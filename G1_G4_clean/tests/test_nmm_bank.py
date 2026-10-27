"""Bank integrity and physical-time replay, independent of expensive TVB runs."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.sparse import eye
from Simulation_G2.g2_simulation_core import G2SimulationConfig, G2SourceSimulator

from Simulation_G2.nmm_bank import (
    initialize_bank, load_bank_sample, prepare_bank_segment, projection_hash,
    read_bank, save_bank_sample,
)


class NMMBankTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.projection = projection_hash(eye(3, format="csc"))
        initialize_bank(self.path, atlas_id="test3", n_regions=3, projection_sha256=self.projection)
        self.times = np.arange(1000) * 2.
        self.source = np.array([np.sin(2 * np.pi * 10 * self.times / 1000 + phase) for phase in (0, .4, .9)]) * 4
        self.manifest = dict(signal_type="alpha_jr", output_srate_hz=500, active_regions=[1],
                             n_regions_output=3, normalize=False, model={"class": "JansenRit"})

    def write(self):
        return save_bank_sample(self.path, "test", self.source, self.times, self.manifest, {"seed": 8})

    def test_roundtrip_amplitude_config_and_joint_phase(self):
        entry = self.write()
        actual, times, loaded = load_bank_sample(self.path, "test", signal_type="alpha_jr")
        np.testing.assert_array_equal(actual, self.source.astype(np.float32))
        np.testing.assert_array_equal(times, self.times)
        self.assertEqual(loaded, entry)
        self.assertEqual(loaded["source_unit"], "mV_PSP")
        self.assertEqual(loaded["provenance"]["seed"], 8)
        self.assertEqual(list(read_bank(self.path)["samples"]), ["test"])
        with self.assertRaises(ValueError):
            load_bank_sample(self.path, "test", signal_type="resting_jr")

    def test_crop_resamples_physical_time_without_stretch_or_rescaling(self):
        result = prepare_bank_segment(self.source, self.times, srate=250, duration_s=1, start_s=.5)
        self.assertEqual(result.shape, (3, 250))
        frequency = np.fft.rfftfreq(250, 1 / 250)[np.argmax(abs(np.fft.rfft(result[0])))]
        self.assertEqual(frequency, 10)
        np.testing.assert_allclose(result[:, 10:-10], self.source[:, 250:750:2][:, 10:-10], atol=.01)
        for kwargs in ({"duration_s": 3}, {"start_s": -.1}, {"start_s": .001}, {"srate": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                prepare_bank_segment(self.source, self.times, **({"srate": 250, "duration_s": 1} | kwargs))

    def test_rejects_wrong_projection_metadata_corruption_and_path_escape(self):
        entry = self.write()
        with self.assertRaises(ValueError):
            initialize_bank(self.path, atlas_id="test3", n_regions=3, projection_sha256="0" * 64)
        index_path = self.path / "bank.json"
        index = read_bank(self.path)
        index["samples"]["test"]["active_regions"] = [2]
        index_path.write_text(json.dumps(index))
        with self.assertRaises(ValueError):
            load_bank_sample(self.path, "test")
        index["samples"]["test"] = entry
        index["samples"]["test"]["source_file"] = "../outside.npz"
        index_path.write_text(json.dumps(index))
        with self.assertRaises(ValueError):
            load_bank_sample(self.path, "test")

    def test_invalid_writes_and_orphans_are_rejected(self):
        for sample, signal, times in (("../bad", self.source, self.times), ("NUL", self.source, self.times),
                                     ("bad", self.source[:2], self.times),
                                     ("bad", self.source * np.nan, self.times),
                                     ("bad", self.source, self.times + 1)):
            with self.subTest(sample=sample), self.assertRaises(ValueError):
                save_bank_sample(self.path, sample, signal, times, self.manifest, {})
        orphan = self.path / "orphan"
        orphan.mkdir()
        (orphan / "unused.npz").write_bytes(b"unknown")
        with self.assertRaises(ValueError):
            initialize_bank(orphan, atlas_id="test3", n_regions=3, projection_sha256=self.projection)

    def test_same_trajectory_different_names_share_split_group(self):
        first = save_bank_sample(self.path, "one", self.source, self.times, self.manifest,
            {"config": {"sample_id": "one", "random_seed": 42, "simulation_length_ms": 2000,
                        "jr_coupling_mode": "dense"}, "connectivity_sha256": "a"})
        second = save_bank_sample(self.path, "two", self.source, self.times, self.manifest,
            {"config": {"sample_id": "two", "random_seed": 42, "simulation_length_ms": 3000,
                        "jr_coupling_mode": "sparse"}, "connectivity_sha256": "a"})
        self.assertEqual(first["simulation_group_id"], second["simulation_group_id"])
        with self.assertRaises(ValueError):
            save_bank_sample(self.path, "ONE", self.source, self.times, self.manifest, {})

    def test_g2_network_and_selected_replay_use_one_coherent_sample(self):
        self.write()
        # Bypass unrelated cortical geometry setup; exercise real G2 source assembly.
        from Simulation_G2.g2_simulation_core import NMMRegionSignalRepository
        simulator = object.__new__(G2SourceSimulator)
        simulator.nmm_repo = NMMRegionSignalRepository(self.path)
        simulator.projection_matrix = eye(3, format="csc")
        simulator.n_regions = simulator.n_vertices = simulator.n_sensors = 3
        simulator.leadfield = np.eye(3)
        config = G2SimulationConfig(time_window=(0, 1), nmm_sample_id="test",
            nmm_start_s=.5, normalize_source=False, normalize_region_signals=False)
        source, region, meta = simulator.generate_source(config)
        np.testing.assert_array_equal(region, self.source.astype(np.float32)[:, 250:750])
        np.testing.assert_array_equal(source, region)
        self.assertEqual(meta["nmm_source_mode"], "network")
        self.assertFalse(meta["spatial_weight_decay_applied"])
        self.assertEqual(meta["configured_regions"], [1])
        config.nmm_source_mode = "selected"
        config.active_region_ids = [0, 2]
        config.weight_decay_mm = None
        _, selected, meta = simulator.generate_source(config)
        np.testing.assert_array_equal(selected[[0, 2]], region[[0, 2]])
        self.assertTrue(np.all(selected[1] == 0))
        self.assertEqual(meta["retained_region_ids"], [0, 2])
        simulator.projection_matrix = 2 * eye(3, format="csc")
        with self.assertRaisesRegex(ValueError, "projection"):
            simulator.generate_source(config)


if __name__ == "__main__":
    unittest.main()
