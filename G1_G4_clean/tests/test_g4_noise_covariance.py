"""Covariance follows each synthetic draw's actual montage, noise and filter."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from Simulation_G4.bank import generate_bank, project_bank
from Simulation_G4.pipeline import analysis_filter, run_dataset


class G4NoiseCovarianceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="g4_covariance_")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.bank = self.directory / "bank"
        rng = np.random.default_rng(73)
        self.connectivity = np.zeros((4, 4))
        self.atlas = dict(channel_names=np.array(["Fp1", "Fp2", "F3", "F4", "C3", "C4", "T7", "T8"]),
            region_names=np.array(["r1", "r2", "r3", "r4"]), aal_roi_numbers=np.arange(1, 5),
            leadfield=rng.normal(size=(8, 4)),
            manifest=dict(atlas_id="fixture", bundle_sha256="fixture", template_limitation="synthetic fixture"))
        self.measured = self.directory / "measured_fixture.npz"
        np.savez(self.measured, eeg=rng.normal(size=(8, 8, 256)) * 1e-6,
            ch_names=self.atlas["channel_names"], units="V", sfreq_hz=256,
            subject=np.full(8, "fixture"), condition=np.full(8, "fixture"),
            split=np.full(8, "test"), trial=np.repeat(["trial1", "trial2"], 4), metadata_json="{}")
        with patch("Simulation_G4.bank.simulate", side_effect=self.simulate), contextlib.redirect_stdout(io.StringIO()):
            generate_bank(self.connectivity, self.bank, n_draws=10, duration_s=1,
                          atlas_id="fixture", atlas_hash="fixture")

    @staticmethod
    def simulate(parameters, connectivity, *, duration_s, sampling_rate_hz, seed, **kwargs):
        times = np.arange(round(duration_s * sampling_rate_hz)) / sampling_rate_hz
        signal = np.sin(2 * np.pi * (6 + np.asarray(parameters["C1"]))[:, None] * times)
        signal += np.random.default_rng(seed).normal(size=signal.shape) * .1
        return signal, times, {"seed": int(seed), "evidence_kind": "synthetic_unit_fixture"}

    def test_optional_noise_preserves_legacy_result_and_scaled_realization(self):
        leadfield = self.atlas["leadfield"]
        old = project_bank(self.bank, leadfield, 256, sensor_snr_db=10, seed=13)
        noisy, targets, noise = project_bank(self.bank, leadfield, 256,
                                            sensor_snr_db=10, seed=13, return_noise=True)
        clean, _, zeros = project_bank(self.bank, leadfield, 256, return_noise=True)
        self.assertEqual(len(old), 2)
        np.testing.assert_array_equal(old[0], noisy)
        np.testing.assert_array_equal(old[1], targets)
        np.testing.assert_array_equal(zeros, 0)
        np.testing.assert_allclose(noisy - clean, noise, atol=8e-7, rtol=1e-5)
        np.testing.assert_allclose(noise.mean(axis=1), 0, atol=1e-15)
        np.testing.assert_allclose(np.mean(clean ** 2, axis=(1, 2)) /
                                   np.mean(noise ** 2, axis=(1, 2)), 10, rtol=2e-7)

    def test_independent_baseline_uses_shared_gain_without_changing_event(self):
        options = dict(sensor_snr_db=10, seed=13)
        old = project_bank(self.bank, self.atlas["leadfield"], 256, return_noise=True, **options)
        new = project_bank(self.bank, self.atlas["leadfield"], 256, return_baseline=True, **options)
        for expected, actual in zip(old, new[:3]):
            np.testing.assert_array_equal(expected, actual)
        baseline = new[3]["baseline_sensor"]
        self.assertEqual(baseline.shape, (10, 8, 512))
        self.assertFalse(np.array_equal(baseline, new[2]))
        for index, info in enumerate(new[3]["metadata"]):
            values = np.random.default_rng(np.random.SeedSequence(info["seed_sequence"])).standard_normal((8, 512))
            values -= values.mean(axis=0, keepdims=True)
            values -= values.mean(axis=1, keepdims=True)
            np.testing.assert_array_equal(baseline[index], values * info["shared_noise_gain"])
            event = np.random.default_rng(np.random.SeedSequence(info["event_seed_sequence"])).standard_normal((8, 256))
            event -= event.mean(axis=0, keepdims=True)
            event -= event.mean(axis=1, keepdims=True)
            np.testing.assert_array_equal(new[2][index], event * info["shared_noise_gain"])
        repeated = project_bank(self.bank, self.atlas["leadfield"], 256, return_baseline=True, **options)
        np.testing.assert_array_equal(repeated[3]["baseline_sensor"], baseline)
        shorter = project_bank(self.bank, self.atlas["leadfield"], 256,
                               return_baseline=True, baseline_duration_s=.5, **options)
        np.testing.assert_array_equal(shorter[2], new[2])
        self.assertEqual(shorter[3]["baseline_sensor"].shape[-1], 128)

    def test_saved_training_covariance_uses_filtered_baseline_and_clean_forward_is_zero(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        leadfield = self.atlas["leadfield"]
        leadfield = leadfield - leadfield.mean(axis=0, keepdims=True)
        for snr in (None, 10):
            with self.subTest(snr=snr):
                name = f"snr_{snr}"
                _, _, raw_noise, baseline = project_bank(self.bank, leadfield, 256,
                    sensor_snr_db=snr, seed=42, return_baseline=True, baseline_duration_s=.5)
                noise = analysis_filter(raw_noise, 256)
                expected_baseline = analysis_filter(baseline["baseline_sensor"], 256)
                with patch("Simulation_G4.pipeline.simulate", side_effect=self.simulate), \
                        contextlib.redirect_stdout(io.StringIO()):
                    report = run_dataset(name, [self.measured], self.atlas, self.connectivity,
                        self.bank, self.directory / "output", epochs=1, batch_size=3,
                        profile="compact", device="cpu", sensor_snr_db=snr, baseline_duration_s=.5)
                output = self.directory / "output" / name
                manifest = json.loads((output / "run_manifest.json").read_text())
                with np.load(output / manifest["noise_covariance"]["file"]) as stored:
                    self.assertEqual(stored["noise_covariance"].dtype, np.dtype("float64"))
                    np.testing.assert_array_equal(stored["draw_indices"], np.arange(10))
                    np.testing.assert_array_equal(stored["oracle_noise_covariance"], stored["sensor_noise_covariance"])
                    np.testing.assert_array_equal(stored["baseline_sensor"], expected_baseline)
                    metadata = json.loads(str(stored["metadata_json"]))
                    for index in range(10):
                        expected = np.cov(expected_baseline[index].astype(np.float64))
                        np.testing.assert_allclose(stored["noise_covariance"][index], expected, atol=1e-15)
                        np.testing.assert_allclose(stored["oracle_noise_covariance"][index],
                                                   np.cov(noise[index].astype(np.float64)), atol=1e-15)
                        info = metadata["draws"][index]["noise_covariance_info"]
                        self.assertEqual(info["rank"], 0 if snr is None else 7)
                        self.assertEqual(info["estimation_window"], "baseline_only")
                        self.assertEqual(info["n_samples"], 128)
                    if snr is not None:
                        self.assertFalse(np.allclose(stored["noise_covariance"][0], np.cov(raw_noise[0])))
                        self.assertFalse(np.allclose(stored["noise_covariance"][0], stored["oracle_noise_covariance"][0]))
                with np.load(output / report["groups"][0]["file"]) as forward:
                    self.assertEqual(forward["noise_covariance"].shape, (3, 8, 8))
                    np.testing.assert_array_equal(forward["noise_covariance"], 0)
                    self.assertEqual(forward["baseline_sensor"].shape, (3, 8, 512))
                    np.testing.assert_array_equal(forward["baseline_sensor"], 0)
                    metadata = json.loads(str(forward["metadata_json"]))
                    self.assertEqual(metadata["noise_status"], "no_added_noise")
                    self.assertEqual(metadata["covariance_target"], "all_simulated_network")
                    self.assertEqual(metadata["noise_covariance_info"]["rank"], 0)
                    self.assertEqual(metadata["baseline"]["status"], "uninformative_no_added_noise")

    def test_short_baseline_is_rejected_before_filtering(self):
        with self.assertRaisesRegex(ValueError, "at least 28 samples"):
            run_dataset("short", [self.measured], self.atlas, self.connectivity,
                self.bank, self.directory / "output", baseline_duration_s=16 / 256)


if __name__ == "__main__":
    unittest.main()
