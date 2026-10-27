"""The public G2 interface replays one coherent bank sample and exports its clock."""
import json
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse

import simulate_source_sensor as pipeline
from Simulation_G2.g2_simulation_core import G2SourceSimulator
from Simulation_G2.nmm_bank import initialize_bank, projection_hash, save_bank_sample


class G2BankCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bank = self.root / "bank"
        projection = sparse.csc_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, .5]])
        initialize_bank(self.bank, atlas_id="test3", n_regions=3, projection_sha256=projection_hash(projection))
        time_ms = np.arange(1000) * 2.
        source = np.array([np.sin(2 * np.pi * 10 * time_ms / 1000 + phase) for phase in (0, .4, .9)])
        save_bank_sample(self.bank, "alpha", source, time_ms,
                         {"signal_type": "alpha_jr", "output_srate_hz": 500, "active_regions": [1],
                          "n_regions_output": 3, "normalize": False, "model": {"class": "JansenRit"}}, {})
        cortex = {"Vertices": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.]]),
                  "Faces": np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]),
                  "VertConn": sparse.csr_matrix(np.ones((4, 4)) - np.eye(4))}
        self.backend = G2SourceSimulator(np.array([[1, 0, .2, 0], [0, 1, 0, .3], [.1, 0, 1, 0]]),
                                         cortex, projection, self.bank)

    def args(self, *options):
        return pipeline.build_parser().parse_args([
            "--nmm-bank", str(self.bank), "--nmm-sample", "alpha", "--sample-id", "case",
            "--srate-hz", "250", "--duration-s", "1.5", "--nmm-start-s", ".25", *options,
        ])

    def test_default_bank_preserves_all_regions_and_exports_physical_clock(self):
        args = self.args()
        self.assertIsNone(args.active_regions)
        with patch.object(pipeline, "load_fs20k_backend", return_value=self.backend):
            manifest = pipeline.run_g2(args, self.root / "output")
        with np.load(self.root / "output/case/case_data.npz") as saved:
            self.assertEqual(saved["region_source"].shape, (3, 375))
            self.assertTrue(np.all(np.std(saved["region_source"], axis=1) > .1))
            np.testing.assert_allclose(saved["time_ms"], np.arange(375) * 4)
            np.testing.assert_allclose(saved["sensor"], saved["clean_sensor"] + saved["sensor_noise"], atol=1e-6)
        self.assertEqual(manifest["source_type"], "alpha_jr")
        self.assertEqual(manifest["nmm_source_mode"], "network")
        loaded = json.loads((self.root / "output/case/case_manifest.json").read_text())
        self.assertIn("time_ms", loaded["arrays"])

    def test_selected_mode_masks_one_trajectory(self):
        with patch.object(pipeline, "load_fs20k_backend", return_value=self.backend):
            pipeline.run_g2(self.args("--nmm-source-mode", "selected", "--active-regions", "2"), self.root)
        with np.load(self.root / "case/case_data.npz") as saved:
            self.assertEqual(np.flatnonzero(np.std(saved["region_source"], axis=1)).tolist(), [2])

    def test_conflicting_flags_fail_before_loading(self):
        for options in (("--keep-nmm-background",), ("--active-regions", "2"),
                        ("--source-type", "resting_jr", "--nmm-state", "alpha_jr")):
            with self.subTest(options=options), self.assertRaises(ValueError):
                pipeline.run_g2(self.args(*options), self.root)
        with self.assertRaises(ValueError):
            pipeline.run_g2(pipeline.build_parser().parse_args(["--nmm-state", "alpha_jr"]), self.root)

    def test_backend_receives_bank_path_and_other_engines_reject_bank_options(self):
        with patch.object(pipeline.G2SourceSimulator, "from_paths") as loader:
            pipeline.load_fs20k_backend(self.args())
            self.assertEqual(loader.call_args.args[-1], self.bank)
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            pipeline.main(["--engine", "g1", "--nmm-bank", str(self.bank)])


if __name__ == "__main__":
    unittest.main()
