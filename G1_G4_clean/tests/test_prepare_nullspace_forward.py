"""Rank diagnostics and alignment checks for the bounded real-forward export."""
import os
os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")

from pathlib import Path
import json
import unittest

import numpy as np

from scripts.prepare_nullspace_forward import covariance_diagnostics, verify_file


class CovarianceDiagnosticsTests(unittest.TestCase):
    def test_rank_is_invariant_to_sensor_units(self):
        rng = np.random.default_rng(7)
        factors = rng.normal(size=(6, 4))
        covariance = factors @ factors.T
        scale = np.asarray([1e-15, 1e-15, 1e-13, 1e-13, 1e-6, 1e-6])
        mixed = covariance * scale[:, None] * scale[None, :]
        self.assertEqual(covariance_diagnostics(covariance)["rank"], 4)
        self.assertEqual(covariance_diagnostics(mixed)["rank"], 4)

    def test_genuinely_indefinite_covariance_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive semidefinite"):
            covariance_diagnostics(np.asarray([[1., 2.], [2., 1.]]))


class RealForwardExportTests(unittest.TestCase):
    def test_local_export_alignment_projection_and_cache_identity(self):
        directory = Path(__file__).resolve().parents[1] / "generated_simulation_dataset/nullspace_study/mne_sample"
        path = directory / "mne_sample_forward.npz"
        if not path.exists():
            self.skipTest("Run scripts/prepare_nullspace_forward.py for the optional public fixture")
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
            fixed, free = data["leadfield_fixed"], data["leadfield_free"]
            self.assertEqual(fixed.shape, (364, 498))
            self.assertEqual(free.shape, (364, 1494))
            self.assertEqual(len(np.unique(data["channel_names"])), 364)
            self.assertNotIn("MEG 2443", data["channel_names"])
            self.assertNotIn("EEG 053", data["channel_names"])
            self.assertEqual(metadata["channel_counts"], dict(eeg=59, grad=203, mag=102))
            reconstructed = np.einsum("cnk,nk->cn", free.reshape(364, 498, 3), data["source_normals"])
            self.assertEqual(fixed.dtype, np.dtype("float64"))
            np.testing.assert_array_equal(fixed, reconstructed)
            self.assertIn("float64", metadata["fixed_construction"])
            self.assertLess(metadata["mne_fixed_vs_float64_contraction_max_row_relative_error"], 1e-6)
            projector = data["sensor_projector"]
            np.testing.assert_allclose(projector @ projector, projector, atol=1e-12)
            self.assertEqual(np.linalg.matrix_rank(projector, tol=1e-10), 360)
            projected = projector @ data["noise_cov"] @ projector.T
            self.assertEqual(covariance_diagnostics(projected, 1e-10)["rank"], 360)
            for name in data.files:
                self.assertFalse(data[name].dtype.hasobject)
            for entry in metadata["files"]:
                verify_file(directory / entry["name"], entry["bytes"], entry["sha256"])


if __name__ == "__main__":
    unittest.main()
