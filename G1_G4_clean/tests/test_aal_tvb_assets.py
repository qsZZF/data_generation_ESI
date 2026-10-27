"""Check the offline AAL-to-TVB export and delay assumptions."""
import io
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.io import loadmat

from Simulation_G3.prepare_aal_tvb import (
    DEFAULT_ATLAS, DEFAULT_SC, build_assets, centroid_distances,
    without_self_connections,
)


class AALTVBTests(unittest.TestCase):
    def test_area_weighted_centres_keep_region_order_and_ignore_unassigned(self):
        vertices = np.array([[0., 0, 0], [4, 0, 0], [0, 3, 0], [999, 999, 999]])
        centres, distances = centroid_distances(vertices, [1, 3, 2, 1], [1, 1, 0, -1], 2)
        np.testing.assert_allclose(centres, [[0, 3, 0], [3, 0, 0]])
        np.testing.assert_allclose(distances, [[0, np.sqrt(18)], [np.sqrt(18), 0]])
        with self.assertRaisesRegex(ValueError, "empty parcel"):
            centroid_distances(vertices, [1, 3, 2, 1], [0, 0, 0, -1], 2)

    def test_diagonal_removal_preserves_asymmetric_raw_offdiagonal_weights(self):
        original = np.array([[1., .7, .1], [.3, 1, .6], [.2, .4, 1]])
        result = without_self_connections(original)
        np.testing.assert_array_equal(result, [[0, .7, .1], [.3, 0, .6], [.2, .4, 0]])
        np.testing.assert_array_equal(np.diag(original), 1)
        with self.assertRaises(ValueError):
            without_self_connections([[0, np.nan], [1, 0]])

    @unittest.skipUnless((DEFAULT_ATLAS / "atlas.npz").exists() and
                         (DEFAULT_SC / "SCmatrices88healthy.mat").exists(), "Local AAL resources absent")
    def test_real_bundle_order_provenance_roundtrip_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = build_assets(directory)
            self.assertEqual(manifest["n_regions"], 78)
            self.assertIn("not measured", manifest["tract_lengths_method"])
            self.assertEqual(manifest["nominal_conduction_speed_mm_per_ms"], 3)
            self.assertEqual(len(manifest["source_files_sha256"]), 6)
            roi = np.array(manifest["aal_roi_numbers"]) - 1
            raw = loadmat(DEFAULT_SC / "SCmatrices88healthy.mat")["SCmatrices"].mean(axis=0).T
            expected = without_self_connections(raw[np.ix_(roi, roi)])
            with ZipFile(Path(directory) / "connectivity_78.zip") as archive:
                weights = np.loadtxt(io.BytesIO(archive.read("weights.txt")))
                lengths = np.loadtxt(io.BytesIO(archive.read("tract_lengths.txt")))
                names = np.loadtxt(io.BytesIO(archive.read("centres.txt")), dtype=str, usecols=0)
            np.testing.assert_array_equal(weights, expected)
            np.testing.assert_array_equal(names, manifest["region_names"])
            np.testing.assert_allclose(lengths, lengths.T)
            np.testing.assert_array_equal(np.diag(lengths), 0)
            self.assertEqual(build_assets(directory)["connectivity_sha256"], manifest["connectivity_sha256"])


if __name__ == "__main__":
    unittest.main()
