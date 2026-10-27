"""Categorical atlas projection and the actual fsaverage5/forward-model contract."""
import csv
from pathlib import Path
import unittest

import nibabel as nib
import numpy as np

from Simulation_G4.atlas import (
    DEFAULT_DIRECTORY, EXCLUDED_PREFIXES, ROOT, categorical_sample,
    load_atlas, parcel_projection, verify_mesh_alignment,
)
from Simulation_G2.g2_simulation_core import load_brainstorm_leadfield


class AtlasTests(unittest.TestCase):
    def test_categorical_sampling_does_not_average_labels_or_fill_background(self):
        data = np.zeros((3, 2, 2), dtype=np.int16)
        data[0] = 2001
        data[1] = 2101
        image = nib.Nifti1Image(data, np.eye(4))
        actual = categorical_sample(image, np.array([[0.49, 0, 0], [0.51, 0, 0], [2, 0, 0], [9, 0, 0]]))
        np.testing.assert_array_equal(actual, [2001, 2101, 0, 0])

    def test_projection_conserves_parcel_mass_and_leaves_unassigned_zero(self):
        projection, indices = parcel_projection(np.array([3, 0, 3, 8]), [3, 8], np.array([1., 2., 3., 5.]))
        np.testing.assert_array_equal(indices, [0, -1, 0, 1])
        np.testing.assert_allclose(projection.toarray(), [[.25, 0], [0, 0], [.75, 0], [0, 1]])
        with self.assertRaisesRegex(ValueError, "empty parcel"):
            parcel_projection(np.array([3, 0]), [3, 8], np.ones(2))

    def test_alignment_rejects_same_cloud_with_wrong_vertex_order(self):
        coordinates = np.array([[0., 0, 0], [2, 0, 0], [0, 3, 0], [0, 0, 4]])
        faces = np.array([[0, 1, 2], [0, 1, 3]])
        info = verify_mesh_alignment(coordinates + [5, -2, 3], faces, coordinates, faces)
        self.assertLess(info["max_vertex_error_mm"], 1e-10)
        with self.assertRaises(ValueError):
            verify_mesh_alignment(coordinates[[1, 0, 2, 3]], faces, coordinates, faces)

    @unittest.skipUnless((DEFAULT_DIRECTORY / "atlas.npz").exists(), "Local atlas not built")
    def test_actual_bundle_preserves_sc_order_forward_projection_and_cortical_mask(self):
        atlas = load_atlas()
        self.assertEqual(atlas["projection"].shape, (20484, 78))
        self.assertEqual(atlas["leadfield"].shape, (64, 78))
        self.assertTrue(np.isfinite(atlas["leadfield"]).all())
        self.assertTrue(np.all(atlas["labels_vertex"][~atlas["cortex_mask"]] == 0))
        self.assertTrue(all(not name.startswith(EXCLUDED_PREFIXES) for name in atlas["region_names"]))
        np.testing.assert_allclose(atlas["projection"].sum(axis=0), 1, atol=1e-12)
        with (ROOT / "Simulation_G4/reference/aal_regions.csv").open() as handle:
            reference = list(csv.DictReader(handle))
        for number, name, label in zip(atlas["aal_roi_numbers"], atlas["region_names"], atlas["region_ids"]):
            self.assertEqual(reference[number - 1]["atlas_name"], name)
            self.assertEqual(int(reference[number - 1]["atlas_mni_id"]), label)
        source = np.random.default_rng(12).normal(size=(78, 5))
        leadfield = load_brainstorm_leadfield(ROOT / "Anatomy/Gain_20k_FS.mat", n_vertices=20484)
        np.testing.assert_allclose(atlas["leadfield"] @ source,
                                   leadfield @ (atlas["projection"] @ source), atol=1e-11)
        self.assertLess(atlas["manifest"]["mesh_alignment"]["max_vertex_error_mm"], 1e-10)
        self.assertFalse(atlas["manifest"]["native_surface_atlas"])


if __name__ == "__main__":
    unittest.main()
