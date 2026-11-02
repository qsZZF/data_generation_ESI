import unittest
import numpy as np

from scripts.run_nullspace_study import source_order,sensor_subsets,nested_patch_bases


class NullspaceDesignTests(unittest.TestCase):
    def test_hemisphere_balanced_source_order_accepts_names(self):
        coords=np.random.default_rng(3).normal(size=(12,3))
        hemis=np.array(['lh']*6+['rh']*6)
        order=source_order(coords,hemis)
        np.testing.assert_array_equal(np.sort(order),np.arange(12))
        self.assertEqual(hemis[order[:4]].tolist(),['lh','rh','lh','rh'])
        np.testing.assert_array_equal(order,source_order(coords,hemis))

    def test_meg_groups_are_nested_and_not_split(self):
        positions=np.repeat(np.random.default_rng(4).normal(size=(34,3)),3,axis=0)
        names=np.array([f'MEG {site:03d}{orientation}' for site in range(34) for orientation in (1,2,3)])
        choices=sensor_subsets(names,positions,'meg',9)
        for parent,child in zip(choices,choices[1:]):self.assertTrue(set(child)<=set(parent))
        for selected in choices:
            for group in np.unique(selected//3):self.assertEqual(np.sum(selected//3==group),3)

    def test_patch_refinement_preserves_common_physical_field_and_energy(self):
        coords=np.random.default_rng(5).normal(size=(16,3));hemis=np.repeat([0,1],8)
        bases=nested_patch_bases(coords,hemis,[2,4,8,12])
        truth=bases[2]@np.random.default_rng(6).normal(size=(2,50))
        for size,basis in bases.items():
            np.testing.assert_allclose(basis.T@basis,np.eye(size),atol=1e-14)
            coefficients=basis.T@truth
            np.testing.assert_allclose(basis@coefficients,truth,atol=1e-14)
            self.assertAlmostEqual(np.sum(coefficients**2),np.sum(truth**2))
