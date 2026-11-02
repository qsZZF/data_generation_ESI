"""Geometric invariants of cortical FEM support dictionaries."""
import unittest

import numpy as np
from scipy import sparse

from resolution_benchmark.support_bases import build_surface_basis, full_green_resolvent, support_dictionaries


class SurfaceBasisTests(unittest.TestCase):
    def setUp(self):
        self.rr = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0],
                            [0, -1, 0], [0, 0, 1], [0, 0, -1]], float) * .05
        self.tris = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
                              [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])

    def test_fem_mass_eigenproblem_and_selected_order(self):
        order = np.array([4, 1, 3, 0, 5, 2])
        result = build_surface_basis(self.rr, self.tris, order, n_modes=4)
        modes = result['mesh_lbo']
        mass = result['mesh_mass_m2']
        np.testing.assert_allclose(modes.T @ (mass[:, None] * modes), np.eye(4), atol=1e-10)
        np.testing.assert_allclose(result['raw_lbo'], modes[order], atol=1e-12)
        self.assertAlmostEqual(result['eigenvalues_m_inv2'][0], 0., places=7)
        np.testing.assert_allclose(result['distances_m'], result['distances_m'].T, atol=1e-12)
        self.assertGreater(result['distances_m'][1, 3], np.linalg.norm(self.rr[1]-self.rr[0]))

    def test_physical_scaling(self):
        a = build_surface_basis(self.rr, self.tris, np.arange(6), n_modes=4)
        b = build_surface_basis(self.rr * 2, self.tris, np.arange(6), n_modes=4)
        np.testing.assert_allclose(b['eigenvalues_m_inv2'], a['eigenvalues_m_inv2']/4, atol=1e-8)
        np.testing.assert_allclose(b['mesh_mass_m2'], a['mesh_mass_m2']*4)
        np.testing.assert_allclose(b['distances_m'], a['distances_m']*2)

    def test_dictionaries_do_not_connect_hemispheres(self):
        distances = np.array([[0., .02, np.inf], [.02, 0., np.inf], [np.inf, np.inf, 0.]])
        raw = np.array([[1., 1., 0.], [1., -1., 0.], [0., 0., 1.]])
        dictionaries = support_dictionaries(raw, np.array([0., 20., 0.]), distances)
        np.testing.assert_allclose(dictionaries['lbo'].T @ dictionaries['lbo'], np.eye(3), atol=1e-12)
        self.assertEqual(dictionaries['patch_30mm'][0, 2], 0.)
        self.assertEqual(dictionaries['green_30mm'][0, 2], 0.)
        for name, dictionary in dictionaries.items():
            np.testing.assert_allclose(np.linalg.norm(dictionary, axis=0), 1., atol=1e-12, err_msg=name)

    def test_rejects_degenerate_or_absent_vertices(self):
        with self.assertRaises(ValueError):
            build_surface_basis(self.rr, np.array([[0, 0, 1]]), [0], n_modes=1)
        with self.assertRaises(ValueError):
            build_surface_basis(self.rr, self.tris, [6], n_modes=2)

    def test_full_green_satisfies_pde_and_has_full_rank(self):
        stiffness = sparse.csr_matrix([[1., -1., 0.], [-1., 2., -1.], [0., -1., 1.]])
        mass = np.array([.001, .002, .003])
        result = full_green_resolvent(stiffness, mass, np.arange(3), length_m=.02)
        raw = result['raw']
        operator = sparse.diags(mass) + .02**2*stiffness
        np.testing.assert_allclose(operator @ raw, np.eye(3), atol=1e-14)
        np.testing.assert_allclose(raw, raw.T, atol=1e-12)
        self.assertEqual(np.linalg.matrix_rank(raw), 3)
        self.assertLess(result['pde_relative_residual'], 1e-13)
        np.testing.assert_allclose(np.linalg.norm(result['normalized'], axis=0), 1.)

    def test_full_green_restricts_after_solve(self):
        stiffness = sparse.csr_matrix([[1., -1., 0.], [-1., 2., -1.], [0., -1., 1.]])
        mass = np.ones(3)*.001
        full = full_green_resolvent(stiffness, mass, np.arange(3))
        subset = full_green_resolvent(stiffness, mass, np.array([2, 0]))
        np.testing.assert_allclose(subset['raw'], full['raw'][np.ix_([2, 0], [2, 0])])
        surface = build_surface_basis(self.rr, self.tris, np.arange(6), n_modes=4)
        self.assertEqual(np.linalg.matrix_rank(surface['green_full_20mm']), 6)


if __name__ == '__main__':
    unittest.main()
