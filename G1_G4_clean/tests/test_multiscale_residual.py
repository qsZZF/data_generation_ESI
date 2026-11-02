"""Residual fields are reproducible and independent of retained coarse modes."""
import unittest
import numpy as np

from Simulation_G1.geometry import compute_basis
from test_spatial_patches import grid_simulator


class MultiscaleResidualTests(unittest.TestCase):
    def setup_mesh(self, masked=False):
        sim = grid_simulator(5)
        if masked:
            sim.cortex_mask[np.arange(25) % 5 == 2] = False
        bundle = compute_basis(sim.vertices * sim.coord_to_mm, sim.faces, n_modes=5)
        return sim, bundle

    def test_mass_orthogonal_modes_constants_and_mask(self):
        from Simulation_G1.multiscale import generate_residual_patterns
        sim, bundle = self.setup_mesh(masked=True)
        patterns, info = generate_residual_patterns(sim, bundle, seed=7, n_components=4, n_modes=3)
        mass = bundle['mass']
        np.testing.assert_array_equal(patterns[~sim.cortex_mask], 0)
        np.testing.assert_allclose(bundle['basis'][:, :3].T @ (mass[:, None] * patterns), 0, atol=1e-12)
        # Mask splits the mesh into independent left and right strips.
        for columns in ([0, 1], [3, 4]):
            mask = np.isin(np.arange(25) % 5, columns)
            np.testing.assert_allclose(mass[mask] @ patterns[mask], 0, atol=1e-12)
        np.testing.assert_allclose(np.sum(mass[:, None] * patterns**2, axis=0), 1, atol=1e-12)
        self.assertEqual(info['eligible_connected_components'], 2)
        self.assertIn('Dirichlet', info['boundary_condition'])

    def test_replay_cache_isolation_and_seed_variation(self):
        from Simulation_G1.multiscale import generate_residual_patterns
        sim, bundle = self.setup_mesh()
        cache = {}
        first, info = generate_residual_patterns(sim, bundle, seed=12, n_components=3, cache=cache)
        repeated, again = generate_residual_patterns(sim, bundle, seed=12, n_components=3, cache=cache)
        changed, _ = generate_residual_patterns(sim, bundle, seed=13, n_components=3, cache=cache)
        np.testing.assert_array_equal(first, repeated)
        self.assertEqual(info['operator_key'], again['operator_key'])
        self.assertFalse(np.array_equal(first, changed))
        for length in (5., 8., 12.):
            generate_residual_patterns(sim, bundle, seed=12, n_components=3, length_mm=length, cache=cache)
        self.assertEqual(len(cache), 2)

    def test_retained_mode_count_and_mask_change_cache_identity(self):
        from Simulation_G1.multiscale import generate_residual_patterns
        sim, bundle = self.setup_mesh()
        cache = {}
        _, a = generate_residual_patterns(sim, bundle, seed=1, n_components=2, n_modes=2, cache=cache)
        _, b = generate_residual_patterns(sim, bundle, seed=1, n_components=2, n_modes=3, cache=cache)
        self.assertNotEqual(a['operator_key'], b['operator_key'])
        sim.cortex_mask[0] = False
        _, c = generate_residual_patterns(sim, bundle, seed=1, n_components=2, n_modes=3, cache=cache)
        self.assertNotEqual(b['operator_key'], c['operator_key'])

    def test_invalid_and_exhausted_residual_rank(self):
        from Simulation_G1.multiscale import generate_residual_patterns
        sim, bundle = self.setup_mesh()
        with self.assertRaisesRegex(ValueError, 'capacity'):
            generate_residual_patterns(sim, bundle, seed=1, n_components=25)
        full = compute_basis(sim.vertices * sim.coord_to_mm, sim.faces, n_modes=24)
        with self.assertRaisesRegex(ValueError, 'capacity'):
            generate_residual_patterns(sim, full, seed=1, n_components=1)
        with self.assertRaisesRegex(ValueError, 'length_mm'):
            generate_residual_patterns(sim, bundle, seed=1, length_mm=0)
        wrong = dict(bundle, mass=bundle['mass']*2)
        with self.assertRaisesRegex(ValueError, 'mass'):
            generate_residual_patterns(sim, wrong, seed=1)

    def test_fraction_vertex_cross_term_and_mass_report(self):
        from Simulation_G1.multiscale import mix_residual
        coarse = np.array([[1., 2], [2., 4]])
        residual = np.array([[1., 2], [-1., -2]])
        mass = np.array([2., 1.])
        mixed, scaled, info = mix_residual(coarse, residual, .2, mass=mass)
        np.testing.assert_array_equal(mixed, coarse+scaled)
        self.assertAlmostEqual(info['realized_component_fraction'], .2)
        self.assertAlmostEqual(info['cross_power'], 2*np.mean(coarse*scaled))
        self.assertNotEqual(info['cross_power'], 0)
        self.assertAlmostEqual(info['mass_orthogonality_error'], 0, places=14)
        self.assertAlmostEqual(np.mean(mixed**2), info['coarse_power']+info['residual_power']+info['cross_power'])
        _, _, area = mix_residual(coarse, residual, .3, metric='area_weighted', mass=mass)
        self.assertAlmostEqual(area['realized_component_fraction'], .3)

    def test_zero_fraction_exact_and_degenerate_energy_validation(self):
        from Simulation_G1.multiscale import mix_residual
        coarse = np.arange(6, dtype=np.float32).reshape(2, 3)
        mixed, scaled, info = mix_residual(coarse, None, 0)
        np.testing.assert_array_equal(mixed, coarse)
        self.assertEqual(mixed.dtype, coarse.dtype)
        self.assertFalse(np.any(scaled))
        self.assertEqual(info['realized_component_fraction'], 0)
        for fraction in (-1, 1, np.nan):
            with self.assertRaises(ValueError):
                mix_residual(coarse, coarse, fraction)
        with self.assertRaisesRegex(ValueError, 'positive'):
            mix_residual(coarse, np.zeros_like(coarse), .2)
        with self.assertRaisesRegex(ValueError, 'positive'):
            mix_residual(np.zeros_like(coarse), coarse, .2)


if __name__ == '__main__':
    unittest.main()
