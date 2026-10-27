"""Foreground ownership and spatial-envelope regression checks on a small mesh."""
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
from scipy import sparse

from Simulation_G1.g1_simulation_core import G1SimulationConfig, G1SourceSimulator


def grid_simulator(n=9):
    vertices = np.array([[100 + 5*x, 100 + 5*y, 0] for y in range(n) for x in range(n)], float)
    faces = []
    for y in range(n-1):
        for x in range(n-1):
            i = y*n+x
            faces.extend([[i, i+1, i+n], [i+1, i+n+1, i+n]])
    faces = np.asarray(faces)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    conn = sparse.coo_matrix((np.ones(len(edges)*2),
        (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
        shape=(n*n, n*n)).tocsc()
    return G1SourceSimulator(np.random.default_rng(0).normal(size=(4, n*n)), vertices, faces, conn)


class PatchSamplingTests(unittest.TestCase):
    def setUp(self):
        self.sim = grid_simulator()
        self.config = G1SimulationConfig(n_patch=3, patch_area_mm2=180, min_seed_distance_mm=0,
            srate=100, time_window=(0, .5), random_seed=4, normalize_source=False)

    def test_random_full_patches_are_disjoint_and_reproducible(self):
        for seed in range(10):
            config = replace(self.config, random_seed=seed)
            source, meta = self.sim.generate_source(config)
            repeated, other = self.sim.generate_source(config)
            np.testing.assert_array_equal(source, repeated)
            self.assertEqual(meta, other)
            supports = [p['vertices'] for p in meta['patches']]
            self.assertEqual(sum(map(len, supports)), len(set(sum(supports, []))))
            for support in supports:
                self.assertGreaterEqual(self.sim.vertex_area_coord2[support].sum(), 180)

    def test_explicit_overlapping_patches_fail(self):
        with self.assertRaisesRegex(ValueError, 'overlap'):
            self.sim.generate_source(replace(self.config, n_patch=2, seed_vertices=[40, 41]))

    def test_exhausted_mesh_is_not_silently_undersized(self):
        with self.assertRaisesRegex((ValueError, RuntimeError), 'area|fit|pack'):
            self.sim.generate_source(replace(self.config, n_patch=1, patch_area_mm2=1e7))

    def test_invalid_seeds_and_counts(self):
        for updates in (dict(n_patch=0), dict(n_patch=1, seed_vertices=[-1]),
                        dict(n_patch=2, seed_vertices=[40,40]), dict(n_patch=1, seed_vertices=[1.5])):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                self.sim.generate_source(replace(self.config, **updates))

    def test_spatial_profile_is_applied_before_demeaning(self):
        from Simulation_G1.spatial import SpatialDynamicsConfig, spatial_envelope
        config = replace(self.config, n_patch=1, seed_vertices=[40],
                         spatial=SpatialDynamicsConfig(mode='diffuse'))
        trace = np.sin(np.arange(50)*.4)[None, :] + 1
        with patch.object(self.sim, '_build_roi_signals', return_value=trace):
            source, meta = self.sim.generate_source(config)
        info = meta['patches'][0]
        weights, _ = spatial_envelope(info['spatial_weights'], np.arange(50)/100, config.spatial,
                                     area_mm2=info['area_mm2'], hop_order=info['hop_order'])
        expected = weights * trace * info['gain']
        expected -= expected.mean(axis=1, keepdims=True)
        np.testing.assert_allclose(source[info['vertices']], expected, atol=3e-7)
        inactive = np.setdiff1d(np.arange(self.sim.n_sources), info['vertices'])
        self.assertTrue(np.all(source[inactive] == 0))

    def test_source_snr_is_calibrated_after_spatial_demeaning(self):
        from Simulation_G1.spatial import SpatialDynamicsConfig
        for mode in ('static', 'diffuse', 'contract', 'pulse'):
            config = replace(self.config, n_patch=1, patch_area_mm2=1000, decay_mm=3,
                             spatial=SpatialDynamicsConfig(mode=mode))
            foreground, _ = self.sim.generate_source(config)
            mixed, meta = self.sim.generate_source(replace(config,
                source_background_type='white', source_snr_db=10))
            background = mixed-foreground
            snr = 10*np.log10(np.mean(foreground.astype(float)**2)/np.mean(background.astype(float)**2))
            with self.subTest(mode=mode):
                self.assertAlmostEqual(snr, 10, places=5)


class SpatialEnvelopeTests(unittest.TestCase):
    def test_modes_and_auto_threshold_units(self):
        from Simulation_G1.spatial import SpatialDynamicsConfig, spatial_envelope
        base = np.array([1., .5, .1])
        times = np.array([0., .15, .3, .6])
        for mode, expected in [('diffuse', (base, np.ones(3))),
                               ('contract', (np.ones(3), base)), ('pulse', (base, base))]:
            weights, meta = spatial_envelope(base, times, SpatialDynamicsConfig(mode=mode),
                                            area_mm2=1200, hop_order=2)
            np.testing.assert_allclose(weights[:, 0], expected[0])
            np.testing.assert_allclose(weights[:, -1], expected[1])
            self.assertTrue(np.all((weights >= base[:, None]) & (weights <= 1)))
            self.assertEqual(meta['mode'], mode)
            if mode == 'pulse':
                np.testing.assert_allclose(weights[:, 1], 1)
        for area, hops, expected in [(999, 4, 'static'), (1000, 0, 'diffuse'), (20, 5, 'diffuse')]:
            _, meta = spatial_envelope(base, times, SpatialDynamicsConfig(), area_mm2=area, hop_order=hops)
            self.assertEqual(meta['mode'], expected)

    def test_invalid_schedule_fails_and_disconnected_support_is_explicit(self):
        from Simulation_G1.spatial import SpatialDynamicsConfig, spatial_envelope, patch_hop_order
        for changes in [dict(duration_s=0), dict(start_s=float('nan')), dict(mode='unknown'),
                        dict(area_threshold_cm2=-1), dict(hop_threshold=1.5)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                spatial_envelope(np.array([1., .2]), np.array([0., 1.]),
                                 SpatialDynamicsConfig(**changes), area_mm2=20, hop_order=1)
        conn = sparse.csc_matrix(np.array([[0,1,0],[1,0,0],[0,0,0]]))
        self.assertEqual(patch_hop_order(np.array([0,1]), 0, conn), 1)
        self.assertIsNone(patch_hop_order(np.array([0,1,2]), 0, conn))


if __name__ == '__main__':
    unittest.main()
