"""Spatially correlated nuisance fields share event/baseline gains and identities."""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from Simulation_G1.geometry import compute_basis, save_basis
from Simulation_G1.g1_simulation_core import G1SimulationConfig
from Simulation_G1.training_recipes import build_recipe_dataset, RecipeDataset, asset_identity
from test_spatial_patches import grid_simulator


class RealisticBackgroundTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        self.simulator = grid_simulator()
        bundle = compute_basis(self.simulator.vertices, self.simulator.faces, n_modes=16)
        save_basis(self.path / 'basis.npz', bundle)
        self.settings = dict(basis_path=str(self.path / 'basis.npz'), n_modes=16,
                             network_fraction=0., spatial_length_mm=20., alpha_hz=10.)
        self.config = G1SimulationConfig(srate=128, time_window=(0., 1.), random_seed=9,
            n_patch=1, patch_area_mm2=80, seed_vertices=[40], normalize_source=False,
            source_background_type='spectral_rest', source_background_spectral=self.settings,
            source_snr_db=12., covariance_estimator='oas')

    def test_spatial_background_baseline_replay_and_vertex_snr(self):
        source, meta, background, baseline = self.simulator._generate_source_components(self.config)
        repeated = self.simulator._generate_source_components(self.config)
        for a, b in zip((source, background, baseline), (repeated[0], repeated[2], repeated[3])):
            np.testing.assert_array_equal(a, b)
        foreground = source - background
        self.assertAlmostEqual(10*np.log10(np.mean(foreground**2)/np.mean(background**2)), 12., places=5)
        self.assertEqual(baseline.shape, (81, 256))
        self.assertFalse(np.array_equal(baseline[:, :128], background))
        self.assertTrue(meta['source_background_metadata']['cortex_mask_applied'])
        sensor = self.simulator.generate_sensor(self.config, compute_diagnostics=False)
        self.assertEqual(sensor[-1]['noise_covariance_info']['estimator'], 'oas_standardized_subspace')
        self.assertEqual(sensor[-1]['baseline_duration_s'], 2.)

    def test_noise_gain_scales_both_intervals_without_redrawing(self):
        source1, meta1, first, first_baseline = self.simulator._generate_source_components(self.config)
        source2, meta2, second, second_baseline = self.simulator._generate_source_components(replace(self.config, source_snr_db=18.))
        gain = 10**(-6/20)
        self.assertAlmostEqual(meta2['source_background_metadata']['shared_source_background_gain'] /
                               meta1['source_background_metadata']['shared_source_background_gain'], gain)
        # Exported nuisance is a residual of float32 source addition, so its
        # absolute rounding error scales with foreground amplitude near zeros.
        bound = 4*np.finfo(np.float32).eps*max(np.max(abs(source1)), np.max(abs(source2)))
        np.testing.assert_allclose(second, first*gain, rtol=0, atol=bound)
        np.testing.assert_allclose(second_baseline, first_baseline*gain, rtol=1e-6, atol=1e-8)

    def test_recipe_binds_mask_and_background_asset(self):
        before = asset_identity(self.simulator)
        changed = grid_simulator()
        changed.cortex_mask_metadata = dict(changed.cortex_mask_metadata, sha256='different')
        self.assertNotEqual(before, asset_identity(changed))
        build_recipe_dataset(self.simulator, self.path / 'recipes', n_supports=3,
            variants_per_support=2, eval_variants_per_support=1, seed=8,
            source_background_spectral=self.settings, covariance_estimator='oas')
        dataset = RecipeDataset(self.path / 'recipes', self.simulator, split=None)
        configs = [dataset.recipe(i)[0] for i in range(len(dataset))]
        for config in configs:
            self.assertEqual(config.covariance_estimator, 'oas')
            if config.source_background_type:
                self.assertEqual(config.source_background_type, 'spectral_rest')
                self.assertIn('basis_sha256', config.source_background_spectral)
