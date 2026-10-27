"""Spatial settings reach projection before physical source/SNR calibration."""
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse

import simulate_source_sensor as cli
from Simulation_G2.nmm_mixing import mix_sources
from nmm_simulation import NMMSimulationConfig, _validate
import test_unified_nmm as unified_tests


class SpatialEntrypointTests(unittest.TestCase):
    def test_cli_passes_shared_settings_to_g1_and_g2(self):
        args = cli.build_parser().parse_args([
            '--source-type', 'erp', '--spatial-mode', 'diffuse',
            '--diffusion-area-cm2', '7', '--diffusion-hops', '0',
            '--diffusion-start-s', '.1', '--diffusion-duration-s', '.4'])
        for config in (cli.build_g1_config(args), cli.build_g2_config(args, [1])):
            self.assertEqual(config.spatial.mode, 'diffuse')
            self.assertEqual(config.spatial.area_threshold_cm2, 7)
            self.assertIsNone(config.spatial.hop_threshold)
            self.assertEqual(config.spatial.start_s, .1)
            self.assertEqual(config.spatial.duration_s, .4)

    def test_nested_json_spatial_config_and_network_validation(self):
        config = NMMSimulationConfig(source_mode='selected', spatial={'mode': 'diffuse'})
        _validate(config)
        self.assertEqual(config.spatial.mode, 'diffuse')
        with self.assertRaisesRegex(ValueError, 'selected'):
            _validate(replace(config, source_mode='network'))
        self.assertEqual(NMMSimulationConfig().spatial.mode, 'static')

    def test_g3_rejects_synthetic_spatial_modes(self):
        args = cli.build_parser().parse_args(['--source-type', 'alpha_jr', '--spatial-mode', 'pulse'])
        with self.assertRaisesRegex(ValueError, 'spatial|G3'):
            cli.build_g3_config(args, [1])

    def test_transformed_vertices_are_demeaned_before_snr(self):
        times = np.linspace(0, 1, 100)
        region = np.array([2 + np.sin(2*np.pi*times), np.cos(4*np.pi*times)])
        projection = sparse.eye(2, format='csc')
        transformed = region * np.array([times, 1-times])
        arrays, meta = mix_sources(region, region[::-1], projection, [1., 3.],
                                  6., False, foreground_vertex=transformed)
        np.testing.assert_allclose(arrays['source_foreground'].mean(axis=1), 0., atol=1e-14)
        expected = transformed - transformed.mean(axis=1, keepdims=True)
        np.testing.assert_allclose(arrays['source_foreground'], expected)
        self.assertAlmostEqual(meta['realized_source_snr_db'], 6., places=12)
        self.assertFalse(meta['source_matches_static_projection'])
        for invalid in (np.zeros((3, 100)), np.full((2, 100), np.nan)):
            with self.assertRaises(ValueError):
                mix_sources(region, None, projection, [1., 3.], foreground_vertex=invalid)


class UnifiedSpatialTests(unittest.TestCase):
    """Reuse the tiny real backend and trajectory fixture without external data."""

    setUp = unified_tests.UnifiedNMMTests.setUp
    fake_trajectory = staticmethod(unified_tests.UnifiedNMMTests.fake_trajectory)

    def test_selected_overlapping_regions_fail(self):
        with self.assertRaisesRegex(ValueError, 'overlap'):
            self.sim.generate_sample(replace(self.config, active_regions=[0, 1]))
        self.resolver.assert_not_called()

    def test_random_selected_regions_are_disjoint_before_simulation(self):
        result = self.sim.generate_sample(replace(self.config, active_regions=None, n_active_regions=2,
                                                 source_snr_db=None))
        selected = self.resolver.call_args.args[0].active_regions
        self.assertEqual(selected, [0, 2])
        self.assertTrue(result['manifest']['spatial_supports_disjoint'])
        self.resolver.reset_mock()
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            self.sim.generate_sample(replace(self.config, active_regions=None, n_active_regions=3,
                                             source_snr_db=None))
        self.resolver.assert_not_called()

    def test_network_ignores_auto_spatial_and_preserves_projection(self):
        config = replace(self.config, source_mode='network', active_regions=None,
                         n_active_regions=1, spatial={'mode': 'auto'})
        result = self.sim.generate_sample(config)
        np.testing.assert_allclose(result['source'], self.backend.projection_matrix @ result['region_source'], atol=1e-7)
        self.assertIn('network', result['manifest']['spatial_skip_reason'])

    def test_selected_dynamic_projection_precedes_calibration(self):
        config = replace(self.config, spatial={'mode': 'diffuse', 'duration_s': .4}, spatial_decay_mm=12.)
        result = self.sim.generate_sample(config)
        self.assertAlmostEqual(result['manifest']['realized_source_snr_db'], 3., places=6)
        self.assertAlmostEqual(result['manifest']['realized_sensor_snr_db'], 10., places=6)
        self.assertFalse(np.allclose(result['source'], self.backend.projection_matrix @ result['region_source']))
        self.assertTrue(result['manifest']['spatial_sources'])

    def test_unified_dynamic_projection_is_invariant_to_psp_baseline(self):
        config=replace(self.config,spatial={'mode':'diffuse','duration_s':.4},spatial_decay_mm=12.)
        expected=self.sim.generate_sample(config)
        def shifted(simulation,**kwargs):
            source,times,entry=self.fake_trajectory(simulation,**kwargs)
            return source+23.,times,entry
        self.resolver.side_effect=shifted
        actual=self.sim.generate_sample(config)
        np.testing.assert_allclose(actual['source_foreground'],expected['source_foreground'],atol=2e-6)
        np.testing.assert_allclose(actual['sensor'],expected['sensor'],atol=2e-6)

    def test_legacy_cli_samples_requested_count_and_uses_shared_projection(self):
        clips = self.root / 'unpacked'
        for region, clip_id in ((0, 7), (1, 1), (2, 7)):
            directory = clips / f'a{region}'
            directory.mkdir(parents=True)
            (directory / f'nmm_{clip_id}.mat').touch()
        args = cli.build_parser().parse_args([
            '--n-active', '2', '--spatial-mode', 'diffuse', '--clip-id', '7', '--nmm-dir', str(clips)])
        times = np.arange(500) / 500
        clip = np.array([np.sin(2*np.pi*7*times + phase) for phase in (0, .3, 1)])
        with patch.object(cli, 'load_fs20k_backend', return_value=self.backend), \
             patch.object(cli, 'load_nmm_spike_clip', return_value=clip), \
             patch.object(self.backend, 'available_regions', side_effect=AssertionError('unpacked clips need no ZIP repository')), \
             patch.object(self.backend, 'sample_regions', wraps=self.backend.sample_regions) as sample, \
             patch.object(self.backend, 'project_regions', wraps=self.backend.project_regions) as project, \
             patch.object(cli, 'save_standard_sample', side_effect=lambda *a, **kw: a[7]):
            manifest = cli.run_g2(args, Path('unused'))
        self.assertEqual(sample.call_args.args[0], 2)
        self.assertEqual(sample.call_args.kwargs['candidate_regions'], [0, 2])
        project.assert_called_once()
        self.assertEqual(set(manifest['selected_regions']), {0, 2})
        self.assertEqual(project.call_args.kwargs['spatial'].mode, 'diffuse')


if __name__ == '__main__':
    unittest.main()
