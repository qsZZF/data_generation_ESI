"""Opt-in residual fields preserve event semantics and shared baseline gains."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from Simulation_G1.geometry import compute_basis, save_basis
from Simulation_G1.g1_simulation_core import G1SimulationConfig
from Simulation_G1.spectral import generate_source_components, generate_rest_background
from test_spatial_patches import grid_simulator


class MultiscaleSpectralTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.sim = grid_simulator(5)
        self.sim.cortex_mask[np.arange(25) % 5 == 2] = False
        self.bundle = compute_basis(self.sim.vertices*self.sim.coord_to_mm, self.sim.faces, n_modes=5)
        self.path = Path(self.temp.name)/'basis.npz'
        save_basis(self.path, self.bundle)
        self.settings = dict(basis_path=str(self.path), n_modes=3, event_onset_s=.2,
                             event_duration_s=.5, erp_latency_jitter_s=.01, ersp_onset_jitter_s=.01)

    def config(self, scenario, fraction=.2):
        config = G1SimulationConfig(source_model='lbo', spectral={**self.settings, 'scenario': scenario},
            srate=128, time_window=(0, 1), random_seed=15, normalize_source=True)
        config.multiscale = dict(fraction=fraction, n_components=3, length_mm=5., seed=91)
        return config

    def test_disabled_is_exact_and_never_draws_patterns(self):
        for scenario in ('rest', 'erp', 'ersp'):
            config = self.config(scenario, 0)
            config.multiscale = {}
            first = generate_source_components(self.sim, config)
            config.multiscale = dict(fraction=0, seed=11)
            components = {}
            with patch('Simulation_G1.multiscale.generate_residual_patterns', side_effect=AssertionError('unexpected RNG')):
                second = generate_source_components(self.sim, config, components=components)
            for index in (0, 2, 3):
                if first[index] is not None:
                    np.testing.assert_array_equal(first[index], second[index])
            self.assertFalse(np.any(components['foreground_residual']))
            for values in (second[0], second[2], second[3]):
                if values is not None:
                    np.testing.assert_array_equal(values[~self.sim.cortex_mask], 0)

    def test_zero_fraction_has_continuous_coarse_field_and_rest_mask(self):
        for scenario in ('rest', 'erp', 'ersp'):
            config = self.config(scenario, 0)
            config.normalize_source = False
            zero_parts, small_parts = {}, {}
            zero = generate_source_components(self.sim, config, components=zero_parts)
            config.multiscale['fraction'] = 1e-10
            small = generate_source_components(self.sim, config, components=small_parts)
            for key in zero_parts:
                if key.endswith('_coarse'):
                    # Disabled truth absorbs float32 output rounding; the enabled
                    # coarse field is retained at float64 before residual closure.
                    tolerance = 4*np.finfo(np.float32).eps*np.max(np.abs(zero_parts[key]))
                    np.testing.assert_allclose(zero_parts[key], small_parts[key], atol=tolerance, rtol=0)
            self.assertLess(np.linalg.norm(small[0]-zero[0])/np.linalg.norm(zero[0]), 1e-4)
        event, baseline, _ = generate_rest_background(self.sim, self.settings, srate=128,
            time_window=(0, 1), baseline_duration_s=2, seed=3)
        np.testing.assert_array_equal(event[~self.sim.cortex_mask], 0)
        np.testing.assert_array_equal(baseline[~self.sim.cortex_mask], 0)

    def test_exported_erp_metadata_describes_temporal_centering(self):
        source, metadata, background, _ = generate_source_components(self.sim, self.config('erp', 0))
        self.assertIn('causal', metadata['coefficient_erp_support'])
        self.assertNotIn('erp_support', metadata)
        self.assertIn('constant offset', metadata['exported_foreground_centering'])
        foreground = source.astype(float)-background
        prestim = np.arange(128)/128 < self.settings['event_onset_s']
        self.assertGreater(np.max(np.abs(foreground[:, prestim])), 0)
        np.testing.assert_allclose(np.ptp(foreground[:, prestim], axis=1), 0, atol=1e-12)

    def test_rest_erp_ersp_fraction_masks_and_component_truth(self):
        for scenario in ('rest', 'erp', 'ersp'):
            with self.subTest(scenario=scenario):
                components = {}
                source, info, background, baseline = generate_source_components(
                    self.sim, self.config(scenario), components=components)
                coarse, residual = components['foreground_coarse'], components['foreground_residual']
                fraction = np.mean(residual**2)/(np.mean(coarse**2)+np.mean(residual**2))
                self.assertAlmostEqual(fraction, .2, places=6)
                np.testing.assert_allclose(coarse+residual, source if background is None else source-background, atol=1e-12)
                for values in components.values():
                    np.testing.assert_array_equal(values[~self.sim.cortex_mask], 0)
                self.assertAlmostEqual(info['multiscale_metadata']['mixing']['realized_component_fraction'], .2)
                self.assertEqual(info['multiscale'], self.config(scenario).multiscale)
                if background is not None:
                    np.testing.assert_allclose(components['background_coarse']+components['background_residual'], background, atol=1e-12)
                    np.testing.assert_allclose(components['baseline_background_coarse']+components['baseline_background_residual'], baseline, atol=1e-12)
                if scenario == 'erp':
                    self.assertEqual(info['erp_latency_s'], info['multiscale_metadata']['temporal']['erp_latency_s'])
                if scenario == 'ersp':
                    self.assertEqual(info['ersp_onset_s'], info['multiscale_metadata']['temporal']['ersp_onset_s'])

    def test_rest_background_shared_event_gain_and_replay(self):
        kwargs = dict(srate=128, time_window=(0, 1), baseline_duration_s=2., seed=13)
        multiscale = dict(fraction=.2, n_components=3, length_mm=5., seed=91)
        parts_a, parts_b = {}, {}
        a = generate_rest_background(self.sim, self.settings, **kwargs, multiscale=multiscale, components=parts_a)
        b = generate_rest_background(self.sim, self.settings, **kwargs,
                multiscale={**multiscale, 'fraction': .4}, components=parts_b)
        gain_ratio = b[2]['multiscale_metadata']['mixing']['residual_scale']/a[2]['multiscale_metadata']['mixing']['residual_scale']
        for prefix in ('background', 'baseline_background'):
            np.testing.assert_allclose(parts_b[prefix+'_residual'], gain_ratio*parts_a[prefix+'_residual'], rtol=1e-12, atol=1e-14)
            np.testing.assert_array_equal(parts_a[prefix+'_coarse'], parts_b[prefix+'_coarse'])
        replay = generate_rest_background(self.sim, self.settings, **kwargs, multiscale=multiscale)
        np.testing.assert_array_equal(a[0], replay[0])
        np.testing.assert_array_equal(a[1], replay[1])
        np.testing.assert_allclose(parts_a['background_coarse']+parts_a['background_residual'], a[0])

    def test_unknown_settings_and_invalid_fraction_fail(self):
        for settings in ({'fraction':1}, {'fraction':-.2}, {'fraction':.2, 'typo':4}):
            config = self.config('rest')
            config.multiscale = settings
            with self.assertRaises(ValueError):
                generate_source_components(self.sim, config)


if __name__ == '__main__':
    unittest.main()
