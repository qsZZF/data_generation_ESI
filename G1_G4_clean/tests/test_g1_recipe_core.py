"""Replayable geometry, isolated random streams, and sensor-noise calibration."""
import json
import unittest
from dataclasses import asdict, replace
from unittest.mock import patch

import numpy as np

from Simulation_G1.g1_simulation_core import G1SimulationConfig, G1ScenarioSampler
from test_spatial_patches import grid_simulator


class RecipeCoreTests(unittest.TestCase):
    def setUp(self):
        self.sim = grid_simulator()
        self.config = G1SimulationConfig(
            n_patch=3, patch_area_mm2=180, min_seed_distance_mm=0,
            srate=100, time_window=(0, .5), random_seed=4, normalize_source=False)

    def test_legacy_seed_is_repeatable(self):
        config = replace(self.config, source_background_type='pink', source_snr_db=10)
        first = self.sim.generate_sensor(config)
        second = self.sim.generate_sensor(config)
        for left, right in zip(first[:4], second[:4]):
            np.testing.assert_array_equal(left, right)
        self.assertEqual(first[4], second[4])

    def resolved_config(self):
        config = replace(self.config,
            component_seeds=dict(spatial=1, waveform=2, gain=3, background=4, sensor=5),
            waveform_params=[('pink', {})], sensor_noise_type='iid_white')
        _, meta = self.sim.generate_source(config)
        return replace(config, seed_vertices=meta['seed_vertices'],
                       patch_vertices=[p['vertices'] for p in meta['patches']],
                       patch_gains=[p['gain'] for p in meta['patches']])

    def test_geometry_resolution_and_json_roundtrip_are_exact(self):
        resolved = self.resolved_config()
        sampled = replace(resolved, seed_vertices=None, patch_vertices=None, patch_gains=None)
        restored = G1SimulationConfig(**json.loads(json.dumps(asdict(resolved))))
        first = self.sim.generate_sensor(sampled)
        for config in (resolved, restored):
            with patch.object(self.sim, '_build_spatial_patches', side_effect=AssertionError('regrowth')):
                repeated = self.sim.generate_sensor(config)
            for left, right in zip(first[:4], repeated[:4]):
                np.testing.assert_array_equal(left, right)

    def test_sensor_seed_changes_only_noise(self):
        config = self.resolved_config()
        first = self.sim.generate_sensor(config)
        second = self.sim.generate_sensor(replace(config,
            component_seeds={**config.component_seeds, 'sensor': 55}))
        for index in (1, 3):
            np.testing.assert_array_equal(first[index], second[index])
        self.assertFalse(np.array_equal(first[2], second[2]))

    def test_background_seed_leaves_foreground_unchanged(self):
        config = self.resolved_config()
        changed = replace(config, component_seeds={**config.component_seeds, 'background': 66})
        first, _ = self.sim.generate_source(config)
        second, _ = self.sim.generate_source(changed)
        np.testing.assert_array_equal(first, second)
        backgrounds = []
        foregrounds = []
        make_background = self.sim._make_source_background
        def capture_background(source, *args, **kwargs):
            np.testing.assert_allclose(source, first, atol=1e-7)
            foregrounds.append(source.copy())
            result = make_background(source, *args, **kwargs)
            backgrounds.append(result)
            return result
        with patch.object(self.sim, '_make_source_background', side_effect=capture_background):
            for candidate in (config, changed):
                self.sim.generate_source(replace(candidate, source_background_type='pink', source_snr_db=10))
        self.assertFalse(np.array_equal(backgrounds[0], backgrounds[1]))
        np.testing.assert_array_equal(foregrounds[0], foregrounds[1])

    def test_synchronous_with_equal_gains_has_identical_center_traces(self):
        config = replace(self.resolved_config(), connectivity_mode='synchronous', patch_gains=[1., 1., 1.])
        source, _ = self.sim.generate_source(config)
        for seed in config.seed_vertices[1:]:
            np.testing.assert_array_equal(source[seed], source[config.seed_vertices[0]])

    def test_malformed_component_seeds_fail(self):
        good = self.resolved_config()
        invalid = [None, -1, 1.5, True, float('inf'), float('nan'), '4']
        mappings = [{}, {'spatial': 1}, {**good.component_seeds, 'extra': 3}]
        mappings.extend({**good.component_seeds, 'sensor': v} for v in invalid)
        for seeds in mappings:
            with self.subTest(seeds=seeds), self.assertRaisesRegex(ValueError, 'component_seeds'):
                self.sim.generate_source(replace(good, component_seeds=seeds))

    def test_invalid_supports_and_gains_fail(self):
        config = self.resolved_config()
        mutations = [
            dict(seed_vertices=None), dict(patch_vertices=[]),
            dict(patch_vertices=[[999]] * 3), dict(patch_vertices=[[1.5]] * 3),
            dict(patch_vertices=[p + [p[0]] for p in config.patch_vertices]),
            dict(patch_vertices=[[s] for s in config.seed_vertices]),
            dict(patch_gains=[1]), dict(patch_gains=[1, 0, 1]),
            dict(patch_gains=[1, float('nan'), 1]),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.sim.generate_source(replace(config, **mutation))
        for supports, centers, message in [([[0, 1, 9], [1, 2, 10]], [0, 2], 'overlap'),
                                           ([[0, 80]], [0], 'connected')]:
            with self.assertRaisesRegex(ValueError, message):
                self.sim.generate_source(replace(config, n_patch=len(centers), seed_vertices=centers,
                    patch_vertices=supports, patch_gains=None, patch_area_mm2=1))

    def test_peak_cap_scales_entire_noise_and_reports_actual_snr(self):
        config = replace(self.resolved_config(), waveform_params=[('gaussian_pulse',
            {'pulse_center_time_s': .2, 'pulse_width_time_s': .01})],
            sensor_snr_db=-20, sensor_snr_mode='global_rms')
        original = self.sim.generate_sensor(config)
        limited = self.sim.generate_sensor(replace(config, sensor_noise_peak_ratio=.2))
        self.assertTrue(limited[4]['noise_peak_limit_applied'])
        self.assertLessEqual(np.max(np.abs(limited[2])), .2*np.max(np.abs(limited[1])) + 1e-7)
        np.testing.assert_allclose(limited[2], original[2]*limited[4]['noise_peak_limit_scale'], rtol=2e-7)
        self.assertGreater(limited[4]['realized_sensor_snr_db'], config.sensor_snr_db)
        actual = 10*np.log10(np.mean(limited[1].astype(float)**2)/np.mean(limited[2].astype(float)**2))
        self.assertAlmostEqual(limited[4]['realized_sensor_snr_db'], actual, places=5)
        self.assertIn('event_sensor_snr_db', limited[4])
        uncapped = self.sim.generate_sensor(replace(config, sensor_noise_peak_ratio=1e6))
        np.testing.assert_array_equal(uncapped[2], original[2])
        for ratio in [0, -1, float('nan'), float('inf'), True]:
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                self.sim.generate_sensor(replace(config, sensor_noise_peak_ratio=ratio))

    def test_iid_white_and_optional_psd(self):
        config = self.resolved_config()
        full = self.sim.generate_sensor(config)
        with patch('Simulation_G1.g1_simulation_core.estimate_power_spectrum_summary',
                   side_effect=AssertionError('PSD disabled')):
            quick = self.sim.generate_sensor(config, compute_diagnostics=False)
        for left, right in zip(full[:4], quick[:4]):
            np.testing.assert_array_equal(left, right)
        self.assertNotIn('source_psd_summary', quick[4])
        self.assertIn('source_psd_summary', full[4])
        np.testing.assert_allclose(quick[2].mean(axis=0), 0, atol=1e-7)

    def test_explicit_waveform_sampling_keeps_requested_kind(self):
        sampler = G1ScenarioSampler(random_seed=8)
        for kind in sampler.waveforms:
            sampled, params = sampler._sample_waveform(kind)
            self.assertEqual(sampled, kind)
            self.assertIsInstance(params, dict)
        with self.assertRaisesRegex(ValueError, 'Unsupported sampled waveform'):
            sampler._sample_waveform('unknown')


if __name__ == '__main__':
    unittest.main()
