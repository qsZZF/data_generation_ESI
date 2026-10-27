"""Whitening covariance comes from the independent baseline, with oracle diagnostics."""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from Simulation_G1.g1_simulation_core import G1SimulationConfig
from test_spatial_patches import grid_simulator


class G1NoiseCovarianceTests(unittest.TestCase):
    def setUp(self):
        self.sim = grid_simulator()
        self.config = G1SimulationConfig(
            n_patch=2, patch_area_mm2=180, min_seed_distance_mm=0,
            srate=100, time_window=(0, 1), random_seed=48,
            sensor_noise_type='iid_white', normalize_source=True)

    def test_sensor_only_covariance_matches_returned_noise(self):
        for average_reference in (False, True):
            for normalize_sensor in (False, True):
                with self.subTest(reference=average_reference, normalize=normalize_sensor):
                    config = replace(self.config, average_reference=average_reference,
                                     normalize_sensor=normalize_sensor, sensor_noise_peak_ratio=.2)
                    _, _, noise, _, metadata = self.sim.generate_sensor(config, compute_diagnostics=False)
                    expected = np.cov(noise.astype(float))
                    np.testing.assert_allclose(metadata['sensor_noise_covariance'], expected, atol=1e-14)
                    np.testing.assert_array_equal(metadata['oracle_noise_covariance'], metadata['sensor_noise_covariance'])
                    baseline = np.asarray(metadata['baseline_sensor'])
                    np.testing.assert_allclose(metadata['noise_covariance'], np.cov(baseline), atol=1e-14)
                    self.assertFalse(np.allclose(metadata['noise_covariance'], expected))
                    self.assertEqual(baseline.shape, (4, 100))
                    self.assertIsInstance(metadata['noise_covariance'], list)
                    json.dumps(metadata)
                    if average_reference:
                        self.assertLessEqual(metadata['noise_covariance_info']['rank'], 3)
                        np.testing.assert_allclose(np.asarray(metadata['noise_covariance']).sum(axis=0),
                                                   0, atol=1e-8 * np.linalg.norm(expected))

    def test_source_background_covariance_tracks_all_scaling_and_cross_terms(self):
        config = replace(self.config, source_background_type='pink', source_snr_db=5,
                         normalize_sensor=True)
        captured = {}
        make_background = self.sim._make_source_background

        def capture(source, *args, **kwargs):
            captured['foreground'] = source.copy()
            result = make_background(source, *args, **kwargs)
            mixed = source + result[:, kwargs.get('baseline_samples', 0):].astype(np.float32)
            mixed -= mixed.mean(axis=1, keepdims=True)
            captured['source_scale'] = max(float(np.max(np.abs(mixed))), 1e-12)
            return result

        with patch.object(self.sim, '_make_source_background', side_effect=capture):
            _, _, noise, source, metadata = self.sim.generate_sensor(config, compute_diagnostics=False)
        foreground = captured['foreground']
        foreground -= foreground.mean(axis=1, keepdims=True)
        foreground /= captured['source_scale']
        background_sensor = self.sim.leadfield @ (source.astype(float) - foreground.astype(float))
        background_sensor -= background_sensor.mean(axis=1, keepdims=True)
        background_sensor -= background_sensor.mean(axis=0, keepdims=True)
        background_sensor /= metadata['sensor_normalization_scale']
        expected = np.cov(background_sensor + noise.astype(float))
        np.testing.assert_allclose(metadata['oracle_noise_covariance'], expected, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(metadata['noise_covariance'], np.cov(np.asarray(metadata['baseline_sensor'])), atol=1e-14)
        np.testing.assert_allclose(metadata['sensor_noise_covariance'], np.cov(noise.astype(float)), atol=1e-14)
        self.assertFalse(np.allclose(expected, np.cov(background_sensor) + np.cov(noise.astype(float))))

    def test_disabled_sensor_noise_keeps_source_background_covariance(self):
        for background in (None, 'pink'):
            config = replace(self.config, sensor_snr_db=None, source_background_type=background,
                             source_snr_db=10 if background else None)
            _, _, noise, _, metadata = self.sim.generate_sensor(config, compute_diagnostics=False)
            np.testing.assert_array_equal(noise, 0)
            np.testing.assert_array_equal(metadata['sensor_noise_covariance'], np.zeros((4, 4)))
            self.assertEqual(metadata['sensor_noise_covariance_info']['rank'], 0)
            self.assertTrue(np.isfinite(metadata['event_sensor_snr_db']))
            if background:
                self.assertGreater(np.trace(metadata['noise_covariance']), 0)
            else:
                np.testing.assert_array_equal(metadata['noise_covariance'], np.zeros((4, 4)))

    def test_covariance_is_saved_even_without_signal_arrays(self):
        _, _, _, _, expected = self.sim.generate_sensor(self.config)
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self.sim.save_sample(tmp, self.config, save_sensor_noise=False)
            stored = json.loads((Path(tmp) / manifest['manifest']).read_text(encoding='utf-8'))
            with np.load(Path(tmp) / stored['noise_covariance_file'], allow_pickle=False) as arrays:
                for key in ('baseline_sensor', 'noise_covariance', 'oracle_noise_covariance', 'sensor_noise_covariance'):
                    self.assertEqual(arrays[key].dtype, np.float32 if key == 'baseline_sensor' else np.float64)
                    np.testing.assert_array_equal(arrays[key], expected[key])
                    self.assertNotIn(key, stored)
            self.assertNotIn('sensor_noise', stored)
            self.assertIn('noise_covariance_info', stored)

    def test_repeated_and_diagnostics_disabled_samples_are_identical(self):
        config = replace(self.config, source_background_type='pink', source_snr_db=10)
        full = self.sim.generate_sensor(config)
        quick = self.sim.generate_sensor(config, compute_diagnostics=False)
        repeated = self.sim.generate_sensor(config)
        self.assertEqual(len(full), 5)
        for index in range(4):
            np.testing.assert_array_equal(full[index], quick[index])
            np.testing.assert_array_equal(full[index], repeated[index])
        np.testing.assert_array_equal(full[3], self.sim.generate_source(config)[0])
        for key in ('baseline_sensor', 'noise_covariance', 'oracle_noise_covariance', 'sensor_noise_covariance'):
            self.assertEqual(full[4][key], quick[4][key])
            self.assertEqual(full[4][key], repeated[4][key])

    def test_baseline_uses_same_peak_and_sensor_normalization_gain(self):
        base = self.sim.generate_sensor(self.config, compute_diagnostics=False)
        capped = self.sim.generate_sensor(replace(self.config, sensor_noise_peak_ratio=.01), compute_diagnostics=False)
        gain = capped[4]['noise_peak_limit_scale']
        self.assertLess(gain, 1)
        np.testing.assert_allclose(capped[4]['baseline_sensor'], np.asarray(base[4]['baseline_sensor']) * gain,
                                   rtol=2e-7, atol=1e-9)
        normalized = self.sim.generate_sensor(replace(self.config, normalize_sensor=True), compute_diagnostics=False)
        np.testing.assert_allclose(normalized[4]['baseline_sensor'],
                                   np.asarray(base[4]['baseline_sensor']) / normalized[4]['sensor_normalization_scale'],
                                   rtol=2e-7, atol=1e-9)

    def test_source_background_baseline_has_no_foreground_and_uses_event_scale(self):
        config = replace(self.config, source_background_type='pink', source_snr_db=5,
                         sensor_snr_db=None, normalize_sensor=True)
        captured = {}
        make_background = self.sim._make_source_background

        def capture(source, *args, **kwargs):
            result = make_background(source, *args, **kwargs)
            count = kwargs['baseline_samples']
            mixed = source + result[:, count:].astype(np.float32)
            mixed -= mixed.mean(axis=1, keepdims=True)
            captured['baseline'] = result[:, :count].astype(np.float32)
            captured['baseline'] /= max(float(np.max(np.abs(mixed))), 1e-12)
            return result

        with patch.object(self.sim, '_make_source_background', side_effect=capture):
            *_, metadata = self.sim.generate_sensor(config, compute_diagnostics=False)
        expected = self.sim.leadfield @ captured['baseline']
        expected -= expected.mean(axis=1, keepdims=True)
        expected -= expected.mean(axis=0, keepdims=True)
        expected /= metadata['sensor_normalization_scale']
        np.testing.assert_array_equal(metadata['baseline_sensor'], expected.astype(np.float32))

    def test_baseline_is_disjoint_and_sensor_seed_controls_both_segments(self):
        config = replace(self.config, component_seeds=dict(spatial=1, waveform=2, gain=3, background=4, sensor=5))
        first = self.sim.generate_sensor(config, compute_diagnostics=False)
        second = self.sim.generate_sensor(replace(config, component_seeds={**config.component_seeds, 'sensor': 8}),
                                          compute_diagnostics=False)
        np.testing.assert_array_equal(first[3], second[3])
        self.assertFalse(np.array_equal(first[4]['baseline_sensor'], second[4]['baseline_sensor']))
        self.assertFalse(np.array_equal(first[4]['baseline_sensor'], first[2]))

    def test_baseline_duration_requires_at_least_two_whole_samples(self):
        for duration in (0, -1, .01, .025, True, float('nan'), float('inf')):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                self.sim.generate_sensor(replace(self.config, baseline_duration_s=duration), compute_diagnostics=False)


if __name__ == '__main__':
    unittest.main()
