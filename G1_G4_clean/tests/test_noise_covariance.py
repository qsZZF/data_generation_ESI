import unittest

import numpy as np

from Simulation_G1.noise_covariance import (covariance_payload, noise_covariance, whitener_from_covariance,
    baseline_covariance_payload, baseline_sample_count, sensor_noise_with_baseline)


class NoiseCovarianceTests(unittest.TestCase):
    def test_default_covariance_uses_baseline_not_activity_noise(self):
        rng = np.random.default_rng(81)
        baseline = rng.normal(size=(4, 800)) * 1e-6
        event_noise = rng.normal(size=(4, 100)) * 2e-6
        arrays, info = baseline_covariance_payload(baseline, event_noise)
        np.testing.assert_allclose(arrays['noise_covariance'], np.cov(baseline.astype(np.float32).astype(float)))
        np.testing.assert_allclose(arrays['oracle_noise_covariance'], np.cov(event_noise))
        self.assertEqual(info['noise_covariance_info']['provenance'], 'generated_baseline')
        self.assertEqual(info['noise_covariance_info']['n_samples'], 800)
        self.assertEqual(info['oracle_noise_covariance_info']['n_samples'], 100)

    def test_joint_sensor_baseline_retains_noise_scale_and_disjoint_data(self):
        from Simulation_G2.g2_simulation_core import G2SourceSimulator
        backend = object.__new__(G2SourceSimulator)
        rng = np.random.default_rng(82)
        clean = rng.normal(size=(4, 200))
        kwargs = dict(srate=200, sensor_noise_type='iid_white', snr_db=10, seed=83,
                      baseline_samples=300, average_reference=False)
        event, baseline, info = sensor_noise_with_baseline(backend, clean, **kwargs)
        again, repeated, _ = sensor_noise_with_baseline(backend, clean, **kwargs)
        np.testing.assert_array_equal(event, again)
        np.testing.assert_array_equal(baseline, repeated)
        self.assertEqual(info['baseline_interval_samples'], [0, 300])
        self.assertEqual(info['event_interval_samples'], [300, 500])
        self.assertAlmostEqual(np.mean(clean**2) / np.mean(event**2), 10)
        scaled_event, scaled_baseline, _ = sensor_noise_with_baseline(backend, clean * 3, **kwargs)
        np.testing.assert_allclose(scaled_event, event * 3)
        np.testing.assert_allclose(scaled_baseline, baseline * 3)
        with self.assertRaisesRegex(ValueError, 'no tiling'):
            sensor_noise_with_baseline(backend, clean, **{**kwargs, 'sensor_noise_type':'real'},
                                       real_sensor_noise=rng.normal(size=clean.shape))

    def test_baseline_sample_grid_validation(self):
        self.assertEqual(baseline_sample_count(1, 256), 256)
        for value in (0, -1, True, .001, np.nan, np.inf):
            with self.assertRaises(ValueError):
                baseline_sample_count(value, 256)

    def test_covariance_whitens_referenced_noise_and_leadfield_consistently(self):
        rng = np.random.default_rng(31)
        noise = rng.normal(size=(6, 500)) * np.arange(1, 7)[:, None] * 1e-6
        noise -= noise.mean(axis=0, keepdims=True)
        covariance, info = noise_covariance(noise, average_reference=True)
        np.testing.assert_allclose(covariance, np.cov(noise), rtol=1e-13)
        self.assertEqual(info['rank'], 5)
        whitener = whitener_from_covariance(covariance)
        self.assertEqual(whitener.shape, (5, 6))
        np.testing.assert_allclose(np.cov(whitener @ noise), np.eye(5), atol=1e-12)
        leadfield, source = rng.normal(size=(6, 9)), rng.normal(size=(9, 500))
        np.testing.assert_allclose(whitener @ (leadfield @ source), (whitener @ leadfield) @ source, atol=1e-8)

    def test_total_includes_cross_terms_and_quadratic_scaling(self):
        rng = np.random.default_rng(32)
        noise = rng.normal(size=(4, 100))
        background = noise * 2
        arrays, _ = covariance_payload(noise, background_sensor=background)
        np.testing.assert_allclose(arrays['noise_covariance'], 9 * np.cov(noise))
        np.testing.assert_allclose(arrays['sensor_noise_covariance'], np.cov(noise))
        covariance, _ = noise_covariance(noise * 0.1)
        np.testing.assert_allclose(covariance, np.cov(noise) * 0.01)

    def test_zero_and_short_window_rank(self):
        covariance, info = noise_covariance(np.zeros((4, 10)))
        self.assertEqual(info['rank'], 0)
        with self.assertRaisesRegex(ValueError, 'skip whitening'):
            whitener_from_covariance(covariance)
        covariance, info = noise_covariance(np.random.default_rng(33).normal(size=(6, 3)))
        self.assertEqual(info['rank'], 2)
        w = whitener_from_covariance(covariance)
        np.testing.assert_allclose(w @ covariance @ w.T, np.eye(2), atol=1e-12)

    def test_invalid_input_is_rejected(self):
        for value in (np.zeros((3, 1)), np.zeros(3), np.full((2, 4), np.nan)):
            with self.assertRaises(ValueError):
                noise_covariance(value)
        for value in (np.diag([1., -1.]), np.array([[1., 1.], [0., 1.]])):
            with self.assertRaises(ValueError):
                whitener_from_covariance(value)
