import unittest
import numpy as np
from scripts.run_nullspace_methods import band_components, point_truth, temporal_sources, fit_key


class MethodDesignTests(unittest.TestCase):
    def test_refinement_key_pairs_methods_but_not_covariance_or_seed(self):
        row = dict(modality='meg', part='supports', covariance='baseline', seed=1,
                   scene='local_patch', nominal_full_array_snr_db=20., n_channels=66,
                   n_sources=400, support='points', method='sbl')
        self.assertEqual(fit_key(row), fit_key(dict(row, method='sbl_lcmv')))
        self.assertNotEqual(fit_key(row), fit_key(dict(row, covariance='reference')))
        self.assertNotEqual(fit_key(row), fit_key(dict(row, seed=2)))

    def test_bands_partition_signal_and_white_noise_variance(self):
        values = np.random.default_rng(1).normal(size=(6, 512))
        components = list(band_components(values))
        np.testing.assert_allclose(sum(x for x, _ in components), values, atol=1e-14)
        self.assertAlmostEqual(sum(f for _, f in components), 1.)
        self.assertAlmostEqual(sum(np.sum(x ** 2) for x, _ in components), np.sum(values ** 2))

    def test_nested_grid_keeps_truth_and_fit_test_have_same_support(self):
        training = point_truth('sparse', 400, 2, 0)
        test = point_truth('sparse', 400, 2, 1)
        np.testing.assert_array_equal(training[100:], 0)
        np.testing.assert_array_equal(np.any(training != 0, axis=1), np.any(test != 0, axis=1))
        self.assertFalse(np.allclose(training, test))

    def test_spectral_source_zero_mean_and_deterministic(self):
        a = temporal_sources(5, np.random.default_rng(42))
        b = temporal_sources(5, np.random.default_rng(42))
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(a.mean(axis=1), 0, atol=1e-15)
