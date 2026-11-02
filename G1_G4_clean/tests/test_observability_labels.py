"""Observable source energy is a subspace diagnostic, not an inverse estimate."""
import json
import unittest

import numpy as np

from Simulation_G1.observability import analyze_observability


class ObservabilityLabelTests(unittest.TestCase):
    def test_known_null_and_distinct_whitened_partition(self):
        gain = np.array([[1., 0., 0.], [0., 1., 0.]])
        source = np.array([[1., -1.], [2., -2.], [3., -3.]])
        result = analyze_observability(gain, source, np.diag([1., 0.]))
        self.assertEqual(result['geometric_rank'], 2)
        self.assertEqual(result['whitened_rank'], 1)
        self.assertEqual(result['lost_noise_directions'], 1)
        self.assertAlmostEqual(result['geometric_null_energy_fraction'], 9 / 14)
        self.assertAlmostEqual(result['white_null_energy_fraction'], 13 / 14)
        self.assertAlmostEqual(result['geometric_energy_lost_by_noise_projection_fraction'], 4 / 14)
        self.assertAlmostEqual(result['geometric_observable_energy_fraction'] + result['geometric_null_energy_fraction'], 1)
        self.assertAlmostEqual(sum(result[k] for k in ('white_stable_energy_fraction', 'white_near_null_energy_fraction', 'white_null_energy_fraction')), 1)
        json.dumps(result, allow_nan=False)

    def test_channel_unit_scaling_does_not_change_ranks_or_energy(self):
        rng = np.random.default_rng(701)
        gain, source = rng.normal(size=(5, 9)), rng.normal(size=(9, 19))
        covariance = np.diag([1., 2., 3., 4., 5.])
        units = np.array([1e-15, 1e-13, 1e-9, 1e-6, 1.])
        original = analyze_observability(gain, source, covariance)
        scaled = analyze_observability(gain * units[:, None], source,
                                       covariance * units[:, None] * units[None, :])
        for key in ('geometric_rank', 'whitened_rank', 'geometric_null_energy_fraction',
                    'white_null_energy_fraction', 'isotropic_wiener_weighted_energy_fraction',
                    'isotropic_wiener_effective_dof', 'whitened_target_to_baseline_power_db'):
            self.assertAlmostEqual(original[key], scaled[key], places=10)

    def test_whitened_target_power_uses_all_noise_directions(self):
        gain = np.array([[2., 0.], [0., 0.], [0., 0.]])
        source = np.array([[1., -1.], [3., -3.]])
        result = analyze_observability(gain, source, np.eye(3))
        self.assertEqual(result['noise_rank'], 3)
        self.assertEqual(result['whitened_rank'], 1)
        self.assertAlmostEqual(result['whitened_target_to_baseline_power_db'], 10 * np.log10(4 / 3))
        # The denominator refers to total baseline nuisance, not measurement-only noise.
        self.assertEqual(result['whitened_target_power_reference'], 'total_baseline_nuisance_unit_variance')
        zero = analyze_observability(gain, np.zeros_like(source), np.eye(3))
        self.assertIsNone(zero['whitened_target_to_baseline_power_db'])
        null = analyze_observability(gain, np.array([[0., 0.], [1., -1.]]), np.eye(3))
        self.assertIsNone(null['whitened_target_to_baseline_power_db'])
        tiny = analyze_observability(np.diag([1., 1e-12]), np.array([[0.], [1.]]), np.eye(2))
        self.assertEqual(tiny['whitened_rank'], 1)
        self.assertAlmostEqual(tiny['whitened_target_to_baseline_power_db'], 10 * np.log10(1e-24 / 2))
        huge = analyze_observability(np.array([[1e200]]), np.ones((1, 3)), np.eye(1))
        self.assertAlmostEqual(huge['whitened_target_to_baseline_power_db'], 4000.)

    def test_reference_constraint_and_zero_noise(self):
        source = np.ones((4, 8))
        result = analyze_observability(np.eye(4), source, np.zeros((4, 4)), average_reference=True)
        self.assertEqual(result['geometric_rank'], 3)
        self.assertAlmostEqual(result['geometric_null_energy_fraction'], 1)
        self.assertEqual(result['noise_status'], 'zero_noise_covariance')
        self.assertIsNone(result['whitened_rank'])
        absent = analyze_observability(np.eye(4), source)
        self.assertEqual(absent['noise_status'], 'not_provided')
        self.assertEqual(absent['geometric_rank'], 4)

    def test_projection_of_entirely_removed_signal_and_noise_stays_zero(self):
        result = analyze_observability(np.ones((4, 2)), np.ones((2, 3)),
                                       np.ones((4, 4)), average_reference=True)
        self.assertEqual(result['geometric_rank'], 0)
        self.assertEqual(result['noise_status'], 'zero_noise_covariance')
        self.assertEqual(result['geometric_null_energy_fraction'], 1.)

    def test_near_null_threshold_is_not_noise_resolvability(self):
        gain = np.diag([1., 1e-4])
        source = np.eye(2)
        result = analyze_observability(gain, source, np.eye(2), near_null_rtol=1e-3)
        self.assertEqual(result['geometric_rank'], 2)
        self.assertEqual(result['whitened_rank'], 2)
        self.assertEqual(result['white_stable_rank'], 1)
        self.assertAlmostEqual(result['white_near_null_energy_fraction'], .5)
        self.assertLess(result['isotropic_wiener_effective_dof'], 1.)

    def test_degenerate_modes_have_rotation_invariant_continuous_weights(self):
        rng = np.random.default_rng(702)
        rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        source = rng.normal(size=(4, 20))
        first = analyze_observability(np.eye(4), source, np.eye(4))
        second = analyze_observability(rotation, source, np.eye(4))
        for key in ('isotropic_wiener_weighted_energy_fraction', 'isotropic_wiener_effective_dof',
                    'white_stable_energy_fraction'):
            self.assertAlmostEqual(first[key], second[key], places=12)

    def test_cache_tracks_covariance_and_recomputes_sample_weights(self):
        cache = {}
        gain, source = np.eye(3), np.eye(3)
        first = analyze_observability(gain, source, np.eye(3), cache=cache)
        repeated = analyze_observability(gain, source * 3, np.eye(3), cache=cache)
        changed = analyze_observability(gain, source, np.eye(3) * 4, cache=cache)
        self.assertFalse(first['cache_hit'])
        self.assertTrue(repeated['cache_hit'])
        self.assertEqual(first['decomposition_key'], repeated['decomposition_key'])
        self.assertNotEqual(first['source_sha256'], repeated['source_sha256'])
        self.assertGreater(repeated['isotropic_wiener_weighted_energy_fraction'], first['isotropic_wiener_weighted_energy_fraction'])
        self.assertNotEqual(first['decomposition_key'], changed['decomposition_key'])
        self.assertFalse(changed['cache_hit'])
        analyze_observability(gain * 2, source, np.eye(3), cache=cache)
        self.assertEqual(len(cache), 2)

    def test_zero_source_and_invalid_inputs(self):
        result = analyze_observability(np.eye(2), np.zeros((2, 4)), np.eye(2))
        self.assertIsNone(result['geometric_null_energy_fraction'])
        self.assertEqual(result['isotropic_wiener_effective_dof'], 0.)
        json.dumps(result, allow_nan=False)
        for kwargs in ({'near_null_rtol': 1e-12}, {'numerical_rtol': 0},
                       {'projector': np.eye(2) * .5}, {'noise_covariance': np.eye(3)},
                       {'noise_covariance': np.array([[1., .5], [0., 1.]])}):
            with self.assertRaises(ValueError):
                analyze_observability(np.eye(2), np.ones((2, 4)), **kwargs)


if __name__ == '__main__':
    unittest.main()
