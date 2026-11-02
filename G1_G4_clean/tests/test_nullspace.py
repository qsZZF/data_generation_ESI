"""Linear-algebra checks with controlled identifiable/unidentifiable sources."""

import json
import unittest

import numpy as np

from resolution_benchmark.nullspace import (
    analyze_operator, projector_from_patterns, solve_svd, unit_safe_whitener,
)


class WhiteningTests(unittest.TestCase):
    def test_pattern_projector_with_dependent_and_empty_columns(self):
        pattern = np.array([[1., 2.], [1., 2.], [1., 2.]])
        projector = projector_from_patterns(pattern)
        np.testing.assert_allclose(projector @ pattern, 0, atol=1e-14)
        np.testing.assert_allclose(projector @ projector, projector, atol=1e-14)
        np.testing.assert_allclose(projector.T, projector, atol=1e-14)
        np.testing.assert_array_equal(projector_from_patterns(np.zeros((3, 0))), np.eye(3))
        np.testing.assert_array_equal(projector_from_patterns(np.zeros((3, 2))), np.eye(3))

    def test_mixed_sensor_units_preserve_rank_and_operator_geometry(self):
        rng = np.random.default_rng(19)
        factor = rng.normal(size=(6, 6))
        covariance = factor @ factor.T + np.eye(6)
        leadfield = rng.normal(size=(6, 9))
        source = rng.normal(size=(9, 4))
        original = analyze_operator(leadfield, covariance, source=source)
        units = np.array([1e-15, 2e-15, 1e-12, 4e-12, 1e-6, 3e-6])
        scaled_cov = covariance * units[:, None] * units[None, :]
        scaled = analyze_operator(leadfield * units[:, None], scaled_cov, source=source)
        np.testing.assert_allclose(scaled["whitener"] @ scaled_cov @ scaled["whitener"].T,
                                   np.eye(6), atol=2e-14)
        np.testing.assert_allclose(original["singular_values"], scaled["singular_values"], rtol=1e-12)
        np.testing.assert_allclose(original["observable_source"], scaled["observable_source"], atol=1e-12)
        np.testing.assert_allclose(original["operator"].T @ original["operator"],
                                   scaled["operator"].T @ scaled["operator"], atol=1e-12)
        data = leadfield @ source
        for method in ("pinv", "tsvd", "ridge"):
            np.testing.assert_allclose(solve_svd(original, data, method=method),
                                       solve_svd(scaled, data * units[:, None], method=method),
                                       atol=1e-12)

    def test_car_covariance_rank_and_zero_variance_sensor(self):
        car = projector_from_patterns(np.ones((5, 1)))
        whitener, info = unit_safe_whitener(car)
        self.assertEqual(info["rank"], 4)
        np.testing.assert_allclose(whitener @ car @ whitener.T, np.eye(4), atol=1e-14)
        whitener, info = unit_safe_whitener(np.diag([1e-30, 1e-12, 0]))
        self.assertEqual(info["rank"], 2)
        self.assertEqual(info["n_zero_variance_channels"], 1)
        np.testing.assert_allclose(whitener @ np.diag([1e-30, 1e-12, 0]) @ whitener.T, np.eye(2))

    def test_indefinite_asymmetric_and_empty_covariances_rejected(self):
        for covariance in (np.zeros((2, 2)), np.array([[1., 2.], [2., 1.]]),
                           np.array([[1., .2], [0., 1.]]), np.array([[0., 1.], [1., 1.]]),
                           np.diag([-1., 1.]), np.ones((2, 3)), np.full((2, 2), np.nan)):
            with self.subTest(covariance=covariance), self.assertRaises(ValueError):
                unit_safe_whitener(covariance)
        almost_psd = np.array([[1., 1 + 1e-11], [1 + 1e-11, 1.]])
        _, info = unit_safe_whitener(almost_psd)
        self.assertEqual(info["rank"], 1)


class OperatorTests(unittest.TestCase):
    def test_underdetermined_pinv_recovers_only_observable_source(self):
        rng = np.random.default_rng(2)
        leadfield = rng.normal(size=(4, 9))
        source = rng.normal(size=(9, 20))
        analysis = analyze_operator(leadfield, np.eye(4), source=source)
        info = analysis["info"]
        self.assertEqual(info["rank"], 4)
        self.assertEqual(info["nullity"], 5)
        self.assertFalse(info["full_column_rank"])
        self.assertIsNone(info["condition_number"])
        np.testing.assert_allclose(leadfield @ analysis["null_source"], 0, atol=1e-13)
        estimate = solve_svd(analysis, leadfield @ source)
        np.testing.assert_allclose(estimate, analysis["observable_source"], atol=1e-13)
        fractions = info["source_energy_fractions"]
        self.assertAlmostEqual(fractions["observable"] + fractions["null"], 1)
        self.assertAlmostEqual(fractions["stable"] + fractions["near_null"] + fractions["null"], 1)
        json.dumps(info, allow_nan=False)

    def test_controlled_near_null_and_threshold_numerical_null(self):
        leadfield = np.diag([1., 1e-4, 1e-12])
        source = np.ones(3)
        analysis = analyze_operator(leadfield, np.eye(3), source=source)
        self.assertEqual(analysis["info"]["rank"], 2)
        self.assertEqual(analysis["info"]["effective_rank"], 1)
        np.testing.assert_array_equal(analysis["near_null_source"], [0, 1, 0])
        np.testing.assert_array_equal(analysis["null_source"], [0, 0, 1])
        np.testing.assert_allclose(solve_svd(analysis, leadfield @ source), [1, 1, 0])
        np.testing.assert_allclose(solve_svd(analysis, leadfield @ source, method="tsvd"), [1, 0, 0])
        self.assertAlmostEqual(analysis["info"]["null_sensor_residual_norm"], 1e-12)

    def test_more_nested_sensor_rows_reduce_null_energy_for_fixed_truth(self):
        rng = np.random.default_rng(14)
        leadfield = rng.normal(size=(8, 12))
        truth = rng.normal(size=(12, 3))
        null_energy = []
        for n_channels in (2, 4, 6, 8):
            result = analyze_operator(leadfield[:n_channels], np.eye(n_channels), source=truth)
            null_energy.append(result["info"]["source_energy_fractions"]["null"])
        self.assertTrue(np.all(np.diff(null_energy) <= 1e-13))

    def test_more_nested_source_columns_increase_null_energy_for_embedded_fixed_truth(self):
        rng = np.random.default_rng(18)
        leadfield = rng.normal(size=(5, 12))
        fixed_truth = rng.normal(size=(3, 4))
        null_energy = []
        for n_sources in (3, 5, 8, 12):
            embedded = np.zeros((n_sources, 4))
            embedded[:3] = fixed_truth
            result = analyze_operator(leadfield[:, :n_sources], np.eye(5), source=embedded)
            null_energy.append(result["info"]["source_energy_fractions"]["null"])
        self.assertTrue(np.all(np.diff(null_energy) >= -1e-13))

    def test_full_column_rank_has_condition_and_recovers_clean_truth(self):
        leadfield = np.array([[1., 0], [0, 2], [0, 0]])
        source = np.array([3., -5.])
        analysis = analyze_operator(leadfield, np.eye(3), source=source)
        self.assertTrue(analysis["info"]["full_column_rank"])
        self.assertEqual(analysis["info"]["condition_number"], 2)
        np.testing.assert_allclose(solve_svd(analysis, leadfield @ source), source)

    def test_positive_definite_ridge_does_not_recover_null_truth(self):
        leadfield = np.array([[1., 0, 0], [0, 2, 0]])
        source = np.array([1., 2., 50.])
        analysis = analyze_operator(leadfield, np.eye(2), source=source)
        estimate = solve_svd(analysis, leadfield @ source, method="ridge", ridge_relative=.1)
        np.testing.assert_allclose(estimate, [1 / 1.04, 8 / 4.04, 0])
        self.assertGreater(np.linalg.eigvalsh(leadfield.T @ leadfield + .04 * np.eye(3))[0], 0)
        self.assertEqual(estimate[2], 0)
        self.assertNotEqual(source[2], 0)

    def test_zero_operator_and_zero_truth_have_explicit_undefined_ratios(self):
        analysis = analyze_operator(np.zeros((2, 3)), np.eye(2), source=np.zeros(3))
        self.assertEqual(analysis["info"]["rank"], 0)
        self.assertIsNone(analysis["info"]["nonzero_condition"])
        self.assertIsNone(analysis["info"]["source_energy_fractions"]["null"])
        for method in ("pinv", "tsvd", "ridge"):
            np.testing.assert_array_equal(solve_svd(analysis, [1, 2], method=method), np.zeros(3))
        json.dumps(analysis["info"], allow_nan=False)

    def test_bad_inputs_rejected(self):
        with self.assertRaises(ValueError):
            analyze_operator(np.eye(2), np.eye(3))
        with self.assertRaises(ValueError):
            analyze_operator(np.eye(2), np.eye(2), effective_rtol=1e-12)
        with self.assertRaises(ValueError):
            analyze_operator(np.eye(2), np.eye(2), source=np.ones(3))
        analysis = analyze_operator(np.eye(2), np.eye(2))
        for options in ({"method": "unknown"}, {"method": "ridge", "ridge_relative": 0}):
            with self.assertRaises(ValueError):
                solve_svd(analysis, [1, 2], **options)
        with self.assertRaises(ValueError):
            solve_svd(analysis, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
