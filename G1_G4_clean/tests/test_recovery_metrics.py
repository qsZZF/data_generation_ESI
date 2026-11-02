import unittest
import numpy as np
from resolution_benchmark.nullspace import analyze_operator
from resolution_benchmark.recovery_metrics import score_recovery, support_diagnostics


class RecoveryMetricsTests(unittest.TestCase):
    def test_correct_support_can_predict_global_null_but_not_arbitrary_null(self):
        a = np.array([[1., 1.]])
        truth = np.array([[1., 2.], [0., 0.]])
        analysis = analyze_operator(a, np.eye(1))
        minimum_norm = np.linalg.pinv(a) @ (a @ truth)
        self.assertAlmostEqual(score_recovery(minimum_norm, truth, analysis)['null_recovery'], 0.)
        supported = np.array([[1.], [0.]]) @ (a @ truth)
        self.assertAlmostEqual(score_recovery(supported, truth, analysis)['null_recovery'], 1.)
        alternative = truth + np.array([[1., 2.], [-1., -2.]])
        np.testing.assert_allclose(a @ alternative, a @ truth)
        self.assertLess(score_recovery(supported, alternative, analysis)['null_recovery'], 1.)

    def test_standardized_map_has_shape_metrics_without_false_current_units(self):
        truth = np.array([[1., 2.], [3., 4.]])
        analysis = analyze_operator(np.array([[1., 1.]]), np.eye(1))
        score = score_recovery(100 * truth, truth, analysis, physical_units=False)
        self.assertAlmostEqual(score['shape_nrmse'], 0.)
        self.assertIsNone(score['source_nrmse'])
        self.assertIsNone(score['null_recovery'])
        self.assertIsNone(score['null_cosine'])
        self.assertIsNone(score['inferred_null_fraction'])

    def test_wrong_support_may_be_identifiable_but_inaccurate(self):
        result = support_diagnostics(np.array([[1., 1.]]), np.array([[1.], [0.]]), np.array([[0.], [1.]]))
        self.assertEqual(result['support_nullity'], 0)
        self.assertEqual(result['support_approximation_nrmse'], 1.)

    def test_orthogonal_error_decomposition(self):
        rng = np.random.default_rng(5)
        analysis = analyze_operator(rng.normal(size=(3, 6)), np.eye(3))
        score = score_recovery(rng.normal(size=(6, 10)), rng.normal(size=(6, 10)), analysis)
        self.assertAlmostEqual(score['source_nrmse'] ** 2,
                               score['observable_error_fraction'] + score['null_error_fraction'])
