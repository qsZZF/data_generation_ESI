"""Separate global-mean recovery from recovery of within-draw regional structure."""
import json
import unittest

import numpy as np

from Simulation_G4.evaluation import recovery_detail


class RecoveryDetailTests(unittest.TestCase):
    def test_global_only_prediction_does_not_recover_regional_residuals(self):
        global_mean = np.arange(10, dtype=float)[:, None] * 10
        regional = np.array([-1., 0., 1.])[None, :]
        truth = global_mean + regional
        prediction = np.broadcast_to(global_mean, truth.shape)
        prior = np.zeros_like(truth)
        report = recovery_detail({'theta': prediction}, {'theta': truth}, {'theta': prior})['theta']
        score = report['prediction']
        self.assertGreater(score['pooled']['r2'], .99)
        self.assertEqual(score['whole_brain_mean'], {'mae': 0., 'r2': 1.})
        self.assertAlmostEqual(score['within_draw_regional_residual']['r2'], 0.)
        self.assertAlmostEqual(score['within_draw_regional_residual']['mae'], 2 / 3)
        self.assertEqual([region['mae'] for region in score['per_region']], [1., 0., 1.])
        self.assertAlmostEqual(report['improvement']['within_draw_regional_residual']['r2_gain'], 0.)
        self.assertGreater(report['improvement']['whole_brain_mean']['mae_reduction'], 0)
        json.dumps(report, allow_nan=False)

    def test_perfect_prediction_scores_and_prior_improvements(self):
        truth = np.array([[1., 4.], [3., 5.], [6., 2.]])
        baseline = np.full_like(truth, 3.)
        report = recovery_detail({'C1': truth}, {'C1': truth}, {'C1': baseline})['C1']
        for key in ('pooled', 'whole_brain_mean', 'within_draw_regional_residual'):
            self.assertEqual(report['prediction'][key], {'mae': 0., 'r2': 1.})
            self.assertEqual(report['improvement'][key]['mae_reduction_fraction'], 1.)
            self.assertEqual(report['improvement'][key]['mae_reduction'], report['prior'][key]['mae'])
        for region in report['prediction']['per_region']:
            self.assertEqual(region, {'mae': 0., 'r2': 1.})

    def test_delay_has_only_scalar_scores_and_undefined_values_are_json_null(self):
        truth = {'C1': np.ones((3, 2)), 'delay_ms': np.ones((3, 1))}
        prediction = {'C1': np.ones((3, 2)), 'delay_ms': np.ones(3)}
        report = recovery_detail(prediction, truth, truth)
        self.assertEqual(set(report['delay_ms']['prediction']), {'pooled'})
        self.assertIsNone(report['delay_ms']['prediction']['pooled']['r2'])
        self.assertIsNone(report['delay_ms']['improvement']['pooled']['mae_reduction_fraction'])
        self.assertIsNone(report['C1']['prediction']['within_draw_regional_residual']['r2'])
        self.assertIn('null', json.dumps(report, allow_nan=False))

    def test_shapes_family_axes_and_nonfinite_inputs_fail(self):
        valid = {'theta': np.ones((3, 2))}
        invalid = [
            {}, {'C1': np.ones((3, 2))}, {'theta': np.ones(3)},
            {'theta': np.ones((3, 2, 1))}, {'theta': np.ones((3, 1))},
            {'theta': np.ones((0, 2))}, {'theta': np.full((3, 2), np.nan)},
            {'theta': np.full((3, 2), np.inf)},
        ]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                recovery_detail(values, valid, valid)
            with self.assertRaises(ValueError):
                recovery_detail(valid, valid, values)
        for values in (
            {'delay_ms': np.ones((3, 2))},
            {'theta': np.ones((3, 2)), 'C1': np.ones((4, 2))},
            {'theta': np.ones((3, 2)), 'C1': np.ones((3, 4))},
            {'theta': np.ones((3, 2)), 'delay_ms': np.ones((4, 1))},
        ):
            with self.subTest(families=list(values)), self.assertRaises(ValueError):
                recovery_detail(values, values, values)

    def test_baseline_is_not_estimated_from_targets_and_inputs_are_unchanged(self):
        target = {'C1': np.array([[1., 2.], [4., 8.]])}
        prediction = {'C1': target['C1'] + 1}
        prior = {'C1': np.full((2, 2), 10.)}
        copies = [values['C1'].copy() for values in (prediction, target, prior)]
        report = recovery_detail(prediction, target, prior)['C1']
        self.assertEqual(report['prior']['pooled']['mae'], np.mean(abs(prior['C1'] - target['C1'])))
        for values, saved in zip((prediction, target, prior), copies):
            np.testing.assert_array_equal(values['C1'], saved)


if __name__ == '__main__':
    unittest.main()
