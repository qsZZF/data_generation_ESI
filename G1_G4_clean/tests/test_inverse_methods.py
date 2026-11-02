"""Analytic and invariance checks for independently fitted inverse operators."""
import unittest

import numpy as np

from resolution_benchmark.inverse_methods import fit_inverse, sbl_active_operator


class TestInverseMethods(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(45)
        self.A = rng.normal(size=(9, 15))
        source = np.zeros((15, 180))
        source[[2, 7]] = 4 * rng.normal(size=(2, 180))
        self.Y = self.A @ source + rng.normal(size=(9, 180))

    def test_unit_gain_for_both_beamformers(self):
        for method in ("lcmv", "sbl_lcmv"):
            fit = fit_inverse(self.A, self.Y, method)
            np.testing.assert_allclose(np.diag(fit["operator"] @ self.A), 1, atol=1e-12)

    def test_sbl_evidence_and_covariance(self):
        fit = fit_inverse(self.A, self.Y, "sbl")
        history = fit["evidence_history"]
        self.assertGreater(len(history), 2)
        self.assertTrue(np.all(np.diff(history) <= 0))
        self.assertLess(history[-1], history[0])
        np.testing.assert_allclose(fit["model_cov"],
                                   (self.A * fit["gamma"]) @ self.A.T + np.eye(9), atol=1e-11)
        reference = fit["gamma"][:, None] * np.linalg.solve(fit["model_cov"], self.A).T
        np.testing.assert_allclose(fit["operator"], reference, atol=1e-12)

    def test_sbl_column_unit_changes_restore_physical_units(self):
        scale = np.geomspace(.001, 1000, self.A.shape[1])
        for method in ("sbl", "sbl_lcmv"):
            original = fit_inverse(self.A, self.Y, method)
            changed = fit_inverse(self.A * scale, self.Y, method)
            np.testing.assert_allclose(changed["operator"] * scale[:, None],
                                       original["operator"], rtol=1e-8, atol=1e-10)
            np.testing.assert_allclose(changed["model_cov"], original["model_cov"], rtol=1e-8)

    def test_sloreta_is_mne_resolution_standardization(self):
        mne = fit_inverse(self.A, self.Y, "mne")["operator"]
        fit = fit_inverse(self.A, self.Y, "sloreta")
        expected = mne / np.sqrt(np.diag(mne @ self.A))[:, None]
        np.testing.assert_allclose(fit["operator"], expected, atol=1e-12)
        self.assertFalse(fit["diagnostics"]["physical_currents"])

    def test_mne_diagonal_solution_and_depth_cap(self):
        A = np.diag([1., .01])
        Y = np.ones((2, 3))
        fit = fit_inverse(A, Y, "mne", ridge_relative=.1)
        np.testing.assert_allclose(fit["operator"], np.diag([1 / 1.01, .01 / .0101]))
        weighted = fit_inverse(A, Y, "wmne")
        np.testing.assert_allclose(weighted["source_weights"], [1, 100])

    def test_no_application_time_fit_or_input_mutation(self):
        A_before, Y_before = self.A.copy(), self.Y.copy()
        for method in ("mne", "wmne", "sloreta", "lcmv", "sbl", "sbl_lcmv"):
            first = fit_inverse(self.A, self.Y, method)
            operator = first["operator"].copy()
            test = self.Y[:, :3].copy()
            prediction = operator @ test
            test *= 100
            np.testing.assert_array_equal(first["operator"], operator)
            repeat = fit_inverse(self.A, self.Y, method)
            np.testing.assert_array_equal(repeat["operator"], operator)
            np.testing.assert_allclose(repeat["operator"] @ (test / 100), prediction)
        np.testing.assert_array_equal(self.A, A_before)
        np.testing.assert_array_equal(self.Y, Y_before)

    def test_zero_columns_are_explicitly_unobservable(self):
        A = self.A.copy()
        A[:, 4] = 0
        for method in ("mne", "wmne", "sloreta", "lcmv", "sbl", "sbl_lcmv"):
            fit = fit_inverse(A, self.Y, method)
            np.testing.assert_array_equal(fit["operator"][4], np.zeros(9))
            self.assertEqual(fit["diagnostics"]["zero_leadfield_columns"], 1)

    def test_nullspace_alternative_cannot_change_estimate(self):
        A = np.array([[1., 0., 1.], [0., 1., 0.]])
        original = np.array([[2., 0., -2.], [0., 1., 3.], [0., 0., 0.]])
        alternative = original + np.array([[1.], [0.], [-1.]]) * np.array([[3., -4., 2.]])
        np.testing.assert_array_equal(A @ original, A @ alternative)
        for method in ("mne", "wmne", "sloreta", "lcmv", "sbl", "sbl_lcmv"):
            first = fit_inverse(A, A @ original, method)
            second = fit_inverse(A, A @ alternative, method)
            np.testing.assert_array_equal(first["operator"], second["operator"])

    def test_model_lcmv_reuses_identical_sbl_fit(self):
        sbl = fit_inverse(self.A, self.Y, "sbl")
        combined = fit_inverse(self.A, self.Y, "sbl_lcmv")
        np.testing.assert_array_equal(sbl["model_lcmv_operator"], combined["operator"])
        np.testing.assert_array_equal(sbl["gamma"], combined["gamma"])

    def test_active_model_lcmv_uses_minimum_variance_prefix(self):
        operator = np.arange(12., dtype=float).reshape(4, 3)
        fit = {"gamma_normalized": np.array([1., 60., 9., 30.]),
               "model_lcmv_operator": operator}
        gated, count = sbl_active_operator(fit)
        self.assertEqual(count, 3)
        np.testing.assert_array_equal(gated[0], np.zeros(3))
        np.testing.assert_array_equal(gated[1:], operator[1:])
        all_rows, count = sbl_active_operator(fit, 1.)
        self.assertEqual(count, 4)
        np.testing.assert_array_equal(all_rows, operator)

    def test_active_model_lcmv_keeps_unit_gain_without_mutating_fit(self):
        fit = fit_inverse(self.A, self.Y, "sbl")
        original_operator = fit["model_lcmv_operator"].copy()
        original_gamma = fit["gamma_normalized"].copy()
        history = list(fit["evidence_history"])
        gated, count = sbl_active_operator(fit)
        kept = np.any(gated != 0, axis=1)
        self.assertEqual(np.sum(kept), count)
        gain = np.diag(gated @ self.A)
        np.testing.assert_allclose(gain[kept], 1., atol=1e-12)
        np.testing.assert_array_equal(gain[~kept], np.zeros(np.sum(~kept)))
        sensitivity = np.sum(self.A * np.linalg.solve(fit["model_cov"], self.A), axis=0)
        np.testing.assert_allclose(fit["operator"],
                                   (fit["gamma"] * sensitivity)[:, None] * original_operator,
                                   atol=1e-12)
        np.testing.assert_array_equal(fit["model_lcmv_operator"], original_operator)
        np.testing.assert_array_equal(fit["gamma_normalized"], original_gamma)
        self.assertEqual(fit["evidence_history"], history)

    def test_active_model_lcmv_zero_power_and_invalid_fraction(self):
        fit = {"gamma_normalized": np.zeros(3), "model_lcmv_operator": np.ones((3, 2))}
        gated, count = sbl_active_operator(fit)
        self.assertEqual(count, 0)
        np.testing.assert_array_equal(gated, np.zeros((3, 2)))
        for fraction in (0, -.1, 1.01, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                sbl_active_operator(fit, fraction)

    def test_invalid_inputs_and_degenerate_training(self):
        for kwargs in ({"ridge_relative": 0}, {"max_iter": 0}, {"tol": float("nan")},
                       {"max_depth_weight": .5}, {"covariance_loading": -1}):
            with self.assertRaises(ValueError):
                fit_inverse(self.A, self.Y, "sbl", **kwargs)
        with self.assertRaises(ValueError):
            fit_inverse(self.A, np.ones_like(self.Y), "lcmv")
        with self.assertRaises(ValueError):
            fit_inverse(self.A, self.Y[:2], "sbl")


if __name__ == "__main__":
    unittest.main()
