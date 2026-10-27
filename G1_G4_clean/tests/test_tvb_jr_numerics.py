"""Exercise the actual installed TVB equations, without anatomical resources."""
import os
from pathlib import Path
import unittest
import warnings

import numpy as np

from Simulation_G3.g3_tvb_core import TVBSignalConfig, TVBSourceGenerator


class TVBJRNumericsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("TVB_USER_HOME", str(Path(__file__).resolve().parents[1] / "Simulation_G3/.tvb_home"))
        try:
            from tvb.simulator.lab import coupling, models
        except ModuleNotFoundError as error:
            if error.name == "tvb":
                raise unittest.SkipTest("TVB is not installed") from error
            raise
        cls.coupling = coupling
        cls.models = models

    def setUp(self):
        self.generator = TVBSourceGenerator(__file__)
        self.config = TVBSignalConfig(signal_type="alpha_jr")
        self.model, _, _ = self.generator._build_model(self.config, 3, [1], self.models)
        self.model.configure()

    def test_extreme_jr_sigmoids_remain_finite_without_warnings(self):
        state = np.zeros((6, 3, 1))
        state[0, :, 0] = [-30, 0, 30]
        state[1, :, 0] = [-4000, 6, 4000]
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with np.errstate(over="raise", invalid="raise"):
                derivative = self.model.dfun(state, np.zeros((2, 3, 1)))
        self.assertTrue(np.isfinite(derivative).all())
        expected = self.model.A * 0.1 * np.array([0, 0.0025, 0.005]) - 0.01 * state[0, :, 0]
        np.testing.assert_allclose(derivative[3, :, 0], expected, atol=1e-14)

    def test_derivative_preserves_tvb_equations_and_spatial_parameters(self):
        original = self.models.JansenRit(A=self.model.A.copy(), mu=self.model.mu.copy(), v0=np.array([6.0]))
        original.configure()
        rng = np.random.default_rng(4)
        state = rng.uniform(-1, 1, (6, 3, 1))
        coupling = rng.uniform(0, 0.1, (2, 3, 1))
        expected = original.dfun(state, coupling, local_coupling=0.2)
        actual = self.model.dfun(state, coupling, local_coupling=0.2)
        self.assertEqual(actual.shape, state.shape)
        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=1e-14)
        np.testing.assert_array_equal(self.model.observe(state), original.observe(state))

    def test_extreme_coupling_sigmoid_has_correct_limits(self):
        coupling = self.generator._build_coupling(self.config, self.coupling)
        state = np.zeros((2, 2, 3, 1))
        state[:, 0, :, 0] = [-4000, 6, 4000]
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with np.errstate(over="raise", invalid="raise"):
                result = coupling.pre(np.zeros_like(state), state)
        self.assertEqual(result.shape, (2, 1, 3, 1))
        np.testing.assert_allclose(result[0, 0, :, 0], [0, 0.0025, 0.005], atol=1e-15)

    def test_coupling_preserves_tvb_equation_and_scaling(self):
        coupling = self.generator._build_coupling(self.config, self.coupling)
        original = self.coupling.SigmoidalJansenRit(a=np.array([self.config.coupling_strength]))
        state = np.random.default_rng(5).uniform(-20, 20, (2, 2, 3, 1))
        np.testing.assert_allclose(coupling.pre(state, state), original.pre(state, state), rtol=2e-14)
        np.testing.assert_array_equal(coupling.post(state), original.post(state))

    def test_genuine_arithmetic_overflow_is_not_hidden(self):
        state = np.full((6, 3, 1), np.finfo(float).max)
        with np.errstate(over="raise", invalid="raise"):
            with self.assertRaises(FloatingPointError):
                self.model.dfun(state, np.zeros((2, 3, 1)))


if __name__ == "__main__":
    unittest.main()
