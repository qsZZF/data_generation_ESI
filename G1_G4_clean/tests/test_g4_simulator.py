"""Small synthetic-graph checks; no atlas/connectome resources are substituted."""
import unittest

import numpy as np

from Simulation_G4.simulator import (
    PARAMETER_BOUNDS, PARAMETER_NAMES, _delayed_output, _sigmoid, _rhs,
    sample_parameters, simulate,
)


class G4SimulatorTests(unittest.TestCase):
    def setUp(self):
        self.parameters = sample_parameters(np.random.default_rng(4), 4)
        self.connectivity = np.full((4, 4), 0.01)

    def test_paper_priors_and_conditional_constraints(self):
        parameters = sample_parameters(np.random.default_rng(7), 994)
        self.assertEqual(len(PARAMETER_NAMES), 11)
        for name, (low, high) in PARAMETER_BOUNDS.items():
            self.assertEqual(parameters[name].shape, (994,))
            self.assertTrue(np.all((parameters[name] >= low) & (parameters[name] <= high)))
        for branch in (1, 2):
            e, i = parameters[f"tau_e{branch}_ms"], parameters[f"tau_i{branch}_ms"]
            self.assertTrue(np.all((i >= e) & (i <= 2.4 * e)))
        self.assertGreaterEqual(parameters["delay_ms"], 0)
        self.assertLessEqual(parameters["delay_ms"], 10)
        repeated = sample_parameters(np.random.default_rng(7), 994)
        for name in parameters:
            np.testing.assert_array_equal(parameters[name], repeated[name])

    def test_sliding_delay_interpolates_physical_time_and_zero_delay(self):
        # A ring containing samples6..9, with newest sample9 at index1.
        ring = np.array([[8.0], [9.0], [6.0], [7.0]])
        np.testing.assert_allclose(_delayed_output(ring, 9, 2.5, 0, np.array([10.])), [6.5])
        np.testing.assert_allclose(_delayed_output(ring, 9, 2.5, 1, np.array([10.])), [7.5])
        np.testing.assert_allclose(_delayed_output(ring, 9, 0, 0, np.array([10.])), [9.])
        np.testing.assert_allclose(_delayed_output(ring, 9, 0, 1, np.array([10.])), [10.])
        np.testing.assert_allclose(_delayed_output(ring, 9, .5, 1, np.array([10.])), [9.5])

    def test_extreme_sigmoid_saturates_without_overflow(self):
        self.assertEqual(_sigmoid(1e6, 6., .56, 5.), 5.)
        self.assertEqual(_sigmoid(-1e6, 6., .56, 5.), 0.)
        self.assertEqual(_sigmoid(6., 6., .56, 5.), 2.5)

    def test_derivative_matches_dual_branch_equations(self):
        parameters = np.array([self.parameters[name] for name in PARAMETER_NAMES])
        state = np.random.default_rng(5).normal(0, .1, (4, 12))
        inputs = np.array([180., 220., 230., 210.])
        expected = np.zeros_like(state)
        for region, x in enumerate(state):
            p = {name: self.parameters[name][region] for name in PARAMETER_NAMES}
            def firing(voltage):
                return p["rmax"] / (1 + np.exp(p["beta"] * (p["theta"] - voltage)))
            voltage = .5 * (x[1] - x[2]) + .5 * (x[7] - x[8])
            common = .5 * (x[0] + x[6])
            drives = [firing(voltage), 135 * p["C2"] * firing(135 * p["C1"] * common) + inputs[region],
                      135 * p["C4"] * firing(135 * p["C3"] * common)]
            for branch in (1, 2):
                offset = 6 * (branch - 1)
                e, i = p[f"tau_e{branch}_ms"] / 1000, p[f"tau_i{branch}_ms"] / 1000
                gains = [3.25, 3.25, 22.] if branch == 1 else [.0325 / e, .0325 / e, .44 / i]
                for population, tau in enumerate((e, e, i)):
                    v, dv = x[offset + population], x[offset + population + 3]
                    expected[region, offset + population] = dv
                    expected[region, offset + population + 3] = gains[population] * drives[population] / tau - 2 * dv / tau - v / tau**2
        np.testing.assert_allclose(_rhs(state, inputs, parameters), expected, rtol=2e-14, atol=1e-12)

    def test_seed_reproducibility_clock_and_warmup_metadata(self):
        first, time, meta = simulate(self.parameters, self.connectivity, duration_s=.5, warmup_s=.25, seed=23)
        repeated, _, _ = simulate(self.parameters, self.connectivity, duration_s=.5, warmup_s=.25, seed=23)
        changed, _, _ = simulate(self.parameters, self.connectivity, duration_s=.5, warmup_s=.25, seed=24)
        np.testing.assert_array_equal(first, repeated)
        self.assertFalse(np.array_equal(first, changed))
        self.assertEqual(first.shape, (4, 128))
        self.assertTrue(np.isfinite(first).all())
        np.testing.assert_allclose(np.diff(time), 1 / 256)
        self.assertEqual(time[0], 0)
        self.assertEqual(meta["warmup_discarded_s"], .25)
        self.assertEqual(meta["drive_sampling_hz"], 256)
        self.assertEqual(meta["source_unit"], "mV_PSP")

    def test_discard_matches_continuous_native_simulation(self):
        # Default internal rate=2048Hz. Without resampling, compare exact states.
        complete, _, _ = simulate(self.parameters, self.connectivity, duration_s=.5, warmup_s=0, sampling_rate_hz=2048, seed=23)
        discarded, _, _ = simulate(self.parameters, self.connectivity, duration_s=.25, warmup_s=.25, sampling_rate_hz=2048, seed=23)
        np.testing.assert_array_equal(discarded, complete[:, 512:])

    def test_timestep_convergence_under_identical_physical_drive(self):
        results = [simulate(self.parameters, self.connectivity, duration_s=2, warmup_s=1, dt_ms=dt)[0]
                   for dt in (.5, .25, .1, .05)]
        errors = [np.linalg.norm(result - results[-1]) / np.linalg.norm(results[-1]) for result in results[:-1]]
        self.assertLess(errors[0], .01)
        self.assertLess(errors[1], errors[0])
        self.assertLess(errors[2], errors[1])

    def test_delay_affects_coupled_but_not_disconnected_system(self):
        zero = dict(self.parameters, delay_ms=0.)
        delayed = dict(self.parameters, delay_ms=10.)
        options = dict(duration_s=.5, warmup_s=.25)
        a = simulate(zero, self.connectivity, **options)[0]
        b = simulate(delayed, self.connectivity, **options)[0]
        self.assertGreater(np.max(np.abs(a - b)), 1e-5)
        no_connections = np.zeros_like(self.connectivity)
        np.testing.assert_array_equal(simulate(zero, no_connections, **options)[0],
                                      simulate(delayed, no_connections, **options)[0])

    def test_orientation_and_explicit_normalization(self):
        graph = np.zeros((4, 4))
        graph[1, 2] = .1
        options = dict(duration_s=.25, warmup_s=.125)
        expected = simulate(self.parameters, graph, **options)[0]
        actual = simulate(self.parameters, graph.T, connectivity_orientation="source_target", **options)[0]
        np.testing.assert_array_equal(actual, expected)
        with self.assertRaisesRegex(ValueError, "weights <= 2"):
            simulate(self.parameters, graph * 100, **options)
        normalized, _, meta = simulate(self.parameters, graph * 100, connectivity_mode="max", **options)
        self.assertTrue(np.isfinite(normalized).all())
        self.assertEqual(meta["connectivity_divisor"], 10)

    def test_invalid_controls_and_parameter_shapes_fail(self):
        for kwargs in ({"dt_ms": 1}, {"warmup_s": -1}, {"duration_s": .001},
                       {"connectivity_mode": "unknown"}, {"dt_ms": np.nan}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                simulate(self.parameters, self.connectivity, **kwargs)
        with self.assertRaises(ValueError):
            simulate(dict(self.parameters, theta=np.ones(3)), self.connectivity)
        with self.assertRaises(ValueError):
            simulate(dict(self.parameters, delay_ms=11.), self.connectivity)

    def test_float32_physical_bound_roundoff_is_repaired_only(self):
        options = dict(duration_s=.125, warmup_s=.125)
        for end in (0, 1):
            parameters = {name: np.full(4, bounds[end], dtype=np.float32)
                          for name, bounds in PARAMETER_BOUNDS.items()}
            parameters["delay_ms"] = np.float32(10 * end)
            before = {name: np.copy(value) for name, value in parameters.items()}
            source, _, meta = simulate(parameters, self.connectivity, **options)
            self.assertTrue(np.isfinite(source).all())
            if end == 1:
                self.assertIn("beta", meta["parameter_roundoff_repairs"])
            for name in parameters:
                np.testing.assert_array_equal(parameters[name], before[name])
        invalid = dict(parameters, beta=np.full(4, .62001, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "beta"):
            simulate(invalid, self.connectivity, **options)

    def test_float32_joint_tau_bound_roundoff_is_repaired_only(self):
        parameters = {name: np.asarray(value, dtype=np.float32) for name, value in self.parameters.items()}
        parameters["tau_e1_ms"] = np.full(4, 10.1, dtype=np.float32)
        parameters["tau_i1_ms"] = np.float32(2.4) * parameters["tau_e1_ms"]
        parameters["tau_e2_ms"] = np.full(4, 8., dtype=np.float32)
        parameters["tau_i2_ms"] = np.nextafter(parameters["tau_e2_ms"], np.float32(0))
        _, _, meta = simulate(parameters, self.connectivity, duration_s=.125, warmup_s=.125)
        self.assertIn("tau_i1_ms_joint", meta["parameter_roundoff_repairs"])
        self.assertIn("tau_i2_ms_joint", meta["parameter_roundoff_repairs"])
        parameters["tau_i1_ms"] += .001
        with self.assertRaisesRegex(ValueError, "inhibitory time constants"):
            simulate(parameters, self.connectivity, duration_s=.125, warmup_s=.125)


if __name__ == "__main__":
    unittest.main()
