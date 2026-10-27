"""Sparse JR coupling preserves TVB's weighted, delayed network equations."""
import os
from pathlib import Path
import unittest

import numpy as np


class SparseJRCouplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("TVB_USER_HOME", str(Path(__file__).resolve().parents[1] / "Simulation_G3/.tvb_home"))
        try:
            from tvb.simulator.history import DenseHistory, SparseHistory
        except ModuleNotFoundError as error:
            if error.name == "tvb":
                raise unittest.SkipTest("TVB is not installed") from error
            raise
        from Simulation_G3.tvb_jr import StableSigmoidalJansenRit, SparseSigmoidalJansenRit
        cls.DenseHistory, cls.SparseHistory = DenseHistory, SparseHistory
        cls.dense_class, cls.sparse_class = StableSigmoidalJansenRit, SparseSigmoidalJansenRit

    def histories(self, weights, delays, modes=1, scale=20):
        rng = np.random.default_rng(17)
        states = rng.uniform(-scale, scale, (int(delays.max()) + 1, 2, len(weights), modes))
        histories = [kind(weights, delays, np.array([0, 1]), modes)
                     for kind in (self.DenseHistory, self.SparseHistory)]
        for history in histories:
            history.initialize(states.copy())
        return histories

    def test_weighted_asymmetric_graph_delays_modes_and_negative_gain(self):
        weights = np.array([[0., .2, 0., -.1], [.7, 0., .3, 0.], [0., 0., 0., 0.], [.1, 0., 0., .4]])
        for modes in (1, 3):
            for delay in (0, 3):
                delays = np.arange(16).reshape(4, 4) % (delay + 1)
                dense_history, sparse_history = self.histories(weights, delays, modes)
                for gain in (.1, -2.):
                    dense = self.dense_class(a=np.array([gain]), cmin=np.array([-.02]))
                    sparse = self.sparse_class(a=np.array([gain]), cmin=np.array([-.02]))
                    for step in range(1, 10):
                        with self.subTest(modes=modes, delay=delay, gain=gain, step=step):
                            actual = sparse(step, sparse_history)
                            self.assertEqual(actual.shape, (2, 4, modes))
                            np.testing.assert_allclose(actual, dense(step, dense_history), rtol=2e-14, atol=1e-16)
                            np.testing.assert_array_equal(actual[:, 2], 0.)

    def test_direction_and_delay_are_target_row_source_column(self):
        weights = np.array([[0., 2.], [0., 0.]])
        delays = np.array([[0, 1], [0, 0]])
        _, history = self.histories(weights, delays)
        history.buffer[:] = 0.
        history.buffer[0, 0, 1, 0] = 6.
        history.buffer[1, 0, 1, 0] = -4000.
        result = self.sparse_class(a=np.array([3.]))(2, history)
        np.testing.assert_allclose(result[0, :, 0], [.015, 0.], atol=1e-16)

    def test_disconnected_graph_and_extreme_sigmoid_are_finite(self):
        for weights in (np.zeros((4, 4)), np.ones((4, 4))):
            dense_history, sparse_history = self.histories(weights, np.zeros((4, 4), dtype=int), scale=4000)
            with np.errstate(over="raise", invalid="raise"):
                result = self.sparse_class()(1, sparse_history)
                expected = self.dense_class()(1, dense_history)
            self.assertTrue(np.isfinite(result).all())
            np.testing.assert_allclose(result, expected, atol=1e-16)

    def test_dense_history_and_nonscalar_parameters_use_dense_contract(self):
        dense_history, sparse_history = self.histories(np.ones((3, 3)), np.zeros((3, 3), dtype=int), modes=2)
        for history, parameters in ((dense_history, {}),
                                    (sparse_history, {"a": np.array([.1, -.2])}),
                                    (sparse_history, {"r": np.array([.3, .6])})):
            np.testing.assert_allclose(self.sparse_class(**parameters)(1, history),
                                       self.dense_class(**parameters)(1, history), rtol=2e-14)

    def test_tiny_seeded_stochastic_tvb_trajectory(self):
        from tvb.datatypes.connectivity import Connectivity
        from tvb.simulator import integrators, monitors, noise, simulator
        from Simulation_G3.tvb_jr import StableJansenRit

        def run(coupling_type):
            weights = np.array([[0., .2, .1], [.7, 0., 0.], [0., .3, 0.]])
            conn = Connectivity(weights=weights, tract_lengths=np.array([[0., 3., 6.], [6., 0., 0.], [0., 3., 0.]]),
                                centres=np.eye(3), region_labels=np.array(["a", "b", "c"]), speed=np.array([3.]))
            sim = simulator.Simulator(
                connectivity=conn, model=StableJansenRit(), coupling=coupling_type(a=np.array([.1])),
                integrator=integrators.HeunStochastic(dt=.5, noise=noise.Additive(
                    nsig=np.array([.001]), random_stream=np.random.RandomState(19))),
                monitors=(monitors.Raw(),),
                initial_conditions=np.random.default_rng(18).uniform(-1, 1, (5, 6, 3, 1)),
            ).configure()
            return sim.run(simulation_length=100.)[0]

        reference_time, reference_state = run(self.dense_class)
        sparse_time, sparse_state = run(self.sparse_class)
        np.testing.assert_array_equal(sparse_time, reference_time)
        np.testing.assert_allclose(sparse_state, reference_state, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
