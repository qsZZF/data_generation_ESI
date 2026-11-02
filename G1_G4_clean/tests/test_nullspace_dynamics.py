"""Known dynamics can reveal initial-state directions lost instantaneously."""
import unittest

import numpy as np

from scripts.run_nullspace_dynamics import diffusion_design, solve_initial


class TestNullspaceDynamics(unittest.TestCase):
    def test_known_decay_adds_rank_but_replication_does_not(self):
        A = np.array([[1., 1.]])
        eigenvalues = np.array([0., np.log(2.)])
        diffusion = diffusion_design(A, eigenvalues, 1., 2, fs=1)
        np.testing.assert_allclose(diffusion, [[1., 1.], [1., .5]])
        self.assertEqual(np.linalg.matrix_rank(A), 1)
        self.assertEqual(np.linalg.matrix_rank(diffusion), 2)
        stationary = diffusion_design(A, eigenvalues, 0., 64, fs=1)
        self.assertEqual(np.linalg.matrix_rank(stationary), 1)

    def test_known_model_recovers_initial_null_component_from_same_data(self):
        A = np.array([[1., 1.]])
        spectrum = np.array([0., np.log(2.)])
        truth = np.array([1., -1.])
        true_design = diffusion_design(A, spectrum, 1., 4, fs=1)
        data = (true_design @ truth).reshape(4, 1).T
        before = data.copy()
        _, correct, info = solve_initial(true_design, data)
        np.testing.assert_allclose(correct, truth, atol=1e-12)
        self.assertEqual(info["observability_rank"], 2)
        wrong_design = diffusion_design(A, spectrum, 2., 4, fs=1)
        _, wrong, _ = solve_initial(wrong_design, data)
        self.assertGreater(np.linalg.norm(wrong - truth), .1)
        np.testing.assert_array_equal(data, before)

    def test_time_major_stack_matches_forward_trajectory(self):
        rng = np.random.default_rng(2)
        A = rng.normal(size=(3, 4))
        eigenvalues = np.arange(4.)
        coefficients = rng.normal(size=4)
        design = diffusion_design(A, eigenvalues, .2, 6, fs=4)
        trajectory = A @ (coefficients[:, None] * np.exp(-.2 * eigenvalues[:, None] * np.arange(6) / 4))
        np.testing.assert_allclose(design @ coefficients, trajectory.T.reshape(-1), atol=1e-14)
        short = diffusion_design(A, eigenvalues, .2, 2, fs=4)
        np.testing.assert_array_equal(design[:6], short)

    def test_invalid_diffusion_is_rejected(self):
        A = np.array([[1., 1.]])
        for spectrum, kappa, samples in (([0., -1.], .1, 2), ([0., 1.], -.1, 2),
                                          ([0., 1.], .1, 0), ([0.], .1, 2)):
            with self.assertRaises(ValueError):
                diffusion_design(A, spectrum, kappa, samples)


if __name__ == "__main__":
    unittest.main()
