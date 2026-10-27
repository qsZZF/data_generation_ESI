"""Small-mesh numerical contracts for the cortical geometry basis."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from Simulation_G1.geometry import compute_basis, load_basis, project_maps, save_basis


def octahedron():
    vertices = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0],
                         [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
    faces = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
                      [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
    return vertices, faces


class GeometryTests(unittest.TestCase):
    def test_mass_orthogonality_and_analytic_eigenvalues(self):
        vertices, faces = octahedron()
        result = compute_basis(vertices, faces, n_modes=4)
        basis, mass = result['basis'], result['mass']
        np.testing.assert_allclose(basis.T @ (mass[:, None] * basis), np.eye(4), atol=1e-12)
        np.testing.assert_allclose(basis.T @ mass, 0, atol=1e-12)
        np.testing.assert_allclose(result['eigenvalues'], [2, 2, 2, 3], atol=1e-12)
        # The regular octahedron has 8 equal triangles, area sqrt(3)/2 each.
        self.assertAlmostEqual(mass.sum(), 4 * np.sqrt(3))
        self.assertLess(result['metadata']['max_relative_eigen_residual'], 1e-12)

    def test_disconnected_components_remove_both_constants(self):
        vertices, faces = octahedron()
        result = compute_basis(np.vstack([vertices, vertices * 2 + 10]),
                               np.vstack([faces, faces + len(vertices)]), n_modes=7)
        self.assertEqual(result['metadata']['n_components'], 2)
        self.assertEqual(result['basis'].shape, (12, 7))
        for label in (0, 1):
            mask = result['component_labels'] == label
            np.testing.assert_allclose(result['mass'][mask] @ result['basis'][mask], 0, atol=1e-12)
        np.testing.assert_allclose(result['eigenvalues'][:3], .5, atol=1e-12)

    def test_save_load_identity_and_coordinate_validation(self):
        vertices, faces = octahedron()
        result = compute_basis(vertices, faces, n_modes=3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'basis.npz'
            save_basis(path, result)
            loaded = load_basis(path, vertices_mm=vertices, faces=faces)
            np.testing.assert_array_equal(loaded['basis'], result['basis'])
            self.assertEqual(loaded['mesh_sha256'], result['mesh_sha256'])
            with self.assertRaisesRegex(ValueError, 'mesh'):
                load_basis(path, vertices_mm=vertices * 2, faces=faces)
            with self.assertRaisesRegex(ValueError, 'together'):
                load_basis(path, vertices_mm=vertices)

    def test_area_normalized_map_projection(self):
        vertices, faces = octahedron()
        result = compute_basis(vertices, faces, n_modes=3)
        coefficients = project_maps(result, result['basis'][:, :2] * 7, vertices_mm=vertices)
        np.testing.assert_allclose(coefficients, np.eye(3)[:, :2], atol=1e-12)
        np.testing.assert_allclose(project_maps(result, np.ones((6, 1))), 0, atol=1e-12)
        for maps in (np.zeros((6, 1)), np.ones((5, 1)), np.full((6, 1), np.nan)):
            with self.assertRaises(ValueError):
                project_maps(result, maps)
        with self.assertRaisesRegex(ValueError, 'coordinates'):
            project_maps(result, np.ones((6, 1)), vertices_mm=vertices[::-1])

    def test_reject_malformed_mesh_and_invalid_mode_count(self):
        vertices, faces = octahedron()
        malformed = [faces.astype(float) + .25, faces[:, :2],
                     np.vstack([faces, faces[0]]), np.array([[0, 0, 1]])]
        for triangles in malformed:
            with self.assertRaises(ValueError):
                compute_basis(vertices, triangles, n_modes=2)
        for n_modes in (0, True, 1.5, 6):
            with self.assertRaises(ValueError):
                compute_basis(vertices, faces, n_modes=n_modes)
        with self.assertRaises(ValueError):
            compute_basis(np.vstack([vertices, [0, 0, 0]]), faces, n_modes=2)

    def test_mesh_scaling_and_repeatability(self):
        vertices, faces = octahedron()
        first = compute_basis(vertices, faces, n_modes=3)
        repeat = compute_basis(vertices, faces, n_modes=3)
        scaled = compute_basis(vertices * 10, faces, n_modes=3)
        np.testing.assert_array_equal(first['basis'], repeat['basis'])
        np.testing.assert_allclose(scaled['eigenvalues'], first['eigenvalues'] / 100, atol=1e-12)
        self.assertNotEqual(first['mesh_sha256'], scaled['mesh_sha256'])

    def test_sparse_solver_on_triangulated_plane(self):
        x, y = np.meshgrid(np.arange(9), np.arange(9))
        vertices = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
        faces = []
        for row in range(8):
            for column in range(8):
                vertex = row * 9 + column
                faces.extend([[vertex, vertex + 1, vertex + 9],
                              [vertex + 1, vertex + 10, vertex + 9]])
        first = compute_basis(vertices, faces, n_modes=5)
        second = compute_basis(vertices, faces, n_modes=5)
        basis, mass = first['basis'], first['mass']
        np.testing.assert_allclose(basis.T @ (mass[:, None] * basis), np.eye(5), atol=1e-10)
        np.testing.assert_allclose(basis.T @ mass, 0, atol=1e-10)
        np.testing.assert_array_equal(basis, second['basis'])
        self.assertLess(first['metadata']['max_relative_eigen_residual'], 1e-8)

    def test_corrupt_cached_dimensions_are_rejected(self):
        vertices, faces = octahedron()
        result = compute_basis(vertices, faces, n_modes=3)
        with tempfile.TemporaryDirectory() as directory:
            result['component_labels'] = np.array([0])
            with self.assertRaisesRegex(ValueError, 'component'):
                save_basis(Path(directory) / 'bad.npz', result)


if __name__ == '__main__':
    unittest.main()
