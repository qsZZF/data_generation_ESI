"""Mesh-edge geodesic growth, complete distance shells and legacy replay."""
import unittest
import numpy as np
from scipy import sparse

from Simulation_G1.g1_simulation_core import G1SimulationConfig
from test_spatial_patches import grid_simulator


def path_graph(points, mask=None):
    from Simulation_G1.geodesic import mesh_edge_graph
    n = len(points)
    conn = sparse.diags([np.ones(n-1), np.ones(n-1)], [-1, 1], shape=(n, n))
    return mesh_edge_graph(np.asarray(points, dtype=float), conn, mask)


class GeodesicPatchTests(unittest.TestCase):
    def test_legacy_fixed_seed_fixture(self):
        sim = grid_simulator()
        cfg = G1SimulationConfig(n_patch=2, patch_area_mm2=100, min_seed_distance_mm=0,
                                 patch_geometry='legacy_bfs')
        seeds, patches, weights = sim._resolve_spatial_patches(cfg, np.random.default_rng(7))
        np.testing.assert_array_equal(seeds, [1, 35])
        self.assertEqual([p.tolist() for p in patches], [[1, 0, 2, 9, 10, 3, 11], [35, 26, 34, 43, 44, 17]])
        np.testing.assert_array_equal(weights[0], np.array([1., .9258747100830078, .9258747100830078,
            .8572439551353455, .9258747100830078, .7348672747612, .8572439551353455], dtype=np.float32))

    def test_folded_surface_has_no_euclidean_shortcut(self):
        from Simulation_G1.geodesic import geodesic_distances, grow_geodesic_patch
        points = np.array([[0., 0, 0], [10, 0, 0], [10, 1, 0], [0, 1, 0]])
        graph = path_graph(points)
        distances = geodesic_distances(graph, 0)
        np.testing.assert_allclose(distances, [0, 10, 11, 21])
        self.assertEqual(np.linalg.norm(points[-1]-points[0]), 1)
        support, selected, info = grow_geodesic_patch(0, graph, np.ones(4), 2.5)
        np.testing.assert_array_equal(support, [0, 1, 2])
        np.testing.assert_array_equal(selected, [0, 10, 11])
        self.assertGreaterEqual(info['actual_area_mm2'], 2.5)
        self.assertLessEqual(info['area_overshoot_mm2'], info['final_shell_area_mm2'])

    def test_masked_bridge_cannot_be_crossed_and_capacity_is_checked(self):
        from Simulation_G1.geodesic import geodesic_distances, grow_geodesic_patch
        graph = path_graph([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [True, False, True])
        self.assertTrue(np.isinf(geodesic_distances(graph, 0)[2]))
        with self.assertRaisesRegex(ValueError, 'connected'):
            grow_geodesic_patch(0, graph, np.ones(3), 1.1)

    def test_equal_distance_shell_is_relabel_invariant(self):
        from Simulation_G1.geodesic import mesh_edge_graph, grow_geodesic_patch
        points = np.array([[0., 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
        conn = sparse.csr_matrix(np.array([[0, 1, 1, 1, 1], [1, 0, 0, 0, 0],
                                           [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]]))
        graph = mesh_edge_graph(points, conn)
        support, _, info = grow_geodesic_patch(0, graph, np.ones(5), 2)
        self.assertEqual(set(support), set(range(5)))
        self.assertEqual(info['final_shell_area_mm2'], 4)
        self.assertEqual(info['area_overshoot_mm2'], 3)
        order = np.array([3, 4, 2, 0, 1])
        changed = mesh_edge_graph(points[order], conn[order][:, order])
        other, _, _ = grow_geodesic_patch(int(np.where(order == 0)[0][0]), changed, np.ones(5), 2)
        self.assertEqual(set(order[other]), set(support))

    def test_invalid_edges_and_geometry_mode(self):
        from Simulation_G1.geodesic import geodesic_distances
        with self.assertRaisesRegex(ValueError, 'positive'):
            path_graph([[0, 0, 0], [0, 0, 0]])
        with self.assertRaisesRegex(ValueError, 'positive'):
            geodesic_distances(sparse.csr_matrix([[0., -1.], [-1., 0.]]), 0)
        with self.assertRaisesRegex(ValueError, 'patch_geometry'):
            G1SimulationConfig(patch_geometry='unknown').validate_recipe()

    def test_explicit_support_distance_scope_allows_outside_support_but_not_mask(self):
        from Simulation_G1.geodesic import mesh_edge_graph, geodesic_distances
        points = np.array([[0., 0, 0], [10, 0, 0], [0, 1, 0], [0, .5, 0]])
        # Support 0-1-2 is connected. Vertex 3 is outside that support and is a
        # shorter valid surface route unless explicitly removed by the mask.
        adjacency = sparse.csr_matrix([[0, 1, 0, 1], [1, 0, 1, 0],
                                       [0, 1, 0, 1], [1, 0, 1, 0]])
        distances = geodesic_distances(mesh_edge_graph(points, adjacency), 0)
        self.assertAlmostEqual(distances[2], 1.)
        masked = mesh_edge_graph(points, adjacency, [True, True, True, False])
        self.assertAlmostEqual(geodesic_distances(masked, 0)[2], 10 + np.sqrt(101))

    def test_default_geodesic_nonoverlap_and_explicit_replay(self):
        from dataclasses import replace
        from Simulation_G1.geodesic import geodesic_distances
        from Simulation_G1.spatial import gaussian_weights
        sim = grid_simulator()
        cfg = G1SimulationConfig(n_patch=2, patch_area_mm2=100, min_seed_distance_mm=0,
                                 random_seed=7, normalize_source=False)
        self.assertEqual(cfg.patch_geometry, 'geodesic')
        source, meta = sim.generate_source(cfg)
        patches = [p['vertices'] for p in meta['patches']]
        self.assertFalse(set(patches[0]) & set(patches[1]))
        for p in meta['patches']:
            distance = geodesic_distances(sim.geodesic_graph, p['seed_vertex'])[p['vertices']]
            np.testing.assert_allclose(p['spatial_weights'], gaussian_weights(distance, 15), rtol=1e-6)
            self.assertLessEqual(p['area_overshoot_mm2'], p['final_shell_area_mm2'] + 1e-9)
        # Explicit replay changes no draws: compare geometry directly.
        replay = replace(cfg, seed_vertices=meta['seed_vertices'], patch_vertices=patches)
        _, support, weights = sim._resolve_spatial_patches(replay, np.random.default_rng(123))
        self.assertEqual([p.tolist() for p in support], patches)
        for p, w in zip(meta['patches'], weights):
            np.testing.assert_array_equal(w, np.array(p['spatial_weights'], dtype=np.float32))


if __name__ == '__main__':
    unittest.main()
