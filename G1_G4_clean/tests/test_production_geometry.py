"""Distance caching reuses computations without stale graphs or unbounded memory."""
import unittest
from unittest.mock import patch
import numpy as np
from scipy import sparse


class ProductionGeometryTests(unittest.TestCase):
    def graph(self):
        return sparse.diags([np.ones(4), np.ones(4)], [-1, 1], shape=(5, 5), format='csr')

    def test_seed_hit_avoids_dijkstra_and_is_readonly(self):
        from Simulation_G1.geodesic import cached_geodesic_distances, geodesic_distances
        cache, graph = {}, self.graph()
        with patch('Simulation_G1.geodesic.dijkstra', wraps=__import__('scipy').sparse.csgraph.dijkstra) as solve:
            first = cached_geodesic_distances(graph, 0, cache=cache)
            repeated = cached_geodesic_distances(graph, 0, cache=cache)
            self.assertEqual(solve.call_count, 1)
        np.testing.assert_array_equal(first, geodesic_distances(graph, 0))
        self.assertIs(first, repeated)
        self.assertFalse(first.flags.writeable)

    def test_changed_graph_invalidates_and_lru_obeys_byte_budget(self):
        from Simulation_G1.geodesic import cached_geodesic_distances
        cache, graph = {}, self.graph()
        cached_geodesic_distances(graph, 0, cache=cache, max_bytes=80)
        cached_geodesic_distances(graph, 1, cache=cache, max_bytes=80)
        cached_geodesic_distances(graph, 0, cache=cache, max_bytes=80)
        cached_geodesic_distances(graph, 2, cache=cache, max_bytes=80)
        self.assertEqual({key[1] for key in cache}, {0, 2})
        self.assertLessEqual(sum(value.nbytes for value in cache.values()), 80)
        graph.data *= 2
        changed = cached_geodesic_distances(graph, 0, cache=cache, max_bytes=80)
        np.testing.assert_array_equal(changed, [0, 2, 4, 6, 8])
        self.assertEqual(len(cache), 1)
        cached_geodesic_distances(graph, 0, cache=cache, max_bytes=0)
        self.assertEqual(cache, {})

    def test_invalid_arguments_cannot_reuse_cached_value(self):
        from Simulation_G1.geodesic import cached_geodesic_distances
        cache, graph = {}, self.graph()
        cached_geodesic_distances(graph, 1, cache=cache)
        with self.assertRaises(ValueError):
            cached_geodesic_distances(graph, True, cache=cache)
        with self.assertRaises(ValueError):
            cached_geodesic_distances(graph, 0, cache=cache, max_bytes=-1)
        graph.data[0] = -1
        with self.assertRaises(ValueError):
            cached_geodesic_distances(graph, 1, cache=cache)


if __name__ == '__main__':
    unittest.main()
