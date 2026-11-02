"""Shortest paths along masked mesh edges, in millimetres.

This is a mesh-edge approximation, not a continuous-surface geodesic solver.
Positive edge lengths ensure that distance-ball supports remain connected.
"""
from collections.abc import MutableMapping
import hashlib

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra


class PatchCapacityError(ValueError):
    """The seed's connected cortical component cannot supply the target area."""


def mesh_edge_graph(vertices_mm, adjacency, mask=None):
    """Build an undirected edge-length graph without reindexing masked vertices."""
    vertices = np.asarray(vertices_mm, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError('Mesh vertices must be finite n-by-3 coordinates in mm')
    n = len(vertices)
    edges = sparse.csr_matrix(adjacency, dtype=float)
    if edges.shape != (n, n) or not np.isfinite(edges.data).all() or np.any(edges.data < 0):
        raise ValueError('Mesh adjacency must be finite, nonnegative and match vertices')
    edges.eliminate_zeros()
    edges = edges.maximum(edges.T).tocoo()
    keep = edges.row != edges.col
    if mask is not None:
        eligible = np.asarray(mask)
        if eligible.shape != (n,) or not np.isin(eligible, [0, 1]).all():
            raise ValueError('Cortex mask must be a binary vector matching vertices')
        eligible = eligible.astype(bool)
        keep &= eligible[edges.row] & eligible[edges.col]
    rows, cols = edges.row[keep], edges.col[keep]
    lengths = np.linalg.norm(vertices[rows] - vertices[cols], axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise ValueError('Retained mesh edges must have finite positive lengths; check duplicate vertices')
    return sparse.csr_matrix((lengths, (rows, cols)), shape=(n, n))


def _validated_graph(graph, seed):
    graph = sparse.csr_matrix(graph)
    if graph.shape[0] != graph.shape[1] or not np.isfinite(graph.data).all() or np.any(graph.data <= 0):
        raise ValueError('Geodesic graph must be square with finite positive edge lengths')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < graph.shape[0]:
        raise ValueError('Geodesic seed must be a valid vertex index')
    return graph


def geodesic_distances(graph, seed):
    """Distances throughout the allowed component, including outside a patch."""
    return dijkstra(_validated_graph(graph, seed), directed=False, indices=int(seed))


def cached_geodesic_distances(graph, seed, *, cache, max_bytes=16*1024**2):
    """Reuse read-only full distance vectors with a bounded byte-LRU cache.

    The complete CSR graph is hashed on every call, including weights and
    connectivity. In-place edge or mask changes therefore cannot return stale
    distances. The caller supplies a dedicated insertion-ordered mapping;
    max_bytes bounds distance-array payloads, excluding small dictionary keys.
    A changed graph clears its predecessor's entries. Oversized vectors are
    returned without retention. This cache is local to a serial caller/worker.
    """
    if not isinstance(cache, MutableMapping):
        raise ValueError('cache must be a dedicated mutable mapping')
    if isinstance(max_bytes, (bool, np.bool_)) or not isinstance(max_bytes, (int, np.integer)) or max_bytes < 0:
        raise ValueError('max_bytes must be a nonnegative integer')
    graph = _validated_graph(graph, seed)
    # This is a local cache fingerprint, not an asset/provenance signature.
    # SHA1 avoids making full-buffer validation slower than the Dijkstra solve
    # on supported Windows builds; provenance continues to use SHA256 elsewhere.
    digest = hashlib.sha1(np.asarray(graph.shape, dtype='<i8').tobytes(), usedforsecurity=False)
    for array in (graph.indptr, graph.indices, graph.data):
        # Hash existing buffers; widening every index to int64 would make cache
        # lookup more expensive than solving a small cortical shortest path.
        contiguous = np.ascontiguousarray(array)
        digest.update(str((contiguous.dtype.str, contiguous.shape)).encode('ascii'))
        digest.update(memoryview(contiguous).cast('B'))
    identity = digest.hexdigest()
    if cache and next(iter(cache))[0] != identity:
        cache.clear()
    key = (identity, int(seed))
    distances = cache.pop(key, None)
    if distances is None:
        distances = geodesic_distances(graph, seed)
        distances.setflags(write=False)
    if distances.nbytes <= max_bytes:
        cache[key] = distances
    while cache and sum(value.nbytes for value in cache.values()) > max_bytes:
        del cache[next(iter(cache))]
    return distances


def grow_geodesic_patch(seed, graph, vertex_areas_mm2, target_area_mm2):
    """Grow a complete distance ball until its vertex-area sum reaches target.

    Include the entire final equal-distance shell (1e-10 mm / 1e-12 relative
    tolerance), avoiding arbitrary vertex-ID tie-breaking. Area overshoot is
    bounded by that shell's area, not by one vertex. No excluded bridge is used.
    """
    areas = np.asarray(vertex_areas_mm2, dtype=float)
    if (areas.shape != (graph.shape[0],) or not np.isfinite(areas).all() or np.any(areas < 0)
            or not np.isfinite(target_area_mm2) or target_area_mm2 <= 0):
        raise ValueError('Vertex areas must be finite/nonnegative and target area finite/positive')
    distances = geodesic_distances(graph, seed)
    reachable = np.flatnonzero(np.isfinite(distances))
    order = reachable[np.argsort(distances[reachable], kind='stable')]
    cumulative = np.cumsum(areas[order])
    if cumulative[-1] + 1e-9 < target_area_mm2:
        raise PatchCapacityError(f'Requested patch area cannot fit seed {seed}\'s connected cortical component')
    cutoff_index = min(int(np.searchsorted(cumulative, target_area_mm2, side='left')), len(order)-1)
    radius = float(distances[order[cutoff_index]])
    shell = np.isclose(distances[order], radius, rtol=1e-12, atol=1e-10)
    support = order[(distances[order] < radius) | shell]
    actual = float(areas[support].sum())
    return support, distances[support], {
        'geodesic_radius_mm': radius, 'actual_area_mm2': actual,
        'area_overshoot_mm2': max(0., actual-float(target_area_mm2)),
        'final_shell_area_mm2': float(areas[order[shell]].sum()),
        'geodesic_tie_policy': 'complete_equal_distance_shell',
    }
