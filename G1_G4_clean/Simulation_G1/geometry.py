"""Area-weighted cortical Laplace--Beltrami modes on a supplied source mesh.

Coordinates are in mm, mass entries in mm**2, and eigenvalues in mm**-2.
Vertex order is never changed: the basis rows must match the leadfield columns.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh


def _array_hash(array, dtype):
    canonical = np.asarray(array, dtype=dtype, order='C')
    digest = hashlib.sha256()
    digest.update(np.asarray(canonical.shape, dtype='<i8').tobytes())
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def _mesh_arrays(vertices_mm, faces):
    vertices = np.asarray(vertices_mm, dtype=np.float64)
    raw_faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
        raise ValueError('Mesh vertices must have shape (N, 3), with N >= 3')
    if not np.isfinite(vertices).all():
        raise ValueError('Mesh vertices must be finite')
    if raw_faces.ndim != 2 or raw_faces.shape[1] != 3 or len(raw_faces) == 0:
        raise ValueError('Mesh faces must be a nonempty (F, 3) triangle array')
    if not np.issubdtype(raw_faces.dtype, np.number) or not np.isfinite(raw_faces).all():
        raise ValueError('Mesh faces must contain finite integer indices')
    if np.any(raw_faces < 0) or np.any(raw_faces >= len(vertices)) or np.any(raw_faces != np.floor(raw_faces)):
        raise ValueError('Mesh faces must contain zero-based integer vertex indices')
    triangles = np.asarray(raw_faces, dtype=np.int64)
    ordered = np.sort(triangles, axis=1)
    if np.any(np.diff(ordered, axis=1) == 0):
        raise ValueError('Mesh triangle repeats a vertex')
    if len(np.unique(ordered, axis=0)) != len(triangles):
        raise ValueError('Mesh contains duplicate triangles')
    return vertices, triangles


def _mesh_hash(vertices, triangles):
    identity = _array_hash(vertices, '<f8') + _array_hash(triangles, '<i8')
    return hashlib.sha256(identity.encode('ascii')).hexdigest()


def _fem_matrices(vertices, triangles):
    points = vertices[triangles]
    twice_area = np.linalg.norm(np.cross(points[:, 1] - points[:, 0],
                                         points[:, 2] - points[:, 0]), axis=1)
    edge_scale = np.sum((points[:, 1] - points[:, 0]) ** 2, axis=1)
    edge_scale = np.maximum(edge_scale, np.sum((points[:, 2] - points[:, 0]) ** 2, axis=1))
    if np.any(twice_area <= np.finfo(float).eps * edge_scale * 10):
        raise ValueError('Mesh contains a degenerate triangle')
    mass = np.bincount(triangles.ravel(), weights=np.repeat(twice_area / 6, 3),
                       minlength=len(vertices))
    if np.any(mass <= 0):
        raise ValueError('Mesh contains isolated vertices with zero area')
    rows, columns, weights = [], [], []
    for center, first, second in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        cotangent = np.sum((points[:, first] - points[:, center]) *
                           (points[:, second] - points[:, center]), axis=1) / twice_area
        rows.append(triangles[:, first])
        columns.append(triangles[:, second])
        weights.append(cotangent / 2)
    rows, columns, weights = np.concatenate(rows), np.concatenate(columns), np.concatenate(weights)
    adjacency = sparse.coo_matrix((np.ones(len(rows)), (rows, columns)),
                                   shape=(len(vertices), len(vertices))).tocsr()
    adjacency = adjacency + adjacency.T
    weighted = sparse.coo_matrix((weights, (rows, columns)), shape=adjacency.shape).tocsr()
    weighted = weighted + weighted.T
    stiffness = sparse.diags(np.asarray(weighted.sum(axis=1)).ravel()) - weighted
    return stiffness.tocsr(), mass, adjacency


def compute_basis(vertices_mm, faces, n_modes=128):
    """Return the lowest ``n_modes`` nonconstant modes across all components.

    Each connected component (normally one hemisphere) contributes its own
    constant null mode, which is omitted. The requested count is TOTAL across
    hemispheres. Repeated eigenvalues define a subspace, not unique individual
    modes; deterministic initialization and signs only fix numerical conventions.
    """
    vertices, triangles = _mesh_arrays(vertices_mm, faces)
    if isinstance(n_modes, (bool, np.bool_)) or not isinstance(n_modes, (int, np.integer)) or n_modes < 1:
        raise ValueError('n_modes must be a positive integer')
    stiffness, mass, adjacency = _fem_matrices(vertices, triangles)
    n_components, labels = connected_components(adjacency, directed=False)
    if n_modes > len(vertices) - n_components:
        raise ValueError('n_modes exceeds the number of nonconstant mesh modes')
    candidates = []
    for label in range(n_components):
        indices = np.flatnonzero(labels == label)
        local_mass = mass[indices]
        inverse_sqrt = 1 / np.sqrt(local_mass)
        local_stiffness = stiffness[indices][:, indices]
        normalized = sparse.diags(inverse_sqrt) @ local_stiffness @ sparse.diags(inverse_sqrt)
        count = min(n_modes, len(indices) - 1)
        if len(indices) <= 64 or count >= len(indices) - 1:
            values, vectors = eigh(normalized.toarray(), subset_by_index=(0, count))
        else:
            scale = max(float(np.median(normalized.diagonal())), np.finfo(float).tiny)
            values, vectors = eigsh(normalized, k=count + 1, sigma=-scale * 1e-8,
                                    which='LM', tol=1e-10,
                                    v0=np.random.default_rng(0).normal(size=len(indices)))
            order = np.argsort(values)
            values, vectors = values[order], vectors[:, order]
        # Connected FEM surfaces have exactly one constant null mode.
        local_basis = inverse_sqrt[:, None] * vectors[:, 1:count + 1]
        for column, value in enumerate(values[1:count + 1]):
            if value <= 0:
                raise ValueError('Mesh eigensolver returned a nonpositive nonconstant eigenvalue')
            vector = local_basis[:, column]
            if vector[np.argmax(np.abs(vector))] < 0:
                vector = -vector
            candidates.append((float(value), label, indices, vector))
    candidates.sort(key=lambda item: item[0])
    selected = candidates[:n_modes]
    basis = np.zeros((len(vertices), n_modes), dtype=np.float64)
    eigenvalues = np.array([item[0] for item in selected])
    counts = [0] * n_components
    for column, (_, label, indices, vector) in enumerate(selected):
        basis[indices, column] = vector
        counts[label] += 1
    left = stiffness @ basis
    right = (mass[:, None] * basis) * eigenvalues
    residual = np.linalg.norm(left - right, axis=0) / np.maximum(
        np.linalg.norm(left, axis=0) + np.linalg.norm(right, axis=0), np.finfo(float).tiny)
    return {
        'basis': basis, 'eigenvalues': eigenvalues, 'mass': mass,
        'mesh_sha256': _mesh_hash(vertices, triangles), 'component_labels': labels,
        'metadata': {
            'schema_version': 1, 'operator': 'cotangent_fem_lumped_mass',
            'coordinate_unit': 'mm', 'mass_unit': 'mm2', 'eigenvalue_unit': 'mm-2',
            'vertices_sha256': _array_hash(vertices, '<f8'),
            'n_vertices': len(vertices), 'n_faces': len(triangles),
            'n_modes': int(n_modes), 'n_components': int(n_components),
            'component_mode_counts': counts, 'constant_modes_omitted': int(n_components),
            'solver': 'component-wise symmetric mass transform; eigh or shift-invert eigsh',
            'degeneracy_convention': 'fixed initial vector and positive largest entry; degenerate subspaces may rotate across solvers',
            'max_relative_eigen_residual': float(np.max(residual)),
        },
    }


def _validate_bundle(bundle):
    basis = np.asarray(bundle['basis'], dtype=float)
    mass = np.asarray(bundle['mass'], dtype=float)
    values = np.asarray(bundle['eigenvalues'], dtype=float)
    if (basis.ndim != 2 or min(basis.shape) == 0 or mass.shape != (basis.shape[0],)
            or values.shape != (basis.shape[1],)):
        raise ValueError('Invalid basis bundle dimensions')
    if not all(np.isfinite(array).all() for array in (basis, mass, values)):
        raise ValueError('Basis bundle contains nonfinite values')
    if np.any(mass <= 0) or np.any(values <= 0) or np.any(np.diff(values) < 0):
        raise ValueError('Basis requires positive mass and ordered positive eigenvalues')
    return basis, mass, values


def _validate_cache_metadata(bundle):
    labels = np.asarray(bundle['component_labels'])
    if (labels.shape != np.asarray(bundle['mass']).shape
            or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0)):
        raise ValueError('Invalid basis component labels')
    metadata = bundle['metadata']
    if metadata.get('schema_version') != 1:
        raise ValueError('Unsupported geometry basis schema version')
    if (metadata.get('n_vertices') != len(labels)
            or metadata.get('n_modes') != bundle['basis'].shape[1]
            or metadata.get('n_components') != len(np.unique(labels))):
        raise ValueError('Geometry basis metadata does not match array dimensions')


def save_basis(path, bundle):
    """Save numeric arrays plus JSON metadata without pickle payloads."""
    basis, mass, values = _validate_bundle(bundle)
    _validate_cache_metadata(bundle)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('wb') as stream:
        np.savez_compressed(stream, basis=basis, mass=mass, eigenvalues=values,
                            mesh_sha256=np.asarray(bundle['mesh_sha256']),
                            component_labels=np.asarray(bundle['component_labels'], dtype=np.int32),
                            metadata_json=np.asarray(json.dumps(bundle['metadata'], sort_keys=True)))


def load_basis(path, vertices_mm=None, faces=None):
    """Load a cache and optionally require the exact mesh and vertex order."""
    if (vertices_mm is None) != (faces is None):
        raise ValueError('Provide vertices_mm and faces together to validate mesh identity')
    with np.load(path, allow_pickle=False) as archive:
        bundle = {name: archive[name] for name in ('basis', 'mass', 'eigenvalues', 'component_labels')}
        bundle['mesh_sha256'] = str(archive['mesh_sha256'].item())
        bundle['metadata'] = json.loads(str(archive['metadata_json'].item()))
    _validate_bundle(bundle)
    _validate_cache_metadata(bundle)
    if vertices_mm is not None:
        vertices, triangles = _mesh_arrays(vertices_mm, faces)
        if bundle['mesh_sha256'] != _mesh_hash(vertices, triangles):
            raise ValueError('Cached basis mesh identity differs from the supplied mesh')
    return bundle


def project_maps(bundle, maps, *, vertices_mm=None):
    """Project area-normalized vertex maps; retain loss from modal truncation.

    Maps have shape (vertices, networks). A nonzero constant map is legal and
    projects to zero because component constants are omitted. Coordinate checks
    certify ordering only; callers must first register foreign cortical maps.
    """
    basis, mass, _ = _validate_bundle(bundle)
    maps = np.asarray(maps, dtype=float)
    if maps.ndim != 2 or maps.shape[0] != len(mass) or maps.shape[1] == 0:
        raise ValueError('Maps must have shape (mesh vertices, nonempty networks)')
    if not np.isfinite(maps).all():
        raise ValueError('Maps must be finite')
    if vertices_mm is not None and _array_hash(vertices_mm, '<f8') != bundle['metadata']['vertices_sha256']:
        raise ValueError('Map coordinates differ from the basis vertex coordinates')
    energy = np.sum(mass[:, None] * maps ** 2, axis=0)
    if np.any(energy <= 0):
        raise ValueError('Maps must have nonzero area-weighted energy')
    return basis.T @ (mass[:, None] * (maps / np.sqrt(energy)))
