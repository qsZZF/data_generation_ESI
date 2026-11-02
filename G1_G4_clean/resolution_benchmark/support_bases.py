"""Anatomical surface FEM bases and geodesic support dictionaries.

The FEM uses the supplied cortical triangulation, never a Euclidean kNN graph.
Distances follow mesh edges and overestimate continuous surface geodesics.
Source coefficients use an unweighted Euclidean metric after restriction;
sampled mesh masses are diagnostic values, not whole-cortex quadrature weights.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.sparse.linalg import eigsh, splu


def full_green_resolvent(stiffness, mass_m2, selected, length_m=.02):
    """Solve (M + length^2 K) G = point impulses on the complete FEM mesh.

    The selected principal block of the inverse is symmetric before column
    normalization. Unit point impulses differ from M times vertex delta only
    by a column scale, removed by the final normalization. All spatial modes
    contribute; this calculation does not use the truncated LBO representation.
    """
    stiffness = sparse.csc_matrix(stiffness, dtype=float)
    mass = np.asarray(mass_m2, dtype=float)
    selected = np.asarray(selected, dtype=np.int64)
    if (stiffness.shape != (len(mass), len(mass)) or mass.ndim != 1
            or not np.isfinite(mass).all() or np.any(mass <= 0)
            or not np.isfinite(stiffness.data).all() or selected.ndim != 1
            or not len(selected) or len(np.unique(selected)) != len(selected)
            or selected.min() < 0 or selected.max() >= len(mass)
            or not np.isfinite(length_m) or length_m <= 0):
        raise ValueError('Expected a finite FEM matrix, positive masses/length and valid selected vertices')
    operator = sparse.diags(mass, format='csc') + length_m**2*stiffness
    factor = splu(operator)
    raw = np.empty((len(selected), len(selected)))
    residual_squared = 0.
    for start in range(0, len(selected), 32):
        stop = min(start+32, len(selected))
        rhs = np.zeros((len(mass), stop-start))
        rhs[selected[start:stop], np.arange(stop-start)] = 1.
        solution = factor.solve(rhs)
        residual_squared += float(np.linalg.norm(operator @ solution-rhs)**2)
        raw[:, start:stop] = solution[selected]
    return dict(raw=raw, normalized=_normalize_columns(raw),
                pde_relative_residual=float(np.sqrt(residual_squared/len(selected))))


def build_surface_basis(rr_m, tris, selected_vertices, n_modes=32):
    """Solve cotangent K phi = lambda M phi on one connected hemisphere.

    ``tris`` indexes ``rr_m``; unused vertices are removed without changing
    their anatomical IDs. M is the lumped barycentric FEM mass in square meters.
    Returns raw mass-normalized modes restricted in requested source order,
    and selected-to-selected mesh-edge distances in meters.
    """
    rr = np.asarray(rr_m, dtype=float)
    triangles = np.asarray(tris, dtype=np.int64)
    selected = np.asarray(selected_vertices, dtype=np.int64)
    if (rr.ndim != 2 or rr.shape[1] != 3 or not np.isfinite(rr).all()
            or triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles)
            or triangles.min() < 0 or triangles.max() >= len(rr)
            or selected.ndim != 1 or not len(selected) or len(np.unique(selected)) != len(selected)):
        raise ValueError('Expected finite N x 3 coordinates, valid triangles and unique selected vertices')
    vertices, inverse = np.unique(triangles, return_inverse=True)
    tri = inverse.reshape(-1, 3)
    selection = np.searchsorted(vertices, selected)
    if np.any(selection >= len(vertices)) or not np.array_equal(vertices[selection], selected):
        raise ValueError('All selected vertices must belong to the supplied triangulation')
    if isinstance(n_modes, bool) or not isinstance(n_modes, (int, np.integer)) or not 1 <= n_modes < len(vertices):
        raise ValueError('n_modes must be an integer below the number of mesh vertices')
    points = rr[vertices]
    a, b, c = (points[tri[:, k]] for k in range(3))
    doubled_area = np.linalg.norm(np.cross(b-a, c-a), axis=1)
    if np.any(doubled_area <= np.finfo(float).eps * max(np.ptp(points, axis=0).max()**2, 1e-30)):
        raise ValueError('Degenerate triangles cannot define a cortical FEM')
    mass = np.bincount(tri.ravel(), weights=np.repeat(doubled_area/6., 3), minlength=len(vertices))
    cot = np.column_stack((np.einsum('ij,ij->i', b-a, c-a),
                           np.einsum('ij,ij->i', a-b, c-b),
                           np.einsum('ij,ij->i', a-c, b-c))) / doubled_area[:, None]
    rows, cols, values = [], [], []
    for k, (i, j) in enumerate(((1, 2), (0, 2), (0, 1))):
        weight = cot[:, k]/2.
        rows.extend([tri[:, i], tri[:, j], tri[:, i], tri[:, j]])
        cols.extend([tri[:, i], tri[:, j], tri[:, j], tri[:, i]])
        values.extend([weight, weight, -weight, -weight])
    stiffness = sparse.coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                                  shape=(len(vertices), len(vertices))).tocsr()
    edges = np.unique(np.sort(np.vstack((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [0, 2]])), axis=1), axis=0)
    edge_lengths = np.linalg.norm(points[edges[:, 0]]-points[edges[:, 1]], axis=1)
    graph = sparse.coo_matrix((np.tile(edge_lengths, 2),
                              (edges.T.ravel(), edges[:, ::-1].T.ravel())), shape=stiffness.shape).tocsr()
    if connected_components(graph, directed=False, return_labels=False) != 1:
        raise ValueError('One connected cortical hemisphere is required')
    if len(vertices) <= 256:
        eigenvalues, modes = eigh(stiffness.toarray(), np.diag(mass), subset_by_index=(0, n_modes-1))
    else:
        scale = float(np.median(stiffness.diagonal()/mass))
        eigenvalues, modes = eigsh(stiffness, k=n_modes, M=sparse.diags(mass),
                                   sigma=-max(scale*1e-8, 1e-12), which='LM',
                                   v0=np.random.default_rng(0).normal(size=len(vertices)), tol=1e-9)
        order = np.argsort(eigenvalues)
        eigenvalues, modes = eigenvalues[order], modes[:, order]
    tolerance = max(float(np.max(np.abs(eigenvalues))), 1.) * 1e-8
    if eigenvalues.min() < -tolerance:
        raise ValueError('FEM stiffness has unexpected negative eigenvalues')
    eigenvalues = np.maximum(eigenvalues, 0.)
    eigenvalues[0] = 0.  # Connected closed cortical surface has one constant mode.
    for column in range(n_modes):
        if modes[np.argmax(np.abs(modes[:, column])), column] < 0:
            modes[:, column] *= -1
    residual = stiffness @ modes - mass[:, None]*modes*eigenvalues
    residual_relative = float(np.linalg.norm(residual) / max(np.linalg.norm(stiffness @ modes), 1e-30))
    distances = dijkstra(graph, directed=False, indices=selection)[:, selection]
    green_full = full_green_resolvent(stiffness, mass, selection)
    return dict(raw_lbo=modes[selection], eigenvalues_m_inv2=eigenvalues,
                distances_m=distances, mesh_mass_m2=mass, mesh_lbo=modes,
                mesh_vertex_ids=vertices, selected_mesh_mass_m2=mass[selection],
                mesh_vertices=len(vertices), mesh_triangles=len(tri),
                mesh_area_m2=float(mass.sum()), fem_relative_residual=residual_relative,
                green_full_20mm=green_full['normalized'],
                green_full_pde_relative_residual=green_full['pde_relative_residual'])


def _normalize_columns(matrix):
    norms = np.linalg.norm(matrix, axis=0)
    if np.any(norms <= np.finfo(float).eps):
        raise ValueError('Support dictionary contains zero columns')
    return matrix/norms


def support_dictionaries(raw_lbo, eigenvalues_m_inv2, distances_m):
    """Euclidean QR LBO, 15/30-mm patches and spectrally truncated Green kernels.

    Green columns are point-source kernels of (I + length^2 Delta)^-1,
    approximated using ONLY the saved FEM modes, then normalized. They are not
    an exact full-band Green solve. QR columns span restricted LBO but need not
    individually be Laplace eigenfunctions.
    """
    raw = np.asarray(raw_lbo, dtype=float)
    eigenvalues = np.asarray(eigenvalues_m_inv2, dtype=float)
    distances = np.asarray(distances_m, dtype=float)
    if (raw.ndim != 2 or raw.shape[1] > raw.shape[0]
            or eigenvalues.shape != (raw.shape[1],) or distances.shape != (raw.shape[0], raw.shape[0])
            or not np.isfinite(raw).all() or not np.isfinite(eigenvalues).all()
            or np.any(eigenvalues < 0) or np.any(np.isnan(distances)) or np.any(distances < 0)):
        raise ValueError('Inconsistent or invalid FEM modes, eigenvalues and distance matrix')
    q, r = np.linalg.qr(raw, mode='reduced')
    if np.min(np.abs(np.diag(r))) < 1e-10*np.max(np.abs(np.diag(r))):
        raise ValueError('Restricted LBO columns are rank deficient')
    dictionaries = {'lbo': q*np.where(np.diag(r) < 0., -1., 1.)}
    for radius in (.015, .03):
        tag = f'{round(radius*1000)}mm'
        dictionaries[f'patch_{tag}'] = _normalize_columns((distances <= radius).astype(float))
        green = (raw/(1.+radius**2*eigenvalues)) @ raw.T
        # Exact disconnected hemispheres remain disconnected despite roundoff.
        green[~np.isfinite(distances)] = 0.
        dictionaries[f'green_{tag}'] = _normalize_columns(green)
    return dictionaries


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_basis_archive(forward_npz, source_fif, output_path, n_modes=32, *, overwrite=False):
    """Prepare a numeric NPZ dictionary, respecting exported source order/IDs.

    Uses MNE ``use_tris`` (anatomical source-space tessellation) when present;
    full cortical triangles are the fallback. No downloads are attempted.
    """
    os.environ.setdefault('MNE_LOGGING_LEVEL', 'ERROR')
    import mne

    forward_npz, source_fif, output_path = map(Path, (forward_npz, source_fif, output_path))
    if output_path.resolve() in (forward_npz.resolve(), source_fif.resolve()):
        raise ValueError('Output must not overwrite an input anatomical asset')
    if output_path.exists() and not overwrite:
        raise FileExistsError(f'Output already exists; use explicit overwrite: {output_path}')
    if not forward_npz.is_file() or not source_fif.is_file():
        raise FileNotFoundError('Both existing forward NPZ and anatomical source FIF are required')
    if 'fwd' in source_fif.name:
        spaces = mne.read_forward_solution(source_fif, verbose='ERROR')['src']
    else:
        spaces = mne.read_source_spaces(source_fif, verbose='ERROR')
    with np.load(forward_npz, allow_pickle=False) as archive:
        vertex_ids = archive['source_vertex_ids'].copy()
        hemisphere = archive['source_hemi'].copy()
    labels = np.unique(hemisphere)
    if len(spaces) != 2 or len(labels) != 2:
        raise ValueError('Expected a two-hemisphere cortical source model')
    n_sources = len(vertex_ids)
    raw = np.zeros((n_sources, 2*n_modes))
    distances = np.full((n_sources, n_sources), np.inf)
    mass = np.zeros(n_sources)
    green_full = np.zeros((n_sources, n_sources))
    eigenvalues, surface_metadata = [], []
    for index, (source, label) in enumerate(zip(spaces, labels)):
        selected_rows = np.flatnonzero(hemisphere == label)
        triangles = source['use_tris']
        triangulation = 'source_space_use_tris'
        if triangles is None:
            triangles = source['tris']
            triangulation = 'full_cortical_tris'
        result = build_surface_basis(source['rr'], triangles, vertex_ids[selected_rows], n_modes)
        raw[selected_rows, index*n_modes:(index+1)*n_modes] = result['raw_lbo']
        distances[np.ix_(selected_rows, selected_rows)] = result['distances_m']
        mass[selected_rows] = result['selected_mesh_mass_m2']
        green_full[np.ix_(selected_rows, selected_rows)] = result['green_full_20mm']
        eigenvalues.extend(result['eigenvalues_m_inv2'])
        surface_metadata.append(dict(hemisphere=str(label), triangulation=triangulation,
            **{key: result[key] for key in ('mesh_vertices', 'mesh_triangles', 'mesh_area_m2',
                                           'fem_relative_residual', 'green_full_pde_relative_residual')}))
    eigenvalues = np.asarray(eigenvalues)
    mode_order = np.argsort(eigenvalues, kind='stable')
    raw, eigenvalues = raw[:, mode_order], eigenvalues[mode_order]
    arrays = dict(raw_lbo=raw, raw_modes=raw, eigenvalues_m_inv2=eigenvalues,
                  eigenvalues=eigenvalues, distances_m=distances, geodesic_distances_m=distances,
                  green_full_20mm=green_full,
                  selected_mesh_mass_m2=mass, source_vertex_ids=vertex_ids, source_hemi=hemisphere,
                  **support_dictionaries(raw, eigenvalues, distances))
    metadata = dict(method='cotangent surface FEM with lumped barycentric mass',
        coefficient_metric='Euclidean after source restriction; lbo is QR of raw_lbo',
        modes_per_hemisphere=n_modes, surfaces=surface_metadata,
        green='column-normalized spectrally truncated point Green kernel; (1+length^2*lambda)^-1',
        green_full_20mm='full FEM sparse-LU resolvent (M + 0.020^2 K)^-1 point impulses; '
                         'restricted to exported vertices and unit-column-normalized; no spectral truncation',
        distance='shortest mesh-edge paths; not exact continuous geodesic',
        mass='sampled mesh-vertex areas only, not full-cortex source quadrature',
        source_fif=str(source_fif.resolve()), source_fif_sha256=_sha256(source_fif),
        forward_sha256=_sha256(forward_npz), builder_sha256=_sha256(__file__))
    arrays['metadata_json'] = np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix('.partial.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(output_path)
    return arrays, metadata
