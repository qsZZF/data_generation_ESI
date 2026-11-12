"""Experimental geodesic network subregions; the first WX pilot stays frozen."""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from .geometry import _mesh_arrays, _mesh_hash
from .geodesic import mesh_edge_graph, geodesic_distances
from .wx_pilot import build_weights


def build_partitioned_weights(bundle, network_maps, cortical_mask, vertices_mm, faces,
                              *, seed=0, length_mm=20., local_width_mm=30.,
                              n_networks=3, local_per_network=2):
    """Preserve shared maps and add smooth signed geodesic subregion columns.

    Width is Gaussian standard deviation in mesh-edge millimetres. Support uses
    |base|/max >= .1, with a smooth amplitude taper to one at .3. Small isolated
    components are excluded before center selection, not amplified by unit-norm
    scaling. Spatial normalization does not orthogonalize overlapping networks.
    """
    if (isinstance(local_width_mm, bool) or not np.isfinite(local_width_mm)
            or local_width_mm <= 0):
        raise ValueError('local_width_mm must be positive and finite')
    vertices, triangles = _mesh_arrays(vertices_mm, faces)
    if len(vertices) != len(bundle['mass']):
        raise ValueError('Geometry and basis must have the same vertex rows')
    if bundle.get('mesh_sha256') is not None and bundle['mesh_sha256'] != _mesh_hash(vertices, triangles):
        raise ValueError('Geometry does not match the saved basis')
    result = build_weights(bundle, network_maps, cortical_mask, seed=seed,
                           length_mm=length_mm, n_networks=n_networks,
                           local_per_network=local_per_network)
    W = result['W'].copy()
    mass, mask = result['mass'], np.asarray(cortical_mask)
    edges = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
    adjacency = sparse.csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                                  shape=(len(vertices), len(vertices)))
    adjacency = adjacency.maximum(adjacency.T)
    rng = np.random.default_rng(seed)
    centers, supports, excluded_counts = [], [], []
    for group in range(n_networks):
        columns = np.flatnonzero(result['groups'] == group)
        base = W[:, columns[0]]
        relative = np.abs(base)/np.max(np.abs(base))
        eligible = mask & (relative >= .1)
        selected = np.flatnonzero(eligible)
        if len(selected) < 3:
            raise ValueError('Network support has fewer than three eligible vertices')
        graph = mesh_edge_graph(vertices, adjacency, eligible)
        count, labels = connected_components(graph[selected][:, selected], directed=False)
        areas = np.bincount(labels, weights=mass[selected], minlength=count)
        energy = np.bincount(labels, weights=mass[selected]*base[selected]**2, minlength=count)
        sizes = np.bincount(labels, minlength=count)
        retained = (sizes >= 3) & (areas >= .02*areas.max()) & (energy >= .02*energy.max())
        if not np.any(retained):
            raise ValueError('No substantial connected network support remains')
        eligible[selected[~retained[labels]]] = False
        graph = mesh_edge_graph(vertices, adjacency, eligible)
        taper = np.sin(.5*np.pi*np.clip((relative-.1)/.2, 0, 1))**2
        taper *= eligible
        candidate_energy = mass*(base*taper)**2
        if np.count_nonzero(candidate_energy) < local_per_network:
            raise ValueError('Not enough energetic vertices for distinct local centers')
        chosen, distances = [], np.full(len(vertices), np.inf)
        for column in columns[1:]:
            if not chosen:
                center = int(rng.choice(len(vertices), p=candidate_energy/candidate_energy.sum()))
            else:
                disconnected = eligible & np.isinf(distances) & (candidate_energy > 0)
                if np.any(disconnected):
                    center = int(np.argmax(np.where(disconnected, candidate_energy, -1)))
                else:
                    score = np.where(eligible, np.minimum(distances, 1e6), 0)
                    score *= np.sqrt(candidate_energy/candidate_energy.max())
                    score[chosen] = -1
                    center = int(np.argmax(score))
            chosen.append(center)
            distance = geodesic_distances(graph, center)
            distances = np.minimum(distances, distance)
            local = base*taper*np.exp(-.5*(distance/local_width_mm)**2)
            norm = np.sqrt(np.sum(mass*local**2))
            if not np.isfinite(norm) or norm <= 1e-14:
                raise ValueError('Degenerate local kernel; increase width or revise support')
            W[:, column] = local/norm
        centers.append(chosen)
        supports.append(int(eligible.sum()))
        excluded_counts.append(int(np.count_nonzero(~retained)))
    basis = bundle['basis']
    coefficients = basis.T @ (mass[:, None]*W)
    residual = W-basis@coefficients
    component_labels = np.asarray(bundle.get('component_labels', np.zeros(len(mass), dtype=int)))
    for label in np.unique(component_labels):
        indices = component_labels == label
        residual[indices] -= np.average(W[indices], weights=mass[indices], axis=0)
    cosine = W.T @ (mass[:, None]*W)
    result['W'] = W
    result['metadata'].update(
        spatial_model='geodesic_network_subregions_v1', local_width_mm=float(local_width_mm),
        centers_zero_based=centers, supported_vertices_per_network=supports,
        excluded_components_per_network=excluded_counts,
        support_rule='abs(shared)/max>=0.1; smooth taper0.1..0.3; components>=3vertices and>=2%largest area/energy',
        center_rule='area-energy weighted first; prioritize substantial disconnected support; then distance*sqrt(energy)',
        distance_metric='masked network-support mesh-edge geodesic, mm',
        weight_cosine=cosine.tolist(),
        unresolved_energy=np.sum(mass[:, None]*residual**2, axis=0).tolist(),
        retained_modal_roughness=np.sum(bundle['eigenvalues'][:, None]*coefficients**2, axis=0).tolist())
    return result
