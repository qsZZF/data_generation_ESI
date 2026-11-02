"""Smooth residual fields outside a retained coarse cortical subspace.

The Gaussian resolvent/SPDE construction is a finite-element approximation,
not an exact continuum Matern sampler. Spatial signs are never rectified.
"""
from collections.abc import MutableMapping
import hashlib
import json

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu

try:
    from .geometry import _array_hash, _mesh_arrays, _mesh_hash, _fem_matrices, _validate_bundle
except ImportError:
    from geometry import _array_hash, _mesh_arrays, _mesh_hash, _fem_matrices, _validate_bundle


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def generate_residual_patterns(simulator, bundle, *, seed, n_components=8,
                               length_mm=10., n_modes=None, cache=None):
    """Return N-by-K signed patterns, each with unit FEM mass norm.

    Smooth white innovations with (M + length_mm**2 K)^-1 M**.5, then
    remove the mass-weighted span of retained low modes and one constant per
    eligible connected component. The full FEM operator is restricted to the
    eligible domain: masked vertices have zero Dirichlet values; any original
    open mesh boundary retains its natural FEM condition. Projection is done
    AFTER this restriction and smoothing, so masking cannot reintroduce a low
    mode. Patterns need not be mutually orthogonal or statistically independent.

    A caller-owned cache stores at most two factorizations/projection bases;
    random innovations and generated patterns are never cached.
    """
    n_components = _positive_integer(n_components, 'n_components')
    if isinstance(length_mm, (bool, np.bool_)) or not np.isscalar(length_mm) or not np.isreal(length_mm) or not np.isfinite(length_mm) or length_mm <= 0:
        raise ValueError('length_mm must be a finite positive scalar')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if cache is not None and not isinstance(cache, MutableMapping):
        raise ValueError('cache must be a dedicated mutable mapping or None')
    vertices, faces = _mesh_arrays(np.asarray(simulator.vertices)*simulator.coord_to_mm, simulator.faces)
    basis, bundle_mass, _ = _validate_bundle(bundle)
    n_modes = basis.shape[1] if n_modes is None else _positive_integer(n_modes, 'n_modes')
    if basis.shape[0] != len(vertices) or n_modes > basis.shape[1]:
        raise ValueError('Retained n_modes/basis dimensions do not match the source mesh')
    mesh_hash = _mesh_hash(vertices, faces)
    if bundle.get('mesh_sha256', mesh_hash) != mesh_hash:
        raise ValueError('LBO bundle belongs to a different mesh')
    mask = np.asarray(simulator.cortex_mask)
    if mask.shape != (len(vertices),) or not np.isin(mask, [0, 1]).all() or not np.any(mask):
        raise ValueError('Cortex mask must retain at least one valid vertex')
    eligible = np.flatnonzero(mask)
    identity = dict(mesh_sha256=mesh_hash, mask_sha256=_array_hash(mask, 'u1'),
                    basis_sha256=_array_hash(basis, '<f8'), mass_sha256=_array_hash(bundle_mass, '<f8'),
                    n_modes=n_modes, length_mm=float(length_mm), algorithm_version=1)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if cache is not None and key in cache:
        prepared = cache.pop(key)
        cache[key] = prepared
    else:
        stiffness, mass, adjacency = _fem_matrices(vertices, faces)
        if not np.allclose(mass, bundle_mass, rtol=1e-8, atol=1e-10):
            raise ValueError('LBO bundle mass differs from the mesh FEM mass')
        local_mass = mass[eligible]
        root_mass = np.sqrt(local_mass)
        count, labels = connected_components(adjacency[eligible][:, eligible], directed=False)
        constants = np.eye(count)[labels]
        candidates = root_mass[:, None] * np.column_stack((basis[eligible, :n_modes], constants))
        norms = np.linalg.norm(candidates, axis=0)
        candidates = candidates[:, norms > 0] / norms[norms > 0]
        u, singular, _ = np.linalg.svd(candidates, full_matrices=False)
        rank = int(np.count_nonzero(singular > singular[0]*1e-12))
        capacity = len(eligible)-rank
        if n_components > capacity:
            raise ValueError(f'Residual capacity {capacity} is smaller than requested n_components={n_components}')
        operator = sparse.diags(local_mass) + float(length_mm)**2 * stiffness[eligible][:, eligible]
        prepared = dict(factor=splu(operator.tocsc()), root_mass=root_mass, q=u[:, :rank],
                        projection_rank=rank, residual_capacity=capacity, component_count=count)
        if cache is not None:
            cache[key] = prepared
            while len(cache) > 2:
                del cache[next(iter(cache))]
    if n_components > prepared['residual_capacity']:
        raise ValueError('Requested n_components exceeds residual capacity')
    root_mass, q = prepared['root_mass'], prepared['q']
    innovations = np.random.default_rng(seed).standard_normal((len(eligible), n_components))
    field = prepared['factor'].solve(root_mass[:, None]*innovations)
    transformed = root_mass[:, None]*field
    # Repeat the small projection to reduce cancellation for strongly smoothed draws.
    for _ in range(2):
        transformed -= q @ (q.T @ transformed)
    norms = np.linalg.norm(transformed, axis=0)
    if not np.isfinite(norms).all() or np.any(norms <= np.finfo(float).tiny):
        raise ValueError('Residual draw has zero/nonfinite energy after coarse-mode projection')
    transformed /= norms
    patterns = np.zeros((len(vertices), n_components))
    patterns[eligible] = transformed / root_mass[:, None]
    info = dict(**identity, operator_key=key, seed=int(seed), n_components=n_components,
                eligible_vertices=len(eligible), eligible_connected_components=prepared['component_count'],
                removed_subspace_rank=prepared['projection_rank'], residual_capacity=prepared['residual_capacity'],
                spatial_model='Gaussian_FEM_resolvent_SPDE_approximation_not_exact_continuum_Matern',
                operator='(M + length_mm^2 K)^-1 M^0.5 epsilon; mass projection after smoothing',
                boundary_condition='Dirichlet zero at masked vertices; natural FEM boundary on original open mesh',
                projection_domain='eligible vertices; retained low LBO modes plus each component constant',
                pattern_normalization='each pattern has unit lumped-FEM-mass norm; signs preserved',
                max_mass_subspace_inner_product=float(np.max(np.abs(q.T @ transformed))),
                max_mass_norm_error=float(np.max(np.abs(np.sum(transformed**2, axis=0)-1))))
    return patterns, info


def mix_residual(coarse, residual, fraction, metric='vertex_mean', mass=None):
    """Keep coarse unchanged; set E_res/(E_coarse+E_res) to fraction.

    This component-power fraction differs from E_res/E_mixed when the fields
    have a nonzero cross term. No RNG is used; fraction=0 accepts residual=None
    and returns the coarse values/dtype unchanged.
    """
    original = np.asarray(coarse)
    if np.iscomplexobj(original) or original.ndim != 2 or min(original.shape) < 1 or not np.isfinite(original).all():
        raise ValueError('coarse must be a finite real vertices-by-time matrix')
    if isinstance(fraction, (bool, np.bool_)) or not np.isscalar(fraction) or not np.isreal(fraction) or not np.isfinite(fraction) or not 0 <= fraction < 1:
        raise ValueError('fraction must be finite and satisfy 0 <= fraction < 1')
    if metric not in ('vertex_mean', 'area_weighted'):
        raise ValueError('metric must be vertex_mean or area_weighted')
    if mass is not None:
        mass = np.asarray(mass, dtype=float)
        if mass.shape != (len(original),) or not np.isfinite(mass).all() or np.any(mass <= 0):
            raise ValueError('mass must be finite positive vertex weights')
    if metric == 'area_weighted' and mass is None:
        raise ValueError('area_weighted metric requires mass')
    weights = mass / mass.sum() if metric == 'area_weighted' else np.full(len(original), 1/len(original))
    values = original.astype(float)
    def power(array):
        return float(weights @ np.mean(array**2, axis=1))
    coarse_power = power(values)
    gain = 0.
    scaled = np.zeros_like(original)
    if fraction:
        raw = np.asarray(residual)
        if np.iscomplexobj(raw) or raw.shape != original.shape or not np.isfinite(raw).all():
            raise ValueError('residual must be finite, real and match coarse shape')
        raw = raw.astype(float)
        energy = power(raw)
        if coarse_power <= 0 or energy <= 0 or not np.isfinite(coarse_power+energy):
            raise ValueError('Coarse and residual energy must be finite and positive for nonzero fraction')
        gain = float(np.exp(.5*(np.log(fraction)-np.log1p(-fraction)+np.log(coarse_power)-np.log(energy))))
        scaled = raw*gain
    mixed = original.copy() if not fraction else values+scaled
    residual_power = power(scaled)
    cross_power = float(2 * weights @ np.mean(values*scaled, axis=1))
    mass_error = None
    if mass is not None:
        a = np.sum(mass[:, None]*values**2, axis=0)
        b = np.sum(mass[:, None]*scaled**2, axis=0)
        denominator = np.sqrt(a)*np.sqrt(b)
        cosine = np.divide(np.sum(mass[:, None]*values*scaled, axis=0), denominator,
                           out=np.zeros_like(a), where=denominator > 0)
        mass_error = float(np.max(np.abs(cosine)))
    info = dict(requested_component_fraction=float(fraction), metric=metric, residual_scale=gain,
                realized_component_fraction=residual_power/(coarse_power+residual_power) if residual_power else 0.,
                fraction_definition='E_residual/(E_coarse+E_residual), excludes cross term',
                coarse_power=coarse_power, residual_power=residual_power, cross_power=cross_power,
                mixed_power=power(mixed), coarse_scale=1., mass_orthogonality_error=mass_error,
                mass_orthogonality_definition='maximum absolute mass-weighted spatial cosine over paired time samples')
    if not np.isfinite(mixed).all() or not all(np.isfinite(info[name]) for name in ('coarse_power', 'residual_power', 'cross_power', 'mixed_power')):
        raise ValueError('Residual mixture exceeds finite numerical range')
    return mixed, scaled, info
