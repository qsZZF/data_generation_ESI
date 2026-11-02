"""Opt-in sample labels for physical and noise-retained observable subspaces.

Source energy uses the supplied per-vertex Euclidean coordinates, not cortical
area weighting. Numerical rank and relative conditioning are distinct from
statistical identifiability. No inverse estimate or dense source projector is
formed. Decomposing a large forward can take seconds; the optional caller-owned
cache retains at most two exact forward/projection/covariance/settings entries.
"""
from __future__ import annotations

from collections.abc import MutableMapping
import hashlib
import json

import numpy as np

try:
    from .noise_covariance import _constraint_basis, _standardized_spectrum, whitener_from_covariance
except ImportError:
    from noise_covariance import _constraint_basis, _standardized_spectrum, whitener_from_covariance


def _matrix(value, name):
    values = np.asarray(value, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) == 0 or not np.isfinite(values).all():
        raise ValueError(f'{name} must be a finite nonempty two-dimensional matrix')
    return values


def _hash(values):
    if values is None:
        return None
    values = np.ascontiguousarray(values, dtype='<f8')
    digest = hashlib.sha256(str(values.shape).encode('ascii'))
    digest.update(memoryview(values).cast('B'))
    return digest.hexdigest()


def _svd(operator, tolerance, *, retain_all=False):
    if not np.isfinite(operator).all():
        raise ValueError('Projected or whitened forward exceeds finite numerical range')
    if not operator.shape[0]:
        return np.empty(0), np.empty((0, operator.shape[1]))
    _, singular, right = np.linalg.svd(operator, full_matrices=False)
    if not np.isfinite(singular).all():
        raise ValueError('Forward singular values exceed finite numerical range')
    if retain_all:
        return singular, right
    retained = singular > singular[0] * tolerance
    return singular[retained], right[retained]


def _mode_energy(right, source):
    energy = np.zeros(len(right))
    for start in range(0, source.shape[1], 256):
        modes = right @ source[:, start:start + 256]
        energy += np.einsum('ij,ij->i', modes, modes)
    return energy


def _decompose(gain, covariance, average_reference, projector, numerical_rtol, near_null_rtol):
    basis = _constraint_basis(len(gain), average_reference, projector)
    projection = basis @ basis.T
    constrained = average_reference or projector is not None
    projection_roundoff_rtol = 8 * np.finfo(float).eps * len(gain)
    projected = projection @ gain if constrained else gain
    if constrained:
        # Do not amplify cancellation roundoff into real rows during equilibration.
        error_bound = projection_roundoff_rtol * (np.abs(projection) @ np.abs(gain))
        projected[np.abs(projected) <= error_bound] = 0.
    # Scale rows before SVD: physical sensor units must not decide geometric rank.
    row_max = np.max(np.abs(projected), axis=1)
    positive = row_max > 0
    equilibrated = projected[positive] / row_max[positive, None]
    equilibrated /= np.linalg.norm(equilibrated, axis=1, keepdims=True)
    _, geometric_right = _svd(equilibrated, numerical_rtol)
    info = dict(projector_sha256=_hash(projection), constraint_rank=int(basis.shape[1]),
                projection_roundoff_rtol=float(projection_roundoff_rtol) if constrained else 0.,
                geometric_rank=len(geometric_right), geometric_nullity=gain.shape[1] - len(geometric_right),
                noise_status='not_provided', noise_rank=None, whitened_rank=None,
                white_nullity=None, white_stable_rank=None, white_near_null_rank=None,
                lost_noise_directions=None, noise_projection_rowspaces_nested=None,
                noise_rowspace_outside_geometric_fraction=None, projected_noise_covariance_sha256=None)
    white_right, singular = np.empty((0, gain.shape[1])), np.empty(0)
    all_white_right, all_white_singular = white_right, singular
    if covariance is not None:
        _standardized_spectrum(covariance, numerical_rtol)
        projected_covariance = projection @ covariance @ projection.T if constrained else covariance
        if constrained:
            error_bound = projection_roundoff_rtol * (np.abs(projection) @ np.abs(covariance) @ np.abs(projection.T))
            projected_covariance[np.abs(projected_covariance) <= error_bound] = 0.
        projected_covariance = (projected_covariance + projected_covariance.T) / 2
        info['projected_noise_covariance_sha256'] = _hash(projected_covariance)
        if not np.any(projected_covariance):
            info['noise_status'] = 'zero_noise_covariance'
        else:
            whitener = whitener_from_covariance(projected_covariance, rtol=numerical_rtol)
            all_white_singular, all_white_right = _svd(whitener @ projected, numerical_rtol, retain_all=True)
            rank = int(np.count_nonzero(all_white_singular > all_white_singular[0] * numerical_rtol))
            singular, white_right = all_white_singular[:rank], all_white_right[:rank]
            stable = int(np.count_nonzero(singular >= singular[0] * near_null_rtol)) if len(singular) else 0
            # Compare row spaces without allocating an N_source x N_source matrix.
            overlap = geometric_right @ white_right.T
            outside = max(0., 1. - float(np.sum(overlap ** 2)) / len(white_right)) if len(white_right) else 0.
            nested = len(white_right) <= len(geometric_right) and outside <= max(1e-8, numerical_rtol * 10)
            info.update(noise_status='available', noise_rank=len(whitener), whitened_rank=len(white_right),
                        white_nullity=gain.shape[1] - len(white_right), white_stable_rank=stable,
                        white_near_null_rank=len(white_right) - stable,
                        lost_noise_directions=len(geometric_right) - len(white_right) if nested else None,
                        noise_projection_rowspaces_nested=bool(nested),
                        noise_rowspace_outside_geometric_fraction=outside)
    for array in (geometric_right, white_right, singular, all_white_right, all_white_singular):
        array.setflags(write=False)
    return dict(info=info, geometric_right=geometric_right, white_right=white_right, singular=singular,
                all_white_right=all_white_right, all_white_singular=all_white_singular)


def analyze_observability(G, source, noise_covariance=None, average_reference=False,
                          projector=None, numerical_rtol=1e-10, near_null_rtol=1e-3, cache=None):
    """Return JSON-safe scalar diagnostics for a forward and a source sample.

    ``geometric_*`` uses row-equilibrated P G. ``white_*`` uses W P G and may
    discard additional directions when baseline covariance has limited rank;
    its null fraction is NOT the physical forward's exact null fraction.

    The optional continuous Wiener diagnostic uses f_i = s_i² v/(1+s_i² v),
    with v equal to this sample's mean squared per-vertex source amplitude.
    It is an isotropic-prior diagnostic, not oracle identifiability. Equal
    singular values receive equal weights, so degenerate-mode rotations do
    not change the aggregate. Near-null cutoffs describe relative conditioning,
    not a signal-to-noise decision. Zero source energy yields null fractions None.
    ``whitened_target_to_baseline_power_db`` uses all W P G modes and averages
    power across every retained noise direction and time point. Its reference
    is total baseline nuisance with unit variance, not sensor-noise-only SNR.
    Missing/zero noise or zero projected target power yields None.

    ``cache`` is a dedicated mutable mapping; cached SVDs depend on exact G,
    covariance, projector, reference and tolerances. Source-dependent energies
    and weights are always recomputed. The mapping is limited to two entries.
    """
    gain = _matrix(G, 'G')
    values = np.asarray(source, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    values = _matrix(values, 'source')
    if values.shape[0] != gain.shape[1]:
        raise ValueError('source rows must match G columns')
    if (not np.isfinite(numerical_rtol) or not np.isfinite(near_null_rtol)
            or not 0 < numerical_rtol <= near_null_rtol < 1):
        raise ValueError('Require 0 < numerical_rtol <= near_null_rtol < 1')
    if cache is not None and not isinstance(cache, MutableMapping):
        raise ValueError('cache must be a dedicated mutable mapping or None')
    covariance = None if noise_covariance is None else _matrix(noise_covariance, 'noise_covariance')
    if covariance is not None and covariance.shape != (len(gain), len(gain)):
        raise ValueError('noise_covariance must match G sensor dimensions')
    projector_array = None if projector is None else _matrix(projector, 'projector')
    identity = dict(leadfield_sha256=_hash(gain), noise_covariance_sha256=_hash(covariance),
                    input_projector_sha256=_hash(projector_array), average_reference=bool(average_reference),
                    numerical_rtol=float(numerical_rtol), near_null_rtol=float(near_null_rtol),
                    algorithm_version=2)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode('ascii')).hexdigest()
    hit = cache is not None and key in cache
    if hit:
        decomposition = cache.pop(key)
        cache[key] = decomposition
    else:
        decomposition = _decompose(gain, covariance, average_reference, projector_array,
                                   numerical_rtol, near_null_rtol)
        if cache is not None:
            cache[key] = decomposition
    if cache is not None:
        while len(cache) > 2:
            del cache[next(iter(cache))]
    total = float(np.einsum('ij,ij->', values, values))
    if not np.isfinite(total):
        raise ValueError('Source energy exceeds finite numerical range')
    result = dict(identity, **decomposition['info'], decomposition_key=key, cache_hit=bool(hit),
                  source_sha256=_hash(values), n_sensors=len(gain), n_sources=gain.shape[1],
                  n_times=values.shape[1], source_energy=total,
                  noise_rank_rtol=float(numerical_rtol),
                  source_energy_metric='unweighted_per_vertex_Euclidean',
                  geometric_rank_basis='row_equilibrated_projected_forward',
                  white_rank_basis='noise_retained_whitened_projected_forward',
                  near_null_interpretation='relative_conditioning_not_SNR_threshold',
                  noise_resolvability_interpretation='isotropic_prior_continuous_Wiener_not_identifiability')
    observable_energy = float(_mode_energy(decomposition['geometric_right'], values).sum())
    observable = float(np.clip(observable_energy / total, 0., 1.)) if total else None
    result.update(geometric_observable_energy_fraction=observable,
                  geometric_null_energy_fraction=1. - observable if total else None,
                  white_stable_energy_fraction=None, white_near_null_energy_fraction=None,
                  white_null_energy_fraction=None, geometric_energy_lost_by_noise_projection_fraction=None,
                  isotropic_wiener_effective_dof=None, isotropic_wiener_weighted_energy_fraction=None,
                  isotropic_source_second_moment=None, whitened_target_to_baseline_power_db=None,
                  whitened_target_power_reference='total_baseline_nuisance_unit_variance')
    if result['noise_status'] == 'available':
        all_energies = _mode_energy(decomposition['all_white_right'], values)
        energies = all_energies[:result['whitened_rank']]
        all_singular = decomposition['all_white_singular']
        positive = (all_singular > 0) & (all_energies > 0)
        if np.any(positive):
            log_power = np.logaddexp.reduce(2 * np.log(all_singular[positive]) + np.log(all_energies[positive]))
            log_mean_power = log_power - np.log(result['noise_rank']) - np.log(values.shape[1])
            result['whitened_target_to_baseline_power_db'] = float(10 / np.log(10) * log_mean_power)
        stable_rank = result['white_stable_rank']
        white_observable = float(np.clip(energies.sum() / total, 0., 1.)) if total else None
        stable = float(np.clip(energies[:stable_rank].sum() / total, 0., white_observable)) if total else None
        variance = total / values.size
        # Logistic evaluation avoids squaring very large or small singular values.
        log_snr = 2 * np.log(decomposition['singular']) + np.log(variance) if variance else np.full(len(energies), -np.inf)
        weights = np.exp(-np.logaddexp(0., -log_snr))
        result.update(white_stable_energy_fraction=stable,
                      white_near_null_energy_fraction=white_observable - stable if total else None,
                      white_null_energy_fraction=1. - white_observable if total else None,
                      isotropic_source_second_moment=variance,
                      isotropic_wiener_effective_dof=float(weights.sum()),
                      isotropic_wiener_weighted_energy_fraction=float(np.clip(np.dot(weights, energies) / total, 0., 1.)) if total else None)
        if total and result['noise_projection_rowspaces_nested']:
            result['geometric_energy_lost_by_noise_projection_fraction'] = max(0., observable - white_observable)
    return result
