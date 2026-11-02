"""Noise-whitened linear inverse diagnostics without dense source projectors.

The effective cutoff describes relative conditioning, not an SNR threshold.
Nullspace and energy statements refer to the supplied source coordinates;
rescaling source columns changes that geometry.
"""

from __future__ import annotations

import numpy as np


def _rtol(value, name):
    value = float(value)
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError(f"{name} must be finite and between zero and one")
    return value


def _matrix(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite two-dimensional array")
    return value


def projector_from_patterns(patterns, rtol=1e-12):
    """Return the sensor projector removing columns of (channels, patterns)."""
    values = _matrix(patterns, "patterns")
    tolerance = _rtol(rtol, "rtol")
    if values.shape[0] == 0:
        raise ValueError("patterns must have at least one sensor row")
    projector = np.eye(values.shape[0])
    if values.shape[1] == 0:
        return projector
    left, singular, _ = np.linalg.svd(values, full_matrices=False)
    retained = left[:, singular > tolerance * singular[0]]
    return projector - retained @ retained.T


def unit_safe_whitener(C, rtol=1e-10):
    """Whiten covariance after diagonal standardization, retaining its rank.

    Standardizing before eigendecomposition prevents magnetometer/gradiometer
    units from deciding rank. Exactly zero-variance sensors are excluded;
    their covariance rows must also be zero. Small negative eigenvalues from
    roundoff are tolerated in the dimensionless standardized matrix.
    """
    covariance = _matrix(C, "C")
    tolerance = _rtol(rtol, "rtol")
    if covariance.shape[0] == 0 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("C must be a nonempty square covariance")
    diagonal = np.diag(covariance)
    if np.any(diagonal < 0):
        raise ValueError("C has negative diagonal variance")
    positive = diagonal > 0
    if not np.any(positive):
        raise ValueError("C has no positive variance")
    if np.any(covariance[~positive, :] != 0) or np.any(covariance[:, ~positive] != 0):
        raise ValueError("Zero-variance sensors must have zero covariance rows and columns")
    scaling = np.sqrt(diagonal[positive])
    standardized = covariance[np.ix_(positive, positive)] / scaling[:, None] / scaling[None, :]
    if not np.all(np.isfinite(standardized)):
        raise ValueError("C cannot be standardized to finite values")
    if not np.allclose(standardized, standardized.T, rtol=0, atol=10 * tolerance):
        raise ValueError("C must be symmetric after unit standardization")
    eigenvalues, eigenvectors = np.linalg.eigh((standardized + standardized.T) / 2)
    scale = float(eigenvalues[-1])
    if eigenvalues[0] < -10 * tolerance * scale:
        raise ValueError("C is not positive semidefinite after unit standardization")
    retained = eigenvalues > tolerance * scale
    whitener = np.zeros((int(np.sum(retained)), covariance.shape[0]))
    whitener[:, positive] = (
        eigenvectors[:, retained].T / np.sqrt(eigenvalues[retained])[:, None]
        / scaling[None, :])
    return whitener, {
        "n_channels": int(covariance.shape[0]), "rank": int(np.sum(retained)),
        "n_zero_variance_channels": int(np.sum(~positive)),
        "rtol": tolerance, "standardized_eigenvalues": eigenvalues.tolist(),
        "standardized_cutoff": tolerance * scale,
        "rank_basis": "diagonally standardized covariance",
    }


def analyze_operator(G, C, *, source=None, numerical_rtol=1e-10,
                     effective_rtol=1e-3):
    """Analyze the observable, near-null, and numerical-null source spaces.

    G is channels x source DOF. Optional source is DOF or DOF x time.
    ``observable_source`` includes the near-null component; equivalently,
    source = stable_source + near_null_source + null_source. Numerical null
    components need not map to exactly zero when a nonzero cutoff is used.
    """
    leadfield = _matrix(G, "G")
    if min(leadfield.shape) == 0:
        raise ValueError("G must have at least one sensor and source")
    numerical_rtol = _rtol(numerical_rtol, "numerical_rtol")
    effective_rtol = _rtol(effective_rtol, "effective_rtol")
    if effective_rtol < numerical_rtol:
        raise ValueError("effective_rtol must be at least numerical_rtol")
    whitener, whitening_info = unit_safe_whitener(C, numerical_rtol)
    if whitener.shape[1] != leadfield.shape[0]:
        raise ValueError("G and C sensor dimensions differ")
    operator = whitener @ leadfield
    left, singular, right = np.linalg.svd(operator, full_matrices=False)
    largest = float(singular[0])
    rank = int(np.sum(singular > numerical_rtol * largest))
    effective_rank = int(np.sum(singular > effective_rtol * largest))
    n_sources = int(leadfield.shape[1])
    nonzero_condition = float(largest / singular[rank - 1]) if rank else None
    info = {
        "n_channels": int(leadfield.shape[0]), "n_sources": n_sources,
        "rank": rank, "nullity": n_sources - rank,
        "rank_nullity": {"rank": rank, "nullity": n_sources - rank},
        "effective_rank": effective_rank, "near_nullity": rank - effective_rank,
        "full_column_rank": rank == n_sources,
        "condition_number": nonzero_condition if rank == n_sources else None,
        "nonzero_condition": nonzero_condition,
        "numerical_rtol": numerical_rtol, "effective_rtol": effective_rtol,
        "effective_cutoff_interpretation": "relative conditioning cutoff, not SNR threshold",
        "whitening": whitening_info,
    }
    result = {"whitener": whitener, "operator": operator,
              "singular_values": singular, "right_vectors": right,
              "left_vectors": left, "info": info}
    if source is not None:
        truth = np.asarray(source, dtype=np.float64)
        if truth.ndim not in (1, 2) or truth.shape[0] != n_sources or not np.all(np.isfinite(truth)):
            raise ValueError("source must be finite, with shape DOF or DOF x time")
        observable = right[:rank].T @ (right[:rank] @ truth)
        near = right[effective_rank:rank].T @ (right[effective_rank:rank] @ truth)
        null = truth - observable
        stable = observable - near
        result.update(observable_source=observable, null_source=null,
                      near_null_source=near, stable_source=stable)
        energy = float(np.sum(truth ** 2))
        info["source_energy_fractions"] = {
            name: float(np.sum(value ** 2) / energy) if energy else None
            for name, value in (("observable", observable), ("null", null),
                                ("near_null", near), ("stable", stable))}
        residual = float(np.linalg.norm(operator @ null))
        signal_norm = float(np.linalg.norm(operator @ truth))
        info["null_sensor_residual_norm"] = residual
        info["null_sensor_relative_residual"] = residual / signal_norm if signal_norm else None
    return result


def solve_svd(analysis, data, *, method="pinv", ridge_relative=.01):
    """Solve from raw sensor data using the analysis whitening/SVD factors.

    Ridge penalizes the Euclidean norm in the supplied source coordinates.
    A unique ridge solution does not identify any true nullspace component.
    """
    if method not in {"pinv", "tsvd", "ridge"}:
        raise ValueError("method must be 'pinv', 'tsvd', or 'ridge'")
    values = np.asarray(data, dtype=np.float64)
    whitener = analysis["whitener"]
    if values.ndim not in (1, 2) or values.shape[0] != whitener.shape[1] or not np.all(np.isfinite(values)):
        raise ValueError("data must be finite, with shape channels or channels x time")
    singular = analysis["singular_values"]
    gain = np.zeros_like(singular)
    if method == "ridge":
        penalty = float(ridge_relative)
        if not np.isfinite(penalty) or penalty <= 0:
            raise ValueError("ridge_relative must be finite and positive")
        if singular[0] > 0:
            relative = singular / singular[0]
            gain = relative / (relative ** 2 + penalty ** 2) / singular[0]
    else:
        rank = analysis["info"]["rank" if method == "pinv" else "effective_rank"]
        gain[:rank] = 1 / singular[:rank]
    coordinates = analysis["left_vectors"].T @ (whitener @ values)
    weighted = gain * coordinates if values.ndim == 1 else gain[:, None] * coordinates
    return analysis["right_vectors"].T @ weighted
