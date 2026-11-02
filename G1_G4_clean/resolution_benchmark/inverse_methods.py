"""Fixed-normal inverse estimators for already noise-whitened data.

Fit uses training data only; apply the returned operator to independent data.
Noise covariance must be identity in the supplied sensor coordinates. sLORETA
returns standardized maps, whose amplitudes must not be scored as currents.

The SBL update is the convex-bound Champagne update in Table 1 of
https://pmc.ncbi.nlm.nih.gov/articles/PMC8433122/ . Its covariance model assumes
zero-mean sources with diagonal, time-stationary variance and known unit noise.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def _inputs(A, Y):
    A, Y = np.asarray(A, dtype=float), np.asarray(Y, dtype=float)
    if (A.ndim != 2 or Y.ndim != 2 or min(A.shape) == 0
            or A.shape[0] != Y.shape[0] or Y.shape[1] < 2
            or not np.all(np.isfinite(A)) or not np.all(np.isfinite(Y))):
        raise ValueError("A and Y must be finite matrices with matching sensor rows and >=2 samples")
    norms = np.linalg.norm(A, axis=0)
    if not np.all(np.isfinite(norms)) or not np.any(norms > 0):
        raise ValueError("A must have finite column norms and at least one nonzero column")
    return A, Y, norms


def _positive(value, name, allow_zero=False):
    value = float(value)
    if not np.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return value


def _unit_gain(A, covariance):
    solved = cho_solve(cho_factor(covariance, lower=True), A)
    gain = np.sum(A * solved, axis=0)
    operator = np.zeros_like(A.T)
    positive = gain > 0
    operator[positive] = solved[:, positive].T / gain[positive, None]
    return operator


def _sbl(A, Y, norms, max_iter, tol):
    scale = np.where(norms > 0, norms, 1.)
    B = A / scale
    m, n = A.shape
    second_moment = Y @ Y.T / Y.shape[1]
    gamma = np.full(n, max((np.trace(second_moment) - m) / n, 1e-6))
    gamma[norms == 0] = 0

    def state(variance):
        covariance = (B * variance) @ B.T + np.eye(m)
        factor = cho_factor(covariance, lower=True)
        inverse_B = cho_solve(factor, B)
        objective = (2 * np.log(np.diag(factor[0])).sum()
                     + np.trace(cho_solve(factor, second_moment)))
        return covariance, inverse_B, float(objective)

    covariance, inverse_B, objective = state(gamma)
    history = [objective]
    converged = False
    stopping_reason = "max_iter"
    for _ in range(max_iter):
        sensitivity = np.sum(B * inverse_B, axis=0)
        power = np.maximum(np.sum(inverse_B * (second_moment @ inverse_B), axis=0), 0)
        ratio = np.divide(power, sensitivity, out=np.zeros(n), where=sensitivity > 0)
        candidate = gamma * np.sqrt(ratio)
        next_cov, next_inverse_B, next_objective = state(candidate)
        # An MM step cannot increase this objective; retain the last valid fit
        # if floating-point roundoff prevents a descent step.
        if next_objective > objective:
            stopping_reason = "numerical_non_descent"
            converged = next_objective - objective <= 1e-10 * max(1., abs(objective))
            break
        improvement = objective - next_objective
        gamma = candidate
        covariance, inverse_B, objective = next_cov, next_inverse_B, next_objective
        history.append(objective)
        if improvement <= tol * max(1., abs(history[-2])):
            converged = True
            stopping_reason = "relative_objective_tolerance"
            break
    return {
        "operator": gamma[:, None] * inverse_B.T / scale[:, None],
        "gamma": gamma / scale ** 2,
        "gamma_normalized": gamma,
        "model_cov": covariance,
        "evidence_history": history,
        "diagnostics": {
            "converged": converged, "iterations": len(history) - 1,
            "stopping_reason": stopping_reason,
            "objective": "logdet(Sigma) + trace(Sigma^-1 Y Y.T / T)",
            "noise_variance": 1., "column_normalized": True,
            "pruning": False, "training_covariance": "uncentered_second_moment",
        },
    }


def sbl_active_operator(fit, retained_power=.99):
    """Gate model-covariance LCMV by training-only SBL variance contributions.

    Keep the smallest descending prefix of ``gamma_normalized`` accounting for
    the requested fraction of their sum. Columns were unit-norm during SBL,
    so these variances measure each atom's modeled sensor power contribution.
    This is an explicit support prior, not ordinary model-covariance LCMV.

    Without this gating, the posterior SBL row satisfies
    K_sbl[i] = gamma[i] * (a_i.T Sigma^-1 a_i) * K_model_lcmv[i].
    Unit-gain model LCMV alone thus omits SBL's learned shrinkage factor.
    """
    retained_power = _positive(retained_power, "retained_power")
    if retained_power > 1:
        raise ValueError("retained_power must be <=1")
    gamma = np.asarray(fit["gamma_normalized"], dtype=float)
    operator = np.asarray(fit["model_lcmv_operator"], dtype=float)
    if (gamma.ndim != 1 or operator.ndim != 2 or operator.shape[0] != gamma.size
            or not np.all(np.isfinite(gamma)) or np.any(gamma < 0)
            or not np.all(np.isfinite(operator))):
        raise ValueError("fit must contain finite nonnegative variances and matching operator rows")
    gated = np.zeros_like(operator)
    if gamma.size == 0 or not np.any(gamma > 0):
        return gated, 0
    order = np.argsort(-gamma, kind="stable")
    relative = gamma[order] / gamma[order[0]]
    cumulative = np.cumsum(relative)
    active_count = int(np.searchsorted(cumulative, retained_power * cumulative[-1], side="left") + 1)
    active = order[:active_count]
    gated[active] = operator[active]
    return gated, active_count


def fit_inverse(A, Y, method, *, ridge_relative=.01, depth_exponent=.8,
                max_depth_weight=100., covariance_loading=.05,
                max_iter=150, tol=1e-5):
    """Return a fitted source-by-sensor operator and reproducibility metadata.

    ``mne`` and ``wmne`` use R A.T (A R A.T + lambda**2 I)^-1,
    lambda = ridge_relative * smax(A sqrt(R)). For wMNE, variance weights
    R_ii = min((max_column_norm / norm_i)**(2*depth_exponent), cap).
    Zero leadfield columns receive zero output in every method.

    ``sloreta`` divides plain MNE rows by sqrt(diag(K_MNE A)); no depth
    weighting or global statistical scaling is applied. ``lcmv`` is unit-gain
    with centered training covariance and loading * trace(C)/m. ``sbl_lcmv``
    uses the learned SBL model covariance (including unit noise) with the same
    unit-gain formula. These are deliberately explicit benchmark variants,
    not claims of identical defaults to a particular software package.
    """
    A, Y, norms = _inputs(A, Y)
    method = str(method).lower()
    allowed = {"mne", "wmne", "sloreta", "lcmv", "sbl", "sbl_lcmv"}
    if method not in allowed:
        raise ValueError(f"method must be one of {sorted(allowed)}")
    ridge_relative = _positive(ridge_relative, "ridge_relative")
    covariance_loading = _positive(covariance_loading, "covariance_loading")
    depth_exponent = _positive(depth_exponent, "depth_exponent", allow_zero=True)
    max_depth_weight = _positive(max_depth_weight, "max_depth_weight")
    if max_depth_weight < 1:
        raise ValueError("max_depth_weight must be >=1")
    tol = _positive(tol, "tol")
    if isinstance(max_iter, bool) or int(max_iter) != max_iter or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    diagnostics = {"method": method, "training_samples": int(Y.shape[1]),
                   "physical_currents": method != "sloreta",
                   "output_kind": "standardized_map" if method == "sloreta" else "source_coefficients",
                   "zero_leadfield_columns": int(np.sum(norms == 0))}
    if method in {"sbl", "sbl_lcmv"}:
        result = _sbl(A, Y, norms, int(max_iter), tol)
        result["diagnostics"].update(diagnostics)
        result["model_lcmv_operator"] = _unit_gain(A, result["model_cov"])
        result["diagnostics"]["model_lcmv_additional_loading"] = 0.
        if method == "sbl_lcmv":
            result["operator"] = result["model_lcmv_operator"]
        return result
    if method == "lcmv":
        centered = Y - Y.mean(axis=1, keepdims=True)
        covariance = centered @ centered.T / (Y.shape[1] - 1)
        loading = covariance_loading * np.trace(covariance) / A.shape[0]
        if loading <= 0:
            raise ValueError("LCMV requires nonzero training sample covariance")
        covariance = covariance + loading * np.eye(A.shape[0])
        operator = _unit_gain(A, covariance)
        diagnostics.update(covariance_loading=covariance_loading, loading_absolute=float(loading),
                           training_covariance="centered_unbiased")
        return {"operator": operator, "diagnostics": diagnostics, "model_cov": covariance}
    weights = np.ones(A.shape[1])
    if method == "wmne":
        positive = norms > 0
        log_weight = 2 * depth_exponent * (np.log(norms.max()) - np.log(norms[positive]))
        weights[positive] = np.exp(np.minimum(log_weight, np.log(max_depth_weight)))
        weights[~positive] = 0
    weighted = A * np.sqrt(weights)
    penalty = ridge_relative * np.linalg.svd(weighted, compute_uv=False)[0]
    covariance = weighted @ weighted.T + penalty ** 2 * np.eye(A.shape[0])
    operator = weights[:, None] * cho_solve(cho_factor(covariance, lower=True), A).T
    result = {"operator": operator, "diagnostics": diagnostics, "source_weights": weights}
    if method == "sloreta":
        resolution_diagonal = np.maximum(np.sum(operator * A.T, axis=1), 0)
        normalization = np.sqrt(resolution_diagonal)
        result["operator"] = np.divide(operator, normalization[:, None],
                                       out=np.zeros_like(operator), where=normalization[:, None] > 0)
        result["normalization"] = normalization
    diagnostics.update(ridge_relative=ridge_relative, penalty_absolute=float(penalty),
                       depth_exponent=depth_exponent if method == "wmne" else 0.,
                       max_depth_weight=max_depth_weight if method == "wmne" else 1.)
    return result
