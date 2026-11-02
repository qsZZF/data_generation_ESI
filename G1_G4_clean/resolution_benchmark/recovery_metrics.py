"""Separate data-space recovery from prior-selected null-space estimates."""
import numpy as np


def score_recovery(estimate, truth, analysis, *, physical_units=True):
    estimate, truth = np.asarray(estimate), np.asarray(truth)
    if estimate.shape != truth.shape or not np.isfinite(estimate).all():
        raise ValueError('Estimate and truth must have matching finite arrays')
    total = float(np.sum(truth ** 2))
    if total <= 0:
        raise ValueError('Truth must have positive energy')
    vectors = analysis['right_vectors']
    rank = analysis['info']['rank']
    effective = analysis['info']['effective_rank']
    visible = vectors[:rank].T @ (vectors[:rank] @ truth)
    inferred_visible = vectors[:rank].T @ (vectors[:rank] @ estimate)
    hidden, inferred_hidden = truth - visible, estimate - inferred_visible
    hidden_energy = float(np.sum(hidden ** 2))
    inferred_energy = float(np.sum(inferred_hidden ** 2))
    norm_estimate = float(np.sum(estimate ** 2))
    alpha = float(np.sum(estimate * truth) / norm_estimate) if norm_estimate else 0.
    # This scalar uses test truth only as a shape diagnostic, never to fit a filter.
    shape_error = float(np.sqrt(np.sum((alpha * estimate - truth) ** 2) / total))
    power, estimated_power = np.mean(truth ** 2, axis=1), np.mean(estimate ** 2, axis=1)
    power_norm = np.linalg.norm(power) * np.linalg.norm(estimated_power)
    result = dict(
        shape_nrmse=shape_error,
        power_cosine=float(power @ estimated_power / power_norm) if power_norm else 0.,
        truth_null_fraction=hidden_energy / total,
        inferred_null_fraction=(inferred_energy / norm_estimate if norm_estimate else 0.) if physical_units else None,
        null_cosine=(float(np.sum(hidden * inferred_hidden) / np.sqrt(hidden_energy * inferred_energy))
                     if physical_units and hidden_energy > 1e-10 * total and inferred_energy > 1e-20 * max(norm_estimate, 1e-300) else None),
        source_nrmse=None, observable_error_fraction=None, near_null_error_fraction=None,
        null_error_fraction=None, null_recovery=None)
    if physical_units:
        error = estimate - truth
        near_error = vectors[effective:rank] @ error
        null_error = float(np.sum((inferred_hidden - hidden) ** 2))
        result.update(source_nrmse=float(np.sqrt(np.sum(error ** 2) / total)),
                      observable_error_fraction=float(np.sum((inferred_visible - visible) ** 2) / total),
                      near_null_error_fraction=float(np.sum(near_error ** 2) / total),
                      null_error_fraction=null_error / total,
                      null_recovery=1 - null_error / hidden_energy if hidden_energy > 1e-10 * total else None)
    return result


def support_diagnostics(operator, basis, truth):
    """Separate model approximation from identifiability inside its span."""
    u, s, _ = np.linalg.svd(basis, full_matrices=False)
    q = u[:, s > 1e-10 * s[0]]
    projected = q @ (q.T @ truth)
    singular = np.linalg.svd(operator @ q, compute_uv=False)
    rank = int(np.sum(singular > 1e-10 * singular[0]))
    return dict(support_dimension=q.shape[1], support_rank=rank,
                support_nullity=q.shape[1] - rank,
                support_condition=float(singular[0] / singular[-1]) if rank == q.shape[1] else None,
                support_approximation_nrmse=float(np.linalg.norm(projected - truth) / np.linalg.norm(truth)))
