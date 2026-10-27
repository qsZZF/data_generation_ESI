"""Synthetic parameter recovery diagnostics; no fitting or held-out tuning."""
import numpy as np


def _metrics(prediction, target):
    error = prediction - target
    mae = float(np.mean(np.abs(error)))
    total = float(np.sum((target - target.mean()) ** 2))
    r2 = float(1 - np.sum(error ** 2) / total) if total > 0 else None
    if not np.isfinite(mae) or (r2 is not None and not np.isfinite(r2)):
        raise ValueError("Recovery metric overflow; check parameter magnitudes")
    return {"mae": mae, "r2": r2}


def _scores(prediction, target, spatial):
    scores = {"pooled": _metrics(prediction, target)}
    if spatial:
        prediction_mean = prediction.mean(axis=1, keepdims=True)
        target_mean = target.mean(axis=1, keepdims=True)
        scores["whole_brain_mean"] = _metrics(prediction_mean, target_mean)
        scores["within_draw_regional_residual"] = _metrics(
            prediction - prediction_mean, target - target_mean)
        scores["per_region"] = [_metrics(prediction[:, region], target[:, region])
                                for region in range(target.shape[1])]
    return scores


def _improvement(prediction, prior):
    reduction = prior["mae"] - prediction["mae"]
    fraction = reduction / prior["mae"] if prior["mae"] > 0 else None
    if fraction is not None and not np.isfinite(fraction):
        raise ValueError("Recovery improvement overflow; check parameter magnitudes")
    return {
        "mae_reduction": reduction,
        "mae_reduction_fraction": fraction,
        "r2_gain": (prediction["r2"] - prior["r2"]
                    if prediction["r2"] is not None and prior["r2"] is not None else None),
    }


def recovery_detail(prediction, target, prior):
    """Compare matched held-out predictions and a previously fitted prior baseline.

    Inputs are matching dictionaries of physical parameter arrays. Regional
    families have shape (draws, regions); ``delay_ms`` is (draws, 1) or (draws,).
    All families share the same draw and regional axes. Prior arrays must already
    match target shapes; this function never estimates a baseline from test data.

    Each returned family contains ``prediction``, ``prior`` and ``improvement``.
    Regional scores include pooled values, whole-brain means across draws, pooled
    within-draw spatial residuals, and a list of across-draw scores in region order.
    Residuals subtract each draw's OWN regional mean, separately for predictions,
    truth and prior; no population regional mean is subtracted. Scalar delay has
    only pooled scores. MAE uses physical units, R2 is 1-SSE/SST, and improvements
    are positive when prediction beats prior. Undefined R2 or fractional MAE
    reduction is None (JSON null); no NaN/Infinity is returned.
    """
    if not target or set(prediction) != set(target) or set(prior) != set(target):
        raise ValueError("prediction, target and prior must have identical nonempty family keys")
    result = {}
    n_draws = n_regions = None
    for name in target:
        arrays = [np.asarray(values[name], dtype=np.float64)
                  for values in (prediction, target, prior)]
        spatial = name != "delay_ms"
        if not spatial:
            arrays = [value[:, None] if value.ndim == 1 else value for value in arrays]
        if any(value.ndim != 2 or 0 in value.shape or not np.isfinite(value).all()
               for value in arrays):
            raise ValueError(f"{name}: expected nonempty finite (draws, regions) arrays")
        p, t, baseline = arrays
        if p.shape != t.shape or baseline.shape != t.shape:
            raise ValueError(f"{name}: prediction, target and prior shapes differ")
        if not spatial and t.shape[1] != 1:
            raise ValueError("delay_ms must have one scalar per draw")
        if n_draws is not None and t.shape[0] != n_draws:
            raise ValueError("Parameter families have inconsistent draw counts")
        n_draws = t.shape[0]
        if spatial:
            if n_regions is not None and t.shape[1] != n_regions:
                raise ValueError("Regional parameter families have inconsistent region counts")
            n_regions = t.shape[1]
        estimated = _scores(p, t, spatial)
        reference = _scores(baseline, t, spatial)
        improvements = {
            key: ([_improvement(a, b) for a, b in zip(value, reference[key])]
                  if key == "per_region" else _improvement(value, reference[key]))
            for key, value in estimated.items()
        }
        result[name] = {"prediction": estimated, "prior": reference,
                        "improvement": improvements}
    return result
