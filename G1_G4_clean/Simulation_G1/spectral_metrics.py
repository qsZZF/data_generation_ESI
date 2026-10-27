"""Descriptive EEG checks, not tests of physiological validity.

Inputs retain their physical amplitude units. Power ratios and spectral slopes
are unit independent. The aperiodic fit is a robust straight log-log slope,
not a periodic/aperiodic decomposition such as specparam.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt, welch
from scipy.stats import theilslopes


_BANDS = {"delta": (1., 4.), "theta": (4., 8.),
          "alpha": (8., 13.), "beta": (13., 30.)}


def _validate(data, sampling_rate_hz, ndim):
    values = np.asarray(data, dtype=np.float64)
    rate = float(sampling_rate_hz)
    if not np.isfinite(rate) or rate < 80:
        raise ValueError("sampling_rate_hz must be finite and at least 80 Hz")
    if values.ndim != ndim or any(size == 0 for size in values.shape):
        raise ValueError(f"Expected nonempty {ndim}-dimensional data")
    if not np.all(np.isfinite(values)):
        raise ValueError("Data must contain only finite values")
    if not np.any(np.var(values, axis=-1) > 0):
        raise ValueError("Data are temporally flat")
    return values, rate


def _band_power(frequency, power, low, high):
    """Integrate with interpolated edges so adjacent bands share no area."""
    inside = (frequency > low) & (frequency < high)
    grid = np.r_[low, frequency[inside], high]
    density = np.interp(grid, frequency, power)
    # Explicit trapezoids also work with NumPy versions before trapezoid().
    return float(np.sum(np.diff(grid) * (density[1:] + density[:-1]) / 2))


def resting_features(eeg, sampling_rate_hz):
    """Summarize channels x time EEG with at least four seconds of data.

    Welch PSDs use four-second Hann windows and 50% overlap. The mean channel
    PSD determines spectral features; autocorrelation/Hjorth values average
    nonflat channels. Band fractions use total 1--40 Hz power as denominator.
    """
    values, rate = _validate(eeg, sampling_rate_hz, 2)
    if values.shape[-1] < int(np.ceil(4 * rate)):
        raise ValueError("resting_features requires at least four seconds")
    centered = values - values.mean(axis=-1, keepdims=True)
    usable = np.var(centered, axis=-1) > 0
    centered = centered[usable]
    frequency, density = welch(centered, fs=rate, nperseg=int(round(4 * rate)),
                               axis=-1, detrend="constant")
    power = density.mean(axis=0)
    total = _band_power(frequency, power, 1, 40)
    if not np.isfinite(total) or total <= 0:
        raise ValueError("No finite positive 1--40 Hz power")

    # Equal log-frequency spacing avoids overweighting the high-frequency bins.
    fit_grid = np.geomspace(2, 35, 64)
    fit_grid = fit_grid[(fit_grid < 7) | (fit_grid > 14)]
    log_power = np.log10(np.maximum(power, np.finfo(float).tiny))
    fit_y = np.interp(np.log10(fit_grid), np.log10(frequency[1:]), log_power[1:])
    slope, intercept, _, _ = theilslopes(fit_y, np.log10(fit_grid))
    alpha_mask = (frequency >= 8) & (frequency <= 13)
    residual_db = 10 * (log_power[alpha_mask] -
                        (intercept + slope * np.log10(frequency[alpha_mask])))
    peak = int(np.argmax(residual_db))
    lags_ms = [25., 50., 100., 250.]
    autocorrelation = []
    for lag_ms in lags_ms:
        lag = max(1, int(round(rate * lag_ms / 1000)))
        # The full-record denominator keeps finite-sample estimates bounded.
        numerator = np.sum(centered[:, :-lag] * centered[:, lag:], axis=-1)
        denominator = np.sum(centered * centered, axis=-1)
        autocorrelation.append(float(np.mean(numerator / denominator)))

    variance = np.var(centered, axis=-1)
    first = np.diff(centered, axis=-1) * rate
    second = np.diff(first, axis=-1) * rate
    first_variance = np.var(first, axis=-1)
    mobility = np.sqrt(first_variance / variance)
    valid_derivative = first_variance > 0
    complexity = np.sqrt(np.var(second, axis=-1)[valid_derivative] /
                         first_variance[valid_derivative]) / mobility[valid_derivative]
    band_power = {name: _band_power(frequency, power, *band) / total
                  for name, band in _BANDS.items()}
    return {
        "aperiodic_exponent": float(-slope),
        "alpha_peak_hz": float(frequency[alpha_mask][peak]),
        "alpha_peak_prominence_db": float(residual_db[peak]),
        "alpha_relative_power": float(band_power["alpha"]),
        "band_relative_power": band_power,
        "autocorrelation": {"lag_ms": lags_ms, "values": autocorrelation},
        "hjorth_mobility_hz": float(np.mean(mobility) / (2 * np.pi)),
        "hjorth_complexity": float(np.mean(complexity)) if complexity.size else None,
        "metadata": {
            "sampling_rate_hz": rate, "duration_s": values.shape[-1] / rate,
            "n_channels": int(values.shape[0]),
            "n_flat_channels_excluded": int(np.sum(~usable)),
            "psd_method": "Welch, 4 s Hann, 50% overlap, mean channel PSD",
            "aperiodic_method": "Theil-Sen log-log slope, 2--35 Hz excluding 7--14 Hz",
            "band_power_denominator_hz": [1., 40.],
            "alpha_peak_is_candidate": True,
            "hjorth_derivative": "first differences; mobility divided by 2*pi",
        },
    }


def _window_mask(time, rate, window, name, *, baseline=False):
    if len(window) != 2 or not np.all(np.isfinite(window)):
        raise ValueError(f"{name} must contain two finite bounds")
    start, stop = map(float, window)
    tolerance = 1e-8 / rate
    if stop <= start or start < time[0] - tolerance or stop > time[-1] + 1 / rate + tolerance:
        raise ValueError(f"{name} must lie inside the epoch with start < stop")
    if baseline and stop > 0:
        raise ValueError("baseline must end at or before event onset")
    if not baseline and start < 0:
        raise ValueError(f"{name} must begin at or after event onset")
    mask = (time >= start - tolerance) & (time < stop - tolerance)
    if np.count_nonzero(mask) < 2:
        raise ValueError(f"{name} must contain at least two samples")
    return mask


def _ersp(values, rate, band, baseline_mask, response_mask):
    sos = butter(4, band, btype="bandpass", fs=rate, output="sos")
    try:
        filtered = sosfiltfilt(sos, values, axis=-1)
    except ValueError as error:
        raise ValueError("Epoch is too short for zero-phase bandpass filtering") from error
    power = np.abs(hilbert(filtered, axis=-1)) ** 2
    baseline_power = float(np.mean(power[..., baseline_mask]))
    response_power = float(np.mean(power[..., response_mask]))
    if baseline_power <= 0 or response_power <= 0:
        return None
    return float(10 * np.log10(response_power / baseline_power))


def evoked_features(epochs, sampling_rate_hz, tmin, *, baseline=(-1.5, -.5),
                    response=(0., .5), ersp_window=(.5, 2.5)):
    """Summarize event-aligned trials x channels x time EEG.

    ERP uses per-trial/channel baseline subtraction. GFP is the spatial
    standard deviation, so a channel-common waveform has zero GFP. Evoked
    power fraction is a finite-trial descriptive ratio, not phase coherence.
    ERSP is the pooled trial/channel band-power ratio in dB. Induced ERSP
    subtracts the trial-average ERP before bandpass/Hilbert processing.
    """
    values, rate = _validate(epochs, sampling_rate_hz, 3)
    if not np.isfinite(tmin):
        raise ValueError("tmin must be finite")
    time = float(tmin) + np.arange(values.shape[-1]) / rate
    baseline_mask = _window_mask(time, rate, baseline, "baseline", baseline=True)
    response_mask = _window_mask(time, rate, response, "response")
    ersp_mask = _window_mask(time, rate, ersp_window, "ersp_window")
    centered = values - values[..., baseline_mask].mean(axis=-1, keepdims=True)
    evoked = centered.mean(axis=0)
    gfp = evoked.std(axis=0)
    response_gfp = gfp[response_mask]
    peak = int(np.argmax(response_gfp))
    evoked_power = float(np.mean(evoked[:, response_mask] ** 2))
    trial_power = float(np.mean(centered[..., response_mask] ** 2))
    induced = centered - evoked[None, ...]
    # One trial has no estimable induced component; identical trials do not
    # justify reporting a ratio of floating-point cancellation residuals.
    induced_valid = values.shape[0] > 1 and float(np.mean(induced ** 2)) > (
        np.finfo(float).eps * float(np.mean(centered ** 2)))
    result = {
        "erp_peak_latency_s": float(time[response_mask][peak]) if response_gfp[peak] > 0 else None,
        "erp_peak_gfp": float(response_gfp[peak]),
        "erp_response_rms": float(np.sqrt(evoked_power)),
        "evoked_power_fraction": evoked_power / trial_power if trial_power > 0 else None,
        "metadata": {
            "sampling_rate_hz": rate, "n_trials": int(values.shape[0]),
            "n_channels": int(values.shape[1]), "tmin_s": float(tmin),
            "baseline_s": list(map(float, baseline)),
            "response_s": list(map(float, response)),
            "ersp_window_s": list(map(float, ersp_window)),
            "window_convention": "start inclusive, stop exclusive",
            "ersp_method": "4th-order zero-phase Butterworth + Hilbert; pooled power ratio",
            "induced_erp_subtracted": True,
            "induced_estimable": bool(induced_valid),
            "amplitude_units": "same as input",
        },
    }
    for name in ("alpha", "beta"):
        result[f"{name}_ersp_db"] = _ersp(centered, rate, _BANDS[name], baseline_mask, ersp_mask)
        result[f"{name}_induced_ersp_db"] = (
            _ersp(induced, rate, _BANDS[name], baseline_mask, ersp_mask)
            if induced_valid else None)
    return result
