"""Cortical dual-timescale Jansen--Rit generator for data-guided inversion.

Equations and priors follow NeuroDyn-EEG, arXiv:2609.36773, Eqs. 2.1--2.5
and Tables S1--S4. This is a custom JR integrator, not TVB's single-branch JR.
Parameters use physical milliseconds, mV, s^-1, and dimensionless C factors.

Deliberate corrections/adaptations to the released implementation:
* Table S1 sigmoid/C3 bounds and shared Gaussian 220 +/- 22 s^-1 drive;
* physical 0--10 ms sliding delay, fractional interpolation, stable sigmoid;
* source-indexed cumulative firing statistics via Welford (linear time);
* zero initial states/history (no artificial 5 mV initial observation);
* fine-step Heun, with drive held on an independent fixed 256 Hz grid.
The latter is a numerical convention, not a claim of bitwise reproduction.
History variance below 1e-12 (s^-1)^2 uses a 1e-6 s^-1 denominator floor.
"""

from __future__ import annotations

from fractions import Fraction
import math

import numpy as np
from numba import njit
from scipy.signal import resample_poly
from scipy.special import ndtr, ndtri


PARAMETER_BOUNDS = {
    "tau_e1_ms": (10.0, 35.0),
    "tau_i1_ms": (10.0, 35.0),
    "tau_e2_ms": (3.9, 8.4),
    "tau_i2_ms": (7.3, 16.8),
    "theta": (5.4, 6.6),
    "beta": (0.5, 0.62),
    "rmax": (2.5, 7.5),
    "C1": (0.5, 1.5),
    "C2": (0.4, 1.2),
    "C3": (0.125, 0.375),
    "C4": (0.125, 0.375),
}
PARAMETER_NAMES = tuple(PARAMETER_BOUNDS)
# Float32 inverse-network heads can cross a physical bound by a few ULPs.
# This is a roundoff allowance, not an extension of the simulation prior.
_BOUND_ROUNDOFF = 4 * np.finfo(np.float32).eps


def _truncated_normal(rng, low, high, mean, std, size):
    lower = ndtr((low - mean) / std)
    upper = ndtr((high - mean) / std)
    return np.clip(mean + std * ndtri(rng.uniform(lower, upper, size)), low, high)


def sample_parameters(rng: np.random.Generator, n_regions: int) -> dict:
    """One hierarchical whole-brain draw; eleven arrays plus scalar delay_ms."""
    if not isinstance(n_regions, (int, np.integer)) or n_regions < 1:
        raise ValueError("n_regions must be a positive integer")
    parameters = {}
    for name, (low, high) in PARAMETER_BOUNDS.items():
        if name.startswith("tau_i"):
            excitatory = parameters[name.replace("tau_i", "tau_e")]
            parameters[name] = rng.uniform(
                np.maximum(excitatory, low), np.minimum(2.4 * excitatory, high)
            )
            continue
        span = high - low
        mean = _truncated_normal(rng, low, high, (low + high) / 2, span / 4, None)
        fraction = 1 / 12 if name.startswith("tau_") else 1 / 6 if name.startswith("C") else 1 / 8
        parameters[name] = _truncated_normal(rng, low, high, mean, span * fraction, n_regions)
    parameters["delay_ms"] = float(rng.uniform(0, 10))
    return parameters


@njit(cache=True)
def _sigmoid(voltage, theta, beta, rmax):
    value = beta * (voltage - theta)
    if value >= 0:
        return rmax / (1 + math.exp(-value))
    exponential = math.exp(value)
    return rmax * exponential / (1 + exponential)


@njit(cache=True)
def _readout(state):
    return 0.5 * (state[:, 1] - state[:, 2] + state[:, 7] - state[:, 8])


@njit(cache=True)
def _rhs(state, inputs, parameters):
    derivative = np.empty_like(state)
    for region in range(state.shape[0]):
        p = parameters[:, region]
        x = state[region]
        voltage = 0.5 * (x[1] - x[2] + x[7] - x[8])
        common = 0.5 * (x[0] + x[6])
        pyramidal = _sigmoid(voltage, p[4], p[5], p[6])
        excitatory = 135 * p[8] * _sigmoid(135 * p[7] * common, p[4], p[5], p[6]) + inputs[region]
        inhibitory = 135 * p[10] * _sigmoid(135 * p[9] * common, p[4], p[5], p[6])
        for branch in range(2):
            offset = branch * 6
            tau_e, tau_i = p[2 * branch] / 1000, p[2 * branch + 1] / 1000
            gain_e = 3.25 if branch == 0 else 0.0325 / tau_e
            gain_i = 22.0 if branch == 0 else 0.44 / tau_i
            derivative[region, offset] = x[offset + 3]
            derivative[region, offset + 1] = x[offset + 4]
            derivative[region, offset + 2] = x[offset + 5]
            derivative[region, offset + 3] = (gain_e * pyramidal - 2 * x[offset + 3] - x[offset] / tau_e) / tau_e
            derivative[region, offset + 4] = (gain_e * excitatory - 2 * x[offset + 4] - x[offset + 1] / tau_e) / tau_e
            derivative[region, offset + 5] = (gain_i * inhibitory - 2 * x[offset + 5] - x[offset + 2] / tau_i) / tau_i
    return derivative


@njit(cache=True)
def _delayed_output(history, current_step, delay_steps, stage, predicted):
    """Interpolate PSP at step+stage-delay; pre-t=0 history is zero."""
    at = current_step + stage - delay_steps
    if at < 0:
        return np.zeros(history.shape[1])
    lower = int(math.floor(at))
    fraction = at - lower
    before = predicted if lower > current_step else history[lower % history.shape[0]]
    after = predicted if lower + 1 > current_step else history[(lower + 1) % history.shape[0]]
    return before * (1 - fraction) + after * fraction


@njit(cache=True)
def _afferent(delayed, mean, m2, count, parameters, targets, sources, weights, drive):
    n_regions = parameters.shape[1]
    centered = np.empty(n_regions)
    for region in range(n_regions):
        sigma = 1.0 if count < 2 else max(math.sqrt(max(m2[region], 0.0) / count), 1e-6)
        firing = _sigmoid(delayed[region], parameters[4, region], parameters[5, region], parameters[6, region])
        centered[region] = 22 * (firing - mean[region]) / sigma
    inputs = np.full(n_regions, drive)
    for edge in range(len(weights)):
        inputs[targets[edge]] += weights[edge] * centered[sources[edge]]
    return inputs


@njit(cache=True)
def _integrate(parameters, targets, sources, weights, delay_steps, dt, warmup_steps, saved_steps, drive, drive_stride):
    n_regions = parameters.shape[1]
    state = np.zeros((n_regions, 12))
    history = np.zeros((int(math.ceil(delay_steps)) + 3, n_regions))
    mean = np.empty(n_regions)
    m2 = np.zeros(n_regions)
    for region in range(n_regions):
        mean[region] = _sigmoid(0.0, parameters[4, region], parameters[5, region], parameters[6, region])
    output = np.empty((n_regions, saved_steps))
    for step in range(warmup_steps + saved_steps):
        if step >= warmup_steps:
            output[:, step - warmup_steps] = _readout(state)
        delayed = _delayed_output(history, step, delay_steps, 0, history[step % len(history)])
        inputs = _afferent(delayed, mean, m2, step + 1, parameters, targets, sources, weights, drive[step // drive_stride])
        first = _rhs(state, inputs, parameters)
        predicted = state + dt * first
        predicted_output = _readout(predicted)
        next_mean, next_m2 = mean.copy(), m2.copy()
        for region in range(n_regions):
            firing = _sigmoid(predicted_output[region], parameters[4, region], parameters[5, region], parameters[6, region])
            delta = firing - mean[region]
            next_mean[region] += delta / (step + 2)
            next_m2[region] += delta * (firing - next_mean[region])
        delayed = _delayed_output(history, step, delay_steps, 1, predicted_output)
        next_inputs = _afferent(delayed, next_mean, next_m2, step + 2, parameters, targets, sources, weights, drive[step // drive_stride])
        state += 0.5 * dt * (first + _rhs(predicted, next_inputs, parameters))
        readout = _readout(state)
        if not np.isfinite(state).all():
            raise FloatingPointError("Nonfinite JR state; inspect parameters and reduce dt_ms")
        history[(step + 1) % len(history)] = readout
        for region in range(n_regions):
            firing = _sigmoid(readout[region], parameters[4, region], parameters[5, region], parameters[6, region])
            delta = firing - mean[region]
            mean[region] += delta / (step + 2)
            m2[region] += delta * (firing - mean[region])
    return output


def simulate(parameters, connectivity, *, duration_s=2.0, warmup_s=1.0,
             sampling_rate_hz=256.0, dt_ms=0.5, seed=0,
             connectivity_orientation="target_source", connectivity_mode="raw",
             drive_sampling_hz=256.0):
    """Return raw source PSP (regions,time) in mV, relative time, and metadata.

    Connectivity must contain dimensionless weights in [0,2]. ``max`` mode
    explicitly scales the matrix to maximum one. No row normalization is applied.
    Diagonal self-connections are removed as required by Eq. 2.4. Sparse and dense
    connectivity are accepted; integration traverses only nonzero edges.
    ``dt_ms`` is a maximum step; the actual step divides the fixed drive period.
    Saved durations must correspond to an integer number of output samples.
    Parameter-bound excesses no larger than four float32 epsilons times the
    bound magnitude (minimum scale one) are repaired and listed in metadata.
    Larger out-of-prior estimates are rejected, including joint tau bounds.
    """
    for name, value in (("duration_s", duration_s), ("sampling_rate_hz", sampling_rate_hz),
                        ("dt_ms", dt_ms), ("drive_sampling_hz", drive_sampling_hz)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not np.isfinite(warmup_s) or warmup_s < 0:
        raise ValueError("warmup_s must be nonnegative and finite")
    if dt_ms > 0.5:
        raise ValueError("dt_ms must be <= 0.5 for the fast-branch kinetics")
    matrix = np.array(connectivity.toarray() if hasattr(connectivity, "toarray") else connectivity, dtype=float, copy=True)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
        raise ValueError("connectivity must be a nonempty square matrix")
    if not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise ValueError("connectivity must be finite and nonnegative")
    if connectivity_orientation == "source_target":
        matrix = matrix.T.copy()
    elif connectivity_orientation != "target_source":
        raise ValueError("connectivity_orientation must be target_source or source_target")
    np.fill_diagonal(matrix, 0)
    scale = 1.0
    if connectivity_mode == "max":
        scale = float(matrix.max()) or 1.0
        matrix /= scale
    elif connectivity_mode != "raw":
        raise ValueError("connectivity_mode must be raw or max")
    if np.any(matrix > 2):
        raise ValueError("paper coupling requires weights <= 2; use explicit connectivity_mode='max' if intended")
    n_regions = matrix.shape[0]
    values = []
    roundoff_repairs = []
    for name, (low, high) in PARAMETER_BOUNDS.items():
        value = np.asarray(parameters[name], dtype=float)
        low_tolerance = _BOUND_ROUNDOFF * max(1, abs(low))
        high_tolerance = _BOUND_ROUNDOFF * max(1, abs(high))
        if (value.shape != (n_regions,) or not np.isfinite(value).all()
                or np.any(value < low - low_tolerance) or np.any(value > high + high_tolerance)):
            raise ValueError(f"{name} must have shape ({n_regions},) and lie within [{low},{high}]")
        if np.any(value < low) or np.any(value > high):
            roundoff_repairs.append(name)
        values.append(np.clip(value, low, high))
    for branch in (1, 2):
        e, i = values[2 * (branch - 1):2 * branch]
        lower_tolerance = _BOUND_ROUNDOFF * np.maximum(1, e)
        upper_tolerance = _BOUND_ROUNDOFF * np.maximum(1, 2.4 * e)
        if np.any(i < e - lower_tolerance) or np.any(i > 2.4 * e + upper_tolerance):
            raise ValueError("inhibitory time constants must be between 1 and 2.4 times excitatory constants")
        if np.any(i < e) or np.any(i > 2.4 * e):
            roundoff_repairs.append(f"tau_i{branch}_ms_joint")
        values[2 * branch - 1] = np.clip(i, e, 2.4 * e)
    delay_ms = float(parameters["delay_ms"])
    if not np.isfinite(delay_ms) or not -_BOUND_ROUNDOFF <= delay_ms <= 10 + 10 * _BOUND_ROUNDOFF:
        raise ValueError("delay_ms must be in [0,10]")
    if not 0 <= delay_ms <= 10:
        roundoff_repairs.append("delay_ms")
        delay_ms = float(np.clip(delay_ms, 0, 10))
    requested_samples = duration_s * sampling_rate_hz
    n_samples = int(round(requested_samples))
    if n_samples < 1 or not np.isclose(requested_samples, n_samples, rtol=0, atol=1e-8):
        raise ValueError("duration_s * sampling_rate_hz must be a positive integer")
    stride = int(math.ceil(1000 / (drive_sampling_hz * dt_ms)))
    internal_rate = drive_sampling_hz * stride
    if sampling_rate_hz > internal_rate:
        raise ValueError("sampling_rate_hz cannot exceed the integration rate")
    warmup_steps = int(math.ceil(warmup_s * internal_rate - 1e-10))
    saved_steps = int(math.ceil(n_samples * internal_rate / sampling_rate_hz - 1e-10))
    total_steps = warmup_steps + saved_steps
    drive = np.random.default_rng(seed).normal(220, 22, int(math.ceil(total_steps / stride)))
    targets, sources = np.nonzero(matrix)
    weights = matrix[targets, sources]
    native = _integrate(np.ascontiguousarray(values), targets, sources,
                        np.sqrt(2 * weights - weights**2), delay_ms * internal_rate / 1000,
                        1 / internal_rate, warmup_steps, saved_steps, drive, stride)
    ratio = Fraction(float(sampling_rate_hz / internal_rate)).limit_denominator(1_000_000)
    if not np.isclose(float(ratio), sampling_rate_hz / internal_rate, rtol=1e-12):
        raise ValueError("sampling rates do not admit an accurate rational resampling ratio")
    # Edge padding avoids imposing zero voltage at either end of a nonzero PSP.
    source = resample_poly(native, ratio.numerator, ratio.denominator, axis=1, padtype="line")[:, :n_samples]
    if source.shape[1] != n_samples or not np.isfinite(source).all():
        raise FloatingPointError("Invalid resampled JR output")
    time = np.arange(n_samples, dtype=float) / sampling_rate_hz
    meta = {
        "model": "paper_dual_timescale_jr", "integrator": "Heun",
        "n_regions": n_regions, "source_unit": "mV_PSP", "time_unit": "s",
        "sampling_rate_hz": float(sampling_rate_hz), "duration_s": float(duration_s),
        "warmup_requested_s": float(warmup_s), "warmup_discarded_s": warmup_steps / internal_rate,
        "dt_ms_requested": float(dt_ms), "dt_ms_actual": 1000 / internal_rate,
        "delay_ms": delay_ms, "delay_interpolation": "linear_PSP_sliding_history",
        "drive": "shared_iid_gaussian_zero_order_hold", "drive_mean_hz": 220.0,
        "drive_std_hz": 22.0, "drive_sampling_hz": float(drive_sampling_hz),
        "connectivity_orientation_input": connectivity_orientation,
        "connectivity_mode": connectivity_mode, "connectivity_divisor": scale,
        "connectivity_self_edges": "removed", "variance_floor_hz": 1e-6,
        "seed": int(seed), "reference": "arxiv:2609.36773",
        "initial_state": "zero", "coupling_statistics": "cumulative_population_Welford",
        "parameter_roundoff_repairs": roundoff_repairs,
    }
    return source, time, meta
