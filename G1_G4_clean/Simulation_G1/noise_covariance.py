"""Per-sample sensor-space covariance and rank-aware spatial whitening.

Use the final scaled, referenced noise, never the signal-plus-noise observation.
Matrices stay float64 even when generated time series are exported as float32.
"""
from __future__ import annotations

import numpy as np


DEFAULT_RANK_RTOL = 1e-7
COVARIANCE_KEYS = ('noise_covariance', 'oracle_noise_covariance', 'sensor_noise_covariance')
NOISE_ARRAY_KEYS = (*COVARIANCE_KEYS, 'baseline_sensor')


def baseline_sample_count(duration_s, srate):
    """Require an explicit, nonempty baseline on the output sampling grid."""
    if (isinstance(duration_s, (bool, np.bool_)) or isinstance(srate, (bool, np.bool_))
            or not np.isfinite(duration_s) or not np.isfinite(srate)
            or duration_s <= 0 or srate <= 0):
        raise ValueError('Baseline duration and sampling rate must be finite and positive')
    count = duration_s * srate
    if count < 2 or not np.isclose(count, round(count), rtol=0, atol=1e-7):
        raise ValueError('Baseline duration must contain at least two whole samples')
    return int(round(count))


def sensor_noise_with_baseline(backend, clean, *, srate, sensor_noise_type, snr_db,
                               seed, baseline_samples, average_reference,
                               snr_mode='global_rms', clip_quantile=.95, real_sensor_noise=None):
    """Generate disjoint baseline/event segments with shared mixing and gain.

    SNR is calibrated using the event only. The baseline never contains target
    signal, and is not independently normalized to a desired noise power.
    """
    try:
        from .g1_simulation_core import TimeSeriesGenerator
    except ImportError:
        from g1_simulation_core import TimeSeriesGenerator

    clean = np.asarray(clean, dtype=np.float64)
    if clean.ndim != 2 or clean.shape[1] < 2 or not np.isfinite(clean).all():
        raise ValueError('Clean sensor signal must be finite channels by time')
    if isinstance(baseline_samples, bool) or int(baseline_samples) != baseline_samples or baseline_samples < 2:
        raise ValueError('baseline_samples must be an integer >=2')
    baseline_samples = int(baseline_samples)
    total = baseline_samples + clean.shape[1]
    info = dict(baseline_samples=baseline_samples, event_samples=clean.shape[1],
                baseline_interval_samples=[0, baseline_samples],
                event_interval_samples=[baseline_samples, total],
                seed=None if seed is None else int(seed),
                generation='joint_noise_stream_disjoint_intervals_shared_gain',
                scale_calibration='event_only', target_present_in_baseline=False)
    if snr_db is None:
        return np.zeros_like(clean), np.zeros((len(clean), baseline_samples)), info
    if not np.isfinite(snr_db):
        raise ValueError('Sensor SNR must be finite or None')
    rng = np.random.default_rng(seed)
    ts = TimeSeriesGenerator(srate, (0., total / srate), rng)
    supplied = None
    if real_sensor_noise is not None:
        supplied = np.asarray(real_sensor_noise, dtype=np.float64)
        if (supplied.ndim != 2 or supplied.shape[0] != len(clean)
                or supplied.shape[1] < total or not np.isfinite(supplied).all()):
            raise ValueError('Real sensor noise must contain disjoint baseline + event samples; no tiling')
        offset = int(rng.integers(supplied.shape[1] - total + 1)) if supplied.shape[1] > total else 0
        supplied = supplied[:, offset:offset + total].copy()
        info['real_noise_start_sample'] = offset
    # The dummy clean waveform only supplies shape/power to the existing noise
    # family generator. Its target waveform is never added to the baseline.
    extended_clean = np.tile(clean, (1, int(np.ceil(total / clean.shape[1]))))[:, :total]
    stream = backend._make_sensor_noise(extended_clean, ts, sensor_noise_type, 0.,
        snr_mode=snr_mode, clip_quantile=clip_quantile,
        real_sensor_noise=supplied, average_reference=average_reference)
    baseline, event = stream[:, :baseline_samples].copy(), stream[:, baseline_samples:].copy()
    baseline -= baseline.mean(axis=1, keepdims=True)
    event -= event.mean(axis=1, keepdims=True)
    if average_reference:
        baseline -= baseline.mean(axis=0, keepdims=True)
        event -= event.mean(axis=0, keepdims=True)
    power = float(np.mean(event ** 2))
    target = backend._snr_reference_power(clean, snr_mode, clip_quantile)
    if power <= 0 or target <= 0:
        raise ValueError('Event signal/noise has zero power; cannot calibrate baseline and event SNR')
    gain = float(np.exp(.5 * (np.log(target) - np.log(power)) - np.log(10) * snr_db / 20))
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError('Sensor SNR exceeds numerical scaling range')
    info['shared_noise_gain'] = gain
    return event * gain, baseline * gain, info


def baseline_covariance_payload(baseline_sensor, sensor_noise, *, background_sensor=None,
                                average_reference=False, baseline_metadata=None):
    """Default whitening uses observed baseline; event components are diagnostics."""
    baseline = np.asarray(baseline_sensor, dtype=np.float32)
    covariance, info = noise_covariance(baseline, average_reference=average_reference)
    if baseline.shape[0] != np.asarray(sensor_noise).shape[0]:
        raise ValueError('Baseline and event channel counts differ')
    diagnostics, details = covariance_payload(sensor_noise, background_sensor=background_sensor,
                                             average_reference=average_reference)
    info.update(provenance='generated_baseline', estimation_window='baseline_only',
                target_present=False)
    arrays = dict(baseline_sensor=baseline, noise_covariance=covariance,
                  oracle_noise_covariance=diagnostics['noise_covariance'],
                  sensor_noise_covariance=diagnostics['sensor_noise_covariance'])
    metadata = dict(noise_covariance_info=info,
                    oracle_noise_covariance_info=details['noise_covariance_info'],
                    sensor_noise_covariance_info=details['sensor_noise_covariance_info'],
                    baseline={} if baseline_metadata is None else dict(baseline_metadata))
    return arrays, metadata


def noise_covariance(noise, *, average_reference=False):
    """Centered empirical covariance (T-1 denominator) of channels x time noise."""
    values = np.asarray(noise, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError('Noise must be a finite channels-by-time matrix with at least two samples')
    centered = values - values.mean(axis=1, keepdims=True)
    covariance = centered @ centered.T / (values.shape[1] - 1)
    covariance = (covariance + covariance.T) / 2
    if not np.isfinite(covariance).all():
        raise ValueError('Noise covariance exceeds the finite numerical range')
    eigenvalues = np.linalg.eigvalsh(covariance)
    cutoff = max(float(eigenvalues[-1]), 0.) * DEFAULT_RANK_RTOL
    rank = int(np.count_nonzero(eigenvalues > cutoff))
    info = dict(estimator='empirical_centered', ddof=1, n_samples=values.shape[1],
                n_channels=values.shape[0], rank=rank, rank_rtol=DEFAULT_RANK_RTOL,
                reference='average' if average_reference else 'as_supplied',
                units='exported_sensor_unit_squared', channel_order='same_as_sensor_rows',
                estimation_window='entire_generated_sample',
                status='no_noise_variance' if rank == 0 else 'estimated',
                whitening='skip_if_zero_otherwise_retained_eigenspace',
                provenance='simulation_noise_oracle')
    return covariance, info


def covariance_payload(sensor_noise, *, background_sensor=None, average_reference=False):
    """Total nuisance covariance includes finite-sample cross terms with background."""
    measurement, measurement_info = noise_covariance(sensor_noise, average_reference=average_reference)
    measurement_info['components'] = ['sensor_noise']
    if background_sensor is None:
        total, total_info = measurement.copy(), dict(measurement_info)
    else:
        background = np.asarray(background_sensor, dtype=np.float64)
        noise = np.asarray(sensor_noise, dtype=np.float64)
        if background.shape != noise.shape:
            raise ValueError('Background and sensor noise must have identical channel/time shapes')
        total, total_info = noise_covariance(background + noise, average_reference=average_reference)
        total_info['components'] = ['background_sensor', 'sensor_noise']
    return ({'noise_covariance': total, 'sensor_noise_covariance': measurement},
            {'noise_covariance_info': total_info, 'sensor_noise_covariance_info': measurement_info})


def whitener_from_covariance(covariance, *, rtol=DEFAULT_RANK_RTOL):
    """Return rank x channels W; apply the SAME W to sensor data and leadfield.

    W C W.T is identity in the retained subspace. Zero noise has no defined
    whitener; callers should skip whitening for explicitly noiseless samples.
    """
    matrix = np.asarray(covariance, dtype=np.float64)
    if (matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]
            or not np.isfinite(matrix).all()):
        raise ValueError('Covariance must be a finite nonempty square matrix')
    if not np.isfinite(rtol) or not 0 < rtol < 1:
        raise ValueError('rtol must lie strictly between zero and one')
    scale = float(np.max(np.abs(matrix)))
    if np.max(np.abs(matrix - matrix.T)) > scale * rtol:
        raise ValueError('Covariance must be symmetric')
    eigenvalues, eigenvectors = np.linalg.eigh((matrix + matrix.T) / 2)
    cutoff = max(float(eigenvalues[-1]), 0.) * rtol
    if eigenvalues[0] < -cutoff:
        raise ValueError('Covariance must be positive semidefinite')
    keep = eigenvalues > cutoff
    if not np.any(keep):
        raise ValueError('Zero noise covariance: skip whitening for this noiseless sample')
    return eigenvectors[:, keep].T / np.sqrt(eigenvalues[keep])[:, None]
