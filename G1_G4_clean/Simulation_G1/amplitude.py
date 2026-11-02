"""Explicit dipole amplitudes and signal-independent sensor noise.

These are unit contracts, not physiological priors. A caller must establish that
its leadfield maps A*m dipole moments to the declared sensor units before opting
in. NMM PSP in mV is not itself a dipole moment.
"""
from __future__ import annotations

import numpy as np


_SENSOR_UNITS = {
    'V': ('V', 1.), 'uV': ('V', 1e-6),
    'T': ('T', 1.), 'fT': ('T', 1e-15),
    'T/m': ('T/m', 1.), 'fT/cm': ('T/m', 1e-13),
}


def _units(sensor_units, n_channels):
    if isinstance(sensor_units, str):
        units = [sensor_units] * n_channels
    else:
        try:
            units = list(sensor_units)
        except TypeError as exc:
            raise ValueError('sensor_units must explicitly declare one unit or one per channel') from exc
    if len(units) != n_channels or any(not isinstance(unit, str) or unit not in _SENSOR_UNITS for unit in units):
        raise ValueError('sensor_units must match channels and use V, uV, T, fT, T/m, or fT/cm')
    return units


def _noise_std(noise_std, units):
    if noise_std is None:
        return np.zeros(len(units))
    original = np.asarray(noise_std)
    if original.dtype.kind in 'bc' or original.dtype.kind not in 'fiu':
        raise ValueError('sensor_noise_std must contain finite nonnegative real numbers')
    values = np.asarray(noise_std, dtype=float)
    if values.ndim == 0:
        if len(set(units)) != 1:
            raise ValueError('Mixed sensor units require one sensor_noise_std per channel')
        values = np.full(len(units), float(values))
    if values.shape != (len(units),) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('sensor_noise_std must be nonnegative and match the sensor channels')
    return values


def _positive_scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f'{name} must be a finite positive scalar')
    if np.iscomplexobj(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a finite positive scalar')
    return float(value)


def validate_amplitude_settings(*, amplitude_mode='relative', leadfield_source_unit=None,
                                sensor_units=None, foreground_peak_am=None,
                                sensor_noise_std=None, sensor_snr_db=10.,
                                normalize_source=True, normalize_sensor=False,
                                sensor_noise_peak_ratio=None, n_channels=None):
    """Validate opt-in physical amplitudes; absent noise amplitude means noiseless.

    Pass n_channels once the forward model is loaded. Scalar noise amplitude is
    allowed only for identical declared sensor units. Returned amplitudes remain
    in those native sensor units; this helper does not modify the forward model.
    """
    if amplitude_mode not in ('relative', 'physical'):
        raise ValueError('amplitude_mode must be relative or physical')
    if amplitude_mode == 'relative':
        if any(value is not None for value in (leadfield_source_unit, sensor_units,
                                               foreground_peak_am, sensor_noise_std)):
            raise ValueError('Physical unit/amplitude fields require amplitude_mode=physical')
        return {'amplitude_mode': 'relative'}
    if leadfield_source_unit != 'A*m':
        raise ValueError('Physical mode requires leadfield_source_unit=A*m; PSP is not dipole current')
    peak = _positive_scalar(foreground_peak_am, 'foreground_peak_am')
    if normalize_source or normalize_sensor or sensor_noise_peak_ratio is not None:
        raise ValueError('Physical mode disallows normalize_source, normalize_sensor and sensor_noise_peak_ratio')
    if sensor_snr_db is not None:
        raise ValueError('Physical mode requires sensor_snr_db=None; specify fixed sensor_noise_std instead')
    if n_channels is None:
        if isinstance(sensor_units, str):
            shape = np.shape(sensor_noise_std)
            n_channels = shape[0] if len(shape) == 1 else 1
        else:
            try:
                n_channels = len(sensor_units)
            except TypeError as exc:
                raise ValueError('Physical mode requires explicit sensor_units') from exc
    if isinstance(n_channels, (bool, np.bool_)) or not isinstance(n_channels, (int, np.integer)) or n_channels < 1:
        raise ValueError('n_channels must be a positive integer')
    units = _units(sensor_units, n_channels)
    std = _noise_std(sensor_noise_std, units)
    return dict(amplitude_mode='physical', leadfield_source_unit='A*m',
                foreground_peak_am=peak, sensor_units=units, sensor_noise_std=std.tolist(),
                source_amplitude_definition='maximum absolute foreground dipole moment across vertices and time',
                physical_amplitude_provenance='user_requested_not_a_physiological_default')


def normalize_sensor_units(leadfield, sensor_units):
    """Return a copied leadfield with rows in SI, plus its per-channel SI units.

    The source unit is unchanged: this cannot convert an uncalibrated PSP model
    to dipole current. fT/cm converts to T/m with factor 1e-13.
    """
    if np.iscomplexobj(leadfield):
        raise ValueError('Leadfield must be real')
    matrix = np.asarray(leadfield, dtype=float)
    if matrix.ndim != 2 or min(matrix.shape) < 1 or not np.isfinite(matrix).all():
        raise ValueError('Leadfield must be finite channels by sources')
    units = _units(sensor_units, len(matrix))
    factors = np.asarray([_SENSOR_UNITS[unit][1] for unit in units])
    return matrix * factors[:, None], [_SENSOR_UNITS[unit][0] for unit in units]


def foreground_amplitude_gain(foreground, foreground_peak_am):
    """One gain for the foreground and ALL associated background/baseline arrays."""
    target = _positive_scalar(foreground_peak_am, 'foreground_peak_am')
    if np.iscomplexobj(foreground):
        raise ValueError('Foreground must be real')
    values = np.asarray(foreground, dtype=float)
    if values.ndim != 2 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError('Foreground must be finite sources by time')
    peak = float(np.max(np.abs(values)))
    if peak <= 0:
        raise ValueError('Zero foreground cannot be scaled to a positive physical peak')
    gain = target / peak
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError('Foreground amplitude exceeds the numerical scaling range')
    return gain


def generate_fixed_sensor_noise(backend, clean, *, srate, noise_type, seed,
                                baseline_samples, average_reference, noise_std,
                                sensor_units, real_sensor_noise=None):
    """Return event, baseline, metadata; never calibrate against clean power.

    IID noise uses its theoretical standard deviation. Other families (and real
    noise) use one per-channel gain measured from the baseline before reference.
    Both segments receive the same gain. Average reference then changes marginal
    noise amplitudes; the realized post-reference deviations are reported.
    """
    if np.iscomplexobj(clean):
        raise ValueError('Clean sensor signal must be real')
    values = np.asarray(clean, dtype=float)
    if values.ndim != 2 or min(values.shape) < 1 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError('Clean sensor signal must be finite channels by time')
    _positive_scalar(srate, 'srate')
    if isinstance(baseline_samples, (bool, np.bool_)) or not isinstance(baseline_samples, (int, np.integer)) or baseline_samples < 2:
        raise ValueError('baseline_samples must be an integer >=2')
    units = _units(sensor_units, len(values))
    requested = _noise_std(noise_std, units)
    if average_reference and len(set(units)) != 1:
        raise ValueError('Average reference requires identical sensor units; convert compatible units first')
    count = baseline_samples + values.shape[1]
    info = dict(sensor_units=units, requested_sensor_noise_std=requested.tolist(),
                seed=None if seed is None else int(seed), baseline_samples=int(baseline_samples),
                event_samples=values.shape[1], baseline_interval_samples=[0, int(baseline_samples)],
                event_interval_samples=[int(baseline_samples), int(count)],
                generation='fixed_amplitude_joint_stream_disjoint_intervals',
                target_present_in_baseline=False, scale_depends_on_clean=False,
                std_definition='population temporal standard deviation before reference',
                scale_calibration='disabled', shared_noise_gain=np.zeros(len(values)).tolist())
    if real_sensor_noise is not None and noise_type != 'real':
        raise ValueError('real_sensor_noise requires noise_type=real')
    if not np.any(requested):
        info.update(noise_status='no_added_noise', baseline_sensor_noise_std=[0.] * len(values),
                    event_sensor_noise_std=[0.] * len(values))
        return np.zeros_like(values), np.zeros((len(values), baseline_samples)), info
    rng = np.random.default_rng(seed)
    if noise_type == 'iid_white':
        stream = rng.standard_normal((len(values), count))
        gain = requested
        info['scale_calibration'] = 'theoretical_iid_standard_deviation'
    else:
        if noise_type == 'real':
            if real_sensor_noise is None or np.iscomplexobj(real_sensor_noise):
                raise ValueError('Real noise requires a finite real baseline plus event recording')
            stream = np.asarray(real_sensor_noise, dtype=float)
            if stream.ndim != 2 or stream.shape[0] != len(values) or stream.shape[1] < count or not np.isfinite(stream).all():
                raise ValueError('Real noise must match channels and contain baseline plus event; no tiling')
            offset = int(rng.integers(stream.shape[1] - count + 1)) if stream.shape[1] > count else 0
            stream = stream[:, offset:offset + count].copy()
            info['real_noise_start_sample'] = offset
        else:
            try:
                from .g1_simulation_core import TimeSeriesGenerator
            except ImportError:
                from g1_simulation_core import TimeSeriesGenerator
            ts = TimeSeriesGenerator(srate, (0., count / srate), rng)
            # Shape-only driver: replacing clean by zero or doubling it cannot change noise.
            dummy = np.broadcast_to(np.where(np.arange(count) % 2, 1., -1.), (len(values), count)).copy()
            stream = backend._make_sensor_noise(dummy, ts, noise_type, 0., snr_mode='global_rms',
                                                average_reference=False)
        stream = np.asarray(stream, dtype=float)
        if stream.shape != (len(values), count) or not np.isfinite(stream).all():
            raise ValueError('Noise family returned an invalid joint stream')
        baseline_std = stream[:, :baseline_samples].std(axis=1)
        if np.any((requested > 0) & (baseline_std <= 0)):
            raise ValueError('Noise baseline has zero variance in a channel requiring noise')
        gain = np.divide(requested, baseline_std, out=np.zeros_like(requested), where=baseline_std > 0)
        info['scale_calibration'] = 'baseline_standard_deviation_before_reference'
    stream = stream * gain[:, None]
    baseline, event = stream[:, :baseline_samples].copy(), stream[:, baseline_samples:].copy()
    for segment in (baseline, event):
        segment -= segment.mean(axis=1, keepdims=True)
        if average_reference:
            segment -= segment.mean(axis=0, keepdims=True)
    if not np.isfinite(baseline).all() or not np.isfinite(event).all():
        raise ValueError('Fixed noise amplitude exceeds numerical scaling range')
    info.update(shared_noise_gain=gain.tolist(), noise_status='added_sensor_noise',
                baseline_sensor_noise_std=baseline.std(axis=1).tolist(),
                event_sensor_noise_std=event.std(axis=1).tolist(), average_reference=bool(average_reference))
    return event, baseline, info
