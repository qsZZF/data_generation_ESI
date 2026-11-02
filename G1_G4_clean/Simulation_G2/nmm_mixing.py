"""Explicit foreground/background and measurement-noise components for NMM data."""
from __future__ import annotations

import numpy as np
from scipy import sparse

from Simulation_G1.noise_covariance import (
    baseline_sample_count, baseline_covariance_payload, sensor_noise_with_baseline)


def validate_sensor_snr_settings(mode, clip_quantile):
    if mode not in ('global_rms', 'robust_global_rms', 'active_rms'):
        raise ValueError('sensor_snr_mode must be global_rms, robust_global_rms or active_rms')
    if (isinstance(clip_quantile, (bool, np.bool_)) or not isinstance(clip_quantile, (int, float, np.number))
            or np.iscomplexobj(clip_quantile) or not np.isfinite(clip_quantile) or not .5 <= clip_quantile <= .999):
        raise ValueError('sensor_snr_clip_quantile must lie in [.5,.999]')


def disjoint_baseline_start(n_samples, event_start, event_stop, baseline_samples):
    """Choose a contiguous same-record baseline without reusing event samples."""
    if event_start >= baseline_samples:
        return event_start - baseline_samples
    if event_stop + baseline_samples <= n_samples:
        return event_stop
    raise ValueError('Recording has no contiguous nonoverlapping baseline window; generate a longer trajectory')


def project_baseline(background_region, projection, source_metadata):
    """Apply the event's calibrated background and normalization gains unchanged."""
    values = _matrix(background_region, 'baseline_region')
    values = values - values.mean(axis=1, keepdims=True)
    return (projection @ values) * (source_metadata['source_background_scale'] *
                                     source_metadata['source_normalization_scale'])


def apply_physical_source_amplitude(arrays, metadata, foreground_peak_am):
    """Apply one explicit current gain to every source component and baseline map."""
    from Simulation_G1.amplitude import foreground_amplitude_gain
    gain = foreground_amplitude_gain(arrays['source_foreground'], foreground_peak_am)
    for key in arrays:
        arrays[key] *= gain
    metadata['source_normalization_scale'] *= gain
    for key in ('source_foreground_power', 'source_background_power'):
        metadata[key] *= gain ** 2
    metadata.update(source_amplitude_gain=gain, amplitude_unit='A*m',
                    source_amplitude_mapping='foreground vertex peak; one gain for all source and baseline components')


def _matrix(value, name):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or min(result.shape) < 1 or result.shape[1] < 2 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite nonempty channels/regions-by-times matrix")
    return result


def _power(value, weights=None):
    power = float(np.average(np.mean(value ** 2, axis=1), weights=weights))
    if not np.isfinite(power):
        raise ValueError("Signal power must be finite")
    return power


def _snr(signal_power, noise_power):
    if signal_power <= 0 or noise_power <= 0:
        return None
    return float(10 / np.log(10) * (np.log(signal_power) - np.log(noise_power)))


def _scale_for_snr(signal_power, noise_power, snr_db):
    if not np.isfinite(snr_db):
        raise ValueError("Requested SNR must be finite; use None to disable noise")
    if signal_power <= 0 or noise_power <= 0:
        raise ValueError("Signal or noise has zero power; cannot achieve requested SNR")
    with np.errstate(over='ignore', under='ignore'):
        scale = float(np.exp(0.5 * (np.log(signal_power) - np.log(noise_power)) - np.log(10) * snr_db / 20))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Requested SNR exceeds numerical scaling range")
    return scale


def mix_sources(foreground_region, background_region, projection, vertex_areas,
                source_snr_db=None, normalize_source=True, *, foreground_vertex=None,
                source_snr_metric='vertex_mean'):
    """Mix at vertex-mean SNR; explicit area_weighted retains legacy calibration.

    Source SNR is a component-power ratio, not the causal effect of an excitation
    parameter. Background may occupy the same regions as the foreground. Supply
    foreground_vertex when spatial dynamics replace the static projection; its
    temporal mean is removed before calibrating background power.
    """
    if source_snr_metric not in ('vertex_mean', 'area_weighted'):
        raise ValueError('source_snr_metric must be vertex_mean or area_weighted')
    fg = _matrix(foreground_region, 'foreground_region')
    if (background_region is None) != (source_snr_db is None):
        raise ValueError("background_region and source_snr_db must be supplied together")
    fg = fg - fg.mean(axis=1, keepdims=True)
    bg = np.zeros_like(fg) if background_region is None else _matrix(background_region, 'background_region')
    if bg.shape != fg.shape:
        raise ValueError("Foreground and background must have identical region/time shapes")
    bg = bg - bg.mean(axis=1, keepdims=True)
    projection = sparse.csc_matrix(projection, dtype=np.float64)
    if projection.shape[1] != fg.shape[0] or not np.isfinite(projection.data).all():
        raise ValueError("Projection must be finite with one column per input region")
    areas = np.asarray(vertex_areas, dtype=np.float64)
    if (areas.shape != (projection.shape[0],) or not np.isfinite(areas).all()
            or np.any(areas < 0) or not np.isfinite(areas.sum()) or areas.sum() <= 0):
        raise ValueError("Vertex areas must be finite, nonnegative and have a positive sum")
    area_weights = areas / areas.sum()
    weights = area_weights if source_snr_metric == 'area_weighted' else None
    fg_vertex = projection @ fg if foreground_vertex is None else _matrix(foreground_vertex, 'foreground_vertex')
    if fg_vertex.shape != (projection.shape[0], fg.shape[1]):
        raise ValueError('foreground_vertex must match the projection vertices and foreground times')
    if foreground_vertex is not None:
        fg_vertex = fg_vertex - fg_vertex.mean(axis=1, keepdims=True)
    bg_vertex = projection @ bg
    raw_fg_power, raw_bg_power = _power(fg_vertex, weights), _power(bg_vertex, weights)
    background_scale = 0. if source_snr_db is None else _scale_for_snr(raw_fg_power, raw_bg_power, source_snr_db)
    bg *= background_scale
    bg_vertex *= background_scale
    vertex_source = fg_vertex + bg_vertex
    normalization_scale = 1.
    if normalize_source:
        maximum = float(np.max(np.abs(vertex_source)))
        if not np.isfinite(maximum) or maximum <= 0:
            raise ValueError("Mixed source has zero or nonfinite amplitude; cannot normalize")
        normalization_scale = 1. / maximum
    arrays = {
        'region_foreground': fg, 'region_background': bg, 'region_source': fg + bg,
        'source_foreground': fg_vertex, 'source_background': bg_vertex, 'source': vertex_source,
    }
    for key in arrays:
        arrays[key] *= normalization_scale
        if not np.isfinite(arrays[key]).all():
            raise ValueError("Source mixing produced nonfinite values")
    foreground_power = _power(arrays['source_foreground'], weights)
    background_power = _power(arrays['source_background'], weights)
    metadata = {
        'requested_source_snr_db': source_snr_db,
        'realized_source_snr_db': _snr(foreground_power, background_power),
        'source_snr_metric': source_snr_metric,
        'mixing_semantics': 'nmm-mixing-v2',
        'source_representation': 'per_vertex_activity',
        'amplitude_unit': 'relative_model_activity',
        'source_snr_definition': f'10log10({source_snr_metric} whole-mesh foreground power / background power), after temporal demeaning',
        'realized_vertex_source_snr_db': _snr(_power(arrays['source_foreground']), _power(arrays['source_background'])),
        'realized_area_weighted_source_snr_db': _snr(_power(arrays['source_foreground'], area_weights),
                                                     _power(arrays['source_background'], area_weights)),
        'source_foreground_power': foreground_power, 'source_background_power': background_power,
        'raw_source_foreground_power': raw_fg_power, 'raw_source_background_power': raw_bg_power,
        'source_background_scale': background_scale, 'source_normalization_scale': normalization_scale,
        'source_matches_static_projection': foreground_vertex is None,
    }
    return arrays, metadata


def make_sensors(backend, source_arrays, *, srate, sensor_noise_type='iid_white',
                 sensor_snr_db=None, sensor_seed=0, average_reference=True, real_sensor_noise=None,
                 baseline_duration_s=2., baseline_source=None, baseline_metadata=None,
                 covariance_estimator='empirical', amplitude_mode='relative',
                 sensor_noise_std=None, sensor_units=None, record_observability=False,
                 observability_settings=None, sensor_snr_mode='active_rms', sensor_snr_clip_quantile=.95):
    """Project both sources and calibrate measurement noise after referencing."""
    if not np.isfinite(srate) or srate <= 0:
        raise ValueError("srate must be positive and finite")
    validate_sensor_snr_settings(sensor_snr_mode, sensor_snr_clip_quantile)
    if amplitude_mode not in ('relative', 'physical'):
        raise ValueError('Unknown amplitude_mode')
    leadfield = np.asarray(backend.leadfield, dtype=np.float64)
    if leadfield.ndim != 2 or min(leadfield.shape) < 1 or not np.isfinite(leadfield).all():
        raise ValueError("Leadfield must be a finite nonempty matrix")
    components = []
    for key in ('source_foreground', 'source_background'):
        source = _matrix(source_arrays[key], key)
        if leadfield.shape[1] != source.shape[0]:
            raise ValueError("Leadfield and source vertex counts differ")
        sensor_component = leadfield @ source
        sensor_component -= sensor_component.mean(axis=1, keepdims=True)
        if average_reference:
            # Subtract an anchor first so exactly common-mode inputs vanish exactly.
            sensor_component -= sensor_component[:1].copy()
            sensor_component -= sensor_component.mean(axis=0, keepdims=True)
        components.append(sensor_component)
    foreground, background = components
    if foreground.shape != background.shape:
        raise ValueError("Source components must have identical time lengths")
    clean = foreground + background
    clean_power = _power(clean)
    baseline_samples = baseline_sample_count(baseline_duration_s, srate)
    if baseline_source is None and np.any(source_arrays['source_background']):
        raise ValueError('Source background requires a distinct nonoverlapping baseline_source window')
    baseline_background = np.zeros((leadfield.shape[0], baseline_samples))
    if baseline_source is not None:
        baseline_source = _matrix(baseline_source, 'baseline_source')
        if baseline_source.shape != (leadfield.shape[1], baseline_samples):
            raise ValueError('baseline_source must match source vertices and baseline duration')
        baseline_background = leadfield @ baseline_source
        baseline_background -= baseline_background.mean(axis=1, keepdims=True)
        if average_reference:
            baseline_background -= baseline_background[:1].copy()
            baseline_background -= baseline_background.mean(axis=0, keepdims=True)
    if real_sensor_noise is not None:
        if sensor_noise_type != 'real' or (sensor_snr_db is None and amplitude_mode != 'physical'):
            raise ValueError("real_sensor_noise requires noise type 'real' and a finite sensor SNR")
        real_sensor_noise = _matrix(real_sensor_noise, 'real_sensor_noise')
        if real_sensor_noise.shape[0] != clean.shape[0] or real_sensor_noise.shape[1] < clean.shape[1] + baseline_samples:
            raise ValueError("Real noise must match channels and contain event plus baseline duration")
        real_sensor_noise = real_sensor_noise - real_sensor_noise.mean(axis=1, keepdims=True)
        if average_reference and amplitude_mode != 'physical':
            real_sensor_noise -= real_sensor_noise[:1].copy()
            real_sensor_noise -= real_sensor_noise.mean(axis=0, keepdims=True)
        if _power(real_sensor_noise) <= 0:
            raise ValueError("Real sensor noise has zero power after referencing")
    if amplitude_mode == 'physical':
        from Simulation_G1.amplitude import generate_fixed_sensor_noise
        if sensor_snr_db is not None:
            raise ValueError('Physical noise uses fixed amplitude, not requested SNR')
        noise, baseline_noise, noise_metadata = generate_fixed_sensor_noise(
            backend, clean, srate=srate, noise_type=sensor_noise_type, seed=sensor_seed,
            baseline_samples=baseline_samples, average_reference=average_reference,
            noise_std=sensor_noise_std, sensor_units=sensor_units, real_sensor_noise=real_sensor_noise)
    else:
        noise, baseline_noise, noise_metadata = sensor_noise_with_baseline(
            backend, clean, srate=srate, sensor_noise_type=sensor_noise_type,
            snr_db=sensor_snr_db, seed=sensor_seed, baseline_samples=baseline_samples,
            snr_mode=sensor_snr_mode, clip_quantile=sensor_snr_clip_quantile,
            average_reference=average_reference, real_sensor_noise=real_sensor_noise)
    arrays = {'foreground_sensor': foreground, 'background_sensor': background,
              'clean_sensor': clean, 'sensor_noise': noise, 'sensor': clean + noise}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Sensor mixing produced nonfinite values")
    noise_power = _power(noise)
    fg_power, bg_power = _power(foreground), _power(background)
    from Simulation_G1.g1_simulation_core import G1SourceSimulator
    reference_power = G1SourceSimulator._snr_reference_power(clean, sensor_snr_mode, sensor_snr_clip_quantile)
    target_gfp = np.sqrt(np.mean(foreground ** 2, axis=0))
    active_threshold = float(np.quantile(target_gfp, sensor_snr_clip_quantile))
    active = target_gfp >= active_threshold
    nuisance = background + noise
    target_active_power = _power(foreground[:, active])
    nuisance_power = _power(nuisance)
    metadata = {
        'requested_sensor_snr_db': sensor_snr_db,
        'realized_sensor_snr_db': _snr(clean_power, noise_power),
        'realized_sensor_snr_reference_db': _snr(reference_power, noise_power),
        'projected_source_snr_db': _snr(fg_power, bg_power),
        'effective_target_snr_db': _snr(fg_power, nuisance_power),
        'effective_target_global_snr_db': _snr(fg_power, nuisance_power),
        'effective_target_active_snr_db': _snr(target_active_power, nuisance_power),
        'effective_target_active_window_snr_db': _snr(target_active_power, _power(nuisance[:, active])),
        'sensor_snr_definition': '10log10(total clean sensor power / measurement noise power), global RMS after referencing',
        'sensor_snr_calibration_definition': f'10log10({sensor_snr_mode} total-clean reference power / GLOBAL event measurement-noise power); baseline shares event-calibrated gain',
        'effective_target_snr_definition': '10log10(foreground sensor power / power of background sensor + measurement noise)',
        'effective_target_active_snr_definition': 'active foreground power / GLOBAL event power of source background plus sensor noise',
        'effective_target_active_window_snr_definition': 'foreground and combined nuisance powers both restricted to the foreground activity window',
        'target_activity_mask_rule': 'foreground GFP >= quantile(sensor_snr_clip_quantile); ties included',
        'target_activity_threshold': active_threshold, 'target_activity_samples': int(active.sum()),
        'reference_mode': 'average' if average_reference else 'as_supplied',
        'sensor_snr_mode': sensor_snr_mode, 'sensor_snr_clip_quantile': float(sensor_snr_clip_quantile),
        'sensor_noise_type': sensor_noise_type,
        'sensor_seed': sensor_seed, 'measurement_noise_enabled': bool(np.any(noise)),
        'clean_sensor_power': clean_power, 'noise_sensor_power': noise_power,
        'foreground_sensor_power': fg_power, 'background_sensor_power': bg_power,
        'real_sensor_noise_used': real_sensor_noise is not None,
    }
    if sensor_noise_type == 'iid_white':
        metadata['sensor_noise_covariance_convention'] = 'IID Gaussian before temporal demeaning and reference; average reference induces channel dependence and rank loss'
    covariance_arrays, covariance_metadata = baseline_covariance_payload(
        baseline_background + baseline_noise, noise, background_sensor=background,
        average_reference=average_reference, estimator=covariance_estimator,
        baseline_metadata={**noise_metadata, **(baseline_metadata or {}),
                           'duration_s': baseline_duration_s,
                           'source_background_included': baseline_source is not None})
    arrays.update(covariance_arrays)
    metadata.update(covariance_metadata)
    if amplitude_mode == 'physical':
        units = [sensor_units] * len(leadfield) if isinstance(sensor_units, str) else list(sensor_units)
        metadata.update(amplitude_mode='physical', sensor_units=units,
                        measurement_noise_calibration='fixed_pre_reference_amplitude',
                        sensor_snr_calibration_definition='no requested SNR; fixed sensor-noise amplitude independent of target')
        for key in ('noise_covariance_info', 'oracle_noise_covariance_info', 'sensor_noise_covariance_info'):
            metadata[key]['channel_units'] = units
            metadata[key]['units'] = 'outer_product_of_declared_sensor_units'
        if len(set(units)) > 1:
            for key in ('realized_sensor_snr_db', 'projected_source_snr_db', 'effective_target_snr_db',
                        'realized_sensor_snr_reference_db', 'effective_target_global_snr_db',
                        'effective_target_active_snr_db', 'effective_target_active_window_snr_db',
                        'target_activity_threshold', 'target_activity_samples',
                        'clean_sensor_power', 'noise_sensor_power', 'foreground_sensor_power', 'background_sensor_power'):
                metadata[key] = None
            metadata['sensor_snr_definition'] = 'mixed units; use whitened target/baseline diagnostic'
    if record_observability:
        from Simulation_G1.observability import analyze_observability
        if not hasattr(backend, '_observability_cache'):
            backend._observability_cache = {}
        metadata['observability'] = analyze_observability(leadfield, source_arrays['source_foreground'].astype(np.float32),
            noise_covariance=arrays['noise_covariance'], average_reference=average_reference,
            cache=backend._observability_cache, **(observability_settings or {}))
        metadata['observability']['target'] = 'source_foreground'
        metadata['observability']['source_stage'] = 'exported_float32_foreground'
        metadata['observability'].pop('cache_hit', None)
    return arrays, metadata
