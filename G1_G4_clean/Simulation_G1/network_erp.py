"""Opt-in geodesic multi-component ERP with explicit ROI power calibration."""
from dataclasses import asdict
import hashlib

import numpy as np

from .wx_partitioned import build_partitioned_weights


def _streams(config):
    if config.component_seeds is not None:
        return {key:int(config.component_seeds[key]) for key in ('spatial','waveform','background')}
    sequence = np.random.SeedSequence(config.random_seed)
    return {name:int(child.generate_state(1, dtype=np.uint64)[0])
            for name,child in zip(('spatial','waveform','background'),sequence.spawn(3))}


def generate_components(simulator, config, settings, assets, *, components=None):
    from .spectral import generate_coefficients, _component_output

    maps = assets['network_maps']
    if maps is None:
        raise ValueError('geodesic ERP requires registered network maps')
    chosen = settings.get('network_indices', list(range(maps.shape[1])))
    raw_indices = np.asarray(chosen)
    if (raw_indices.ndim != 1 or not len(raw_indices) or raw_indices.dtype.kind not in 'iu'
            or len(np.unique(raw_indices)) != len(raw_indices)
            or np.any(raw_indices < 0) or np.any(raw_indices >= maps.shape[1])):
        raise ValueError('network_indices must be unique valid integer map indices')
    bundle = assets['bundle']
    modes = settings.get('n_modes', bundle['basis'].shape[1])
    if isinstance(modes,bool) or not isinstance(modes,(int,np.integer)) or not 1 <= modes <= bundle['basis'].shape[1]:
        raise ValueError('n_modes outside saved basis')
    if settings['spatial_length_mm'] <= 0:
        raise ValueError('geodesic ERP spatial_length_mm must be positive')
    seeds = _streams(config)
    mask_hash = hashlib.sha256(simulator.cortex_mask.astype(np.uint8).tobytes()).hexdigest()
    key = (assets['key'],mask_hash,seeds['spatial'],modes,tuple(raw_indices),
           settings['spatial_length_mm'],settings['erp_local_width_mm'])
    cached = getattr(simulator,'_network_erp_weights_cache',None)
    if cached is None or cached[0] != key:
        local_bundle = {**bundle,'basis':bundle['basis'][:,:modes],'eigenvalues':bundle['eigenvalues'][:modes]}
        weights = build_partitioned_weights(local_bundle,maps[:,raw_indices],simulator.cortex_mask,
            simulator.vertices*simulator.coord_to_mm,simulator.faces,seed=seeds['spatial'],
            length_mm=settings['spatial_length_mm'],local_width_mm=settings['erp_local_width_mm'],
            n_networks=len(raw_indices),local_per_network=2)
        for value in weights.values():
            if isinstance(value,np.ndarray):
                value.setflags(write=False)
        simulator._network_erp_weights_cache = (key,weights)
    else:
        weights = cached[1]
    background_settings = {**settings,'scenario':'rest','erp_spatial_mode':'random'}
    generated = generate_coefficients(bundle,background_settings,srate=config.srate,
        time_window=config.time_window,baseline_duration_s=config.baseline_duration_s,
        rng=np.random.default_rng(seeds['background']),network_coefficients=assets['network_coefficients'],
        _rest_as_background=True)
    basis = bundle['basis'][:,:modes]
    background = basis @ generated['background']
    baseline = basis @ generated['baseline_background']
    background[~simulator.cortex_mask] = 0
    baseline[~simulator.cortex_mask] = 0
    background -= background.mean(axis=1,keepdims=True)
    baseline -= baseline.mean(axis=1,keepdims=True)
    time = config.time_window[0] + np.arange(background.shape[1])/config.srate
    onset = settings['event_onset_s']
    if not config.time_window[0] <= onset < config.time_window[1]:
        raise ValueError('ERP onset must lie in time_window')
    offsets = np.arange(weights['W'].shape[1]) % 3
    jitter = float(np.random.default_rng(seeds['waveform']).normal(0,settings['erp_latency_jitter_s']))
    latency = onset+settings['erp_latency_s']+jitter+offsets*settings['erp_component_interval_s']
    if np.any(latency < onset) or np.any(latency >= config.time_window[1]):
        raise ValueError('ERP component centers must lie between onset and end of time_window')
    widths = settings['erp_width_s']*(1+.25*offsets)
    ramp = np.sin(.5*np.pi*np.clip((time-onset)/min(.05,settings['erp_width_s']),0,1))**2
    gains = np.where(weights['groups']==0,1.,settings['erp_secondary_gain'])
    polarity = np.where(offsets==1,-.7,1.)
    x = gains[:,None]*polarity[:,None]*np.exp(-.5*((time[None,:]-latency[:,None])/widths[:,None])**2)*ramp
    raw = weights['W'] @ x
    offset = raw.mean(axis=1,keepdims=True)
    foreground = raw-offset
    # Match the legacy exported centering before measuring the local power ratio.
    roi = weights['roi']
    active_stop = min(config.time_window[1],float(np.max(latency+3*widths)))
    active = (time>=onset)&(time<=active_stop)
    if np.count_nonzero(active)<2:
        raise ValueError('ERP calibration window requires at least two samples')
    area = bundle['mass'][roi]
    power = lambda values:float(np.average(np.mean(values[roi][:,active]**2,axis=1),weights=area))
    foreground_power,background_power = power(foreground),power(background)
    if min(foreground_power,background_power)<=0:
        raise ValueError('ERP and ROI background must both have nonzero power')
    gain = np.sqrt(settings['erp_roi_power_ratio']*background_power/foreground_power)
    foreground *= gain
    source = (foreground+background).astype(np.float32)
    source -= source.mean(axis=1,keepdims=True)
    scale = max(float(np.max(abs(source))),1e-12) if config.normalize_source else 1.
    source /= scale
    foreground = (foreground/scale).astype(np.float32)
    background = source.astype(float)-foreground.astype(float)
    baseline = (baseline/scale).astype(np.float32)
    actual_ratio = power(foreground)/power(background)
    if not np.isclose(actual_ratio,settings['erp_roi_power_ratio'],rtol=1e-5,atol=1e-8):
        raise ValueError('Export precision changed the requested ROI power ratio')
    actual = dict(foreground=foreground,background=background,baseline_background=baseline)
    _component_output(components,actual,actual,1.,False)
    meta = {**asdict(config),**generated['metadata'],'generator_type':'G1_spectral',
        'spectral_parameters':dict(settings),'target_definition':'primary_and_secondary_geodesic_evoked_response',
        'scenario':'erp','erp_spatial_mode':'geodesic','erp_spatial_seed':seeds['spatial'],
        'spectral_seed':seeds['waveform'],'source_background_seed':seeds['background'],
        'n_vertices':simulator.n_sources,'n_sensors':simulator.n_sensors,
        'basis_sha256':assets['basis_sha256'],'network_sha256':assets['network_sha256'],
        'cortex_mask_applied':True,'cortex_mask_sha256':mask_hash,
        'source_normalization_scale':scale,'coefficient_energy':'masked vertex field energy; W columns may overlap',
        'exported_foreground_centering':'whole-event per-vertex mean removed; causal ERP has a constant pre-onset offset',
        'active_vertices':np.flatnonzero(np.any(foreground!=0,axis=1)).tolist(),
        'active_vertices_definition':'nonzero foreground target including secondary response, not nuisance',
        'network_erp':dict(version=1,weights_sha256=hashlib.sha256(weights['W'].tobytes()).hexdigest(),
            spatial=weights['metadata'],network_indices=raw_indices.tolist(),seeds=seeds,
            roi_vertices=np.flatnonzero(roi).tolist(),roi_metric='surface-area-weighted mean squared activity',
            roi_window_s=[float(onset),active_stop],roi_power_ratio_requested=settings['erp_roi_power_ratio'],
            roi_power_ratio_realized=actual_ratio,latencies_s=latency.tolist(),widths_s=widths.tolist(),
            secondary_gain=settings['erp_secondary_gain'],polarity=polarity.tolist(),foreground_gain=float(gain),
            raw_pre_onset_zero=True,validation_status='mechanism-validated; not empirical canonical ERP validation')}
    return source,meta,background,baseline
