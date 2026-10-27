"""Mass-normalized cortical modes with resting, evoked and induced activity.

The fast coefficient API does not project to vertices. A basis with Phi.T M Phi=I
makes squared coefficient norm equal to surface-area-weighted source energy.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

import numpy as np

try:
    from .noise_covariance import baseline_sample_count
except ImportError:
    from noise_covariance import baseline_sample_count


DEFAULTS = dict(
    spatial_length_mm=20., spatial_exponent=2., temporal_exponent=1.,
    low_frequency_hz=.5, alpha_hz=10., alpha_width_hz=2., alpha_power_ratio=1.,
    beta_hz=20., beta_width_hz=3., beta_power_ratio=0.,
    network_fraction=.3, scenario='rest', event_onset_s=.2, event_duration_s=.5,
    erp_latency_s=.1, erp_width_s=.05, erp_amplitude=3., erp_latency_jitter_s=0.,
    ersp_depth=.5, ersp_beta_depth=0., ersp_onset_jitter_s=0.)
ASSET_KEYS = {'basis_path', 'basis_sha256', 'network_path', 'network_sha256',
              'network_indices', 'n_modes'}


def _settings(settings):
    if not isinstance(settings, dict):
        raise ValueError('spectral settings must be a dictionary')
    unknown = set(settings) - set(DEFAULTS) - ASSET_KEYS
    if unknown:
        raise ValueError(f'Unknown spectral settings: {sorted(unknown)}')
    resolved = {**DEFAULTS, **settings}
    if resolved['scenario'] not in ('rest', 'erp', 'ersp'):
        raise ValueError('spectral scenario must be rest, erp or ersp')
    for key in set(DEFAULTS) - {'scenario'}:
        value = resolved[key]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) or not np.isfinite(value):
            raise ValueError(f'{key} must be a finite number')
    for key in ('spatial_length_mm', 'spatial_exponent', 'temporal_exponent',
                'alpha_power_ratio', 'beta_power_ratio', 'erp_amplitude', 'erp_latency_s', 'erp_latency_jitter_s', 'ersp_onset_jitter_s'):
        if resolved[key] < 0:
            raise ValueError(f'{key} must be nonnegative')
    for key in ('low_frequency_hz', 'alpha_hz', 'alpha_width_hz', 'beta_hz', 'beta_width_hz', 'event_duration_s', 'erp_width_s'):
        if resolved[key] <= 0:
            raise ValueError(f'{key} must be positive')
    if not 0 <= resolved['network_fraction'] <= 1:
        raise ValueError('network_fraction must be between zero and one')
    if resolved['ersp_depth'] < -1:
        raise ValueError('ersp_depth must be >= -1 to keep the alpha amplitude nonnegative')
    if resolved['ersp_beta_depth'] < -1:
        raise ValueError('ersp_beta_depth must be >= -1 to keep the beta amplitude nonnegative')
    return resolved


def _fft_process(rng, n_components, n_samples, spectrum):
    """Gaussian stationary draws with unit EXPECTED variance, without sample rescaling."""
    multiplicity = np.full(len(spectrum), 2.)
    multiplicity[0] = 1
    if n_samples % 2 == 0:
        multiplicity[-1] = 1
    expected_variance = float(np.dot(multiplicity, spectrum) / n_samples)
    if not np.isfinite(expected_variance) or expected_variance <= 0:
        raise ValueError('Requested spectrum has no power on the sampling grid')
    white = rng.standard_normal((n_components, n_samples))
    return np.fft.irfft(np.fft.rfft(white, axis=1) * np.sqrt(spectrum / expected_variance),
                        n=n_samples, axis=1)


def generate_coefficients(bundle, settings, *, srate, time_window, baseline_duration_s,
                          rng, network_coefficients=None):
    """Return foreground/background K x T and independent baseline-background K x B.

    Baseline and event occupy disjoint portions of a jointly generated stationary
    process. ERP adds a phase-locked response. ERSP target is the modulation delta
    relative to the unmodulated resting process, retaining random alpha/beta phases.
    """
    resolved = _settings(settings)
    baseline_n = baseline_sample_count(baseline_duration_s, srate)
    window = np.asarray(time_window, dtype=float)
    if window.shape != (2,) or not np.isfinite(window).all() or window[1] <= window[0]:
        raise ValueError('time_window must contain two finite increasing times')
    event_n = baseline_sample_count(float(window[1] - window[0]), srate)
    if resolved['alpha_hz'] >= srate / 2:
        raise ValueError('alpha_hz must lie below the sampling Nyquist frequency')
    if resolved['beta_power_ratio'] > 0 and resolved['beta_hz'] >= srate / 2:
        raise ValueError('Active beta_hz must lie below the sampling Nyquist frequency')
    eigenvalues = np.asarray(bundle['eigenvalues'], dtype=float)
    if eigenvalues.ndim != 1 or not len(eigenvalues) or not np.isfinite(eigenvalues).all() or np.any(eigenvalues < 0):
        raise ValueError('Basis eigenvalues must be a finite nonnegative vector')
    n_modes = resolved.get('n_modes', len(eigenvalues))
    if isinstance(n_modes, (bool, np.bool_)) or not isinstance(n_modes, (int, np.integer)) or not 1 <= n_modes <= len(eigenvalues):
        raise ValueError('n_modes must be an integer within the saved basis size')
    eigenvalues = eigenvalues[:n_modes]
    weights = (1 + resolved['spatial_length_mm'] ** 2 * eigenvalues) ** (-resolved['spatial_exponent'] / 2)
    weights /= np.linalg.norm(weights)

    networks = None
    selected = None
    if network_coefficients is not None:
        networks = np.asarray(network_coefficients, dtype=float)
        if networks.ndim != 2 or networks.shape[0] < n_modes or not networks.shape[1] or not np.isfinite(networks).all():
            raise ValueError('network_coefficients must be finite modes x networks')
        selected = resolved.get('network_indices', list(range(networks.shape[1])))
        indices = np.asarray(selected)
        if (indices.ndim != 1 or not len(indices) or indices.dtype.kind not in 'iu'
                or len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= networks.shape[1])):
            raise ValueError('network_indices must be unique valid integer map indices')
        selected = indices.tolist()
        networks = networks[:n_modes, indices]
        norms = np.linalg.norm(networks, axis=0)
        if np.any(norms <= np.finfo(float).eps):
            raise ValueError('Selected network has zero energy in the retained basis')
        networks = networks / norms[None, :] / np.sqrt(len(indices))
    elif 'network_indices' in settings or ('network_fraction' in settings and settings['network_fraction'] != 0):
        raise ValueError('Network settings require network_coefficients')
    fraction = resolved['network_fraction'] if networks is not None else 0.
    resolved['network_fraction'] = fraction

    n_total = baseline_n + event_n
    freqs = np.fft.rfftfreq(n_total, 1 / srate)
    aperiodic_psd = np.maximum(freqs, resolved['low_frequency_hz']) ** (-resolved['temporal_exponent'])
    alpha_psd = np.exp(-.5 * ((freqs - resolved['alpha_hz']) / resolved['alpha_width_hz']) ** 2)
    aperiodic_psd[0] = alpha_psd[0] = 0

    def draw(spectrum):
        coefficients = np.sqrt(1 - fraction) * weights[:, None] * _fft_process(rng, n_modes, n_total, spectrum)
        if fraction:
            coefficients += np.sqrt(fraction) * (networks @ _fft_process(rng, networks.shape[1], n_total, spectrum))
        return coefficients

    ratio = resolved['alpha_power_ratio']
    beta_ratio = resolved['beta_power_ratio']
    total_power = 1 + ratio + beta_ratio
    aperiodic = draw(aperiodic_psd) / np.sqrt(total_power)
    alpha = np.sqrt(ratio / total_power) * draw(alpha_psd)
    joint = aperiodic + alpha
    beta = None
    if beta_ratio > 0:
        beta_psd = np.exp(-.5 * ((freqs - resolved['beta_hz']) / resolved['beta_width_hz']) ** 2)
        beta_psd[0] = 0
        beta = np.sqrt(beta_ratio / total_power) * draw(beta_psd)
        joint += beta
    background = joint[:, baseline_n:].copy()
    baseline = joint[:, :baseline_n].copy()
    # Subtract only temporal offsets; never independently normalize the segments.
    background -= background.mean(axis=1, keepdims=True)
    baseline -= baseline.mean(axis=1, keepdims=True)
    time = window[0] + np.arange(event_n) / srate
    scenario = resolved['scenario']
    realized = {}
    if scenario == 'rest':
        foreground = background
        background = np.zeros_like(foreground)
        baseline.fill(0)
        target = 'whole_resting_field'
    elif scenario == 'erp':
        if not window[0] <= resolved['event_onset_s'] < window[1]:
            raise ValueError('ERP onset must lie inside time_window')
        latency = resolved['event_onset_s'] + resolved['erp_latency_s'] + rng.normal(0, resolved['erp_latency_jitter_s'])
        if not resolved['event_onset_s'] <= latency < window[1]:
            raise ValueError('Realized ERP latency lies outside time_window; reduce jitter or adjust timing')
        pattern = np.sqrt(1 - fraction) * weights
        if fraction:
            pattern = pattern + np.sqrt(fraction) * networks.sum(axis=1)
        pattern_norm = np.linalg.norm(pattern)
        if pattern_norm <= np.finfo(float).eps:
            raise ValueError('Evoked spatial pattern cancels to zero; adjust the network selection')
        response = resolved['erp_amplitude'] * np.exp(-.5 * ((time - latency) / resolved['erp_width_s']) ** 2)
        # A broad Gaussian must not inject evoked activity into the prestimulus baseline.
        ramp = np.clip((time-resolved['event_onset_s']) / min(.05,resolved['erp_width_s']),0,1)
        response *= np.sin(.5*np.pi*ramp)**2
        foreground = pattern[:, None] / pattern_norm * response[None, :]
        realized['erp_latency_s'] = float(latency)
        realized['erp_support'] = 'causal; smooth onset ramp of min(50ms,width)'
        target = 'evoked_response_over_stationary_rest_background'
    else:
        onset = resolved['event_onset_s'] + rng.normal(0, resolved['ersp_onset_jitter_s'])
        duration = resolved['event_duration_s']
        if onset < window[0] or onset + duration > window[1]:
            raise ValueError('Realized ERSP interval lies outside time_window; reduce jitter or adjust timing')
        phase = (time - onset) / duration
        envelope = np.where((phase >= 0) & (phase <= 1), .5 - .5 * np.cos(2 * np.pi * phase), 0.)
        foreground = resolved['ersp_depth'] * envelope[None, :] * alpha[:, baseline_n:]
        if beta is not None:
            foreground += resolved['ersp_beta_depth'] * envelope[None, :] * beta[:, baseline_n:]
        realized.update(ersp_onset_s=float(onset), ersp_stop_s=float(onset + duration))
        target = 'alpha_amplitude_modulation_delta_over_unmodulated_rest_background'
        if beta is not None:
            target = 'alpha_beta_amplitude_modulation_delta_over_unmodulated_rest_background'
    metadata = dict(spectral_parameters=resolved, n_modes=int(n_modes),
                    mesh_sha256=str(bundle.get('mesh_sha256', 'unspecified')),
                    source_units='model_current_arbitrary', target_definition=target,
                    coefficient_energy='squared coefficient norm equals surface mass-weighted field energy',
                    temporal_model=('independent Gaussian FFT aperiodic and Gaussian alpha/beta PSD'
                                    if beta is not None else 'independent Gaussian FFT aperiodic and Gaussian alpha PSD'),
                    alpha_width_definition='Gaussian PSD standard deviation in Hz',
                    beta_width_definition='Gaussian PSD standard deviation in Hz',
                    network_indices=selected, network_normalization='unit retained modal energy per map',
                    baseline_interval_samples=[0, baseline_n], event_interval_samples=[baseline_n, n_total],
                    baseline_contains_source_background=scenario != 'rest', **realized)
    return dict(foreground=foreground, background=background, baseline_background=baseline, metadata=metadata)


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _asset_signature(path):
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return str(resolved), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _mesh_signature(vertices_mm, faces):
    digest = hashlib.sha256()
    for values in (vertices_mm, faces):
        array = np.ascontiguousarray(values)
        digest.update(array.dtype.str.encode('ascii'))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _check_asset_hashes(resolved, basis_hash, network_hash):
    if resolved.get('basis_sha256') is not None and resolved['basis_sha256'] != basis_hash:
        raise ValueError('Spectral basis SHA256 does not match the recipe')
    if resolved.get('network_sha256') is not None and resolved['network_sha256'] != network_hash:
        raise ValueError('Network asset SHA256 does not match the recipe')


def _load_assets(simulator, resolved, source_model):
    """Keep only the last validated resource pair; never cache generated samples."""
    try:
        from .geometry import load_basis, project_maps
    except ImportError:
        from geometry import load_basis, project_maps

    if not resolved.get('basis_path'):
        raise ValueError('Spectral sources require spectral.basis_path')
    if source_model == 'lbo_network' and not resolved.get('network_path'):
        raise ValueError('lbo_network requires spectral.network_path')
    if source_model != 'lbo_network' and resolved.get('network_path'):
        raise ValueError('network_path requires source_model=lbo_network')
    vertices_mm = simulator.vertices * simulator.coord_to_mm
    basis_signature = _asset_signature(resolved['basis_path'])
    network_signature = _asset_signature(resolved['network_path']) if source_model == 'lbo_network' else None
    key = (basis_signature, network_signature, _mesh_signature(vertices_mm, simulator.faces))
    cached = getattr(simulator, '_spectral_asset_cache', None)
    if cached is not None and cached['key'] == key:
        _check_asset_hashes(resolved, cached['basis_sha256'], cached['network_sha256'])
        return cached

    basis_hash = _file_hash(basis_signature[0])
    network_hash = _file_hash(network_signature[0]) if network_signature is not None else None
    _check_asset_hashes(resolved, basis_hash, network_hash)
    bundle = load_basis(basis_signature[0], vertices_mm=vertices_mm, faces=simulator.faces)
    network_coefficients = None
    if network_signature is not None:
        network_path = network_signature[0]
        with np.load(network_path, allow_pickle=False) as data:
            if 'vertices_brainstorm_mm' not in data or 'faces' not in data or 'source_prior_raw_values' not in data:
                raise ValueError('Network asset must be the signed Brainstorm-aligned source-prior NPZ')
            coordinates = data['vertices_brainstorm_mm'].astype(float)
            if coordinates.shape != vertices_mm.shape or not np.allclose(coordinates, vertices_mm, rtol=0, atol=1e-4):
                raise ValueError('Network vertex rows do not match the forward-model mesh')
            if not np.array_equal(data['faces'], simulator.faces):
                raise ValueError('Network topology does not match the forward-model mesh')
            maps = np.asarray(data['source_prior_weights'], dtype=float)
        network_coefficients = project_maps(bundle, maps)
        network_coefficients.setflags(write=False)
    for values in bundle.values():
        if isinstance(values, np.ndarray):
            values.setflags(write=False)
    cached = dict(key=key, bundle=bundle, network_coefficients=network_coefficients,
                  basis_sha256=basis_hash, network_sha256=network_hash)
    simulator._spectral_asset_cache = cached
    return cached


def generate_source_components(simulator, config):
    """Validate assets against the actual forward-model mesh and project modal draws."""
    resolved = _settings(config.spectral)
    if config.source_snr_db is not None or config.source_background_type is not None:
        raise ValueError('Spectral sources use erp_amplitude/ersp_depth, not legacy source SNR/background settings')
    assets = _load_assets(simulator, resolved, config.source_model)
    bundle = assets['bundle']
    network_coefficients = assets['network_coefficients']
    if config.source_model != 'lbo_network':
        if config.spectral.get('network_fraction', 0) != 0:
            raise ValueError('Nonzero network_fraction requires source_model=lbo_network')
        resolved['network_fraction'] = 0.
    seed = config.component_seeds['waveform'] if config.component_seeds is not None else config.random_seed
    generated = generate_coefficients(bundle, resolved, srate=config.srate,
        time_window=config.time_window, baseline_duration_s=config.baseline_duration_s,
        rng=np.random.default_rng(seed), network_coefficients=network_coefficients)
    basis = np.asarray(bundle['basis'])[:, :generated['metadata']['n_modes']]
    foreground = basis @ generated['foreground']
    foreground -= foreground.mean(axis=1, keepdims=True)
    has_background = resolved['scenario'] != 'rest'
    if has_background:
        background = basis @ generated['background']
        baseline = basis @ generated['baseline_background']
        background -= background.mean(axis=1, keepdims=True)
        source = (foreground + background).astype(np.float32)
    else:
        source = foreground.astype(np.float32)
    source -= source.mean(axis=1, keepdims=True)
    scale = max(float(np.max(np.abs(source))), 1e-12) if config.normalize_source else 1.
    source /= scale
    if has_background:
        foreground = (foreground / scale).astype(np.float32)
        background = source.astype(float) - foreground.astype(float)
        baseline = (baseline / scale).astype(np.float32)
    else:
        background = baseline = None
    metadata = {**asdict(config), **generated['metadata'], 'generator_type': 'G1_spectral',
                'n_vertices': simulator.n_sources, 'n_sensors': simulator.n_sensors,
                'active_vertices': np.flatnonzero(np.any(foreground != 0, axis=1)).tolist(),
                'active_vertices_definition': 'nonzero foreground target, not nuisance background'}
    metadata.update(basis_sha256=assets['basis_sha256'], network_sha256=assets['network_sha256'],
                    spectral_seed=seed, source_normalization_scale=scale)
    return source, metadata, background, baseline
