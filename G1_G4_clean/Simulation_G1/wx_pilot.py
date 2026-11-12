"""Experimental spatially constrained WX model; production defaults are untouched."""
import numpy as np

from .geometry import _validate_bundle


def build_weights(bundle, network_maps, cortical_mask, *, seed=0, length_mm=20.,
                  n_networks=3, local_per_network=2):
    """Build area-normalized smooth shared/local columns, retaining signed maps.

    Localization uses a soft map envelope, not a thresholded LBO eigenvector.
    Unresolved energy is relative to the retained nonconstant basis, not a
    measurement of all high spatial frequencies.
    """
    basis, mass, values = _validate_bundle(bundle)
    maps = np.asarray(network_maps, dtype=float)
    mask = np.asarray(cortical_mask)
    if (maps.ndim != 2 or maps.shape[0] != len(mass) or not np.isfinite(maps).all()
            or mask.shape != mass.shape or mask.dtype.kind != 'b' or not mask.any()):
        raise ValueError('Finite vertex-by-network maps and a nonempty Boolean mask required')
    if (isinstance(length_mm, bool) or not np.isfinite(length_mm) or length_mm <= 0
            or isinstance(n_networks, bool) or not isinstance(n_networks, (int, np.integer))
            or not 1 <= n_networks <= maps.shape[1]
            or isinstance(local_per_network, bool) or not isinstance(local_per_network, (int, np.integer))
            or not 1 <= local_per_network <= 8):
        raise ValueError('Invalid length, network count or local component count')
    rng = np.random.default_rng(seed)
    attenuation = np.exp(-.5 * length_mm**2 * values)
    labels = np.asarray(bundle.get('component_labels', np.zeros(len(mass), dtype=int)))
    def smooth(data):
        result = basis @ (attenuation[:, None] * (basis.T @ (mass[:, None] * data)))
        # The cached LBO omits hemisphere constants; restore them when smoothing maps.
        for label in np.unique(labels):
            rows = labels == label
            result[rows] += np.average(data[rows], weights=mass[rows], axis=0)
        return result
    shared = smooth(maps[:, :n_networks])
    columns, roles, groups = [], [], []
    roi = None
    for group in range(n_networks):
        raw = np.abs(maps[:, group]) * mask
        peak = raw.max()
        if peak <= 0:
            raise ValueError('Selected network has no cortical support')
        envelope = np.clip(smooth((raw / peak)[:, None])[:, 0], 0, 1) * mask
        if group == 0:
            roi = mask & (envelope >= .5 * envelope.max())
        base = shared[:, group] * envelope
        columns.append(base)
        roles.append('shared')
        groups.append(group)
        patterns = basis @ (attenuation[:, None] * rng.normal(size=(len(values), local_per_network)))
        for pattern in patterns.T:
            rms = np.sqrt(np.average(pattern[mask]**2, weights=mass[mask]))
            local = base * (1 + .65 * np.tanh(pattern / max(rms, 1e-15)))
            columns.append(local)
            roles.append('local')
            groups.append(group)
    W = np.column_stack(columns)
    norm = np.sqrt(np.sum(mass[:, None] * W**2, axis=0))
    if np.any(norm <= 1e-14):
        raise ValueError('Degenerate spatial component')
    W /= norm
    coeff = basis.T @ (mass[:, None] * W)
    residual = W - basis @ coeff
    for label in np.unique(labels):
        rows = labels == label
        residual[rows] -= np.average(W[rows], weights=mass[rows], axis=0)
    return dict(W=W, roles=roles, groups=np.asarray(groups), roi=roi, mass=mass,
                metadata=dict(length_mm=float(length_mm), spatial_seed=int(seed),
                    n_networks=int(n_networks), local_per_network=int(local_per_network),
                    unresolved_energy=np.sum(mass[:, None]*residual**2, axis=0).tolist(),
                    retained_modal_roughness=np.sum(values[:, None]*coeff**2, axis=0).tolist(),
                    roughness_scope='retained modes only; final mesh roughness must be measured separately',
                    roi_definition='primary smoothed absolute map >= half maximum',
                    weight_normalization='unit surface-area-weighted column energy'))


def _power(W, X, mass):
    gram = W.T @ (mass[:, None] * W)
    return float(np.sum(X * (gram @ X)) / X.shape[1] / mass.sum())


def generate_timecourses(weights, *, scenario, seed, srate=256, duration_s=2,
                         baseline_s=2, variant='wx', rho=.5):
    """Return reproducible component arrays; foreground + background is total.

    ERP backgrounds are identical across variants. For rest the whole field is
    the target, so baseline has no source field (sensor nuisance is added later).
    ERSP foreground is modulation delta; it may be negative during suppression.
    """
    if scenario not in {'rest', 'erp', 'ersp'} or variant not in {'synchronous', 'independent', 'wx'}:
        raise ValueError('Invalid scenario or variant')
    if any(isinstance(v, bool) or not np.isfinite(v) or v <= 0 for v in (srate, duration_s, baseline_s)):
        raise ValueError('Rate and durations must be finite and positive')
    if isinstance(rho, bool) or not np.isfinite(rho) or not 0 <= rho <= 1 or srate <= 50:
        raise ValueError('rho must be in [0,1] and sampling rate above 50 Hz')
    n, b = int(round(duration_s*srate)), int(round(baseline_s*srate))
    if (min(n, b) < 4 or not np.isclose(n, duration_s*srate, rtol=0, atol=1e-7)
            or not np.isclose(b, baseline_s*srate, rtol=0, atol=1e-7)):
        raise ValueError('Durations must resolve to integer sample counts >=4')
    W, mass, roi = (np.asarray(weights[key]) for key in ('W', 'mass', 'roi'))
    groups = np.asarray(weights['groups'])
    if (W.ndim != 2 or min(W.shape) < 1 or mass.shape != (W.shape[0],) or groups.shape != (W.shape[1],)
            or roi.shape != mass.shape or roi.dtype.kind != 'b' or not roi.any()
            or not np.isfinite(W).all() or not np.isfinite(mass).all() or np.any(mass <= 0)
            or groups.dtype.kind not in 'iu' or groups.min() < 0):
        raise ValueError('Invalid weight bundle')
    k, total = W.shape[1], n+b
    time = np.arange(n)/srate
    freqs = np.fft.rfftfreq(total, 1/srate)
    streams = np.random.SeedSequence(seed).spawn(3)
    rng = np.random.default_rng(streams[0])
    alpha_rng = np.random.default_rng(streams[1])
    event_rng = np.random.default_rng(streams[2])
    # Fixed ERP background isolates foreground structure in paired comparisons.
    process_variant = 'independent' if scenario == 'erp' else variant
    heterogeneous = process_variant == 'wx'
    exponents = 1.2 + (.15*np.sin(np.arange(k)) if heterogeneous else np.zeros(k))
    alpha_hz = 10.5 + (.8*np.cos(np.arange(k)) if heterogeneous else np.zeros(k))
    shared_white = rng.normal(size=(int(groups.max())+1, total))
    shared_alpha = alpha_rng.normal(size=shared_white.shape)
    rest, alpha = np.empty((k,total)), np.empty((k,total))
    for j in range(k):
        if process_variant == 'synchronous':
            white, narrow = shared_white[0], shared_alpha[0]
        elif process_variant == 'independent':
            white, narrow = rng.normal(size=total), alpha_rng.normal(size=total)
        else:
            white = np.sqrt(rho)*shared_white[groups[j]] + np.sqrt(1-rho)*rng.normal(size=total)
            narrow = np.sqrt(rho)*shared_alpha[groups[j]] + np.sqrt(1-rho)*alpha_rng.normal(size=total)
        def filter_draw(draw, psd):
            psd[0] = 0
            multiplicity = np.full(len(psd), 2.)
            multiplicity[0] = 1
            if total % 2 == 0:
                multiplicity[-1] = 1
            expected = np.dot(multiplicity, psd)/total
            return np.fft.irfft(np.fft.rfft(draw)*np.sqrt(psd/expected), n=total)
        aperiodic = filter_draw(white, np.maximum(freqs,.5)**(-exponents[j]))
        alpha[j] = filter_draw(narrow, np.exp(-.5*((freqs-alpha_hz[j])/1.5)**2))
        rest[j] = (aperiodic + np.sqrt(.5)*alpha[j])/np.sqrt(1.5*k)
    background, baseline = rest[:, b:].copy(), rest[:, :b].copy()
    foreground = np.zeros_like(background)
    metadata = dict(scenario=scenario, variant=variant, seed=int(seed), rho=float(rho),
                    alpha_hz=alpha_hz.tolist(), temporal_exponent=exponents.tolist(),
                    baseline_source_semantics='unmodulated background, disjoint continuous-process segment',
                    centering='no per-window source centering; causal ERP remains zero before onset')
    if scenario == 'rest':
        foreground, background = background, foreground
        baseline.fill(0)
        metadata.update(target='whole_resting_field', baseline_source_semantics='zero: rest is target, not nuisance')
    elif scenario == 'erp':
        if duration_s <= 1.2:
            raise ValueError('ERP pilot requires duration_s >1.2')
        jitter = float(event_rng.normal(0,.015))
        latencies = .35 + jitter + (.2*(np.arange(k)%3) if variant == 'wx' else np.zeros(k))
        widths = .06 + (.015*(np.arange(k)%3) if variant == 'wx' else np.zeros(k))
        gains = np.where(groups == 0, 1., .2)
        polarity = np.where(np.arange(k)%3 == 1, -.7, 1.) if variant == 'wx' else np.ones(k)
        ramp = np.sin(.5*np.pi*np.clip((time-.2)/.05,0,1))**2
        foreground = gains[:,None]*polarity[:,None]*np.exp(-.5*((time[None,:]-latencies[:,None])/widths[:,None])**2)*ramp
        active = (time >= .2) & (time <= 1.2)
        before = _power(W[roi], foreground[:,active], mass[roi])
        other = _power(W[roi], background[:,active], mass[roi])
        gain = np.sqrt(2*other / before)
        foreground *= gain
        metadata.update(target='all primary and secondary event-related response',
            latencies_s=latencies.tolist(), widths_s=widths.tolist(), erp_gain=float(gain),
            roi_component_power_ratio=_power(W[roi], foreground[:,active], mass[roi])/other,
            roi_cross_power=2*np.mean(np.sum(foreground[:,active]*(W[roi].T@(mass[roi,None]*W[roi])@background[:,active]),axis=0))/mass[roi].sum())
    else:
        if duration_s <= 1.5:
            raise ValueError('ERSP pilot requires duration_s >1.5')
        onsets = .3 + (.08*(np.arange(k)%3) if variant == 'wx' else np.zeros(k))
        phase = (time[None,:]-onsets[:,None])/.8
        envelope = np.where((phase>=0)&(phase<=1), .5-.5*np.cos(2*np.pi*phase),0)
        depths = np.where(np.arange(k)%2, -.5, .8) if variant == 'wx' else np.full(k,.8)
        foreground = depths[:,None]*envelope*alpha[:,b:]*np.sqrt(.5/(1.5*k))
        metadata.update(target='alpha modulation delta over unmodulated rest',
            onsets_s=onsets.tolist(), modulation_depths=depths.tolist(),
            min_alpha_amplitude_multiplier=float(np.min(1+depths[:,None]*envelope)))
    return dict(X_foreground=foreground, X_background=background,
                X_baseline=baseline, time=time, metadata=metadata)
