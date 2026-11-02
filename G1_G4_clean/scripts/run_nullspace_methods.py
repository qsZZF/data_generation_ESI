"""Held-out inverse-method and anatomical-support pilot on saved forward assets."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resolution_benchmark.inverse_methods import fit_inverse, sbl_active_operator
from resolution_benchmark.nullspace import analyze_operator, projector_from_patterns, unit_safe_whitener
from resolution_benchmark.recovery_metrics import score_recovery, support_diagnostics
from scripts.run_nullspace_study import covariance_factor, source_order, sensor_subsets

FS = 256
SAMPLES = 512


def normalize_columns(values):
    norms = np.linalg.norm(values, axis=0)
    if np.any(norms <= 0):
        raise ValueError('Dictionary has an empty atom')
    return values / norms


def temporal_sources(count, rng):
    frequencies = np.fft.rfftfreq(SAMPLES, 1 / FS)
    power = np.maximum(frequencies, 0.5) ** -1.5
    power += 0.5 * np.exp(-0.5 * ((frequencies - 10.5) / 1.5) ** 2)
    power[0] = 0
    spectrum = np.fft.rfft(rng.normal(size=(count, SAMPLES)), axis=1)
    values = np.fft.irfft(spectrum * np.sqrt(power), n=SAMPLES, axis=1)
    return values / np.std(values, axis=1, keepdims=True)


def dictionaries(bases, order):
    modes = bases['raw_modes'][order]
    eigenvalues = bases['eigenvalues']
    lbo = np.linalg.qr(modes, mode='reduced')[0]
    distances = bases['geodesic_distances_m'][np.ix_(order, order)]
    centers = np.arange(32)  # source ordering is a geometry-only balanced FPS
    patch = normalize_columns((distances[:, centers] <= .025).astype(float))
    green = normalize_columns((modes / (1 + .020 ** 2 * eigenvalues)) @ modes[centers].T)
    full_green = normalize_columns(bases['green_full_20mm'][np.ix_(order, order)])
    local = np.column_stack([normalize_columns(np.exp(-.5 * (distances / radius) ** 2))
                             for radius in (.012, .024, .048)])
    return dict(points=np.eye(len(order)), patch32=patch, green32=green,
                lbo32=lbo[:, :32], patch_green64=np.column_stack((patch, green)),
                hybrid64=np.column_stack((lbo[:, :32], green)),
                green_full400=full_green, local_multiscale1200=local,
                local_green1600=np.column_stack((local, full_green))), lbo, distances


def support_truth(scene, lbo, distances, seed, realization):
    spatial_rng = np.random.default_rng(seed + 1000)
    rng = np.random.default_rng(seed + 10000 + realization * 100000)
    count = len(distances)
    centers = spatial_rng.choice(np.arange(32, count), size=3, replace=False)
    patch = normalize_columns(np.exp(-0.5 * (distances[:, centers] / .018) ** 2))
    if scene in ('lbo_matched', 'lbo_mismatch'):
        modes = 20 if scene == 'lbo_matched' else 56
        coefficients = temporal_sources(modes, rng) / np.sqrt(1 + np.arange(modes))[:, None]
        return lbo[:, :modes] @ coefficients
    if scene == 'local_patch':
        return patch @ temporal_sources(3, rng)
    if scene == 'mixed':
        alpha = temporal_sources(12, rng)
        frequencies = np.fft.rfftfreq(SAMPLES, 1 / FS)
        alpha = np.fft.irfft(np.fft.rfft(alpha, axis=1) * ((frequencies >= 8) & (frequencies < 13)), n=SAMPLES, axis=1)
        time_axis = np.arange(SAMPLES) / FS
        transient = np.stack([np.exp(-.5 * ((time_axis - t) / .09) ** 2) for t in (.5, 1., 1.5)])
        transient -= transient.mean(axis=1, keepdims=True)
        local = patch @ transient
        global_field = lbo[:, :12] @ alpha
        return local / np.linalg.norm(local) + global_field / np.linalg.norm(global_field)
    if scene == 'propagating':
        waves = temporal_sources(3, rng)
        spectrum = np.fft.rfft(waves, axis=1)
        frequencies = np.fft.rfftfreq(SAMPLES, 1 / FS)
        result = np.zeros((count, SAMPLES))
        for k, center in enumerate(centers):
            distance = distances[:, center]
            delay = np.where(np.isfinite(distance), distance / .5, 0.)
            shifted = np.fft.irfft(spectrum[k] * np.exp(-2j * np.pi * delay[:, None] * frequencies), n=SAMPLES, axis=1)
            result += patch[:, k, None] * shifted
        return result
    raise ValueError(scene)


def point_truth(scene, count, seed, realization):
    spatial_rng = np.random.default_rng(seed + 2000)
    rng = np.random.default_rng(seed + 20000 + realization * 100000)
    active = spatial_rng.choice(100, size=4, replace=False) if scene != 'distributed' else np.arange(100)
    waves = temporal_sources(len(active), rng)
    if scene == 'synchronous':
        shared = temporal_sources(1, rng)
        waves = np.sqrt(.95) * shared + np.sqrt(.05) * waves
    truth = np.zeros((count, SAMPLES))
    truth[active] = waves
    return truth


def band_components(values):
    frequencies = np.fft.rfftfreq(values.shape[1], 1 / FS)
    spectrum = np.fft.rfft(values, axis=1)
    weights = np.full(len(frequencies), 2.)
    weights[0] = weights[-1] = 1.
    for lower, upper in ((0, 8), (8, 13), (13, 30), (30, np.inf)):
        mask = (frequencies >= lower) & (frequencies < upper)
        fraction = float(weights[mask].sum() / values.shape[1])
        yield np.fft.irfft(spectrum * mask, n=values.shape[1], axis=1), fraction


def load_case(directory, kind):
    folder = directory / ('mne_sample' if kind == 'meg' else 'hdeeg_template')
    path = folder / ('mne_sample_forward.npz' if kind == 'meg' else 'forward.npz')
    with np.load(path) as archive:
        data = dict(archive)
    with np.load(folder / 'bases.npz') as archive:
        bases = dict(archive)
    np.testing.assert_array_equal(bases['source_vertex_ids'], data['source_vertex_ids'])
    mask = np.isin(data['channel_types'], ['mag', 'grad']) if kind == 'meg' else data['channel_types'] == 'eeg'
    channels = np.flatnonzero(mask)
    order = source_order(data['source_rr_m'], data['source_hemi'])[:400]
    patterns = data['sensor_projector_basis'][channels]
    projection = projector_from_patterns(patterns)
    covariance = data['noise_cov'][np.ix_(channels, channels)]
    covariance = projection @ covariance @ projection.T
    choices = sensor_subsets(data['channel_names'][channels], data['channel_positions_head_m'][channels], kind, 41)
    selected = [choices[0], min(choices, key=lambda a: abs(len(a) - 150)), min(choices, key=lambda a: abs(len(a) - 66))]
    support, lbo, distances = dictionaries(bases, order)
    return dict(gain=data['leadfield_fixed'][channels][:, order], covariance=covariance,
                projection=projection, patterns=patterns, selections=selected,
                dictionaries=support, lbo=lbo, distances=distances, order=order,
                asset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                bases_sha256=hashlib.sha256((folder / 'bases.npz').read_bytes()).hexdigest())


def observations(case, truth_train, truth_test, seed, snr):
    rng = np.random.default_rng(seed + 50000)
    factor = covariance_factor(case['covariance'])
    baseline = factor @ rng.normal(size=(factor.shape[1], FS * 2))
    noise_train = factor @ rng.normal(size=(factor.shape[1], SAMPLES))
    noise_test = factor @ rng.normal(size=(factor.shape[1], SAMPLES))
    baseline_cov = np.cov(baseline)
    p = case['projection']
    whitener, _ = unit_safe_whitener(p @ baseline_cov @ p.T)
    clean = p @ case['gain'] @ truth_train
    scaling = np.sqrt(10 ** (snr / 10) * np.mean((whitener @ noise_train) ** 2) / np.mean((whitener @ clean) ** 2))
    truth_train, truth_test = scaling * truth_train, scaling * truth_test
    return dict(truth_train=truth_train, truth_test=truth_test, baseline_cov=baseline_cov,
                train=case['gain'] @ truth_train + noise_train,
                test=case['gain'] @ truth_test + noise_test, noise_test=noise_test)


def sensor_problem(case, data, selected, count, covariance_kind='baseline'):
    p = projector_from_patterns(case['patterns'][selected])
    covariance_source = data['baseline_cov'] if covariance_kind == 'baseline' else case['covariance']
    covariance = p @ covariance_source[np.ix_(selected, selected)] @ p.T
    analysis = analyze_operator(p @ case['gain'][selected, :count], covariance)
    w = analysis['whitener']
    test_noise = w @ p @ data['noise_test'][selected]
    test_clean = analysis['operator'] @ data['truth_test'][:count]
    analysis['test_snr_db'] = float(10 * np.log10(np.sum(test_clean ** 2) / np.sum(test_noise ** 2)))
    return analysis, w @ p @ data['train'][selected], w @ p @ data['test'][selected]


def fit_records(analysis, training, test, truth, basis, methods, max_iter):
    operator = analysis['operator'] @ basis
    output = []
    sbl = None
    sbl_seconds = 0.
    for method in methods:
        started = time.perf_counter()
        active_count = None
        if method in ('sbl_lcmv', 'sbl_lcmv_active'):
            inverse = sbl['model_lcmv_operator']
            if method == 'sbl_lcmv_active':
                inverse, active_count = sbl_active_operator(sbl)
            diagnostics = sbl['diagnostics']
        else:
            fit = fit_inverse(operator, training, method, max_iter=max_iter)
            inverse, diagnostics = fit['operator'], fit['diagnostics']
            if method == 'sbl':
                sbl = fit
                sbl_seconds = time.perf_counter() - started
        estimate = basis @ inverse @ test
        record = dict(method=method, fit_seconds=time.perf_counter() - started + (sbl_seconds if method.startswith('sbl_lcmv') else 0.),
                      converged=diagnostics.get('converged'), iterations=diagnostics.get('iterations'),
                      active_atoms=active_count)
        record.update(score_recovery(estimate, truth, analysis, physical_units=method != 'sloreta'))
        record['test_sensor_residual'] = float(np.linalg.norm(analysis['operator'] @ estimate - test) / np.linalg.norm(test)) if method != 'sloreta' else None
        output.append(record)
    return output


def fit_bands(analysis, training, test, truth, basis, max_iter):
    estimates = {method: np.zeros_like(truth) for method in ('sbl', 'sbl_lcmv')}
    diagnostics = []
    started = time.perf_counter()
    for (train_band, fraction), (test_band, test_fraction) in zip(band_components(training), band_components(test)):
        assert fraction == test_fraction
        scale = np.sqrt(fraction)
        fit = fit_inverse(analysis['operator'] @ basis / scale, train_band / scale, 'sbl', max_iter=max_iter)
        estimates['sbl'] += basis @ fit['operator'] @ (test_band / scale)
        estimates['sbl_lcmv'] += basis @ fit['model_lcmv_operator'] @ (test_band / scale)
        diagnostics.append(fit['diagnostics'])
    records = []
    for method, estimate in estimates.items():
        record = dict(method=method, fit_seconds=time.perf_counter() - started, active_atoms=None,
                      converged=all(d['converged'] for d in diagnostics),
                      iterations=max(d['iterations'] for d in diagnostics))
        record.update(score_recovery(estimate, truth, analysis))
        record['test_sensor_residual'] = float(np.linalg.norm(analysis['operator'] @ estimate - test) / np.linalg.norm(test))
        records.append(record)
    return records


def run(case, kind, part, seeds, max_iter, directory, covariance_kind='baseline', snrs=(10., 20.)):
    records = []
    scenes = ('sparse', 'synchronous', 'distributed') if part == 'methods' else ('local_patch', 'lbo_matched', 'lbo_mismatch', 'mixed', 'propagating')
    started = time.perf_counter()
    for seed in range(seeds):
        for scene in scenes:
            generator = (lambda r: point_truth(scene, 400, seed, r)) if part == 'methods' else (lambda r: support_truth(scene, case['lbo'], case['distances'], seed, r))
            train_truth, test_truth = generator(0), generator(1)
            for snr in snrs:
                data = observations(case, train_truth, test_truth, seed, snr)
                settings = [(case['selections'][0], n) for n in (100, 200, 400)] + [(s, 400) for s in case['selections'][1:]] if part == 'methods' else [(s, 400) for s in (case['selections'][0], case['selections'][-1])]
                for selected, count in settings:
                    analysis, training, test = sensor_problem(case, data, selected, count, covariance_kind)
                    truth = data['truth_test'][:count]
                    common = dict(modality=kind, part=part, seed=seed, scene=scene, nominal_full_array_snr_db=snr,
                                  n_channels=len(selected), n_sources=count, rank=analysis['info']['rank'],
                                  nullity=analysis['info']['nullity'], effective_rank=analysis['info']['effective_rank'],
                                  covariance=covariance_kind, test_snr_db=analysis['test_snr_db'])
                    supports = {'points': np.eye(count)} if part == 'methods' else case['dictionaries']
                    for name, basis in supports.items():
                        methods = ('mne', 'wmne', 'sloreta', 'lcmv', 'sbl', 'sbl_lcmv', 'sbl_lcmv_active') if part == 'methods' else ('mne', 'sbl', 'sbl_lcmv', 'sbl_lcmv_active')
                        details = support_diagnostics(analysis['operator'], basis, truth)
                        fitted = fit_records(analysis, training, test, truth, basis, methods, max_iter)
                        records.extend(dict(common, support=name, **details, **row) for row in fitted)
                    if part == 'supports':
                        basis = supports['hybrid64']
                        details = support_diagnostics(analysis['operator'], basis, truth)
                        fitted = fit_bands(analysis, training, test, truth, basis, max_iter)
                        records.extend(dict(common, support='hybrid64_bands', **details, **row) for row in fitted)
                print(json.dumps(dict(case=kind, part=part, seed=seed, scene=scene, snr=snr,
                                      rows=len(records), seconds=round(time.perf_counter() - started, 1))), flush=True)
    destination = directory / f'{kind}_{part}_{covariance_kind}.csv'
    with destination.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return len(records)


def fit_key(row):
    return tuple(str(row[key]) for key in ('modality', 'part', 'covariance', 'seed', 'scene',
                 'nominal_full_array_snr_db', 'n_channels', 'n_sources', 'support'))


def refine_unconverged(directory, max_iter):
    """Refit flagged training problems only; retain paired method rows and audit changes."""
    output = directory / 'methods_supports'
    audit = []
    cases = {}
    for path in sorted(output.glob('*_*.csv')):
        if not path.name.endswith(('_baseline.csv', '_reference.csv')):
            continue
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        pending = [row.copy() for row in rows if row['method'] == 'sbl' and row['converged'] == 'False']
        for row in pending:
            kind, scene, seed = row['modality'], row['scene'], int(row['seed'])
            if kind not in cases:
                cases[kind] = load_case(directory, kind)
            case = cases[kind]
            generator = (lambda r: point_truth(scene, 400, seed, r)) if row['part'] == 'methods' else (lambda r: support_truth(scene, case['lbo'], case['distances'], seed, r))
            data = observations(case, generator(0), generator(1), seed, float(row['nominal_full_array_snr_db']))
            selected = next(s for s in case['selections'] if len(s) == int(row['n_channels']))
            count = int(row['n_sources'])
            analysis, training, test = sensor_problem(case, data, selected, count, row['covariance'])
            truth = data['truth_test'][:count]
            if row['support'] == 'hybrid64_bands':
                fitted = fit_bands(analysis, training, test, truth, case['dictionaries']['hybrid64'], max_iter)
            else:
                basis = np.eye(count) if row['part'] == 'methods' else case['dictionaries'][row['support']]
                fitted = fit_records(analysis, training, test, truth, basis,
                                     ('sbl', 'sbl_lcmv', 'sbl_lcmv_active'), max_iter)
            replacements = {item['method']: item for item in fitted}
            for original in rows:
                if fit_key(original) == fit_key(row) and original['method'] in replacements:
                    original.update(replacements[original['method']])
            audit.append(dict(file=path.name, before=row, after=replacements['sbl'], max_iter=max_iter))
        if pending:
            temporary = path.with_suffix('.partial.csv')
            with temporary.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            temporary.replace(path)
    report = dict(refitted_groups=len(audit), fits=audit,
                  remaining=sum(not item['after']['converged'] for item in audit),
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output / 'convergence_refinement.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'fits'}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('meg', 'hdeeg', 'all'), default='all')
    parser.add_argument('--part', choices=('methods', 'supports', 'all'), default='all')
    parser.add_argument('--seeds', type=int, default=2)
    parser.add_argument('--max-iter', type=int, default=150)
    parser.add_argument('--covariance', choices=('baseline', 'reference'), default='baseline')
    parser.add_argument('--snrs', type=float, nargs='+', default=[10., 20.])
    parser.add_argument('--refine-unconverged', action='store_true')
    parser.add_argument('--directory', type=Path, default=ROOT / 'generated_simulation_dataset/nullspace_study')
    args = parser.parse_args()
    if args.seeds < 1 or args.max_iter < 1:
        parser.error('seeds and max-iter must be positive')
    if args.refine_unconverged:
        refine_unconverged(args.directory, args.max_iter)
        return
    output = args.directory / 'methods_supports'
    output.mkdir(parents=True, exist_ok=True)
    for kind in ('meg', 'hdeeg'):
        if args.case not in (kind, 'all'):
            continue
        case = load_case(args.directory, kind)
        for part in ('methods', 'supports'):
            if args.part not in (part, 'all'):
                continue
            rows = run(case, kind, part, args.seeds, args.max_iter, output, args.covariance, args.snrs)
            protocol = dict(case=kind, part=part, seeds=args.seeds, rows=rows, max_iter=args.max_iter,
                            fs_hz=FS, train_samples=SAMPLES, test_samples=SAMPLES, baseline_s=2,
                            covariance=args.covariance, snrs_db=args.snrs,
                            forward_sha256=case['asset_sha256'], bases_sha256=case['bases_sha256'],
                            source_metric='Euclidean dipole coefficients',
                            hyperparameters='Fixed in advance; no test-truth selection',
                            bands_hz=[[0, 8], [8, 13], [13, 30], [30, 128]],
                            code_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                (Path(__file__), ROOT / 'resolution_benchmark/inverse_methods.py', ROOT / 'resolution_benchmark/recovery_metrics.py')})
            (output / f'{kind}_{part}_{args.covariance}_protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')


if __name__ == '__main__':
    main()
