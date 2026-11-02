"""Offline BioSemi256 template forward for source/sensor nullspace experiments.

Uses the existing fsaverage three-layer BEM without decimation. Sensor noise is
an explicit 1 microvolt IID model, not an empirical or recorded covariance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUBJECT = Path.home() / 'mne_data/MNE-fsaverage-data/fsaverage'
DEFAULT_OUTPUT = ROOT / 'generated_simulation_dataset/nullspace_study/hdeeg_template'


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def farthest_point_indices(points, count):
    """Deterministic Euclidean FPS; ties follow the original source vertex order."""
    points = np.asarray(points, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
            or isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer))
            or not 1 <= count <= len(points)):
        raise ValueError('FPS needs finite N x 3 positions and an integer count in [1,N]')
    selected = np.empty(count, dtype=np.int64)
    distance = np.full(len(points), np.inf)
    next_index = int(np.argmax(np.sum((points - points.mean(axis=0)) ** 2, axis=1)))
    for index in range(count):
        selected[index] = next_index
        distance = np.minimum(distance, np.sum((points - points[next_index]) ** 2, axis=1))
        distance[selected[:index + 1]] = -np.inf
        next_index = int(np.argmax(distance))
    return selected


def select_sources(source_spaces, per_hemisphere):
    """Restrict active vertices while retaining the original geometry and normals."""
    if len(source_spaces) != 2 or any(source['type'] != 'surf' for source in source_spaces):
        raise ValueError('Expected the two fsaverage cortical surface source spaces')
    selected = source_spaces.copy()
    orders = []
    for source in selected:
        original_vertices = np.asarray(source['vertno'])
        order = original_vertices[farthest_point_indices(source['rr'][original_vertices], per_hemisphere)]
        orders.append(order.copy())
        source['vertno'] = np.sort(order)
        source['inuse'] = np.zeros(len(source['rr']), dtype=np.int64)
        source['inuse'][source['vertno']] = 1
        source['nuse'] = len(order)
        # Original sampling triangulation and patch assignments are stale after FPS.
        source['use_tris'] = None
        source['nuse_tri'] = 0
        for key in ('patch_inds', 'pinfo', 'nearest', 'nearest_dist', 'dist', 'dist_limit'):
            source[key] = None
    return selected, orders


def prepare_forward(output_dir=DEFAULT_OUTPUT, subject_dir=DEFAULT_SUBJECT, *, per_hemisphere=256, jobs=2):
    output_dir, subject_dir = Path(output_dir), Path(subject_dir)
    destination = output_dir / 'forward.npz'
    if destination.exists():
        raise FileExistsError(f'Output already exists: {destination}')
    if (isinstance(per_hemisphere, bool) or not isinstance(per_hemisphere, int)
            or not 150 <= per_hemisphere <= 256):
        raise ValueError('per_hemisphere must be an integer from 150 through 256')
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1:
        raise ValueError('jobs must be a positive integer')
    source_path = subject_dir / 'bem/fsaverage-ico-5-src.fif'
    bem_path = subject_dir / 'bem/fsaverage-5120-5120-5120-bem-sol.fif'
    transform_path = subject_dir / 'bem/fsaverage-trans.fif'
    for path in (source_path, bem_path, transform_path):
        if not path.is_file():
            raise FileNotFoundError(f'Offline asset missing; no download is attempted: {path}')
    # Keep joblib scratch local and avoid writing MNE's user-level config lock.
    runtime_cache = output_dir / '.runtime_cache'
    runtime_cache.mkdir(parents=True, exist_ok=True)
    os.environ['MNE_CACHE_DIR'] = str(runtime_cache.resolve())
    os.environ['MNE_MEMMAP_MIN_SIZE'] = '10M'
    os.environ.setdefault('MNE_LOGGING_LEVEL', 'ERROR')
    import mne

    started = time.perf_counter()
    original_sources = mne.read_source_spaces(source_path, verbose='ERROR')
    sources, fps_orders = select_sources(original_sources, per_hemisphere)
    montage = mne.channels.make_standard_montage('biosemi256')
    info = mne.create_info(montage.ch_names, sfreq=256., ch_types='eeg')
    info.set_montage(montage, on_missing='raise')
    print(json.dumps(dict(stage='forward', channels=len(info['ch_names']),
                          selected_sources=sum(source['nuse'] for source in sources))), flush=True)
    forward = mne.make_forward_solution(info, trans=transform_path, src=sources, bem=bem_path,
        eeg=True, meg=False, mindist=5., n_jobs=jobs, verbose='INFO')
    free = mne.convert_forward_solution(forward, surf_ori=False, force_fixed=False,
                                        copy=True, verbose='ERROR')
    fixed = mne.convert_forward_solution(forward, surf_ori=True, force_fixed=True,
                                         use_cps=False, copy=True, verbose='ERROR')
    if int(free['coord_frame']) != 4:
        raise ValueError('Expected the forward and source normals in head coordinates')
    if int(free['nsource']) < 300:
        raise ValueError(f"Only {free['nsource']} sources survived mindist=5 mm; adjust source selection")
    names = list(free['sol']['row_names'])
    if len(names) != 256 or names != info['ch_names']:
        raise ValueError('Forward channel order differs from the 256-channel montage')
    gain_free = np.asarray(free['sol']['data'], dtype=np.float64)
    normals = np.asarray(fixed['source_nn'], dtype=np.float64)
    if not np.allclose(np.linalg.norm(normals, axis=1), 1., atol=1e-6):
        raise ValueError('Cortical source normals are not unit vectors')
    # MNE's fixed converter rounds its gain to float32; retain the precise Cartesian contraction.
    gain_fixed = np.einsum('cnk,nk->cn', gain_free.reshape(len(names), -1, 3), normals)
    fixed_error = np.linalg.norm(gain_fixed-fixed['sol']['data'], axis=1) / np.linalg.norm(gain_fixed, axis=1)
    if np.max(fixed_error) > 1e-6:
        raise ValueError('Cartesian gain dotted with normals disagrees with MNE fixed forward')
    projector = np.eye(len(names)) - np.ones((len(names), len(names))) / len(names)
    channel_std = np.full(len(names), 1e-6)
    noise_cov = np.diag(channel_std ** 2)
    montage_path = Path(mne.__file__).parent / 'channels/data/montages/biosemi256.txt'
    assets = [source_path, bem_path, transform_path, montage_path]
    metadata = dict(
        evidence_kind='computed_template_forward_with_modelled_noise', source_subject='fsaverage',
        source_space='FPS subset of cached ico5 cortical vertices; original surface normals',
        coordinate_frame='head', spatial_units='m', source_current_units='A m',
        channel_unit_by_type={'eeg': 'V'}, channel_counts={'eeg': 256}, mne_version=mne.__version__,
        source_selection='deterministic Euclidean farthest-point sampling independently per hemisphere',
        requested_per_hemisphere=per_hemisphere, requested_sources=2*per_hemisphere,
        n_sources=int(free['nsource']), retained_per_hemisphere=[len(s['vertno']) for s in free['src']],
        mindist_mm=5., bem='unchanged cached three-layer 5120-5120-5120 triangle BEM solution',
        bem_decimated=False, montage='MNE standard biosemi256 template; not subject digitization',
        fixed_orientation='surface normal, use_cps=False; float64 Cartesian projection',
        fixed_conversion_max_row_relative_error=float(np.max(fixed_error)),
        covariance_method='modelled IID sensor noise, standard deviation 1 microvolt',
        covariance_empirical=False, covariance_nfree=None, covariance_reference='before average reference',
        covariance_export='diagonal 1e-12 V^2; no measured baseline or empty-room recording',
        ssp_application='Apply exported P to G and P C P.T before whitening; P is EEG average reference',
        ssp_independent_vectors=1, sss_applied_by_exporter=False, sss_history='none; simulated template EEG',
        source_assets=[dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=file_hash(path)) for path in assets],
        builder_sha256=file_hash(__file__), elapsed_seconds=time.perf_counter()-started)
    retained_rank = []
    for source, order in zip(free['src'], fps_orders):
        rank = {int(vertex): index for index, vertex in enumerate(order)}
        retained_rank.extend(rank[int(vertex)] for vertex in source['vertno'])
    arrays = dict(leadfield_fixed=gain_fixed, leadfield_free=gain_free,
        source_rr_m=np.asarray(free['source_rr'], dtype=np.float64), source_normals=normals,
        source_hemi=np.concatenate([np.full(len(s['vertno']), hemi) for s, hemi in zip(free['src'], ('lh','rh'))]),
        source_vertex_ids=np.concatenate([s['vertno'] for s in free['src']]),
        source_fps_rank_within_hemisphere=np.asarray(retained_rank, dtype=np.int64),
        channel_names=np.asarray(names), channel_types=np.full(len(names),'eeg'),
        channel_positions_head_m=np.asarray([ch['loc'][:3] for ch in info['chs']]),
        channel_std=channel_std, noise_cov=noise_cov, sensor_projector=projector,
        sensor_projector_basis=np.ones((len(names),1))/np.sqrt(len(names)),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)))
    for key, array in arrays.items():
        if array.dtype.hasobject or (array.dtype.kind in 'fci' and not np.isfinite(array).all()):
            raise ValueError(f'Invalid output array: {key}')
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.partial.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(destination)
    (output_dir/'forward_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(destination), channels=len(names), sources=int(free['nsource']),
                          elapsed_seconds=metadata['elapsed_seconds'])),flush=True)
    return arrays, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--subject-dir', type=Path, default=DEFAULT_SUBJECT)
    parser.add_argument('--per-hemisphere', type=int, default=256)
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args(argv)
    prepare_forward(args.output_dir,args.subject_dir,per_hemisphere=args.per_hemisphere,jobs=args.jobs)


if __name__ == '__main__':
    main()
