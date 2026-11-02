"""Export verified G4 fitted-model draws into the standard regional NMM bank.

The exports are model-generated PSP, not recovered human source truth. All
repetitions of one parameter fit retain the same simulation group and supplied
measured-data ancestry. Historical artifacts without ancestry are rejected.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile

import numpy as np

from Simulation_G1.sample_io import _publish_directory
from Simulation_G2.nmm_bank import initialize_bank, projection_hash, save_bank_sample
from .atlas import load_atlas
from .simulator import PARAMETER_NAMES, drive_config


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def _strings(value, field):
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f'G4 export requires nonempty {field}; legacy missing ancestry cannot be inferred')
    if len(set(value)) != len(value):
        raise ValueError(f'G4 {field} contains duplicate identities')
    return value


def _group_plan(run_dir, group, run, atlas, expected_projection):
    relative = Path(group.get('file', ''))
    artifact = (run_dir / relative).resolve()
    source_dir = (run_dir / 'reconstructed_sources').resolve()
    if (artifact.parent != source_dir or re.fullmatch(r'group_[0-9]+\.npz', artifact.name) is None):
        raise ValueError('G4 artifact path must identify a local reconstructed_sources/group_*.npz')
    sidecar_encoded = artifact.with_suffix('.json').read_bytes()
    sidecar = json.loads(sidecar_encoded)
    sidecar_hash = hashlib.sha256(sidecar_encoded).hexdigest()
    encoded = artifact.read_bytes()
    artifact_hash = hashlib.sha256(encoded).hexdigest()
    if artifact_hash != sidecar.get('artifact_sha256'):
        raise ValueError('G4 reconstruction artifact checksum mismatch')
    if sidecar.get('result') != group:
        raise ValueError('G4 reconstruction report differs from the completed group sidecar')
    with np.load(io.BytesIO(encoded), allow_pickle=False) as data:
        fingerprint = str(data['fingerprint'])
        if fingerprint != sidecar.get('fingerprint') or re.fullmatch(r'[a-f0-9]{64}', fingerprint) is None:
            raise ValueError('G4 reconstruction fingerprint differs from its sidecar')
        metadata = json.loads(str(data['metadata_json']))
        for key in ('dataset', 'subject', 'condition'):
            if not isinstance(metadata.get(key), str) or not metadata[key]:
                raise ValueError(f'G4 artifact lacks trustworthy {key}')
        if metadata['dataset'] != run['dataset'] or any(metadata[key] != group.get(key) for key in ('subject', 'condition')):
            raise ValueError('G4 dataset/subject/condition identity mismatch')
        expected = dict(atlas_id=atlas['manifest']['atlas_id'],
                        atlas_bundle_sha256=atlas['manifest']['bundle_sha256'],
                        projection_sha256=expected_projection,
                        connectivity_sha256=run['bank_config']['connectivity_sha256'], source_unit='mV_PSP')
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f'G4 artifact {key} differs from the validated run/atlas')
        parents = _strings(metadata.get('parent_data_group_ids'), 'parent_data_group_ids')
        fitting_recordings = _strings(metadata.get('fitting_recording_ids'), 'fitting_recording_ids')
        evaluation_recordings = _strings(metadata.get('evaluation_recording_ids'), 'evaluation_recording_ids')
        fitting = _strings(data['fitting_window_ids'].tolist(), 'fitting_window_ids')
        evaluation = _strings(data['evaluation_window_ids'].tolist(), 'evaluation_window_ids')
        if set(fitting) & set(evaluation):
            raise ValueError('G4 fitting and evaluation windows overlap')
        if len(fitting) != group.get('fitting_windows') or len(evaluation) != group.get('evaluation_windows'):
            raise ValueError('G4 window counts differ from the reconstruction report')
        if not np.array_equal(data['region_names'], atlas['region_names']):
            raise ValueError('G4 reconstructed region order differs from the atlas')
        sources = data['region_source_mV']
        n_regions = len(atlas['region_names'])
        if (sources.dtype.kind not in 'fiu' or sources.ndim != 3 or sources.shape[0] < 1
                or sources.shape[1] != n_regions or sources.shape[2] < 2 or not np.isfinite(sources).all()):
            raise ValueError('G4 source must be finite repetitions x regions x time')
        times = np.asarray(data['time_s'], dtype=float)
        rate = run.get('sampling_rate_hz')
        if (not isinstance(rate, (int, float)) or isinstance(rate, bool) or not np.isfinite(rate) or rate <= 0
                or times.shape != (sources.shape[2],) or not np.isfinite(times).all()
                or not np.allclose(times, np.arange(sources.shape[2]) / rate, rtol=1e-8, atol=1e-10)):
            raise ValueError('G4 source clock differs from the run sampling rate')
        parameters = {}
        for name in (*PARAMETER_NAMES, 'delay_ms'):
            values = np.asarray(data[name], dtype=float)
            shape = () if name == 'delay_ms' else (n_regions,)
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError(f'Invalid G4 fitted parameter {name}')
            parameters[name] = values.tolist()
        simulations = metadata.get('simulations')
        if not isinstance(simulations, list) or len(simulations) != len(sources):
            raise ValueError('G4 repetitions lack per-draw simulation provenance')
        expected_drive = drive_config(run['bank_config'].get('drive_mode', 'shared'), run['bank_config'].get('drive_correlation'))
        seeds = []
        for simulation in simulations:
            if (simulation.get('model') != 'paper_dual_timescale_jr' or simulation.get('source_unit') != 'mV_PSP'
                    or simulation.get('sampling_rate_hz') != rate
                    or type(simulation.get('seed')) is not int or simulation['seed'] < 0):
                raise ValueError('G4 simulation model/unit/rate/seed metadata is incomplete or inconsistent')
            if drive_config(simulation.get('drive_mode', 'shared'), simulation.get('drive_correlation')) != expected_drive:
                raise ValueError('G4 simulation drive differs from fitted bank configuration')
            seeds.append(simulation['seed'])
        if len(set(seeds)) != len(seeds):
            raise ValueError('G4 stochastic repetitions must have distinct recorded seeds')
    fit_identity = dict(dataset=metadata['dataset'], subject=metadata['subject'], condition=metadata['condition'],
        atlas=expected, parameters=parameters, fitting_window_ids=fitting,
        fitting_recording_ids=fitting_recordings, parent_data_group_ids=sorted(parents), **expected_drive)
    return dict(artifact=artifact, artifact_sha256=artifact_hash, sidecar_sha256=sidecar_hash, fingerprint=fingerprint,
        metadata=metadata, n_repeats=len(simulations), n_samples=len(times), rate=rate,
        parameters=parameters, fitting_window_ids=fitting, evaluation_window_ids=evaluation,
        fitting_recording_ids=fitting_recordings, evaluation_recording_ids=evaluation_recordings,
        simulation_group_id=_hash_json(fit_identity), **expected_drive)


def export_reconstructions(run_dir, output_dir, *, atlas_dir, sample_prefix=None):
    """Export every completed reported G4 group to a fresh schema-2 dense bank.

    The atlas loader validates the actual projection/bundle. SC identity is
    cross-checked between artifact and run bank metadata; no SC is downloaded.
    Publication is exclusive and atomic. Existing outputs are never overwritten.
    """
    run_dir, output = Path(run_dir).resolve(), Path(output_dir).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'G4 export requires a new output directory: {output}')
    prefix = 'g4' if sample_prefix is None else sample_prefix
    if not isinstance(prefix, str) or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,69}', prefix) is None:
        raise ValueError('sample_prefix must be a safe identifier of at most 70 characters')
    atlas = load_atlas(atlas_dir)
    projection_digest = projection_hash(atlas['projection'])
    run_path, report_path = run_dir / 'run_manifest.json', run_dir / 'reconstruction_report.json'
    run_hash, report_hash = _sha(run_path), _sha(report_path)
    run = json.loads(run_path.read_text(encoding='utf-8'))
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if not run.get('dataset') or report.get('dataset') != run['dataset']:
        raise ValueError('G4 run/report dataset identity mismatch')
    bank = run.get('bank_config', {})
    if (bank.get('atlas_id') != atlas['manifest']['atlas_id']
            or bank.get('atlas_hash') != atlas['manifest']['bundle_sha256']
            or run.get('atlas_manifest', {}).get('bundle_sha256') != atlas['manifest']['bundle_sha256']
            or re.fullmatch(r'[a-f0-9]{64}', str(bank.get('connectivity_sha256', ''))) is None):
        raise ValueError('G4 fitted bank atlas or structural-connectivity identity is missing/mismatched')
    groups = [group for group in report.get('groups', []) if group.get('status') == 'forward_checked']
    if not groups or len({group.get('file') for group in groups}) != len(groups):
        raise ValueError('G4 report must contain distinct completed reconstruction groups')
    plans = [_group_plan(run_dir, group, run, atlas, projection_digest) for group in groups]
    subjects = {}
    for plan in plans:
        metadata = plan['metadata']
        key = (metadata['dataset'], metadata['subject'])
        parents = set(metadata['parent_data_group_ids'])
        subjects[key] = subjects[key] & parents if key in subjects else parents
        if not subjects[key]:
            raise ValueError('G4 conditions from the same subject lack shared measured-data ancestry')
    summary = dict(schema='g4-nmm-export-v1', dataset=run['dataset'], n_groups=len(plans),
        n_records=sum(plan['n_repeats'] for plan in plans), atlas_id=atlas['manifest']['atlas_id'],
        atlas_bundle_sha256=atlas['manifest']['bundle_sha256'], projection_sha256=projection_digest,
        connectivity_sha256=bank['connectivity_sha256'], source_unit='mV_PSP',
        run_manifest_sha256=run_hash, reconstruction_report_sha256=report_hash,
        activation_truth='not_available; full data-guided network trajectories',
        physiological_parameter_recovery_validated=False, records=[])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.g4_export_', dir=output.parent) as temporary:
        staging = Path(temporary) / 'bank'
        initialize_bank(staging, atlas_id=summary['atlas_id'], n_regions=len(atlas['region_names']),
                        projection_sha256=projection_digest, schema_version=2)
        for group_index, plan in enumerate(plans):
            encoded = plan['artifact'].read_bytes()
            if hashlib.sha256(encoded).hexdigest() != plan['artifact_sha256']:
                raise ValueError('G4 artifact checksum changed during export')
            if _sha(plan['artifact'].with_suffix('.json')) != plan['sidecar_sha256']:
                raise ValueError('G4 group sidecar checksum changed during export')
            with np.load(io.BytesIO(encoded), allow_pickle=False) as artifact:
                sources = artifact['region_source_mV']
                times_ms = artifact['time_s'] * 1000.
                for repeat in range(plan['n_repeats']):
                    sample_id = f'{prefix}_{group_index:04d}_{repeat:02d}'
                    simulation = plan['metadata']['simulations'][repeat]
                    metadata = plan['metadata']
                    manifest = dict(signal_type='data_guided_jr', output_srate_hz=plan['rate'],
                        n_regions_output=len(atlas['region_names']), normalize=False, source_unit='mV_PSP',
                        active_regions=[], region_names=atlas['region_names'].tolist(), random_seed=simulation['seed'],
                        model={'class': 'DualTimescaleJansenRit', 'readout': 'mean dual-branch pyramidal PSP',
                               'parameters': plan['parameters'], 'connectivity_sha256': bank['connectivity_sha256']},
                        dataset=metadata['dataset'], subject=metadata['subject'], condition=metadata['condition'],
                        parent_data_group_ids=metadata['parent_data_group_ids'],
                        parameter_status='model_conditioned_estimate', human_source_truth_available=False,
                        configured_regions_are_activation_truth=False,
                        drive_mode=plan['drive_mode'], drive_correlation=plan['drive_correlation'])
                    provenance = dict(producer='Simulation_G4.export_nmm', run_dir=str(run_dir),
                        source_artifact=str(plan['artifact']), artifact_sha256=plan['artifact_sha256'],
                        sidecar_sha256=plan['sidecar_sha256'],
                        reconstruction_fingerprint=plan['fingerprint'], repetition_index=repeat,
                        run_manifest_sha256=run_hash, reconstruction_report_sha256=report_hash,
                        fitting_window_ids=plan['fitting_window_ids'], evaluation_window_ids=plan['evaluation_window_ids'],
                        fitting_recording_ids=plan['fitting_recording_ids'], evaluation_recording_ids=plan['evaluation_recording_ids'],
                        atlas_bundle_sha256=summary['atlas_bundle_sha256'], simulation=simulation,
                        parent_data_group_ids=metadata['parent_data_group_ids'])
                    entry = save_bank_sample(staging, sample_id, sources[repeat], times_ms, manifest, provenance,
                        storage={'codec': 'dense'}, simulation_group_id=plan['simulation_group_id'])
                    summary['records'].append(dict(sample_id=sample_id, simulation_group_id=entry['simulation_group_id'],
                        source_sha256=entry['source_sha256'], subject=metadata['subject'], condition=metadata['condition']))
        if _sha(run_path) != run_hash or _sha(report_path) != report_hash:
            raise ValueError('G4 run/report changed during export')
        (staging / 'g4_export.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        _publish_directory(staging, output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--atlas-dir', type=Path, required=True)
    parser.add_argument('--sample-prefix')
    args = parser.parse_args()
    print(json.dumps(export_reconstructions(args.run_dir, args.output_dir,
        atlas_dir=args.atlas_dir, sample_prefix=args.sample_prefix), indent=2))


if __name__ == '__main__':
    main()
