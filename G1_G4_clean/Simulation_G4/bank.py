"""Resumable synthetic source bank shared across dataset-specific EEG montages."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from Simulation_G1.noise_covariance import baseline_sample_count, sensor_noise_with_baseline
from Simulation_G1.real_noise import NoiseWindowReader, load_noise_registry

from .simulator import PARAMETER_NAMES, drive_config, sample_parameters, simulate


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()


def _validate_sample(data, config, index, path):
    """Check cached draw identity and targets before either reuse or projection."""
    required = ("source", "draw_seed", "noise_seed", *PARAMETER_NAMES, "delay_ms")
    if not all(name in data for name in required):
        raise ValueError(f"Missing cached source, seeds or parameter targets: {path}")
    draw_seed, noise_seed = np.random.SeedSequence([config["seed"], index]).generate_state(2)
    for name, expected in (("draw_seed", draw_seed), ("noise_seed", noise_seed)):
        if data[name].shape != () or data[name] != expected:
            raise ValueError(f"Cached {name} does not match draw {index}: {path}")
    expected_parameters = sample_parameters(np.random.default_rng(int(draw_seed)), config["n_regions"])
    for name, expected in expected_parameters.items():
        if not np.array_equal(data[name], expected):
            raise ValueError(f"Cached parameter {name} does not match draw {index}: {path}")
    source = data["source"]
    shape = (config["n_regions"], round(config["duration_s"] * config["sampling_rate_hz"]))
    if source.shape != shape:
        raise ValueError(f"Invalid cached source shape: {path}")
    if not np.isfinite(source).all():
        raise ValueError(f"Nonfinite cached source: {path}")
    expected_drive = drive_config(config.get("drive_mode", "shared"), config.get("drive_correlation"))
    metadata = json.loads(str(data["metadata_json"])) if "metadata_json" in data else {}
    cached_drive = drive_config(metadata.get("drive_mode", "shared"), metadata.get("drive_correlation"))
    if cached_drive != expected_drive:
        raise ValueError(f"Cached drive configuration differs: {path}")
    return source


def _generate_one(job):
    directory, index, connectivity, config = job
    path = Path(directory) / f"sample_{index:06d}.npz"
    draw_seed, noise_seed = np.random.SeedSequence([config["seed"], index]).generate_state(2)
    parameters = sample_parameters(np.random.default_rng(int(draw_seed)), len(connectivity))
    source, times, metadata = simulate(
        parameters, connectivity, duration_s=config["duration_s"],
        warmup_s=config["warmup_s"], sampling_rate_hz=config["sampling_rate_hz"],
        dt_ms=config["dt_ms"], seed=int(noise_seed),
        **drive_config(config.get("drive_mode", "shared"), config.get("drive_correlation")),
    )
    metadata.update(drive_config(config.get("drive_mode", "shared"), config.get("drive_correlation")))
    temporary = path.with_suffix(".part")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, source=source.astype(np.float32), time_s=times,
                            **parameters, draw_seed=int(draw_seed), noise_seed=int(noise_seed),
                            metadata_json=json.dumps(metadata))
    temporary.replace(path)
    return index


def generate_bank(connectivity, directory, *, n_draws=2048, duration_s=10,
                  sampling_rate_hz=256, warmup_s=1, dt_ms=0.5, seed=20261007,
                  workers=1, atlas_id, atlas_hash, drive_mode="independent", drive_correlation=None):
    """New banks use independent inputs; shared mode is an explicit legacy option.

    Config/hash mismatches block reuse; changing input statistics requires a new
    bank and inverse training, not relabelling already generated trajectories.
    """
    drive_settings = drive_config(drive_mode, drive_correlation)
    matrix = np.asarray(connectivity, dtype=float)
    if workers < 1 or n_draws < 10:
        raise ValueError("At least ten draws and one worker are required")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = dict(n_draws=int(n_draws), duration_s=float(duration_s),
                  sampling_rate_hz=float(sampling_rate_hz), warmup_s=float(warmup_s),
                  dt_ms=float(dt_ms), seed=int(seed), n_regions=len(matrix),
                  atlas_id=atlas_id, atlas_hash=atlas_hash, connectivity_sha256=array_hash(matrix),
                  simulator_sha256=hashlib.sha256(Path(__file__).with_name("simulator.py").read_bytes()).hexdigest(),
                  **drive_settings)
    config_path = directory / "config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        existing.update(drive_config(existing.get("drive_mode", "shared"), existing.get("drive_correlation")))
        if existing != config:
            raise ValueError("Existing bank configuration differs; use a new output directory")
    if not config_path.exists():
        if any(directory.glob("sample_*")):
            raise ValueError("Orphan sample cache has no config.json; use a new output directory")
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    pending = []
    for index in range(n_draws):
        path = directory / f"sample_{index:06d}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                _validate_sample(data, config, index, path)
        else:
            pending.append((str(directory), index, matrix, config))
    print(json.dumps({"stage": "bank", "existing": n_draws - len(pending), "pending": len(pending)}), flush=True)
    if workers == 1:
        results = map(_generate_one, pending)
        for count, index in enumerate(results, 1):
            if count % 20 == 0 or count == len(pending):
                print(json.dumps({"stage": "bank", "generated": count, "last_index": index}), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for count, index in enumerate(executor.map(_generate_one, pending, chunksize=1), 1):
                if count % 20 == 0 or count == len(pending):
                    print(json.dumps({"stage": "bank", "generated": count, "last_index": index}), flush=True)
    # Split draw identities once; all channel/noise/duration views inherit them.
    order = np.random.default_rng(seed + 1).permutation(n_draws)
    fit_stop, tune_stop, calibrate_stop = [int(n_draws * fraction) for fraction in (0.7, 0.8, 0.9)]
    splits = {"train": order[:fit_stop].tolist(), "validation": order[fit_stop:tune_stop].tolist(),
              "calibration": order[tune_stop:calibrate_stop].tolist(), "test": order[calibrate_stop:].tolist()}
    (directory / "splits.json").write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")
    return config, splits


def _training_noise_plan(registry_path, mapping, splits, n_draws, channels, rate,
                         baseline_samples, event_samples, excluded_subject_ids):
    """Validate all donor/fold metadata before source projection or signal loading."""
    folds = {'train': 'train', 'validation': 'val', 'calibration': 'val', 'test': 'test'}
    if not isinstance(mapping, dict) or set(mapping) != set(folds):
        raise ValueError('Noise donors must cover train/validation/calibration/test exactly')
    if not isinstance(splits, dict) or set(splits) != set(folds):
        raise ValueError('G4 draw splits must form the four-fold partition')
    draw_folds = [None] * n_draws
    for fold, indices in splits.items():
        if not isinstance(indices, list):
            raise ValueError('G4 draw splits must form a complete partition')
        for index in indices:
            if type(index) is not int or not 0 <= index < n_draws or draw_folds[index] is not None:
                raise ValueError('G4 draw splits must form a disjoint complete partition')
            draw_folds[index] = fold
    if any(fold is None for fold in draw_folds):
        raise ValueError('G4 draw splits must form a complete partition')
    if channels is None or isinstance(channels, str) or not len(channels) or len(set(channels)) != len(channels):
        raise ValueError('Registered training noise requires unique explicit channel_names')
    channels = list(channels)
    registry_path = Path(registry_path).resolve()
    registry_hash = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    registry = load_noise_registry(registry_path)
    if hashlib.sha256(registry_path.read_bytes()).hexdigest() != registry_hash:
        raise ValueError('Noise registry changed during preflight')
    records = {record['record_id']: record for record in registry['records']}
    owners, canonical, donors = {}, {}, {}
    for fold, split in folds.items():
        selected = mapping[fold]
        if (not isinstance(selected, list) or not selected or any(not isinstance(value, str) for value in selected)
                or len(set(selected)) != len(selected)):
            raise ValueError('Each G4 fold requires nonempty distinct donor record IDs')
        canonical[fold] = sorted(selected)
        for record_id in canonical[fold]:
            if record_id not in records:
                raise ValueError(f'Unknown registered noise donor: {record_id}')
            record = records[record_id]
            if record['split'] != split:
                raise ValueError(f'Noise donor {record_id} registry split differs from {fold}')
            if (record['fs_hz'] != rate or record['reference'] != 'average'
                    or record['sensor_units'] != ['V'] * len(channels)
                    or set(record['channel_names']) != set(channels)
                    or record['shape'][0] != len(channels)
                    or record['shape'][1] < baseline_samples + event_samples):
                raise ValueError('Noise donor rate/reference/units/channel/window mismatch; prepare an explicit common EEG view')
            if record.get('subject_id') in excluded_subject_ids:
                raise ValueError('Noise donor subject overlaps a measured forward-evaluation subject')
            for parent in record['parent_data_group_ids']:
                if parent in owners and owners[parent] != fold:
                    raise ValueError('Noise donor ancestry crosses G4 folds, including validation/calibration')
                owners[parent] = fold
            donors[record_id] = {key: record[key] for key in ('subject_id', 'recording_id', 'noise_role',
                'parent_data_group_ids', 'file_sha256', 'numeric_sha256', 'split')}
    info = dict(kind='registered_measured_noise', registry_path=str(registry_path), registry_sha256=registry_hash,
        noise_records_by_split=canonical, donors=donors, channel_names=channels, native_sensor_units='V',
        sampling_rate_hz=rate, reference='average', baseline_samples=baseline_samples, event_samples=event_samples,
        calibration='global event SNR in relative projected-model units; same gain on baseline and event',
        subject_overlap_check='exact dataset-qualified subject IDs; raw/missing identities are not guessed')
    return info, draw_folds


def project_bank(directory, leadfield, n_times, *, sensor_snr_db=None, seed=0,
                 return_noise=False, return_baseline=False, baseline_duration_s=2.,
                 real_noise_registry=None, noise_records_by_split=None, channel_names=None,
                 excluded_noise_subject_ids=(), draw_indices=None):
    """Build a montage view, optionally returning the actual added sensor noise.

    The optional third return shares the EEG's draw/channel/time axes. Keeping
    the realization allows downstream filters to precede covariance estimation.
    ``return_baseline=True`` adds a fourth return containing ``baseline_sensor``
    and per-draw ``metadata``. Its independent noise uses the event's same gain.
    """
    from .inverse import pack_parameters

    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    n = config["n_draws"]
    selected_draws = np.arange(n) if draw_indices is None else np.asarray(draw_indices)
    if (selected_draws.ndim != 1 or selected_draws.dtype.kind not in 'iu' or not len(selected_draws)
            or len(np.unique(selected_draws)) != len(selected_draws)
            or np.any(selected_draws < 0) or np.any(selected_draws >= n)):
        raise ValueError('draw_indices must contain unique valid integer draw IDs')
    selected_draws = selected_draws.tolist()
    leadfield = np.asarray(leadfield, dtype=float)
    if leadfield.ndim != 2 or leadfield.shape[1] != config["n_regions"]:
        raise ValueError("Leadfield and source bank region counts differ")
    if n_times < 32 or n_times > round(config["duration_s"] * config["sampling_rate_hz"]):
        raise ValueError("Requested EEG window does not fit the simulated source bank")
    if sensor_snr_db is not None and not np.isfinite(sensor_snr_db):
        raise ValueError("Sensor SNR must be finite or None")
    if (real_noise_registry is None) != (noise_records_by_split is None):
        raise ValueError('real_noise_registry and noise_records_by_split must be supplied together')
    noise_plan = None
    if real_noise_registry is not None:
        if sensor_snr_db is None:
            raise ValueError('Registered G4 noise requires a finite sensor_snr_db')
        if channel_names is None or len(channel_names) != leadfield.shape[0]:
            raise ValueError('Registered training noise channel_names must match leadfield rows')
        baseline_samples = baseline_sample_count(baseline_duration_s, config['sampling_rate_hz'])
        noise_plan, draw_folds = _training_noise_plan(real_noise_registry, noise_records_by_split,
            json.loads((directory / 'splits.json').read_text()), n, channel_names,
            config['sampling_rate_hz'], baseline_samples, n_times, set(excluded_noise_subject_ids))
        reader = NoiseWindowReader()
        read_options = dict(expected_srate=config['sampling_rate_hz'], expected_channels=list(channel_names),
            expected_reference='average', expected_units='V', baseline_samples=baseline_samples, event_samples=n_times)
        for record_id, donor in noise_plan['donors'].items():
            _, provenance = reader.read(real_noise_registry, record_id, seed=0, split=donor['split'], **read_options)
            if provenance['registry_sha256'] != noise_plan['registry_sha256']:
                raise ValueError('Noise registry changed while validating donor files')
        from Simulation_G1.g1_simulation_core import G1SourceSimulator
        noise_backend = object.__new__(G1SourceSimulator)
    eeg = np.empty((len(selected_draws), leadfield.shape[0], n_times), dtype=np.float32)
    sensor_noise = np.zeros(eeg.shape, dtype=np.float64) if return_noise or return_baseline else None
    if return_baseline:
        baseline_samples = baseline_sample_count(baseline_duration_s, config["sampling_rate_hz"])
        baseline = np.zeros((len(selected_draws), leadfield.shape[0], baseline_samples), dtype=np.float64)
        baseline_metadata = []
    targets = []
    for row_index, index in enumerate(selected_draws):
        path = directory / f"sample_{index:06d}.npz"
        with np.load(path, allow_pickle=False) as data:
            source = _validate_sample(data, config, index, path)
            projected = leadfield @ source[:, :n_times]
            projected -= projected.mean(axis=0, keepdims=True)
            projected -= projected.mean(axis=1, keepdims=True)
            gain = 0.
            if noise_plan is not None:
                fold = draw_folds[index]
                chooser = np.random.default_rng(np.random.SeedSequence([seed, index, 37]))
                choices = noise_plan['noise_records_by_split'][fold]
                record_id = choices[int(chooser.integers(len(choices)))]
                crop_seed = int(np.random.SeedSequence([seed, index, 38]).generate_state(1)[0])
                real, provenance = reader.read(real_noise_registry, record_id, seed=crop_seed,
                    split=noise_plan['donors'][record_id]['split'], **read_options)
                if provenance['registry_sha256'] != noise_plan['registry_sha256']:
                    raise ValueError('Noise registry changed during training projection')
                noise, real_baseline, noise_info = sensor_noise_with_baseline(noise_backend, projected,
                    srate=config['sampling_rate_hz'], sensor_noise_type='real', snr_db=sensor_snr_db,
                    seed=crop_seed, baseline_samples=baseline_samples, average_reference=True,
                    snr_mode='global_rms', real_sensor_noise=real)
                raw_event = real[:, baseline_samples:] - real[:, baseline_samples:].mean(axis=1, keepdims=True)
                raw_event -= raw_event.mean(axis=0, keepdims=True)
                noise_info['registered_input_to_output_gain'] = float(np.sqrt(np.mean(noise ** 2) / np.mean(raw_event ** 2)))
                noise_info['registered_gain_definition'] = 'same scalar on centered/referenced registered event and baseline; output uses relative projected-model units'
                projected += noise
                if return_noise or return_baseline:
                    sensor_noise[row_index] = noise
                if return_baseline:
                    baseline[row_index] = real_baseline
                    baseline_metadata.append(dict(noise_info, real_noise_provenance=provenance,
                        g4_fold=fold, draw_index=index, noise_role=provenance['noise_role'],
                        duration_s=float(baseline_duration_s), sampling_rate_hz=config['sampling_rate_hz'],
                        n_samples=baseline_samples, target_present=False, status='generated'))
            elif sensor_snr_db is not None:
                rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
                noise = rng.standard_normal(projected.shape)
                noise -= noise.mean(axis=0, keepdims=True)
                noise -= noise.mean(axis=1, keepdims=True)
                gain = float(np.sqrt(np.mean(projected**2) / (10**(sensor_snr_db / 10) * np.mean(noise**2))))
                noise *= gain
                projected += noise
                if return_noise or return_baseline:
                    sensor_noise[row_index] = noise
            if return_baseline and noise_plan is None:
                baseline_seed = [int(seed), index, 1]
                if sensor_snr_db is not None:
                    values = np.random.default_rng(np.random.SeedSequence(baseline_seed)).standard_normal(
                        (leadfield.shape[0], baseline_samples))
                    values -= values.mean(axis=0, keepdims=True)
                    values -= values.mean(axis=1, keepdims=True)
                    baseline[row_index] = values * gain
                baseline_metadata.append(dict(seed_sequence=baseline_seed, draw_index=index,
                    event_seed_sequence=[int(seed), index], generation="independent_iid_noise_shared_gain",
                    duration_s=float(baseline_duration_s), sampling_rate_hz=config["sampling_rate_hz"],
                    n_samples=baseline_samples, shared_noise_gain=gain, scale_calibration="event_only",
                    target_present=False, status="uninformative_no_added_noise" if sensor_snr_db is None else "generated"))
            eeg[row_index] = projected
            targets.append(pack_parameters({name: data[name] for name in (*PARAMETER_NAMES, "delay_ms")}))
    if not np.isfinite(eeg).all():
        raise FloatingPointError("Nonfinite projected training EEG")
    if noise_plan is not None and hashlib.sha256(Path(real_noise_registry).read_bytes()).hexdigest() != noise_plan['registry_sha256']:
        raise ValueError('Noise registry changed during training projection')
    result = (eeg, np.asarray(targets, dtype=np.float32))
    if return_baseline:
        baseline_result = dict(baseline_sensor=baseline, metadata=baseline_metadata)
        if noise_plan is not None:
            baseline_result['noise_plan'] = noise_plan
        return (*result, sensor_noise, baseline_result)
    return (*result, sensor_noise) if return_noise else result


def main():
    from .atlas import load_atlas
    from .resources import load_cortical_connectivity

    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=root / "generated_simulation_dataset/G4/aal78_membership_bank_256hz_v1")
    parser.add_argument("--atlas-dir", type=Path, default=root / "Anatomy/G4_AAL_cortical_membership")
    parser.add_argument("--connectivity-dir", type=Path, default=root / "Anatomy/G4_resources")
    parser.add_argument("--draws", type=int, default=2048)
    parser.add_argument("--duration-s", type=float, default=10)
    parser.add_argument("--sampling-rate", type=float, default=256)
    parser.add_argument("--warmup-s", type=float, default=1)
    parser.add_argument("--dt-ms", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--drive-mode", choices=["independent", "shared", "mixed"], default="independent",
                        help="New ESI banks default to independent regional inputs; shared reproduces legacy drive")
    parser.add_argument("--drive-correlation", type=float, default=None,
                        help="Mixed input correlation in [0,1] (default 0.15); independent=0, shared=1")
    args = parser.parse_args()
    atlas = load_atlas(args.atlas_dir)
    connectivity = load_cortical_connectivity(args.connectivity_dir, atlas["aal_roi_numbers"])
    generate_bank(connectivity, args.output_dir, n_draws=args.draws, duration_s=args.duration_s,
                  sampling_rate_hz=args.sampling_rate, warmup_s=args.warmup_s, dt_ms=args.dt_ms,
                  seed=args.seed, workers=args.workers, atlas_id=atlas["manifest"]["atlas_id"],
                  atlas_hash=atlas["manifest"]["bundle_sha256"],
                  drive_mode=args.drive_mode, drive_correlation=args.drive_correlation)


if __name__ == "__main__":
    main()
