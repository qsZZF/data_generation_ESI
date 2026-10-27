"""Resumable synthetic source bank shared across dataset-specific EEG montages."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from Simulation_G1.noise_covariance import baseline_sample_count

from .simulator import PARAMETER_NAMES, sample_parameters, simulate


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
    )
    temporary = path.with_suffix(".part")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, source=source.astype(np.float32), time_s=times,
                            **parameters, draw_seed=int(draw_seed), noise_seed=int(noise_seed),
                            metadata_json=json.dumps(metadata))
    temporary.replace(path)
    return index


def generate_bank(connectivity, directory, *, n_draws=2048, duration_s=10,
                  sampling_rate_hz=256, warmup_s=1, dt_ms=0.5, seed=20261007,
                  workers=1, atlas_id, atlas_hash):
    """Generate deterministic per-draw files; config/hash mismatches block reuse."""
    matrix = np.asarray(connectivity, dtype=float)
    if workers < 1 or n_draws < 10:
        raise ValueError("At least ten draws and one worker are required")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = dict(n_draws=int(n_draws), duration_s=float(duration_s),
                  sampling_rate_hz=float(sampling_rate_hz), warmup_s=float(warmup_s),
                  dt_ms=float(dt_ms), seed=int(seed), n_regions=len(matrix),
                  atlas_id=atlas_id, atlas_hash=atlas_hash, connectivity_sha256=array_hash(matrix),
                  simulator_sha256=hashlib.sha256(Path(__file__).with_name("simulator.py").read_bytes()).hexdigest())
    config_path = directory / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
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


def project_bank(directory, leadfield, n_times, *, sensor_snr_db=None, seed=0,
                 return_noise=False, return_baseline=False, baseline_duration_s=1.):
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
    leadfield = np.asarray(leadfield, dtype=float)
    if leadfield.ndim != 2 or leadfield.shape[1] != config["n_regions"]:
        raise ValueError("Leadfield and source bank region counts differ")
    if n_times < 32 or n_times > round(config["duration_s"] * config["sampling_rate_hz"]):
        raise ValueError("Requested EEG window does not fit the simulated source bank")
    if sensor_snr_db is not None and not np.isfinite(sensor_snr_db):
        raise ValueError("Sensor SNR must be finite or None")
    eeg = np.empty((n, leadfield.shape[0], n_times), dtype=np.float32)
    sensor_noise = np.zeros(eeg.shape, dtype=np.float64) if return_noise or return_baseline else None
    if return_baseline:
        baseline_samples = baseline_sample_count(baseline_duration_s, config["sampling_rate_hz"])
        baseline = np.zeros((n, leadfield.shape[0], baseline_samples), dtype=np.float64)
        baseline_metadata = []
    targets = []
    for index in range(n):
        path = directory / f"sample_{index:06d}.npz"
        with np.load(path, allow_pickle=False) as data:
            source = _validate_sample(data, config, index, path)
            projected = leadfield @ source[:, :n_times]
            projected -= projected.mean(axis=0, keepdims=True)
            projected -= projected.mean(axis=1, keepdims=True)
            gain = 0.
            if sensor_snr_db is not None:
                rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
                noise = rng.standard_normal(projected.shape)
                noise -= noise.mean(axis=0, keepdims=True)
                noise -= noise.mean(axis=1, keepdims=True)
                gain = float(np.sqrt(np.mean(projected**2) / (10**(sensor_snr_db / 10) * np.mean(noise**2))))
                noise *= gain
                projected += noise
                if return_noise or return_baseline:
                    sensor_noise[index] = noise
            if return_baseline:
                baseline_seed = [int(seed), index, 1]
                if sensor_snr_db is not None:
                    values = np.random.default_rng(np.random.SeedSequence(baseline_seed)).standard_normal(
                        (leadfield.shape[0], baseline_samples))
                    values -= values.mean(axis=0, keepdims=True)
                    values -= values.mean(axis=1, keepdims=True)
                    baseline[index] = values * gain
                baseline_metadata.append(dict(seed_sequence=baseline_seed,
                    event_seed_sequence=[int(seed), index], generation="independent_iid_noise_shared_gain",
                    duration_s=float(baseline_duration_s), sampling_rate_hz=config["sampling_rate_hz"],
                    n_samples=baseline_samples, shared_noise_gain=gain, scale_calibration="event_only",
                    target_present=False, status="uninformative_no_added_noise" if sensor_snr_db is None else "generated"))
            eeg[index] = projected
            targets.append(pack_parameters({name: data[name] for name in (*PARAMETER_NAMES, "delay_ms")}))
    if not np.isfinite(eeg).all():
        raise FloatingPointError("Nonfinite projected training EEG")
    result = (eeg, np.asarray(targets, dtype=np.float32))
    if return_baseline:
        return (*result, sensor_noise, dict(baseline_sensor=baseline, metadata=baseline_metadata))
    return (*result, sensor_noise) if return_noise else result


def main():
    from .atlas import load_atlas
    from .resources import load_cortical_connectivity

    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=root / "generated_simulation_dataset/G4/aal78_bank_256hz_v1")
    parser.add_argument("--atlas-dir", type=Path, default=root / "Anatomy/G4_AAL_cortical")
    parser.add_argument("--connectivity-dir", type=Path, default=root / "Anatomy/G4_resources")
    parser.add_argument("--draws", type=int, default=2048)
    parser.add_argument("--duration-s", type=float, default=10)
    parser.add_argument("--sampling-rate", type=float, default=256)
    parser.add_argument("--warmup-s", type=float, default=1)
    parser.add_argument("--dt-ms", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    atlas = load_atlas(args.atlas_dir)
    connectivity = load_cortical_connectivity(args.connectivity_dir, atlas["aal_roi_numbers"])
    generate_bank(connectivity, args.output_dir, n_draws=args.draws, duration_s=args.duration_s,
                  sampling_rate_hz=args.sampling_rate, warmup_s=args.warmup_s, dt_ms=args.dt_ms,
                  seed=args.seed, workers=args.workers, atlas_id=atlas["manifest"]["atlas_id"],
                  atlas_hash=atlas["manifest"]["bundle_sha256"])


if __name__ == "__main__":
    main()
