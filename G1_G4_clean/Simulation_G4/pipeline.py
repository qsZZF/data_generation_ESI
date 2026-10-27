"""Train a montage-specific paper-style inverse and reconstruct measured EEG.

Input EEG must come from the provenance-checked loaders in data.py. Synthetic
parameters supervise training; measured EEG never masquerades as parameter truth.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import multiprocessing
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt, welch
import torch

from Simulation_G1.noise_covariance import baseline_covariance_payload, baseline_sample_count

from .atlas import load_atlas
from .bank import array_hash, project_bank
from .inverse import create_model, parameter_metrics, predict, train_from_arrays, unpack_parameters
from .resources import load_cortical_connectivity
from .simulator import PARAMETER_NAMES, simulate

ROOT = Path(__file__).resolve().parents[1]
CHANNEL_ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


def channel_key(name):
    name = str(name).strip().upper().removeprefix("EEG ").removesuffix("-REF").rstrip(".")
    return CHANNEL_ALIASES.get(name, name)


def analysis_filter(eeg, sampling_rate_hz):
    """Same finite-window observation filter for simulated and measured signals."""
    if sampling_rate_hz < 100:
        raise ValueError("The initial 1-40 Hz analysis requires sampling rate >=100 Hz")
    filt = butter(4, [1, 40], btype="bandpass", fs=sampling_rate_hz, output="sos")
    values = sosfiltfilt(filt, eeg, axis=-1)
    values -= values.mean(axis=-1, keepdims=True)
    values -= values.mean(axis=-2, keepdims=True)
    return values.astype(np.float32)


def load_measured(paths, atlas, *, sampling_rate_hz=256):
    """Align measured EEG to the bank grid, in volts, for 1-40 Hz analysis.

    Upsampling changes the grid only; it does not add measured bandwidth.
    Epoch start/stop sample provenance remains on the native recording grid.
    """
    if not np.isfinite(sampling_rate_hz) or sampling_rate_hz < 100:
        raise ValueError("The 1-40 Hz analysis requires a finite target sampling rate >=100 Hz")
    records = []
    common = set(map(channel_key, atlas["channel_names"]))
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if str(data["units"]) != "V":
                raise ValueError(f"Measured units are not verified volts: {path}")
            names = [channel_key(name) for name in data["ch_names"]]
            if len(names) != len(set(names)):
                raise ValueError(f"Channel aliases collide in {path}")
            common &= set(names)
    ordered = [channel_key(name) for name in atlas["channel_names"] if channel_key(name) in common]
    if len(ordered) < 8:
        raise ValueError(f"Only {len(ordered)} channels match the actual leadfield; cannot use this initial inverse")
    atlas_idx = [list(map(channel_key, atlas["channel_names"])).index(name) for name in ordered]
    n_times = None
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
            names = [channel_key(name) for name in data["ch_names"]]
            take = [names.index(name) for name in ordered]
            eeg = np.asarray(data["eeg"][:, take], dtype=float)
            native_rate = float(data["sfreq_hz"])
            if not np.isfinite(native_rate) or native_rate < 100:
                raise ValueError(f"The 1-40 Hz analysis requires native sampling rate >=100 Hz: {path}")
            if native_rate != sampling_rate_hz:
                ratio = Fraction(sampling_rate_hz / native_rate).limit_denominator(10000)
                eeg = resample_poly(eeg, ratio.numerator, ratio.denominator, axis=-1)
            eeg = analysis_filter(eeg, sampling_rate_hz)
            if n_times is None:
                n_times = eeg.shape[-1]
            if eeg.shape[-1] != n_times:
                raise ValueError("Dataset contains different window lengths; train separate duration domains")
            if not np.isfinite(eeg).all() or np.any(eeg.std(axis=(1, 2)) < 1e-12):
                raise ValueError(f"Nonfinite or flat measured EEG in {path}")
            rows = {}
            for key in ("subject", "condition", "split"):
                values = np.asarray(data[key]).astype(str)
                if values.shape != (len(eeg),):
                    raise ValueError(f"Invalid epoch provenance {key} in {path}")
                rows[key] = values
            for key in ("trial", "run", "block", "start_sample", "stop_sample"):
                rows[key] = data[key].astype(str) if key in data else np.full(len(eeg), "")
                if rows[key].shape != (len(eeg),):
                    raise ValueError(f"Invalid epoch provenance {key} in {path}")
            rows["recording"] = np.full(len(eeg), path.stem)
            rows["window_id"] = np.asarray([f"{path.name}:{i}" for i in range(len(eeg))])
            records.append((eeg, rows, {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                        "native_sfreq_hz": native_rate, "metadata": metadata,
                                        "analysis_sfreq_hz": float(sampling_rate_hz),
                                        "analysis_band_hz": [1, 40],
                                        "grid_upsampled": native_rate < sampling_rate_hz,
                                        "resampling_method": "resample_poly" if native_rate != sampling_rate_hz else "none",
                                        "epoch_sample_indices_grid": "native",
                                        "dropped_channels": [name for name in names if name not in common]}))
    return (np.concatenate([record[0] for record in records]),
            {key: np.concatenate([record[1][key] for record in records]) for key in records[0][1]},
            ordered, np.asarray(atlas["leadfield"])[atlas_idx], [record[2] for record in records])


def atomic_checkpoint(path, model, leadfield, metadata):
    temporary = path.with_suffix(".part")
    torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "model_config": model.config, "leadfield": torch.as_tensor(leadfield),
                "metadata": json.loads(json.dumps(metadata))}, temporary)
    temporary.replace(path)


def _forward_metrics(observed, reconstructed, rate):
    # Distributional checks; stochastic waveforms are not expected to match phase.
    frequency, real_psd = welch(observed, fs=rate, nperseg=min(observed.shape[-1], int(2 * rate)), axis=-1)
    _, sim_psd = welch(reconstructed, fs=rate, nperseg=min(reconstructed.shape[-1], int(2 * rate)), axis=-1)
    select = (frequency >= 1) & (frequency <= 40)
    real = np.log10(np.maximum(real_psd.mean(axis=0)[..., select], 1e-30))
    simulated = np.log10(np.maximum(sim_psd.mean(axis=0)[..., select], 1e-30))
    return {"mean_abs_log10_psd_error": float(np.mean(abs(real - simulated))),
            "log_psd_correlation": float(np.corrcoef(real.ravel(), simulated.ravel())[0, 1])}


def _atomic_npz(path, **arrays):
    temporary = path.with_suffix(".part")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _group_fingerprint(job):
    """Hash every group input, numerical convention and reconstruction code."""
    identity = {key: job[key] for key in ("subject", "condition", "seed", "sampling_rate_hz",
                                        "warmup_s", "dt_ms", "code_identity")}
    identity["region_names"] = np.asarray(job["region_names"]).tolist()
    identity["aal_roi_numbers"] = np.asarray(job["aal_roi_numbers"]).tolist()
    identity["provenance"] = {key: np.asarray(value).tolist() for key, value in job["provenance"].items()}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8"))
    arrays = {key: job[key] for key in ("eeg", "leadfield", "connectivity")}
    arrays.update({"estimated_" + key: value for key, value in job["parameters"].items()})
    arrays.update({"prior_" + key: value for key, value in job["prior_parameters"].items()})
    for key, value in sorted(arrays.items()):
        array = np.ascontiguousarray(value, dtype=np.float64)
        digest.update(json.dumps([key, list(array.shape)]).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _reconstruct_group(job):
    """One subject/condition job; no model or full dataset enters a worker."""
    eeg, provenance, parameters = job["eeg"], job["provenance"], job["parameters"]
    subject, condition = job["subject"], job["condition"]
    leadfield, connectivity = job["leadfield"], job["connectivity"]
    sampling_rate_hz, seed = job["sampling_rate_hz"], job["seed"]
    warmup_s, dt_ms = job["warmup_s"], job["dt_ms"]
    prior_parameters = job["prior_parameters"]
    chosen = np.arange(len(eeg))
    name = f"group_{job['group_index']:04d}.npz"
    artifact = Path(job["output"]) / name
    cache_path = artifact.with_suffix(".json")
    fingerprint = _group_fingerprint(job)
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if (cached["fingerprint"] == fingerprint
                and cached["result"]["file"] == str(Path("reconstructed_sources") / name)
                and cached["artifact_sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()):
            with np.load(artifact, allow_pickle=False) as data:
                if str(data["fingerprint"]) == fingerprint:
                    return cached["result"]
    except (OSError, ValueError, KeyError, TypeError):
        pass  # Missing, stale, corrupt, or interrupted cache: regenerate this group.
    # Trial numbers can restart in every recording/run.
    trials = np.asarray([f"{record}:{trial}" if trial else "" for record, trial in
                         zip(provenance["recording"][chosen], provenance["trial"][chosen])])
    unique_trials = sorted(set(trials) - {""})
    recordings = sorted(set(provenance["recording"]))
    if len(unique_trials) >= 2:
        fitting = chosen[np.isin(trials, unique_trials[:max(1, len(unique_trials) // 2)])]
        evaluation = np.setdiff1d(chosen, fitting)
        split_kind = "held_out_trials"
    elif not unique_trials and len(recordings) >= 2:
        fitting = chosen[np.isin(provenance["recording"], recordings[:max(1, len(recordings) // 2)])]
        evaluation = np.setdiff1d(chosen, fitting)
        split_kind = "held_out_recordings"
    elif len(chosen) >= 5:
        midpoint = len(chosen) // 2
        fitting, evaluation = chosen[:midpoint - 1], chosen[midpoint + 1:]
        split_kind = "within_record_blocks_two_window_guard"
    else:
        return {"subject": subject, "condition": condition, "status": "parameters_only_insufficient_independent_windows"}
    # Linear constraints of the joint tau prior are preserved by averaging.
    point = {key: value[fitting].mean(axis=0, dtype=float) for key, value in parameters.items()}
    point["delay_ms"] = float(point["delay_ms"][0])
    regional, projected, baseline_projected, metas = [], [], [], []
    for repetition in range(3):
        source, times, meta = simulate(point, connectivity,
            duration_s=eeg.shape[-1] / sampling_rate_hz, warmup_s=warmup_s, dt_ms=dt_ms,
            sampling_rate_hz=sampling_rate_hz, seed=seed + repetition)
        regional.append(source.astype(np.float32))
        projected.append(analysis_filter((leadfield @ source)[None], sampling_rate_hz)[0])
        metas.append(meta)
        baseline_source, _, _ = simulate(prior_parameters, connectivity,
            duration_s=eeg.shape[-1] / sampling_rate_hz, warmup_s=warmup_s, dt_ms=dt_ms,
            sampling_rate_hz=sampling_rate_hz, seed=seed + repetition)
        baseline_projected.append(analysis_filter((leadfield @ baseline_source)[None], sampling_rate_hz)[0])
    projected = np.asarray(projected)
    gain = float(np.sqrt(np.mean(eeg[fitting] ** 2) / max(float(np.mean(projected ** 2)), np.finfo(float).tiny)))
    reconstructed = projected * gain
    baseline_projected = np.asarray(baseline_projected)
    baseline_gain = float(np.sqrt(np.mean(eeg[fitting] ** 2) /
                          max(float(np.mean(baseline_projected ** 2)), np.finfo(float).tiny)))
    baseline_metrics = _forward_metrics(eeg[evaluation], baseline_projected * baseline_gain, sampling_rate_hz)
    fitted_metrics = _forward_metrics(eeg[evaluation], reconstructed, sampling_rate_hz)
    baseline_samples = baseline_sample_count(1., sampling_rate_hz)
    covariance_arrays, covariance_metadata = baseline_covariance_payload(
        np.zeros((leadfield.shape[0], baseline_samples)), np.zeros_like(reconstructed[0]),
        average_reference=True, baseline_metadata=dict(duration_s=1., sampling_rate_hz=sampling_rate_hz,
            n_samples=baseline_samples, target_present=False, status="uninformative_no_added_noise",
            generation="zeros_no_added_noise"))
    covariance_arrays = {key: np.repeat(value[None], len(reconstructed), axis=0)
                         for key, value in covariance_arrays.items()}
    covariance_metadata.update(noise_status="no_added_noise",
        covariance_target="all_simulated_network", covariance_units="V^2",
        covariance_scope="generated_clean_sensor_only; not measured EEG noise")
    _atomic_npz(artifact, fingerprint=fingerprint,
        region_source_mV=np.asarray(regional), clean_sensor_V=reconstructed, time_s=times,
        **covariance_arrays,
        observation_gain=gain, prior_observation_gain=baseline_gain,
        prior_sensor_V=baseline_projected * baseline_gain, region_names=job["region_names"], **point,
        fitting_window_ids=provenance["window_id"][fitting], evaluation_window_ids=provenance["window_id"][evaluation],
        metadata_json=json.dumps({"subject": subject, "condition": condition, "split_kind": split_kind,
            "human_source_truth_available": False, "parameter_status": "model_conditioned_estimate",
            "source_unit": "mV_PSP", "simulations": metas, **covariance_metadata}))
    result = {"subject": subject, "condition": condition, "status": "forward_checked",
              "fitting_windows": len(fitting), "evaluation_windows": len(evaluation),
              "split_kind": split_kind, "observation_gain": gain, "file": str(Path("reconstructed_sources") / name),
              **fitted_metrics, "prior_baseline": baseline_metrics,
              "log_psd_error_improvement_over_prior": baseline_metrics["mean_abs_log10_psd_error"] -
                  fitted_metrics["mean_abs_log10_psd_error"]}
    result["forward_quality"] = ("lower_psd_error_than_prior" if result["log_psd_error_improvement_over_prior"] > 0
                                 else "no_psd_error_improvement_over_prior")
    result["physiological_parameter_recovery_validated"] = False
    # Commit the archive before the JSON completion record. An interruption
    # between these replaces is detected by the stored archive checksum.
    temporary = cache_path.with_suffix(".part")
    temporary.write_text(json.dumps({"fingerprint": fingerprint, "result": result,
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}, indent=2) + "\n", encoding="utf-8")
    temporary.replace(cache_path)
    return result


def _reconstruction_worker_init():
    torch.set_num_threads(1)


def reconstruct_groups(model, eeg, provenance, atlas, connectivity, leadfield, output,
                       *, sampling_rate_hz, prior_parameters, seed=42, device="auto", max_groups=None,
                       warmup_s=1, dt_ms=0.5, reconstruction_workers=1):
    """Estimate windows once, then resume/parallelize independent forward checks."""
    if not isinstance(reconstruction_workers, int) or reconstruction_workers < 1:
        raise ValueError("reconstruction_workers must be a positive integer")
    parameters = predict(model, eeg, device=device)
    _atomic_npz(output / "estimated_parameters.npz", **parameters, **provenance,
                region_names=atlas["region_names"], aal_roi_numbers=atlas["aal_roi_numbers"],
                parameter_unit_note="time constants/delay ms; theta mV; beta 1/mV; rmax Hz; C dimensionless")
    keys = sorted(set(zip(provenance["subject"], provenance["condition"])))
    if max_groups is not None:
        keys = keys[:max_groups]
    reconstruction_dir = output / "reconstructed_sources"
    reconstruction_dir.mkdir(exist_ok=True)
    code_identity = hashlib.sha256(Path(__file__).read_bytes() +
                                    Path(__file__).with_name("simulator.py").read_bytes() +
                                    (ROOT / "Simulation_G1/noise_covariance.py").read_bytes()).hexdigest()

    def jobs():
        for index, (subject, condition) in enumerate(keys):
            chosen = np.flatnonzero((provenance["subject"] == subject) & (provenance["condition"] == condition))
            yield index, {"group_index": index, "subject": str(subject), "condition": str(condition),
                "eeg": eeg[chosen], "parameters": {key: value[chosen] for key, value in parameters.items()},
                "provenance": {key: value[chosen] for key, value in provenance.items()},
                "leadfield": leadfield, "connectivity": connectivity, "region_names": atlas["region_names"],
                "aal_roi_numbers": atlas["aal_roi_numbers"], "prior_parameters": prior_parameters,
                "sampling_rate_hz": sampling_rate_hz, "seed": seed + index * 3, "warmup_s": warmup_s,
                "dt_ms": dt_ms, "output": str(reconstruction_dir), "code_identity": code_identity}

    results = [None] * len(keys)

    def record(index, result):
        results[index] = result
        print(json.dumps({"stage": "reconstruction", **result}), flush=True)

    if reconstruction_workers == 1 or len(keys) < 2:
        for index, job in jobs():
            record(index, _reconstruct_group(job))
    else:
        # Bound queued group copies; never serialize the full EEG dataset or
        # CUDA model into each process. Spawn also avoids forking CUDA state.
        with ProcessPoolExecutor(max_workers=reconstruction_workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_reconstruction_worker_init) as executor:
            iterator = iter(jobs())
            pending = {}
            for _ in range(min(len(keys), 2 * reconstruction_workers)):
                index, job = next(iterator)
                pending[executor.submit(_reconstruct_group, job)] = index
            while pending:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    record(pending.pop(future), future.result())
                    item = next(iterator, None)
                    if item is not None:
                        index, job = item
                        pending[executor.submit(_reconstruct_group, job)] = index
    return results


def validate_bank_identity(config, atlas, connectivity):
    expected = {"atlas_id": atlas["manifest"]["atlas_id"],
                "atlas_hash": atlas["manifest"]["bundle_sha256"],
                "n_regions": len(atlas["region_names"]),
                "connectivity_sha256": array_hash(connectivity),
                "simulator_sha256": hashlib.sha256(Path(__file__).with_name("simulator.py").read_bytes()).hexdigest()}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Source bank {key} differs from the reconstruction inputs")


def run_dataset(name, paths, atlas, connectivity, bank_dir, output_dir, *, epochs=150,
                batch_size=8, profile="paper", device="auto", seed=42, sensor_snr_db=None,
                max_reconstruction_groups=None, train_only=False, reconstruction_workers=1,
                baseline_duration_s=1.):
    output = Path(output_dir) / name
    output.mkdir(parents=True, exist_ok=True)
    bank_config = json.loads((Path(bank_dir) / "config.json").read_text())
    validate_bank_identity(bank_config, atlas, connectivity)
    splits = json.loads((Path(bank_dir) / "splits.json").read_text())
    rate = bank_config["sampling_rate_hz"]
    eeg_real, provenance, channels, leadfield, sources = load_measured(paths, atlas, sampling_rate_hz=rate)
    leadfield = leadfield - leadfield.mean(axis=0, keepdims=True)
    metadata = {"dataset": name, "measured_window_count": len(eeg_real), "measured_subjects": sorted(set(provenance["subject"])),
                "channels": channels, "sampling_rate_hz": rate, "window_samples": eeg_real.shape[-1],
                "atlas_manifest": atlas["manifest"], "bank_config": bank_config,
                "source_files": sources, "sensor_training_snr_db": sensor_snr_db,
                "observation_filter": "within-window fourth-order Butterworth 1-40Hz; same for synthetic/measured",
                "human_parameter_truth_available": False, "reference_reproduction": "adapted cortical model; not million-sample replication"}
    # The within-window filter needs more than its 27-sample padding length.
    if baseline_sample_count(baseline_duration_s, rate) <= 27:
        raise ValueError("G4 baseline needs at least 28 samples for the 1-40 Hz analysis filter")
    synthetic, targets, sensor_noise, baseline = project_bank(bank_dir, leadfield, eeg_real.shape[-1],
        sensor_snr_db=sensor_snr_db, seed=seed, return_baseline=True,
        baseline_duration_s=baseline_duration_s)
    synthetic = analysis_filter(synthetic, rate)
    sensor_noise = analysis_filter(sensor_noise, rate)
    baseline_sensor = analysis_filter(baseline["baseline_sensor"], rate)
    covariance_arrays, covariance_metadata = [], []
    for index, noise in enumerate(sensor_noise):
        arrays, info = baseline_covariance_payload(baseline_sensor[index], noise,
            average_reference=True, baseline_metadata=baseline["metadata"][index])
        covariance_arrays.append(arrays)
        covariance_metadata.append(info)
    covariance_file = "training_noise_covariance.npz"
    covariance_manifest = dict(file=covariance_file, units="projected_sensor_units_squared",
        channels=channels, covariance_target="all_simulated_network",
        default_covariance="baseline_only", baseline_duration_s=float(baseline_duration_s),
        signal_stage="after_analysis_filter_before_inverse_global_zscore",
        normalization_rule="if sensor is divided by scale, divide covariance by scale squared",
        preprocessing=metadata["observation_filter"], average_reference=True,
        noise_status="no_added_noise" if sensor_snr_db is None else "added_sensor_noise",
        scope="synthetic_training_draws_only; not measured EEG noise")
    _atomic_npz(output / covariance_file, draw_indices=np.arange(len(synthetic)),
        **{key: np.asarray([arrays[key] for arrays in covariance_arrays])
           for key in covariance_arrays[0]},
        metadata_json=json.dumps({**covariance_manifest, "draws": covariance_metadata}))
    metadata["noise_covariance"] = covariance_manifest
    (output / "run_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    del sensor_noise, covariance_arrays, baseline, baseline_sensor
    model = create_model(len(atlas["region_names"]), len(channels), eeg_real.shape[-1], leadfield,
                         seed=seed, profile=profile, sampling_rate_hz=rate)
    started = time.perf_counter()

    def progress(record):
        print(json.dumps({"stage": "training", "dataset": name, **record}), flush=True)
        with (output / "training_progress.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if record["is_best"]:
            atomic_checkpoint(output / "best_checkpoint.pt", model, leadfield, metadata)

    history = train_from_arrays(model, synthetic, targets, train_indices=splits["train"],
        validation_indices=splits["validation"], groups=np.arange(len(targets)), epochs=epochs,
        batch_size=batch_size, device=device, seed=seed, progress=progress)
    history["training_seconds"] = time.perf_counter() - started
    truth = unpack_parameters(targets, model.N)
    prior_parameters = {key: value[splits["train"]].mean(axis=0, dtype=float) for key, value in truth.items()}
    prior_parameters["delay_ms"] = float(prior_parameters["delay_ms"][0])
    test = np.asarray(splits["test"])
    predictions = predict(model, synthetic[test], batch_size=batch_size, device=device)
    history["heldout_test_metrics"] = parameter_metrics(predictions, {key: value[test] for key, value in truth.items()})
    prior = {key: np.broadcast_to(value[splits["train"]].mean(axis=0), value[test].shape) for key, value in truth.items()}
    history["prior_mean_test_metrics"] = parameter_metrics(prior, {key: value[test] for key, value in truth.items()})
    calibration = np.asarray(splits["calibration"])
    calibrated = predict(model, synthetic[calibration], batch_size=batch_size, device=device)
    history["synthetic_marginal_calibration"] = {}
    for key in truth:
        residual = np.abs(calibrated[key] - truth[key][calibration])
        radius = float(np.quantile(residual, .9, method="higher"))
        coverage = float(np.mean(np.abs(predictions[key] - truth[key][test]) <= radius))
        history["synthetic_marginal_calibration"][key] = {"absolute_radius": radius, "heldout_coverage": coverage,
            "scope": "pooled synthetic family marginal; not real-data or aggregated-fit coverage"}
    history["parameter_count"] = sum(p.numel() for p in model.parameters())
    history["status"] = "trained_recovery_metrics_reported_not_physiologically_validated"
    (output / "training_report.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    atomic_checkpoint(output / "best_checkpoint.pt", model, leadfield, metadata)
    np.savez(output / "training_prior.npz", **prior_parameters)
    if train_only:
        return history
    results = reconstruct_groups(model, eeg_real, provenance, atlas, connectivity, leadfield, output,
        sampling_rate_hz=rate, prior_parameters=prior_parameters, seed=seed, device=device,
        max_groups=max_reconstruction_groups, warmup_s=bank_config["warmup_s"], dt_ms=bank_config["dt_ms"],
        reconstruction_workers=reconstruction_workers)
    report = {"dataset": name, "training_report": "training_report.json", "checkpoint": "best_checkpoint.pt",
              "estimated_parameters": "estimated_parameters.npz", "groups": results,
              "atlas_boundary_limitation": atlas["manifest"]["template_limitation"],
              "scope": "conditional fitted parameters and stochastic model-generated sources"}
    (output / "reconstruction_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def reconstruct_dataset(name, paths, atlas, connectivity, bank_dir, output_dir, *, device="auto", seed=42,
                        max_reconstruction_groups=None, reconstruction_workers=1):
    """Apply a completed checkpoint to additional subjects without retraining."""
    output = Path(output_dir) / name
    checkpoint = torch.load(output / "best_checkpoint.pt", map_location="cpu", weights_only=True)
    config = json.loads((Path(bank_dir) / "config.json").read_text())
    validate_bank_identity(config, atlas, connectivity)
    previous = checkpoint["metadata"]
    if previous["dataset"] != name or previous["bank_config"] != config:
        raise ValueError("Checkpoint dataset or source bank differs")
    rate = config["sampling_rate_hz"]
    eeg, provenance, channels, leadfield, sources = load_measured(paths, atlas, sampling_rate_hz=rate)
    leadfield -= leadfield.mean(axis=0, keepdims=True)
    if channels != previous["channels"] or eeg.shape[-1] != previous["window_samples"]:
        raise ValueError("Measured montage/window differs from trained model")
    np.testing.assert_allclose(leadfield, checkpoint["leadfield"].numpy(), rtol=1e-12, atol=0)
    model = create_model(leadfield=leadfield, **checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    with np.load(output / "training_prior.npz", allow_pickle=False) as data:
        prior = dict(data)
    prior["delay_ms"] = float(prior["delay_ms"])
    results = reconstruct_groups(model, eeg, provenance, atlas, connectivity, leadfield, output,
        sampling_rate_hz=rate, prior_parameters=prior, seed=seed, device=device,
        max_groups=max_reconstruction_groups, warmup_s=config["warmup_s"], dt_ms=config["dt_ms"],
        reconstruction_workers=reconstruction_workers)
    report = {"dataset": name, "training_report": "training_report.json", "checkpoint": "best_checkpoint.pt",
              "estimated_parameters": "estimated_parameters.npz", "groups": results,
              "measured_window_count": len(eeg), "measured_subjects": sorted(set(provenance["subject"])),
              "source_files": sources, "atlas_boundary_limitation": atlas["manifest"]["template_limitation"],
              "scope": "conditional fitted parameters and stochastic model-generated sources"}
    (output / "reconstruction_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--atlas-dir", type=Path, default=ROOT / "Anatomy/G4_AAL_cortical")
    parser.add_argument("--connectivity-dir", type=Path, default=ROOT / "Anatomy/G4_resources")
    parser.add_argument("--bank-dir", type=Path, default=ROOT / "generated_simulation_dataset/G4/aal78_bank_256hz_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "generated_simulation_dataset/G4/trained_256hz")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--profile", choices=["paper", "compact"], default="paper")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sensor-snr-db", type=float, default=None)
    parser.add_argument("--baseline-duration-s", type=float, default=1.)
    parser.add_argument("--max-reconstruction-groups", type=int, default=None)
    parser.add_argument("--reconstruction-workers", type=int, default=1)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train-only", action="store_true")
    mode.add_argument("--reconstruct-only", action="store_true")
    args = parser.parse_args()
    paths = sorted(args.data_dir.glob("*.npz"))
    if not paths:
        parser.error("No prepared measured-EEG NPZ files found")
    atlas = load_atlas(args.atlas_dir)
    connectivity = load_cortical_connectivity(args.connectivity_dir, atlas["aal_roi_numbers"])
    if args.reconstruct_only:
        reconstruct_dataset(args.dataset, paths, atlas, connectivity, args.bank_dir, args.output_dir,
                            device=args.device, max_reconstruction_groups=args.max_reconstruction_groups,
                            reconstruction_workers=args.reconstruction_workers)
        return
    run_dataset(args.dataset, paths, atlas, connectivity, args.bank_dir, args.output_dir,
                epochs=args.epochs, batch_size=args.batch_size, profile=args.profile,
                device=args.device, sensor_snr_db=args.sensor_snr_db,
                baseline_duration_s=args.baseline_duration_s,
                max_reconstruction_groups=args.max_reconstruction_groups, train_only=args.train_only,
                reconstruction_workers=args.reconstruction_workers)


if __name__ == "__main__":
    main()
