"""Measured EEG access, provenance and non-overlapping window export for G4.

OpenEEGBench revisions/splits live in dataset_registry.json. Public original
PhysioNet EDFs provide a small, auditable fallback when Hub access is unavailable.
This fallback is explicitly *not* the benchmark's preprocessed distribution.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from email.utils import parsedate_to_datetime
import hashlib
import json
import ntpath
from pathlib import Path, PureWindowsPath
import sys
import time

import numpy as np


REGISTRY_PATH = Path(__file__).with_name("dataset_registry.json")
DEFAULT_DATASETS = ("arithmetic_zyma2019", "physionet", "faced")


def _get(url, **kwargs):
    """At most six attempts, honoring Retry-After without retrying ahead of the server."""
    import requests
    for attempt in range(6):
        try:
            response = requests.get(url, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            status = getattr(exc.response, "status_code", None)
            retry_after = getattr(exc.response, "headers", {}).get("Retry-After")
            if exc.response is not None:
                exc.response.close()
            if attempt == 5 or (status is not None and status not in {429, 500, 502, 503, 504}):
                raise
            delay = min(5. * 2 ** attempt, 60.)
            if retry_after:
                try:
                    requested = float(retry_after)
                except ValueError:
                    try:
                        requested = parsedate_to_datetime(retry_after).timestamp() - time.time()
                    except (ValueError, TypeError, OverflowError):
                        requested = 0.
                if np.isfinite(requested):
                    if requested > 60:
                        print(f"HTTP {status}: Retry-After={requested:g}s exceeds the 60s wait cap; "
                              "resume this command after the server cooldown.", file=sys.stderr, flush=True)
                        raise
                    delay = max(delay, requested)
            print(f"HTTP {status or 'connection error'}: retry {attempt + 1}/5 in {delay:g}s: {url}",
                  file=sys.stderr, flush=True)
            time.sleep(delay)


def load_registry(names=None):
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    entries = registry["datasets"]
    if names is None:
        return entries
    unknown = set(names) - entries.keys()
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")
    return {name: entries[name] for name in names}


def subject_split(dataset, subject):
    """Use explicit benchmark subject splits; never infer them from window order."""
    entry = load_registry([dataset])[dataset]
    split = entry["benchmark_split"]
    if split["metadata_key"] != "subject":
        raise ValueError(f"{dataset} uses {split['metadata_key']} splits, not subjects")
    for name in ("train", "val", "test"):
        if subject in split[name]:
            return name
    raise ValueError(f"Subject {subject!r} not in pinned {dataset} partition")


def download_file(url, destination, *, sha256=None, max_bytes=32_000_000):
    """Cache only complete downloads; verify supplied publisher SHA256 on reuse."""
    import requests

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha256:
        if hashlib.sha256(destination.read_bytes()).hexdigest() == sha256:
            return destination
    elif destination.exists() and destination.stat().st_size:
        return destination
    temporary = destination.with_suffix(destination.suffix + ".partial")
    for attempt in range(3):
        # Header failures exhaust _get's existing budget and propagate directly.
        response = _get(url, timeout=(15, 45), stream=True)
        try:
            with response:
                if int(response.headers.get("Content-Length", 0)) > max_bytes:
                    raise ValueError(f"Download exceeds {max_bytes} byte limit: {url}")
                size = 0
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(1 << 20):
                        size += len(chunk)
                        if size > max_bytes:
                            raise ValueError(f"Download exceeds {max_bytes} byte limit: {url}")
                        handle.write(chunk)
        except requests.RequestException:
            if attempt == 2:
                raise
            delay = 5. * (attempt + 1)
            print(f"Payload interrupted: retry {attempt + 1}/2 in {delay:g}s: {url}",
                  file=sys.stderr, flush=True)
            time.sleep(delay)
        else:
            break
    if sha256 and hashlib.sha256(temporary.read_bytes()).hexdigest() != sha256:
        raise ValueError(f"SHA256 mismatch for {url}")
    temporary.replace(destination)
    return destination


def preflight(names=DEFAULT_DATASETS, output_dir="generated_simulation_dataset/G4/data_access"):
    """Check pinned public Hub metadata; errors are recorded separately per dataset."""
    import requests

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, entry in load_registry(names).items():
        url = f"https://huggingface.co/api/datasets/{entry['hf_id']}/revision/{entry['revision']}"
        result = {"dataset": name, "hf_id": entry["hf_id"], "revision": entry["revision"],
                  "url": url, "training_status": "not_started"}
        try:
            response = requests.get(url, timeout=(8, 20))
            response.raise_for_status()
            metadata = response.json()
            if metadata.get("sha") != entry["revision"]:
                raise ValueError("Hub response revision differs from registry pin")
            result.update(status="metadata_available", gated=metadata.get("gated"),
                          private=metadata.get("private"),
                          license=metadata.get("cardData", {}).get("license", "unknown"))
            (destination / f"{name}.hub.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        except (requests.RequestException, ValueError) as exc:
            result.update(status="access_error", error=str(exc))
        results[name] = result
        (destination / "preflight.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def _window_recording(raw, *, dataset, subject, run, condition, duration_s,
                      max_windows=None, source=None):
    """Keep physical volts and exact channel order; no amplitude normalization."""
    from mne.io.constants import FIFF

    if duration_s <= 0 or not np.isfinite(duration_s):
        raise ValueError("duration_s must be finite and positive")
    picks = [i for i, name in enumerate(raw.ch_names) if raw.get_channel_types()[i] == "eeg"]
    if not picks:
        raise ValueError("No EEG channels remain")
    if any(raw.info["chs"][i]["unit"] != FIFF.FIFF_UNIT_V for i in picks):
        raise ValueError("EEG units must be explicitly volts in MNE metadata")
    sfreq = float(raw.info["sfreq"])
    n_times = int(round(duration_s * sfreq))
    if n_times < 2 or not np.isclose(n_times, duration_s * sfreq):
        raise ValueError("Window duration must represent an integer number of samples")
    starts = np.arange(0, raw.n_times - n_times + 1, n_times, dtype=np.int64)
    if max_windows is not None:
        if max_windows < 1:
            raise ValueError("max_windows must be positive")
        starts = starts[:max_windows]
    if not len(starts):
        raise ValueError("Recording is shorter than one complete window")
    eeg = np.stack([raw.get_data(picks=picks, start=int(s), stop=int(s + n_times)) for s in starts])
    if not np.isfinite(eeg).all():
        raise ValueError("Recording contains nonfinite data")
    n = len(starts)
    return dict(eeg=eeg.astype(np.float32), sfreq_hz=np.asarray(sfreq),
                ch_names=np.asarray([raw.ch_names[i] for i in picks]), units=np.asarray("V"),
                subject=np.full(n, str(subject)), condition=np.full(n, str(condition)),
                split=np.full(n, subject_split(dataset, subject)), run=np.full(n, str(run)),
                trial=np.full(n, ""), block=np.arange(n), start_sample=starts,
                stop_sample=starts + n_times,
                metadata_json=np.asarray(json.dumps(source or {}, sort_keys=True)))


def save_windows(destination, windows):
    """Validate safe numeric/string NPZ export, preserving all provenance arrays."""
    required = {"eeg", "sfreq_hz", "ch_names", "units", "subject", "condition", "split", "metadata_json"}
    if not required.issubset(windows):
        raise ValueError(f"Missing fields: {sorted(required - windows.keys())}")
    if any(np.asarray(value).dtype.hasobject for value in windows.values()):
        raise ValueError("NPZ fields must be numeric or Unicode arrays, never pickle objects")
    eeg = np.asarray(windows["eeg"])
    if eeg.ndim != 3 or 0 in eeg.shape or not np.isfinite(eeg).all():
        raise ValueError("eeg must be finite nonempty [epochs, channels, time]")
    if len(windows["ch_names"]) != eeg.shape[1] or len(set(windows["ch_names"])) != eeg.shape[1]:
        raise ValueError("Channels must have unique names matching EEG channel dimension")
    if str(windows["units"]) != "V":
        raise ValueError("Expected measured EEG explicitly converted to volts")
    if not np.isfinite(float(windows["sfreq_hz"])) or float(windows["sfreq_hz"]) <= 0:
        raise ValueError("Sampling rate must be finite and positive")
    for key in ("subject", "condition", "split"):
        if np.asarray(windows[key]).shape != (len(eeg),):
            raise ValueError(f"{key} must have one value per epoch")
    if not set(windows["split"]).issubset({"train", "val", "test"}):
        raise ValueError("Unrecognized dataset split")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **windows)
    temporary.replace(destination)
    return destination


def prepare_eegmat(output_dir="generated_simulation_dataset/G4/data_access", *,
                   subjects=("00",), duration_s=5.0, max_windows=None):
    """Download verified original EEGMAT EDFs, export paired rest/arithmetic windows.

    EDF physical-dimension headers are read by MNE and converted to SI volts.
    Original acquisition/preprocessing is kept: this is not the HF 200 Hz variant.
    Subjects retain the benchmark's explicit subject split.
    """
    import mne

    root = Path(output_dir) / "eegmat"
    base = "https://physionet.org/files/eegmat/1.0.0/"
    checksums = download_file(base + "SHA256SUMS.txt", root / "SHA256SUMS.txt")
    expected = {line.split()[-1].lstrip("*"): line.split()[0]
                for line in checksums.read_text().splitlines() if line.strip()}
    for filename in ("README.txt", "subject-info.csv"):
        download_file(base + filename, root / filename, sha256=expected.get(filename))
    exports = []
    for subject in subjects:
        subject = f"{int(subject):02d}"
        subject_split("arithmetic_zyma2019", subject)
        for run, condition in ((1, "rest"), (2, "arithmetic")):
            filename = f"Subject{subject}_{run}.edf"
            path = download_file(base + filename, root / filename, sha256=expected[filename])
            raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
            # The EDF also contains ECG and a status channel, neither is a cortical EEG observation.
            excluded_channels = [name for name in raw.ch_names
                                 if not name.startswith("EEG ") or name == "EEG A2-A1"]
            eeg_channels = [name for name in raw.ch_names if name not in excluded_channels]
            if not eeg_channels:
                raise ValueError("Expected documented EEGMAT EEG-prefixed channels")
            raw.pick(eeg_channels)
            original_names = list(raw.ch_names)
            raw.rename_channels({name: name.removeprefix("EEG ") for name in raw.ch_names})
            source = {"dataset": "arithmetic_zyma2019", "distribution": "original_physionet_eegmat",
                      "version": "1.0.0", "url": base + filename, "sha256": expected[filename],
                      "license_url": "https://physionet.org/content/eegmat/1.0.0/",
                      "unit_conversion": "EDF physical units converted to SI volts by MNE read_raw_edf",
                      "edf_physical_units": raw._orig_units, "original_channel_names": original_names,
                      "excluded_channels": excluded_channels,
                      "preprocessing": "native EDF; no filtering, resampling, rereferencing or amplitude normalization",
                      "windowing": "non-overlapping complete windows; trailing incomplete window discarded",
                      "limited_windows": max_windows,
                      "benchmark_equivalent": False}
            windows = _window_recording(raw, dataset="arithmetic_zyma2019", subject=subject,
                                        run=run, condition=condition, duration_s=duration_s,
                                        max_windows=max_windows, source=source)
            exports.append(save_windows(root / f"subject{subject}_{condition}_{duration_s:g}s.npz", windows))
            raw.close()
    return exports


def _decode_pipeline(value):
    """Braindecode 1.3.2 stores JSON with a second quote-escaping layer."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return json.loads(value.replace('\\"', '"'))


def _hub_voltage_scale(attributes, info):
    """Undo recorded array scaling; MNE's unit code alone can be stale after apply_function."""
    unit_codes = [channel["unit"] for channel in info["chs"]]
    if any((unit.get("value") if isinstance(unit, dict) else unit) != 107 for unit in unit_codes):
        raise ValueError("Hub EEG metadata does not declare original SI volts")
    pipeline = _decode_pipeline(attributes.get("raw_preproc_kwargs", "[]"))
    scale = 1.0
    for transform in pipeline:
        name = transform["fn"].rsplit(".", 1)[-1]
        if name == "_scale_data":
            scale *= float(transform["kwargs"]["factor"])
        elif transform.get("apply_on_array"):
            raise ValueError(f"Cannot establish physical units after array transform {name}")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid recorded voltage scaling factor")
    return 1.0 / scale


def load_hub_recording(dataset, store, recording, *, max_windows=None):
    """Read an already cached native Zarr recording without importing/executing remote code."""
    import pandas as pd
    import zarr

    store = Path(store)
    root_attributes = json.loads((store / "zarr.json").read_text(encoding="utf-8"))["attributes"]
    group = store / f"recording_{recording}"
    attributes = json.loads((group / "zarr.json").read_text(encoding="utf-8"))["attributes"]
    info, description = attributes["info"], attributes["description"]
    metadata = pd.read_csv(group / "metadata.tsv", sep="\t", index_col=0)
    data = zarr.open_array(str(group / "data"), mode="r")
    if data.ndim != 3 or len(metadata) != data.shape[0] or data.shape[1] != len(info["ch_names"]):
        raise ValueError("Native Zarr data dimensions disagree with recording metadata")
    voltage_scale = _hub_voltage_scale(root_attributes, info)
    selection = np.arange(data.shape[0])
    if max_windows is not None:
        if max_windows < 1:
            raise ValueError("max_windows must be positive")
        selection = selection[:max_windows]
    metadata = metadata.iloc[selection]
    subject = description["subject"]
    split = subject_split(dataset, subject)
    # Recover class names from the exact saved event mapping, not a dataset-card guess.
    mapping = {}
    for item in _decode_pipeline(root_attributes.get("window_kwargs", "[]")):
        if len(item) == 2 and isinstance(item[1], dict):
            mapping.update({str(v): k for k, v in (item[1].get("mapping") or {}).items()})
    if "emotion_label" in metadata:
        conditions = metadata["emotion_label"].to_numpy(dtype=str)
    elif "target" in metadata:
        fallback = description.get("task") if metadata["target"].nunique() == 1 else None
        conditions = np.asarray([mapping.get(str(t), fallback or str(t)) for t in metadata["target"]])
    else:
        raise ValueError("Recording lacks condition/target metadata")
    n = len(selection)
    entry = load_registry([dataset])[dataset]
    manifest = store.parent.parent.parent / "download_manifest.json"
    provenance = {
        "dataset": dataset, "distribution": "OpenEEGBench_HuggingFace",
        "hf_id": entry["hf_id"], "revision": entry["revision"],
        "recording": recording, "native_description": description,
        "raw_preproc_kwargs": root_attributes.get("raw_preproc_kwargs"),
        "window_kwargs": root_attributes.get("window_kwargs"),
        "stored_values_to_volts": voltage_scale,
        "unit_conversion": "MNE original volts with recorded _scale_data factors reversed",
        "sfreq_hz": float(info["sfreq"]), "limited_windows": max_windows,
        "window_metadata": json.loads(metadata.to_json(orient="records")),
        "trial_identity": "video_index" if "video_index" in metadata else "unavailable",
        "reference": info.get("custom_ref_applied"),
        "download_manifest": str(manifest) if manifest.exists() else None,
    }
    result = dict(eeg=np.asarray(data[:n], dtype=np.float32) * voltage_scale,
                  ch_names=np.asarray(info["ch_names"]), sfreq_hz=np.asarray(float(info["sfreq"])),
                  units=np.asarray("V"), subject=np.full(n, str(subject)),
                  split=np.full(n, split), condition=conditions,
                  run=np.full(n, "" if description.get("run") is None else str(description["run"])),
                  trial=metadata["video_index"].to_numpy(dtype=str) if "video_index" in metadata else np.full(n, ""),
                  block=selection,
                  metadata_json=np.asarray(json.dumps(provenance, sort_keys=True)))
    if "target" in metadata:
        result["target"] = metadata["target"].to_numpy()
    for column, key in (("i_start_in_trial", "start_sample"), ("i_stop_in_trial", "stop_sample"),
                        ("i_window_in_trial", "window_in_trial")):
        if column in metadata:
            result[key] = metadata[column].to_numpy()
    return result


def _verify_hub_file(root, item):
    path = Path(root) / item["path"]
    if not path.exists() or path.stat().st_size != item["size"]:
        return False
    value = path.read_bytes()
    digest = item.get("lfs", {}).get("oid")
    if digest:
        return hashlib.sha256(value).hexdigest() == digest
    return hashlib.sha1(f"blob {len(value)}\0".encode() + value).hexdigest() == item["oid"]


def _resolved_path_within_root(destination, root):
    """Compare resolved paths with equivalent Win32 drive/UNC namespace spelling."""
    def comparable(path):
        if not isinstance(path, PureWindowsPath):
            return path
        value = str(path)
        if value[:8].lower() == "\\\\?\\unc\\":
            value = "\\\\" + value[8:]
        elif value.startswith("\\\\?\\"):
            value = value[4:]
        return PureWindowsPath(ntpath.normpath(value))

    return comparable(destination).is_relative_to(comparable(root))


def prepare_hub_recordings(dataset, recordings, output_dir="generated_simulation_dataset/G4/data_access",
                           *, endpoint="https://huggingface.co", max_windows=None,
                           max_recording_bytes=128_000_000, workers=2, record_workers=1):
    """Cache explicit recordings only, validating Git blobs / LFS SHA256 and pinned revision.

    An explicit public mirror endpoint is supported; every stored blob is checked
    against the pinned tree's content digest. Missing chunks must never be filled
    with Zarr zeros and interpreted as real EEG, so all recording files are fetched.
    Records run serially by default. Optional concurrency queues at most twice
    record_workers tasks; only the coordinator writes the shared manifest.
    """
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if not isinstance(record_workers, int) or record_workers < 1:
        raise ValueError("record_workers must be a positive integer")
    recordings = list(dict.fromkeys(recordings))
    if any(not isinstance(recording, int) or recording < 0 for recording in recordings):
        raise ValueError("Recording IDs must be nonnegative integers")
    entry = load_registry([dataset])[dataset]
    endpoint = endpoint.rstrip("/")
    api = f"{endpoint}/api/datasets/{entry['hf_id']}"
    revision = entry["revision"]
    root = (Path(output_dir) / dataset / revision).resolve()
    root.mkdir(parents=True, exist_ok=True)
    store_path = "sourcedata/braindecode/dataset.zarr"
    cached_manifest = root / "download_manifest.json"
    previous = json.loads(cached_manifest.read_text(encoding="utf-8")) if cached_manifest.exists() else {}
    if previous and previous.get("revision") != revision:
        raise ValueError("Cached manifest revision differs from registry")
    metadata_file = root / "hub_metadata.json"
    revision_info = (json.loads(metadata_file.read_text(encoding="utf-8")) if metadata_file.exists()
                     else _get(f"{api}/revision/{revision}", timeout=(12, 30)).json())
    if revision_info.get("sha") != revision:
        raise ValueError("Remote dataset revision differs from registry")
    if revision_info.get("gated") or revision_info.get("private"):
        raise PermissionError("Dataset requires its normal authorized access procedure")
    metadata_file.write_text(json.dumps(revision_info, indent=2), encoding="utf-8")
    root_tree_file = root / "root-tree.json"
    root_tree = (json.loads(root_tree_file.read_text(encoding="utf-8")) if root_tree_file.exists()
                 else _get(f"{api}/tree/{revision}/{store_path}?recursive=false", timeout=(12, 30)).json())
    root_tree_file.write_text(json.dumps(root_tree, indent=2), encoding="utf-8")
    root_files = [item for item in root_tree if item["path"] == store_path + "/zarr.json"]
    if len(root_files) != 1:
        raise ValueError("Missing root Zarr metadata")

    def fetch(item):
        relative = Path(item["path"])
        destination = (root / relative).resolve()
        if not _resolved_path_within_root(destination, root.resolve()):
            raise ValueError(f"Unsafe path in remote manifest: {relative!s} -> {destination!s}")
        if _verify_hub_file(root, item):
            return item
        digest = item.get("lfs", {}).get("oid")
        if destination.exists() and not digest:
            destination.unlink()  # The manifest hash rejected this cache file; fetch it again.
        path = download_file(f"{endpoint}/datasets/{entry['hf_id']}/resolve/{revision}/{item['path']}",
                             destination, sha256=digest, max_bytes=max_recording_bytes)
        if not digest:
            value = path.read_bytes()
            actual = hashlib.sha1(f"blob {len(value)}\0".encode() + value).hexdigest()
            if actual != item["oid"]:
                raise ValueError(f"Git blob hash mismatch: {item['path']}")
        return item

    fetch(root_files[0])
    combined = {item["path"]: item for item in previous.get("files", [])}
    combined[root_files[0]["path"]] = root_files[0]
    completed_recordings = set(previous.get("recordings", []))
    n_recordings = json.loads((root / store_path / "zarr.json").read_text(encoding="utf-8"))["attributes"]["n_datasets"]
    if any(recording >= n_recordings for recording in recordings):
        raise ValueError(f"Recording index exceeds dataset size {n_recordings}")
    def prepare_record(position, recording):
        print(f"{dataset} [{position}/{len(recordings)}] recording_{recording}: checking cache/download",
              flush=True)
        url = f"{api}/tree/{revision}/{store_path}/recording_{recording}?recursive=true"
        tree_file = root / f"recording_{recording}-tree.json"
        entries = json.loads(tree_file.read_text(encoding="utf-8")) if tree_file.exists() else []
        if entries:
            url = None
        while url:
            response = _get(url, timeout=(12, 30))
            entries.extend(response.json())
            url = response.links.get("next", {}).get("url")
        tree_file.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        selected = [item for item in entries if item["type"] == "file"]
        if not selected or sum(item["size"] for item in selected) > max_recording_bytes:
            raise ValueError("Empty recording or recording download exceeds configured byte cap")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            verified = list(pool.map(fetch, selected))
        windows = load_hub_recording(dataset, root / store_path, recording, max_windows=max_windows)
        path = save_windows(root / f"recording_{recording}.npz", windows)
        return path, verified, len(windows["eeg"])

    results = {}
    queue_limit = 1 if record_workers == 1 else 2 * record_workers
    remaining = iter(enumerate(recordings, 1))
    first_error = None
    with ThreadPoolExecutor(max_workers=record_workers) as record_pool:
        pending = {}
        while True:
            while first_error is None and len(pending) < queue_limit:
                task = next(remaining, None)
                if task is None:
                    break
                position, recording = task
                pending[record_pool.submit(prepare_record, position, recording)] = task
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                position, recording = pending.pop(future)
                if future.cancelled():
                    continue
                try:
                    path, verified, n_windows = future.result()
                except Exception as exc:
                    first_error = first_error or exc
                    for queued in pending:
                        queued.cancel()
                    continue
                # Single coordinator: completed records are committed even if a peer failed.
                results[recording] = path
                combined.update({item["path"]: item for item in verified})
                completed_recordings.add(recording)
                manifest = {"dataset": dataset, "revision": revision, "endpoint": endpoint,
                            "files": list(combined.values()), "status": "downloaded_hash_verified",
                            "recordings": sorted(completed_recordings),
                            "full_dataset": completed_recordings == set(range(n_recordings))}
                temporary = cached_manifest.with_suffix(".partial.json")
                temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                temporary.replace(cached_manifest)
                print(f"{dataset} [{position}/{len(recordings)}] recording_{recording}: "
                      f"verified/exported {n_windows} windows", flush=True)
    if first_error is not None:
        raise first_error
    return [results[recording] for recording in recordings]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "eegmat", "hub"))
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--subjects", nargs="+", default=["00"])
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--recordings", type=int, nargs="+", default=[0])
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--record-workers", type=int, default=1,
                        help="Concurrent records; maximum simultaneous HTTP requests is workers times record-workers.")
    parser.add_argument("--output-dir", default="generated_simulation_dataset/G4/data_access")
    args = parser.parse_args()
    if args.action == "preflight":
        print(json.dumps(preflight(args.datasets, args.output_dir), indent=2))
    elif args.action == "eegmat":
        for path in prepare_eegmat(args.output_dir, subjects=args.subjects,
                                   duration_s=args.duration_s, max_windows=args.max_windows):
            print(path)
    else:
        for dataset in args.datasets:
            for path in prepare_hub_recordings(dataset, args.recordings, args.output_dir,
                                               endpoint=args.endpoint, max_windows=args.max_windows,
                                               workers=args.workers, record_workers=args.record_workers):
                print(path)


if __name__ == "__main__":
    main()
