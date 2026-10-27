"""Prepare a fixed, small PhysioNet rest/imagery validation cohort.

No filtering, normalization, rereferencing or resampling is applied. EDF physical
units are converted to volts by MNE. Imagery epochs include verified T0 baseline;
their averages describe cue-related responses, not canonical sensory ERP trials.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")
import mne
import numpy as np
import requests


BASE_URL = "https://physionet.org/files/eegmmidb/1.0.0/"
TRANSPORT_URL = "https://physionet-open.s3.amazonaws.com/eegmmidb/1.0.0/"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "generated_simulation_dataset/G1_spectral_validation"


def read_edf(path):
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    if raw.info["sfreq"] != 160 or len(raw.ch_names) != 64:
        raw.close()
        raise ValueError(f"Expected 64 EEG channels at 160 Hz: {path}")
    return raw


def download_edf(url, destination, expected_hash=None):
    """Bounded streaming download; validate EDF before promoting the cache file."""
    if destination.exists():
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if expected_hash is not None and digest != expected_hash:
            raise ValueError(f"Cached EDF differs from the recorded SHA256: {destination}")
        read_edf(destination).close()
        return digest
    temporary = destination.with_suffix(".partial.edf")
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=(15, 45)) as response:
                response.raise_for_status()
                size = 0
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(1 << 16):
                        size += len(chunk)
                        if size > 8_000_000:
                            raise ValueError(f"Unexpected EDF size over 8 MB: {url}")
                        handle.write(chunk)
                expected_size = response.headers.get("Content-Length")
                if expected_size is not None and size != int(expected_size):
                    raise ValueError(f"Incomplete HTTP payload: {url}")
            read_edf(temporary).close()
            digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
            if expected_hash is not None and digest != expected_hash:
                raise ValueError(f"Downloaded EDF differs from the recorded SHA256: {destination}")
            temporary.replace(destination)
            return digest
        except (requests.RequestException, OSError) as exc:
            if attempt == 2:
                raise
            print(f"Retry {attempt + 1}/2: {url}: {exc}", flush=True)
            time.sleep(2 * (attempt + 1))


def imagery_intervals(annotations, n_times, sfreq):
    """Return [-2, 3) s intervals whose full baseline is explicitly T0."""
    annotations = sorted(annotations, key=lambda item: item[0])
    selected = []
    for index, (onset, duration, code) in enumerate(annotations):
        if code not in ("T1", "T2") or index == 0 or duration < 3:
            continue
        previous_onset, previous_duration, previous_code = annotations[index - 1]
        if previous_code != "T0" or previous_onset > onset - 2 + 1e-7:
            continue
        if previous_onset + previous_duration < onset - 1e-7:
            continue
        if index + 1 < len(annotations) and annotations[index + 1][0] < onset + 3 - 1e-7:
            continue
        start, stop = round((onset - 2) * sfreq), round((onset + 3) * sfreq)
        if start >= 0 and stop <= n_times:
            selected.append((index, start, stop, onset, code))
    return selected


def export_record(raw, destination, *, subject, run, provenance):
    sfreq = float(raw.info["sfreq"])
    annotations = list(zip(raw.annotations.onset, raw.annotations.duration,
                           raw.annotations.description))
    if run in (1, 2):
        condition = "eyes_open" if run == 1 else "eyes_closed"
        starts = np.arange(round(4 * sfreq), raw.n_times - round(8 * sfreq) + 1,
                           round(8 * sfreq), dtype=np.int64)
        intervals = [(index, int(start), int(start + 8 * sfreq), start / sfreq, "T0")
                     for index, start in enumerate(starts)]
        tmin = 0.0
    else:
        condition = "motor_imagery"
        intervals = imagery_intervals(annotations, raw.n_times, sfreq)
        tmin = -2.0
    if not intervals:
        raise ValueError(f"No eligible epochs: subject={subject}, run={run}")
    eeg = np.stack([raw.get_data(start=item[1], stop=item[2]) for item in intervals])
    if not np.isfinite(eeg).all():
        raise ValueError("Nonfinite EEG samples")
    codes = np.asarray([item[4] for item in intervals])
    conditions = (np.where(codes == "T1", "left_hand", "right_hand") if run == 4
                  else np.full(len(intervals), condition))
    metadata = dict(provenance, preprocessing="native EDF; MNE conversion to V only",
                    reference="native acquisition reference; not rereferenced",
                    epoch_interval="half-open", baseline_s=[-2.0, 0.0] if run == 4 else None,
                    first_rest_seconds_discarded=4 if run in (1, 2) else None,
                    response_interpretation="cue-related imagery; not canonical sensory ERP",
                    original_annotations=[dict(onset_s=float(a), duration_s=float(b), code=str(c))
                                          for a, b, c in annotations])
    fields = dict(eeg=eeg.astype(np.float32), sfreq_hz=np.asarray(sfreq), units=np.asarray("V"),
                  ch_names=np.asarray(raw.ch_names), subject=np.asarray(f"S{subject:03d}"),
                  split=np.asarray("train" if subject <= 4 else "test"),
                  condition=np.asarray(condition), event_condition=conditions,
                  run=np.asarray(run), tmin=np.asarray(tmin),
                  trial_ids=np.asarray([f"S{subject:03d}R{run:02d}_annotation{item[0]}" for item in intervals]),
                  start_sample=np.asarray([item[1] for item in intervals]),
                  stop_sample=np.asarray([item[2] for item in intervals]),
                  onsets=np.asarray([item[3] for item in intervals]), annotation_code=codes,
                  metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    assert all(not value.dtype.hasobject for value in fields.values())
    temporary = destination.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **fields)
    with np.load(temporary, allow_pickle=False) as check:
        assert check["eeg"].shape == eeg.shape
        assert np.all(check["stop_sample"] - check["start_sample"] == eeg.shape[-1])
        if run == 4:
            assert eeg.shape[-1] == 800 and float(check["tmin"]) == -2.0
    temporary.replace(destination)
    return dict(file=str(destination), epochs=len(eeg), channels=eeg.shape[1],
                n_times=eeg.shape[2], sfreq_hz=sfreq, tmin_s=tmin,
                conditions={str(key): int(np.sum(conditions == key)) for key in np.unique(conditions)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    root = args.output_dir
    raw_root, prepared_root = root / "raw_physionet", root / "prepared"
    raw_root.mkdir(parents=True, exist_ok=True)
    prepared_root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "physionet_manifest.json"
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    previous = {item["record"]: item for item in existing.get("records", [])}
    manifest = dict(dataset="PhysioNet EEG Motor Movement/Imagery Dataset", version="1.0.0",
                    source_url="https://physionet.org/content/eegmmidb/1.0.0/",
                    doi="10.13026/C28G6P", license="Open Data Commons Attribution License v1.0",
                    license_url="https://physionet.org/content/eegmmidb/view-license/1.0.0/",
                    cohort_fixed_before_feature_analysis=True,
                    train_subjects=[f"S{s:03d}" for s in range(1, 5)],
                    test_subjects=[f"S{s:03d}" for s in range(5, 9)],
                    sha256_scope="Locally computed file-integrity hashes, not publisher-verified checksums",
                    records=[], errors=[])
    for subject in range(1, 9):
        for run in (1, 2, 4):
            record = f"S{subject:03d}R{run:02d}"
            url = f"{BASE_URL}S{subject:03d}/{record}.edf"
            transport_url = f"{TRANSPORT_URL}S{subject:03d}/{record}.edf"
            path = raw_root / f"{record}.edf"
            try:
                digest = download_edf(transport_url, path, previous.get(record, {}).get("sha256"))
                provenance = dict(record=record, url=url, transport_url=transport_url,
                                  sha256=digest, native_sfreq_hz=160.0)
                raw = read_edf(path)
                try:
                    result = export_record(raw, prepared_root / f"{record}.npz", subject=subject, run=run,
                                           provenance=provenance)
                    result.update(duration_s=raw.n_times / raw.info["sfreq"])
                finally:
                    raw.close()
                manifest["records"].append(dict(provenance, **result, edf_bytes=path.stat().st_size))
                print(f"Prepared {record}: {result['epochs']} epochs", flush=True)
            except Exception as exc:
                manifest["errors"].append(dict(record=record, error=f"{type(exc).__name__}: {exc}"))
                print(f"FAILED {record}: {exc}", flush=True)
            temporary = manifest_path.with_suffix(".partial.json")
            temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            temporary.replace(manifest_path)
    if manifest["errors"]:
        raise SystemExit(f"Prepared {len(manifest['records'])}/24 records; see {manifest_path}")


if __name__ == "__main__":
    # Covers rejection of short/gapped/non-rest baselines and clipped task epochs.
    assert imagery_intervals([(0, 2, "T0"), (2, 3, "T1")], 800, 160) == [(1, 0, 800, 2, "T1")]
    assert not imagery_intervals([(0, 1, "T0"), (2, 3, "T1")], 800, 160)
    assert not imagery_intervals([(0, 2, "T2"), (2, 3, "T1")], 800, 160)
    assert not imagery_intervals([(0, 2, "T0"), (2, 2.9, "T1")], 800, 160)
    main()
