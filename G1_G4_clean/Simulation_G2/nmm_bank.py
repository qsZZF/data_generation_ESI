"""Versioned regional NMM recordings: G3 writes, G2 replays without TVB."""
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np
from scipy.signal import resample_poly
from scipy.sparse import csc_matrix

from Simulation_G2.nmm_codec import (decode_regions, encode_regions, _protected_indices,
                                     validate_regions_payload)


def projection_hash(matrix):
    matrix = csc_matrix(matrix, dtype=np.float64, copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    digest = hashlib.sha256()
    for values, dtype in ((matrix.shape, "<i8"), (matrix.indptr, "<i8"),
                          (matrix.indices, "<i8"), (matrix.data, "<f8")):
        digest.update(np.asarray(values, dtype=dtype).tobytes())
    return digest.hexdigest()


def _write_json(path, value):
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(".part")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _valid_id(value):
    return (isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", value)
            and not re.fullmatch(r"(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])", value))


def read_bank(path):
    bank = json.loads((Path(path) / "bank.json").read_text(encoding="utf-8"))
    if (type(bank.get("schema_version")) is not int or bank["schema_version"] not in (1, 2)
            or not isinstance(bank.get("samples"), dict)
            or not isinstance(bank.get("n_regions"), int) or isinstance(bank["n_regions"], bool)
            or bank["n_regions"] < 1 or not isinstance(bank.get("atlas_id"), str)
            or not re.fullmatch(r"[a-f0-9]{64}", str(bank.get("projection_sha256", "")))):
        raise ValueError("Invalid NMM bank header")
    return bank


def parent_group_ids(entry):
    """Recover declared trajectory/data ancestry, including historical conversion wrappers."""
    parents=set()
    pending=[entry]
    seen=set()
    while pending:
        metadata=pending.pop()
        if not isinstance(metadata,dict):
            raise ValueError("Parent provenance must be an object")
        if id(metadata) in seen:
            continue
        seen.add(id(metadata))
        for key in ("simulation_group_id","parent_simulation_group_id"):
            if key in metadata:
                value=metadata[key]
                if not isinstance(value,str) or not value:
                    raise ValueError("Parent group identity must be a nonempty string")
                parents.add(value)
        for key in ("parent_data_group_ids","parent_simulation_group_ids"):
            values=metadata.get(key,[])
            if not isinstance(values,list) or any(not isinstance(v,str) or not v for v in values):
                raise ValueError("Parent group identities must be nonempty strings")
            parents.update(values)
        for key in ("manifest","provenance","original_provenance"):
            if key in metadata: pending.append(metadata[key])
    return sorted(parents)


def initialize_bank(path, *, atlas_id, n_regions, projection_sha256, schema_version=1):
    if (not isinstance(n_regions, (int, np.integer)) or isinstance(n_regions, bool) or n_regions < 1
            or not isinstance(atlas_id, str) or not atlas_id
            or not isinstance(projection_sha256, str)
            or not re.fullmatch(r"[a-f0-9]{64}", projection_sha256)
            or type(schema_version) is not int or schema_version not in (1, 2)):
        raise ValueError("Bank requires an atlas identity, positive region count and projection hash")
    path = Path(path)
    expected = dict(schema_version=schema_version, atlas_id=atlas_id, n_regions=int(n_regions), projection_sha256=projection_sha256)
    if (path / "bank.json").exists():
        bank = read_bank(path)
        if any(bank.get(key) != value for key, value in expected.items()):
            raise ValueError("Existing NMM bank atlas/projection differs; use another directory")
        return bank
    if path.exists() and any(path.glob("*.npz")):
        raise ValueError("Unidentified orphan recordings exist without bank.json")
    path.mkdir(parents=True, exist_ok=True)
    bank = {**expected, "samples": {}}
    _write_json(path / "bank.json", bank)
    return bank


def _validate_clock(source, times, rate, n_regions):
    if (source.ndim != 2 or source.shape[0] != n_regions or source.shape[1] < 2
            or times.shape != (source.shape[1],) or not np.isfinite(source).all()
            or not np.isfinite(times).all() or not np.isfinite(rate) or rate <= 0):
        raise ValueError("Invalid NMM source shape, clock, rate or finite values")
    expected = np.arange(source.shape[1], dtype=float) * (1000 / rate)
    if not np.allclose(times, expected, rtol=1e-7, atol=1e-6):
        raise ValueError("NMM bank clock must start at zero and match its sampling rate")


def save_bank_sample(path, sample_id, region_source, time_ms, manifest, provenance, *,
                     storage=None, simulation_group_id=None):
    """Commit one complete recording. One writer per bank; no implicit overwrite."""
    path = Path(path)
    bank = read_bank(path)
    if not _valid_id(sample_id):
        raise ValueError("sample_id must be a safe, nonempty filename identifier")
    if sample_id.casefold() in {name.casefold() for name in bank["samples"]}:
        raise ValueError(f"NMM sample already exists: {sample_id}")
    target = path / f"{sample_id}.npz"
    if any(file.name.casefold() == target.name.casefold() for file in path.glob("*.npz")):
        raise ValueError(f"Unindexed NMM recording already exists: {sample_id}")
    if storage is not None and bank["schema_version"] != 2:
        raise ValueError("Regional storage codecs require a schema 2 bank")
    if simulation_group_id is not None and (not isinstance(simulation_group_id, str)
            or not re.fullmatch(r"[a-f0-9]{64}", simulation_group_id)):
        raise ValueError("simulation_group_id must be a lowercase SHA256 identifier")
    source = np.asarray(region_source)
    times = np.asarray(time_ms, dtype=float)
    rate = float(manifest["output_srate_hz"])
    _validate_clock(source, times, rate, bank["n_regions"])
    if manifest.get("n_regions_output", source.shape[0]) != source.shape[0]:
        raise ValueError("Manifest region count differs from recording")
    signal_type = manifest.get("signal_type")
    active = manifest.get("active_regions", [])
    if (not isinstance(signal_type, str) or not signal_type or len(set(active)) != len(active)
            or any(not isinstance(i, (int, np.integer)) or isinstance(i, bool)
                   or i < 0 or i >= source.shape[0] for i in active)):
        raise ValueError("Invalid signal type or configured region indices")
    unit = ("normalized_model_state" if manifest.get("normalize", False) else
            "mV_PSP" if manifest.get("model", {}).get("class") == "JansenRit" else "model_state")
    record = dict(sample_id=sample_id, signal_type=signal_type, active_regions=[int(i) for i in active],
                  native_srate_hz=rate, n_samples=source.shape[1], source_unit=unit,
                  atlas_id=bank["atlas_id"], projection_sha256=bank["projection_sha256"],
                  manifest=manifest, provenance=provenance)
    if simulation_group_id is not None:
        group_bytes = None
    elif isinstance(provenance.get("config"), dict):
        # Keep crop/normalization/duration views of a seeded parameter draw in one split.
        view_fields = {"sample_id", "connectivity_path", "simulation_length_ms", "warmup_ms", "discard_ms",
                       "output_srate_hz", "normalize", "reshape_to_994", "dt_ms", "jr_coupling_mode"}
        family = {**provenance, "config": {key: value for key, value in provenance["config"].items()
                                           if key not in view_fields}}
        if "active_regions" in family["config"]:
            family["config"]["active_regions"] = sorted(family["config"]["active_regions"])
        group_bytes = json.dumps(family, sort_keys=True, allow_nan=False).encode()
    else:
        group_bytes = np.ascontiguousarray(source, dtype="<f4").tobytes() + times.astype("<f8").tobytes()
    record["simulation_group_id"] = simulation_group_id or hashlib.sha256(group_bytes).hexdigest()
    if bank["schema_version"] == 2:
        if storage is not None and not isinstance(storage, dict):
            raise ValueError("Regional storage options must be a dictionary")
        options = {"codec": "dense"} if storage is None else dict(storage)
        protected = _protected_indices(options.get("protected_regions", []), bank["n_regions"]).tolist()
        options["protected_regions"] = protected + [int(i) for i in active if i not in protected]
        payload, record["storage"] = encode_regions(source, options)
    else:
        with np.errstate(over="raise", invalid="raise"):
            payload = {"region_source": source.astype(np.float32)}
    record_text = json.dumps(record, sort_keys=True, allow_nan=False)
    record = json.loads(record_text)
    temporary = target.with_suffix(".part")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload, time_ms=times, record_json=record_text)
    temporary.replace(target)
    entry = {**record, "source_file": target.name, "source_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    bank["samples"][sample_id] = entry
    _write_json(path / "bank.json", bank)
    return entry


def load_bank_sample(path, sample_id=None, *, signal_type=None, rng=None):
    bank = read_bank(path)
    if sample_id is None:
        choices = sorted(key for key, entry in bank["samples"].items()
                         if signal_type is None or entry["signal_type"] == signal_type)
        if not choices:
            raise ValueError("No NMM bank samples match the requested state")
        rng = np.random.default_rng() if rng is None else rng
        sample_id = choices[int(rng.integers(len(choices)))]
    payload, times, entry = load_bank_payload(path, sample_id, bank=bank)
    if signal_type is not None and entry["signal_type"] != signal_type:
        raise ValueError("Selected NMM sample does not match requested state")
    source = (decode_regions(payload, entry["storage"], bank["n_regions"], entry["n_samples"])
              if bank["schema_version"] == 2 else payload["region_source"])
    return source, times, entry


def load_bank_payload(path, sample_id, *, bank=None):
    """Read and verify one complete NPZ payload without decoding regional factors.

    ``bank`` may be a previously validated index snapshot. Callers retaining it
    must reject index changes. Payload arrays are owned by the caller.
    """
    bank = read_bank(path) if bank is None else bank
    if not _valid_id(sample_id):
        raise ValueError("Invalid NMM bank sample identifier")
    if sample_id not in bank["samples"]:
        raise ValueError(f"Unknown NMM bank sample: {sample_id}")
    entry = bank["samples"][sample_id]
    if entry.get("source_file") != f"{sample_id}.npz":
        raise ValueError("Unsafe source path in NMM bank")
    source_path = Path(path) / entry["source_file"]
    if not source_path.resolve().is_relative_to(Path(path).resolve()):
        raise ValueError("NMM bank recording escapes its directory")
    encoded = source_path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != entry["source_sha256"]:
        raise ValueError("NMM bank recording checksum mismatch")
    # Decode the exact bytes we hashed, even if an external writer replaces the file.
    with np.load(io.BytesIO(encoded), allow_pickle=False) as data:
        record = json.loads(str(data["record_json"]))
        if record != {key: value for key, value in entry.items() if key not in {"source_file", "source_sha256"}}:
            raise ValueError("NMM bank metadata differs from its recording")
        payload = {name: data[name] for name in data.files if name not in {"record_json", "time_ms"}}
        times = data["time_ms"]
    if (record["sample_id"] != sample_id or record["atlas_id"] != bank["atlas_id"]
            or record["projection_sha256"] != bank["projection_sha256"]
            or record.get("source_unit") not in {"normalized_model_state", "mV_PSP", "model_state"}
            or not re.fullmatch(r"[a-f0-9]{64}", str(record.get("simulation_group_id", "")))):
        raise ValueError("NMM recording identity differs from bank")
    n_samples, rate = record["n_samples"], record["native_srate_hz"]
    if (type(n_samples) is not int or n_samples < 2 or times.shape != (n_samples,)
            or not isinstance(rate, (int, float)) or isinstance(rate, bool)
            or not np.isfinite(rate) or rate <= 0 or not np.isfinite(times).all()
            or not np.allclose(times, np.arange(n_samples) * (1000 / rate), rtol=1e-7, atol=1e-6)):
        raise ValueError("Invalid NMM bank clock or sampling rate")
    active = _protected_indices(record["active_regions"], bank["n_regions"])
    if bank["schema_version"] == 2:
        validate_regions_payload(payload, record.get("storage"), bank["n_regions"], n_samples)
        if not set(active).issubset(record["storage"]["protected_regions"]):
            raise ValueError("Configured regions must be protected in a schema 2 bank")
    else:
        if "storage" in record or set(payload) != {"region_source"}:
            raise ValueError("Invalid schema 1 regional payload")
        source = payload["region_source"]
        if source.shape != (bank["n_regions"], n_samples) or source.dtype != np.float32:
            raise ValueError("Invalid schema 1 regional dimensions or dtype")
        _validate_clock(source, times, rate, bank["n_regions"])
    return payload, times, entry


def prepare_bank_segment(source, time_ms, *, srate, duration_s, start_s=0):
    """Crop in native physical time, then resample; never stretch a recording."""
    source, times = np.asarray(source), np.asarray(time_ms, dtype=float)
    if times.ndim != 1 or len(times) < 2 or times[1] <= times[0]:
        raise ValueError("Invalid NMM bank clock")
    native_rate = 1000 / (times[1] - times[0])
    _validate_clock(source, times, native_rate, source.shape[0] if source.ndim == 2 else 0)
    if not all(np.isfinite(x) for x in (srate, duration_s, start_s)) or srate <= 0 or duration_s <= 0 or start_s < 0:
        raise ValueError("Rates/duration must be positive and offset nonnegative")
    counts = [start_s * native_rate, duration_s * native_rate, duration_s * srate]
    if any(not np.isclose(x, round(x), rtol=0, atol=1e-6) for x in counts):
        raise ValueError("Segment boundaries and duration must fall on the sample grids")
    start, length, output_length = map(lambda x: int(round(x)), counts)
    if length < 2 or output_length < 2 or start + length > source.shape[1]:
        raise ValueError("Requested duration/offset does not fit NMM bank recording")
    segment = source[:, start:start + length]
    ratio = Fraction(float(srate / native_rate)).limit_denominator(1_000_000)
    if not np.isclose(float(ratio), srate / native_rate, rtol=1e-10, atol=0):
        raise ValueError("Inaccurate resampling ratio")
    if ratio != 1:
        segment = resample_poly(segment, ratio.numerator, ratio.denominator, axis=-1, padtype="line")
    if segment.shape[1] != output_length or not np.isfinite(segment).all():
        raise ValueError("Invalid resampled NMM segment")
    return segment.copy()
