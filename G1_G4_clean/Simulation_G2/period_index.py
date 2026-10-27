"""Build reproducible window recipes without decoding or duplicating NMM recordings."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .nmm_bank import read_bank, parent_group_ids
from .nmm_mixing import disjoint_baseline_start
from Simulation_G1.noise_covariance import baseline_sample_count


SPLITS = ("train", "val", "test")
NOISE_TYPES = {"iid_white", "white", "pink", "powerlaw", "specparam_resting",
               "mixed", "eegsourcesim_resting"}
SPATIAL_PROFILES = {"uniform", "decay", "diffuse", "contract", "pulse"}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _parents(entry):
    """Include inherited data identities, even when a trajectory has a new seed."""
    group = entry.get("simulation_group_id")
    if not isinstance(group, str) or not group:
        raise ValueError("Each recording needs a simulation_group_id")
    return set(parent_group_ids(entry))


def _components(records):
    roots = {}

    def find(value):
        roots.setdefault(value, value)
        while roots[value] != value:
            roots[value] = roots[roots[value]]
            value = roots[value]
        return value

    for record in records:
        parents = sorted(record["parents"])
        for parent in parents[1:]:
            a, b = find(parents[0]), find(parent)
            roots[max(a, b)] = min(a, b)
        find(parents[0])
    return {parent: find(parent) for parent in roots}


def _assign_splits(groups, fractions, rng):
    groups = sorted(set(groups))
    positive = np.flatnonzero(fractions)
    if len(groups) < len(positive):
        raise ValueError("Not enough independent parent groups for the requested positive splits")
    desired = len(groups) * np.asarray(fractions)
    counts = np.floor(desired).astype(int)
    for index in np.argsort(-(desired - counts), kind="stable")[:len(groups)-counts.sum()]:
        counts[index] += 1
    for index in positive:
        if counts[index] == 0:
            donor = int(np.argmax(counts))
            counts[donor] -= 1
            counts[index] += 1
    order = rng.permutation(len(groups))
    assignments = {}
    cursor = 0
    for name, count in zip(SPLITS, counts):
        for index in order[cursor:cursor+count]:
            assignments[groups[index]] = name
        cursor += count
    return assignments


def _sample_count(seconds, rate, name, minimum=1):
    value = seconds * rate
    if (isinstance(seconds, bool) or not np.isfinite(value) or value < minimum
            or not np.isclose(value, round(value), rtol=0, atol=1e-6)):
        raise ValueError(f"{name} must cover at least {minimum} samples and lie on the sampling grid")
    return int(round(value))


def _seed(entry):
    seed = entry.get("manifest", {}).get("random_seed")
    if seed is None:
        seed = entry.get("provenance", {}).get("config", {}).get("random_seed")
    return seed if type(seed) is int and seed >= 0 else None


def _resting(entry):
    metadata = entry.get("manifest", {})
    model = metadata.get("model", {})
    if not isinstance(model,dict):
        return False
    background_a, active_a = model.get("background_A"), model.get("active_A")
    return (entry.get("signal_type") == "resting_jr" and entry.get("active_regions") == []
            and metadata.get("normalize") is False and entry.get("source_unit") == "mV_PSP"
            and model.get("class") == "JansenRit" and _seed(entry) is not None
            and isinstance(background_a, (int, float)) and np.isfinite(background_a)
            and active_a == background_a)


def _regions(values, count):
    values = list(values)
    if (not values or len(set(values)) != len(values)
            or any(type(v) is not int or not 0 <= v < count for v in values)):
        raise ValueError("Selected regions must be nonempty, unique integer indices in atlas order")
    return sorted(values)


def _reference(record, start, length):
    entry = record["entry"]
    return dict(bank_index=record["bank_index"], sample_id=entry["sample_id"],
                simulation_group_id=entry["simulation_group_id"], source_sha256=entry["source_sha256"],
                start_sample=int(start), stop_sample=int(start+length))


def _variant_rng(seed, group, start, variant):
    identity = np.frombuffer(hashlib.sha256(group.encode()).digest(), dtype="<u4").tolist()
    return np.random.default_rng(np.random.SeedSequence([seed, *identity, start, variant]))


def build_period_index(bank_dirs, output_dir, *, period_s=1., stride_s=None,
                       tail_policy="drop", variants_per_period=1,
                       split_fractions=(.8, .1, .1), seed=42, signal_types=None, sample_ids=None,
                       source_mode="network", selection="protected", selected_regions=None,
                       source_snr_db=None, background_bank_dirs=(), sensor_snrs=(10.,),
                       sensor_noise_types=("iid_white",), spatial_profiles=("uniform",),
                       decay_mm=12., spatial_start_s=0., spatial_duration_s=.3,
                       average_reference=True, normalize_source=True, baseline_duration_s=1.):
    """Write a JSONL table with byte offsets and parent-disjoint split indices.

    Input banks must already share one sampling rate, atlas, projection and unit.
    Tail windows are never padded. Independent background draws stay in the same
    split and cannot share a parent identity or simulation seed with foreground.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    fractions = np.asarray(split_fractions, dtype=float)
    if (fractions.shape != (3,) or not np.isfinite(fractions).all() or (fractions < 0).any()
            or not np.isclose(fractions.sum(), 1, rtol=0, atol=1e-9)):
        raise ValueError("split_fractions must contain three nonnegative fractions summing to one")
    if type(seed) is not int or seed < 0 or type(variants_per_period) is not int or variants_per_period < 1:
        raise ValueError("Seed must be a nonnegative integer and variants_per_period a positive integer")
    if tail_policy not in {"drop", "error"} or source_mode not in {"network", "selected", "partition"}:
        raise ValueError("Choose drop/error tail policy and network/selected/partition source mode")
    if selection not in {"configured", "protected"}:
        raise ValueError("selection must be configured or protected")
    if source_mode == "network" and selected_regions is not None:
        raise ValueError("Explicit selected regions require selected source mode")
    profiles, noises, snrs = tuple(spatial_profiles), tuple(sensor_noise_types), tuple(sensor_snrs)
    if not profiles or set(profiles)-SPATIAL_PROFILES or not noises or set(noises)-NOISE_TYPES:
        raise ValueError("Unknown or empty spatial profiles / synthetic sensor noise choices")
    if source_mode == "network" and set(profiles) != {"uniform"}:
        raise ValueError("Network mode preserves the atlas projection and requires uniform spatial profiles")
    if not snrs or any(value is not None and (isinstance(value, bool) or not np.isfinite(value)) for value in snrs):
        raise ValueError("sensor_snrs must contain finite SNR values or None (noise disabled)")
    if (not np.isfinite([decay_mm, spatial_start_s, spatial_duration_s]).all()
            or decay_mm < 0 or spatial_start_s < 0 or spatial_duration_s <= 0):
        raise ValueError("Invalid spatial decay, start or duration")
    if type(average_reference) is not bool or type(normalize_source) is not bool:
        raise ValueError("Reference and normalization options must be boolean")
    foreground_paths = sorted({Path(path).resolve() for path in bank_dirs})
    background_paths = sorted({Path(path).resolve() for path in background_bank_dirs})
    if not foreground_paths:
        raise ValueError("At least one foreground bank is required")
    if sample_ids is not None:
        sample_ids=list(sample_ids)
        if not sample_ids or len(set(sample_ids))!=len(sample_ids) or any(not isinstance(s,str) or not s for s in sample_ids):
            raise ValueError("sample_ids must contain distinct nonempty recording IDs")
    if source_snr_db is not None and (isinstance(source_snr_db, bool) or not np.isfinite(source_snr_db)):
        raise ValueError("source_snr_db must be finite or None")
    if source_mode == "partition" and background_paths:
        raise ValueError("Partition uses same-record background and cannot also use external background")
    if source_mode != "partition" and source_snr_db is not None and not background_paths:
        raise ValueError("Network/selected source SNR requires an independent background bank")
    if (source_mode == "partition" or background_paths) and source_snr_db is None:
        raise ValueError("Partition / independent background requires finite source_snr_db")
    paths = sorted(set(foreground_paths+background_paths))
    banks, foreground, background, ancestry = [], [], [], []
    signature = None
    for bank_index, path in enumerate(paths):
        index_hash = _sha256(path/"bank.json")
        bank = read_bank(path)
        if _sha256(path/"bank.json") != index_hash:
            raise ValueError("Bank index changed while building the snapshot")
        if banks and any(bank[key] != banks[0][key] for key in ("atlas_id", "n_regions", "projection_sha256")):
            raise ValueError("All banks must share atlas, projection and region count")
        banks.append(dict(path=str(path), index_sha256=index_hash,
                          atlas_id=bank["atlas_id"], n_regions=bank["n_regions"],
                          projection_sha256=bank["projection_sha256"]))
        for sample_id, entry in sorted(bank["samples"].items()):
            ancestry.append(dict(parents=_parents(entry)))
            is_foreground = (path in foreground_paths and (signal_types is None or entry["signal_type"] in signal_types)
                             and (sample_ids is None or sample_id in sample_ids))
            is_background = path in background_paths and _resting(entry)
            if not is_foreground and not is_background:
                continue
            rate, count = entry.get("native_srate_hz"), entry.get("n_samples")
            if (not isinstance(rate, (float, int)) or isinstance(rate, bool) or not np.isfinite(rate)
                    or rate <= 0 or type(count) is not int or count < 2):
                raise ValueError("Invalid recording sampling rate or sample count")
            current = (bank["atlas_id"], bank["n_regions"], bank["projection_sha256"], rate, entry["source_unit"])
            if signature is not None and current != signature:
                raise ValueError("All selected recordings must share atlas, projection, region count, rate and source unit")
            signature = current
            if entry.get("sample_id") != sample_id or len(entry.get("source_sha256", "")) != 64:
                raise ValueError("Invalid recording identity or source checksum")
            record = dict(bank_index=bank_index, entry=entry, parents=_parents(entry))
            if is_foreground:
                foreground.append(record)
            if is_background:
                background.append(record)
    if not foreground:
        raise ValueError("No foreground recordings match the requested states")
    if sample_ids is not None and set(sample_ids)-{record['entry']['sample_id'] for record in foreground}:
        raise ValueError("Requested sample IDs are absent or excluded by the state filter")
    rate = signature[3]
    baseline_length = baseline_sample_count(baseline_duration_s, rate)
    length = _sample_count(period_s, rate, "period_s", minimum=2)
    stride = _sample_count(period_s if stride_s is None else stride_s, rate, "stride_s")
    discarded, gap_samples = 0, 0
    for record in foreground:
        entry = record["entry"]
        count = entry["n_samples"]
        starts = range(0, max(0, count-length+1), stride)
        if not starts:
            raise ValueError(f"Recording {entry['sample_id']} is shorter than one period")
        tail = count-(starts[-1]+length)
        if tail and tail_policy == "error":
            raise ValueError(f"Recording {entry['sample_id']} leaves {tail} tail samples")
        discarded += tail
        gap_samples += max(0, stride-length)*(len(starts)-1)
        record["starts"] = starts
        if source_mode == "partition":
            for start in starts:
                disjoint_baseline_start(count, start, start+length, baseline_length)
        if source_mode in {"selected", "partition"}:
            values = selected_regions if selected_regions is not None else (
                entry.get("storage", {}).get("protected_regions", entry["active_regions"])
                if selection == "protected" else entry["active_regions"])
            record["selected"] = _regions(values, signature[1])
            if source_mode == "partition" and len(record["selected"]) == signature[1]:
                raise ValueError("Same-record background requires at least one unselected region")
        else:
            record["selected"] = None
    background = [record for record in background if record["entry"]["n_samples"] >= length+baseline_length]
    if background_paths and not background:
        raise ValueError("No eligible raw JansenRit resting background fits the period plus baseline")
    # An excluded state may still connect two included recordings to a shared parent.
    components = _components(ancestry)
    rng = np.random.default_rng(seed)
    foreground_groups = {components[next(iter(record["parents"]))] for record in foreground}
    assignments = _assign_splits(foreground_groups, fractions, rng)
    extra_groups = {components[next(iter(record["parents"]))] for record in background}-foreground_groups
    if extra_groups:
        # Allocate independent backgrounds across positive folds; never borrow across splits.
        assignments.update(_assign_splits(extra_groups, fractions, rng))
    parent_splits = {parent: assignments[component] for parent, component in sorted(components.items())
                     if component in assignments}
    for record in foreground:
        entry = record["entry"]
        record["split"] = parent_splits[entry["simulation_group_id"]]
        record["backgrounds"] = [candidate for candidate in background
            if parent_splits[candidate["entry"]["simulation_group_id"]] == record["split"]
            and components[entry["simulation_group_id"]] != components[candidate["entry"]["simulation_group_id"]]
            and _seed(entry) is not None and _seed(entry) != _seed(candidate["entry"])]
        if background_paths and not record["backgrounds"]:
            raise ValueError(f"No independent same-split background with a different recorded seed for {entry['sample_id']}")
    if any(_sha256(Path(bank["path"])/"bank.json") != bank["index_sha256"] for bank in banks):
        raise ValueError("Bank index changed during recipe preparation")
    output_dir.mkdir(parents=True, exist_ok=False)
    offsets, split_rows = [], {name: [] for name in SPLITS}
    periods = 0
    with (output_dir/"periods.jsonl").open("wb") as handle:
        for record in foreground:
            entry = record["entry"]
            for start in record["starts"]:
                periods += 1
                identity = f"{entry['source_sha256']}:{entry['simulation_group_id']}:{entry['sample_id']}:{rate}:{start}:{start+length}"
                period_id = hashlib.sha256(identity.encode()).hexdigest()
                for variant in range(variants_per_period):
                    vrng = _variant_rng(seed, entry["simulation_group_id"], start, variant)
                    profile = profiles[int(vrng.integers(len(profiles)))]
                    selected_background = None
                    baseline_background = None
                    if source_mode == "partition":
                        baseline_start = disjoint_baseline_start(entry['n_samples'], start, start+length, baseline_length)
                        baseline_background = _reference(record, baseline_start, baseline_length)
                    parents = set(record["parents"])
                    if record["backgrounds"]:
                        candidate = record["backgrounds"][int(vrng.integers(len(record["backgrounds"])))]
                        bg_start = int(vrng.integers(candidate["entry"]["n_samples"]-length-baseline_length+1))
                        selected_background = _reference(candidate, bg_start, length)
                        baseline_background = _reference(candidate, bg_start+length, baseline_length)
                        parents.update(candidate["parents"])
                    config = dict(source_mode=source_mode, selected_regions=record["selected"],
                        baseline_duration_s=float(baseline_duration_s),
                        source_snr_db=None if source_snr_db is None else float(source_snr_db),
                        sensor_noise_type=noises[int(vrng.integers(len(noises)))],
                        sensor_snr_db=snrs[int(vrng.integers(len(snrs)))], sensor_seed=int(vrng.integers(0, 2**32)),
                        spatial_profile=profile, spatial_decay_mm=0. if profile == "uniform" else float(decay_mm),
                        spatial=dict(mode="static" if profile in {"uniform", "decay"} else profile,
                            start_s=float(spatial_start_s), duration_s=float(spatial_duration_s),
                            area_threshold_cm2=None, hop_threshold=None),
                        average_reference=average_reference, normalize_source=normalize_source)
                    if config["sensor_snr_db"] is not None:
                        config["sensor_snr_db"] = float(config["sensor_snr_db"])
                    row = dict(period_id=period_id, recipe_id=f"{period_id}-{variant:04d}", split=record["split"],
                        foreground=_reference(record, start, length), background=selected_background,
                        baseline_background=baseline_background,
                        parent_data_group_ids=sorted(parents), config=config)
                    split_rows[record["split"]].append(len(offsets))
                    offsets.append(handle.tell())
                    handle.write((json.dumps(row, separators=(",", ":"), allow_nan=False)+"\n").encode())
    np.save(output_dir/"offsets.npy", np.asarray(offsets, dtype=np.int64), allow_pickle=False)
    splits = {}
    for name, indices in split_rows.items():
        filename = f"{name}_indices.npy"
        np.save(output_dir/filename, np.asarray(indices, dtype=np.int64), allow_pickle=False)
        splits[name] = dict(file=filename, sha256=_sha256(output_dir/filename), count=len(indices))
    manifest = dict(schema="nmm-periods-v1", banks=banks, rate_hz=rate,
        period_samples=length, stride_samples=stride, tail_policy=tail_policy,
        table=dict(file="periods.jsonl", index="offsets.npy", sha256=_sha256(output_dir/"periods.jsonl"),
                   index_sha256=_sha256(output_dir/"offsets.npy"), count=len(offsets)),
        splits=splits, parent_splits=parent_splits, seed=seed,
        statistics=dict(recordings=len(foreground), periods=periods, variants_per_period=variants_per_period,
                        discarded_tail_samples=discarded, stride_gap_samples=gap_samples),
        source_unit=signature[4], background_policy="independent same-split raw JR rest" if background_paths else "same-record partition" if source_mode == "partition" else "none" if source_mode == "selected" else "original network")
    (output_dir/"dataset.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--banks", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--period-s", type=float, default=1.)
    parser.add_argument("--baseline-duration-s", type=float, default=1.)
    parser.add_argument("--stride-s", type=float)
    parser.add_argument("--tail-policy", choices=("drop", "error"), default="drop")
    parser.add_argument("--variants-per-period", type=int, default=1)
    parser.add_argument("--split-fractions", nargs=3, type=float, default=(.8, .1, .1))
    parser.add_argument("--signal-types", nargs="+")
    parser.add_argument("--sample-ids", nargs="+")
    parser.add_argument("--background-banks", nargs="*", default=[])
    parser.add_argument("--source-mode", choices=("network", "selected", "partition"), default="network")
    parser.add_argument("--selection", choices=("protected", "configured"), default="protected")
    parser.add_argument("--regions", nargs="+", type=int)
    parser.add_argument("--source-snr-db", type=float)
    sensor_group=parser.add_mutually_exclusive_group()
    sensor_group.add_argument("--sensor-snrs", nargs="+", type=float, default=[10.])
    sensor_group.add_argument("--no-sensor-noise", action="store_true")
    parser.add_argument("--sensor-noise-types", nargs="+", choices=sorted(NOISE_TYPES), default=["iid_white"])
    parser.add_argument("--spatial-profiles", nargs="+", choices=sorted(SPATIAL_PROFILES), default=["uniform"])
    parser.add_argument("--decay-mm", type=float, default=12.)
    parser.add_argument("--spatial-start-s", type=float, default=0.)
    parser.add_argument("--spatial-duration-s", type=float, default=.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-average-reference", action="store_true")
    parser.add_argument("--no-normalize-source", action="store_true")
    args = vars(parser.parse_args(argv))
    args["bank_dirs"] = args.pop("banks")
    args["background_bank_dirs"] = args.pop("background_banks")
    args["selected_regions"] = args.pop("regions")
    args["average_reference"] = not args.pop("no_average_reference")
    args["normalize_source"] = not args.pop("no_normalize_source")
    if args.pop("no_sensor_noise"):
        args["sensor_snrs"]=(None,)
    result = build_period_index(**args)
    print(json.dumps({"output_dir": args["output_dir"], **result["statistics"]}, indent=2))


if __name__ == "__main__":
    main()
