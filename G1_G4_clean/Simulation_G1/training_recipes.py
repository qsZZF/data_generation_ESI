"""Build configuration-only G1 datasets and replay indexed recipes on demand."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, OrderedDict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy

from .g1_simulation_core import G1SimulationConfig, G1ScenarioSampler, G1SourceSimulator

SCHEMA = "g1-recipes-v1"
STREAMS = ("spatial", "waveform", "gain", "background", "sensor")
PROFILES = ("uniform", "decay", "diffuse", "contract", "pulse")
# label, fraction, possible source counts, area range(cm2), sensor SNR, source SNR, seed spacing(mm)
LEVELS = (
    ("easy", .20, (1, 1, 1, 2, 2), (.5, 4.), (15., 30.), (25., 40.), 30.),
    ("medium", .35, (2, 3, 4), (.5, 8.), (5., 20.), (15., 30.), 25.),
    ("hard", .35, (3, 4, 5, 6), (.25, 12.), (-5., 10.), (8., 20.), 20.),
    ("extreme", .10, (4, 5, 6, 7, 8), (.25, 20.), (-10., 5.), (0., 12.), 15.),
)


def _seed(root, *keys):
    return int(np.random.SeedSequence([root, *keys]).generate_state(1)[0])


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _array_hash(*arrays):
    result = hashlib.sha256()
    for value in arrays:
        value = np.ascontiguousarray(value)
        result.update(str((value.shape, value.dtype.str)).encode())
        result.update(value.tobytes())
    return result.hexdigest()


def asset_identity(simulator):
    conn = simulator.vert_conn
    return {"mesh_sha256": _array_hash(simulator.vertices, simulator.faces, conn.indptr, conn.indices, conn.data),
            "leadfield_sha256": _array_hash(simulator.leadfield),
            "n_vertices": simulator.n_sources, "n_sensors": simulator.n_sensors,
            "coordinate_scale_to_mm": simulator.coord_to_mm}


def generator_identity():
    folder = Path(__file__).resolve().parent
    return {"files": {name: _digest(folder/name) for name in
                      ("g1_simulation_core.py", "spatial.py", "training_recipes.py", "noise_covariance.py")},
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}


def _counts(total, fractions):
    raw = np.asarray(fractions)*total
    counts = np.floor(raw).astype(int)
    for index in np.argsort(-(raw-counts), kind="stable")[:total-int(counts.sum())]:
        counts[index] += 1
    return counts


def _write_rows(path, rows):
    offsets = []
    with path.open("xb") as handle:
        for row in rows:
            offsets.append(handle.tell())
            handle.write((_json(row)+"\n").encode("utf-8"))
    index_path = path.with_suffix(".offsets.npy")
    np.save(index_path, np.asarray(offsets, dtype=np.int64), allow_pickle=False)
    return {"file": path.name, "index": index_path.name, "count": len(offsets),
            "sha256": _digest(path), "index_sha256": _digest(index_path)}


def _variant(support, index, base, dataset_seed, families):
    # Five profile variants can share a temporal realization without repeating geometry.
    group = index//len(PROFILES)
    root = _seed(dataset_seed, support["index"], group, 23)
    rng = np.random.default_rng(root)
    sampler = G1ScenarioSampler(random_seed=_seed(root, 1))
    level = next(item for item in LEVELS if item[0] == support["difficulty"])
    count = len(support["seed_vertices"])
    family = families[(support["index"]+group) % len(families)]
    arrangement = ("same_family", "mixed", "synchronous", "mixed")[group % 4] if count > 1 else "single"
    kinds = [family]*count if arrangement != "mixed" else [families[(families.index(family)+i) % len(families)] for i in range(count)]
    waveforms = [sampler._sample_waveform(kind) for kind in kinds]
    t0, t1 = base["time_window"]
    duration = t1-t0
    event_times = []
    for kind, params in waveforms:
        if kind == "erp":
            for component in params["components"]:
                component["latency_s"] = float(t0+rng.uniform(.25, .65)*duration)
                event_times.append(component["latency_s"]-t0)
        elif kind == "spike":
            params["latency_s"] = float(t0+rng.uniform(.25, .65)*duration)
            event_times.append(params["latency_s"]-t0)
        elif kind == "ersp" and params["modulation"]["type"] != "ampmod":
            params["modulation"]["latency_s"] = float(t0+rng.uniform(.25, .65)*duration)
            event_times.append(params["modulation"]["latency_s"]-t0)
    mode = "independent"
    coupling = 0.
    if arrangement == "synchronous":
        mode = "synchronous"
        waveforms = [waveforms[0]]*count
    elif count > 1 and level[0] != "easy":
        mode = ("independent", "correlated", "causal")[(support["index"]+group) % 3]
        if mode == "correlated":
            coupling = float(rng.uniform(.2, .85))
        elif mode == "causal":
            coupling = rng.uniform(-.15, .15, count*count).tolist()
    profile = PROFILES[index % len(PROFILES)]
    decays = rng.uniform(5., 25., count).tolist()
    if profile == "uniform":
        decays = [0.]*count
    transition = float(rng.uniform(.15, .45)*duration)
    center = float(np.median(event_times)) if event_times else .5*duration
    start = float(np.clip(center-transition/2, 0., duration-transition))
    variant_seed = _seed(dataset_seed, support["index"], index, 31)
    noise_rng = np.random.default_rng(variant_seed)
    bg_type = str(rng.choice(["none", "pink", "powerlaw", "specparam", "alpha", "synaptic"]))
    config = dict(
        sample_id=f"g1_{support['support_id'][:16]}_{index:03d}",
        random_seed=variant_seed,
        component_seeds={name: _seed(root, stream) for stream, name in enumerate(STREAMS)},
        patch_gains=[1.]*count if arrangement == "synchronous" else rng.uniform(.9, 1.1, count).tolist(),
        decay_mm=decays, waveform_params=waveforms, connectivity_mode=mode, coupling=coupling,
        spatial=dict(mode="static" if profile in {"uniform", "decay"} else profile,
                     area_threshold_cm2=None, hop_threshold=None, start_s=start, duration_s=transition),
        source_background_type=None if bg_type == "none" else bg_type,
        source_snr_db=None if bg_type == "none" else float(rng.uniform(*level[5])),
        sensor_noise_type=str(noise_rng.choice(["iid_white", "white", "pink", "powerlaw", "specparam_resting", "mixed"])),
        sensor_snr_db=float(noise_rng.uniform(*level[4])),
    )
    config["component_seeds"]["sensor"] = _seed(variant_seed, 4)
    return dict(support_index=support["index"], support_id=support["support_id"], split=support["split"],
                difficulty=support["difficulty"], variant_index=index, temporal_group=group,
                spatial_profile=profile, waveform_arrangement=arrangement, config=config)


def build_recipe_dataset(simulator, output_dir, *, n_supports=25000, variants_per_support=32,
                         eval_variants_per_support=16, seed=42, srate=250., duration_s=2.,
                         shard_size=4096, sensor_noise_peak_ratio=1., resource_paths=None):
    """Write recipes only; no source signals, sensor arrays or PSDs are computed."""
    for name, value in (("n_supports", n_supports), ("variants_per_support", variants_per_support),
                        ("eval_variants_per_support", eval_variants_per_support), ("shard_size", shard_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not np.isfinite(srate) or srate < 100 or not np.isfinite(duration_s) or duration_s < 1:
        raise ValueError("The current waveform plan requires srate>=100 and duration_s>=1")
    if not np.isclose(srate*duration_s, round(srate*duration_s)):
        raise ValueError("duration_s must contain whole samples")
    if sensor_noise_peak_ratio is not None and (not np.isfinite(sensor_noise_peak_ratio) or sensor_noise_peak_ratio <= 0):
        raise ValueError("sensor_noise_peak_ratio must be positive or None")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    generator_version = generator_identity()
    snapshot = output/"generator_snapshot"/"Simulation_G1"
    snapshot.mkdir(parents=True)
    for name in generator_version["files"]:
        (snapshot/name).write_bytes((Path(__file__).resolve().parent/name).read_bytes())
    assets = asset_identity(simulator)
    base = asdict(G1SimulationConfig(srate=float(srate), time_window=(0., float(duration_s)),
        source_space_id=assets["mesh_sha256"], leadfield_id=assets["leadfield_sha256"],
        sensor_snr_mode="global_rms", sensor_noise_peak_ratio=sensor_noise_peak_ratio))
    split_counts = _counts(n_supports, [.8, .1, .1])
    splits = np.repeat(["train", "val", "test"], split_counts)
    np.random.default_rng(_seed(seed, 0)).shuffle(splits)
    levels = []
    for label in ("train", "val", "test"):
        indices = np.flatnonzero(splits == label)
        names = np.repeat([item[0] for item in LEVELS], _counts(len(indices), [item[1] for item in LEVELS]))
        np.random.default_rng(_seed(seed, 1, len(levels))).shuffle(names)
        levels.extend(zip(indices.tolist(), names.tolist()))
    difficulty = dict(levels)
    supports, seen = [], set()
    for index in range(n_supports):
        level = next(item for item in LEVELS if item[0] == difficulty[index])
        for attempt in range(32):
            geometry_seed = _seed(seed, index, attempt, 17)
            rng = np.random.default_rng(geometry_seed)
            count = int(rng.choice(level[2]))
            areas = (100*np.exp(rng.uniform(np.log(level[3][0]), np.log(level[3][1]), count))).tolist()
            geometry = G1SimulationConfig(n_patch=count, patch_area_mm2=areas, decay_mm=0.,
                min_seed_distance_mm=level[6], random_seed=geometry_seed)
            try:
                centers, patches, _ = simulator._resolve_spatial_patches(geometry, rng)
            except RuntimeError:
                continue
            identity = dict(mesh=assets["mesh_sha256"], vertices=sorted(sorted(p.tolist()) for p in patches))
            support_id = hashlib.sha256(_json(identity).encode()).hexdigest()
            if support_id in seen:
                continue
            seen.add(support_id)
            supports.append(dict(index=index, support_id=support_id, split=str(splits[index]),
                difficulty=level[0], spatial_seed=geometry_seed,
                seed_vertices=centers.tolist(), patch_vertices=[p.tolist() for p in patches],
                patch_area_mm2=areas, min_seed_distance_mm=level[6],
                actual_area_mm2=[float(simulator.vertex_area_coord2[p].sum()*simulator.coord_to_mm**2) for p in patches]))
            break
        else:
            raise RuntimeError(f"Could not sample unique disjoint support {index}; reduce area/count or use a larger mesh")
    support_record = _write_rows(output/"supports.jsonl", supports)
    families = G1ScenarioSampler().waveforms
    batch, shards, total = [], [], 0
    split_indices = {label: [] for label in ("train", "val", "test")}
    counts = {name: Counter() for name in ("split", "difficulty", "source_count", "spatial_profile", "waveform_family", "connectivity_mode")}
    for support in supports:
        count = variants_per_support if support["split"] == "train" else eval_variants_per_support
        for variant_index in range(count):
            row = _variant(support, variant_index, base, seed, families)
            for key in ("split", "difficulty", "spatial_profile"):
                counts[key][row[key]] += 1
            counts["source_count"][len(support["seed_vertices"])]+=1
            counts["connectivity_mode"][row["config"]["connectivity_mode"]]+=1
            for family, _ in row["config"]["waveform_params"]:
                counts["waveform_family"][family]+=1
            split_indices[row["split"]].append(total+len(batch))
            batch.append(row)
            if len(batch) == shard_size:
                entry = _write_rows(output/f"recipes-{len(shards):05d}.jsonl", batch)
                shards.append(dict(start=total, **entry)); total += len(batch); batch=[]
    if batch:
        entry = _write_rows(output/f"recipes-{len(shards):05d}.jsonl", batch)
        shards.append(dict(start=total, **entry)); total += len(batch)
    split_indexes = {}
    for label, indices in split_indices.items():
        path = output/f"{label}.indices.npy"
        np.save(path, np.asarray(indices, dtype=np.int64), allow_pickle=False)
        split_indexes[label] = dict(file=path.name, sha256=_digest(path), count=len(indices))
    manifest = dict(schema=SCHEMA, generator=generator_version, assets=assets,
        resource_paths=resource_paths or {}, base_config=base, dataset_seed=seed,
        n_supports=n_supports, n_samples=total, variants_per_support=variants_per_support,
        eval_variants_per_support=eval_variants_per_support,
        split_support_counts=dict(zip(("train", "val", "test"), map(int, split_counts))),
        support_table=support_record, shards=shards, split_indexes=split_indexes,
        counts={k: dict(v) for k,v in counts.items()},
        difficulty_definition="nominal requested SNR/source count/geometry stratum; actual capped SNR is reported at replay",
        real_sensor_noise_enabled=False, metadata_only=True,
        split_policy="exact foreground-support identity held out; not anatomical-region or forward-model holdout",
        generator_snapshot="generator_snapshot/Simulation_G1")
    (output/"dataset.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return manifest


class RecipeDataset:
    """Random access with one resident simulator; returns G1's five output objects."""
    def __init__(self, path, simulator, *, split="train"):
        self.path = Path(path)
        self.manifest = json.loads((self.path/"dataset.json").read_text(encoding="utf-8"))
        if self.manifest["schema"] != SCHEMA:
            raise ValueError("Unsupported G1 recipe schema")
        if self.manifest["generator"] != generator_identity():
            raise ValueError("Generator code or environment differs from the frozen recipe version")
        if self.manifest["assets"] != asset_identity(simulator):
            raise ValueError("Recipe mesh/leadfield differs from the supplied simulator")
        self.simulator = simulator
        self._indexes, self._support_cache = {}, OrderedDict()
        self._starts = [row["start"] for row in self.manifest["shards"]]
        self._selection = None
        if split is not None:
            if split not in self.manifest["split_indexes"]:
                raise ValueError("split must be train, val, test or None for audit")
            entry = self.manifest["split_indexes"][split]
            if _digest(self.path/entry["file"]) != entry["sha256"]:
                raise ValueError("Split index checksum mismatch")
            self._selection = np.load(self.path/entry["file"], allow_pickle=False)

    def __len__(self):
        return self.manifest["n_samples"] if self._selection is None else len(self._selection)

    def _read(self, table, index):
        key = table["file"]
        if key not in self._indexes:
            for name, digest_key in (("file", "sha256"), ("index", "index_sha256")):
                if _digest(self.path/table[name]) != table[digest_key]:
                    raise ValueError(f"Recipe checksum mismatch: {table[name]}")
            self._indexes[key] = np.load(self.path/table["index"], allow_pickle=False)
        offsets = self._indexes[key]
        with (self.path/table["file"]).open("rb") as handle:
            handle.seek(int(offsets[index]))
            return json.loads(handle.readline())

    def recipe(self, index):
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or not 0 <= index < len(self):
            raise IndexError("Recipe index outside dataset")
        if self._selection is not None:
            index = int(self._selection[index])
        shard = self.manifest["shards"][bisect_right(self._starts, index)-1]
        row = self._read(shard, index-shard["start"])
        support_index = row["support_index"]
        if support_index not in self._support_cache:
            self._support_cache[support_index] = self._read(self.manifest["support_table"], support_index)
            if len(self._support_cache) > 128:
                self._support_cache.popitem(last=False)
        support = self._support_cache[support_index]
        self._support_cache.move_to_end(support_index)
        if support["support_id"] != row["support_id"] or support["split"] != row["split"]:
            raise ValueError("Recipe support/split identity mismatch")
        config = {**self.manifest["base_config"], **row["config"],
                  "n_patch": len(support["seed_vertices"]),
                  **{key: support[key] for key in ("seed_vertices", "patch_vertices", "patch_area_mm2", "min_seed_distance_mm")}}
        return G1SimulationConfig(**config), row

    def __getitem__(self, index):
        config, row = self.recipe(index)
        sensor, clean, noise, source, metadata = self.simulator.generate_sensor(config, compute_diagnostics=False)
        metadata.update(recipe_schema=SCHEMA, support_id=row["support_id"], split=row["split"],
                        difficulty=row["difficulty"], variant_index=row["variant_index"],
                        temporal_group=row["temporal_group"], spatial_profile=row["spatial_profile"])
        return sensor, clean, noise, source, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-supports", type=int, default=25000)
    parser.add_argument("--variants-per-support", type=int, default=32)
    parser.add_argument("--eval-variants-per-support", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--srate", type=float, default=250.)
    parser.add_argument("--duration-s", type=float, default=2.)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--sensor-noise-peak-ratio", type=float, default=1., help="0 disables the peak limit")
    anatomy = Path(__file__).resolve().parents[1]/"Anatomy"
    parser.add_argument("--cortex", type=Path, default=anatomy/"Cortex_FS_20k.mat")
    parser.add_argument("--leadfield", type=Path, default=anatomy/"Gain_20k_FS.mat")
    args = parser.parse_args()
    from Simulation_G2.g2_simulation_core import load_brainstorm_cortex, load_brainstorm_leadfield
    cortex = load_brainstorm_cortex(args.cortex)
    leadfield = load_brainstorm_leadfield(args.leadfield, n_vertices=np.asarray(cortex["Vertices"]).shape[-1])
    simulator = G1SourceSimulator.from_brainstorm_cortex(leadfield, cortex)
    manifest = build_recipe_dataset(simulator, args.output_dir, n_supports=args.n_supports,
        variants_per_support=args.variants_per_support, eval_variants_per_support=args.eval_variants_per_support,
        seed=args.seed, srate=args.srate, duration_s=args.duration_s, shard_size=args.shard_size,
        sensor_noise_peak_ratio=None if args.sensor_noise_peak_ratio == 0 else args.sensor_noise_peak_ratio,
        resource_paths={"cortex": str(args.cortex.resolve()), "leadfield": str(args.leadfield.resolve())})
    print(json.dumps({key: manifest[key] for key in ("n_supports", "n_samples", "split_support_counts", "counts")}, indent=2))


if __name__ == "__main__":
    main()
