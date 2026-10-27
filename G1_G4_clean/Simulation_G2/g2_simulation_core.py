from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import h5py
import numpy as np
import scipy.io
import scipy.signal
import scipy.sparse


if __package__ in {None, ""}:
    root_dir = Path(__file__).resolve().parents[1]
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

from Simulation_G2.nmm_bank import load_bank_sample, prepare_bank_segment, projection_hash, read_bank
from Simulation_G1.noise_covariance import (NOISE_ARRAY_KEYS, baseline_sample_count,
    baseline_covariance_payload, sensor_noise_with_baseline)
from Simulation_G1.spatial import (
    SpatialDynamicsConfig, gaussian_weights, patch_hop_order,
    spatial_envelope, validate_disjoint_supports, resolve_spatial_config,
)

try:
    from Simulation_G1.g1_simulation_core import (
        TimeSeriesGenerator,
        _json_safe,
        coordinate_unit_scale_to_mm,
        estimate_power_spectrum_summary,
        mesh_vertex_areas,
        normalize_cortex,
    )
except ImportError:
    from g1_simulation_core import (
        TimeSeriesGenerator,
        _json_safe,
        coordinate_unit_scale_to_mm,
        estimate_power_spectrum_summary,
        mesh_vertex_areas,
        normalize_cortex,
    )


Array = np.ndarray
NMM_SRATE_HZ = 500.0
NMM_DURATION_S = 1.0


def prepare_nmm_clip(clip: Array, srate: float, time_window: tuple[float, float]) -> Array:
    """Resample a native one-second NMM clip without changing its physical duration."""
    duration = float(time_window[1] - time_window[0])
    if not np.isclose(duration, NMM_DURATION_S, rtol=0, atol=1e-9):
        raise ValueError("NMM clips have a fixed one-second duration; use duration-s=1")
    n_samples = float(srate) * duration
    if not np.isfinite(n_samples) or n_samples < 2 or not np.isclose(n_samples, round(n_samples), rtol=0, atol=1e-9):
        raise ValueError("NMM sampling rate must produce an integer number of at least two samples")
    clip = np.asarray(clip)
    if clip.ndim != 2 or clip.shape[1] != int(NMM_SRATE_HZ * NMM_DURATION_S) or not np.isfinite(clip).all():
        raise ValueError("Expected a finite regions-by-500 native NMM clip")
    if round(n_samples) != clip.shape[1]:
        clip = scipy.signal.resample(clip, num=round(n_samples), axis=1)
    return clip


def load_projection_matrix(path: str | Path, n_vertices: int | None = None) -> scipy.sparse.csc_matrix:
    if Path(path).suffix.lower() == ".npz":
        projection = scipy.sparse.csc_matrix(scipy.sparse.load_npz(path))
        if n_vertices is not None and projection.shape[0] != n_vertices:
            raise ValueError("Projection vertices do not match the cortical mesh")
        if not np.isfinite(projection.data).all():
            raise ValueError("Projection must be finite")
        return projection
    with h5py.File(path, "r") as h5_file:
        if "Wnmm2source" in h5_file:
            group = h5_file["Wnmm2source"]
        else:
            candidate_keys = [key for key in h5_file.keys() if key != "#refs#"]
            if len(candidate_keys) != 1:
                raise KeyError("Could not infer projection matrix group")
            group = h5_file[candidate_keys[0]]
        data = np.asarray(group["data"][:], dtype=float).reshape(-1)
        ir = np.asarray(group["ir"][:], dtype=np.int64).reshape(-1)
        jc = np.asarray(group["jc"][:], dtype=np.int64).reshape(-1)
        if n_vertices is None:
            n_vertices = int(group.attrs["MATLAB_sparse"]) if "MATLAB_sparse" in group.attrs else 20484
        n_regions = len(jc) - 1
    return scipy.sparse.csc_matrix((data, ir, jc), shape=(n_vertices, n_regions))


def load_brainstorm_leadfield(leadfield_path: str | Path, n_vertices: int | None = None) -> np.ndarray:
    """Load the supplied forward model; referencing belongs to sensor generation."""
    with h5py.File(leadfield_path, "r") as h5_file:
        if "Gain_default" in h5_file:
            group = h5_file["Gain_default"]
        elif "Gain" in h5_file:
            group = h5_file
        elif "LeadField_fs_20k" in h5_file:
            group = h5_file["LeadField_fs_20k"]
        else:
            candidate_keys = [key for key in h5_file.keys() if key != "#refs#"]
            if len(candidate_keys) == 1 and isinstance(h5_file[candidate_keys[0]], h5py.Group):
                group = h5_file[candidate_keys[0]]
            else:
                raise KeyError("Expected Brainstorm leadfield field")
        gain = np.asarray(group["Gain"][:], dtype=float)
        if n_vertices is not None and gain.shape[0] == 3 * n_vertices and "GridOrient" in group:
            orientations = np.asarray(group["GridOrient"][:], dtype=float)
            if orientations.shape[0] != 3:
                orientations = orientations.T
            leadfield = np.empty((gain.shape[1], n_vertices), dtype=float)
            for vertex_idx in range(n_vertices):
                components = gain[3 * vertex_idx : 3 * vertex_idx + 3, :]
                leadfield[:, vertex_idx] = orientations[:, vertex_idx] @ components
        else:
            leadfield = gain.T
    return leadfield


def load_brainstorm_cortex(cortex_path: str | Path) -> dict[str, Any]:
    with h5py.File(cortex_path, "r") as h5_file:
        if "Cortex" in h5_file:
            group = h5_file["Cortex"]
        else:
            candidate_keys = [key for key in h5_file.keys() if key != "#refs#"]
            if len(candidate_keys) == 1 and isinstance(h5_file[candidate_keys[0]], h5py.Group):
                group = h5_file[candidate_keys[0]]
            else:
                group = h5_file
        cortex = {
            "Vertices": np.asarray(group["Vertices"][:], dtype=float),
            "Faces": np.asarray(group["Faces"][:], dtype=int),
        }
        vert_conn = group["VertConn"]
        if isinstance(vert_conn, h5py.Group) and {"data", "ir", "jc"}.issubset(vert_conn.keys()):
            cortex["VertConn"] = scipy.sparse.csc_matrix(
                (
                    np.asarray(vert_conn["data"][:]).reshape(-1),
                    np.asarray(vert_conn["ir"][:]).reshape(-1),
                    np.asarray(vert_conn["jc"][:]).reshape(-1),
                )
            )
        else:
            cortex["VertConn"] = scipy.sparse.csc_matrix(vert_conn[:])
    return cortex


def load_region_filter(path: str | Path) -> scipy.sparse.csc_matrix:
    return load_projection_matrix(path)


class NMMRegionSignalRepository:
    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.bank = read_bank(self.root_dir) if (self.root_dir / "bank.json").exists() else None
        self.region_files = {} if self.bank is not None else self._scan()

    def _scan(self) -> dict[int, list[tuple[Path, str]]]:
        region_files: dict[int, list[tuple[Path, str]]] = {}
        for zip_path in sorted(self.root_dir.glob("*.zip")):
            with ZipFile(zip_path) as zip_file:
                for name in zip_file.namelist():
                    match = re.fullmatch(r"a(\d+)/nmm_(\d+)\.mat", name)
                    if not match:
                        continue
                    region_id = int(match.group(1))
                    region_files.setdefault(region_id, []).append((zip_path, name))
        return region_files

    def available_regions(self) -> list[int]:
        if self.bank is not None:
            return list(range(self.bank["n_regions"]))
        return sorted(self.region_files.keys())

    def sample_signal(self, region_id: int, rng: np.random.Generator) -> tuple[Array, dict[str, Any]]:
        if self.bank is not None:
            if region_id not in self.available_regions():
                raise KeyError(f"No NMM region {region_id} in bank")
            source, _, info = load_bank_sample(self.root_dir, rng=rng)
            return source, info
        if region_id not in self.region_files:
            raise KeyError(f"No NMM traces available for region {region_id}")
        zip_path, member_name = self.region_files[region_id][int(rng.integers(0, len(self.region_files[region_id])))]
        with ZipFile(zip_path) as zip_file:
            mat = scipy.io.loadmat(BytesIO(zip_file.read(member_name)), squeeze_me=True, struct_as_record=False)
        key = next(k for k in mat.keys() if not k.startswith("__"))
        data = np.asarray(mat[key], dtype=float)
        if data.ndim != 2:
            raise ValueError(f"Expected 2D NMM data in {member_name}, got {data.shape}")
        region_time = data.T
        maxabs = float(np.max(np.abs(region_time)))
        return region_time, {"zip_path": zip_path.name, "member_name": member_name, "source_key": key, "maxabs": maxabs}


@dataclass
class G2SimulationConfig:
    sample_id: str = "G2_sample"
    template_id: str = "fsaverage5"
    source_space_id: str = "Cortex_FS_20k"
    leadfield_id: str = "Gain_20k_FS"
    projection_id: str = "FS_994nmm_to_20k"
    modality: str = "EEG"
    srate: float = 500.0
    time_window: tuple[float, float] = (0.0, 1.0)
    n_active_regions: int = 2
    active_region_ids: list[int] | None = None
    exclude_region_ids: list[int] = field(
        default_factory=lambda: [7, 243, 325, 610, 921, 949, 281, 464, 490, 546, 735, 843, 972, 943]
    )
    weight_decay_mm: float | None = 12.0
    weight_decay_hops: int = 1  # Legacy metadata only; spatial.hop_threshold controls dynamics.
    spatial: SpatialDynamicsConfig = field(default_factory=SpatialDynamicsConfig)
    normalize_region_signals: bool = True
    normalize_source: bool = True
    sensor_noise_type: str = "white"
    sensor_snr_db: float | None = 5.0
    sensor_snr_mode: str = "robust_global_rms"
    sensor_snr_clip_quantile: float = 0.95
    baseline_duration_s: float = 1.0
    average_reference: bool = True
    random_seed: int | None = None
    nmm_sample_id: str | None = None
    nmm_state: str | None = None
    nmm_source_mode: str = "auto"
    nmm_start_s: float = 0.0


class G2SourceSimulator:
    def __init__(
        self,
        leadfield: Array,
        cortex: dict[str, Any],
        projection_matrix: scipy.sparse.csc_matrix,
        nmm_root_dir: str | Path,
    ):
        self.leadfield = np.asarray(leadfield, dtype=float)
        self.cortex = cortex
        self.vertices, self.faces, self.vert_conn = normalize_cortex(cortex["Vertices"], cortex["Faces"], cortex["VertConn"])
        self.coord_to_mm = coordinate_unit_scale_to_mm(self.vertices)
        _, self.vertex_area = mesh_vertex_areas(self.vertices, self.faces)
        self.projection_matrix = projection_matrix.tocsc(copy=True)
        self.projection_matrix.sum_duplicates()
        self.projection_matrix.eliminate_zeros()
        if not np.isfinite(self.projection_matrix.data).all():
            raise ValueError("Projection must be finite")
        if self.projection_matrix.shape[0] != self.vertices.shape[0]:
            raise ValueError("projection matrix row count must match cortex vertex count")
        self.nmm_repo = NMMRegionSignalRepository(nmm_root_dir)
        self.region_vertex_lists = [self.projection_matrix[:, idx].indices for idx in range(self.projection_matrix.shape[1])]
        self.region_centroids = [
            np.mean(self.vertices[vertex_ids], axis=0) if len(vertex_ids) > 0 else np.zeros(3)
            for vertex_ids in self.region_vertex_lists
        ]
        self.n_vertices = self.vertices.shape[0]
        self.n_regions = self.projection_matrix.shape[1]
        self.n_sensors = self.leadfield.shape[0]

    @classmethod
    def from_paths(
        cls,
        leadfield_path: str | Path,
        cortex_path: str | Path,
        projection_path: str | Path,
        nmm_root_dir: str | Path,
    ) -> "G2SourceSimulator":
        cortex = load_brainstorm_cortex(cortex_path)
        n_vertices = np.asarray(cortex["Vertices"]).shape[-1]
        leadfield = load_brainstorm_leadfield(leadfield_path, n_vertices=n_vertices)
        projection_matrix = load_projection_matrix(projection_path, n_vertices=n_vertices)
        return cls(leadfield, cortex, projection_matrix, nmm_root_dir)

    def available_regions(self) -> list[int]:
        return [region for region in self.nmm_repo.available_regions() if region < self.n_regions]

    def sample_regions(self, n_regions: int, rng: np.random.Generator, exclude: set[int] | None = None,
                       *, candidate_regions: list[int] | None = None) -> list[int]:
        """Choose nonempty, disjoint parcel supports without silently shrinking them."""
        if isinstance(n_regions, bool) or not isinstance(n_regions, (int, np.integer)) or n_regions < 1:
            raise ValueError("n_regions must be a positive integer")
        exclude = set(exclude or [])
        exclude.update({int(v) for v in self._default_excludes()})
        available = self.available_regions() if candidate_regions is None else candidate_regions
        if any(isinstance(r, (bool, np.bool_)) or not isinstance(r, (int, np.integer))
               or r < 0 or r >= self.n_regions for r in available):
            raise ValueError("Candidate regions must be valid integer IDs")
        supports = {r: set(self._region_support(r).tolist()) for r in available if r not in exclude}
        candidates = [r for r, support in supports.items() if support]
        if len(candidates) < n_regions:
            raise RuntimeError("Not enough candidate regions available")
        for _ in range(64):
            chosen, occupied = [], set()
            for candidate in rng.permutation(candidates):
                region = int(candidate)
                if not occupied.isdisjoint(supports[region]):
                    continue
                chosen.append(region)
                occupied.update(supports[region])
                if len(chosen) == n_regions:
                    return chosen
        raise RuntimeError("Could not select disjoint regions after 64 attempts; reduce the source count or change the atlas")

    def _region_support(self, region_id: int) -> Array:
        column = self.projection_matrix.getcol(region_id)
        return column.indices[column.data != 0]

    def _validate_selected_regions(self, selected_regions: list[int]) -> list[Array]:
        if (not selected_regions or any(
                isinstance(region, (bool, np.bool_)) or not isinstance(region, (int, np.integer))
                or region < 0 or region >= self.n_regions for region in selected_regions)
                or len(set(selected_regions)) != len(selected_regions)):
            raise ValueError("Active regions must be nonempty unique valid integer IDs")
        supports = [self._region_support(region) for region in selected_regions]
        if any(len(support) == 0 for support in supports):
            raise ValueError("Active regions must have nonempty cortical supports")
        validate_disjoint_supports(supports)
        return supports

    def project_regions(
        self, region_source: Array, selected_regions: list[int], *, srate: float,
        time_window: tuple[float, float], decay_mm: float | None = None,
        spatial: SpatialDynamicsConfig | None = None,
    ) -> tuple[Array, dict[str, Any]]:
        """Transform selected parcels separately, leaving background projection intact."""
        spatial = resolve_spatial_config(spatial)
        region_source = np.asarray(region_source)
        if (region_source.ndim != 2 or region_source.shape[0] != self.n_regions
                or not np.isfinite(region_source).all()):
            raise ValueError("Expected finite regions-by-time signals matching the projection")
        if not np.isfinite(srate) or srate <= 0:
            raise ValueError("srate must be finite and positive")
        duration = float(time_window[1] - time_window[0])
        if not np.isfinite(duration) or duration <= 0 or not np.isclose(duration * srate, region_source.shape[1]):
            raise ValueError("time_window and srate must match the signal length")
        supports = self._validate_selected_regions(selected_regions)
        # Project background separately to avoid cancellation and tapering shared nuisance vertices.
        background = region_source.copy()
        background[selected_regions] = 0
        vertex_source = np.asarray(self.projection_matrix @ background, dtype=float)
        times = np.arange(region_source.shape[1], dtype=float) / srate
        entries = []
        for region_id, indices in zip(selected_regions, supports):
            column = self.projection_matrix.getcol(region_id)
            amplitudes = column.data[column.data != 0]
            if hasattr(self, "vertices"):
                centroid = np.mean(self.vertices[indices], axis=0)
                distances_mm = np.linalg.norm(self.vertices[indices] - centroid, axis=1) * self.coord_to_mm
                seed = int(indices[np.argmin(distances_mm)])
                area_mm2 = float(np.sum(self.vertex_area[indices]) * self.coord_to_mm ** 2)
                hops = patch_hop_order(indices, seed, self.vert_conn)
            else:
                if decay_mm not in (None, 0) or spatial.mode not in {"static", "auto"}:
                    raise ValueError("Spatial weighting requires cortical geometry")
                distances_mm = np.zeros(len(indices))
                seed, area_mm2, hops = None, None, None
            base = gaussian_weights(distances_mm, decay_mm)
            weights, metadata = spatial_envelope(base, times, spatial, area_mm2=area_mm2, hop_order=hops)
            vertex_source[indices] += amplitudes[:, None] * weights * region_source[region_id]
            entries.append(dict(region_id=int(region_id), seed_vertex_id=seed,
                vertex_indices=indices.tolist(), base_weights=base.tolist(),
                projection_weights=amplitudes.tolist(), area_mm2=area_mm2,
                area_cm2=None if area_mm2 is None else area_mm2 / 100.0,
                hop_order=hops, spatial_dynamics=metadata))
        return vertex_source, dict(spatial_sources=entries,
            spatial_weight_decay_applied=decay_mm is not None and decay_mm > 0,
            spatial_supports_disjoint=True)

    def generate_source(self, config: G2SimulationConfig) -> tuple[Array, Array, dict[str, Any]]:
        config.spatial = resolve_spatial_config(config.spatial)
        if getattr(self.nmm_repo, "bank", None) is not None:
            return self._generate_bank_source(config)
        if (config.nmm_sample_id is not None or config.nmm_state is not None
                or config.nmm_source_mode not in {"auto", "selected"} or config.nmm_start_s != 0):
            raise ValueError("Generated-bank options require an NMM bank directory")
        rng = np.random.default_rng(config.random_seed)
        selected_regions = (self.sample_regions(config.n_active_regions, rng, set(config.exclude_region_ids))
                            if config.active_region_ids is None else config.active_region_ids)
        self._validate_selected_regions(selected_regions)
        region_info = []
        region_source = None
        for region_id in selected_regions:
            trace, info = self.nmm_repo.sample_signal(region_id, rng)
            trace = prepare_nmm_clip(trace, config.srate, config.time_window)
            if trace.shape[0] != self.n_regions:
                raise ValueError("NMM clip region count must match the projection matrix")
            if config.normalize_region_signals:
                trace = trace / max(float(np.max(np.abs(trace))), 1e-12)
            if region_source is None:
                region_source = np.zeros_like(trace, dtype=np.float32)
            region_source[region_id] = trace[region_id]
            signal_row = np.asarray(trace[region_id], dtype=float)
            background_mask = np.ones(trace.shape[0], dtype=bool)
            background_mask[region_id] = False
            background = trace[background_mask]
            signal_power = float(np.mean(signal_row**2))
            background_power = float(np.mean(background**2)) if background.size else 0.0
            info["region_id"] = int(region_id)
            info["signal_power"] = signal_power
            info["background_power"] = background_power
            info["source_level_snr_db"] = float(10.0 * np.log10(max(signal_power, 1e-12) / max(background_power, 1e-12)))
            region_info.append(info)
        if region_source is None:
            raise RuntimeError("No region signals selected")
        vertex_source, spatial_meta = self.project_regions(region_source, selected_regions,
            srate=config.srate, time_window=config.time_window, decay_mm=config.weight_decay_mm,
            spatial=config.spatial)
        if config.normalize_source:
            vertex_source = vertex_source - vertex_source.mean(axis=1, keepdims=True)
            vertex_source /= max(float(np.max(np.abs(vertex_source))), 1e-12)
        manifest = self._source_metadata(config, selected_regions, region_info, region_source, vertex_source)
        manifest.update(spatial_meta)
        return vertex_source.astype(np.float32), region_source.astype(np.float32), manifest

    def _generate_bank_source(self, config: G2SimulationConfig) -> tuple[Array, Array, dict[str, Any]]:
        bank = read_bank(self.nmm_repo.root_dir)
        if (bank["n_regions"] != self.n_regions
                or bank["projection_sha256"] != projection_hash(self.projection_matrix)):
            raise ValueError("NMM bank atlas/projection differs from G2 projection")
        mode = "network" if config.nmm_source_mode == "auto" else config.nmm_source_mode
        if mode not in {"network", "selected"}:
            raise ValueError("NMM source mode must be auto, network or selected")
        if mode == "network" and config.active_region_ids is not None:
            raise ValueError("Explicit region selection requires nmm_source_mode='selected'")
        if mode == "network" and config.spatial.mode not in {"auto", "static"}:
            raise ValueError("Spatial dynamics require nmm_source_mode='selected'; network mode preserves the simulation")
        if mode == "selected" and config.active_region_ids is not None:
            self._validate_selected_regions(config.active_region_ids)
        source, times, entry = load_bank_sample(self.nmm_repo.root_dir, config.nmm_sample_id,
            signal_type=config.nmm_state, rng=np.random.default_rng(config.random_seed))
        region_source = prepare_bank_segment(source, times, srate=config.srate,
            duration_s=config.time_window[1] - config.time_window[0], start_s=config.nmm_start_s)
        selected = list(entry["active_regions"] if config.active_region_ids is None else config.active_region_ids)
        if (len(set(selected)) != len(selected)
                or any(not isinstance(i, (int, np.integer)) or isinstance(i, bool)
                       or i < 0 or i >= self.n_regions for i in selected)):
            raise ValueError("Selected bank regions must be unique valid integer indices")
        if mode == "selected":
            if not selected:
                raise ValueError("Selected mode requires region IDs; this sample has no configured regions")
            retained = np.zeros(self.n_regions, dtype=bool)
            retained[selected] = True
            region_source[~retained] = 0
        if config.normalize_region_signals:
            region_source /= max(float(np.max(np.abs(region_source))), 1e-12)
        spatial_meta = {"spatial_weight_decay_applied": False, "spatial_sources": [],
                        "spatial_skip_reason": "network preserves the original NMM trajectory"}
        if mode == "selected":
            vertex_source, spatial_meta = self.project_regions(region_source, selected,
                srate=config.srate, time_window=config.time_window, decay_mm=config.weight_decay_mm,
                spatial=config.spatial)
        else:
            vertex_source = self.projection_matrix @ region_source
        if config.normalize_source:
            vertex_source -= vertex_source.mean(axis=1, keepdims=True)
            vertex_source /= max(float(np.max(np.abs(vertex_source))), 1e-12)
        manifest = self._source_metadata(config, selected, [entry], region_source, vertex_source)
        manifest.update(generator_type="G2_NMM_bank", nmm_sample_id=entry["sample_id"],
            nmm_state=entry["signal_type"], source_type=entry["signal_type"], nmm_source_mode=mode,
            nmm_bank=str(self.nmm_repo.root_dir), nmm_bank_atlas_id=bank["atlas_id"],
            nmm_native_srate_hz=entry["native_srate_hz"], nmm_source_unit=entry["source_unit"],
            region_source_unit="normalized_model_state" if config.normalize_region_signals else entry["source_unit"],
            vertex_source_unit="normalized_source" if config.normalize_source or config.normalize_region_signals else entry["source_unit"],
            configured_regions=entry["active_regions"],
            retained_region_ids=list(range(self.n_regions)) if mode == "network" else selected,
            nmm_background_policy="same_simulation" if mode == "network" else "none",
            parameter_preset_is_activation_truth=False,
            simulation_group_id=entry["simulation_group_id"])
        manifest.update(spatial_meta)
        if not np.isfinite(vertex_source).all():
            raise FloatingPointError("Nonfinite projected NMM source")
        with np.errstate(over="raise", invalid="raise"):
            return vertex_source.astype(np.float32), region_source.astype(np.float32), manifest

    def generate_sensor(
        self,
        config: G2SimulationConfig,
        real_sensor_noise: Array | None = None,
    ) -> tuple[Array, Array, Array, Array, dict[str, Any]]:
        source, region_source, meta = self.generate_source(config)
        return self._finalize_sensor_from_source(source, region_source, meta, config, real_sensor_noise)

    def _finalize_sensor_from_source(
        self,
        source: Array,
        region_source: Array,
        meta: dict[str, Any],
        config: G2SimulationConfig,
        real_sensor_noise: Array | None = None,
    ) -> tuple[Array, Array, Array, Array, dict[str, Any]]:
        clean = self.leadfield @ source
        clean = clean - clean.mean(axis=1, keepdims=True)
        average_reference = config.average_reference and config.modality.upper() == "EEG"
        if average_reference:
            clean -= clean.mean(axis=0, keepdims=True)
        meta["reference_mode"] = "average" if average_reference else "as_supplied"
        baseline_samples = baseline_sample_count(config.baseline_duration_s, config.srate)
        noise, baseline, baseline_info = sensor_noise_with_baseline(self, clean,
            srate=config.srate, sensor_noise_type=config.sensor_noise_type, snr_db=config.sensor_snr_db,
            seed=None if config.random_seed is None else config.random_seed + 1,
            baseline_samples=baseline_samples, average_reference=average_reference,
            snr_mode=config.sensor_snr_mode, clip_quantile=config.sensor_snr_clip_quantile,
            real_sensor_noise=real_sensor_noise)
        sensor = clean + noise
        meta["sensor_shape"] = list(sensor.shape)
        meta["vertex_source_shape"] = list(source.shape)
        meta["region_source_shape"] = list(region_source.shape)
        meta["clean_sensor_power"] = float(np.mean(clean**2))
        meta["noise_sensor_power"] = float(np.mean(noise**2))
        source_snrs = [info["source_level_snr_db"] for info in meta.get("region_info", [])
                       if info.get("source_level_snr_db") is not None]
        meta["source_level_snr_db"] = float(np.mean(source_snrs)) if source_snrs else None
        meta["sensor_snr_mode"] = config.sensor_snr_mode
        power_floor = np.finfo(float).tiny
        noise_log_power = np.log10(max(float(np.mean(noise**2)), power_floor))
        meta["realized_sensor_snr_db"] = float(10.0 * (np.log10(max(float(np.mean(clean**2)), power_floor)) - noise_log_power))
        meta["realized_sensor_snr_reference_db"] = float(
            10.0
            * (
                np.log10(max(self._snr_reference_power(clean, config.sensor_snr_mode, config.sensor_snr_clip_quantile), power_floor))
                - noise_log_power
            )
        )
        meta["vertex_source_psd_summary"] = estimate_power_spectrum_summary(source, config.srate)
        meta["region_source_psd_summary"] = estimate_power_spectrum_summary(region_source, config.srate)
        meta["clean_sensor_psd_summary"] = estimate_power_spectrum_summary(clean, config.srate)
        meta["noise_sensor_psd_summary"] = estimate_power_spectrum_summary(noise, config.srate)
        meta["real_sensor_noise_used"] = real_sensor_noise is not None
        covariances, covariance_info = baseline_covariance_payload(baseline, noise.astype(np.float32),
            average_reference=average_reference, baseline_metadata=baseline_info)
        meta.update(covariance_info)
        meta.update({key: value.tolist() for key, value in covariances.items()})
        meta["noise_covariance_target"] = "entire_exported_source; no separate source-background target in legacy API"
        return sensor.astype(np.float32), clean.astype(np.float32), noise.astype(np.float32), source.astype(np.float32), meta

    def save_sample(
        self,
        output_dir: str | Path,
        config: G2SimulationConfig,
        real_sensor_noise: Array | None = None,
        save_sensor: bool = False,
        save_clean_sensor: bool = False,
        save_sensor_noise: bool = True,
        save_source: bool = False,
        save_region_source: bool = True,
    ) -> dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        source, region_source, meta = self.generate_source(config)
        sensor, clean, noise, source, manifest = self._finalize_sensor_from_source(
            source,
            region_source,
            meta,
            config,
            real_sensor_noise=real_sensor_noise,
        )
        if save_sensor:
            path = output_dir / f"{config.sample_id}_sensor.npz"
            np.savez_compressed(path, sensor=sensor)
            manifest["sensor_data"] = path.name
        if save_clean_sensor:
            path = output_dir / f"{config.sample_id}_sensor_clean.npz"
            np.savez_compressed(path, sensor_clean=clean)
            manifest["sensor_clean"] = path.name
        if save_sensor_noise:
            path = output_dir / f"{config.sample_id}_sensor_noise.npz"
            np.savez_compressed(path, sensor_noise=noise)
            manifest["sensor_noise"] = path.name
        if save_source:
            path = output_dir / f"{config.sample_id}_source.npz"
            np.savez_compressed(path, source=source)
            manifest["source_truth"] = path.name
        if save_region_source:
            path = output_dir / f"{config.sample_id}_region_source.npz"
            np.savez_compressed(path, region_source=region_source)
            manifest["region_source"] = path.name
        covariance_path = output_dir / f"{config.sample_id}_noise_covariance.npz"
        np.savez_compressed(covariance_path, **{key: np.asarray(manifest.pop(key),
            dtype=np.float32 if key == 'baseline_sensor' else np.float64) for key in NOISE_ARRAY_KEYS})
        manifest["noise_covariance_file"] = covariance_path.name
        manifest_path = output_dir / f"{config.sample_id}_manifest.json"
        manifest["manifest"] = manifest_path.name
        manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
        return manifest

    def manifest_only(self, config: G2SimulationConfig) -> dict[str, Any]:
        _, _, _, _, meta = self.generate_sensor(config)
        return meta

    def _apply_weight_decay(
        self,
        vertex_source: Array,
        selected_regions: list[int],
        decay_mm: float,
    ) -> Array:
        self._validate_selected_regions(selected_regions)
        decayed = np.array(vertex_source, dtype=np.float32, copy=True)
        for region_id in selected_regions:
            region_vertices = set(self.region_vertex_lists[region_id].tolist())
            if not region_vertices:
                continue
            region_vertices = np.array(sorted(region_vertices), dtype=int)
            centroid = self.region_centroids[region_id]
            dist_mm = np.linalg.norm(self.vertices[region_vertices] - centroid, axis=1) * self.coord_to_mm
            weights = gaussian_weights(dist_mm, decay_mm).astype(np.float32)
            decayed[region_vertices, :] *= weights[:, None]
        return decayed

    def _source_metadata(
        self,
        config: G2SimulationConfig,
        selected_regions: list[int],
        region_info: list[dict[str, Any]],
        region_source: Array,
        vertex_source: Array,
    ) -> dict[str, Any]:
        manifest = asdict(config)
        manifest.update(
            {
                "generator_type": "G2_NMM_spike",
                "storage_policy": "region_source_plus_vertex_projection",
                "n_vertices": self.n_vertices,
                "n_regions": self.n_regions,
                "n_sensors": self.n_sensors,
                "selected_regions": [int(v) for v in selected_regions],
                "region_info": region_info,
                "source_representation": "vertex_projected",
                "source_shape": list(vertex_source.shape),
                "vertex_source_shape": list(vertex_source.shape),
                "region_source_shape": list(region_source.shape),
                "leadfield_shape": list(self.leadfield.shape),
                "projection_shape": list(self.projection_matrix.shape),
                "projection_nnz": int(self.projection_matrix.nnz),
            }
        )
        return _json_safe(manifest)

    def _make_sensor_noise(
        self,
        clean: Array,
        ts: TimeSeriesGenerator,
        noise_type: str,
        snr_db: float,
        snr_mode: str = "robust_global_rms",
        clip_quantile: float = 0.95,
        real_sensor_noise: Array | None = None,
        average_reference: bool = False,
    ) -> Array:
        if real_sensor_noise is not None:
            noise = self._prepare_real_sensor_noise(real_sensor_noise, clean.shape, ts.rng)
        elif noise_type == "real":
            raise ValueError("sensor_noise_type='real' requires real_sensor_noise")
        elif noise_type == "iid_white":
            noise = ts.rng.standard_normal(clean.shape)
        elif noise_type == "white":
            noise = self._make_broadband_sensor_noise(clean.shape[0], clean.shape[1], ts)
        elif noise_type == "pink":
            noise = self._make_aperiodic_sensor_noise(clean.shape[0], clean.shape[1], ts)
        elif noise_type in {"powerlaw", "neurodsp_powerlaw"}:
            noise = self._make_powerlaw_sensor_noise(clean.shape[0], clean.shape[1], ts)
        elif noise_type in {"specparam_resting", "knee_powerlaw"}:
            noise = self._make_specparam_sensor_noise(clean.shape[0], clean.shape[1], ts)
        elif noise_type == "mixed":
            noise = 0.55 * self._make_aperiodic_sensor_noise(clean.shape[0], clean.shape[1], ts)
            noise += 0.35 * self._make_broadband_sensor_noise(clean.shape[0], clean.shape[1], ts)
            noise += 0.10 * self._make_eegsourcesim_sensor_noise(clean.shape[0], clean.shape[1], ts)
        elif noise_type == "eegsourcesim_resting":
            noise = self._make_eegsourcesim_sensor_noise(clean.shape[0], clean.shape[1], ts)
        else:
            raise ValueError(f"Unsupported sensor_noise_type: {noise_type}")
        noise -= noise.mean(axis=1, keepdims=True)
        if average_reference:
            noise -= noise.mean(axis=0, keepdims=True)
        clean_power = self._snr_reference_power(clean, snr_mode, clip_quantile)
        noise_power = float(np.mean(noise**2))
        target_power = clean_power / max(10.0 ** (float(snr_db) / 10.0), 1e-12)
        if noise_power == 0 and target_power > 0:
            raise ValueError("Sensor noise has zero power after referencing; cannot reach the requested SNR")
        return noise * np.sqrt(target_power / max(noise_power, np.finfo(float).tiny))

    @staticmethod
    def _snr_reference_power(clean: Array, mode: str, clip_quantile: float) -> float:
        if mode == "global_rms":
            return float(np.mean(clean**2))
        if mode == "robust_global_rms":
            limit = float(np.quantile(np.abs(clean), np.clip(clip_quantile, 0.5, 0.999)))
            clipped = np.clip(clean, -limit, limit)
            return float(np.mean(clipped**2))
        if mode == "active_rms":
            global_field_power = np.sqrt(np.mean(clean**2, axis=0))
            threshold = float(np.quantile(global_field_power, np.clip(clip_quantile, 0.5, 0.999)))
            active = global_field_power >= threshold
            return float(np.mean(clean[:, active] ** 2)) if np.any(active) else float(np.mean(clean**2))
        raise ValueError(f"Unsupported sensor_snr_mode: {mode}")

    @staticmethod
    def _prepare_real_sensor_noise(real_noise: Array, target_shape: tuple[int, int], rng: np.random.Generator) -> Array:
        noise = np.array(real_noise, dtype=float, copy=True)
        if noise.ndim == 1:
            noise = np.tile(noise[None, :], (target_shape[0], 1))
        if noise.ndim != 2:
            raise ValueError("real_sensor_noise must be [channels, times] or [times]")
        if noise.shape[0] != target_shape[0]:
            raise ValueError(f"real noise has {noise.shape[0]} channels, expected {target_shape[0]}")
        if noise.shape[1] < target_shape[1]:
            repeats = int(np.ceil(target_shape[1] / noise.shape[1]))
            noise = np.tile(noise, (1, repeats))
        if noise.shape[1] > target_shape[1]:
            start = int(rng.integers(0, noise.shape[1] - target_shape[1] + 1))
            noise = noise[:, start : start + target_shape[1]]
        else:
            noise = noise[:, : target_shape[1]]
        noise -= np.mean(noise, axis=1, keepdims=True)
        return noise

    def _make_broadband_sensor_noise(self, n_channels: int, n_times: int, ts: TimeSeriesGenerator) -> Array:
        common = np.asarray([ts.broadband_noise(n_times) for _ in range(min(4, n_channels))])
        mixing = ts.rng.standard_normal((n_channels, common.shape[0]))
        shared = mixing @ common
        independent = np.asarray([ts.broadband_noise(n_times) for _ in range(n_channels)])
        noise = 0.30 * shared + 0.70 * independent
        noise -= noise.mean(axis=1, keepdims=True)
        return noise

    def _make_powerlaw_sensor_noise(self, n_channels: int, n_times: int, ts: TimeSeriesGenerator) -> Array:
        common = np.asarray(
            [
                ts.powerlaw_noise(
                    n_times,
                    exponent=float(ts.rng.uniform(0.8, 1.8)),
                    white_floor=float(ts.rng.uniform(0.02, 0.12)),
                )
                for _ in range(min(6, n_channels))
            ]
        )
        mixing = ts.rng.standard_normal((n_channels, common.shape[0]))
        shared = mixing @ common
        independent = np.asarray(
            [
                ts.powerlaw_noise(
                    n_times,
                    exponent=float(ts.rng.uniform(0.6, 1.6)),
                    white_floor=float(ts.rng.uniform(0.04, 0.16)),
                )
                for _ in range(n_channels)
            ]
        )
        noise = 0.50 * shared + 0.45 * independent + 0.05 * ts.rng.standard_normal((n_channels, n_times))
        noise -= noise.mean(axis=1, keepdims=True)
        return noise

    def _make_specparam_sensor_noise(self, n_channels: int, n_times: int, ts: TimeSeriesGenerator) -> Array:
        common = np.asarray(
            [
                ts.specparam_signal(
                    n_times,
                    exponent=float(ts.rng.uniform(0.8, 1.8)),
                    knee_hz=float(ts.rng.uniform(0.3, 2.0)),
                    peaks=[
                        {
                            "center_hz": float(ts.rng.uniform(8.0, 12.0)),
                            "bandwidth_hz": float(ts.rng.uniform(1.0, 3.5)),
                            "height": float(ts.rng.uniform(0.25, 0.9)),
                        }
                    ],
                    white_floor=float(ts.rng.uniform(0.01, 0.08)),
                )
                for _ in range(min(6, n_channels))
            ]
        )
        mixing = ts.rng.standard_normal((n_channels, common.shape[0]))
        shared = mixing @ common
        independent = np.asarray(
            [
                ts.specparam_signal(
                    n_times,
                    exponent=float(ts.rng.uniform(0.7, 1.5)),
                    peaks=[
                        {
                            "center_hz": float(ts.rng.uniform(8.0, 12.0)),
                            "bandwidth_hz": float(ts.rng.uniform(1.5, 4.0)),
                            "height": float(ts.rng.uniform(0.1, 0.6)),
                        }
                    ],
                    white_floor=float(ts.rng.uniform(0.03, 0.12)),
                )
                for _ in range(n_channels)
            ]
        )
        noise = 0.45 * shared + 0.45 * independent + 0.10 * ts.rng.standard_normal((n_channels, n_times))
        noise -= noise.mean(axis=1, keepdims=True)
        return noise

    def _make_aperiodic_sensor_noise(self, n_channels: int, n_times: int, ts: TimeSeriesGenerator) -> Array:
        common = np.asarray(
            [
                ts.aperiodic_noise(
                    n_times,
                    exponent=float(ts.rng.uniform(0.8, 1.4)),
                    white_floor=float(ts.rng.uniform(0.06, 0.16)),
                    envelope_depth=float(ts.rng.uniform(0.05, 0.20)),
                )
                for _ in range(min(5, n_channels))
            ]
        )
        mixing = ts.rng.standard_normal((n_channels, common.shape[0]))
        shared = mixing @ common
        independent = np.asarray(
            [
                ts.aperiodic_noise(
                    n_times,
                    exponent=float(ts.rng.uniform(0.7, 1.3)),
                    white_floor=float(ts.rng.uniform(0.08, 0.20)),
                    envelope_depth=float(ts.rng.uniform(0.05, 0.18)),
                )
                for _ in range(n_channels)
            ]
        )
        white_floor = ts.rng.standard_normal((n_channels, n_times)) * 0.08
        noise = 0.45 * shared + 0.50 * independent + white_floor
        noise -= noise.mean(axis=1, keepdims=True)
        return noise

    def _make_eegsourcesim_sensor_noise(self, n_channels: int, n_times: int, ts: TimeSeriesGenerator) -> Array:
        n_common = min(6, n_channels)
        common_basis = np.asarray(
            [
                ts.eegsourcesim_resting_noise(
                    n_times,
                    exponent=float(ts.rng.uniform(0.9, 1.4)),
                    alpha_peak_hz=float(ts.rng.uniform(8.0, 12.0)),
                    alpha_width_hz=float(ts.rng.uniform(2.0, 4.0)),
                    alpha_gain=float(ts.rng.uniform(0.8, 2.5)),
                    white_floor=float(ts.rng.uniform(0.03, 0.08)),
                    envelope_depth=float(ts.rng.uniform(0.10, 0.25)),
                )
                for _ in range(n_common)
            ]
        )
        channel_mixing = ts.rng.standard_normal((n_channels, n_common))
        common_noise = channel_mixing @ common_basis
        independent_noise = np.asarray(
            [
                ts.eegsourcesim_resting_noise(
                    n_times,
                    exponent=float(ts.rng.uniform(0.8, 1.3)),
                    alpha_peak_hz=float(ts.rng.uniform(8.0, 12.0)),
                    alpha_width_hz=float(ts.rng.uniform(2.0, 5.0)),
                    alpha_gain=float(ts.rng.uniform(0.3, 1.5)),
                    white_floor=float(ts.rng.uniform(0.05, 0.12)),
                    envelope_depth=float(ts.rng.uniform(0.05, 0.20)),
                )
                for _ in range(n_channels)
            ]
        )
        white_floor = ts.rng.standard_normal((n_channels, n_times)) * 0.15
        noise = 0.55 * common_noise + 0.35 * independent_noise + white_floor
        noise -= noise.mean(axis=1, keepdims=True)
        return noise

    def _default_excludes(self) -> set[int]:
        return {7, 243, 325, 610, 921, 949, 281, 464, 490, 546, 735, 843, 972, 943}


class G2ScenarioSampler:
    def __init__(self, random_seed: int | None = None):
        self.rng = np.random.default_rng(random_seed)

    def sample(self, sample_id: str, template_id: str, source_space_id: str, leadfield_id: str, projection_id: str) -> G2SimulationConfig:
        n_active = int(self.rng.integers(1, 4))
        noise_choice = str(
            self.rng.choice(
                ["white", "pink", "powerlaw", "specparam_resting", "mixed", "eegsourcesim_resting"],
                p=[0.16, 0.18, 0.18, 0.18, 0.12, 0.18],
            )
        )
        return G2SimulationConfig(
            sample_id=sample_id,
            template_id=template_id,
            source_space_id=source_space_id,
            leadfield_id=leadfield_id,
            projection_id=projection_id,
            n_active_regions=n_active,
            weight_decay_mm=float(self.rng.uniform(8.0, 16.0)),
            weight_decay_hops=int(self.rng.integers(1, 3)),
            sensor_noise_type=noise_choice,
            sensor_snr_db=float(self.rng.uniform(0.0, 15.0)),
            random_seed=int(self.rng.integers(1, 2**31 - 1)),
        )
