from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
ANATOMY_DIR = ROOT / "Anatomy"
NMM_SPIKES_DIR = ROOT / "nmm_spikes"
DEFAULT_OUTPUT_DIR = ROOT / "generated_simulation_dataset"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Simulation_G1.g1_simulation_core import G1SimulationConfig, G1SourceSimulator, _json_safe, estimate_power_spectrum_summary
from Simulation_G1.spatial import SpatialDynamicsConfig
from Simulation_G1.noise_covariance import NOISE_ARRAY_KEYS
from Simulation_G1.sample_io import save_sample_directory
from Simulation_G2.g2_simulation_core import (
    G2SimulationConfig,
    G2SourceSimulator,
    load_brainstorm_cortex,
    load_brainstorm_leadfield,
    load_projection_matrix,
    prepare_nmm_clip,
)
from Simulation_G2.nmm_bank import projection_hash, read_bank
from Simulation_G3.g3_tvb_core import TVBSignalConfig, TVBSourceGenerator
from Simulation_G3.atlas_profiles import ATLAS_CHOICES, get_atlas_profile, validate_atlas_connectivity
from Simulation_G3.generate_nmm_spike_reference_demo import load_nmm_spike_clip


G1_WAVEFORM_PRESETS: dict[str, tuple[str, dict[str, Any]]] = {
    "erp": (
        "erp",
        {
            "components": [
                {"type": "gaussian", "latency_s": 0.12, "width_s": 0.045, "amplitude_rel": 1.0},
                {"type": "gaussian", "latency_s": 0.28, "width_s": 0.075, "amplitude_rel": -0.55},
            ],
            "amplitude": 1.0,
        },
    ),
    "gabor_erp": (
        "erp",
        {
            "components": [
                {
                    "type": "gabor",
                    "latency_s": 0.18,
                    "std_dev_s": 0.055,
                    "frequency_Hz": 12.0,
                    "phase_rad": 0.0,
                    "amplitude_rel": 1.0,
                }
            ],
            "amplitude": 1.0,
        },
    ),
    "spike": ("spike", {"latency_s": 0.16, "duration_s": 0.075, "shape": "exponential_decay", "amplitude": 1.0}),
    "alpha_burst": (
        "ersp",
        {
            "base_frequency_band_Hz": 10.0,
            "modulation": {"type": "burst", "latency_s": 0.25, "width_s": 0.35, "min_rel_amplitude": 0.08},
            "amplitude": 1.0,
        },
    ),
    "resting_alpha": (
        "resting",
        {
            "band": "alpha",
            "envelope": "smooth",
            "min_gain": 0.3,
            "max_gain": 1.2,
            "control_interval_s": 0.35,
            "amplitude": 1.0,
        },
    ),
    "powerlaw": ("powerlaw", {"exponent": 1.4, "white_floor": 0.04, "amplitude": 1.0}),
    "specparam_alpha": (
        "specparam",
        {
            "exponent": 1.2,
            "knee_hz": 0.8,
            "peaks": [{"center_hz": 10.0, "bandwidth_hz": 1.6, "height": 0.8}],
            "envelope_depth": 0.12,
            "amplitude": 1.0,
        },
    ),
    "synaptic_noise": (
        "synaptic_noise",
        {"event_rate_hz": 45.0, "tau_rise_s": 0.006, "tau_decay_s": 0.070, "amplitude": 1.0},
    ),
    "nonsinusoidal_alpha": (
        "nonsinusoidal_oscillation",
        {"frequency_hz": 10.0, "harmonics": {2: 0.35, 3: 0.15}, "amplitude": 1.0},
    ),
}


G3_TVB_PRESETS: dict[str, dict[str, Any]] = {
    "spike_jr": {"coupling_strength": 1.0, "jr_active_A": 3.6, "jr_mu": 0.087, "jr_noise_scale": 1.0},
    "resting_jr": {"coupling_strength": 1.0, "jr_active_A": 3.25, "jr_mu": 0.087, "jr_noise_scale": 1.0},
    "alpha_jr": {"coupling_strength": 0.8, "jr_active_A": 3.35, "jr_mu": 0.10, "jr_noise_scale": 0.8},
    "oscillator": {"coupling_strength": 0.003},
    "wilson_cowan": {"coupling_strength": 0.003},
}


def parse_int_list(text: str | None) -> list[int]:
    if not text:
        return []
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def get_projection_path(args: argparse.Namespace) -> Path:
    if args.projection_path:
        return Path(args.projection_path)
    if args.atlas in ("aal78", "aal78_membership"):
        return Path(get_atlas_profile(args.atlas)["projection_path"])
    return Path(args.anatomy_dir) / "FS_994nmm_to_20k.mat"


def load_fs20k_backend(args: argparse.Namespace) -> G2SourceSimulator:
    anatomy_dir = Path(args.anatomy_dir)
    cortex_path = Path(args.cortex_path) if args.cortex_path else anatomy_dir / "Cortex_FS_20k.mat"
    leadfield_path = Path(args.leadfield_path) if args.leadfield_path else anatomy_dir / "Gain_20k_FS.mat"
    projection_path = get_projection_path(args)
    nmm_root = args.nmm_bank or args.nmm_dir
    return G2SourceSimulator.from_paths(leadfield_path, cortex_path, projection_path, Path(nmm_root))


def validate_named_projection(args: argparse.Namespace, backend: G2SourceSimulator) -> dict | None:
    if args.atlas is None:
        return None
    profile = get_atlas_profile(args.atlas)
    if backend.n_regions != profile["n_regions"]:
        raise ValueError("The projection region count does not match --atlas")
    expected_projection = load_projection_matrix(profile["projection_path"])
    if projection_hash(backend.projection_matrix) != projection_hash(expected_projection):
        raise ValueError("Projection differs from the named atlas; use a custom bank plan without a named atlas")
    return profile


def build_spatial_config(args: argparse.Namespace) -> SpatialDynamicsConfig:
    return SpatialDynamicsConfig(
        mode=args.spatial_mode,
        area_threshold_cm2=args.diffusion_area_cm2 or None,
        hop_threshold=args.diffusion_hops or None,
        start_s=args.diffusion_start_s,
        duration_s=args.diffusion_duration_s,
    )


def build_g1_config(args: argparse.Namespace) -> G1SimulationConfig:
    spectral = (json.loads(Path(args.g1_spectral_config).read_text(encoding="utf-8"))
                if args.g1_spectral_config else {})
    if args.source_type not in G1_WAVEFORM_PRESETS:
        raise ValueError(f"Unsupported G1 source_type {args.source_type!r}; choose from {sorted(G1_WAVEFORM_PRESETS)}")
    duration_s = float(args.duration_s)
    return G1SimulationConfig(
        source_model=args.g1_source_model, spectral=spectral,
        multiscale=(json.loads(args.g1_multiscale_config.read_text(encoding='utf-8'))
                    if args.g1_multiscale_config else {}),
        source_background_multiscale=(json.loads(args.background_multiscale_config.read_text(encoding='utf-8'))
                                     if args.background_multiscale_config else {}),
        sample_id=args.sample_id,
        template_id="fsaverage5",
        source_space_id="Cortex_FS_20k",
        leadfield_id=Path(args.leadfield_path).name if args.leadfield_path else "Gain_20k_FS",
        modality=args.modality,
        srate=float(args.srate_hz),
        time_window=(0.0, duration_s),
        n_patch=int(args.n_active),
        patch_area_mm2=float(args.patch_area_mm2),
        patch_geometry=args.patch_geometry,
        decay_mm=float(args.decay_mm),
        min_seed_distance_mm=float(args.min_seed_distance_mm),
        spatial=build_spatial_config(args),
        waveform_params=[G1_WAVEFORM_PRESETS[args.source_type]],
        source_background_type=args.source_background_type,
        source_background_spectral=(json.loads(Path(args.source_background_config).read_text(encoding='utf-8'))
                                    if args.source_background_config else {}),
        source_snr_db=args.source_snr_db,
        sensor_noise_type=args.sensor_noise_type,
        real_noise_registry=str(args.real_noise_registry) if args.real_noise_registry else None,
        real_noise_record_id=args.real_noise_record_id,
        real_noise_split=args.real_noise_split,
        allow_combined_real_source_background=args.allow_combined_real_source_background,
        sensor_snr_db=args.sensor_snr_db,
        sensor_snr_mode=args.sensor_snr_mode,
        sensor_snr_clip_quantile=args.sensor_snr_clip_quantile,
        baseline_duration_s=args.baseline_duration_s,
        covariance_estimator=args.covariance_estimator,
        average_reference=args.average_reference,
        normalize_source=args.amplitude_mode != 'physical',
        normalize_sensor=False,
        random_seed=args.random_seed,
        amplitude_mode=args.amplitude_mode, foreground_peak_am=args.foreground_peak_am,
        leadfield_source_unit=args.leadfield_source_unit,
        sensor_units=(args.sensor_units[0] if len(args.sensor_units) == 1 else args.sensor_units) if args.sensor_units else None,
        sensor_noise_std=(args.sensor_noise_std[0] if len(args.sensor_noise_std) == 1 else args.sensor_noise_std) if args.sensor_noise_std else None,
        record_observability=args.record_observability,
    )


def run_g1(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    if args.atlas is not None:
        raise ValueError("--atlas is available only with --engine g2 or g3")
    anatomy_dir = Path(args.anatomy_dir)
    cortex_path = Path(args.cortex_path) if args.cortex_path else anatomy_dir / "Cortex_FS_20k.mat"
    leadfield_path = Path(args.leadfield_path) if args.leadfield_path else anatomy_dir / "Gain_20k_FS.mat"
    cortex = load_brainstorm_cortex(cortex_path, use_cortex_mask=args.cortex_mask)
    n_vertices = np.asarray(cortex["Vertices"]).shape[-1]
    leadfield = load_brainstorm_leadfield(leadfield_path, n_vertices=n_vertices)
    channel_names = json.loads(args.channel_names.read_text(encoding='utf-8')) if args.channel_names else None
    simulator = G1SourceSimulator.from_brainstorm_cortex(leadfield, cortex, channel_names=channel_names)
    config = build_g1_config(args)
    sample = simulator.generate_sample(config)
    manifest = dict(sample['manifest'])
    arrays = {key: value for key, value in sample.items() if key != 'manifest'}
    sensor, clean, noise = arrays['sensor'], arrays['clean_sensor'], arrays['sensor_noise']
    source = arrays.get('source_total', arrays.get('source'))
    manifest.update({"high_level_engine": "g1", "source_type": args.source_type
                     if config.source_model == "patch" else config.spectral.get("scenario", "rest")})
    return save_standard_sample(output_dir, args.sample_id, sensor, clean, noise, source, None,
                                manifest, args.plot, sample_arrays=arrays, resume=args.resume)


def build_g2_config(args: argparse.Namespace, active_regions: list[int] | None) -> G2SimulationConfig:
    return G2SimulationConfig(
        record_observability=args.record_observability,
        sample_id=args.sample_id,
        template_id="fsaverage5",
        source_space_id="Cortex_FS_20k",
        leadfield_id=Path(args.leadfield_path).name if args.leadfield_path else "Gain_20k_FS",
        projection_id=(get_projection_path(args).name
                       if args.projection_path or args.atlas in ("aal78", "aal78_membership") else "FS_994nmm_to_20k"),
        modality=args.modality,
        srate=float(args.srate_hz),
        time_window=(0.0, float(args.duration_s)),
        n_active_regions=len(active_regions) if active_regions is not None else args.n_active,
        active_region_ids=active_regions,
        weight_decay_mm=float(args.decay_mm),
        weight_decay_hops=1,
        spatial=build_spatial_config(args),
        normalize_region_signals=False,
        normalize_source=True,
        sensor_noise_type=args.sensor_noise_type,
        sensor_snr_db=args.sensor_snr_db,
        sensor_snr_mode=args.sensor_snr_mode,
        sensor_snr_clip_quantile=args.sensor_snr_clip_quantile,
        baseline_duration_s=args.baseline_duration_s,
        covariance_estimator=args.covariance_estimator,
        average_reference=args.average_reference,
        random_seed=args.random_seed,
        nmm_sample_id=args.nmm_sample,
        nmm_state=args.nmm_state,
        nmm_source_mode=args.nmm_source_mode,
        nmm_start_s=args.nmm_start_s,
    )


def run_g2_bank(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    if args.keep_nmm_background:
        raise ValueError("--keep-nmm-background is legacy-only; use --nmm-source-mode network for a bank")
    if args.source_type and args.nmm_state and args.source_type != args.nmm_state:
        raise ValueError("--source-type and --nmm-state must match when both are supplied")
    active_regions = parse_int_list(args.active_regions) or None
    if active_regions is not None and args.nmm_source_mode != "selected":
        raise ValueError("--active-regions requires --nmm-source-mode selected for a bank")
    backend = load_fs20k_backend(args)
    profile = validate_named_projection(args, backend)
    if profile is not None and read_bank(args.nmm_bank)["atlas_id"] != profile["atlas_id"]:
        raise ValueError("Bank atlas_id does not match --atlas; omit --atlas to replay a custom bank")
    config = build_g2_config(args, active_regions)
    config.nmm_state = args.nmm_state or args.source_type
    source, region_source, meta = backend.generate_source(config)
    sensor, clean, noise, source, manifest = backend._finalize_sensor_from_source(  # noqa: SLF001
        source, region_source, meta, config,
    )
    manifest.update({"high_level_engine": "g2", "source_type": manifest.get("nmm_state", config.nmm_state)})
    time_ms = np.arange(region_source.shape[1], dtype=float) * 1000.0 / config.srate
    return save_standard_sample(output_dir, args.sample_id, sensor, clean, noise, source,
                                region_source, manifest, args.plot, time_ms=time_ms, resume=args.resume)


def run_g2(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    if args.nmm_bank:
        return run_g2_bank(args, output_dir)
    if args.atlas in ("aal78", "aal78_membership"):
        raise ValueError("--atlas aal78 requires --nmm-bank for G2; the legacy spike library has 994 regions")
    if args.nmm_sample or args.nmm_state or args.nmm_source_mode != "auto" or args.nmm_start_s != 0:
        raise ValueError("Bank sample/state/mode/offset options require --nmm-bank")
    if not np.isclose(float(args.duration_s), 1.0, rtol=0, atol=1e-9):
        raise ValueError("NMM clips have a fixed one-second duration; use --duration-s 1")
    backend = load_fs20k_backend(args)
    active_regions = parse_int_list(args.active_regions)
    if not active_regions:
        candidates = [region for region in range(backend.n_regions)
                      if (Path(args.nmm_dir) / f"a{region}" / f"nmm_{int(args.clip_id)}.mat").is_file()]
        active_regions = backend.sample_regions(
            args.n_active, np.random.default_rng(args.random_seed), candidate_regions=candidates)
    if len(set(active_regions)) != len(active_regions) or any(r < 0 or r >= backend.n_regions for r in active_regions):
        raise ValueError("Active regions must be unique valid output-region IDs")
    config = build_g2_config(args, active_regions)
    region_source = None
    region_info = []
    for region_id in active_regions:
        clip = load_nmm_spike_clip(Path(args.nmm_dir), region_id, int(args.clip_id))
        clip = prepare_nmm_clip(clip, config.srate, config.time_window).astype(np.float32)
        if clip.shape[0] != backend.n_regions:
            raise ValueError("NMM clip region count must match the projection matrix")
        if region_source is None:
            # Keep one background realization, then replace every selected active row.
            region_source = clip.copy() if args.keep_nmm_background else np.zeros_like(clip)
        region_source[region_id] = clip[region_id]
        signal = np.asarray(clip[region_id], dtype=float)
        background_mask = np.ones(clip.shape[0], dtype=bool)
        background_mask[region_id] = False
        background = clip[background_mask, : region_source.shape[1]]
        signal_power = float(np.mean(signal**2))
        background_power = float(np.mean(background**2)) if background.size else 0.0
        peak_idx = int(np.argmax(signal))
        peak_time_s = peak_idx / config.srate
        region_info.append(
            {
                "region_id": int(region_id),
                "clip_id": int(args.clip_id),
                "source_mat": str(Path(args.nmm_dir) / f"a{region_id}" / f"nmm_{int(args.clip_id)}.mat"),
                "signal_power": signal_power,
                "background_power": background_power,
                "source_level_snr_db": float(10.0 * np.log10(max(signal_power, 1e-12) / max(background_power, 1e-12))),
                "peak_idx": peak_idx,
                "peak_time_ms": 1000.0 * peak_time_s,
                "peak_is_boundary": peak_time_s < 0.05 or peak_time_s > 0.95,
            }
        )
    source, spatial_meta = backend.project_regions(
        region_source, active_regions, srate=config.srate, time_window=config.time_window,
        decay_mm=config.weight_decay_mm, spatial=config.spatial)
    if config.normalize_source:
        source = source - source.mean(axis=1, keepdims=True)
        source = source / max(float(np.max(np.abs(source))), 1e-12)
    meta = {
        **asdict(config),
        "generator_type": "G2_NMM_spike_clip_high_level",
        "high_level_engine": "g2",
        "source_type": "nmm_spike",
        "selected_regions": active_regions,
        "nmm_native_srate_hz": 500.0,
        "nmm_background_policy": "first_clip" if args.keep_nmm_background else "none",
        "nmm_background_region": active_regions[0] if args.keep_nmm_background else None,
        "region_info": region_info,
        "n_vertices": backend.n_vertices,
        "n_regions": backend.n_regions,
        "n_sensors": backend.n_sensors,
        "leadfield_shape": list(backend.leadfield.shape),
        "projection_shape": list(backend.projection_matrix.shape),
        **spatial_meta,
    }
    sensor, clean, noise, source, manifest = backend._finalize_sensor_from_source(  # noqa: SLF001
        np.asarray(source, dtype=np.float32),
        region_source,
        _json_safe(meta),
        config,
    )
    return save_standard_sample(output_dir, args.sample_id, sensor, clean, noise, source, region_source, manifest, args.plot, resume=args.resume)


def build_g3_config(args: argparse.Namespace, active_regions: list[int]) -> TVBSignalConfig:
    if build_spatial_config(args).mode not in {"static", "auto"}:
        raise ValueError("G3 network trajectories do not support synthetic spatial modes; use G2 selected projection")
    if args.source_type not in G3_TVB_PRESETS:
        raise ValueError(f"Unsupported G3 source_type {args.source_type!r}; choose from {sorted(G3_TVB_PRESETS)}")
    preset = G3_TVB_PRESETS[args.source_type]
    parameters = {"jr_active_A": None, "jr_mu": None, "jr_noise_scale": 1.0, **preset}
    if args.source_type == "resting_jr":
        parameters["jr_active_A"] = args.jr_background_A
    for name in parameters:
        if getattr(args, name) is not None:
            parameters[name] = getattr(args, name)
    profile = get_atlas_profile(args.atlas)
    connectivity_path = args.connectivity_path
    if connectivity_path is None and args.atlas is not None:
        connectivity_path = str(profile["connectivity_path"])
    return TVBSignalConfig(
        sample_id=args.sample_id,
        signal_type=args.source_type,
        connectivity_path=connectivity_path,
        active_regions=active_regions,
        simulation_length_ms=1000.0 * float(args.duration_s),
        warmup_ms=float(args.warmup_ms),
        dt_ms=float(args.tvb_dt_ms),
        output_srate_hz=float(args.srate_hz),
        random_seed=args.random_seed or 0,
        reshape_to_994=profile["reshape_to_994"],
        normalize=True,
        discard_ms=0.0,
        jr_background_A=float(args.jr_background_A),
        jr_coupling_mode=args.jr_coupling_mode,
        **parameters,
    )


def run_g3(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    backend = load_fs20k_backend(args)
    validate_named_projection(args, backend)
    active_regions = parse_int_list(args.active_regions) or [10]
    tvb_config = build_g3_config(args, active_regions)
    if args.atlas is not None:
        validate_atlas_connectivity(get_atlas_profile(args.atlas), tvb_config.connectivity_path)
    region_source, time_ms, tvb_manifest = TVBSourceGenerator(tvb_config.connectivity_path).generate(tvb_config)
    if region_source.shape[0] != backend.n_regions:
        raise ValueError("TVB output region count must match the projection matrix and selected atlas")
    source = backend.projection_matrix @ region_source
    source = np.asarray(source, dtype=np.float32)
    source -= source.mean(axis=1, keepdims=True)
    source /= max(float(np.max(np.abs(source))), 1e-12)
    g2_config = build_g2_config(args, active_regions)
    g2_config.normalize_region_signals = tvb_config.normalize
    g2_config.time_window = (0.0, region_source.shape[1] / float(args.srate_hz))
    meta = {
        **asdict(g2_config),
        "generator_type": "G3_TVB_projected_high_level",
        "high_level_engine": "g3",
        "source_type": args.source_type,
        "selected_regions": active_regions,
        "tvb_manifest": tvb_manifest,
        "n_vertices": backend.n_vertices,
        "n_regions": backend.n_regions,
        "n_sensors": backend.n_sensors,
        "leadfield_shape": list(backend.leadfield.shape),
        "projection_shape": list(backend.projection_matrix.shape),
    }
    if args.atlas is not None:
        profile = get_atlas_profile(args.atlas)
        meta.update({"atlas": args.atlas, "atlas_id": profile["atlas_id"],
                     "region_names": profile["region_names"]})
    sensor, clean, noise, source, manifest = backend._finalize_sensor_from_source(  # noqa: SLF001
        source,
        region_source,
        _json_safe(meta),
        g2_config,
    )
    return save_standard_sample(output_dir, args.sample_id, sensor, clean, noise, source, region_source, manifest, args.plot, time_ms=time_ms, resume=args.resume)


def save_standard_sample(
    output_dir: Path,
    sample_id: str,
    sensor: np.ndarray,
    clean: np.ndarray,
    noise: np.ndarray,
    source: np.ndarray,
    region_source: np.ndarray | None,
    manifest: dict[str, Any],
    plot: bool,
    time_ms: np.ndarray | None = None,
    source_components: dict[str, np.ndarray] | None = None,
    *, sample_arrays: dict[str, np.ndarray] | None = None,
    request_identity: dict[str, Any] | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Save once; generic resume verifies generated arrays rather than skipping work."""
    manifest = dict(manifest)

    data_name = f"{sample_id}_data.npz"
    save_payload: dict[str, Any] = {
        "sensor": sensor,
        "clean_sensor": clean,
        "sensor_noise": noise,
        "source": source,
    }
    if sample_arrays is not None:
        save_payload.update(sample_arrays)
    covariance_source = manifest if sample_arrays is None else save_payload
    if not all(key in covariance_source for key in NOISE_ARRAY_KEYS):
        raise ValueError('Generated sensor samples must include a baseline and its covariance; regenerate with the current generator')
    for key in NOISE_ARRAY_KEYS:
        values = np.asarray(covariance_source[key], dtype=np.float32 if key == 'baseline_sensor' else np.float64)
        valid_shape = (values.ndim == 2 and values.shape[0] == sensor.shape[0]
                       and (values.shape[1] >= 2 if key == 'baseline_sensor' else values.shape[1] == sensor.shape[0]))
        if not valid_shape or not np.isfinite(values).all():
            raise ValueError(f'{key} must be finite and match the exported sensor channels')
        save_payload[key] = values
    for key in NOISE_ARRAY_KEYS:
        manifest.pop(key, None)
    manifest["noise_covariance_file"] = data_name
    if region_source is not None:
        save_payload["region_source"] = region_source
    if time_ms is not None:
        save_payload["time_ms"] = time_ms
    if source_components:
        save_payload.update(source_components)
        manifest['source_components_file'] = data_name
        manifest['source_component_keys'] = sorted(source_components)
    manifest.update(
        {
            "standard_data": data_name,
            "files": {
                "npz": data_name,
                "manifest": f"{sample_id}_manifest.json",
            },
            "arrays": sorted(save_payload.keys()),
            "sensor_shape": list(sensor.shape),
            "clean_sensor_shape": list(clean.shape),
            "sensor_noise_shape": list(noise.shape),
            "source_shape": list(source.shape),
            "region_source_shape": list(region_source.shape) if region_source is not None else None,
        }
    )
    if 'source_psd_summary' not in manifest:
        manifest['source_psd_summary'] = estimate_power_spectrum_summary(source, float(manifest.get('srate', 500.0)))
    artifacts = {}
    if plot:
        figure_name = f"{sample_id}_overview.png"
        buffer = BytesIO()
        plot_standard_overview(buffer, sample_id, sensor, clean, noise, source, region_source, manifest)
        artifacts[figure_name] = buffer.getvalue()
        manifest["overview_figure"] = figure_name
    manifest = _json_safe(manifest)
    if request_identity is None:
        array_hashes = {}
        for key, value in save_payload.items():
            array = np.ascontiguousarray(value)
            digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode('ascii'))
            digest.update(array.tobytes())
            array_hashes[key] = digest.hexdigest()
        request_identity = dict(manifest=manifest, arrays_sha256=array_hashes,
                                high_level_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return save_sample_directory(output_dir, sample_id, save_payload, manifest,
                                 request_identity=request_identity, resume=resume, artifacts=artifacts)


def plot_standard_overview(
    output_path: Path,
    sample_id: str,
    sensor: np.ndarray,
    clean: np.ndarray,
    noise: np.ndarray,
    source: np.ndarray,
    region_source: np.ndarray | None,
    manifest: dict[str, Any],
) -> None:
    srate = float(manifest.get("srate", 500.0))
    time = np.arange(sensor.shape[1]) / srate
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), constrained_layout=True)
    fig.suptitle(f"{sample_id} | {manifest.get('high_level_engine')}:{manifest.get('source_type')}", fontsize=13)

    channels = np.arange(min(8, sensor.shape[0]))
    sensor_offset = max(float(np.std(sensor[channels])), 1e-8) * 2.0
    for idx, ch in enumerate(channels):
        axes[0].plot(time, sensor[ch] + idx * sensor_offset, linewidth=0.7)
    axes[0].set_title(f"Noisy {manifest.get('modality', 'EEG')} sensor traces")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("channel + offset")

    clean_offset = max(float(np.std(clean[channels])), 1e-8) * 2.0
    for idx, ch in enumerate(channels):
        axes[1].plot(time, clean[ch] + idx * clean_offset, linewidth=0.7, color="tab:green")
    axes[1].set_title("Clean sensor projection")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("channel + offset")

    noise_offset = max(float(np.std(noise[channels])), 1e-8) * 2.0
    for idx, ch in enumerate(channels):
        axes[2].plot(time, noise[ch] + idx * noise_offset, linewidth=0.7, color="tab:red")
    axes[2].set_title(f"Sensor noise: {manifest.get('sensor_noise_type')}")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("channel + offset")

    if region_source is not None:
        selected = manifest.get("selected_regions") or manifest.get("active_regions") or [0]
        region_id = int(selected[0])
        region_id = max(0, min(region_id, region_source.shape[0] - 1))
        axes[3].plot(np.arange(region_source.shape[1]) / srate, region_source[region_id], color="tab:purple", linewidth=1.0)
        axes[3].set_title(f"Region source trace: region {region_id}")
    else:
        active = np.argsort(np.mean(source**2, axis=1))[-min(8, source.shape[0]) :]
        source_offset = max(float(np.std(source[active])), 1e-8) * 2.0
        for idx, vertex in enumerate(active):
            axes[3].plot(time, source[vertex] + idx * source_offset, linewidth=0.7, color="tab:purple")
        axes[3].set_title("High-power vertex source preview")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("source amplitude")

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="High-level interface for simulated source activity and EEG/MEG-like sensor data.")
    parser.add_argument("--engine", choices=["g1", "g2", "g3"], default="g2")
    parser.add_argument("--atlas", choices=ATLAS_CHOICES, default=None,
                        help="G2 bank/G3 atlas: legacy 998 nodes (994 exported), or 78 cortical AAL regions")
    parser.add_argument("--source-type", default=None)
    parser.add_argument("--sample-id", default="sim_sample")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--anatomy-dir", default=str(ANATOMY_DIR))
    parser.add_argument("--cortex-path", default=None)
    parser.add_argument("--leadfield-path", default=None)
    parser.add_argument("--projection-path", default=None)
    parser.add_argument("--connectivity-path", default=None)
    parser.add_argument("--nmm-dir", default=str(NMM_SPIKES_DIR), help="Legacy one-second NMM spike library")
    parser.add_argument("--nmm-bank", default=None, help="G3-generated bank directory containing bank.json (G2 only)")
    parser.add_argument("--nmm-sample", default=None, help="Bank sample ID; otherwise sampled with --random-seed")
    parser.add_argument("--nmm-state", default=None, help="Filter bank by signal_type, e.g. alpha_jr")
    parser.add_argument("--nmm-source-mode", choices=["auto", "network", "selected"], default="auto",
                        help="Bank auto=network; selected masks output regions without rerunning the NMM")
    parser.add_argument("--nmm-start-s", type=float, default=0.0, help="Offset within the saved, already warmed-up bank signal")
    parser.add_argument("--modality", choices=["EEG", "MEG"], default="EEG")
    parser.add_argument("--srate-hz", type=float, default=500.0)
    parser.add_argument("--duration-s", type=float, default=1.0,
                        help="Saved duration in seconds; G3 warm-up is additional (default: 1)")
    parser.add_argument("--active-regions", default=None,
                        help="Comma-separated output-region IDs; legacy randomly samples --n-active; G3 default: 10; bank selected uses entry metadata")
    parser.add_argument("--n-active", type=int, default=1)
    parser.add_argument("--clip-id", type=int, default=1)
    parser.add_argument("--keep-nmm-background", action="store_true")
    parser.add_argument("--patch-area-mm2", type=float, default=180.0)
    parser.add_argument('--patch-geometry', choices=('geodesic', 'legacy_bfs'), default='geodesic')
    parser.add_argument("--decay-mm", type=float, default=12.0)
    parser.add_argument("--min-seed-distance-mm", type=float, default=25.0)
    parser.add_argument("--spatial-mode", choices=["static", "auto", "diffuse", "contract", "pulse"], default="auto",
                        help="Patch weighting: static, large-source diffusion, forced diffusion, contraction or spread/recovery")
    parser.add_argument("--diffusion-area-cm2", type=float, default=10., help="Automatic diffusion area threshold; 0 disables it")
    parser.add_argument("--diffusion-hops", type=int, default=5, help="Automatic diffusion graph-radius threshold; 0 disables it")
    parser.add_argument("--diffusion-start-s", type=float, default=0.)
    parser.add_argument("--diffusion-duration-s", type=float, default=.3)
    parser.add_argument("--source-background-type", default=None)
    parser.add_argument('--source-background-config', type=Path, help='G1 spectral_rest background settings JSON')
    parser.add_argument('--cortex-mask', action=argparse.BooleanOptionalAction, default=True,
                        help='Apply available cortical scouts to G1 patch placement; --no-cortex-mask is legacy replay')
    parser.add_argument("--source-snr-db", type=float, default=None)
    parser.add_argument("--sensor-noise-type", default="iid_white", choices=["iid_white", "white", "pink", "powerlaw", "specparam_resting", "mixed", "eegsourcesim_resting", "real"])
    parser.add_argument('--real-noise-registry', type=Path)
    parser.add_argument('--real-noise-record-id')
    parser.add_argument('--real-noise-split', choices=('train', 'val', 'test'))
    parser.add_argument('--channel-names', type=Path, help='JSON list in the exact leadfield row order')
    parser.add_argument('--allow-combined-real-source-background', action='store_true')
    parser.add_argument("--sensor-snr-db", type=lambda value: None if value.lower() == 'none' else float(value), default=10.0)
    parser.add_argument('--amplitude-mode', choices=('relative', 'physical'), default='relative')
    parser.add_argument('--foreground-peak-am', type=float)
    parser.add_argument('--leadfield-source-unit', choices=('A*m',))
    parser.add_argument('--sensor-units', nargs='+')
    parser.add_argument('--sensor-noise-std', nargs='+', type=float)
    parser.add_argument('--record-observability', action='store_true')
    parser.add_argument("--sensor-snr-mode", default="active_rms", choices=["global_rms", "robust_global_rms", "active_rms"])
    parser.add_argument("--sensor-snr-clip-quantile", type=float, default=0.95)
    parser.add_argument("--baseline-duration-s", type=float, default=2.0)
    parser.add_argument('--covariance-estimator', choices=('empirical', 'oas'), default='empirical')
    parser.add_argument("--g1-source-model", choices=("patch", "lbo", "lbo_network"), default="patch")
    parser.add_argument("--g1-spectral-config", type=Path, help="JSON settings for the G1 spectral source model")
    parser.add_argument('--g1-multiscale-config', type=Path, help='Optional spectral foreground residual settings JSON')
    parser.add_argument('--background-multiscale-config', type=Path, help='Optional spectral_rest background residual settings JSON')
    parser.add_argument("--average-reference", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--random-seed", type=int, default=20240629)
    parser.add_argument("--warmup-ms", type=float, default=1000.0,
                        help="G3 startup time simulated and automatically discarded (default: 1000 ms)")
    parser.add_argument("--tvb-dt-ms", type=float, default=0.5)
    parser.add_argument("--jr-background-A", type=float, default=3.25)
    parser.add_argument("--jr-active-A", type=float, default=None, help="Override the source-type preset")
    parser.add_argument("--jr-mu", type=float, default=None, help="Override the source-type preset")
    parser.add_argument("--jr-coupling-mode", choices=("sparse", "dense"), default="sparse")
    parser.add_argument("--jr-noise-scale", type=float, default=None, help="Override the source-type preset")
    parser.add_argument("--coupling-strength", type=float, default=None, help="Override the source-type preset")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument('--resume', action='store_true',
                        help='Verify and reuse identical committed output after generation; never overwrite')
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.engine != 'g1' and (args.sensor_noise_type == 'real' or args.real_noise_registry
            or args.real_noise_record_id or args.real_noise_split or args.channel_names
            or args.allow_combined_real_source_background or args.g1_multiscale_config
            or args.background_multiscale_config):
        parser.error('Registered NMM noise uses nmm_simulation.py; multiscale settings are G1-only')
    if args.engine != 'g1' and (args.amplitude_mode != 'relative' or any(
            value is not None for value in (args.foreground_peak_am, args.leadfield_source_unit, args.sensor_units, args.sensor_noise_std))):
        parser.error('Physical NMM generation uses nmm_simulation.py with an explicit JSON unit contract')
    if args.engine != 'g1' and args.source_background_config is not None:
        parser.error('--source-background-config is only available with G1')
    if args.engine == "g1" and args.atlas is not None:
        parser.error("--atlas is available only with --engine g2 or g3")
    if args.engine != "g2" and (args.nmm_bank or args.nmm_sample or args.nmm_state
                               or args.nmm_source_mode != "auto" or args.nmm_start_s != 0):
        parser.error("--nmm-bank and its sample/state/mode/offset options are only available with --engine g2")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.engine == "g2":
        if not args.nmm_bank:
            args.source_type = args.source_type or "nmm_spike"
        manifest = run_g2(args, output_dir)
    elif args.engine == "g1":
        args.source_type = args.source_type or "erp"
        manifest = run_g1(args, output_dir)
    else:
        args.source_type = args.source_type or "resting_jr"
        manifest = run_g3(args, output_dir)

    print(
        json.dumps(
            _json_safe(
                {
                    "sample_id": manifest["sample_id"],
                    "engine": manifest.get("high_level_engine"),
                    "source_type": manifest.get("source_type"),
                    "output_dir": str(output_dir / manifest["sample_id"]),
                    "arrays": manifest["arrays"],
                    "sensor_shape": manifest["sensor_shape"],
                    "source_shape": manifest["source_shape"],
                    "region_source_shape": manifest.get("region_source_shape"),
                    "realized_sensor_snr_db": manifest.get("realized_sensor_snr_db"),
                    "overview_figure": manifest.get("overview_figure"),
                }
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
