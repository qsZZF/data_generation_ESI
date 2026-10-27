from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse

try:
    from .g1_simulation import G1DifficultyScenarioSampler, G1SimulationConfig, G1SourceSimulator
except ImportError:
    from g1_simulation import G1DifficultyScenarioSampler, G1SimulationConfig, G1SourceSimulator


ROOT = Path(__file__).resolve().parents[1]
ANATOMY_DIR = ROOT / "Anatomy"
OUTPUT_DIR = ROOT / "Simulation_G1" / "generated_fs20k_demo"


def load_fs20k_cortex(cortex_path: Path) -> dict[str, Any]:
    with h5py.File(cortex_path, "r") as h5_file:
        group = h5_file["Cortex_FS_20k"]
        vert_conn = group["VertConn"]
        return {
            "Vertices": np.asarray(group["Vertices"][:], dtype=float),
            "Faces": np.asarray(group["Faces"][:], dtype=int),
            "VertConn": scipy.sparse.csc_matrix(
                (
                    np.asarray(vert_conn["data"][:]).reshape(-1),
                    np.asarray(vert_conn["ir"][:]).reshape(-1),
                    np.asarray(vert_conn["jc"][:]).reshape(-1),
                )
            ),
        }


def load_fixed_orientation_leadfield(leadfield_path: Path, n_vertices: int) -> np.ndarray:
    with h5py.File(leadfield_path, "r") as h5_file:
        group = h5_file["LeadField_fs_20k"]
        gain = np.asarray(group["Gain"][:], dtype=float)
        orientations = np.asarray(group["GridOrient"][:], dtype=float)
        if orientations.shape[0] != 3:
            orientations = orientations.T
        if gain.shape[0] != 3 * n_vertices:
            raise ValueError(f"Expected Gain rows = 3*n_vertices, got {gain.shape[0]} for {n_vertices} vertices")
        leadfield = np.empty((gain.shape[1], n_vertices), dtype=np.float32)
        for vertex_idx in range(n_vertices):
            components = gain[3 * vertex_idx : 3 * vertex_idx + 3, :]
            leadfield[:, vertex_idx] = orientations[:, vertex_idx] @ components
    leadfield -= np.mean(leadfield, axis=0, keepdims=True)
    return leadfield


def base_config(sample_id: str, waveform: tuple[str, dict[str, Any]], seed: int) -> G1SimulationConfig:
    return G1SimulationConfig(
        sample_id=sample_id,
        template_id="fsaverage5",
        source_space_id="Cortex_FS_20k",
        leadfield_id="Gain_20k_FS_fixed_orientation",
        modality="EEG",
        srate=250,
        time_window=(-0.2, 0.8),
        n_patch=1,
        patch_area_mm2=180.0,
        decay_mm=12.0,
        min_seed_distance_mm=25.0,
        connectivity_mode="independent",
        waveform_params=[waveform],
        source_background_type=None,
        source_snr_db=None,
        sensor_noise_type="pink",
        sensor_snr_db=12.0,
        random_seed=seed,
    )


def build_demo_configs() -> list[G1SimulationConfig]:
    configs = [
        base_config(
            "G1_fs20k_erp",
            (
                "erp",
                {
                    "components": [
                        {"type": "gaussian", "latency_s": 0.12, "width_s": 0.045, "amplitude_rel": 1.0},
                        {"type": "gaussian", "latency_s": 0.28, "width_s": 0.075, "amplitude_rel": -0.55},
                    ],
                    "amplitude": 1.0,
                },
            ),
            101,
        ),
        base_config(
            "G1_fs20k_gabor_erp",
            (
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
            102,
        ),
        base_config(
            "G1_fs20k_ersp_alpha_burst",
            (
                "ersp",
                {
                    "base_frequency_band_Hz": 10.0,
                    "modulation": {"type": "burst", "latency_s": 0.25, "width_s": 0.35, "min_rel_amplitude": 0.08},
                    "amplitude": 1.0,
                },
            ),
            103,
        ),
        base_config(
            "G1_fs20k_spike",
            ("spike", {"latency_s": 0.16, "duration_s": 0.075, "shape": "exponential_decay", "amplitude": 1.0}),
            104,
        ),
        base_config(
            "G1_fs20k_resting_alpha",
            (
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
            105,
        ),
        base_config(
            "G1_fs20k_eegsourcesim_resting",
            (
                "eegsourcesim_resting_noise",
                {
                    "exponent": 1.1,
                    "alpha_peak_hz": 10.0,
                    "alpha_width_hz": 1.8,
                    "alpha_gain": 1.4,
                    "amplitude": 1.0,
                },
            ),
            106,
        ),
        base_config(
            "G1_fs20k_powerlaw_aperiodic",
            ("powerlaw", {"exponent": 1.4, "white_floor": 0.04, "amplitude": 1.0}),
            109,
        ),
        base_config(
            "G1_fs20k_knee_powerlaw",
            ("knee_powerlaw", {"exponent": 1.6, "knee_hz": 1.2, "white_floor": 0.03, "amplitude": 1.0}),
            110,
        ),
        base_config(
            "G1_fs20k_specparam_alpha",
            (
                "specparam",
                {
                    "exponent": 1.2,
                    "knee_hz": 0.8,
                    "peaks": [{"center_hz": 10.0, "bandwidth_hz": 1.6, "height": 0.8}],
                    "envelope_depth": 0.12,
                    "amplitude": 1.0,
                },
            ),
            111,
        ),
        base_config(
            "G1_fs20k_synaptic_shot_noise",
            (
                "synaptic_noise",
                {
                    "event_rate_hz": 45.0,
                    "tau_rise_s": 0.006,
                    "tau_decay_s": 0.070,
                    "amplitude": 1.0,
                },
            ),
            112,
        ),
        base_config(
            "G1_fs20k_burst_oscillation",
            (
                "burst_oscillation",
                {
                    "frequency_hz": 10.0,
                    "duty_cycle": 0.45,
                    "min_duration_s": 0.08,
                    "max_duration_s": 0.30,
                    "amplitude": 1.0,
                },
            ),
            113,
        ),
        base_config(
            "G1_fs20k_nonsinusoidal_alpha",
            (
                "nonsinusoidal_oscillation",
                {
                    "frequency_hz": 10.0,
                    "harmonics": {2: 0.35, 3: 0.15},
                    "amplitude": 1.0,
                },
            ),
            114,
        ),
    ]

    configs[-6].sensor_noise_type = "powerlaw"
    configs[-5].sensor_noise_type = "specparam_resting"
    configs[-4].sensor_noise_type = "specparam_resting"
    configs[-3].source_background_type = "synaptic"
    configs[-3].source_snr_db = 30.0
    configs[-2].sensor_noise_type = "mixed"
    configs[-1].sensor_noise_type = "powerlaw"

    correlated = base_config("G1_fs20k_correlated_3patch", ("erp", {"components": [{"latency_s": 0.14, "width_s": 0.05}]}), 107)
    correlated.n_patch = 3
    correlated.patch_area_mm2 = [120.0, 180.0, 240.0]
    correlated.decay_mm = [10.0, 14.0, 18.0]
    correlated.connectivity_mode = "correlated"
    correlated.coupling = 0.75
    correlated.waveform_params = [
        ("erp", {"components": [{"type": "gaussian", "latency_s": 0.12, "width_s": 0.04, "amplitude_rel": 1.0}], "amplitude": 1.0}),
        ("ersp", {"base_frequency_band_Hz": 12.0, "modulation": {"type": "burst", "latency_s": 0.24, "width_s": 0.3}, "amplitude": 1.0}),
        ("spike", {"latency_s": 0.18, "duration_s": 0.07, "amplitude": 1.0}),
    ]
    correlated.source_background_type = "eegsourcesim_resting"
    correlated.source_snr_db = 28.0
    correlated.sensor_noise_type = "eegsourcesim_resting"
    correlated.sensor_snr_db = 8.0
    configs.append(correlated)

    causal = base_config("G1_fs20k_causal_4patch", ("ar", {"freq": 10.0, "amplitude": 1.0}), 108)
    causal.n_patch = 4
    causal.patch_area_mm2 = [80.0, 120.0, 160.0, 200.0]
    causal.decay_mm = [8.0, 12.0, 16.0, 20.0]
    causal.connectivity_mode = "causal"
    causal.coupling = [
        0.45, 0.10, 0.00, 0.00,
        0.20, 0.45, 0.12, 0.00,
        0.00, 0.15, 0.45, 0.12,
        0.00, 0.00, 0.15, 0.45,
    ]
    causal.waveform_params = [
        ("ar", {"freq": 8.0, "amplitude": 1.0}),
        ("ar", {"freq": 10.0, "amplitude": 1.0}),
        ("ar", {"freq": 15.0, "amplitude": 1.0}),
        ("ar", {"freq": 20.0, "amplitude": 1.0}),
    ]
    causal.source_background_type = "pink"
    causal.source_snr_db = 25.0
    causal.sensor_noise_type = "mixed"
    causal.sensor_snr_db = 6.0
    configs.append(causal)

    for difficulty, seed in [("easy", 201), ("medium", 202), ("hard", 203), ("extreme", 204)]:
        sampler = G1DifficultyScenarioSampler(difficulty=difficulty, random_seed=seed)
        config = sampler.sample(
            sample_id=f"G1_fs20k_random_{difficulty}",
            template_id="fsaverage5",
            source_space_id="Cortex_FS_20k",
            leadfield_id="Gain_20k_FS_fixed_orientation",
        )
        config.srate = 250
        config.time_window = (-0.2, 0.8)
        configs.append(config)

    return configs


def extract_patch_timeseries(source: np.ndarray, manifest: dict[str, Any]) -> np.ndarray:
    traces = []
    for patch in manifest["patches"]:
        vertices = np.asarray(patch["vertices"], dtype=int)
        weights = np.asarray(patch["spatial_weights"], dtype=float)
        weights = weights / max(weights.sum(), 1e-12)
        traces.append(weights @ source[vertices, :])
    return np.asarray(traces, dtype=np.float32)


def waveform_label(manifest: dict[str, Any], patch_index: int) -> str:
    waveform_params = manifest.get("waveform_params", [])
    if not waveform_params:
        return f"patch {patch_index}"
    waveform_type, params = waveform_params[patch_index % len(waveform_params)]
    if waveform_type == "resting":
        return f"patch {patch_index}: resting_{params.get('band', 'unknown')}"
    if waveform_type == "erp":
        components = params.get("components", [])
        if components and components[0].get("type") == "gabor":
            return f"patch {patch_index}: gabor_erp"
    return f"patch {patch_index}: {waveform_type}"


def waveform_summary(manifest: dict[str, Any], max_items: int | None = None) -> str:
    labels = [waveform_label(manifest, idx).split(": ", 1)[-1] for idx in range(int(manifest["n_patch"]))]
    if max_items is not None and len(labels) > max_items:
        return ", ".join(labels[:max_items]) + f", +{len(labels) - max_items} more"
    return ", ".join(labels)


def source_waveform_psd(patch_ts: np.ndarray, srate: float) -> tuple[np.ndarray, np.ndarray]:
    freqs = np.fft.rfftfreq(patch_ts.shape[1], d=1.0 / srate)
    psd = np.abs(np.fft.rfft(patch_ts, axis=1)) ** 2
    return freqs, psd


def plot_sample_summary(
    output_path: Path,
    vertices: np.ndarray,
    sensor: np.ndarray,
    clean: np.ndarray,
    noise: np.ndarray,
    source: np.ndarray,
    manifest: dict[str, Any],
) -> None:
    time = np.linspace(manifest["time_window"][0], manifest["time_window"][1], sensor.shape[1], endpoint=False)
    patch_ts = extract_patch_timeseries(source, manifest)
    active_vertices = np.asarray(manifest["active_vertices"], dtype=int)
    rng = np.random.default_rng(0)
    bg_idx = rng.choice(vertices.shape[0], size=min(4000, vertices.shape[0]), replace=False)

    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)

    ax = fig.add_subplot(gs[0, 0])
    for idx, trace in enumerate(patch_ts[:6]):
        ax.plot(time, trace + idx * 1.2, lw=1.5, label=waveform_label(manifest, idx))
    ax.set_title(f"{manifest['sample_id']} source time series: {waveform_summary(manifest)}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Patch trace + offset")
    ax.legend(fontsize=7, loc="upper right")

    ax = fig.add_subplot(gs[0, 1])
    for idx in range(min(8, sensor.shape[0])):
        ax.plot(time, sensor[idx] + idx * 0.8, lw=0.8)
    ax.set_title("Noisy EEG traces")
    ax.set_xlabel("Time (s)")

    ax = fig.add_subplot(gs[0, 2])
    for idx in range(min(8, noise.shape[0])):
        ax.plot(time, noise[idx] + idx * 0.8, lw=0.8)
    ax.set_title(f"Noise EEG traces: {manifest['sensor_noise_type']}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Noise + offset")

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(time, clean[0], label="clean ch0", lw=1.0)
    ax.plot(time, noise[0], label="noise ch0", lw=1.0)
    ax.plot(time, sensor[0], label="sensor ch0", lw=1.0)
    ax.set_title("Clean/noise/sensor example")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    freqs, psd = source_waveform_psd(patch_ts, manifest["srate"])
    for idx in range(min(6, psd.shape[0])):
        ax.semilogy(freqs[1:], psd[idx, 1:] + 1e-12, lw=1.0, label=waveform_label(manifest, idx))
    ax.axvspan(8, 13, color="orange", alpha=0.2, label="alpha")
    ax.set_xlim(1, 45)
    ax.set_title("Source waveform PSD")
    ax.set_xlabel("Hz")
    ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[1, 2], projection="3d")
    ax.scatter(vertices[bg_idx, 0], vertices[bg_idx, 1], vertices[bg_idx, 2], s=0.2, c="lightgray", alpha=0.25)
    ax.scatter(vertices[active_vertices, 0], vertices[active_vertices, 1], vertices[active_vertices, 2], s=8, c="crimson", alpha=0.95)
    ax.set_title("Active cortical vertices")
    ax.set_axis_off()

    ax = fig.add_subplot(gs[2, :])
    summary = {
        "n_patch": manifest["n_patch"],
        "connectivity": manifest["connectivity_mode"],
        "sensor_snr_db": round(float(manifest["sensor_snr_db"]), 3) if manifest["sensor_snr_db"] is not None else None,
        "source_background_type": manifest["source_background_type"],
        "sensor_noise_type": manifest["sensor_noise_type"],
        "source_waveforms": waveform_summary(manifest),
        "n_active_vertices": len(manifest["active_vertices"]),
        "sensor_shape": manifest["sensor_shape"],
        "source_shape": manifest["source_shape"],
    }
    ax.axis("off")
    ax.text(0.01, 0.95, json.dumps(summary, indent=2), va="top", ha="left", family="monospace", fontsize=10)

    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_overview(output_path: Path, sample_records: list[dict[str, Any]]) -> None:
    n_samples = len(sample_records)
    fig, axes = plt.subplots(n_samples, 4, figsize=(18, 2.3 * n_samples), constrained_layout=True)
    if n_samples == 1:
        axes = axes[None, :]
    for row, record in enumerate(sample_records):
        time = record["time"]
        manifest = record["manifest"]
        patch_ts = record["patch_ts"]
        sensor = record["sensor"]
        noise = record["noise"]
        axes[row, 0].plot(time, patch_ts.T, lw=0.9)
        axes[row, 0].set_title(
            f"{manifest['sample_id']} | {manifest['connectivity_mode']} | {waveform_summary(manifest, max_items=4)}",
            fontsize=8,
        )
        axes[row, 0].set_ylabel("source")
        for idx in range(min(6, sensor.shape[0])):
            axes[row, 1].plot(time, sensor[idx] + idx * 0.8, lw=0.6)
        axes[row, 1].set_title("EEG traces")
        for idx in range(min(6, noise.shape[0])):
            axes[row, 2].plot(time, noise[idx] + idx * 0.8, lw=0.6)
        axes[row, 2].set_title(f"Noise traces: {manifest['sensor_noise_type']}")
        freqs, psd = source_waveform_psd(patch_ts, manifest["srate"])
        for idx in range(min(4, psd.shape[0])):
            axes[row, 3].semilogy(freqs[1:], psd[idx, 1:] + 1e-12, lw=0.8)
        axes[row, 3].axvspan(8, 13, color="orange", alpha=0.2)
        axes[row, 3].set_xlim(1, 45)
        axes[row, 3].set_title("Source PSD")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cortex = load_fs20k_cortex(ANATOMY_DIR / "Cortex_FS_20k.mat")
    vertices = cortex["Vertices"].T if cortex["Vertices"].shape[0] == 3 else cortex["Vertices"]
    n_vertices = vertices.shape[0]
    leadfield = load_fixed_orientation_leadfield(ANATOMY_DIR / "Gain_20k_FS.mat", n_vertices)
    simulator = G1SourceSimulator.from_brainstorm_cortex(leadfield, cortex)

    sample_records = []
    manifests = []
    for config in build_demo_configs():
        sensor, clean, noise, source, manifest = simulator.generate_sensor(config)
        patch_ts = extract_patch_timeseries(source, manifest)
        manifests.append(manifest)
        np.savez_compressed(
            OUTPUT_DIR / f"{config.sample_id}.npz",
            sensor=sensor,
            clean_sensor=clean,
            sensor_noise=noise,
            patch_timeseries=patch_ts,
            active_vertices=np.asarray(manifest["active_vertices"], dtype=np.int32),
        )
        (OUTPUT_DIR / f"{config.sample_id}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        plot_sample_summary(OUTPUT_DIR / f"{config.sample_id}_summary.png", vertices, sensor, clean, noise, source, manifest)
        sample_records.append(
            {
                "manifest": manifest,
                "time": np.linspace(config.time_window[0], config.time_window[1], sensor.shape[1], endpoint=False),
                "patch_ts": patch_ts,
                "sensor": sensor,
                "noise": noise,
            }
        )

    plot_overview(OUTPUT_DIR / "G1_fs20k_overview.png", sample_records)
    (OUTPUT_DIR / "manifest_index.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(OUTPUT_DIR),
                "n_vertices": int(n_vertices),
                "leadfield_shape": list(leadfield.shape),
                "n_samples": len(sample_records),
                "overview": "G1_fs20k_overview.png",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
