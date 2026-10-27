from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

try:
    from .g2_simulation_core import (
        G2SimulationConfig,
        G2SourceSimulator,
        load_brainstorm_cortex,
        load_brainstorm_leadfield,
        load_projection_matrix,
    )
except ImportError:
    from g2_simulation_core import (
        G2SimulationConfig,
        G2SourceSimulator,
        load_brainstorm_cortex,
        load_brainstorm_leadfield,
        load_projection_matrix,
    )


ANATOMY_DIR = ROOT / "Anatomy"
NMM_DIR = ROOT / "nmm_spikes"
OUTPUT_DIR = ROOT / "Simulation_G2" / "generated_fs20k_demo"


def build_simulator() -> G2SourceSimulator:
    cortex = load_brainstorm_cortex(ANATOMY_DIR / "Cortex_FS_20k.mat")
    n_vertices = np.asarray(cortex["Vertices"]).shape[-1]
    leadfield = load_brainstorm_leadfield(ANATOMY_DIR / "Gain_20k_FS.mat", n_vertices=n_vertices)
    projection_matrix = load_projection_matrix(ANATOMY_DIR / "FS_994nmm_to_20k.mat", n_vertices=n_vertices)
    return G2SourceSimulator(leadfield, cortex, projection_matrix, NMM_DIR)


def make_config(
    sample_id: str,
    active_region_ids: list[int],
    sensor_noise_type: str,
    sensor_snr_db: float,
    weight_decay_mm: float | None,
    weight_decay_hops: int,
    random_seed: int,
) -> G2SimulationConfig:
    return G2SimulationConfig(
        sample_id=sample_id,
        template_id="fsaverage5",
        source_space_id="Cortex_FS_20k",
        leadfield_id="Gain_20k_FS",
        projection_id="FS_994nmm_to_20k",
        modality="EEG",
        srate=500.0,
        time_window=(0.0, 1.0),
        n_active_regions=len(active_region_ids),
        active_region_ids=active_region_ids,
        weight_decay_mm=weight_decay_mm,
        weight_decay_hops=weight_decay_hops,
        normalize_region_signals=True,
        normalize_source=True,
        sensor_noise_type=sensor_noise_type,
        sensor_snr_db=sensor_snr_db,
        sensor_snr_mode="robust_global_rms",
        sensor_snr_clip_quantile=0.95,
        average_reference=True,
        random_seed=random_seed,
    )


def source_waveform_psd(source_traces: np.ndarray, srate: float) -> tuple[np.ndarray, np.ndarray]:
    if source_traces.ndim == 1:
        source_traces = source_traces[None, :]
    freqs, psd = scipy.signal.welch(source_traces, fs=srate, axis=1, nperseg=min(256, source_traces.shape[1]))
    return freqs, psd


def plot_sample_summary(
    output_path: Path,
    sample_name: str,
    config: G2SimulationConfig,
    manifest: dict[str, Any],
    sensor: np.ndarray,
    clean: np.ndarray,
    noise: np.ndarray,
    source: np.ndarray,
    region_source: np.ndarray,
    region_ids: list[int],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    fig.suptitle(
        f"{sample_name} | regions={region_ids} | noise={config.sensor_noise_type} | snr={config.sensor_snr_db:.1f} dB",
        fontsize=13,
    )

    time = np.arange(source.shape[1]) / config.srate + config.time_window[0]
    sensor_idx = np.arange(min(8, sensor.shape[0]))

    # Source traces
    ax = axes[0, 0]
    source_lines = region_source[region_ids]
    offset = max(np.std(source_lines, axis=1).max(), 1e-8) * 1.6
    for idx, region_id in enumerate(region_ids):
        y = source_lines[idx] + idx * offset
        ax.plot(time, y, linewidth=1.0, label=f"region {region_id}")
    ax.set_title("Region source time series")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude + offset")
    ax.legend(loc="upper right", fontsize=8)

    # Source PSD
    ax = axes[0, 1]
    freqs, psd = source_waveform_psd(source_lines, config.srate)
    for idx, region_id in enumerate(region_ids):
        ax.plot(freqs, 10 * np.log10(psd[idx] + 1e-12), label=f"region {region_id}")
    ax.set_xlim(0, 80)
    ax.set_title("Region source PSD")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.legend(loc="upper right", fontsize=8)

    # Sensor traces
    ax = axes[1, 0]
    sensor_offset = max(np.std(sensor[sensor_idx], axis=1).max(), 1e-8) * 1.8
    for idx, ch in enumerate(sensor_idx):
        ax.plot(time, sensor[ch] + idx * sensor_offset, color="k", linewidth=0.7)
    ax.set_title("Sensor EEG traces")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude + offset")

    # Noise traces
    ax = axes[1, 1]
    noise_offset = max(np.std(noise[sensor_idx], axis=1).max(), 1e-8) * 1.8
    for idx, ch in enumerate(sensor_idx):
        ax.plot(time, noise[ch] + idx * noise_offset, color="tab:red", linewidth=0.7)
    ax.set_title("Sensor noise traces")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude + offset")

    fig.text(0.01, 0.01, json.dumps(
        {
            "selected_regions": manifest.get("selected_regions", region_ids),
            "region_source_shape": manifest.get("region_source_shape"),
            "vertex_source_shape": manifest.get("vertex_source_shape"),
            "sensor_shape": manifest.get("sensor_shape"),
            "projection_shape": manifest.get("projection_shape"),
        },
        ensure_ascii=False,
    ), fontsize=8)

    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    simulator = build_simulator()
    rng = np.random.default_rng(20240608)

    active_sets = [
        simulator.sample_regions(1, rng),
        simulator.sample_regions(2, rng),
        simulator.sample_regions(3, rng),
        simulator.sample_regions(2, rng),
        simulator.sample_regions(1, rng),
        simulator.sample_regions(2, rng),
    ]

    configs = [
        make_config("G2_fs20k_single_white", active_sets[0], "white", 12.0, None, 0, 301),
        make_config("G2_fs20k_pair_pink_decay", active_sets[1], "pink", 10.0, 12.0, 1, 302),
        make_config("G2_fs20k_triple_mixed_decay", active_sets[2], "mixed", 8.0, 14.0, 2, 303),
        make_config("G2_fs20k_pair_eegsourcesim", active_sets[3], "eegsourcesim_resting", 10.0, 10.0, 1, 304),
        make_config("G2_fs20k_single_powerlaw", active_sets[4], "powerlaw", 9.0, 12.0, 1, 305),
        make_config("G2_fs20k_pair_specparam_resting", active_sets[5], "specparam_resting", 9.0, 10.0, 1, 306),
    ]

    manifests: list[dict[str, Any]] = []
    overview_path = OUTPUT_DIR / "G2_fs20k_overview.png"

    for index, config in enumerate(configs):
        sample_dir = OUTPUT_DIR / config.sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        source, region_source, source_manifest = simulator.generate_source(config)
        sensor, clean, noise, source_out, manifest = simulator._finalize_sensor_from_source(  # noqa: SLF001
            source,
            region_source,
            source_manifest,
            config,
        )

        assert np.allclose(source, source_out)

        np.savez_compressed(
            sample_dir / f"{config.sample_id}.npz",
            sensor=sensor,
            clean_sensor=clean,
            sensor_noise=noise,
            source=source,
            region_source=region_source,
        )

        manifest.update(
            {
                "sensor_data": f"{config.sample_id}.npz",
                "sample_dir": sample_dir.name,
                "files": {
                    "npz": f"{config.sample_id}.npz",
                    "manifest": f"{config.sample_id}_manifest.json",
                },
            }
        )
        manifest_path = sample_dir / f"{config.sample_id}_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        plot_sample_summary(
            sample_dir / f"{config.sample_id}_summary.png",
            config.sample_id,
            config,
            manifest,
            sensor,
            clean,
            noise,
            source,
            region_source,
            manifest["selected_regions"],
        )
        manifests.append(manifest)

        if index == 0:
            plot_sample_summary(
                overview_path,
                "G2_fs20k_overview",
                config,
                manifest,
                sensor,
                clean,
                noise,
                source,
                region_source,
                manifest["selected_regions"],
            )

    (OUTPUT_DIR / "manifest_index.json").write_text(
        json.dumps([manifest["sample_id"] for manifest in manifests], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
