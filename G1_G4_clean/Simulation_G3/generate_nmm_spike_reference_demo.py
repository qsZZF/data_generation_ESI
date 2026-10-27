from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import scipy.signal


ROOT = Path(__file__).resolve().parents[1]
NMM_SPIKES_DIR = ROOT / "nmm_spikes"
OUTPUT_DIR = ROOT / "Simulation_G3" / "generated_nmm_spike_reference"


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def load_nmm_spike_clip(nmm_dir: Path, region_id: int, clip_id: int) -> np.ndarray:
    mat_path = nmm_dir / f"a{region_id}" / f"nmm_{clip_id}.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"Could not find NMM spike clip: {mat_path}")

    mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if "data" not in mat:
        raise KeyError(f"{mat_path} does not contain variable 'data'")

    data = np.asarray(mat["data"], dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected 2-D NMM data, got shape {data.shape} from {mat_path}")

    # stESI spike .mat files are time x regions. Convert to regions x time.
    if data.shape[0] == 500 and data.shape[1] >= 900:
        return data.T
    if data.shape[1] == 500 and data.shape[0] >= 900:
        return data
    raise ValueError(f"Unexpected NMM data shape {data.shape}; expected 500 x 994 or 994 x 500")


def robust_normalize(region_source: np.ndarray) -> np.ndarray:
    centered = region_source - np.median(region_source, axis=1, keepdims=True)
    scale = np.percentile(np.abs(centered), 99.5)
    return centered / max(float(scale), 1e-12)


def active_spike_metrics(region_source: np.ndarray, region_id: int, srate_hz: float) -> dict[str, Any]:
    trace = np.asarray(region_source[region_id], dtype=float)
    peak_idx = int(np.nanargmax(trace))
    trough_idx = int(np.nanargmin(trace))
    peak_time_ms = 1000.0 * peak_idx / float(srate_hz)
    trough_time_ms = 1000.0 * trough_idx / float(srate_hz)
    prominence = scipy.signal.peak_prominences(trace, [peak_idx])[0][0]
    return {
        "active_region": int(region_id),
        "peak_idx": peak_idx,
        "peak_time_ms": peak_time_ms,
        "peak_value": float(trace[peak_idx]),
        "trough_idx": trough_idx,
        "trough_time_ms": trough_time_ms,
        "trough_value": float(trace[trough_idx]),
        "peak_prominence": float(prominence),
        "peak_is_boundary": bool(peak_idx < 25 or peak_idx > trace.size - 25),
    }


def plot_reference_sample(
    output_path: Path,
    sample_id: str,
    region_source: np.ndarray,
    time_ms: np.ndarray,
    active_region: int,
    metrics: dict[str, Any],
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), constrained_layout=True)
    fig.suptitle(f"{sample_id} | NMM spike reference", fontsize=13)

    active_trace = region_source[active_region]
    peak_idx = int(metrics["peak_idx"])

    axes[0].plot(time_ms, active_trace, color="tab:red", linewidth=1.2)
    axes[0].axvline(time_ms[peak_idx], color="k", linestyle="--", linewidth=0.8)
    axes[0].set_title(f"Active region {active_region}")
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("Raw NMM amplitude")

    centered_time = time_ms - time_ms[peak_idx]
    axes[1].plot(centered_time, active_trace, color="tab:purple", linewidth=1.2)
    axes[1].axvline(0.0, color="k", linestyle="--", linewidth=0.8)
    axes[1].set_title("Peak-centered active-region spike")
    axes[1].set_xlabel("Time from peak (ms)")
    axes[1].set_ylabel("Raw NMM amplitude")

    preview_count = min(12, region_source.shape[0])
    preview_regions = np.unique(np.r_[active_region, np.linspace(0, region_source.shape[0] - 1, preview_count, dtype=int)])
    offset = max(float(np.std(region_source[preview_regions])), 1e-6) * 1.6
    for idx, region in enumerate(preview_regions):
        axes[2].plot(time_ms, region_source[region] + idx * offset, linewidth=0.8)
    axes[2].set_title("Region trace preview")
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_ylabel("Region + offset")

    heat_regions = np.unique(np.r_[active_region, np.linspace(0, region_source.shape[0] - 1, min(80, region_source.shape[0]), dtype=int)])
    heatmap = robust_normalize(region_source[heat_regions])
    im = axes[3].imshow(
        heatmap,
        aspect="auto",
        interpolation="nearest",
        extent=[float(time_ms[0]), float(time_ms[-1]), len(heat_regions), 0],
        cmap="coolwarm",
        vmin=-2.5,
        vmax=2.5,
    )
    axes[3].set_title("Robust-normalized region heatmap")
    axes[3].set_xlabel("Time (ms)")
    axes[3].set_ylabel("Sampled regions")
    fig.colorbar(im, ax=axes[3], shrink=0.85)

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_reference_demo(region_id: int, clip_id: int, srate_hz: float, nmm_dir: Path, output_dir: Path) -> dict[str, Any]:
    region_source = load_nmm_spike_clip(nmm_dir, region_id, clip_id)
    time_ms = np.arange(region_source.shape[1], dtype=np.float32) * (1000.0 / float(srate_hz))
    metrics = active_spike_metrics(region_source, region_id, srate_hz)

    sample_id = f"G3_nmm_spike_region{region_id}_clip{clip_id}"
    sample_dir = output_dir / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    npz_name = f"{sample_id}_region_source.npz"
    manifest_name = f"{sample_id}_manifest.json"
    figure_name = f"{sample_id}_overview.png"

    np.savez_compressed(sample_dir / npz_name, region_source=region_source.astype(np.float32), time_ms=time_ms)
    plot_reference_sample(sample_dir / figure_name, sample_id, region_source, time_ms, region_id, metrics)

    manifest = {
        "generator_type": "G3_NMM_spike_reference_clip",
        "sample_id": sample_id,
        "source_dataset": str(nmm_dir),
        "source_mat": str(nmm_dir / f"a{region_id}" / f"nmm_{clip_id}.mat"),
        "active_regions": [int(region_id)],
        "clip_id": int(clip_id),
        "srate_hz": float(srate_hz),
        "time_window_ms": [float(time_ms[0]), float(time_ms[-1])],
        "region_source_shape": list(region_source.shape),
        "region_source": npz_name,
        "manifest": manifest_name,
        "overview_figure": figure_name,
        "spike_metrics": metrics,
        "notes": [
            "This file visualizes an already-detected stESI NMM spike clip.",
            "It is the correct reference shape for NMM spike demos; short TVB runs can show startup transients instead.",
        ],
    }
    (sample_dir / manifest_name).write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export and visualize a reference stESI NMM spike clip.")
    parser.add_argument("--region-id", type=int, default=10)
    parser.add_argument("--clip-id", type=int, default=1)
    parser.add_argument("--srate-hz", type=float, default=500.0)
    parser.add_argument("--nmm-dir", default=str(NMM_SPIKES_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args(argv)

    manifest = save_reference_demo(
        region_id=args.region_id,
        clip_id=args.clip_id,
        srate_hz=args.srate_hz,
        nmm_dir=Path(args.nmm_dir),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(_json_safe({
        "sample_id": manifest["sample_id"],
        "output_dir": str(Path(args.output_dir) / manifest["sample_id"]),
        "region_source_shape": manifest["region_source_shape"],
        "overview_figure": manifest["overview_figure"],
        "spike_metrics": manifest["spike_metrics"],
    }), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
