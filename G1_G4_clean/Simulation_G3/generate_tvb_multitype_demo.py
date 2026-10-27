from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    root_dir = Path(__file__).resolve().parents[1]
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

try:
    from .g3_tvb_core import ROOT, TVBSignalConfig, TVBSourceGenerator
except ImportError:
    from g3_tvb_core import ROOT, TVBSignalConfig, TVBSourceGenerator


DEFAULT_SIGNAL_TYPES = ["resting_jr", "alpha_jr", "oscillator", "wilson_cowan"]


def build_demo_configs(signal_types: list[str], connectivity_path: str | None, short: bool) -> list[TVBSignalConfig]:
    simulation_length_ms = 500.0 if short else 1000.0
    warmup_ms = 1000.0 if short else 2000.0
    configs = []
    for idx, signal_type in enumerate(signal_types):
        coupling_strength = 1.0 if signal_type.endswith("_jr") else 0.003
        jr_noise_scale = 1.0
        if signal_type == "seizure_epileptor":
            coupling_strength = 0.0005
        elif signal_type == "alpha_jr":
            coupling_strength = 0.8
            jr_noise_scale = 0.8
        configs.append(TVBSignalConfig(
            sample_id=f"G3_tvb_{signal_type}",
            signal_type=signal_type,
            connectivity_path=connectivity_path,
            active_regions=[10],
            simulation_length_ms=simulation_length_ms,
            warmup_ms=warmup_ms,
            dt_ms=0.5,
            output_srate_hz=500.0,
            random_seed=100 + idx,
            discard_ms=0.0,
            coupling_strength=coupling_strength,
            jr_noise_scale=jr_noise_scale,
        ))
    return configs


def extract_event_clip(
    region_source: np.ndarray,
    time_ms: np.ndarray,
    active_region: int,
    pre_ms: float = 200.0,
    post_ms: float = 700.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    x = np.asarray(region_source[active_region], dtype=float)
    centered = x - np.median(x)
    peak_idx = int(np.argmax(np.abs(centered)))
    dt_ms = float(np.median(np.diff(time_ms))) if time_ms.size > 1 else 1.0
    pre_n = int(round(pre_ms / max(dt_ms, 1e-12)))
    post_n = int(round(post_ms / max(dt_ms, 1e-12)))
    start = max(0, peak_idx - pre_n)
    stop = min(region_source.shape[1], peak_idx + post_n)
    clip = region_source[:, start:stop]
    clip_time = time_ms[start:stop] - time_ms[peak_idx]
    return clip, clip_time, peak_idx


def plot_tvb_sample(sample_dir: Path, manifest: dict) -> Path:
    import matplotlib.pyplot as plt

    npz_path = sample_dir / manifest["region_source"]
    data = np.load(npz_path)
    region_source = data["region_source"]
    time_ms = data["time_ms"]
    active_regions = [int(v) for v in manifest.get("active_regions", [])]
    active_region = active_regions[0] if active_regions else 0
    active_region = max(0, min(active_region, region_source.shape[0] - 1))
    event_clip, event_time_ms, peak_idx = extract_event_clip(region_source, time_ms, active_region)

    overview_path = sample_dir / f"{manifest['sample_id']}_overview.png"
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), constrained_layout=True)
    fig.suptitle(f"{manifest['sample_id']} | {manifest['signal_type']}", fontsize=13)

    axes[0].plot(time_ms, region_source[active_region], color="tab:red", linewidth=1.2)
    axes[0].axvline(time_ms[peak_idx], color="k", linestyle="--", linewidth=0.8)
    axes[0].set_title(f"Active region {active_region}")
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("Normalized amplitude")

    preview_count = min(12, region_source.shape[0])
    preview_regions = np.linspace(0, region_source.shape[0] - 1, preview_count, dtype=int)
    offset = max(float(np.std(region_source[preview_regions])), 1e-6) * 2.0
    for idx, region in enumerate(preview_regions):
        axes[1].plot(time_ms, region_source[region] + idx * offset, linewidth=0.8)
    axes[1].set_title("Region trace preview")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Region + offset")

    heat_regions = np.unique(np.r_[active_region, np.linspace(0, region_source.shape[0] - 1, min(80, region_source.shape[0]), dtype=int)])
    im = axes[2].imshow(
        region_source[heat_regions],
        aspect="auto",
        interpolation="nearest",
        extent=[float(time_ms[0]), float(time_ms[-1]), len(heat_regions), 0],
        cmap="coolwarm",
    )
    axes[2].set_title("Region activity heatmap")
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_ylabel("Sampled regions")
    fig.colorbar(im, ax=axes[2], shrink=0.85)

    axes[3].plot(event_time_ms, event_clip[active_region], color="tab:purple", linewidth=1.2)
    axes[3].axvline(0.0, color="k", linestyle="--", linewidth=0.8)
    axes[3].set_title("Peak-centered event clip from active region")
    axes[3].set_xlabel("Time from peak (ms)")
    axes[3].set_ylabel("Normalized amplitude")

    fig.savefig(overview_path, dpi=150)
    plt.close(fig)
    return overview_path


def plot_manifest_summary(output_dir: Path, manifests: list[dict]) -> Path:
    import matplotlib.pyplot as plt

    summary_path = output_dir / "tvb_waveform_summary.png"
    fig, axes = plt.subplots(len(manifests), 1, figsize=(12, max(2.2 * len(manifests), 4)), sharex=False, constrained_layout=True)
    if len(manifests) == 1:
        axes = [axes]
    for ax, manifest in zip(axes, manifests):
        sample_dir = output_dir / manifest["sample_id"]
        data = np.load(sample_dir / manifest["region_source"])
        region_source = data["region_source"]
        time_ms = data["time_ms"]
        active_region = int(manifest.get("active_regions", [0])[0])
        active_region = max(0, min(active_region, region_source.shape[0] - 1))
        ax.plot(time_ms, region_source[active_region], linewidth=1.0)
        ax.set_title(f"{manifest['signal_type']} | active region {active_region}", fontsize=10)
        ax.set_ylabel("amp")
    axes[-1].set_xlabel("Time (ms)")
    fig.savefig(summary_path, dpi=150)
    plt.close(fig)
    return summary_path


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generate a small multi-type TVB source activity demo.")
    parser.add_argument(
        "--signal-types",
        default=",".join(DEFAULT_SIGNAL_TYPES),
        help="Comma-separated TVB source types, e.g. resting_jr,spike_jr,alpha_jr.",
    )
    parser.add_argument("--connectivity-path", default=None)
    parser.add_argument("--output-dir", default=str(ROOT / "Simulation_G3" / "generated_tvb_demo"))
    parser.add_argument("--short", action="store_true", help="Use shorter 500 ms output windows for quick smoke tests.")
    parser.add_argument("--plot", action="store_true", help="Save an overview PNG for each generated source sample.")
    args = parser.parse_args(argv)

    signal_types = [item.strip() for item in args.signal_types.split(",") if item.strip()]
    generator = TVBSourceGenerator(args.connectivity_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifests = []
    for config in build_demo_configs(signal_types, args.connectivity_path, args.short):
        sample_dir = output_dir / config.sample_id
        manifest = generator.save_sample(sample_dir, config)
        if args.plot:
            manifest["overview_figure"] = plot_tvb_sample(sample_dir, manifest).name
            (sample_dir / f"{config.sample_id}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifests.append(manifest)

    index_path = output_dir / "manifest_index.json"
    index_path.write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    summary_figure = None
    if args.plot:
        summary_figure = plot_manifest_summary(output_dir, manifests)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "n_samples": len(manifests),
                "signal_types": [manifest["signal_type"] for manifest in manifests],
                "shapes": {manifest["sample_id"]: manifest["region_source_shape"] for manifest in manifests},
                "summary_figure": str(summary_figure) if summary_figure else None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
