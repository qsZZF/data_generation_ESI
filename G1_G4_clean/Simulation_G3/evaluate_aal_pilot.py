"""Describe short AAL78 bank recordings without treating threshold peaks as labels."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Simulation_G2.nmm_bank import load_bank_sample, read_bank
from Simulation_G3.compare_spike_bank import (
    EVENT_MS, FS, detect_events, event_metrics, normalized_event,
)

REGIONS = [25, 30, 42, 48, 52, 72]
SAMPLE_IDS = ["aal_rest_20261016", "aal_spike_20261016", "aal_spike_20261017"]
COLORS = ["#687987", "#267b9a", "#be7e43"]


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".svg"))


def evaluate(bank, output):
    metadata = read_bank(bank)
    if metadata["n_regions"] != 78 or any(s not in metadata["samples"] for s in SAMPLE_IDS):
        raise ValueError("The complete three-sample AAL78 pilot bank is required")
    output.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 9, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "axes.spines.right": False, "axes.spines.top": False})
    rows, event_rows, summaries, plot_data = [], [], [], []
    for sample_index, sample_id in enumerate(SAMPLE_IDS):
        source, times, entry = load_bank_sample(bank, sample_id)
        if entry["native_srate_hz"] != FS or entry["source_unit"] != "mV_PSP":
            raise ValueError("Fixed event protocol requires raw PSP at 500 Hz")
        if source.shape != (78, 5000) or not np.isfinite(source).all():
            raise ValueError("Expected a finite 78 x 5000 pilot recording")
        configured = entry["active_regions"]
        if configured != ([] if sample_index == 0 else REGIONS):
            raise ValueError("Pilot configured regions differ from the fixed design")
        names = entry["manifest"]["region_names"]
        frequencies, powers = welch(source.astype(float), fs=FS, nperseg=1000, axis=-1)
        band = (frequencies >= 1) & (frequencies <= 45)
        denominator = powers[:, band].sum(axis=1)
        if np.any(denominator <= 0):
            raise ValueError("Undefined spectral fractions for zero-power signal")
        shapes, peaks_by_region, sample_rows = {}, {}, []
        for region, trace in enumerate(source):
            candidates, edge_count = detect_events(trace)
            shapes[region], peaks_by_region[region], region_events = [], [], []
            for peak in candidates:
                event = normalized_event(trace, int(peak))
                if event is None:
                    continue
                crop, shape = event
                metrics = event_metrics(crop, shape, None)
                row = dict(sample_id=sample_id, region=region, region_name=names[region],
                           selected=region in REGIONS, peak_index=int(peak),
                           peak_time_ms=float(times[peak]),
                           half_prominence_width_ms=metrics["half_prominence_width_ms"],
                           raw_peak_mV=metrics["raw_peak"], prominence_mV=metrics["prominence"])
                event_rows.append(row)
                region_events.append(row)
                shapes[region].append(shape)
                peaks_by_region[region].append(int(peak))
            widths = [r["half_prominence_width_ms"] for r in region_events]
            row = dict(sample_id=sample_id, region=region, region_name=names[region],
                       selected=region in REGIONS, configured_active=region in configured,
                       minimum_mV=float(trace.min()), maximum_mV=float(trace.max()),
                       rms_mV=float(np.sqrt(np.mean(trace.astype(float) ** 2))),
                       centered_rms_mV=float(np.std(trace.astype(float))),
                       valid_events=len(region_events), edge_excluded=edge_count,
                       nonpositive_excluded=len(candidates) - len(region_events),
                       median_half_prominence_width_ms=float(np.median(widths)) if widths else None,
                       dominant_frequency_1_45_hz=float(frequencies[band][np.argmax(powers[region, band])]))
            for name, low, high in (("delta", 1, 4), ("theta", 4, 8), ("alpha", 8, 13),
                                    ("beta", 13, 30), ("gamma", 30, 45.5)):
                frequency_mask = (frequencies >= low) & (frequencies < high) & band
                row[f"{name}_fraction_1_45"] = float(powers[region, frequency_mask].sum() / denominator[region])
            assert abs(sum(row[f"{b}_fraction_1_45"] for b in ("delta", "theta", "alpha", "beta", "gamma")) - 1) < 1e-12
            sample_rows.append(row)
        rows.extend(sample_rows)
        selected = [row for row in sample_rows if row["selected"]]
        background = [row for row in sample_rows if not row["selected"]]
        summary = dict(sample_id=sample_id, source_sha256=entry["source_sha256"], finite=True,
                       shape=list(source.shape), duration_s=source.shape[1] / FS,
                       configured_active=configured, evaluation_regions=REGIONS,
                       minimum_mV=float(source.min()), maximum_mV=float(source.max()),
                       rms_mV=float(np.sqrt(np.mean(source.astype(float) ** 2))))
        for label, subset in (("selected", selected), ("background", background)):
            summary[label] = dict(n_regions=len(subset), valid_events=sum(r["valid_events"] for r in subset),
                                 regions_with_events=sum(r["valid_events"] > 0 for r in subset),
                                 edge_excluded=sum(r["edge_excluded"] for r in subset),
                                 nonpositive_excluded=sum(r["nonpositive_excluded"] for r in subset),
                                 counts={str(r["region"]): r["valid_events"] for r in subset})
        assert summary["selected"]["valid_events"] + summary["background"]["valid_events"] == sum(len(v) for v in shapes.values())
        summaries.append(summary)
        plot_data.append((sample_id, names, shapes, powers, frequencies))
        fig, axes = plt.subplots(6, 1, figsize=(10, 8), sharex=True, sharey=True, layout="constrained")
        for ax, region in zip(axes, REGIONS):
            ax.plot(times / 1000, source[region], lw=.65, color=COLORS[sample_index])
            ax.axhline(8, color="0.65", linestyle=":", lw=.7)
            peaks = peaks_by_region[region]
            ax.scatter(times[peaks] / 1000, source[region, peaks], s=7, color="#a74032", zorder=3)
            ax.set_ylabel("PSP (mV)")
            ax.set_title(f"{region}: {names[region]}", loc="left", fontsize=8, pad=3)
            ax.set_title(f"n = {len(peaks)}", loc="right", fontsize=8, pad=3)
            ax.set_ylim(-4, 14)
        axes[-1].set_xlabel("Time after discarded warm-up (s)")
        fig.suptitle(f"{sample_id}: all retained samples\nSix fixed evaluation regions; dots = valid threshold events")
        save_figure(fig, output / f"{sample_id}_traces")
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    x = np.arange(3)
    for offset, group, color in ((-.18, "selected", "#267b9a"), (.18, "background", "#a8adb2")):
        values = [s[group]["valid_events"] for s in summaries]
        bars = axes[0].bar(x + offset, values, .36, color=color, label=f"{group} ({6 if group == 'selected' else 72} regions)")
        axes[0].bar_label(bars, padding=3, fontsize=8)
    axes[0].set_xticks(x, ["Rest\nseed 20261016", "Spike\nseed 20261016", "Spike\nseed 20261017"])
    axes[0].set_ylabel("Valid events / 10 s (sum across regions)")
    axes[0].margins(y=.25)
    axes[0].legend(fontsize=8)
    axes[0].set_title("a  Fixed threshold event counts", loc="left")
    for index, (sample_id, _, _, powers, frequencies) in enumerate(plot_data):
        mask = (frequencies >= 1) & (frequencies <= 45)
        axes[1].semilogy(frequencies[mask], powers[:, mask].mean(axis=0), color=COLORS[index], label=sample_id.replace("aal_", ""))
    axes[1].set(xlabel="Frequency (Hz)", ylabel="Mean regional PSD (mV²/Hz)", xlim=(1, 45))
    axes[1].set_title("b  Welch PSD: all 78 regions", loc="left")
    axes[1].legend(fontsize=8)
    save_figure(fig, output / "aal78_summary")
    plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(10, 7), sharex=True, sharey=True, layout="constrained")
    for ax, region in zip(axes.flat, REGIONS):
        for index, (sample_id, names, shapes, _, _) in enumerate(plot_data):
            events = shapes[region]
            label = f"{sample_id.replace('aal_', '')} (n={len(events)})"
            if events:
                q25, median, q75 = np.quantile(events, [.25, .5, .75], axis=0)
                ax.fill_between(EVENT_MS, q25, q75, color=COLORS[index], alpha=.16, lw=0)
                ax.plot(EVENT_MS, median, color=COLORS[index], label=label)
            else:
                ax.plot([], [], color=COLORS[index], label=label)
        ax.axvline(0, color="0.8", lw=.7)
        ax.set_title(f"{region}: {names[region]}", loc="left", fontsize=9)
        ax.set(xlabel="Time from peak (ms)", ylabel="Baseline-subtracted / peak amplitude")
        ax.legend(fontsize=7)
    fig.suptitle("All valid events: median and interquartile range\nNo time warping; no legacy atlas correspondence assumed")
    save_figure(fig, output / "aal78_event_shapes")
    plt.close(fig)
    write_csv(output / "regional_metrics.csv", rows)
    write_csv(output / "events.csv", event_rows)
    report = dict(atlas_id=metadata["atlas_id"], event_protocol=dict(threshold_mV="strictly >8",
                  minimum_spacing_ms="strictly >200", crop_ms=[-100, 300],
                  baseline_ms=[-100, -52], amplitude_normalization="positive peak minus baseline"),
                  limitations=["Threshold events are not clinical spike labels.",
                               "Three ten-second runs do not establish population or physiological validity.",
                               "The resting control uses the six fixed regions only for comparison; no regions have active A gain.",
                               "No region-matched legacy994/AAL comparison was performed."], samples=summaries)
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=ROOT / "generated_simulation_dataset/G3_AAL78_pilot/bank")
    parser.add_argument("--output", type=Path, default=ROOT / "generated_simulation_dataset/G3_AAL78_pilot/evaluation")
    args = parser.parse_args()
    result = evaluate(args.bank, args.output)
    print(json.dumps([{key: sample[key] for key in ("sample_id", "selected", "background")} for sample in result["samples"]], indent=2))
