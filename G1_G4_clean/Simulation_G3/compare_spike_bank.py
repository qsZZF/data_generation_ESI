"""Fixed-protocol morphology comparison: coupled G3 runs versus legacy G2 clips.

Legacy focus intervals were amplitude-scaled during export. Their raw amplitudes
and selected-clip frequencies therefore cannot establish amplitude/rate parity.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat
from scipy.signal import find_peaks, peak_prominences, peak_widths, periodogram

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Simulation_G2.nmm_bank import load_bank_sample, read_bank

FS = 500
PRE, POST = 50, 150
EVENT_MS = np.arange(-PRE, POST + 1) * 1000 / FS
METRICS = ("correlation", "half_prominence_width_ms", "trough_latency_ms",
           "trough_ratio", "normalized_psd_l1", "raw_peak", "prominence")


def normalized_event(trace, peak):
    """Return the fixed crop and shape; reject edge or nonpositive events."""
    trace = np.asarray(trace, dtype=float)
    if peak < PRE or peak + POST >= len(trace):
        return None
    crop = trace[peak - PRE:peak + POST + 1].copy()
    if not np.isfinite(crop).all():
        raise ValueError("Nonfinite event samples")
    baseline = float(crop[:25].mean())
    amplitude = float(crop[PRE] - baseline)
    if amplitude <= 0:
        return None
    return crop, (crop - baseline) / amplitude


def detect_events(trace):
    """Positive local peaks strictly above 8 mV, strictly >200 ms apart."""
    trace = np.asarray(trace, dtype=float)
    if not np.isfinite(trace).all():
        raise ValueError("Nonfinite source trace")
    peaks, _ = find_peaks(trace, height=np.nextafter(8., np.inf), distance=101)
    valid = peaks[(peaks >= PRE) & (peaks + POST < len(trace))]
    return valid, int(len(peaks) - len(valid))


def normalized_psd(shape):
    _, power = periodogram(shape, fs=FS, window="hann", detrend="constant")
    total = power.sum()
    return power / total if total > 0 else np.zeros_like(power)


def event_metrics(crop, shape, template):
    prominence = peak_prominences(crop, [PRE])
    width = peak_widths(crop, [PRE], rel_height=.5, prominence_data=prominence)[0][0]
    trough = PRE + 1 + int(np.argmin(shape[PRE + 1:]))
    correlation = None
    psd_distance = None
    if template is not None and np.std(template) > 0 and np.std(shape) > 0:
        correlation = float(np.corrcoef(shape, template)[0, 1])
        psd_distance = float(np.abs(normalized_psd(shape) - normalized_psd(template)).sum())
    return dict(correlation=correlation, half_prominence_width_ms=float(width * 2),
                trough_latency_ms=float((trough - PRE) * 2), trough_ratio=float(shape[trough]),
                normalized_psd_l1=psd_distance, raw_peak=float(crop[PRE]),
                prominence=float(prominence[0][0]))


def summarize(rows):
    """Use null rather than NaN for an empty or unavailable measurement."""
    result = {"n_events": len(rows)}
    for key in METRICS:
        values = [row[key] for row in rows if row.get(key) is not None]
        result[key] = ({"median": float(np.median(values)),
                        "q25": float(np.quantile(values, .25)),
                        "q75": float(np.quantile(values, .75))} if values else None)
    return result


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_figure(fig, path):
    fig.savefig(f"{path}.png", dpi=180)
    fig.savefig(f"{path}.pdf")
    fig.savefig(f"{path}.svg")


def reference_paths(folder):
    """Select by filename alone; retain a development/heldout split if scarce."""
    paths = sorted(folder.glob("nmm_*.mat"), key=lambda p: int(p.stem.split("_")[-1]))
    if len(paths) < 21:
        raise ValueError(f"{folder.name} has {len(paths)} reference clips; at least 21 required")
    return paths[:40], len(paths)


def _load_reference_trace(path, region):
    data = np.asarray(loadmat(path, variable_names=["data"])["data"])
    if data.ndim != 2:
        raise ValueError(f"Expected 2D legacy data: {path}")
    if data.shape[0] == 500 and data.shape[1] >= 900:
        trace = data[:, region]
    elif data.shape[1] == 500 and data.shape[0] >= 900:
        trace = data[region]
    else:
        raise ValueError(f"Unexpected legacy data shape {data.shape}: {path}")
    trace = np.asarray(trace, dtype="<f8")
    if not np.isfinite(trace).all():
        raise ValueError(f"Nonfinite legacy trace: {path}")
    # Canonicalize signed zero so numerically identical traces share identity.
    trace[trace == 0] = 0
    return trace


def load_references(regions):
    references, provenance, all_rows = {}, [], []
    for region in regions:
        folder = ROOT / "nmm_spikes" / f"a{region}"
        paths, available = reference_paths(folder)
        groups = {"development": [], "heldout": []}
        excluded = {"development": 0, "heldout": 0}
        exclusions = {split: dict(duplicate=0, edge=0, nonpositive=0) for split in groups}
        seen_waveforms = {}
        for index, path in enumerate(paths):
            split = "development" if index < 20 else "heldout"
            trace = _load_reference_trace(path, region)
            waveform_sha256 = hashlib.sha256(trace.tobytes()).hexdigest()
            relative_path = str(path.relative_to(ROOT))
            duplicate_of = seen_waveforms.get(waveform_sha256)
            seen_waveforms.setdefault(waveform_sha256, relative_path)
            peak = int(np.argmax(trace))
            event = normalized_event(trace, peak)
            reason = ("duplicate" if duplicate_of is not None else
                      "edge" if peak < PRE or peak + POST >= len(trace) else
                      "nonpositive" if event is None else None)
            provenance.append(dict(region=region, split=split, path=relative_path,
                                   sha256=_sha256(path), waveform_sha256=waveform_sha256,
                                   duplicate_of=duplicate_of, duplicate_exclusion=duplicate_of is not None,
                                   exclusion_reason=reason, peak_index=peak, included=reason is None))
            if reason is not None:
                excluded[split] += 1
                exclusions[split][reason] += 1
            else:
                groups[split].append((path.name, peak, *event))
        development = np.array([event[3] for event in groups["development"]])
        template = np.median(development, axis=0) if len(development) else None
        heldout_rows = []
        for split, events in groups.items():
            for name, peak, crop, shape in events:
                row = dict(sample_id="legacy", region=region, split=split, clip=name,
                           peak_index=peak, peak_time_ms=peak * 2,
                           **event_metrics(crop, shape, template))
                all_rows.append(row)
                if split == "heldout":
                    heldout_rows.append(row)
        references[region] = dict(template=template, groups=groups, excluded=excluded,
                                  heldout_rows=heldout_rows,
                                  exclusion_details=exclusions,
                                  availability=dict(total_files=available, selected_files=len(paths),
                                                    development=20, heldout=len(paths) - 20))
    return references, provenance, all_rows


def _plot_trial(output, sample_id, source, times, regions, references, events, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 8, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "axes.spines.right": False, "axes.spines.top": False})
    fig, axes = plt.subplots(len(regions), 1, figsize=(10, 8), sharex=True, layout="constrained")
    for ax, region in zip(np.atleast_1d(axes), regions):
        ax.plot(times / 1000, source[region], color="#346c96", lw=.6)
        ax.axhline(8, color="0.6", linestyle=":", lw=.7)
        peaks = [event[0] for event in events[region]]
        if peaks:
            ax.scatter(times[peaks] / 1000, source[region, peaks], color="#be793b", s=8, zorder=3)
        ax.set_ylabel(f"R{region}\nPSP (mV)")
        ax.text(.99, .82, f"n = {len(peaks)}", ha="right", transform=ax.transAxes)
    axes[-1].set_xlabel("Time after discarded warm-up (s)")
    fig.suptitle(f"{sample_id}: simultaneous coupled foci; all retained samples")
    _save_figure(fig, output / f"{sample_id}_continuous")
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(8, 8), sharex=True, sharey=True, layout="constrained")
    for ax, region in zip(axes.flat, regions):
        reference = references[region]
        heldout = [event[3] for event in reference["groups"]["heldout"]]
        generated = [event[2] for event in events[region]]
        for shapes, color, label in ((heldout, "#707070", "Legacy heldout"),
                                      (generated, "#346c96", "G3 all events")):
            if shapes:
                q25, median, q75 = np.quantile(shapes, [.25, .5, .75], axis=0)
                ax.fill_between(EVENT_MS, q25, q75, color=color, alpha=.18, linewidth=0)
                ax.plot(EVENT_MS, median, color=color, lw=1.3, label=f"{label} (n={len(shapes)})")
        if reference["template"] is not None:
            ax.plot(EVENT_MS, reference["template"], "--", color="#be793b", lw=.9, label="Legacy dev. median")
        ax.axvline(0, color="0.8", lw=.6)
        score = summaries[str(region)]["generated"]["correlation"]
        label = f"r median = {score['median']:.3f}" if score else "No valid generated events"
        ax.set_title(f"Region {region}: {label}", loc="left")
        ax.legend(loc="lower right", fontsize=6)
        ax.set_xlabel("Time from positive peak (ms)")
        ax.set_ylabel("Baseline-subtracted / peak amplitude")
    fig.suptitle(f"{sample_id}: fixed-window morphology\nMedian and IQR; no time warping or polarity reversal")
    _save_figure(fig, output / f"{sample_id}_comparison")
    plt.close(fig)


def compare_bank(bank, output, selection):
    samples = read_bank(bank)["samples"]
    if not samples:
        raise ValueError("Bank has no samples")
    output.mkdir(parents=True, exist_ok=True)
    regions = json.loads(selection.read_text(encoding="utf-8"))["active_regions"]
    if len(regions) != 6 or len(set(regions)) != 6:
        raise ValueError("This fixed comparison requires the six preregistered distinct regions")
    contract = dict(
        conclusion="Assess shape resemblance and event coverage; no presumed equivalence.",
        archetype="quantitative grid", backend="python",
        evidence="Full continuous selected traces, all-event median/IQR, fixed heldout reference split.",
        exports="10x8 inch traces, 8x8 inch shape grid; 180 dpi screen PNG and editable-text vector PDF/SVG.",
        event_rule="Per-region local maxima strictly >8 mV and >200 ms apart; no prominence filter.",
        crop="201 samples at 500 Hz, -100 to +300 ms; edge events excluded and counted.",
        normalization="Subtract mean first25 samples (-100 to -52 ms); divide positive peak, no polarity flip.",
        legacy_rule="Numerically first up to40 files per region: first20 development, remaining up to20 heldout; minimum21 files; maximum peak per clip. Full focus-trace canonical float64 SHA256 duplicates are excluded per region with first-appearance/development priority, without refill. Availability shortfall is not a morphology exclusion; duplicate/edge/nonpositive exclusions are separate.",
        limitations=["Legacy amplitudes were partly rescaled to target 15 dB, so raw amplitude parity is not tested.",
                     "Legacy selected clips cannot identify continuous event rates.",
                     "Events in one trajectory are dependent; no p-values or biological replication claims.",
                     "Removing exact waveform duplicates does not prove independent stochastic draws: legacy trajectory provenance is unknown.",
                     "Per-region peak spacing differs from original cross-region global detector.",
                     "Single-focus legacy clips do not establish jointly coupled multi-focus timing equivalence.",
                     "Post-peak minimum at window boundary is a censored trough latency, not necessarily a slow wave."])
    (output / "figure_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    references, provenance, rows = load_references(regions)
    (output / "legacy_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    report = dict(regions=regions, protocol=contract, trials={})
    summary_rows = []
    for sample_id in sorted(samples):
        source, times, entry = load_bank_sample(bank, sample_id)
        if entry["source_unit"] != "mV_PSP" or not np.allclose(np.diff(times), 2., rtol=0, atol=1e-6):
            raise ValueError("Comparison requires unnormalized PSP at 500 Hz")
        if set(entry["active_regions"]) != set(regions):
            raise ValueError(f"{sample_id}: active regions differ from fixed selection")
        trial = dict(source_sha256=entry["source_sha256"], manifest=entry["manifest"], regions={},
                     background_event_counts={}, background_edge_exclusions={})
        events = {}
        for region in range(source.shape[0]):
            peaks, excluded = detect_events(source[region])
            if region not in regions:
                trial["background_event_counts"][str(region)] = len(peaks)
                trial["background_edge_exclusions"][str(region)] = excluded
                continue
            events[region], region_rows = [], []
            rejected = 0
            for peak in peaks:
                event = normalized_event(source[region], peak)
                if event is None:
                    rejected += 1
                    continue
                crop, shape = event
                events[region].append((int(peak), crop, shape))
                row = dict(sample_id=sample_id, region=region, split="generated", clip="",
                           peak_index=int(peak), peak_time_ms=float(times[peak]),
                           **event_metrics(crop, shape, references[region]["template"]))
                region_rows.append(row)
                rows.append(row)
            ref = references[region]
            trial["regions"][str(region)] = dict(generated=summarize(region_rows),
                edge_exclusions=excluded, nonpositive_baseline_amplitude_exclusions=rejected,
                legacy_heldout=summarize(ref["heldout_rows"]), legacy_exclusions=ref["excluded"],
                legacy_exclusion_details=ref["exclusion_details"],
                legacy_availability=ref["availability"],
                development_count=len(ref["groups"]["development"]))
            for split, split_rows in (("generated", region_rows), ("legacy_heldout", ref["heldout_rows"])):
                summary = summarize(split_rows)
                summary_rows.append(dict(sample_id=sample_id, region=region, split=split,
                    n_events=summary["n_events"],
                    excluded=(excluded + rejected if split == "generated" else ref["excluded"]["heldout"]),
                    **{f"{metric}_{stat}": summary[metric][stat] if summary[metric] else None
                       for metric in METRICS for stat in ("median", "q25", "q75")}))
        trial["selected_event_total"] = sum(len(value) for value in events.values())
        trial["background_event_total"] = sum(trial["background_event_counts"].values())
        trial["active_regions_with_events"] = sum(bool(value) for value in events.values())
        report["trials"][sample_id] = trial
        crops = {f"region_{region}": np.array([event[2] for event in events[region]]).reshape(-1, len(EVENT_MS))
                 for region in regions}
        np.savez_compressed(output / f"{sample_id}_normalized_events.npz", event_time_ms=EVENT_MS, **crops)
        _plot_trial(output, sample_id, source, times, regions, references, events, trial["regions"])
    fields = ["sample_id", "region", "split", "clip", "peak_index", "peak_time_ms", *METRICS]
    with (output / "event_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (output / "region_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    (output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path,
                        default=ROOT / "generated_simulation_dataset/G3_spike_comparison/region_selection.json")
    args = parser.parse_args()
    report = compare_bank(args.bank, args.output_dir, args.selection)
    print(json.dumps({key: {name: value[name] for name in
                           ("selected_event_total", "background_event_total", "active_regions_with_events")}
                      for key, value in report["trials"].items()}, indent=2))


if __name__ == "__main__":
    main()
