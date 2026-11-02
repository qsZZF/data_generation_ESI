"""Oracle known-diffusion observability pilot, not a resting-state model.

The experiment fixes an anatomical basis and assumes deterministic diffusion
without process noise. It tests whether later measurements constrain an initial
state that is invisible instantaneously. The same observations are supplied to
correct, stationary, and mismatched dynamical models.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resolution_benchmark.nullspace import analyze_operator, projector_from_patterns
from resolution_benchmark.recovery_metrics import score_recovery
from scripts.run_nullspace_study import covariance_factor, sensor_subsets

FS = 256
WINDOWS = (1, 4, 16, 64)
TRUE_KAPPA = .002
ASSUMPTIONS = {"correct_diffusion": TRUE_KAPPA, "stationary": 0., "mismatched_diffusion": .004}


def diffusion_design(projected_basis, eigenvalues, kappa, samples, fs=FS):
    """Stack A B exp(-kappa Lambda t), with time-major sensor rows."""
    projected_basis = np.asarray(projected_basis, dtype=float)
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    if (projected_basis.ndim != 2 or eigenvalues.ndim != 1
            or projected_basis.shape[1] != len(eigenvalues)
            or not np.all(np.isfinite(projected_basis)) or not np.all(np.isfinite(eigenvalues))
            or np.any(eigenvalues < 0) or not np.isfinite(kappa) or kappa < 0
            or not np.isfinite(fs) or fs <= 0 or isinstance(samples, bool)
            or int(samples) != samples or samples < 1):
        raise ValueError("Invalid projected basis, spectrum, diffusion, or sampling parameters")
    decay = np.exp(-kappa * np.arange(samples)[:, None] * eigenvalues[None, :] / fs)
    return (decay[:, None, :] * projected_basis[None, :, :]).reshape(-1, len(eigenvalues))


def solve_initial(design, data, *, ridge_relative=.01, numerical_rtol=1e-10):
    """Fit only an initial coefficient vector from time-major observations."""
    design = np.asarray(design, dtype=float)
    data = np.asarray(data, dtype=float)
    if (design.ndim != 2 or data.ndim != 2 or min(design.shape) == 0
            or design.shape[0] != data.size or not np.isfinite(design).all()
            or not np.isfinite(data).all() or ridge_relative <= 0
            or not np.isfinite(ridge_relative) or not 0 < numerical_rtol < 1):
        raise ValueError("Invalid design, channel-by-time data, or solver parameters")
    u, s, vt = np.linalg.svd(design, full_matrices=False)
    if s[0] <= 0:
        raise ValueError("The design has no observable component")
    projected_data = u.T @ data.T.reshape(-1)
    rank = int(np.sum(s > numerical_rtol * s[0]))
    ridge_gain = (s / s[0]) / ((s / s[0]) ** 2 + ridge_relative ** 2) / s[0]
    ridge = vt.T @ (ridge_gain * projected_data)
    pinv = vt[:rank].T @ (projected_data[:rank] / s[:rank])
    diagnostics = dict(observability_rank=rank, observability_nullity=design.shape[1] - rank,
                       observability_effective_rank=int(np.sum(s > 1e-3 * s[0])),
                       observability_condition=float(s[0] / s[-1]) if rank == design.shape[1] else None,
                       observability_nonzero_condition=float(s[0] / s[rank - 1]),
                       ridge_penalty=float(ridge_relative * s[0]))
    return ridge, pinv, diagnostics


def load_forward(directory, modality):
    folder = directory / ("mne_sample" if modality == "meg" else "hdeeg_template")
    path = folder / ("mne_sample_forward.npz" if modality == "meg" else "forward.npz")
    with np.load(path) as archive:
        data = dict(archive)
    with np.load(folder / "bases.npz") as archive:
        bases = dict(archive)
    mask = (np.isin(data["channel_types"], ["mag", "grad"]) if modality == "meg"
            else data["channel_types"] == "eeg")
    channels = np.flatnonzero(mask)
    selected = sensor_subsets(data["channel_names"][channels],
                              data["channel_positions_head_m"][channels], modality, 41)[-1]
    patterns = data["sensor_projector_basis"][channels]
    full_projection = projector_from_patterns(patterns)
    covariance = data["noise_cov"][np.ix_(channels, channels)]
    covariance = full_projection @ covariance @ full_projection.T
    projection = projector_from_patterns(patterns[selected])
    modes = np.argsort(bases["eigenvalues"], kind="stable")[:64]
    basis = bases["raw_modes"][:, modes]
    basis = basis / np.linalg.norm(basis, axis=0)
    if basis.shape[0] != data["leadfield_fixed"].shape[1]:
        raise ValueError("Anatomical basis and forward source dimensions differ")
    for name in ("source_vertex_ids", "source_hemi"):
        np.testing.assert_array_equal(bases[name], data[name], err_msg=f"Basis/forward mismatch: {name}")
    return dict(gain=data["leadfield_fixed"][channels][selected], covariance=covariance,
                projection=projection, selected=selected, basis=basis,
                eigenvalues=bases["eigenvalues"][modes],
                hashes={"forward": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "bases": hashlib.sha256((folder / "bases.npz").read_bytes()).hexdigest()})


def run_case(case, modality, seeds):
    records = []
    factor = covariance_factor(case["covariance"])
    selected, p, basis = case["selected"], case["projection"], case["basis"]
    true_covariance = p @ case["covariance"][np.ix_(selected, selected)] @ p.T
    for seed in range(seeds):
        rng = np.random.default_rng(70000 + seed)
        baseline = (factor @ rng.normal(size=(factor.shape[1], 2 * FS)))[selected]
        noise = p @ (factor @ rng.normal(size=(factor.shape[1], max(WINDOWS))))[selected]
        covariance = p @ np.cov(baseline) @ p.T
        analysis = analyze_operator(p @ case["gain"], covariance)
        w, A = analysis["whitener"], analysis["operator"]
        noise = w @ noise
        projected_basis = A @ basis
        z0 = rng.normal(size=basis.shape[1])
        expected_noise_power = np.trace(w @ true_covariance @ w.T) / w.shape[0]
        initial_signal_power = np.mean((projected_basis @ z0) ** 2)
        for snr in (20., 30.):
            scale = np.sqrt(10 ** (snr / 10) * expected_noise_power / initial_signal_power)
            initial = basis @ (scale * z0)
            decay = np.exp(-TRUE_KAPPA * case["eigenvalues"][:, None]
                           * np.arange(max(WINDOWS))[None, :] / FS)
            clean = projected_basis @ ((scale * z0)[:, None] * decay)
            data = clean + noise
            observation_hash = hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()
            for name, kappa in ASSUMPTIONS.items():
                for samples in WINDOWS:
                    design = diffusion_design(projected_basis, case["eigenvalues"], kappa, samples)
                    ridge, _, diagnostics = solve_initial(design, data[:, :samples])
                    _, clean_pinv, _ = solve_initial(design, clean[:, :samples])
                    recovery = score_recovery((basis @ ridge)[:, None], initial[:, None], analysis)
                    clean_recovery = score_recovery((basis @ clean_pinv)[:, None], initial[:, None], analysis)
                    record = dict(modality=modality, seed=seed, nominal_initial_sparse_array_snr_db=snr,
                                  n_channels=len(selected), instantaneous_physical_rank=analysis["info"]["rank"],
                                  n_physical_sources=len(initial), n_modes=basis.shape[1], assumption=name,
                                  true_kappa_m2_per_s=TRUE_KAPPA, assumed_kappa_m2_per_s=kappa,
                                  samples=samples, elapsed_span_s=(samples - 1) / FS,
                                  observed_window_snr_db=float(10 * np.log10(np.sum(clean[:, :samples] ** 2)
                                                                        / np.sum(noise[:, :samples] ** 2))),
                                  observation_sha256=observation_hash,
                                  expected_whitened_noise_power=float(expected_noise_power),
                                  **diagnostics, **recovery)
                    record.update(clean_pinv_source_nrmse=clean_recovery["source_nrmse"],
                                  clean_pinv_null_recovery=clean_recovery["null_recovery"],
                                  clean_pinv_sensor_residual=float(np.linalg.norm(design @ clean_pinv
                                                                               - clean[:, :samples].T.reshape(-1))
                                                                   / np.linalg.norm(clean[:, :samples])),
                                  noisy_ridge_sensor_residual=float(np.linalg.norm(design @ ridge
                                                                                - data[:, :samples].T.reshape(-1))
                                                                    / np.linalg.norm(data[:, :samples])))
                    records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "generated_simulation_dataset/nullspace_study")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--case", choices=("meg", "hdeeg", "all"), default="all")
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("seeds must be positive")
    output = args.directory / "methods_supports"
    output.mkdir(parents=True, exist_ok=True)
    rows, hashes = [], {}
    for modality in ("meg", "hdeeg"):
        if args.case not in (modality, "all"):
            continue
        case = load_forward(args.directory, modality)
        rows.extend(run_case(case, modality, args.seeds))
        hashes[modality] = case["hashes"]
    with (output / "dynamics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    protocol = dict(rows=len(rows), seeds=args.seeds, fs_hz=FS, baseline_s=2, windows_samples=WINDOWS,
                    source_prior="64 globally eigenvalue-sorted raw LBO eigenvectors, unit Euclidean column norm",
                    initial_coefficients="independent standard Gaussian; one shared initial state per seed",
                    metric="Euclidean physical dipole coefficients at t=0",
                    temporal_model="deterministic diagonal cortical diffusion; no process noise",
                    oracle_assumptions="known exact anatomical basis and, in correct arm, known exact diffusion",
                    snr_definition="selected sparse array t=0 whitened signal power / expected true noise power",
                    pairings="same observations and their nested prefixes for every assumed model and time window",
                    true_kappa_m2_per_s=TRUE_KAPPA, assumptions=ASSUMPTIONS,
                    ridge_relative=.01, numerical_rtol=1e-10, effective_rtol=1e-3,
                    limitations=["Not evidence of recovery under unknown real resting-state dynamics",
                                 "Estimated two-second noise whitening has finite-sample error",
                                 "Rank increase does not guarantee noise-stable observability",
                                 "Regularization remains a prior, including within the known basis"],
                    input_sha256=hashes,
                    code_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 (Path(__file__), ROOT / "resolution_benchmark/nullspace.py",
                                  ROOT / "resolution_benchmark/recovery_metrics.py")})
    (output / "dynamics_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(rows=len(rows), output=str(output / "dynamics.csv"))))


if __name__ == "__main__":
    main()
