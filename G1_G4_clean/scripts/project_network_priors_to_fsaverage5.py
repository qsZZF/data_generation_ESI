#!/usr/bin/env python
"""Project NeuroVault-derived network maps to fsaverage5 cortex vertices.

The input maps are signed PCA/bootstrap-PCA component maps derived from
NeuroVault Z/T statistical maps. This script writes two source-prior versions:

1. signed_maxabs: preserves positive/negative values and scales each map by
   max(abs(value)), so weights are in [-1, 1]. The unscaled sampled values are
   saved in the same file as ``sourcePriorRawValues``.
2. abs_sum1: uses abs(raw value) and normalizes each map to sum to 1.

The Brainstorm cortex in Anatomy/Cortex_FS_20k.mat uses a different axis/origin
than Nilearn/MNE fsaverage5. By default the script estimates the axis transform
against Nilearn fsaverage5 and uses the aligned MNI-like fsaverage coordinates
for sampling volumetric NIfTI maps.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import re
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
from scipy.io import savemat
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree


def natural_key(path: Path) -> list[object]:
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", path.name)]


def load_brainstorm_cortex(cortex_file: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(cortex_file, "r") as h5:
        cortex = h5["Cortex_FS_20k"]
        vertices = np.asarray(cortex["Vertices"], dtype=np.float64).T
        faces = np.asarray(cortex["Faces"], dtype=np.int64).T

    if vertices.shape != (20484, 3):
        raise ValueError(f"Expected 20484x3 vertices, got {vertices.shape}")
    if np.nanmax(np.abs(vertices)) < 1.0:
        vertices = vertices * 1000.0
    if faces.min() == 1:
        faces = faces - 1
    return vertices, faces


def load_nilearn_fsaverage5() -> np.ndarray:
    from nilearn import datasets, surface

    fs = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    left_coords, _ = surface.load_surf_mesh(fs.pial_left)
    right_coords, _ = surface.load_surf_mesh(fs.pial_right)
    return np.vstack([left_coords, right_coords]).astype(np.float64)


def align_brainstorm_to_fsaverage(vertices_bst_mm: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    """Find the signed axis permutation that best matches Nilearn fsaverage5."""
    reference = load_nilearn_fsaverage5()
    best = None
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            transformed = vertices_bst_mm[:, perm] * np.asarray(signs)
            translation = reference.mean(axis=0) - transformed.mean(axis=0)
            candidate = transformed + translation
            distances, _ = cKDTree(reference).query(candidate, k=1)
            score = float(np.median(distances))
            if best is None or score < best[0]:
                best = (
                    score,
                    float(np.percentile(distances, 95)),
                    perm,
                    signs,
                    translation,
                    candidate,
                )

    assert best is not None
    median_distance, p95_distance, perm, signs, translation, vertices_mni = best
    info = {
        "coordinate_mode": "brainstorm_to_nilearn_fsaverage5",
        "axis_permutation_0based": np.asarray(perm, dtype=np.int32),
        "axis_signs": np.asarray(signs, dtype=np.int32),
        "translation_mm": translation.astype(np.float64),
        "median_nn_distance_to_nilearn_fsaverage5_mm": median_distance,
        "p95_nn_distance_to_nilearn_fsaverage5_mm": p95_distance,
    }
    return vertices_mni, info


def sample_raw_values(nii_file: Path, vertices_mni_mm: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    img = nib.load(str(nii_file))
    data = np.asarray(img.dataobj, dtype=np.float64)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    ijk = nib.affines.apply_affine(np.linalg.inv(img.affine), vertices_mni_mm)
    inside = np.all((ijk >= 0) & (ijk <= (np.asarray(img.shape) - 1)), axis=1)
    sampled = map_coordinates(
        data,
        ijk.T,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).astype(np.float32)

    abs_data = np.abs(data)
    abs_sampled = np.abs(sampled)
    peak_voxel = np.unravel_index(np.argmax(abs_data), data.shape)
    volume_peak = nib.affines.apply_affine(img.affine, np.asarray(peak_voxel))
    peak_vertex = int(np.argmax(abs_sampled)) if np.any(abs_sampled) else -1
    surface_peak = vertices_mni_mm[peak_vertex] if peak_vertex >= 0 else np.full(3, np.nan)

    qc = {
        "name": nii_file.stem.replace(".nii", ""),
        "source_file": str(nii_file),
        "input_min": float(data.min()),
        "input_max": float(data.max()),
        "input_abs_max": float(abs_data.max()),
        "sampled_min": float(sampled.min()),
        "sampled_max": float(sampled.max()),
        "sampled_abs_max": float(abs_sampled.max()),
        "sampled_nonzero": int(np.count_nonzero(sampled)),
        "inside_vertices": int(inside.sum()),
        "outside_vertices": int((~inside).sum()),
        "abs_peak_vertex": peak_vertex + 1 if peak_vertex >= 0 else 0,
        "surface_abs_peak_x": float(surface_peak[0]),
        "surface_abs_peak_y": float(surface_peak[1]),
        "surface_abs_peak_z": float(surface_peak[2]),
        "volume_abs_peak_x": float(volume_peak[0]),
        "volume_abs_peak_y": float(volume_peak[1]),
        "volume_abs_peak_z": float(volume_peak[2]),
        "abs_peak_distance_mm": float(np.linalg.norm(surface_peak - volume_peak)),
    }
    return sampled, qc


def signed_maxabs(raw_values: np.ndarray) -> np.ndarray:
    denom = np.max(np.abs(raw_values), axis=0, keepdims=True)
    return np.divide(raw_values, denom, out=np.zeros_like(raw_values), where=denom > 0)


def abs_sum1(raw_values: np.ndarray) -> np.ndarray:
    values = np.abs(raw_values)
    denom = values.sum(axis=0, keepdims=True)
    return np.divide(values, denom, out=np.zeros_like(values), where=denom > 0)


def weighted_centroid(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mass = weights.sum()
    if mass <= 0:
        return np.full(3, np.nan)
    return (points * weights[:, None]).sum(axis=0) / mass


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_prior_files(
    dataset_dir: Path,
    label: str,
    suffix: str,
    weights: np.ndarray,
    raw_values: np.ndarray | None,
    names: list[str],
    files: list[Path],
    vertices_mni_mm: np.ndarray,
    vertices_brainstorm_mm: np.ndarray,
    faces: np.ndarray,
    coordinate_info: dict[str, object],
) -> None:
    payload = {
        "source_prior_weights": weights.astype(np.float32),
        "network_names": np.asarray(names, dtype=object),
        "source_paths": np.asarray([str(p) for p in files], dtype=object),
        "vertices_mni_mm": vertices_mni_mm.astype(np.float32),
        "vertices_brainstorm_mm": vertices_brainstorm_mm.astype(np.float32),
        "faces": faces.astype(np.int32),
        **coordinate_info,
    }
    mat_payload = {
        "sourcePriorWeights": weights.astype(np.float32),
        "networkNames": np.asarray(names, dtype=object),
        "sourcePaths": np.asarray([str(p) for p in files], dtype=object),
        "VerticesMNI_mm": vertices_mni_mm.astype(np.float32),
        "VerticesBrainstorm_mm": vertices_brainstorm_mm.astype(np.float32),
        "Faces": (faces + 1).astype(np.int32),
        **coordinate_info,
    }
    if raw_values is not None:
        payload["source_prior_raw_values"] = raw_values.astype(np.float32)
        mat_payload["sourcePriorRawValues"] = raw_values.astype(np.float32)

    stem = f"{label}_fsaverage5_20k_source_priors_{suffix}"
    np.savez_compressed(dataset_dir / f"{stem}.npz", **payload)
    savemat(dataset_dir / f"{stem}.mat", mat_payload, do_compression=True)


def plot_original_slices(axs, nii_file: Path) -> None:
    img = nib.load(str(nii_file))
    data = np.nan_to_num(np.asarray(img.dataobj, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    peak = np.unravel_index(np.argmax(np.abs(data)), data.shape)
    vmax = float(np.percentile(np.abs(data), 99.5))
    vmax = vmax if vmax > 0 else 1.0
    slices = [
        (data[peak[0], :, :], "original sagittal"),
        (data[:, peak[1], :], "original coronal"),
        (data[:, :, peak[2]], "original axial"),
    ]
    for ax, (slc, title) in zip(axs, slices):
        ax.imshow(np.rot90(slc), cmap="coolwarm", vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.axis("off")


def plot_surface_panel(ax, vertices: np.ndarray, values: np.ndarray, axes: tuple[int, int], title: str) -> None:
    xy = vertices[:, axes]
    vmax = float(np.percentile(np.abs(values), 99.5))
    vmax = vmax if vmax > 0 else 1.0
    order = np.argsort(np.abs(values))
    ax.scatter(xy[order, 0], xy[order, 1], c=values[order], s=2.0, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.axis("off")


def write_comparison_figure(
    out_file: Path,
    nii_file: Path,
    vertices_mni_mm: np.ndarray,
    signed_values: np.ndarray,
    abs_values: np.ndarray,
    name: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 3, figsize=(12, 7), dpi=160)
    plot_original_slices(axs[0], nii_file)
    plot_surface_panel(axs[1, 0], vertices_mni_mm, signed_values, (1, 2), "projected signed lateral")
    plot_surface_panel(axs[1, 1], vertices_mni_mm, signed_values, (0, 2), "projected signed coronal")
    plot_surface_panel(axs[1, 2], vertices_mni_mm, abs_values, (0, 1), "projected abs dorsal")
    fig.suptitle(name, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)


def project_dataset(
    label: str,
    input_dir: Path,
    output_dir: Path,
    vertices_mni_mm: np.ndarray,
    vertices_brainstorm_mm: np.ndarray,
    faces: np.ndarray,
    coordinate_info: dict[str, object],
    n_random_figures: int,
    random_seed: int,
) -> None:
    files = sorted(input_dir.glob("*.nii.gz"), key=natural_key)
    if not files:
        raise FileNotFoundError(f"No .nii.gz files found in {input_dir}")

    dataset_dir = output_dir / label
    figure_dir = dataset_dir / "comparison_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    raw_values = np.zeros((vertices_mni_mm.shape[0], len(files)), dtype=np.float32)
    rows: list[dict[str, object]] = []
    names: list[str] = []
    for col, nii_file in enumerate(files):
        sampled, qc = sample_raw_values(nii_file, vertices_mni_mm)
        raw_values[:, col] = sampled
        rows.append(qc)
        names.append(str(qc["name"]))

    signed_weights = signed_maxabs(raw_values)
    abs_weights = abs_sum1(raw_values)

    for col, row in enumerate(rows):
        abs_col = abs_weights[:, col]
        centroid = weighted_centroid(vertices_mni_mm, abs_col)
        row["signed_norm_min"] = float(signed_weights[:, col].min())
        row["signed_norm_max"] = float(signed_weights[:, col].max())
        row["abs_sum1_sum"] = float(abs_col.sum())
        row["abs_sum1_nonzero"] = int(np.count_nonzero(abs_col))
        row["abs_surface_centroid_x"] = float(centroid[0])
        row["abs_surface_centroid_y"] = float(centroid[1])
        row["abs_surface_centroid_z"] = float(centroid[2])

    save_prior_files(
        dataset_dir,
        label,
        "signed_maxabs",
        signed_weights,
        raw_values,
        names,
        files,
        vertices_mni_mm,
        vertices_brainstorm_mm,
        faces,
        coordinate_info,
    )
    save_prior_files(
        dataset_dir,
        label,
        "abs_sum1",
        abs_weights,
        None,
        names,
        files,
        vertices_mni_mm,
        vertices_brainstorm_mm,
        faces,
        coordinate_info,
    )
    write_csv(dataset_dir / f"{label}_projection_qc.csv", rows)

    rng = np.random.default_rng(random_seed)
    selected = rng.choice(len(files), size=min(n_random_figures, len(files)), replace=False)
    for col in selected:
        write_comparison_figure(
            figure_dir / f"{col + 1:03d}_{names[col]}_original_vs_projected.png",
            files[col],
            vertices_mni_mm,
            signed_weights[:, col],
            abs_weights[:, col],
            names[col],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cortex", type=Path, default=Path("Anatomy") / "Cortex_FS_20k.mat")
    parser.add_argument("--sim-data", type=Path, default=Path("assets") / "SIM_data")
    parser.add_argument("--motifs", type=Path, default=Path("neuro_motifs_results") / "NIfTI_maps")
    parser.add_argument("--output", type=Path, default=Path("projected_source_priors") / "fsaverage5_20k")
    parser.add_argument("--coordinate-mode", choices=["auto", "direct"], default="auto")
    parser.add_argument("--random-figures", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=20260719)
    args = parser.parse_args()

    vertices_bst_mm, faces = load_brainstorm_cortex(args.cortex)
    if args.coordinate_mode == "auto":
        vertices_mni_mm, coordinate_info = align_brainstorm_to_fsaverage(vertices_bst_mm)
    else:
        vertices_mni_mm = vertices_bst_mm
        coordinate_info = {
            "coordinate_mode": "direct_brainstorm_vertices_times_1000",
            "axis_permutation_0based": np.asarray([0, 1, 2], dtype=np.int32),
            "axis_signs": np.asarray([1, 1, 1], dtype=np.int32),
            "translation_mm": np.asarray([0.0, 0.0, 0.0]),
            "median_nn_distance_to_nilearn_fsaverage5_mm": np.nan,
            "p95_nn_distance_to_nilearn_fsaverage5_mm": np.nan,
        }

    args.output.mkdir(parents=True, exist_ok=True)
    for label, input_dir in [("SIM_data", args.sim_data), ("neuro_motifs", args.motifs)]:
        project_dataset(
            label,
            input_dir,
            args.output,
            vertices_mni_mm,
            vertices_bst_mm,
            faces,
            coordinate_info,
            args.random_figures,
            args.random_seed,
        )

    print(f"Done. Outputs written to: {args.output.resolve()}")
    print(
        "Coordinate mode: {coordinate_mode}; perm={axis_permutation_0based}; "
        "signs={axis_signs}; translation_mm={translation_mm}; "
        "median_nn={median_nn_distance_to_nilearn_fsaverage5_mm:.3f} mm; "
        "p95_nn={p95_nn_distance_to_nilearn_fsaverage5_mm:.3f} mm".format(**coordinate_info)
    )


if __name__ == "__main__":
    main()
