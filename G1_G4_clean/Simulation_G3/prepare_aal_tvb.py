"""Export verified local AAL78 anatomy and structural weights to a TVB ZIP.

No downloads: tract lengths are explicitly Euclidean centroid-distance proxies,
not measured fibre lengths. Existing G4 anatomy is only read, never changed.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Simulation_G2.nmm_bank import projection_hash
from Simulation_G4.resources import SC_RESOURCES, load_cortical_connectivity

DEFAULT_OUTPUT = ROOT / "Anatomy/G3_AAL78_TVB"
DEFAULT_ATLAS = ROOT / "Anatomy/G4_AAL_cortical"
DEFAULT_SC = ROOT / "Anatomy/G4_resources"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def centroid_distances(vertices, areas, vertex_to_region, n_regions):
    """Area-weighted parcel centroids and straight-line distances, both in mm."""
    vertices, areas, labels = map(np.asarray, (vertices, areas, vertex_to_region))
    if (vertices.shape != (len(areas), 3) or labels.shape != areas.shape
            or not np.isfinite(vertices).all() or not np.isfinite(areas).all()
            or np.any(areas <= 0) or np.any(labels < -1) or np.any(labels >= n_regions)
            or not np.issubdtype(labels.dtype, np.integer)):
        raise ValueError("Invalid parcel geometry or area weights")
    assigned = labels >= 0
    totals = np.bincount(labels[assigned], weights=areas[assigned], minlength=n_regions)
    if np.any(totals <= 0):
        raise ValueError("Cannot compute a centroid for an empty parcel")
    centres = np.column_stack([
        np.bincount(labels[assigned], weights=areas[assigned] * vertices[assigned, axis],
                    minlength=n_regions) / totals for axis in range(3)
    ])
    distances = np.linalg.norm(centres[:, None] - centres[None, :], axis=2)
    return centres, distances


def without_self_connections(weights):
    """Preserve published off-diagonal weights, removing local self coupling."""
    result = np.array(weights, dtype=float, copy=True)
    if result.ndim != 2 or result.shape[0] != result.shape[1] or not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("Structural weights must be a finite nonnegative square matrix")
    np.fill_diagonal(result, 0)
    return result


def build_assets(output_dir=DEFAULT_OUTPUT, *, atlas_dir=DEFAULT_ATLAS, sc_dir=DEFAULT_SC):
    atlas_dir, sc_dir, output_dir = map(Path, (atlas_dir, sc_dir, output_dir))
    sc_path = sc_dir / "SCmatrices88healthy.mat"
    if sha256(sc_path) != SC_RESOURCES[sc_path.name][1]:
        raise ValueError("Structural connectivity SHA256 differs from verified published resource")
    atlas_manifest = json.loads((atlas_dir / "manifest.json").read_text(encoding="utf-8"))
    if sha256(atlas_dir / "atlas.npz") != atlas_manifest["bundle_sha256"]:
        raise ValueError("Atlas SHA256 differs from its recorded bundle")
    with np.load(atlas_dir / "atlas.npz", allow_pickle=False) as source:
        atlas = {key: source[key] for key in source.files}
    names, ids, roi = atlas["region_names"], atlas["region_ids"], atlas["aal_roi_numbers"]
    if (len(names) != 78 or len(set(names.tolist())) != 78 or len(set(ids.tolist())) != 78
            or roi.shape != (78,) or ids.shape != (78,)):
        raise ValueError("Expected exactly 78 uniquely labelled cortical AAL parcels")
    with (atlas_dir / "parcels.csv").open(encoding="utf-8", newline="") as handle:
        parcels = list(csv.DictReader(handle))
    reference_path = ROOT / "Simulation_G4/reference/aal_regions.csv"
    with reference_path.open(encoding="utf-8", newline="") as handle:
        reference = list(csv.DictReader(handle))
    if len(parcels) != 78:
        raise ValueError("Parcel table and atlas region count disagree")
    for index, (name, label, number, parcel) in enumerate(zip(names, ids, roi, parcels)):
        if not 1 <= int(number) <= 90:
            raise ValueError("AAL cortical ROI index must lie within the original AAL90")
        original = reference[int(number) - 1]
        if (int(parcel["region_index_0based"]) != index or parcel["name"] != name
                or int(parcel["atlas_id"]) != label or int(parcel["aal_roi_number"]) != number
                or original["atlas_name"] != name or int(original["atlas_mni_id"]) != label):
            raise ValueError("AAL name, label and connectivity ROI ordering disagree")
    projection = sparse.load_npz(atlas_dir / "projection.npz").tocsc()
    indices = atlas["vertex_to_region"]
    vertices, areas = atlas["vertices_fsaverage_mm"], atlas["vertex_area_mm2"]
    if projection.shape != (20484, 78) or indices.shape != (20484,):
        raise ValueError("AAL projection must map 78 parcels to the current 20484 vertices")
    centres, distances = centroid_distances(vertices, areas, indices, 78)
    assigned = indices >= 0
    if (not np.array_equal(atlas["labels_vertex"][assigned], ids[indices[assigned]])
            or np.any(atlas["labels_vertex"][~assigned] != 0)):
        raise ValueError("Atlas vertex labels do not match projection column ordering")
    totals = np.bincount(indices[assigned], weights=areas[assigned], minlength=78)
    expected = sparse.csc_matrix((areas[assigned] / totals[indices[assigned]],
                                 (np.flatnonzero(assigned), indices[assigned])), shape=projection.shape)
    if projection_hash(projection) != projection_hash(expected):
        raise ValueError("Projection differs from recorded area-weighted atlas assignment")
    original_weights = load_cortical_connectivity(sc_dir, roi)
    weights = without_self_connections(original_weights)
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "connectivity_78.zip"
    temporary = zip_path.with_suffix(".part")
    arrays = {"weights.txt": weights, "tract_lengths.txt": distances,
              "cortical.txt": np.ones(78, dtype=int),
              "hemispheres.txt": np.array([name.endswith("_R") for name in names], dtype=int),
              "areas.txt": totals}
    with ZipFile(temporary, "w") as archive:
        # Fixed timestamps and round-trip text precision make offline exports reproducible.
        for filename, values in arrays.items():
            stream = io.StringIO()
            np.savetxt(stream, values, fmt="%.17g")
            entry = ZipInfo(filename, date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = ZIP_DEFLATED
            archive.writestr(entry, stream.getvalue())
        lines = [name + " " + " ".join(format(float(v), ".17g") for v in centre)
                 for name, centre in zip(names, centres)]
        entry = ZipInfo("centres.txt", date_time=(1980, 1, 1, 0, 0, 0))
        entry.compress_type = ZIP_DEFLATED
        archive.writestr(entry, "\n".join(lines) + "\n")
    temporary.replace(zip_path)
    strengths = weights.sum(axis=1)
    manifest = {
        "atlas_id": atlas_manifest["atlas_id"], "n_regions": 78, "n_vertices": 20484,
        "connectivity_file": zip_path.name, "connectivity_sha256": sha256(zip_path),
        "projection_path": str((atlas_dir / "projection.npz").resolve()),
        "projection_sha256": projection_hash(projection),
        "source_files_sha256": {str(path.resolve()): sha256(path) for path in
                                (sc_path, atlas_dir / "atlas.npz", atlas_dir / "manifest.json",
                                 atlas_dir / "projection.npz", atlas_dir / "parcels.csv", reference_path)},
        "structural_source": "https://doi.org/10.1038/s41597-022-01596-9",
        "structural_license": "CC-BY-4.0",
        "connectivity_orientation": "target_source (published source_target mean transposed)",
        "connectivity_aggregation": "mean of 88 healthy participants, then subset both axes by AAL ROI number",
        "weight_scaling": "none: all published mean off-diagonal weights unchanged",
        "self_connections": "diagonal explicitly removed; local recurrence is in the neural mass model",
        "original_diagonal": np.diag(original_weights).tolist(),
        "nonzero_edges": int(np.count_nonzero(weights)),
        "incoming_strength": {"min": float(strengths.min()), "median": float(np.median(strengths)),
                              "max": float(strengths.max()), "per_region": strengths.tolist()},
        "region_names": names.tolist(), "region_ids": ids.tolist(), "aal_roi_numbers": roi.tolist(),
        "centres_mm": centres.tolist(), "centroid_space": "existing fsaverage5 surface, millimetres",
        "centroid_method": "vertex-area-weighted parcel mean",
        "tract_lengths_method": "Euclidean distance between parcel centroids; approximate delay proxy, not measured fibre length",
        "nominal_conduction_speed_mm_per_ms": 3.0,
        "delay_assumption": "TVB delays = Euclidean proxy length / speed; fixed nominal speed 3 mm/ms",
        "template_limitation": atlas_manifest["template_limitation"],
        "scope": "cortical AAL78 only; subcortical and cerebellar regions excluded",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build_assets(args.output_dir)
    print(json.dumps({"directory": str(args.output_dir), "n_regions": result["n_regions"],
                      "connectivity_sha256": result["connectivity_sha256"]}))
