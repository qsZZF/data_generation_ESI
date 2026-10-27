"""Derived cortical AAL on the repository's exact Brainstorm fsaverage5 mesh.

Labels are sampled categorically using neuromaps registration-fusion MNI lookup
coordinates. They are never interpolated as continuous label numbers or filled
across unassigned vertices. See the bundle manifest for template limitations.
"""
from pathlib import Path
import csv
import hashlib
import json

import h5py
import nibabel as nib
import numpy as np
from scipy import sparse
from scipy.ndimage import map_coordinates

from Simulation_G1.g1_simulation_core import mesh_vertex_areas
from Simulation_G2.g2_simulation_core import load_brainstorm_leadfield
from scripts.project_network_priors_to_fsaverage5 import load_brainstorm_cortex

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRECTORY = ROOT / "Anatomy/G4_AAL_cortical"
EXCLUDED_PREFIXES = ("Hippocampus_", "Amygdala_", "Caudate_", "Putamen_", "Pallidum_", "Thalamus_")


def verify_mesh_alignment(vertices, faces, reference, reference_faces):
    """Verify vertex correspondence, not merely proximity of unordered clouds."""
    if vertices.shape != reference.shape or faces.shape != reference_faces.shape:
        raise ValueError("Brainstorm/reference mesh shapes differ")
    if not np.array_equal(np.sort(faces, axis=1), np.sort(reference_faces, axis=1)):
        raise ValueError("Mesh topology or vertex order differs from fsaverage5")
    left, right = vertices - vertices.mean(0), reference - reference.mean(0)
    u, _, vt = np.linalg.svd(left.T @ right)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        raise ValueError("Mesh matching would require a reflection")
    translation = reference.mean(0) - vertices.mean(0) @ rotation
    distances = np.linalg.norm(vertices @ rotation + translation - reference, axis=1)
    if distances.max() > 1e-3:
        raise ValueError(f"fsaverage5 vertex correspondence failed: max error {distances.max()} mm")
    return {"rotation_row_vectors": rotation.tolist(), "translation_mm": translation.tolist(),
            "median_vertex_error_mm": float(np.median(distances)),
            "p95_vertex_error_mm": float(np.percentile(distances, 95)),
            "max_vertex_error_mm": float(distances.max()), "topology_matches": True}


def categorical_sample(image, coordinates):
    """Nearest voxel label in the supplied image's world coordinate system."""
    data = np.asarray(image.dataobj)
    if data.ndim != 3 or not np.isfinite(data).all() or not np.allclose(data, np.rint(data)):
        raise ValueError("Expected a finite 3D categorical atlas")
    ijk = nib.affines.apply_affine(np.linalg.inv(image.affine), coordinates)
    return map_coordinates(data, ijk.T, order=0, mode="constant", cval=0,
                           prefilter=False).astype(np.int32)


def parcel_projection(labels, region_ids, vertex_areas):
    """Area-weighted unit-sum source distribution for each retained parcel."""
    labels, region_ids, vertex_areas = map(np.asarray, (labels, region_ids, vertex_areas))
    if labels.shape != vertex_areas.shape or np.any(vertex_areas <= 0) or not np.isfinite(vertex_areas).all():
        raise ValueError("Vertex areas must be finite, positive and match labels")
    if len(set(region_ids.tolist())) != len(region_ids):
        raise ValueError("Parcel IDs must be unique")
    lookup = {int(label): i for i, label in enumerate(region_ids)}
    columns = np.array([lookup.get(int(label), -1) for label in labels], dtype=np.int32)
    rows = np.flatnonzero(columns >= 0)
    totals = np.bincount(columns[rows], weights=vertex_areas[rows], minlength=len(region_ids))
    if np.any(totals == 0):
        raise ValueError("Cannot project an empty parcel")
    weights = vertex_areas[rows] / totals[columns[rows]]
    return sparse.csc_matrix((weights, (rows, columns[rows])), shape=(len(labels), len(region_ids))), columns


def _matlab_text(file, ref):
    return "".join(chr(int(c)) for c in np.asarray(file[ref]).flat)


def _cortex_mask(cortex_path, n_vertices):
    """Embedded Desikan scouts define cortex; corpus callosum and unknown stay out."""
    mask = np.zeros(n_vertices, dtype=bool)
    with h5py.File(cortex_path) as file:
        atlases = file["Cortex_FS_20k/Atlas"]
        names = [_matlab_text(file, ref) for ref in np.asarray(atlases["Name"]).flat]
        scout = file[np.asarray(atlases["Scouts"]).flat[names.index("Desikan-Killiany")]]
        for name_ref, vertices_ref in zip(np.asarray(scout["Label"]).flat,
                                           np.asarray(scout["Vertices"]).flat):
            name = _matlab_text(file, name_ref)
            if not name.startswith(("corpuscallosum", "unknown")):
                mask[np.asarray(file[vertices_ref], dtype=int).ravel() - 1] = True
    return mask


def build_atlas(directory=DEFAULT_DIRECTORY):
    """Build the offline bundle from downloaded source resources; no implicit fetch."""
    from nilearn import datasets, surface

    directory = Path(directory)
    resource = directory / "resources"
    cortex_path = ROOT / "Anatomy/Cortex_FS_20k.mat"
    vertices, faces = load_brainstorm_cortex(cortex_path)
    fs = datasets.fetch_surf_fsaverage("fsaverage5")
    left, left_faces = surface.load_surf_mesh(fs.pial_left)
    right, right_faces = surface.load_surf_mesh(fs.pial_right)
    reference = np.vstack([left, right]).astype(float)
    ref_faces = np.vstack([left_faces, right_faces + len(left)])
    alignment = verify_mesh_alignment(vertices, faces, reference, ref_faces)
    lookup_files = [resource / f"tpl-MNI152_space-fsaverage_den-10k_hemi-{h}_regfusion.txt" for h in ("L", "R")]
    coordinates = np.vstack([np.loadtxt(path) for path in lookup_files])
    if coordinates.shape != vertices.shape or not np.isfinite(coordinates).all():
        raise ValueError("Invalid registration-fusion lookup shape or coordinates")
    table = [line.split() for line in (resource / "ROI_MNI_V4.txt").read_text().splitlines() if line.strip()]
    # The first 90 original AAL labels are the cerebral ROIs; the final 26 are cerebellar.
    if len(table) != 116 or table[0][1] != "Precentral_L" or table[89][1] != "Temporal_Inf_R":
        raise ValueError("Expected original AAL116 ordering, matching the AAL90 SC resource")
    expected = [(i + 1, name, int(label)) for i, (_, name, label) in enumerate(table[:90])
                if not name.startswith(EXCLUDED_PREFIXES)]
    raw_labels = categorical_sample(nib.load(resource / "ROI_MNI_V4.nii"), coordinates)
    cortex_mask = _cortex_mask(cortex_path, len(vertices))
    labels = raw_labels.copy()
    labels[~cortex_mask] = 0
    labels[~np.isin(labels, [label for _, _, label in expected])] = 0
    kept = [row for row in expected if np.any(labels == row[2])]
    dropped = [name for _, name, label in expected if not np.any(labels == label)]
    roi, names, ids = zip(*kept)
    # Boundary voxels can cross the interhemispheric fissure; reject instead of relabel.
    wrong_hemi = np.zeros(len(vertices), dtype=bool)
    for _, name, label in kept:
        wrong_hemi |= (labels == label) & ((np.arange(len(labels)) >= len(left)) if name.endswith("_L")
                                         else (np.arange(len(labels)) < len(left)))
    labels[wrong_hemi] = 0
    _, areas = mesh_vertex_areas(vertices, faces)
    projection, vertex_to_region = parcel_projection(labels, ids, areas)
    leadfield = load_brainstorm_leadfield(ROOT / "Anatomy/Gain_20k_FS.mat", n_vertices=len(vertices)) @ projection
    with h5py.File(ROOT / "Anatomy/Chanlocs.mat") as file:
        channels = [_matlab_text(file, ref) for ref in np.asarray(file["chanlocs/labels"]).flat]
    if len(channels) != leadfield.shape[0]:
        raise ValueError("Channel table and forward model row counts disagree")
    directory.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(directory / "projection.npz", projection)
    np.savez_compressed(directory / "atlas.npz", labels_vertex=labels, region_names=np.array(names),
                        region_ids=np.array(ids), aal_roi_numbers=np.array(roi),
                        vertex_to_region=vertex_to_region, leadfield=np.asarray(leadfield),
                        channel_names=np.array(channels), vertices=vertices, faces=faces,
                        vertices_fsaverage_mm=reference, regfusion_mni152_mm=coordinates,
                        cortex_mask=cortex_mask, vertex_area_mm2=areas)
    source_files = [resource / "ROI_MNI_V4.nii", resource / "ROI_MNI_V4.txt", *lookup_files, cortex_path,
                    ROOT / "Anatomy/Gain_20k_FS.mat", ROOT / "Anatomy/Chanlocs.mat"]
    manifest = {
        "atlas_id": "aal78_cortical_fsaverage5_regfusion", "n_regions": len(ids),
        "n_vertices": len(vertices), "n_channels": len(channels),
        "native_surface_atlas": False, "registration_status": "derived_registration_fusion_nominal_MNI",
        "template_limitation": "AAL is originally Colin27; neuromaps lookup is MNI152. AAL world coordinates are treated as nominal MNI. No Colin27-to-MNI152 nonlinear deformation was available; parcel boundaries remain approximate.",
        "registration_source": "https://github.com/netneurolab/neuromaps",
        "registration_url": "https://files.osf.io/v1/resources/4mw3a/providers/osfstorage/60b684c03a6df1020ed525f6",
        "registration_archive_md5": "8e6c4679366031d44b030e9566e101a0",
        "atlas_source": "https://github.com/fieldtrip/fieldtrip/tree/master/template/atlas/aal",
        "atlas_original": "https://www.gin.cnrs.fr/fr/outils/aal/",
        "atlas_reference": "https://doi.org/10.1006/nimg.2001.0978",
        "registration_reference": "https://doi.org/10.1002/hbm.24213",
        "mesh_alignment": alignment, "interpolation": "nearest voxel, no filling",
        "projection_normalization": "vertex area / parcel total area; each column sums to 1",
        "leadfield_reference": "original supplied reference; no automatic average referencing",
        "expected_cortical_regions": len(expected), "dropped_cortical_labels": dropped,
        "excluded_prefixes": list(EXCLUDED_PREFIXES),
        "unassigned_vertex_count": int(np.sum(labels == 0)),
        "medialwall_or_unscouted_vertex_count": int(np.sum(~cortex_mask)),
        "unassigned_cortical_vertex_count": int(np.sum((labels == 0) & cortex_mask)),
        "wrong_hemisphere_rejected_count": int(wrong_hemi.sum()),
        "cortical_coverage_fraction": float(np.sum(labels != 0) / cortex_mask.sum()),
        "files_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files},
        "bundle_sha256": hashlib.sha256((directory / "atlas.npz").read_bytes()).hexdigest(),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with (directory / "parcels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["region_index_0based", "aal_roi_number", "atlas_id", "name", "vertex_count", "area_mm2"])
        for index, (number, name, label) in enumerate(kept):
            writer.writerow([index, number, label, name, int(np.sum(labels == label)), float(areas[labels == label].sum())])
    return load_atlas(directory)


def load_atlas(directory=DEFAULT_DIRECTORY):
    directory = Path(directory)
    with np.load(directory / "atlas.npz", allow_pickle=False) as data:
        atlas = {key: data[key] for key in data.files}
    atlas["projection"] = sparse.load_npz(directory / "projection.npz")
    atlas["manifest"] = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    return atlas


if __name__ == "__main__":
    print(json.dumps(build_atlas()["manifest"], indent=2))
