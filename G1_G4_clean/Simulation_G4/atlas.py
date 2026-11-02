"""Derived cortical AAL on the repository's exact Brainstorm fsaverage5 mesh.

Labels are sampled categorically using neuromaps registration-fusion MNI lookup
coordinates. They are never interpolated as continuous label numbers or filled
across unassigned vertices. See the bundle manifest for template limitations.
"""
from pathlib import Path
import csv
import hashlib
import json
import shutil

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
PROJECTION_MODES = ('area_unit_sum', 'membership')
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


def parcel_projection(labels, region_ids, vertex_areas, mode='area_unit_sum'):
    """Map parcel coefficients to vertices with an explicit amplitude convention.

    area_unit_sum retains the historical area-weighted, unit-column-sum map.
    membership broadcasts the same coefficient to every vertex of a parcel.
    These conventions change source and sensor amplitudes, not parcel labels.
    """
    if mode not in PROJECTION_MODES:
        raise ValueError(f'Unknown projection mode {mode!r}; choose from {PROJECTION_MODES}')
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
    weights = (vertex_areas[rows] / totals[columns[rows]] if mode == 'area_unit_sum'
               else np.ones(len(rows), dtype=float))
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


def build_atlas(directory=DEFAULT_DIRECTORY, *, projection_mode='area_unit_sum'):
    """Build the offline bundle from downloaded source resources; no implicit fetch."""
    directory = Path(directory)
    if projection_mode not in PROJECTION_MODES:
        raise ValueError(f'Unknown projection mode {projection_mode!r}')
    if (directory / 'atlas.npz').exists() or (directory / 'projection.npz').exists():
        manifest_path = directory / 'manifest.json'
        if not manifest_path.exists():
            raise FileExistsError('Existing atlas has no manifest; use a separate output directory')
        existing = json.loads(manifest_path.read_text(encoding='utf-8'))
        if existing.get('projection_mode', 'area_unit_sum') != projection_mode:
            raise FileExistsError('Refusing to overwrite a different projection mode; use a separate output directory')
    from nilearn import datasets, surface

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
    projection, vertex_to_region = parcel_projection(labels, ids, areas, mode=projection_mode)
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
                        cortex_mask=cortex_mask, vertex_area_mm2=areas,
                        projection_mode=np.array(projection_mode))
    source_files = [resource / "ROI_MNI_V4.nii", resource / "ROI_MNI_V4.txt", *lookup_files, cortex_path,
                    ROOT / "Anatomy/Gain_20k_FS.mat", ROOT / "Anatomy/Chanlocs.mat"]
    manifest = {
        "atlas_id": "aal78_cortical_fsaverage5_regfusion" + ('_membership' if projection_mode == 'membership' else ''), "n_regions": len(ids),
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
        "projection_mode": projection_mode,
        "projection_normalization": projection_description(projection_mode),
        "projection_file_sha256": hashlib.sha256((directory / 'projection.npz').read_bytes()).hexdigest(),
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
    manifest = atlas['manifest']
    if hashlib.sha256((directory / 'atlas.npz').read_bytes()).hexdigest() != manifest['bundle_sha256']:
        raise ValueError('Atlas SHA256 differs from its recorded bundle')
    mode = projection_mode_for_bundle(atlas, manifest)
    expected, indices = parcel_projection(atlas['labels_vertex'], atlas['region_ids'],
                                         atlas['vertex_area_mm2'], mode=mode)
    projection = atlas['projection']
    if (projection.shape != expected.shape or not np.array_equal(indices, atlas['vertex_to_region'])
            or not np.isfinite(projection.data).all()):
        raise ValueError('Projection geometry differs from recorded atlas assignment')
    difference = projection - expected
    if difference.nnz and np.max(np.abs(difference.data)) > 1e-14:
        raise ValueError('Projection differs from recorded projection mode and atlas assignment')
    if ('projection_file_sha256' in manifest and hashlib.sha256((directory / 'projection.npz').read_bytes()).hexdigest()
            != manifest['projection_file_sha256']):
        raise ValueError('Projection SHA256 differs from its recorded bundle')
    return atlas


def projection_mode_for_bundle(atlas, manifest):
    """Missing mode means the original area convention; disagreement is an error."""
    mode = manifest.get('projection_mode', 'area_unit_sum')
    array_mode = str(np.asarray(atlas.get('projection_mode', mode)).item())
    if mode not in PROJECTION_MODES or array_mode != mode:
        raise ValueError('Atlas projection mode and manifest projection mode disagree')
    return mode


def projection_description(mode):
    return ('vertex area / parcel total area; each column sums to 1' if mode == 'area_unit_sum'
            else 'binary membership; equal per-vertex amplitude within each parcel; no column normalization')


def derive_projection_view(existing_atlas, new_directory, vertex_leadfield, *, projection_mode='membership'):
    """Create a separate offline projection view without rebuilding atlas geometry.

    vertex_leadfield is the original full-mesh matrix or its Brainstorm MAT path.
    It must reproduce the existing regional leadfield before a new view is saved.
    No NMM trajectories or connectivity weights are modified.
    """
    existing_atlas, new_directory = Path(existing_atlas), Path(new_directory)
    if new_directory.exists() and any(new_directory.iterdir()):
        raise FileExistsError('Projection view requires a new or empty output directory')
    atlas = load_atlas(existing_atlas)
    n_vertices = len(atlas['labels_vertex'])
    if isinstance(vertex_leadfield, (str, Path)):
        gain_path = Path(vertex_leadfield)
        gain = load_brainstorm_leadfield(gain_path, n_vertices=n_vertices)
        gain_source = {'path': str(gain_path.resolve()),
                       'sha256': hashlib.sha256(gain_path.read_bytes()).hexdigest()}
    else:
        gain = np.asarray(vertex_leadfield, dtype=float)
        gain_source = {'array_sha256_float64_c_order': hashlib.sha256(np.ascontiguousarray(gain, dtype='<f8').tobytes()).hexdigest()}
    if gain.shape != (atlas['leadfield'].shape[0], n_vertices) or not np.isfinite(gain).all():
        raise ValueError('Vertex leadfield dimensions/values do not match the existing atlas')
    if not np.allclose(gain @ atlas['projection'], atlas['leadfield'], rtol=1e-10, atol=1e-12):
        raise ValueError('Vertex leadfield does not reproduce the existing regional leadfield')
    projection, indices = parcel_projection(atlas['labels_vertex'], atlas['region_ids'],
                                           atlas['vertex_area_mm2'], mode=projection_mode)
    arrays = {key: value for key, value in atlas.items() if key not in ('manifest', 'projection')}
    arrays.update(leadfield=np.asarray(gain @ projection), vertex_to_region=indices,
                  projection_mode=np.array(projection_mode))
    new_directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(new_directory / 'atlas.npz', **arrays)
    sparse.save_npz(new_directory / 'projection.npz', projection)
    shutil.copyfile(existing_atlas / 'parcels.csv', new_directory / 'parcels.csv')
    manifest = dict(atlas['manifest'])
    base_id = manifest['atlas_id'].removesuffix('_membership')
    manifest.update(atlas_id=base_id + ('_membership' if projection_mode == 'membership' else ''),
                    projection_mode=projection_mode,
                    projection_normalization=projection_description(projection_mode),
                    derived_from_projection_mode=projection_mode_for_bundle(atlas, atlas['manifest']),
                    derivation='projection-only offline view; labels, geometry, parcel ordering and NMM parameters unchanged',
                    source_bundle_files_sha256={str((existing_atlas / name).resolve()):
                        hashlib.sha256((existing_atlas / name).read_bytes()).hexdigest()
                        for name in ('atlas.npz', 'projection.npz', 'manifest.json', 'parcels.csv')},
                    vertex_leadfield_source=gain_source,
                    bundle_sha256=hashlib.sha256((new_directory / 'atlas.npz').read_bytes()).hexdigest(),
                    projection_file_sha256=hashlib.sha256((new_directory / 'projection.npz').read_bytes()).hexdigest())
    (new_directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return load_atlas(new_directory)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument('--projection-mode', choices=PROJECTION_MODES, default='area_unit_sum')
    parser.add_argument('--derive-from', type=Path, help='Offline existing atlas directory; never resample labels')
    parser.add_argument('--vertex-leadfield', type=Path, default=ROOT / 'Anatomy/Gain_20k_FS.mat')
    args = parser.parse_args()
    result = (derive_projection_view(args.derive_from, args.output_dir, args.vertex_leadfield,
                                     projection_mode=args.projection_mode) if args.derive_from
              else build_atlas(args.output_dir, projection_mode=args.projection_mode))
    print(json.dumps(result['manifest'], indent=2))
