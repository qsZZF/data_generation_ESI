"""Build a directed conditional-association surface proxy with explicit registration.

The historical function name is retained, but scores are not inferential Z
statistics or covariance. Import never downloads data, loads matrices or plots.
Existing legacy NPZ files are never silently replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree


LOGGER = logging.getLogger(__name__)
LEGACY_BUILDER_SHA256 = '05385f936d6e1688710b0e11cfc72360a6c88ad11a37bcd9bf348b0f934c1286'


def validate_surface_transform(value):
    """Require a supplied nonsingular affine, not an inferred identity mapping."""
    if value is None:
        raise ValueError('surface_to_mask_world_mm is required: supply a validated surface-RAS to mask-world affine')
    transform = np.asarray(value, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError('surface_to_mask_world_mm must be a finite 4x4 affine')
    if not np.allclose(transform[3], [0., 0., 0., 1.], rtol=0, atol=1e-12):
        raise ValueError('Affine last row must be [0, 0, 0, 1]')
    if np.linalg.matrix_rank(transform[:3, :3]) != 3:
        raise ValueError('Surface-to-mask affine must be nonsingular')
    return transform.copy()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _vertex_average(values, voxel_to_surface, counts):
    sums = np.bincount(voxel_to_surface, weights=values, minlength=len(counts))
    return np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)


def _method_metadata(radius, min_studies):
    return dict(
        schema='conditional_surface_association_v2',
        statistic='(conditional mean of MKDA maps - all-study mean) / all-study sample standard deviation',
        interpretation='directed standardized conditional association; not permutation Z, correlation, or covariance',
        symmetry='not enforced; rows condition on seed-specific study subsets',
        radius_mm=float(radius), min_studies=int(min_studies), ddof=1,
        constant_voxel_policy='replace zero standard deviation with one; constant conditional difference remains zero',
        unprocessed_seed_policy='zero row when fewer than min_studies activate the seed voxel',
        empty_surface_bin_policy='zero value',
        surface_coordinate_frame='MRI surface RAS, millimeters',
        mapped_coordinate_frame='mask-image world coordinates, millimeters; MNI identity not inferred',
        registration='explicit user-supplied affine; anatomical accuracy not independently validated',
        projection='all mask voxels assigned to nearest mapped surface vertex, then averaged',
        limitation='dense voxel-nearest-surface proxy; no cortical-ribbon or distance constraint; not validated cortical MACM',
        voxel_order='np.argwhere(mask_data), C-order; array length checked but kernel column identity not independently verified',
        legacy_builder_sha256=LEGACY_BUILDER_SHA256,
        legacy_hash_scope='original builder source only; does not certify provenance of existing legacy artifacts')


def build_macm_surface_z_matrix(dset_path, fsaverage_src_path=None, radius=10.0,
                                n_jobs=1, min_studies=5, output_npz='macm_surface_z.npz',
                                *, surface_to_mask_world_mm=None):
    """Return association matrix and original surface-RAS coordinates in mm.

    The supplied affine maps MRI surface-RAS mm to the dataset mask's world mm.
    If source space is omitted, fsaverage is fetched only on this explicit call.
    Output must be new; legacy artifacts are never silently overwritten.
    """
    transform = validate_surface_transform(surface_to_mask_world_mm)
    if isinstance(radius, (bool, np.bool_)) or not np.isfinite(radius) or radius <= 0:
        raise ValueError('radius must be finite and positive')
    for name, value in (('n_jobs', n_jobs), ('min_studies', min_studies)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    destination = Path(output_npz)
    if destination.exists():
        raise FileExistsError(f'Refusing to replace existing coactivation artifact: {destination}')

    # Optional heavy dependencies are needed only for an explicitly requested build.
    import nibabel as nib
    import mne
    from nimare.dataset import Dataset
    from nimare.meta.kernel import MKDAKernel

    dataset_path = Path(dset_path)
    LOGGER.info('Loading dataset: %s', dataset_path)
    dataset = Dataset.load(str(dataset_path))
    if len(dataset.ids) < 2:
        raise ValueError('At least two study maps are required for sample standard deviation')
    mask_img = dataset.masker.mask_img
    mask_data = mask_img.get_fdata().astype(bool)
    if mask_data.ndim != 3 or not mask_data.any():
        raise ValueError('Dataset mask must be a nonempty 3D image')
    ma_maps = MKDAKernel(r=radius).transform(dataset, return_type='array')
    if sparse.issparse(ma_maps):
        ma_maps = ma_maps.toarray()
    ma_maps = np.asarray(ma_maps, dtype=np.float32)
    if (ma_maps.shape != (len(dataset.ids), int(mask_data.sum()))
            or not np.isfinite(ma_maps).all()):
        raise ValueError('Kernel study/voxel array shape differs from dataset mask, or contains nonfinite values')
    activation = sparse.csc_matrix(ma_maps > 0)
    mean_all = ma_maps.mean(axis=0, dtype=np.float64).astype(np.float32)
    std_all = ma_maps.std(axis=0, ddof=1, dtype=np.float64).astype(np.float32)
    std_all[std_all == 0] = 1.

    if fsaverage_src_path is None:
        fsaverage_src_path = Path(mne.datasets.fetch_fsaverage()) / 'bem/fsaverage-ico-5-src.fif'
    source_path = Path(fsaverage_src_path)
    src = mne.read_source_spaces(str(source_path), verbose=False)
    # FIFFV_COORD_MRI == 5 denotes MRI surface RAS, not an MNI coordinate frame.
    if len(src) != 2 or any(int(hemisphere['coord_frame']) != 5 for hemisphere in src):
        raise ValueError('Expected two source-space hemispheres in MRI surface-RAS coordinates')
    vertices = [np.asarray(hemisphere['vertno']) for hemisphere in src]
    coords = np.concatenate([hemisphere['rr'][indices] * 1000.
                             for hemisphere, indices in zip(src, vertices)]).astype(np.float32)
    mapped_coords = nib.affines.apply_affine(transform, coords)
    voxel_world = nib.affines.apply_affine(mask_img.affine, np.argwhere(mask_data))
    if not np.isfinite(mapped_coords).all() or not np.isfinite(voxel_world).all():
        raise ValueError('Mapped surface or mask world coordinates are nonfinite')
    _, seed_voxel_indices = cKDTree(voxel_world).query(mapped_coords)
    _, voxel_to_surface = cKDTree(mapped_coords).query(voxel_world)
    counts = np.bincount(voxel_to_surface, minlength=len(coords))
    seed_study_counts = np.diff(activation.indptr)[seed_voxel_indices]

    def process_vertex(index):
        if seed_study_counts[index] < min_studies:
            return index, None
        voxel = seed_voxel_indices[index]
        studies = activation.indices[activation.indptr[voxel]:activation.indptr[voxel + 1]]
        conditional_mean = ma_maps[studies].mean(axis=0, dtype=np.float64).astype(np.float32)
        association = (conditional_mean - mean_all) / std_all
        return index, _vertex_average(association, voxel_to_surface, counts)

    matrix = np.zeros((len(coords), len(coords)), dtype=np.float32)
    if n_jobs > 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs)(delayed(process_vertex)(index) for index in range(len(coords)))
    else:
        results = (process_vertex(index) for index in range(len(coords)))
    for index, values in results:
        if values is not None:
            matrix[index] = values
    metadata = _method_metadata(radius, min_studies)
    metadata.update(dataset_path=str(dataset_path), dataset_sha256=_sha256(dataset_path),
                    source_space_path=str(source_path), source_space_sha256=_sha256(source_path),
                    builder_sha256=_sha256(__file__), n_studies=len(dataset.ids),
                    n_vertices=len(coords), n_mask_voxels=int(mask_data.sum()),
                    processed_seed_count=int(np.count_nonzero(seed_study_counts >= min_studies)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects against an output appearing during the build.
    with destination.open('xb') as handle:
        np.savez_compressed(handle, M=matrix, coords=coords, coords_mask_world_mm=mapped_coords,
            surface_to_mask_world_mm=transform, vertno_lh=vertices[0], vertno_rh=vertices[1],
            voxel_mask_world_mm=voxel_world, seed_vox_idx=seed_voxel_indices,
            vox2surf=voxel_to_surface, vertex_voxel_counts=counts, seed_study_counts=seed_study_counts,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)))
    LOGGER.info('Saved %d x %d association proxy to %s', len(coords), len(coords), destination)
    return matrix, coords


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--source-space', type=Path, help='MRI-coordinate source FIF; otherwise fetch fsaverage')
    parser.add_argument('--output', required=True, type=Path, help='New output NPZ; existing files are protected')
    parser.add_argument('--surface-to-mask-world-mm', required=True, type=Path,
                        help='Text file containing the externally validated 4x4 affine')
    parser.add_argument('--radius', default=10., type=float)
    parser.add_argument('--min-studies', default=5, type=int)
    parser.add_argument('--jobs', default=1, type=int)
    if argv is None:
        import sys
        argv = sys.argv[1:]
    if not argv:
        parser.print_help()
        return 0
    args = parser.parse_args(argv)
    transform = validate_surface_transform(np.loadtxt(args.surface_to_mask_world_mm))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    build_macm_surface_z_matrix(args.dataset, args.source_space, args.radius, args.jobs,
        args.min_studies, args.output, surface_to_mask_world_mm=transform)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
