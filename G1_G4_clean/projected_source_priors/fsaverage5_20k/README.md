# fsaverage5 20k Network Source Priors

## Published assets (2026-10-29)

This release contains 200 CLASS maps and 15 neuro_motifs maps, each in signed_maxabs
and abs_sum1 variants, as NPZ and MATLAB files, with their QC CSVs. The NPZ labels
are Unicode and every array is readable with `allow_pickle=False`. Numeric arrays
are unchanged from the historical inputs. MATLAB companions are unchanged.
See `ASSET_MANIFEST.json` for original and release SHA256 hashes. If a simulation
recipe pins the original NPZ file hash, update it deliberately for this label-only
container conversion; do not disable hash validation.

```python
import numpy as np
from pathlib import Path
root = Path('projected_source_priors/fsaverage5_20k')  # from G1_G4_clean/
path = root / 'neuro_motifs/neuro_motifs_fsaverage5_20k_source_priors_signed_maxabs.npz'
with np.load(path, allow_pickle=False) as data:
    weights = data['source_prior_weights']  # (20484, 15)
    names = data['network_names']          # Unicode, no pickle
```

These network rows match the supplied Brainstorm fsaverage5 cortex ordering.
The geometry-based LBO source modes can use this network file directly when the
matching anatomy/leadfield is supplied. Row agreement is NOT a certificate of
accurate MNI registration; see the coordinate limitation below.


This folder contains NeuroVault-derived network maps projected to the
fsaverage5 cortex surface.

## Inputs

- `assets/SIM_data/*.nii.gz`: 200 CLASS maps from clustered PCA components.
- `neuro_motifs_results/NIfTI_maps/*.nii.gz`: 15 motif maps from bootstrap PCA components.
- `Anatomy/Cortex_FS_20k.mat`: Brainstorm-style fsaverage5 cortex with 20484 vertices.

The input NIfTI maps are signed statistical/component maps derived from
NeuroVault Z/T maps. They are not probability maps.

## Coordinate Handling

`Cortex_FS_20k.mat` stores Brainstorm vertices in meters. Directly multiplying
these vertices by 1000 gives a surface that is rotated/shifted relative to the
Nilearn/MNE fsaverage5 surface.

The projection script estimates the Brainstorm-to-fsaverage5 correspondence by
matching to Nilearn fsaverage5 pial vertices. The current transform is:

```text
axis permutation: [1, 0, 2]
axis signs:       [-1, 1, 1]
translation mm:   [-0.10203189, -18.88167467, -46.73649980]
nearest-neighbor median distance to Nilearn fsaverage5: 0.935 mm
nearest-neighbor 95th percentile distance:              1.577 mm
```

The legacy `VerticesMNI_mm` / `vertices_mni_mm` field was used for NIfTI sampling.
It contains coordinates after this approximate surface alignment, not a verified
MRI-to-MNI registration. No nonlinear registration or validated surface-RAS to
MNI152 transform is stored. The original Brainstorm coordinates are also saved as
`VerticesBrainstorm_mm` / `vertices_brainstorm_mm`. Matching vertex rows/topology
does not by itself validate the anatomical accuracy of volume sampling.

## Output Versions

For each input set, two versions are saved.

### signed_maxabs

Files:

- `*_fsaverage5_20k_source_priors_signed_maxabs.mat`
- `*_fsaverage5_20k_source_priors_signed_maxabs.npz`

Arrays:

- `sourcePriorWeights` / `source_prior_weights`: signed values normalized by
  each map's maximum absolute sampled value, so each network is in `[-1, 1]`.
- `sourcePriorRawValues` / `source_prior_raw_values`: unnormalized signed
  values sampled directly from the original NIfTI map.

Use this version when the positive/negative component direction matters.

### abs_sum1

Files:

- `*_fsaverage5_20k_source_priors_abs_sum1.mat`
- `*_fsaverage5_20k_source_priors_abs_sum1.npz`

Arrays:

- `sourcePriorWeights` / `source_prior_weights`: `abs(raw sampled value)`,
  normalized so each network sums to 1 over 20484 dipoles.

Use this version as a non-negative source covariance or spatial probability
prior.

## QC

Each dataset folder contains:

- `*_projection_qc.csv`: numeric projection checks.
- `comparison_figures/*.png`: 5 randomly selected maps comparing original
  volumetric slices with projected cortical maps.

The generated random examples are deterministic under the script's default
`--random-seed 20260719`.

## Usage

Recommended command:

```powershell
.\.venv\Scripts\python.exe .\scripts\project_network_priors_to_fsaverage5.py
```

The `.venv` Python is recommended because it contains `nilearn` and
`matplotlib`, which are used for fsaverage5 coordinate matching and QC figures.

## Historical coactivation archive

`coactive_legacy/macm_surface_z.npz` contains the ORIGINAL 1,492,922,912-byte archive,
unchanged. Its SHA256 is
`85f3c235b9f58dbde604872d6ff6407bbb8340fd1b84d04761b57376534c3609`.
The matrix `M` has shape (20484, 20484), dtype float32, seed rows and target columns.
All values were streamed and checked finite for this release. Other fields include
`coords`, `vertno_lh`, `vertno_rh`, `voxel_mni`, `seed_vox_idx`, and `vox2surf`.

It is stored with Git LFS. From the repository checkout, run:

```sh
git lfs install
git lfs pull --include="G1_G4_clean/projected_source_priors/fsaverage5_20k/coactive_legacy/*"
```

A small text file beginning `version https://git-lfs.github.com/spec/v1` is only a
pointer, not the NPZ. GitHub ZIP downloads may not contain LFS payloads; use Git LFS.
Loading `M` through NumPy decompresses about 1.68 GB; NPZ is not a memory-mappable
row store. The publication inspector scans it with bounded memory.

Read `coactive_legacy/provenance.json` BEFORE scientific use. The artifact did not
record its coordinate transform at generation. The inspected legacy code compared
source surface-RAS and volume-world positions without an explicit registration,
and averaged all mask voxels into nearest surface bins. Its conditional standardized
association scores can be negative/asymmetric; they are NOT covariance, structural
connectivity, or inferential statistical Z scores. Brainstorm correspondence is
unverified. Do not silently reorder by nearest neighbours, symmetrize, or substitute
this matrix for EEG noise covariance. It is not connected to the current G1 network
loader.

The updated `build_fs_coactive.py` requires an explicit surface-to-mask-world affine
and records frame/transform/input provenance for NEW outputs. This release does
not retroactively repair or regenerate the historical matrix.
