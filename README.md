# data_generation_ESI

## G1-G4 simulation code

The self-contained code collection is in [G1_G4_clean](G1_G4_clean/README.md).
It includes synthetic G1 sources, unified G2-G3 neural mass simulation and
period datasets, and G4 data-guided parameter estimation/reconstruction.

Sensor samples include a separately generated baseline. The default noise
covariance is estimated from that baseline; event-component covariance is
retained only as an oracle diagnostic. See the
[baseline and whitening guide](G1_G4_clean/NMM_USAGE.md#per-sample-baseline-covariance-for-esi-whitening),
[asset setup](G1_G4_clean/EXTERNAL_ASSETS.md), and
[validation record](G1_G4_clean/VALIDATION.md).

EEG recordings, anatomical binaries, NMM signal banks, trained checkpoints,
virtual environments, and local caches are not included. The existing analysis
scripts in the repository root are retained.

## fsaverage5 network and coactivation assets

The [curated fsaverage5 asset collection](G1_G4_clean/projected_source_priors/fsaverage5_20k/README.md)
contains 200 CLASS maps, 15 network motifs (NPZ/MAT variants and QC), and the
original 20484-by-20484 coactivation archive. Network NPZ metadata is pickle-free.
The 1.49 GB coactivation archive is stored in Git LFS; run `git lfs pull` after cloning.

The historical coactivation matrix has unverified coordinate registration and is
not a covariance matrix. Its original bytes and provenance limitations are retained.
The reviewed `build_fs_coactive.py` now runs only through an explicit CLI and
requires a supplied surface-to-mask coordinate transform for new builds.
