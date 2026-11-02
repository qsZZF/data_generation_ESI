# data_generation_ESI

## G1-G4 simulation code

The self-contained code collection is in [G1_G4_clean](G1_G4_clean/README.md).
It includes synthetic G1 sources, unified G2-G3 neural mass simulation and
period datasets, and G4 data-guided parameter estimation/reconstruction.

The November 3, 2026 source snapshot includes the named-array sample interface,
transactional saving and verified resume, explicit SNR conventions, cortical
multiscale sources, and online/offline sample quality checks. Production recipes
default to IID measurement noise and use spatially correlated source backgrounds
only with explicitly supplied cortical resources.

- [Current training policy and QC](G1_G4_clean/QUALITY_AND_TRAINING_POLICY.md)
- [Recipe audit and measured-noise comparison](G1_G4_clean/OPTIMIZATION_RESULTS.md)
- [Named-array API and sample storage](G1_G4_clean/PRODUCTION_SAMPLE_INTERFACE.md)
- [G4 export to the unified NMM bank](G1_G4_clean/G4_UNIFIED_EXPORT.md)
- [Current G4 training pilot results](G1_G4_clean/G4_REFRESH_RESULTS.md)
- [Nullspace methods and support experiments](G1_G4_clean/NULLSPACE_METHODS_STUDY.md)

Measured-noise augmentation remains opt-in: the bounded experiment improved
held-out measured PSD fitting but reduced performance on the IID test domain.
G4 regional physiological parameter recovery remains unvalidated. Historical
reports retain their original run scope and publication-status notes.

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
