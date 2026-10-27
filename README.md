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
