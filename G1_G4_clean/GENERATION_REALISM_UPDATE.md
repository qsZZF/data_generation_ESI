# Generator correctness and realism update

The subsequent [2026-11-01 geometry/units/observability extension](GEOMETRY_UNITS_OBSERVABILITY.md)
adds geodesic patches and physical-amplitude interfaces. The remaining-scope notes
below describe this earlier 2026-10-31 release.

This release fixes confirmed defects and adds controlled spatial/noise realism.
It does not establish physiological current calibration or a new fit to measured
EEG/MEG. Source representation and physical amplitude units are kept separate.

## Changes

- **G1 spectral ERP:** random spatial coefficients are now the default. The
  independent spatial stream changes topography without changing the temporal
  response or stationary background draw. `erp_spatial_mode="fixed"` preserves
  the old controlled pattern; rest/ERSP numerical regression fixtures pass.
- **G1 cortical placement:** Brainstorm loaders read available Desikan scouts.
  Automatic/explicit seeds and full patch supports exclude unknown/corpus-callosum
  vertices. Custom meshes without a mask are explicitly marked as unspecified.
  `--no-cortex-mask` is available for legacy G1 CLI replay. Full spectral fields
  still use their supplied basis domain; this is not a claim that every spectral
  mode has been rebuilt on a masked surface.
- **Patch source background:** `spectral_rest` reuses cortical LBO/network
  stationary processes with 1/f and alpha/beta settings. The event and baseline
  use the same basis, cortical mask, noise parameters and calibrated gain.
  They occupy disjoint time intervals. Background settings and hashes are saved.
- **Whitening:** ranks are identified after unit standardization, then filters
  respect the retained physical covariance support. Mixed EEG/MEG units no
  longer decide which sensor types survive the rank cutoff.
- **Covariance:** optional `covariance_estimator="oas"` shrinks in the known
  CAR/SSP subspace; discarded reference directions are not reintroduced.
  `empirical` remains the compatibility default. The default baseline stays
  **two seconds**; no simulator-only configuration covariance is required.
- **G4 driving:** new banks default to `independent`; `shared` and `mixed`
  (default correlation 0.15) are explicit choices. Drive identity travels through
  sample caches, training checkpoints and reconstruction. Low-level shared mode
  reproduces the prior trajectory exactly. NN `paper/compact` profiles retain
  their original architecture meaning.
- **NMM source SNR:** new direct calls and new period plans use `vertex_mean`.
  `area_weighted` is an explicit legacy option, and old period recipes missing
  the field retain area weighting. New plans use `nmm-periods-v2` and require
  explicit SNR/covariance semantics, so older readers reject them rather than
  silently replaying changed defaults. Both actual SNR metrics are recorded.
- **AAL:** `aal78_membership` uses a separate derived projection/bundle. Region
  order and structural connectivity remain identical to the old `aal78` profile;
  region-to-sensor leadfields are recomputed for the new projection.
- **Network scripts:** fix actual-count image/term alignment, remove exploratory
  cross-image smoothing, seed random draws and record effective-input provenance.
  Imports no longer load imaging dependencies or configure application logging.
  The existing clustering method and NeuroVault download filters are preserved.

## G1 example

Run from the project root after supplying the anatomy and a matching LBO bundle.
`Simulation_G1/realistic_background.example.json` points to the existing local
fsaverage5 validation basis; change `basis_path` for another installation.

```powershell
.\.venv\Scripts\python.exe simulate_source_sensor.py --engine g1 --source-type erp --sample-id cortical_rest_demo --n-active 4 --patch-area-mm2 180 --source-background-type spectral_rest --source-background-config Simulation_G1/realistic_background.example.json --source-snr-db 12 --sensor-noise-type iid_white --sensor-snr-db 10 --sensor-snr-mode active_rms --covariance-estimator oas --duration-s 1 --srate-hz 256 --output-dir generated_simulation_dataset/realism_validation/g1_samples
```

Here 10 dB is the requested **active-window reference** SNR; the global realized
sensor SNR can differ. Both remain in metadata. Empirical covariance and existing
`global_rms/robust_global_rms/active_rms` choices are preserved.

For configuration-only training, add `--background-spectral-config
Simulation_G1/realistic_background.example.json --covariance-estimator oas` to
`python -m Simulation_G1.training_recipes`. Variants that select a background
use the specified cortical field; no-background variants remain available.
Recipes bind the mask identity, basis/network hashes and new source code snapshot.

## AAL and unified NMM

The membership assets have been derived locally in new directories. On another
installation, derive them from the existing area-unit-sum bundle:

```powershell
python -m Simulation_G4.atlas --derive-from Anatomy/G4_AAL_cortical --output-dir Anatomy/G4_AAL_cortical_membership --projection-mode membership
python -m Simulation_G3.prepare_aal_tvb --atlas-dir Anatomy/G4_AAL_cortical_membership --output-dir Anatomy/G3_AAL78_membership_TVB --sc-dir Anatomy/G4_resources
python nmm_simulation.py --config Simulation_G2/realistic_nmm.example.json --output-dir generated_simulation_dataset/realism_validation/nmm_samples
```

The first command intentionally refuses a nonempty destination. Existing old
bundles were not overwritten. The last command also refuses an existing sample.
The alpha preset remains a candidate parameter recipe, not a guarantee that each
trajectory matches measured alpha activity.

## Compatibility and migration

Source banks and prior published assets have not been rewritten. A different G4
drive changes the dynamics and requires a new source bank and trained inverse
model. Even shared replay retains the strict simulator-code identity guard; do
not edit old config hashes to bypass it. Old checkpoints should be used with
their recorded source/code environment.

A projection-only change does not mathematically require new regional ODE
trajectories, but current bank identities bind projection hashes. This release
does not silently relabel old banks. Use a new bank/profile or a future explicit
audited derived view, then rebuild indices, sensor views and affected G4 models.

The unit convention is `source_representation=per_vertex_activity`; G4 temporal
trajectories remain mV PSP and normalized sources remain relative model activity.
No arbitrary normalization is relabeled as A·m. Area-unit-sum/membership and
coefficient mass metrics stay explicit.

## Measured short-run checks

- fs5 eligibility: **18,426 retained / 2,058 excluded** vertices.
- Twenty ERP seeds: spatial rank **1 → 20**, median absolute cosine **1 → 0.08384**.
- Six real-geometry patch/background samples reproduced exactly, obeyed the
  cortical mask, and achieved 12 dB source SNR within 2.5e-7 dB.
- Eligible-edge normalized background roughness: legacy pink **2.0004**, cortical
  spectral rest **0.01483**. Alpha 8–13 / 1–40 Hz power: **0.15765 → 0.42655**.
  These show controllable spatial smoothness and spectral structure, not a
  demonstrated match to empirical recordings.
- MEG 305 channels, 512 baseline samples and 4,096 independent held-out samples:
  mean whitened covariance error/√rank **3.3732 → 1.0215** with OAS; mean eigenvalue
  condition **66.278 → 56.864**. Both retain rank 302. Shrinkage helps here but does
  not make a two-second covariance estimate exact.
- Mixed 364-channel EEG/MEG covariance retains **360** dimensions instead of the
  previous raw-unit cutoff's **58**.
- Three-seed uncoupled heterogeneous G4 pilot: mean output |r| **0.197 → 0.104**
  from shared to independent drive; effective rank **2.994 → 3.422**. Independent
  inputs do not guarantee zero short-window output correlations.
- One actual AAL membership NMM sample achieved **15 dB source / 10 dB sensor**
  SNR and a two-second baseline through the unified interface.

## Remaining scope

Patch growth still uses the existing BFS compatibility geometry; a new geodesic
growth model has not been substituted. Network maps/coactivation registration and
large datasets were not rebuilt. No new physiological-current default, posterior
inference network, or long G4 bank training was introduced. Optional TVB surface
geodesic/orientation warnings are unrelated to these fixed-normal parcel runs.
