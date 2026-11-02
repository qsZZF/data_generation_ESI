# Prior-dependent recovery in underdetermined source imaging

## Completed results

The pilot produced **4,850 method/support estimates and 96 known-dynamics
configurations**. These are crossed numerical conditions, not independent
subjects. Baseline-covariance conditions use two source/noise seeds at 10 and
20 dB; the known-covariance control uses one seed at 20 dB. All 1,150 SBL result
records now satisfy the objective stopping criterion. Ten initially flagged
training problems were refitted with a 600-iteration cap; the longest required
226 iterations. `convergence_refinement.json` preserves before/after records.

For 400 candidate sources, four active point sources and 20 dB full-array
training SNR, the mean held-out physical NRMSE over two seeds was:

| Method | MEG, 66 channels | hdEEG, 64 channels |
|---|---:|---:|
| MNE | 0.868 | 0.973 |
| wMNE | 1.019 | 0.962 |
| sLORETA | standardized map; no current NRMSE | standardized map; no current NRMSE |
| LCMV | 0.990 | 0.768 |
| SBL | 0.068 | 0.053 |
| SBL model covariance + LCMV | 0.793 | 0.647 |
| SBL-selected support + model LCMV | 0.276 | 0.047 |

sLORETA remains in the scale-aligned shape and spatial-power comparisons. Its
shape NRMSE here was 0.945/0.968 (MEG/hdEEG); it cannot be assigned a physical
null-recovery score without a current model. With strongly synchronous point
sources, ordinary LCMV's shape error increased to about 0.996 in both cases,
whereas SBL retained low errors in these particular sparse simulations.
With 100 distributed active sources, SBL NRMSE worsened to 0.813/0.939 and null
recovery to 0.208/-0.023. Sparse-point success therefore does not generalize to
arbitrary extended activity.

Define null recovery as
`1 - ||Pnull (Xhat-X)||_F^2 / ||Pnull X||_F^2`.
One is exact recovery, zero matches an estimate with no null component, and
negative values mean worse null error than that zero-null baseline. At 400 source
locations and 66 MEG / 64 EEG channels, SBL gave these mean scores at 20 dB:

| True field | LBO32, MEG / EEG | Full Green, MEG / EEG | Multiscale local, MEG / EEG | Local + full Green, MEG / EEG |
|---|---:|---:|---:|---:|
| Local patches | -1.327 / 0.004 | 0.804 / 0.360 | 0.942 / 0.890 | 0.942 / 0.886 |
| Matched smooth LBO field | 0.986 / 0.803 | 0.423 / -0.322 | 0.953 / 0.700 | 0.951 / 0.690 |
| Smooth field with omitted modes | 0.327 / 0.479 | 0.473 / -0.070 | 0.710 / 0.424 | 0.706 / 0.405 |
| Local transient + global alpha | -0.144 / 0.386 | 0.452 / -0.368 | 0.938 / 0.670 | 0.930 / 0.656 |
| Spatially delayed propagation | -3.633 / -0.179 | 0.435 / 0.311 | 0.176 / 0.393 | 0.460 / 0.429 |

Thus local multiscale sparsity is a promising prior for local and mixed fields;
low-frequency LBO is appropriate for genuinely smooth fields. Full Green adds
useful flexibility for propagation in this pilot. Adding it to the local dictionary
does not consistently improve local or mixed fields. A few fixed patch centers
are a poor substitute for adequately covering unknown locations and scales.

The frequency-band hybrid is not a universal improvement: on matched smooth
fields it gave null recovery 0.991/0.911, but on MEG propagation it failed badly
(-319.203). That failure remains in the figure and CSV. The model assumes separate
stationary diagonal variances per band; it does not represent a propagating wave's
joint phase/delay structure.

## What changes the null space?

The data operator's null space stayed identical across methods and dictionaries
for a fixed sensor/source geometry. Increasing source count or removing channels
changes that operator; choosing a prior changes which null components an estimator
selects. Plain MNE remained in the row space (null recovery zero). Weighted,
standardized, Bayesian and beamforming estimates can select other directions,
but only a correctly informative prior makes those selections useful.

Known temporal dynamics can change the operator for an *initial state*. With
`X(t)=B D^t z0`, the relevant matrix is `O=[A B; A B D; A B D^2; ...]`.
The diffusion control used 64 known LBO coefficients, 33 MEG / 32 EEG channels,
and no process noise. Correct dynamics raised O's rank from 30/31 to 64 by four
samples. Repeating a stationary operator left the rank at 30/31. At 64 samples
and 30 dB initial sparse-array SNR, two-seed means were:

| Assumed evolution | MEG initial NRMSE / null recovery | EEG initial NRMSE / null recovery |
|---|---:|---:|
| Correct diffusion | 0.355 / 0.862 | 0.500 / 0.688 |
| Stationary | 0.841 / 0.265 | 0.760 / 0.332 |
| Diffusion speed doubled | 1.341 / -0.940 | 0.868 / 0.073 |

This is an oracle demonstration: exact basis, deterministic evolution and known
diffusion in the correct arm. It supports studying explicit dynamic coupling,
not claiming recovery of arbitrary unknown resting-state null trajectories.
Noiseless correct-model pseudoinverse errors at 64 samples were about 1e-14;
noise and dynamics mismatch still materially limited reconstruction.

## Comparison and acceptance criteria

This extension compares fixed-normal wMNE, sLORETA, unit-gain LCMV, diagonal
multiple-measurement SBL, and LCMV using the SBL model covariance. Plain MNE is
included as a row-space control. All operate on the same projected, baseline-
whitened forward and observations. sLORETA is a standardized activity map rather
than a physical-current estimate: its shape/localization scores are comparable,
but its uncalibrated amplitude and null-energy recovery are not.

For a fixed forward A, its null space is independent of the reconstruction
method. An estimator can predict a true null component only by imposing a
restriction on admissible sources. The key test is whether that restriction
continues to help when the true source violates it.

The implementation adds separate method, basis, and scoring modules without
changing the G1–G4 generators. It reuses the existing projection, whitening,
source/sensor ordering and null-space decomposition. Tests must cover analytic
unit gain, SBL objective descent, basis geometry/normalization, null projection,
and indistinguishable sources. No new dependencies are required.

## Experimental separation

- Method sweep: nested point-source grids and channel subsets; the physical
  truth remains supported on the first 100 locations. Sparse independent,
  synchronous, and distributed sources are crossed with noise/SNR conditions.
- Support sweep: a common 400-location source model, with point, local patch,
  Green/resolvent, cortical FEM LBO and combined dictionaries. Smooth, local,
  mixed and propagating source fields test different assumptions. Geometry
  selects dictionary centers; test truth never chooses the fitted support.
- A time-frequency hybrid fits separate SBL variances in predefined bands.
  Train and test are independent realizations with the same spatial parameters;
  neither test covariance nor test source truth is used to fit inverse filters.
- Every simulation has an independent two-second noise baseline. Activity
  segments are generated separately for fitting and evaluation. SNR calibration
  occurs once on the full array and is preserved when channels are removed.

The study reports total, observable, near-null and null-component errors, spatial
power similarity, and support approximation error. Near-null uses a declared
relative singular-value cutoff, not an asserted noise threshold. Null recovery
is relative to the zero-null estimate: 1 is exact, 0 is no improvement, and
negative values mean the inferred null component makes the error worse.

## Interpretation boundaries

Full-column-rank A B can identify coefficients inside a known basis B even when
A is underdetermined on the full cortex. An invertible spatial smoothing matrix
does not remove null ambiguity by itself. Likewise, temporal smoothing alone
cannot identify an arbitrary null trajectory for a static forward. A spatial-
temporal combination helps only through additional assumptions about which
sources share a waveform, spectrum, support, or dynamics.

Hyperparameters are fixed in advance for this pilot, not optimized against test
truth. Method ranking therefore refers to these implementations and settings.
The MEG and hdEEG cases retain the previous study's distinct geometry and noise
assumptions and are not a device-performance comparison. Finite-element modes
are computed on actual cortical triangulations; resolvent kernels truncated to
those modes are explicitly labeled as approximations. Results use a common
Euclidean dipole-coefficient norm, not cortical current-density area integrals.

The completed dictionary design also includes a full FEM Green resolvent,
computed by sparse solution of `(M + length^2 K) g = delta` before restriction
to source vertices; it is not limited to the saved 64 eigenmodes. All-position
Gaussian geodesic dictionaries use 12, 24 and 48 mm widths (1,200 atoms), and
their union with the full Green dictionary has 1,600 atoms. These overcomplete
dictionaries do not eliminate nullity by rank restriction; SBL selects sparse
variance patterns inside them. The 32-center dictionaries remain as restricted-
coverage controls, not as representatives of all patch-based approaches.

The additional `sbl_lcmv_active` control keeps the smallest set of training-derived,
column-normalized SBL variances accounting for 99% of their sum. It zeros other
model-LCMV rows without looking at test truth. The threshold is a preset heuristic.
For each source, SBL's posterior operator is model-LCMV multiplied by
`gamma_i * (a_i.T Sigma^-1 a_i)`. Thus model covariance alone does not retain the
posterior's shrinkage. With an anatomical dictionary these filters operate on
dictionary coefficients before mapping back to the common physical source grid.

## Figure contract

Python quantitative grids test the claim that successful null prediction depends
on matching the support prior, and method performance depends on source structure.
The support heatmaps show all ten dictionaries, five truth families, full and
reduced arrays, both modalities, and 20 dB baseline-covariance results averaged
over two source/noise seeds. Colors clip null recovery to [-1,1], while labels
show unmodified values, including arbitrarily negative failures. The method
figure includes sLORETA using a clearly labeled scale-aligned shape score; this
diagnostic fits one scalar against test truth and is not an amplitude estimate.
All 10 dB, reference covariance, per-seed and physical-error data remain in CSV.
Figures are wide report layouts, exported as editable SVG/PDF and 300 dpi PNG;
there is no submission-specific TIFF or biological significance claim.

## Reproduction

After preparing the forwards described in `NULLSPACE_STUDY.md`, prepare bases:

```powershell
python scripts/prepare_nullspace_bases.py --forward generated_simulation_dataset/nullspace_study/mne_sample/mne_sample_forward.npz --source-fif generated_simulation_dataset/nullspace_study/mne_sample/sample_audvis_trunc-meg-eeg-oct-4-fwd.fif --output generated_simulation_dataset/nullspace_study/mne_sample/bases.npz
python scripts/prepare_nullspace_bases.py --forward generated_simulation_dataset/nullspace_study/hdeeg_template/forward.npz --source-fif <fsaverage>/bem/fsaverage-ico-5-src.fif --output generated_simulation_dataset/nullspace_study/hdeeg_template/bases.npz
python scripts/run_nullspace_methods.py
python scripts/run_nullspace_methods.py --covariance reference --seeds 1 --snrs 20
python scripts/run_nullspace_methods.py --refine-unconverged --max-iter 600
python scripts/run_nullspace_dynamics.py
python scripts/summarize_nullspace_methods.py
python scripts/plot_nullspace_methods.py
```

Outputs reside in `generated_simulation_dataset/nullspace_study/methods_supports`.
Only files ending in `_baseline.csv` or `_reference.csv` are final sweep inputs;
earlier development outputs without these suffixes are not used. Runtime and data
hashes are saved per case. SBL+LCMV timing includes shared SBL fitting work;
the two stages reuse one fit rather than optimizing twice.

## Verification and remaining limits

The original suite passed 395 tests with two explicit skips (397 total). The
isolated clean collection passed 380 with ten explicit skips (390 total).
Checks cover whitening, FEM/Green equations, SBL objective descent, unit gain,
support-restricted identifiability, unchanged data under a null perturbation,
fit/test separation, FFT band noise scaling, and dynamic observability. Every
physical estimate also passed the orthogonal observable/null error identity.
No new dependencies were installed; unavailable dedicated lint/type tools were
not run. Source syntax and report-figure preflight checks passed.

Important practical limits are the two simulation seeds, one geometry per
modality, preset hyperparameters, noise drawn independently over time, and no
subject/head-model perturbation. The repeated ERP template in the mixed condition
is shared between training and test; alpha and noise realizations are independent.
The MEG LBO mesh is the original coarse oct4 triangulation, whereas the EEG basis
uses ico5; these are not a matched device comparison.

Two seconds of baseline for hundreds of sensors introduces substantial covariance
estimation error. For example, seed-zero local-field SBL with the multiscale
dictionary improved on the full array from NRMSE 0.257 to 0.191 (MEG) and 0.364
to 0.216 (EEG) when the estimated covariance was replaced by its known reference.
Some apparent improvement after deleting channels is therefore an estimation/
regularization effect, not new information. Null-recovery denominators also change
with geometry; inspect total error and true null fraction together.

Future validation should cross anatomical errors and source priors, use more
realizations and measured baseline noise, and compare selected local supports
with dynamic local/global models. Sensor prediction alone cannot establish that
an unobservable source component is correct; a wrong prior can fit the same data.

## Primary implementation references

- [MNE inverse implementation and sLORETA normalization](https://mne.tools/stable/documentation/implementation.html).
- [MNE LCMV implementation](https://github.com/mne-tools/mne-python/blob/main/mne/beamformer/_lcmv.py).
- [Wipf et al., Champagne](https://pmc.ncbi.nlm.nih.gov/articles/PMC4083006/).
- [Cai et al., empirical Bayesian time-frequency reconstruction](https://pmc.ncbi.nlm.nih.gov/articles/PMC10411635/): model covariance and optimized beamformers.
