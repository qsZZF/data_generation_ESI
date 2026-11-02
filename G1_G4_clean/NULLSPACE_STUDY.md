# MEG / high-density EEG null-space pilot

## Question and design

Can reducing the cortical parameter count give a stable starting point, and do
removing channels and adding source parameters produce similar loss of information?

The relevant operator is **A = W P G B**: G is the fixed head-model forward,
P removes nuisance/reference directions, W whitens the corresponding noise
covariance, and B defines the source basis. A has a positive-definite normal
matrix only when it has full column rank. This does not guarantee a well-conditioned
inverse. Free orientation introduces three unknowns per source location.

The pilot keeps the volume-conductor geometry unchanged. It changes inverse
source subsets or nested coherent groups, not BEM mesh resolution. Main source
sets contain 100, 150, 200, 250, 300, and 400 fixed-normal dipoles; the same 100-source
physical activity is embedded in every model. Channel selections are nested
spatial farthest-point sets, repeated for three selection seeds. MEG co-located
sensor groups remain together. Source energy uses Euclidean dipole-coefficient
norms; these results are not area-weighted cortical current-density estimates.

## Cases and provenance

- **MEG:** pinned MNE testing-data sample forward and covariance; 305 good MEG
  channels, 498 cortical locations, three SSP directions, noise-subspace rank 302.
  This is sample geometry with synthetic activity and Gaussian noise drawn from
  the supplied covariance. It is not a recorded 302-rank brain signal and is not
  a device-performance evaluation. Download identities and usage notices are saved.
- **hdEEG:** standard BioSemi256 montage on cached fsaverage, unchanged three-layer
  BEM, 512 selected cortical locations. Average reference leaves 255 sensor
  dimensions. Noise is explicitly a model: independent 1 microvolt standard
  deviation before referencing. There is no empirical hdEEG recording here.

The cases use different anatomy and noise assumptions; their errors must not be
interpreted as MEG-versus-EEG performance superiority. Physical channel count,
preprocessed noise-subspace rank, activity PCA rank, and leadfield rank are distinct.

## Signal, covariance and inverse controls

G1 generates four seconds at 256 Hz, with a 1/f temporal component and 10.5 Hz
alpha activity. Independent Gaussian noise draws provide a two-second baseline
and activity noise. Whitening standardizes channel units before covariance rank
estimation, which is essential when mixing magnetometers and gradiometers.

Full-array whitened SNR is calibrated to 20 dB once. It is not recalibrated when
channels or source count change. Every cell compares the two-second estimated
covariance with the supplied covariance. Restricted SSP/reference patterns are
recomputed consistently for G, data and covariance in each channel subset.

Numerical rank uses relative tolerance 1e-10. A separate 1e-3 singular-value cutoff
defines weak, near-null directions; sensitivity checks also save 1e-2 and 1e-4.
This cutoff is a conditioning convention, not an SNR threshold. Actual per-mode
signal/noise powers are saved separately. Inverses include pseudoinverse, truncated
SVD, and ridge; ridge lambda is fixed from the full-array 100-source baseline case.

For X, the numerical-null component is Xnull = (I - Vr Vr.T) X. The noiseless
minimum-norm estimate recovers only Xobservable = Vr Vr.T X. In the matched model,
the squared error separates into null energy and error in the observable subspace.
For isotropic source covariance only, expected null-energy fraction equals
(N-rank(A))/N. Structured activity need not follow that dimensional ratio.

Controls include free orientations, spatially distributed versus forward-QR-selected
locations, off-grid activity, and hierarchical coherent groups whose refinement
preserves exactly the same physical source field. QR selection favors sensor-visible
locations and is not unbiased whole-cortex coverage. Off-grid error is evaluated on
a common fine point grid; it is not a localization-distance metric.

## Run and outputs

The completed pilot produced 252 MEG and 216 hdEEG sweep cells. In all three
channel-selection orderings, numerical-null energy was nondecreasing both when
removing channels and when adding source parameters. The maximum relative
sensor residual of the numerical-null component was below 2.5e-15.

| Case | Channels | Sources | Rank | Condition number | Null energy | Pseudoinverse NRMSE |
|---|---:|---:|---:|---:|---:|---:|
| MEG | 305 | 100 | 100 | 4,147 | ~0% | 11.58 |
| MEG | 305 | 300 | 300 | 1.71e8 | ~0% | 178,975 |
| MEG | 305 | 400 | 302 | singular | 24.47% | 1,714 |
| hdEEG | 256 | 100 | 100 | 159,135 | ~0% | 204.88 |
| hdEEG | 256 | 300 | 255 | singular | 16.94% | 103,604 |

These are the first ordering, estimated baseline covariance and a 20 dB full-array
calibration. NRMSE means ||estimate-truth|| / ||truth||, so values above one indicate
error larger than the true source norm. Full-column rank alone clearly fails to
ensure reliable reconstruction. Ridge lowers the 100-source full-array NRMSE to
0.609 (MEG) and 0.752 (hdEEG), still not accurate recovery.

For 100 MEG source locations, reducing channels from 305 to 102, 66 and 33 raises
null energy from approximately zero to 0.88%, 37.73% and 71.50%. Pseudoinverse
NRMSE instead rises to 134.20 at 102 channels and then falls to 1.76 and 0.87:
dropping unstable directions can reduce variance while losing more true signal.
This is not improved identifiability. Numerical-null and near-null fractions are
disjoint; a drop in the near-null fraction can reflect directions becoming null.

In the forward-dependent QR control, 100 MEG locations have condition number
39.95 and pseudoinverse NRMSE 0.278 at 20 dB, or 0.088 at 30 dB. This demonstrates
a possible stable starting condition but relies on favorable source selection;
it cannot establish reliable arbitrary whole-cortex activity reconstruction.

The figure is a wide report graphic (274 mm), not a journal submission layout.

```powershell
.\.venv\Scripts\python.exe scripts/prepare_nullspace_forward.py
.\.venv\Scripts\python.exe scripts/prepare_nullspace_hdeeg.py --jobs 1
.\.venv\Scripts\python.exe scripts/run_nullspace_study.py --case all
.\.venv\Scripts\python.exe scripts/plot_nullspace_study.py
```

Preparation needs the documented public MEG files and existing fsaverage assets;
the hdEEG preparation is offline and refuses to overwrite an existing export.
Outputs are in `generated_simulation_dataset/nullspace_study/`: complete sweep
CSVs, source/noise/baseline arrays, spectra, a null-perturbation example, protocols,
asset/code hashes, summaries and figures. Forward files and generated arrays are
local experiment outputs, not dependencies bundled with the clean code collection.

## Figure contract and limitations

The quantitative grid tests one claim: numerical-null energy and noisy reconstruction
error need not vary together. Source-count and channel-count panels show actual
null and near-null fractions; error panels retain the pseudoinverse error peak.
Python/matplotlib produces editable SVG/PDF and PNG. Plotted paths use the first
of three sensor orderings and the estimated baseline covariance; every ordering
and covariance comparison remains in the CSV. These are deterministic numerical
paths, not biological replicates, confidence intervals, or significance tests.

This pilot is limited to one noise realization, one temporal/source realization,
one MEG anatomy and one template EEG anatomy. Next validation should add multiple
noise/source seeds, measured baseline/empty-room data, head-position and conductivity
perturbations, and area-normalized nested cortical bases. Identifiability inside a
coarse assumed basis does not establish uniqueness in the full biological cortex.

## Sources

- [MNE testing sample assets](https://github.com/mne-tools/mne-testing-data/tree/33ef2a6f6345a1adf8764f8f63725f0ed9e4bb92/MEG/sample).
- [MNE covariance whitening](https://mne.tools/stable/generated/mne.cov.compute_whitener.html).
- [MNE Maxwell filtering](https://mne.tools/stable/generated/mne.preprocessing.maxwell_filter.html):
  preprocessing can limit rank well below physical channel count; this pilot does
  not apply Maxwell filtering.
