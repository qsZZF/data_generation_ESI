# Unified G3 generation / G2 replay

## Period index and training dataset (2026-10-24)

The offline path is now `G3 generation → training-bank conversion → period index
→ NMMRecipeDataset`. Indexing does not run TVB or duplicate trajectories. It uses
integer half-open sample intervals and supports fixed or sliding windows.
Index construction checks declared metadata; payload checksums and actual cortical
support compatibility are verified when records are first replayed.
`--tail-policy drop` reports unused tail samples; `error` rejects incomplete tails.
There is no implicit zero padding or concatenation across recordings. Strides
larger than a window are allowed and their uncovered gaps are reported separately.

```powershell
# Three existing AAL trajectories: 2s windows,1s stride,3 observation variants each.
python -m Simulation_G2.period_index --banks generated_simulation_dataset/NMM_training_views/aal78_256_v1/lowrank_k032 --output-dir generated_simulation_dataset/NMM_periods/my_network --period-s 2 --stride-s 1 --variants-per-period 3 --sensor-snrs 0 10 20 --sensor-noise-types iid_white specparam_resting

# Same-trajectory foreground/background repartitioning; this small pilot is training-only.
python -m Simulation_G2.period_index --banks generated_simulation_dataset/NMM_training_views/aal78_256_v1/lowrank_k032 --output-dir generated_simulation_dataset/NMM_periods/my_partition --signal-types spike_jr --source-mode partition --selection protected --source-snr-db 20 --spatial-profiles uniform decay diffuse pulse --period-s 2 --stride-s 1 --variants-per-period 4 --split-fractions 1 0 0
```

| Mode | Foreground / background |
|---|---|
| `network` | Complete recorded trajectory is the foreground; no spatial remapping. An independent resting bank may be added explicitly. |
| `selected` | Configured, protected or explicitly supplied IDs are the foreground. Other rows are discarded; an independent resting bank is optional. |
| `partition` | Selected rows are foreground; remaining rows of the same trajectory are background. Requires explicit source SNR; this is a reweighted synthetic view, not unchanged network physiology. |

`--selection protected` uses codec-protected IDs, falling back to configured IDs
for schema1. `--regions` overrides selection. `--sample-ids` and `--signal-types`
filter foreground records. Network mode requires uniform projection; selected
and partition modes support uniform/decay/diffuse/contract/pulse. Variant choices
are seeded draws from the requested lists, not a guarantee of exhaustive factorial
coverage. Dynamic envelopes restart at each period's first sample. Regional PSP
means are removed before dynamic weighting, and vertex means again before source
SNR calibration, so a constant baseline is not turned into artificial slow activity.

Add `--background-banks <paths> --source-snr-db 20` for independent resting NMM
background. Candidates must be unnormalized resting JR, without configured
high-excitability regions, with a different recorded seed and a distinct ancestry
component in the same split. A shared seed is rejected even when parameter values
differ. No suitable background means an explicit error, not borrowing from a
validation/test pool. The existing `aal_spike_20261016` and `aal_rest_20261016`
share a seed and cannot form such a pair; `--sample-ids aal_spike_20261017` can use
that resting record in a training-only smoke plan.

All selected inputs must share atlas, projection, region order/count, source unit
and sampling rate. Convert/resample first if they differ. Per-period source SNR
is optional except for partition or independent background modes. Use
`--no-sensor-noise` for clean observations; otherwise sensor SNR is global RMS
after reference. Real/PCA/residual noise banks are not part of this reader yet.

### Parent-aware splits

The default ratio is80/10/10 by connected parent component, not by window. A
component joins all records sharing any declared simulation or original-data
parent, including historical `original_provenance` conversion wrappers. Even
filtered-out records can connect ancestry. Original and compressed views therefore
remain together. Separate background components are allocated before pairing.
Every positive fold needs at least one component; small corpora must explicitly
choose training-only or fewer folds. Three components give one per fold and are
only an interface check, not a reliable generalization experiment.

The producer must declare direct shared data ancestry in `parent_data_group_ids`.
Different parameter draws from one fitted subject need that shared subject identity.
If different configurations intentionally share a stochastic drive for paired
experiments, give them a shared parent identity too; distinct simulation hashes
alone do not express that additional dependence. Learned model/PCA artifact IDs
belong in fit provenance, not as substitutes for direct reused sample parents.

### Random access, batches and cache

```python
from Simulation_G2.g2_simulation_core import G2SourceSimulator
from Simulation_G2.training_dataset import NMMRecipeDataset

bank = "generated_simulation_dataset/NMM_training_views/aal78_256_v1/lowrank_k032"
backend = G2SourceSimulator.from_paths(
    "Anatomy/Gain_20k_FS.mat", "Anatomy/Cortex_FS_20k.mat",
    "Anatomy/G4_AAL_cortical/projection.npz", bank,
)
data = NMMRecipeDataset("generated_simulation_dataset/NMM_periods/my_network",
                        backend, split="train", cache_bytes=256*1024**2)
sample = data[0]  # component arrays plus manifest
recipe = data.recipe(0)  # metadata only
for batch in data.iter_batches(4, shuffle=True, seed=42, workers=2, prefetch=2,
                              fields=("sensor", "source_foreground")):
    eeg, target = batch["sensor"], batch["source_foreground"]
    metadata = batch["manifests"]  # keep heterogeneous metadata as a list
    break
```

`source_foreground` is the localization target; `source` also includes modeled
background. Region and sensor components are separately available. Targets and
sensors use the same decoded approximation. Seeds are stored per recipe and do
not depend on worker count, access order or prefetch. `split=None` is for audits;
the default is training only. This is a NumPy map-style dataset; a framework
DataLoader should use a collator that preserves manifests instead of stacking
their heterogeneous contents.

`BankWindowReader` caches immutable encoded arrays under a byte budget and decodes
only the requested columns on hits. Cache misses still read/decompress an entire
NPZ payload. The budget is per dataset/worker, shared across its banks; it counts
retained array buffers, not all process memory. Concurrent replay and dense
cortical truth can consume much more RAM. Select only needed batch fields and
keep prefetch small. Readers reset on worker serialization/process changes.
Backend anatomical/leadfield arrays are expected to remain immutable.

### Verification and boundaries

Tests cover window tiling, overlapping windows, tail errors, transitive parent
splits, inherited conversion ancestry, background leakage rejection, deterministic
batches/prefetch, worker serialization, SNR and PSP-offset invariance. Actual
AAL78 lowrank banks produced81 network recipes,72 training-only partition recipes
and5 training-only independent-background recipes. Ten adjacent1s decoded windows
reassembled the original10s trajectory exactly. Threaded batches matched serial
outputs and source/sensor SNR checks passed. No new TVB simulations were needed.

Event-centered spike periods and oscillation-cycle segmentation remain specialized
analysis tasks rather than this fixed/sliding-window index. NPZ on-disk chunking,
direct lowrank-to-sensor projection and GPU generation remain performance work;
the implemented cache/reader is not a claim of maximum GPU training throughput.


## 256 Hz training views and fixed-k background storage (2026-10-24)

`python -m Simulation_G2.prepare_training_bank` converts existing regional banks
without running TVB or modifying the input. It creates separate schema2 banks
for dense float32, int16 background quantization and fixed-rank background SVD.
Schema1 remains supported. The default output rate is256Hz;200Hz is also supported.
This step changes the exported view, not the NMM integration timestep.

The entire retained recording is resampled with `resample_poly`, Kaiser5 window
and line boundary padding, before training windows are cropped. Downsampling may
change sharp peaks and removes frequencies outside the retained band; it is not
lossless relative to the original high-rate recording. Metadata preserves input
hashes, parent trajectory identity, clocks, resampling settings and code/library
versions. New conversions also report changes in protected-region peak amplitude
and dominant peak time; these are diagnostics, not an event-matching guarantee.

### Foreground protection

Protected IDs are the union of configured regions, previously protected IDs,
explicit `--protect-regions`, and optional event candidates. Default `auto`
detects positive peaks strictly above8mV only for raw `mV_PSP` records, at native
and output rates; it also conservatively protects threshold exceedances at the
record boundaries. `--protection configured` disables event additions;
`--protection events` requires PSP units. Adjust the threshold with
`--spike-threshold-mv`. The detector identifies protection candidates, not verified
physiological activity or causal origin.

Legacy G2 exports can have rescaled amplitudes: their unit must not be relabelled
as raw mV solely to apply the8mV threshold. Use configured/manual protection for
such exports. A bank writer always protects configured active IDs, even if they
were omitted from an explicit codec request.

All protected rows are stored in float32, exactly matching the canonical
**downsampled float32** reference. Background rows are stored using:

- `dense`: unchanged float32 arrays with NPZ compression;
- `int16`: per-row offset/scale and signed16-bit samples;
- `lowrank`: per-row mean plus `U[N_background,k]` and `C[k,T]`.

Fixed k is accepted without a retained-energy threshold. Only the dimensional
limit `min(N_background,T-1)` can reduce it; requested and stored ranks are both
recorded. All-protected records naturally have rank0. Sensor-domain noise is
unchanged by this codec: it can continue to be generated by seed or saved as a
complete normally compressed array.

### Build and replay

Run from the project root, using the same Python environment as the generator:

```powershell
# Compare codecs and k values, including current EEG leadfield and cortical-area QC.
python -m Simulation_G2.prepare_training_bank --input-bank generated_simulation_dataset/G3_AAL78_pilot/bank --output-dir generated_simulation_dataset/NMM_training_views/my_aal256 --ranks 8 16 32 --projection Anatomy/G4_AAL_cortical/projection.npz --leadfield Anatomy/Gain_20k_FS.mat --cortex Anatomy/Cortex_FS_20k.mat

# Produce only a fixed-k training bank, preserving configured and detected events.
python -m Simulation_G2.prepare_training_bank --input-bank generated_simulation_dataset/G3_spike_comparison/bank --sample-ids origin_noise --output-dir generated_simulation_dataset/NMM_training_views/my_fixed32 --codecs lowrank --ranks 32

# Replay using the existing G2 entry point; set output sampling rate explicitly.
python simulate_source_sensor.py --engine g2 --nmm-bank generated_simulation_dataset/NMM_training_views/my_fixed32/lowrank_k032 --nmm-sample origin_noise --nmm-source-mode network --srate-hz 256 --duration-s 1 --sample-id fixed32_replay
```

The output directory must be new. Each codec subdirectory contains a regular
`bank.json` and NPZ records, so `load_bank_sample` and existing G2 bank replay
decode transparently. `comparison.json` contains detailed QC; `comparison.csv`
contains a compact table. Supplying only `--input-bank/--output-dir` performs
source-space QC; sensor QC requires both projection and leadfield. Supply cortex
for area-weighted source-SNR calculations. `--no-average-reference` is available
for MEG or an explicitly supplied reference. Leadfield units/order remain the
caller's responsibility; the projection hash must match the bank.

Use these derived banks for replay. Do not point a resumable schema1 G3 writer
at them. Codec variants, resampling views and crops retain the parent's
`simulation_group_id`; keep the whole group within one dataset split.

Protection controls storage fidelity, not source selection. G2 `selected` mode
still follows the requested/configured IDs. To retain propagated activity, use
`network`, or explicitly select the additional protected IDs.

For old unpacked G2 clips, the existing bank API can first register the complete
regional clip with500Hz clock, its focus ID, projection hash, original-file hash,
and honest amplitude provenance. Use a non-JR model label for rescaled/unknown
amplitude units. One such `a10/nmm_1.mat` example was registered and converted in
the original workspace; bulk import of the historical archive is not automatic.

### How much background energy is necessary?

Training nuisance backgrounds need not retain99% of regional energy. What matters
is the intended background model, its spectrum/covariance after projection, and
its effect relative to the foreground at the chosen source SNR. However, a low
rank of `LW B` does not imply that the largest singular vectors of `B` are the
best directions for `LW B`. Regional SVD and sensor-weighted approximations have
different objectives. This implementation uses regional SVD and reports the
resulting sensor error; it does not optimize against a particular sensor layout.

QC compares the decoded background with the same256Hz dense reference, with
temporal means removed. It reports source and sensor relative errors,1–80Hz PSD
differences, sensor covariance error and the sensor rank needed for99% energy.
For records with nonzero protected foreground, it additionally reports errors
relative to the projected foreground at sourceSNR10/20/30dB, both with the same
background gain and after recalibrating the decoded background to the target SNR.
These SNR contexts are diagnostics only: conversion does not rescale the stored
bank to10/20/30dB. Apply the desired foreground/background mixing policy at training
time and record its realized SNR separately.
Rest-only records have no such foreground and explicitly lack this comparison.

The source-power metric is `W.T @ diag(vertex_area/sum(vertex_area)) @ W` when
cortex is supplied; otherwise it is mean regional power. Foreground here means
all protected rows. These comparisons use uniform external parcel projection,
not a dynamically changed spatial recipe. Other geometries, MEG operators and
spatial envelopes need their own validation.

### Initial actual-data results

All rows below compare to compressed dense NPZ at the same256Hz. Sensor errors
are relative Frobenius norms, after background SNR recalibration, using the current
64-channel EEG forward model and cortical-area weighting.

| Example, k=32 | Protected rows | File size / dense NPZ | Sensor error / foreground at20dB | At30dB |
|---|---:|---:|---:|---:|
| AAL78 spike,10s (`aal_spike_20261017`) | 6 | 54.6% | 5.61% | 1.77% |
| G3 legacy994 spike,10s (`origin_noise`) | 10:6configured +4additional candidates | 6.04% | 8.84% | 2.79% |
| Old G2 single-focus clip,1s (`legacy_r10_c1`) | 1 | 18.09% | 10.47% | 3.31% |

Int16 files are about58–62% of their dense NPZ references in these examples,
with much smaller waveform errors. Fixed k16/32 are practical *candidate training
background models*, particularly at high source SNR. They are not lossless
archives, and these tests do not establish ESI accuracy. Decode and use the same
approximate source for both training sensor synthesis and source labels; do not
pair compressed sensors with original dense source truth.

The current format deliberately retains per-record compressed NPZ. Cached encoded factors decode only the requested window; cold reads still decompress the complete payload. There is no on-disk time chunking or automatic codec choice.
Reported encode/decode timings are single local measurements, not a training
throughput benchmark. Keep dense or int16 holdout references to assess whether
training only on lowrank backgrounds makes the inverse overly specialized.


2026-10-20: Selected foreground supports must be disjoint. Diffusion, contraction and spread/recovery are documented in [spatial scenarios](../SIMULATION_GUIDELINE.md#g1g2-spatial-scenarios-2026-10-20). Network replay is unchanged.

G3 generates an offline NMM bank for explicit states, parameters, and seeds. G2
loads a saved trajectory, crops/resamples it in physical time, projects it to the
cortical mesh and sensors, and adds the requested sensor noise. Existing G2 spike
libraries and the on-demand `--engine g3` route remain available.

## Generate once

Prepare a JSON plan listing the simulations to generate. Each sample overrides
the defaults with fields of `TVBSignalConfig`; sample IDs must be unique. For
example, a minimal plan in the repository root is:

```json
{
  "atlas_id": "legacy998_export994",
  "projection_path": "Anatomy/FS_994nmm_to_20k.mat",
  "defaults": {
    "simulation_length_ms": 10000.0,
    "warmup_ms": 1000.0,
    "dt_ms": 0.5,
    "output_srate_hz": 500.0,
    "reshape_to_994": true,
    "normalize": false
  },
  "samples": [
    {"sample_id": "rest_seed1", "signal_type": "resting_jr", "active_regions": [], "random_seed": 1},
    {"sample_id": "alpha_r10_seed2", "signal_type": "alpha_jr", "active_regions": [10], "random_seed": 2},
    {"sample_id": "spike_r10_seed3", "signal_type": "spike_jr", "active_regions": [10], "random_seed": 3}
  ]
}
```

```powershell
.\.venv\Scripts\python.exe -m Simulation_G3.generate_nmm_bank --plan nmm_plan.json --output-dir generated_simulation_dataset/NMM_bank --dry-run
.\.venv\Scripts\python.exe -m Simulation_G3.generate_nmm_bank --plan nmm_plan.json --output-dir generated_simulation_dataset/NMM_bank
```

Paths in the plan resolve relative to that plan. Generation validates the plan
before simulation and resumes matching verified entries. Altered parameters,
projection, connectivity, or generator identity require a separate bank/sample
as indicated by validation errors. Full 998-node simulations can take substantial
time; dry-run checks the plan without running them.

JR simulation now defaults to `jr_coupling_mode: "sparse"`: it evaluates only
existing connections, retaining the weights, transmission delays, equations,
time step and random stream. Set `"dense"` in the plan (or
`--jr-coupling-mode dense` for direct G3 generation) to use the reference
implementation. Backend choice and actual noise `nsig` are recorded in the
manifest. Dense/sparse views share a simulation split group for the same seed
and model configuration.

For the six-focus spike comparison, use
`Simulation_G3/spike_comparison_plan.json`. Its regions were selected once with
seed 20261015 from eligible legacy-library regions. The plan compares the current
G3 noise setting with the stESI origin setting and one declared DeepSIF parameter
variant. G3's historical noise scale remains unchanged: `nsig[4] = (.05*scale)^2`.
To reproduce the stESI base intensity `.000190125`, use
`jr_noise_scale = 0.2757716446627535` (and multiply this by any origin noise
multiplier). This conversion affects model dynamics and is separate from the
equivalent coupling speed optimization. Legacy saved clips also include event
amplitude scaling, so raw PSP amplitude should not be interpreted as a direct
match to the saved library.

Each bank contains `bank.json` and per-sample NPZ arrays with regional source
signals and time coordinates. Provenance records parameters, seed, solver,
connectivity and projection identity. A `spike_jr` preset is a parameter setting,
**not a guarantee of a detected, centred or morphology-validated spike**. Event
detection, acceptance criteria and alignment are still needed for a spike-event
library equivalent to legacy G2 clips.

## Replay through G2

### Cortical AAL78 option

Use `"atlas": "aal78"` in a bank plan, or `--atlas aal78` at the bank CLI. This
selects the current cortical AAL projection and its78-node TVB graph together.
Then pass `--atlas aal78` when replaying through G2 or generating with high-level
G3. Region IDs refer to the AAL78 ordering, not the original994 library. Named
profiles validate projection identity and region order; custom plans can still
provide an explicit atlas_id/graph/projection without a named atlas selector.

The completed [AAL setup and commands](../Simulation_G3/README.md)
include three10s samples and real G2/G3 verification. Its tract lengths are
explicit centroid-distance proxies, not measured fibre lengths; see the report
before interpreting network timing. Omit the atlas selector to retain the old
default998-to994 route.

Preserve the coupled trajectory of the complete network:

```powershell
.\.venv\Scripts\python.exe simulate_source_sensor.py --engine g2 --nmm-bank generated_simulation_dataset/NMM_bank --nmm-sample alpha_r10_seed2 --sample-id alpha_network --duration-s 2 --nmm-start-s 1 --srate-hz 250 --sensor-snr-db 10 --plot
```

Omit `--nmm-sample` to sample reproducibly using `--random-seed`; use
`--nmm-state alpha_jr` to restrict the eligible state. `--source-type` is also
accepted as a state filter for bank replay; if both filters are supplied they
must agree. State filtering matches the saved `signal_type` exactly.

For a source-localization benchmark with only selected projected sources:

```powershell
.\.venv\Scripts\python.exe simulate_source_sensor.py --engine g2 --nmm-bank generated_simulation_dataset/NMM_bank --nmm-sample alpha_r10_seed2 --nmm-source-mode selected --active-regions 10 --sample-id alpha_selected --duration-s 1 --sensor-snr-db 5
```

`auto` defaults to `network` for banks. Network mode preserves all regional
trajectories, including their shared background and coupling relationships; it
does not apply selected-patch spatial decay. Explicit `--active-regions` requires
`selected` mode. Selected mode masks the saved trajectory to the chosen regions
and applies configured patch decay; without explicit IDs it uses the sample's
configured active-region IDs. **This is an output mask, not a rerun with other
regions switched off.** Configured active regions describe model perturbations,
not independently validated physiological activation.

G3 simulates and discards warm-up before saving the bank. G2 does not discard it
again: `--nmm-start-s` is relative to the already retained signal. Crop duration
must fit inside that recording. Output timestamps start at zero, and resampling
changes sampling density without stretching physical time. Bank duration is
variable; the legacy G2 MAT/ZIP contract remains a one-second clip at native
500 Hz, resampled if requested.

The bank defaults to unnormalized model signals (JR PSP in mV when supported by
the model metadata). The high-level G2 interface preserves those regional values
but centres and globally normalizes the projected vertex source for its existing
simulation/noise convention. Exported vertex/sensor amplitudes therefore are not
calibrated physical dipole currents or measured EEG volts. The manifest records
these transformations. `--keep-nmm-background` belongs only to legacy spike
replay; bank `network` mode already retains its complete coupled background.

Exports use the existing standard sample directory, with `region_source`,
`source`, `clean_sensor`, `sensor_noise`, `sensor`, `time_ms` and a JSON manifest.
No G4 model or dataset is changed by this integration.

Keep all crops and replay variants with the same `simulation_group_id` in one
train/validation/test partition. It identifies a seeded simulator configuration
independently of sample names and duration/resampling/normalization views. For
parameter-generalization experiments, also group equal parameter settings across
different noise seeds; the stored complete configuration permits that stricter
split. Legacy clips without their original run provenance do not acquire such a
guarantee retroactively.

This integration caches simulations; it does not speed up the underlying G3
solver. Long recordings still allocate the TVB Raw monitor and are loaded in
full before cropping through the legacy full loader. The period reader caches encoded arrays and decodes requested windows; NPZ cold reads still decompress the entire payload. Use one writer per bank and bounded recording durations.
The legacy 998-node-to-994-output anatomical mapping is preserved, not replaced
by a newly validated 994-node connectome.

## Validation

The bank/producer/replay regression tests are included. Historical generated
recordings and diagnostic logs are excluded from this code-only snapshot.
See [copy validation](../VALIDATION.md) and [external assets](../EXTERNAL_ASSETS.md).
