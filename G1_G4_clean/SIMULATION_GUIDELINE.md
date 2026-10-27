# Simulation Guideline

## Noise covariance supplied with every sensor sample

G1 and legacy G2 keep their five-return API. The returned metadata now contains
`baseline_sensor`, `noise_covariance`, `oracle_noise_covariance`,
`sensor_noise_covariance` and corresponding `*_info` records
(rank, sample count, reference and units). Default covariance is estimated only from a separately generated baseline.
G1 includes background activity in that baseline when configured; the event noise
covariance is retained only in the diagnostic oracle field. `save_sample` always writes a compressed
covariance NPZ referenced by `noise_covariance_file`, even when noise time series
are not saved. The unified high-level NPZ stores both matrices alongside signals.
G1 metadata-only recipe files remain compact: covariance is computed deterministically
when each recipe is generated, not stored in every recipe in advance.

See [covariance and whitening conventions](NMM_USAGE.md#per-sample-baseline-covariance-for-esi-whitening)
for G2–G3 batches, G4, rank-aware whitening, and matching the leadfield reference.
Previously frozen recipes retain their original generator snapshot; use that
snapshot to replay them, or generate a new recipe collection with the updated code.

This guideline describes the recommended high-level workflow for generating simulated source activity, sensor noise, and EEG/MEG-like sensor data in this repository.

## Recommended Entry Point

Use the high-level interface from the repository root:

```powershell
.\.venv\Scripts\python.exe simulate_source_sensor.py --engine g2 --sample-id demo_nmm_spike --active-regions 10 --clip-id 1 --plot
```

Each run creates one sample directory:

```text
generated_simulation_dataset/<sample_id>/
|-- <sample_id>_data.npz
|-- <sample_id>_manifest.json
`-- <sample_id>_overview.png        # only with --plot
```

The standard `.npz` arrays are:

| Array | Shape | Meaning |
| --- | --- | --- |
| `sensor` | channels x time | noisy EEG/MEG-like data |
| `clean_sensor` | channels x time | leadfield projection before sensor noise |
| `sensor_noise` | channels x time | generated sensor noise scaled to requested SNR |
| `source` | vertices x time | fsaverage20k vertex-level source truth |
| `region_source` | regions x time | region-level source truth, available for G2/G3 |
| `time_ms` | time | TVB time vector, available for G3 |

## Engine Selection

### `--engine g1`: mathematical source waveforms

Use G1 when you need controlled, interpretable source activity on fsaverage20k patches.

Recommended source types:

```text
erp
gabor_erp
spike
alpha_burst
resting_alpha
powerlaw
specparam_alpha
synaptic_noise
nonsinusoidal_alpha
```

Example:

```powershell
.\.venv\Scripts\python.exe simulate_source_sensor.py --engine g1 --source-type erp --sample-id g1_erp_demo --sensor-noise-type mixed --sensor-snr-db 8 --plot
```

Reasonable parameters:

| Parameter | Suggested range | Notes |
| --- | --- | --- |
| `--n-active` | 1-4 | Number of cortical source patches |
| `--patch-area-mm2` | 80-250 | Larger patches produce broader source spread |
| `--decay-mm` | 8-20 | Spatial falloff around patch seed |
| `--sensor-snr-db` | 0-20 | Lower values are harder/noisier |

### `--engine g2`: stESI NMM spike clips

Use G2 when you need the canonical NMM spike shape. This route reads already detected and cropped clips from `nmm_spikes/a*/nmm_*.mat`, then projects them through the fsaverage20k source model and leadfield.

Example:

```powershell
.\.venv\Scripts\python.exe simulate_source_sensor.py --engine g2 --sample-id g2_nmm_spike_demo --active-regions 10 --clip-id 1 --sensor-noise-type pink --sensor-snr-db 10 --plot
```

Recommended usage:

| Parameter | Suggested value | Notes |
| --- | --- | --- |
| `--active-regions` | e.g. `10` or `10,25` | Region IDs must exist under `nmm_spikes/a<id>/` |
| `--clip-id` | 1+ | Uses `nmm_<clip-id>.mat` |
| `--keep-nmm-background` | off by default | Keep it off for clean single-region source truth |

Do not use a short TVB run as an NMM spike clip. The correct stESI-style spike workflow runs a long Jansen-Rit simulation, detects valid local maxima away from boundaries, then crops a 500-sample event window. The G2 high-level path uses those cropped clips directly.

G2 clips contain one second at a native 500 Hz. Use `--duration-s 1`;
other durations are rejected rather than stretching the waveform. `--srate-hz`
resamples that same one-second interval and must produce an integer sample count.
The low-level `G2SourceSimulator` follows the same timing contract.

For multiple active regions, each active row comes from its own region-specific
clip. With `--keep-nmm-background`, the first selected clip supplies background
once, then all requested active rows are inserted. The manifest records this
policy. This is a composite of separate simulations, not a jointly simulated
multi-focus NMM network. Without that flag, unselected regional rows are zero.
High-level regional output preserves loaded amplitudes; cortical output is
centered and normalized. Low-level `normalize_region_signals=False` also preserves
loaded amplitudes; set `normalize_source=False` to disable cortical normalization.

### `--engine g3`: TVB neural mass source activity

Use G3 when you need continuous region-level neural mass activity from TVB.

Recommended source types:

```text
resting_jr
alpha_jr
oscillator
wilson_cowan
```

Example:

```powershell
.\.venv\Scripts\python.exe simulate_source_sensor.py --engine g3 --source-type oscillator --sample-id g3_osc_demo --duration-s 0.5 --warmup-ms 500 --sensor-noise-type white --sensor-snr-db 12 --plot
```

Recommended parameter policy:

| Source type | Role | Default policy |
| --- | --- | --- |
| `resting_jr` | Jansen-Rit baseline | `A=3.25`, `mu=0.087`, coupling `1.0` |
| `alpha_jr` | JR alpha-like activity | active `A=3.35`, `mu=0.10`, coupling `0.8` |
| `oscillator` | generic oscillator control | linear coupling `0.003` |
| `wilson_cowan` | excitatory/inhibitory dynamics | linear coupling `0.003` |

TVB simulations should include warm-up. Use at least:

```text
--warmup-ms 1000
```

for JR models. Short smoke tests can use `--duration-s 0.5`, but publication-quality sequences should use longer windows and inspect the generated overview.

Warm-up is already automatic: the default is **1000 ms of extra simulation**, and
those startup samples are removed before resampling, normalization, projection,
plotting, or saving. For example, `--duration-s 5` simulates six seconds and saves
only the final five seconds, with saved time starting at zero. At 500 Hz this is
2500 samples. No manual trimming is needed.

```powershell
.\.venv\Scripts\python.exe simulate_source_sensor.py --engine g3 --source-type alpha_jr --sample-id alpha_post_warmup --duration-s 5 --warmup-ms 1000
```

The TVB manifest records `simulated_duration_ms`, `discarded_initial_ms`, and
`retained_duration_ms`. In the high-level package these are under `tvb_manifest`.
The standalone G3 core also accepts `--discard-ms` for an additional trim after
warm-up; that option shortens its post-warm-up window. A fixed warm-up duration
does not automatically guarantee equilibration for every model parameter setting.

Explicit `--jr-active-A`, `--jr-mu`, `--jr-noise-scale`, and
`--coupling-strength` values override the source-type presets. The unused
`--oscillator-frequency-hz` option has been removed: the generic oscillator does
not provide a calibrated frequency control.

With the default 994-region export, `--active-regions` uses output-atlas IDs
0 through 993. Regions 7, 325, 921, and 949 map internally to TVB regions 994,
997, 996, and 995. Manifests retain both output and raw active IDs.
Warm-up and duration must be multiples of the integration step; retained duration
must yield an integer number of output samples. Exported `time_ms` starts at zero,
has spacing `1000 / srate_hz`, and excludes the duration endpoint.

TVB `spike_jr` output is continuous dynamics, not a validated detected spike clip.
The peak-centered demo is for inspection only. Finite short-run output does not
establish long-run stability or timestep convergence.

The JR presets use local TVB subclasses with `scipy.special.expit` for sigmoid
evaluation in both the model and coupling. TVB's broad default random initial
velocities can produce startup voltages large enough to overflow a direct
exponential even though the sigmoid and derivatives remain finite. Stable
evaluation preserves the equations, initialization, seed, and integration step;
it does not clip the state or suppress numerical warnings. Remaining JR integration
overflows raise an error with model and timestep context. Other model families
retain the caller's overflow policy; invalid operations fail in every family.
Warm-up is still necessary: a warning-free short segment may contain startup
relaxation. The manifest records `model.sigmoid_evaluation` for JR runs.

## Sensor Noise

Supported sensor noise types:

```text
white
pink
powerlaw
specparam_resting
mixed
eegsourcesim_resting
```

Recommended choices:

| Noise type | Use case |
| --- | --- |
| `white` | quick debugging and unit tests |
| `pink` | simple 1/f-like EEG background |
| `powerlaw` | stronger aperiodic background |
| `specparam_resting` | resting-like aperiodic + alpha peak |
| `mixed` | mixed broadband + aperiodic + resting-like noise |
| `eegsourcesim_resting` | EEGSourceSim-inspired resting noise |

Use `--sensor-snr-mode robust_global_rms` by default. It clips extreme sensor values before computing the noise scale and is less sensitive to spike peaks than plain RMS.

## EEG and MEG

The current bundled anatomy contains an EEG-style leadfield:

```text
anatomy/Gain_20k_FS.mat
```

The high-level interface accepts `--modality EEG` and `--modality MEG`, but the sensor physics are determined by the leadfield file. For MEG, pass a MEG leadfield explicitly:

```powershell
.\.venv\Scripts\python.exe simulate_source_sensor.py --engine g1 --modality MEG --leadfield-path path\to\MEG_gain.mat
```

Do not label data as MEG unless the leadfield is actually MEG.

Leadfield loading preserves the supplied forward matrix. Average referencing is
applied only for EEG when enabled, to both clean data and noise before SNR scaling.
MEG is never average-referenced. `--no-average-reference` preserves the reference
already present in the supplied leadfield; it cannot undo upstream referencing.
The manifest's `reference_mode` records the applied policy.

## Parameter Checks

Before using generated data for model training or benchmarking, inspect:

- `overview.png`: source event timing, clean projection, noise, and noisy sensor traces.
- `manifest.json`: shapes, selected regions, SNR, PSD summaries, TVB model parameters.
- `peak_is_boundary`: for NMM spike clips, this should be `false`.
- `realized_sensor_snr_reference_db`: should match the requested SNR using the selected reference-power definition after referencing.
- `realized_sensor_snr_db`: plain global-power SNR; it can differ when robust or active-window scaling is selected.

Run the focused regression checks with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The older `G3_optimized_plan.md` describes a future real-data-prior branch;
the current executable G3 engine is TVB. `G2_optimized_plan.md` references
`Simulation_NMMreal.m` and `SimulationDataset.py`, which are not shipped in the
current directory. The original NMM clip-generation provenance cannot be fully
reconstructed from these local scripts alone.

## Quick Commands

NMM spike with pink noise:

```powershell
.\.venv\Scripts\python.exe simulate_source_sensor.py --engine g2 --sample-id nmm_spike_r10 --active-regions 10 --clip-id 1 --sensor-noise-type pink --sensor-snr-db 10 --plot
```

ERP with mixed noise:

```powershell
.\.venv\Scripts\python.exe simulate_source_sensor.py --engine g1 --source-type erp --sample-id erp_mixed --sensor-noise-type mixed --sensor-snr-db 8 --plot
```

TVB oscillator:

```powershell
.\.venv\Scripts\python.exe simulate_source_sensor.py --engine g3 --source-type oscillator --sample-id tvb_osc --duration-s 0.5 --warmup-ms 500 --plot
```

TVB alpha-like JR:

```powershell
.\.venv\Scripts\python.exe simulate_source_sensor.py --engine g3 --source-type alpha_jr --sample-id tvb_alpha --duration-s 1.0 --warmup-ms 1000 --plot
```

## G1/G2 spatial scenarios (2026-10-20)

G1 now packs **complete, connected, disjoint foreground patches**. Seed separation
alone is insufficient: every proposed patch is checked against all accepted
vertices. Random sampling retries within a bounded budget; explicit overlapping
patches fail with an error. Requested areas that cannot fit are not silently
truncated. Decrease patch area/count/spacing if packing fails. With a fixed seed,
the new sampler is reproducible, but its samples can differ from older versions.

G2 uses actual nonzero parcel projection supports for the same ownership check.
Distinct region IDs alone do not guarantee distinct vertex supports in custom
soft atlases. Each selected region is weighted before summation. Background
components may intentionally overlap foreground, including for source SNR.

### Spatial weight profiles

Shared implementation: `Simulation_G1/spatial.py`; import `SpatialDynamicsConfig`
from either G1 or G2's public wrapper. Config field: `spatial`.

| Mode | Behavior |
|---|---|
| `static` | Fixed Gaussian center decay; decay 0 gives uniform weights |
| `auto` | Diffusion when actual patch area >= threshold **or** graph radius >= threshold; otherwise static |
| `diffuse` | Gaussian center decay gradually becomes uniform weight 1 |
| `contract` | Uniform weights gradually return to Gaussian center decay |
| `pulse` | Center decay → uniform → center decay during one transition interval |

Defaults in G1/G2 configs and `simulate_source_sensor.py`: `auto`, area threshold
**10 cm² = 1000 mm²**, hop threshold **5**, start **0 s**, duration **0.3 s**.
These are adjustable scenario defaults, not physiological thresholds. Radius is
the maximum shortest-path hop count inside the patch from its seed; in G2 the
seed is the vertex nearest the parcel centroid. A disconnected parcel records
`hop_order=null`, so only its area can trigger automatic diffusion. This can
occur with surface-mapped AAL parcels.

With baseline Gaussian weights `b`, the envelope is `w(t)=b+(1-b)*a(t)`. Diffusion
uses smoothstep `a=u²(3−2u)`, `u=clip((elapsed-start)/duration,0,1)`; contraction
reverses it, and pulse rises then falls. The schedule starts from the **first
exported sample**, including when an epoch has negative absolute time. Longer
transitions can be only partially observed in a short clip; metadata records
completion. Uniform spatial weights do not imply equal EEG channels or equal
time signals across independently driven patches.

Support stays fixed and disjoint throughout the sample. This is a phenomenological
amplitude-envelope model, not a cortical diffusion PDE or conduction-velocity
estimate. It does not expand beyond the reserved patch. Existing projection
coefficients still multiply G2 weights: a custom nonuniform projection remains
nonuniform even after the envelope reaches one. Decay 0 disables center contrast,
so an explicit transition with uniform baseline has no spatial effect.

### Commands and Python configuration

Run from the original root using its environment, or from the clean collection
using the same configured Python and explicitly supplied external assets.

```powershell
# Six disjoint G1 patches, each at least 12 cm²; large-patch diffusion is automatic.
python simulate_source_sensor.py --engine g1 --source-type erp --n-active 6 --patch-area-mm2 1200 --spatial-mode auto --diffusion-area-cm2 10 --diffusion-hops 5 --diffusion-duration-s 0.3 --sample-id g1_large_diffuse

# Legacy G2 clips: six random disjoint foreground parcels, expand then contract.
python simulate_source_sensor.py --engine g2 --n-active 6 --duration-s 1 --spatial-mode pulse --diffusion-start-s 0.1 --diffusion-duration-s 0.6 --sample-id g2_spatial_pulse

# Disable automatic spatial dynamics while retaining nonoverlapping foregrounds.
python simulate_source_sensor.py --engine g1 --source-type erp --n-active 4 --spatial-mode static --sample-id g1_static
```

`--diffusion-area-cm2 0` / `--diffusion-hops 0` disable the respective automatic
trigger. Python uses `None` for a disabled trigger. Explicit `diffuse`, `contract`
and `pulse` apply regardless of patch size. G2 legacy defaults still require its
local NMM spike library; the new flags do not download data. For generated-bank
G2, add `--nmm-bank <path> --nmm-source-mode selected` and optionally explicit
`--active-regions`; the configured bank sample determines the selected regions
when they are not supplied. `--n-active` controls random legacy selection, not
the active-region count of an already generated bank trajectory.

```python
from Simulation_G1.g1_simulation import G1SimulationConfig, SpatialDynamicsConfig

config = G1SimulationConfig(
    n_patch=6, patch_area_mm2=1200, random_seed=42,
    spatial=SpatialDynamicsConfig(
        mode="auto", area_threshold_cm2=10, hop_threshold=5,
        start_s=0.1, duration_s=0.4,
    ),
)
```

The unified `nmm_simulation.py` API retains its existing default **static,
untapered** selected-region projection; opt into the same shared profiles:

```python
from nmm_simulation import NMMSimulationConfig

config = NMMSimulationConfig(
    source_mode="selected", n_active_regions=6,
    spatial={"mode": "auto", "area_threshold_cm2": 10, "hop_threshold": 5,
             "start_s": 0, "duration_s": 0.3},
    spatial_decay_mm=12,
    source_snr_db=3, sensor_snr_db=10,
)
```

The same nested `spatial` mapping works in its existing JSON config input. Without
an explicit decay, unified non-static profiles use 12 mm; static keeps its old
no-decay behavior. Network replay preserves the complete coherent NMM trajectory:
`auto/static` do not spatially modify it, and explicit dynamic modes require
`selected`. G3 network generation likewise rejects these explicit spatial modes.
The old G2 `weight_decay_hops` field is retained only for compatibility and is not
an expansion control; use `spatial.hop_threshold` for automatic triggering.

### Truth, SNR and verification

Manifests save actual patch supports, area, graph radius, baseline weights and
the transition schedule/endpoints. G2 also saves projection coefficients.
The schedule describes weights **before** temporal demeaning and normalization;
the final source waveform may therefore have a different apparent endpoint ratio.
Dynamic vertex truth is not generally `projection @ region_source`; retain the
vertex output or replay the spatial metadata and normalization convention.

G1 now demeans its modulated foreground before background-SNR calibration; its
source-SNR convention remains whole-mesh mean vertex power. Unified G2–G3 uses
whole-mesh **area-weighted** component power after modulation/demeaning and then
adds sensor noise after projection/reference. These two source-SNR definitions
must not be treated as identical on an irregular mesh.

Coverage includes toy meshes with deliberate overlap, packing failures, fixed-seed
replay, area/hop triggers, envelope endpoints, source/noise composition, JSON/CLI
forwarding, and source/sensor SNR after modulation. The original workspace also
ran a small 20,484-vertex anatomy check: three six-patch G1 scenes (each patch
about12 cm²) had zero overlapping vertices, and both 994-region and AAL78 spatial
projections retained a requested3 dB source SNR. These checks used synthetic
regional traces for G2. Separate CLI checks also generated one G1 six-patch
scene and replayed six real legacy G2 spike clips with a pulse envelope; both
had zero foreground overlap and10 dB sensor SNR under the requested robust
reference definition. No new long NMM simulation runs were needed.
