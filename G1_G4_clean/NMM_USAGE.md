# Unified G2–G3 neural mass simulation

## Per-sample baseline covariance for ESI whitening

Every generated sensor sample now supplies `baseline_sensor` (float32,
channels x baseline samples) and three float64 channel-by-channel matrices:

- `noise_covariance`: centered empirical covariance of the exported baseline,
  with denominator `B-1`. This is the default for whitening.
- `oracle_noise_covariance`: covariance of the known event background plus
  measurement noise. Diagnostic simulation truth only, not an inverse input.
- `sensor_noise_covariance`: covariance of the known event measurement noise,
  also diagnostic only.

The baseline contains no localization target. Its default duration is two seconds
(`baseline_duration_s`, or `--baseline-duration-s` where exposed). Baseline and
activity share the noise model, channel mixing/reference, and calibrated amplitude
scales. SNR is calibrated on the activity segment; the baseline is never separately
rescaled to force a desired noise power. Do not derive baseline covariance by
subtracting a known simulated target from an activity window.

G1 and synthetic measurement noise use disjoint intervals of a joint noise stream.
G2-G3 source backgrounds use a nonoverlapping segment of the same resting parent
trajectory. Online/cache generation reserves the extra duration; offline banks
must contain an unused baseline interval. Period recipes record `baseline_background`
and keep its parent in the same data split. Partition mode masks selected target
regions in that separate background interval. Insufficient background duration
raises an error instead of recycling event samples. Older period plans without a baseline reference can derive one deterministically
from unused samples of that same parent (prefer preceding, otherwise following);
insufficient duration raises an error. New plans store the reference explicitly.

Real noise input must contain enough consecutive samples for baseline plus activity;
it is not tiled. Source-only banks do not yet have a sensor montage: their baseline
covariance is formed at the sensor projection/mixing stage.

`generate_sample` and saved NPZ expose these arrays directly. G1 and the NMM
five-return `generate_sensor` API supply them as JSON-safe lists in metadata;
`save_sample` writes arrays to NPZ and replaces large JSON lists with file references.
Period `get_batch` / `iter_batches` include `noise_covariance` by default, shaped
batch x channels x channels. Use explicit `fields` to include `baseline_sensor`
or diagnostic covariance. Legacy G2/G3 APIs target the whole exported source,
so their baseline contains measurement noise only. Use the unified foreground/
background interface when a separate brain background is part of the scenario.

```python
from Simulation_G1.noise_covariance import whitener_from_covariance

C = sample['noise_covariance']
if sample['manifest']['noise_covariance_info']['rank']:
    W = whitener_from_covariance(C)  # retained_rank x channels
    sensor_white = W @ sample['sensor']
    leadfield_white = W @ referenced_leadfield
```

Use the sample's channel order, reference and forward scaling for the leadfield.
Stored activity is not automatically whitened. The helper removes eigenvalues below
`1e-7 * max_eigenvalue`, preserving the average-reference nullspace; zero covariance
means skip whitening, not multiply by a zero whitener. Baseline covariance is an
estimate and does not exactly whiten the held-out activity noise. Short, narrow-band
or nonstationary baselines can give uncertain estimates; no shrinkage or independent
baseline estimator is silently applied. This is spatial, not temporal whitening.

If channels/reference/filtering/scaling change, transform or re-estimate covariance
consistently. Dividing signals by a scalar s divides covariance by s**2. Keep baseline
out of supervised activity windows. G1 metadata-only recipes generate baseline at
load time and record seeds/configuration rather than storing every waveform in JSON.
New generator code changes the frozen recipe identity; use an old snapshot to replay
old recipes or create a new recipe collection.

G4 projected training draws save baseline and matrices in
`training_noise_covariance.npz`, after the same 1-40 Hz filter and before inverse
input z-scoring, with per-draw seeds, channels and calibration metadata. Clean
reconstruction currently adds no nuisance: it exports a zero baseline and zero
covariance explicitly marked uninformative. This does not estimate measured EEG
noise; data-driven PCA/residual backgrounds remain a separate planned step.
Existing saved samples are not automatically rewritten.

For offline training windows, use `Simulation_G2.period_index` and
`Simulation_G2.training_dataset.NMMRecipeDataset`. They add integer-sample periods,
parent-aware splits, bounded encoded caches and deterministic batch prefetch while
reusing the projection/SNR components below. See the
[period dataset guide](Simulation_G2/NMM_bank_usage.md#period-index-and-training-dataset-2026-10-24).
Dynamic selected-region projection now removes regional PSP baselines before
applying a time-varying envelope; a static offset therefore does not become a new signal.

Selected-region spatial envelopes are available through `spatial` and
`spatial_decay_mm`; defaults preserve the static projection. See the shared
[G1/G2 spatial guide](SIMULATION_GUIDELINE.md#g1g2-spatial-scenarios-2026-10-20)
for nonoverlap, diffusion/contraction, JSON configuration and SNR semantics.

`nmm_simulation.py` exposes one NMM interface with G1's five-return convention.
It reuses G3 for dynamics and bank generation, and G2 for anatomical projection
and measurement-noise construction. Existing G1/G2/G3 entry points remain valid.

```python
from nmm_simulation import NMMSimulationConfig, NMMSourceSimulator

simulator = NMMSourceSimulator()
config = NMMSimulationConfig(
    sample_id="aal_spike_with_rest",
    atlas="aal78",                       # also: legacy998 (998 raw -> 994 output)
    preset="spike_origin",
    provider="cache",
    active_regions=[25, 30, 42, 48, 52, 72],
    source_mode="selected",
    duration_s=2.0,
    trajectory_duration_s=4.0,             # generate once; crop/reuse for other views
    warmup_s=1.0,
    srate=500.0,
    random_seed=20261016,
    background_seed=20261018,
    source_snr_db=10.0,
    sensor_noise_type="iid_white",
    sensor_seed=20261019,
    sensor_snr_db=10.0,
)
sensor, clean, noise, source, manifest = simulator.generate_sensor(config)
```

Here `source` is the **total** cortical source, including the resting background;
`clean` is its projected EEG before added measurement noise. For focal ESI targets
or exact component supervision, use the richer result instead:

```python
sample = simulator.generate_sample(config)
target = sample["source_foreground"]
brain_background = sample["source_background"]
observed_eeg = sample["sensor"]

manifest = simulator.save_sample("generated_simulation_dataset/NMM_examples", config)
```

Every saved sample contains `region_foreground`, `region_background`,
`region_source`, `source_foreground`, `source_background`, `source`,
`foreground_sensor`, `background_sensor`, `clean_sensor`, `sensor_noise`,
`sensor`, `time_ms`, and a separate JSON manifest. `save_sample` refuses to
overwrite an existing sample directory. The five-return and rich methods each
run the requested operation; use one result when multiple arrays are needed.

## Providers and presets

| Provider | Behavior |
|---|---|
| `cache` | Default: generate missing trajectories, otherwise verify and reuse them. An explicit `nmm_sample_id` replays that record. |
| `bank` | Requires `bank_dir` and an explicit `nmm_sample_id`; uses an existing record without TVB integration. Source-background mixing additionally needs `background_sample_id`. |
| `online` | Always integrates a new trajectory through the same producer and verified temporary bank format; no persistent trajectory cache. |

Without `bank_dir`, the cache is `generated_simulation_dataset/NMM_unified/<atlas>`.
Cache identity includes model parameters, simulation timing/seed, connectivity,
atlas projection, implementation and TVB version. Integer/float spellings of
the same numerical configuration share a cache record. Source/sensor SNR and
sensor noise seeds do not trigger another NMM run. Changing actual dynamics,
retained trajectory duration or its output rate can create another record.

`start_s` selects the foreground crop; `background_start_s` selects the resting
crop. `duration_s` is the returned duration. If `trajectory_duration_s` is omitted,
the generated recording covers the requested crop. Warm-up is additional time
and is discarded before storing the recording.

For an explicit stored ID, the stored dynamics and provenance are authoritative;
`preset`/`model_overrides` do not retroactively change that record. The manifest
records whether the requested preset was actually used for generation. Region
masking is still available through `source_mode="selected"`.

| Preset | Purpose and status |
|---|---|
| `spike_origin` | Origin-noise JR spike recipe; short-run morphology checked on both atlases. |
| `rest_control` | Uniform-A JR parameter control; no configured high-A regions. Not certified as realistic measured resting EEG. Use network mode. |
| `spike_variant` | Lower A/mean drive parameter variant; previously checked on legacy998 only. |
| `alpha_candidate` | Existing alpha-oriented parameter recipe; actual spectrum requires evaluation. |

`model_overrides` accepts JR gains, mean drive, noise scale and coupling strength.
An overridden recipe needs its own waveform validation. Preset names are not
detected waveform labels. Old994 spike ZIPs remain supported by the legacy G2
entry point; this unified provider operates on the explicit G3 bank format.
The initial unified presets use JR observables with common PSP units. Other G3
model families remain accessible through their original entry points until their
observables and background-mixing units are explicitly aligned.

## Source-level SNR: independently seeded resting NMM

Let `S` be the projected foreground cortical activity and `B` an independently
seeded resting JR cortical record. Both are temporally demeaned. The mixture is

`X = S + alpha * B`

with

`alpha = sqrt(P_area(S) / (P_area(B) * 10**(source_snr_db/10)))`.

`P_area` is the temporal mean-square activity averaged over the **entire mesh**
using vertex-area weights, including zero foreground values outside the selected
patches. Calibration occurs after parcel-to-vertex projection, not in parcel
space. This makes the reference explicit despite different parcel sizes.

- The automatically generated background uses `rest_control`, with no perturbed
  regions, a different recorded simulation seed and the same atlas/clock.
- Background is present in all supported parcels, including foreground parcels;
  it retains its within-trajectory spatial relationships instead of assigning
  independent noise to every vertex.
- `source_mode="selected"` retains the requested regions from the coupled
  foreground trajectory. This is the useful default choice for controlled focal
  ESI experiments with an added resting background; it is explicit in examples.
- `source_mode="network"` retains the entire foreground network, including its
  endogenous background. In this mode source SNR means that full network versus
  the **additional** resting component; it is not focal-event/background SNR.
- `source_snr_db=None` disables the additional resting component.
- Existing background records must be raw resting JR records with no configured
  perturbed regions. Same-seed foreground/background histories are rejected.
- `normalize_source=True` applies one shared scale to both components after
  mixing. It preserves their SNR. Set it false to avoid this final common
  normalization. Raw unnormalized NMM trajectories remain in the bank.

This is a controlled additive mixture of two simulations, not a new integration
in which the added resting background is dynamically coupled back into the focal
network. Source SNR is a component-power ratio, not a causal parameter effect.

## Sensor-level SNR: separate measurement noise

After projection and the configured EEG reference, let

`Yclean = R L (S + alpha B)` and `Y = Yclean + beta R E`.

Measurement noise is calibrated **after** temporal demeaning and referencing:

`sensor_snr_db = 10 log10(P(Yclean) / P(beta R E))`.

The reference is global mean-square power over all channels and time points.
It is not a per-channel SNR, not clipped/robust power and not event-window-only
power. Zero-power inputs cannot satisfy a finite requested SNR and are rejected.
Use `sensor_snr_db=None` to disable added measurement noise.

Recommended uses:

1. **`iid_white` for a controlled baseline.** It draws independent Gaussian
   samples before reference. Average referencing induces channel dependence and
   rank loss, so the final referenced noise is not channel-independent.
2. **`pink`, `powerlaw` or `mixed` for additional robustness conditions.** These
   reuse the existing G2 implementations and their spatial mixing. They are
   synthetic perturbations, not a validated instrument-noise model.
3. **`real` for an explicitly identified multichannel recording.** Preserve its
   channel covariance by cropping all channels together and scaling once.

The historical G2 option called `white` includes filtering, drift and shared
components. It is retained for backward compatibility; `iid_white` is the new
unambiguous Gaussian baseline.

The manifest also reports `projected_source_snr_db` (projected foreground versus
projected resting background), and `effective_target_snr_db` (foreground versus
the actual sum of projected background and measurement noise). Neither is
generally equal to either requested SNR, because projection changes component
powers and the final nuisance contains both components.

For example, the measured AAL test with source10dB and sensor10dB gave projected
source SNR about11.14dB and effective target SNR about7.41dB. These are different
definitions, not a failed calibration. Both requested component SNRs were10dB.

## Supplied real noise

```python
from dataclasses import replace

real_config = replace(config, sensor_noise_type="real")
sample = simulator.generate_sample(
    real_config,
    real_sensor_noise=noise_channels_by_time,
    real_noise_srate=500,
    real_noise_channel_names=recorded_channel_names,
    real_noise_recording_id="recording_001",
)
```

Sampling rate must already match. Channel names must match the forward model;
the interface reorders them explicitly. The recording must cover the requested
duration: there is no silent tiling or resampling. A deterministic common crop
is selected using `sensor_seed`. Input/selected-segment hashes, crop offset,
sampling rate, channel order and recording identity are saved.

Use a stable `real_noise_recording_id` across excerpts of the same recording.
Without it, the fallback content hash only recognizes identical full inputs,
not different excerpts from one recording. A real resting EEG recording contains
brain activity as well: if used here alongside source-level rest, interpret it
as a composite sensor perturbation rather than pure instrument noise.

## Dataset generation and splitting

The CLI accepts one JSON configuration or a list of configurations:

```powershell
.\.venv\Scripts\python.exe -m nmm_simulation --config Simulation_G3/nmm_unified_example.json --output-dir generated_simulation_dataset/NMM_examples
```

The [example](Simulation_G3/nmm_unified_example.json) uses the tested AAL recipe.
Changing `sample_id`, `source_snr_db`, `sensor_snr_db`, `sensor_noise_type` and
`sensor_seed` builds different observations from cached NMM trajectories.
Call `save_sample` in a Python loop or provide a JSON list for a batch. This is
not yet an automatic parameter-search or morphology-balancing dataset builder.

`parent_simulation_group_ids` records both NMM parents; `parent_data_group_ids`
also includes a supplied real-noise recording. **Any samples sharing any parent
must remain in one split.** Allocate foreground/background/noise parents to
splits before mixing, or group the resulting sample-parent graph by connected
components. A hash of each foreground/background pair alone is insufficient.

Keep raw-trajectory morphology QC separate from augmented views: the8mV peak
detector belongs on the raw PSP bank, not on normalized or SNR-scaled arrays.

## Verification and implementation

Real AAL anatomy was exercised for a2x3 grid of source SNR0/10dB and sensor
SNR0/10/20dB. Both achieved SNRs were within1e-4dB of their targets after float32
export, while reusing exactly two underlying NMM records. Pink, mixed and supplied
synthetic matrix noise were also checked. Legacy998-to994 replay plus a newly
generated resting background passed the same source/sensor SNR checks. Real
online and cached AAL output arrays and parent groups were identical.

The supplied-noise path was tested using synthetic matrices, not a measured EEG
recording. Additional morphology classes and physiological validation remain out
of scope. [Numerical verification](generated_simulation_dataset/NMM_unified_validation/verification.json)
and [component preview](generated_simulation_dataset/NMM_unified_validation/component_preview.png).

The final full regression run passed **163 tests**, with installed-TVB and local
anatomy checks enabled. Compilation and the real JSON CLI example passed as well.

New implementation files are `nmm_simulation.py`, `Simulation_G3/nmm_sources.py`
and `Simulation_G2/nmm_mixing.py`, with corresponding tests. Existing G2 changes
are limited to the additional `iid_white` noise branch. No dependencies or G4
assets were added or changed. Banks remain single-writer; batch generation is
sequential. Cached dynamics still use the existing verified atlas conventions,
including the AAL centroid-distance delay approximation.
