# Geodesic patches, explicit amplitude units and observability labels

This 2026-11-01 extension implements the next three generator changes. It adds
no physiological amplitude distribution, new external dependency, or fitted
real-data model. The default baseline remains two seconds.

## 1. Cortical geometry

New G1 configurations default to `patch_geometry="geodesic"`. Both patch growth
and Gaussian decay use shortest paths over length-weighted mesh edges, restricted
by the cortical mask. This is a mesh-edge approximation, not an exact continuous
surface geodesic. Seed-spacing remains a separate Euclidean-distance constraint.

Growth retains the complete final equal-distance shell, preventing a cut based
on arbitrary vertex IDs. Consequently, requested area can be exceeded by the
final **shell** area, rather than necessarily one vertex. Actual area, shell area,
radius and metric are recorded. Supports remain connected and disjoint; paths
for explicit support decay may traverse valid cortex outside the support, but
never an excluded vertex.

Use `patch_geometry="legacy_bfs"` / `--patch-geometry legacy_bfs` to reproduce the
previous BFS/Euclidean geometry. A prechange fixed-seed support/weight fixture
passes exactly. Old G1 recipe snapshots still require their recorded generator
hash; do not bypass that guard. NMM parcel geometry is unchanged by this G1 patch
option; placing NMM trajectories into new subpatches remains a separate adapter task.

## 2. Relative versus physical amplitude

The existing `amplitude_mode="relative"` remains the amplitude default. It retains
requested SNR calibration and optional normalization. Physical mode requires:

```python
config = G1SimulationConfig(
    amplitude_mode="physical",
    leadfield_source_unit="A*m",  # only after checking the actual forward model
    sensor_units="V",           # or a per-channel list for mixed MEG types
    foreground_peak_am=1e-9,     # explicit example, not a physiological default
    sensor_noise_std=1e-6,       # native declared sensor units, before referencing
    sensor_snr_db=None,
    normalize_source=False,
    normalize_sensor=False,
    sensor_noise_peak_ratio=None,
    sensor_noise_type="iid_white",
    record_observability=True,
)
```

The source gain fixes the maximum absolute foreground vertex amplitude to the
requested dipole moment. The same gain applies to source background and its
baseline. Source-background SNR can still be specified; the fixed-noise contract
concerns **measurement noise**, not a separately calibrated biological background.
No density or vertex-area factor is silently added.

The leadfield must map A·m to its declared sensor units. Supported native units
are `V`, `uV`, `T`, `fT`, `T/m`, and `fT/cm`. `normalize_sensor_units` converts
forward rows to SI when useful. Declaring units does not verify a file's actual
calibration. The existing Brainstorm forward is not automatically relabeled SI;
G4 PSP remains mV model state until an explicit source-amplitude mapping is chosen.

IID measurement noise is scaled by its requested theoretical standard deviation,
independent of clean power. Colored or real-noise families use a single baseline-
estimated per-channel gain, shared by activity and baseline. No activity-dependent
noise calibration or post-hoc normalization is permitted in physical mode.
`sensor_noise_std=None` or zero means explicitly no measurement noise.

Raw mixed V/T/Tm powers, pooled PSDs and SNRs are not meaningful; those aggregate
fields are omitted or set to `None`. Use per-unit analysis or the explicitly
baseline-whitened power diagnostic. Average reference requires identical declared
sensor units; convert consistently before applying it.

`NMMSimulationConfig` exposes the same amplitude fields. Its regional ODE bank
remains unchanged; a shared gain maps the cropped regional/vertex components
to an explicitly requested current. Native temporal units are retained in metadata.

## 3. Sample observability labels

Set `record_observability=True` or CLI `--record-observability`. This is opt-in
because decomposing a large forward per changing noise covariance can be costly.
`observability_settings` accepts `numerical_rtol` and `near_null_rtol`, defaulting
to 1e-10 and 1e-3. No dense vertex-by-vertex projector is formed.

The returned `observability` metadata distinguishes:

| Field family | Meaning |
|---|---|
| `geometric_rank`, `geometric_nullity` | Numerical row space of the referenced forward, with row scaling to avoid sensor-unit bias |
| `geometric_observable_energy_fraction`, `geometric_null_energy_fraction` | Source-energy partition in the supplied coordinate system |
| `white_stable_energy_fraction`, `white_near_null_energy_fraction`, `white_null_energy_fraction` | Partition for the noise-retained whitened operator; not automatically the same null space |
| `lost_noise_directions` | Additional lost directions only when row-space nesting is verified |
| `isotropic_wiener_effective_dof`, `isotropic_wiener_weighted_energy_fraction` | Continuous isotropic-prior diagnostic; not rounded into a hard rank or treated as proven recoverability |
| `whitened_target_to_baseline_power_db` | Actual projected target power relative to unit noise after whitening with the total-nuisance baseline covariance |

Zero/no noise yields geometric labels but no fabricated noise-dependent values.
A zero source has no defined energy fraction. A conditioning cutoff is not an SNR
threshold. For G1 and unified NMM, the target is foreground excluding source
background; the legacy G2 interface labels the entire exported source.

G4 training/reconstruction also accepts `--record-observability`. Its labels are
explicitly for **regional PSP coefficients**, after the temporal observation
filter and before inverse z-scoring. No regional average reference is applied to
sources. Reconstruction includes its fitted observation gain. Metadata sets
`full_vertex_nullspace_evaluated=False`; these results cannot be presented as
full-cortex vertex null fractions. Clean reconstructed signals with zero added
noise receive no noise-power claim.

Each simulator caches at most two decompositions keyed by exact forward,
projection, covariance and tolerance hashes. Source-dependent values are always
recomputed. Runtime cache-hit flags are not saved in deterministic sample labels.

## Calling the new options

Relative-unit G1 on the current anatomy:

```powershell
python simulate_source_sensor.py --engine g1 --sample-id geodesic_labels_demo --source-type erp --n-active 4 --patch-area-mm2 180 --patch-geometry geodesic --covariance-estimator oas --record-observability --duration-s 1 --srate-hz 256 --output-dir generated_simulation_dataset/geometry_units_validation
```

For a **verified physical forward**, additionally provide `--amplitude-mode
physical --leadfield-source-unit "A*m" --sensor-units V --foreground-peak-am
1e-9 --sensor-noise-std 1e-6 --sensor-snr-db none`. The high-level G1 CLI disables
source normalization in physical mode. It does not infer calibration from the
forward filename. Physical NMM generation uses the explicit JSON contract through
`nmm_simulation.py`; the legacy high-level G2/G3 wrapper rejects unsupported
physical fields instead of ignoring them.

G1 recipe generation adds `--patch-geometry`, `--amplitude-config`, and
`--record-observability`. A physical amplitude JSON contains only the five
amplitude/unit fields above; the recipe builder disables SNR-driven measurement
noise, source normalization and peak caps for that explicit mode.

New NMM period plans use **`nmm-periods-v3`**, carrying amplitude and observability
settings explicitly. Existing v1/v2 plans remain supported with their recorded
relative/SNR semantics; v1 missing-metric fallback remains area weighted. Old
readers reject v3 rather than silently ignoring physical amplitude settings.
For physical period plans pass `--amplitude-config ... --sensor-snrs none
--no-normalize-source`. `--no-sensor-noise` cannot be combined with a nonzero
physical noise amplitude.

## Verification and measured limits

- Seven geodesic tests cover folded geometry, masking, connected/disjoint supports,
  equal-distance shell handling, relabeling and the legacy numeric fixture.
- Actual fs5 six-patch probe: 50–600 mm² requests, no excluded vertices or overlap;
  all area overshoots bounded by final-shell area, approximately 0.055 seconds.
- Actual 20,484-vertex / 64-channel G1 sample: four 180 mm² requests produced
  182.28/185.02/182.85/183.13 mm²; geometric, noise and whitened ranks were 63.
  Exact repeated arrays and labels passed. This small pilot took approximately
  0.505 seconds initially and 0.115 seconds on a repeated cached sample.
- Physical MNE MEG forward, 305 channels / 498 sources: changing peak current from
  1 to 2 nAm doubled the sensor target, left noise/baseline/covariance bitwise
  unchanged, and increased whitened target power by 6.0206 dB.
- The MEG noise in that check was synthetic IID noise using empirical marginal
  standard deviations. It did not reproduce the full measured noise covariance.

No physiological amplitude range was fitted, no measured EEG parameter inversion
was retrained, and no large production dataset was generated. Timings are short
local probes, not promised throughput or performance on every forward model.
