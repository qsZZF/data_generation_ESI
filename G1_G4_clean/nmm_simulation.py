"""Unified G2/G3 API: NMM trajectories, independent resting background and EEG.

generate_sensor follows G1's five-return contract. generate_sample/save_sample
also expose the foreground and nuisance components needed for ESI supervision.
"""
from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re

import h5py
import numpy as np

from Simulation_G1.g1_simulation_core import _json_safe
from Simulation_G1.noise_covariance import baseline_sample_count, COVARIANCE_KEYS, NOISE_ARRAY_KEYS
from Simulation_G1.spatial import SpatialDynamicsConfig, resolve_spatial_config
from Simulation_G1.amplitude import validate_amplitude_settings
from Simulation_G1.real_noise import load_noise_window
from Simulation_G2.g2_simulation_core import G2SourceSimulator, load_projection_matrix
from Simulation_G2.nmm_bank import prepare_bank_segment, projection_hash, parent_group_ids
from Simulation_G2.nmm_mixing import (mix_sources, make_sensors, disjoint_baseline_start,
                                     project_baseline, apply_physical_source_amplitude, validate_sensor_snr_settings)
from Simulation_G3.atlas_profiles import get_atlas_profile
from Simulation_G3.g3_tvb_core import TVBSignalConfig, TVBSourceGenerator
from Simulation_G3.nmm_sources import PRESETS, resolve_trajectory

ROOT = Path(__file__).resolve().parent
MODEL_OVERRIDES = {'jr_background_A', 'jr_active_A', 'jr_mu', 'jr_noise_scale', 'coupling_strength'}
SENSOR_NOISE_TYPES = {'iid_white', 'white', 'pink', 'powerlaw', 'neurodsp_powerlaw',
                      'specparam_resting', 'knee_powerlaw', 'mixed', 'eegsourcesim_resting', 'real'}


@dataclass
class NMMSimulationConfig:
    sample_id: str = 'NMM_sample'
    atlas: str = 'legacy998'
    preset: str = 'spike_origin'
    provider: str = 'cache'
    bank_dir: str | None = None
    nmm_sample_id: str | None = None
    active_regions: list[int] | None = None
    n_active_regions: int = 6
    source_mode: str = 'network'
    spatial: SpatialDynamicsConfig = field(default_factory=lambda: SpatialDynamicsConfig(mode='static'))
    spatial_decay_mm: float | None = None
    duration_s: float = 2.0
    baseline_duration_s: float = 2.0
    trajectory_duration_s: float | None = None
    start_s: float = 0.0
    warmup_s: float = 1.0
    dt_ms: float = 0.5
    srate: float = 500.0
    random_seed: int = 42
    model_overrides: dict = field(default_factory=dict)
    source_snr_db: float | None = None
    source_snr_metric: str = 'vertex_mean'
    background_bank_dir: str | None = None
    background_sample_id: str | None = None
    background_seed: int | None = None
    background_start_s: float = 0.0
    sensor_noise_type: str = 'iid_white'
    sensor_snr_db: float | None = 10.0
    sensor_snr_mode: str = 'active_rms'
    sensor_snr_clip_quantile: float = .95
    sensor_seed: int | None = None
    real_noise_registry: str | None = None
    real_noise_record_id: str | None = None
    real_noise_split: str | None = None
    allow_combined_real_source_background: bool = False
    average_reference: bool = True
    normalize_source: bool = True
    covariance_estimator: str = 'empirical'
    amplitude_mode: str = 'relative'
    foreground_peak_am: float | None = None
    leadfield_source_unit: str | None = None
    sensor_units: str | list[str] | None = None
    sensor_noise_std: float | list[float] | None = None
    record_observability: bool = False
    observability_settings: dict = field(default_factory=dict)


def _seed(root, supplied, stream):
    return int(supplied) if supplied is not None else int(np.random.SeedSequence([root, stream]).generate_state(1)[0])


def _validate(config):
    validate_sensor_snr_settings(config.sensor_snr_mode, config.sensor_snr_clip_quantile)
    if not isinstance(config.allow_combined_real_source_background, bool):
        raise ValueError('allow_combined_real_source_background must be Boolean')
    if (config.real_noise_registry is None) != (config.real_noise_record_id is None):
        raise ValueError('real_noise_registry and real_noise_record_id must be supplied together')
    if config.real_noise_split is not None and (config.real_noise_registry is None or config.real_noise_split not in ('train', 'val', 'test')):
        raise ValueError('real_noise_split requires a registered train/val/test record')
    if config.real_noise_registry is not None and config.sensor_noise_type != 'real':
        raise ValueError('Registered noise requires sensor_noise_type=real')
    if config.real_noise_registry is not None and config.amplitude_mode == 'relative' and config.sensor_snr_db is None:
        raise ValueError('Registered noise in relative mode requires a finite sensor_snr_db')
    validate_amplitude_settings(amplitude_mode=config.amplitude_mode,
        foreground_peak_am=config.foreground_peak_am, leadfield_source_unit=config.leadfield_source_unit,
        sensor_units=config.sensor_units, sensor_noise_std=config.sensor_noise_std,
        sensor_snr_db=config.sensor_snr_db, normalize_source=config.normalize_source)
    if (config.real_noise_registry is not None and config.amplitude_mode == 'physical'
            and (config.sensor_noise_std is None or not np.any(np.asarray(config.sensor_noise_std) > 0))):
        raise ValueError('Registered physical noise requires at least one positive sensor_noise_std')
    if not isinstance(config.record_observability, bool):
        raise ValueError('record_observability must be Boolean')
    if (not isinstance(config.observability_settings, dict)
            or set(config.observability_settings) - {'numerical_rtol', 'near_null_rtol'}):
        raise ValueError('Unknown observability settings')
    if config.source_snr_metric not in ('vertex_mean', 'area_weighted'):
        raise ValueError('source_snr_metric must be vertex_mean or area_weighted')
    if config.covariance_estimator not in ('empirical', 'oas'):
        raise ValueError('covariance_estimator must be empirical or oas')
    baseline_sample_count(config.baseline_duration_s, config.srate)
    config.spatial = resolve_spatial_config(config.spatial)
    if config.source_mode == 'network' and config.spatial.mode not in {'static', 'auto'}:
        raise ValueError('Spatial dynamics require source_mode=selected; network trajectories remain unchanged')
    if config.spatial_decay_mm is not None and (not np.isfinite(config.spatial_decay_mm) or config.spatial_decay_mm < 0):
        raise ValueError('spatial_decay_mm must be finite and nonnegative or None')
    if (not isinstance(config.sample_id, str)
            or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,99}', config.sample_id) is None
            or re.fullmatch(r'(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])', config.sample_id)):
        raise ValueError('sample_id must be a safe filename identifier')
    if config.preset not in PRESETS or config.provider not in {'bank', 'cache', 'online'}:
        raise ValueError('Unknown NMM preset or provider')
    if config.source_mode not in {'network', 'selected'}:
        raise ValueError('source_mode must be network or selected')
    if config.sensor_noise_type not in SENSOR_NOISE_TYPES:
        raise ValueError('Unknown sensor_noise_type')
    if not isinstance(config.average_reference, bool) or not isinstance(config.normalize_source, bool):
        raise ValueError('average_reference and normalize_source must be booleans')
    for name in ('duration_s', 'srate', 'dt_ms'):
        value = getattr(config, name)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    samples = config.duration_s * config.srate
    if samples < 2 or not np.isclose(samples, round(samples), rtol=0, atol=1e-7):
        raise ValueError('duration_s must contain at least two whole output samples')
    for name in ('start_s', 'background_start_s', 'warmup_s'):
        value = getattr(config, name)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if config.trajectory_duration_s is not None:
        if not np.isfinite(config.trajectory_duration_s) or config.trajectory_duration_s < config.start_s + config.duration_s:
            raise ValueError('trajectory_duration_s must cover start_s + duration_s')
    for name in ('source_snr_db', 'sensor_snr_db'):
        value = getattr(config, name)
        if value is not None and not np.isfinite(value):
            raise ValueError(f'{name} must be finite or None')
    for name in ('random_seed', 'background_seed', 'sensor_seed'):
        value = getattr(config, name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 0 <= value < 2**32):
            raise ValueError(f'{name} must be an integer in [0, 2**32)')
    if config.random_seed is None:
        raise ValueError('random_seed is required for reproducible NMM simulations')
    if not isinstance(config.model_overrides, dict) or set(config.model_overrides) - MODEL_OVERRIDES:
        raise ValueError(f'model_overrides may contain only {sorted(MODEL_OVERRIDES)}')
    if not isinstance(config.n_active_regions, int) or isinstance(config.n_active_regions, bool) or config.n_active_regions < 0:
        raise ValueError('n_active_regions must be a nonnegative integer')
    if config.provider == 'bank' and not config.nmm_sample_id:
        raise ValueError('bank provider requires nmm_sample_id')
    if config.provider == 'bank' and config.source_snr_db is not None and not config.background_sample_id:
        raise ValueError('Bank-only source SNR requires background_sample_id')
    if config.source_snr_db is None and config.background_sample_id is not None:
        raise ValueError('background_sample_id requires source_snr_db')


class NMMSourceSimulator:
    def __init__(self, backend=None, *, channel_names=None, noise_reader=None):
        self._backend = backend
        self._channel_names = channel_names
        self._loaded_atlas = None
        self.noise_reader = noise_reader

    def _get_backend(self, profile, bank_dir):
        if self._backend is None or (self._loaded_atlas is not None and self._loaded_atlas != profile['atlas_id']):
            self._backend = G2SourceSimulator.from_paths(
                ROOT / 'Anatomy/Gain_20k_FS.mat', ROOT / 'Anatomy/Cortex_FS_20k.mat',
                profile['projection_path'], bank_dir)
            self._loaded_atlas = profile['atlas_id']
            with h5py.File(ROOT / 'Anatomy/Chanlocs.mat') as handle:
                self._channel_names = [''.join(chr(int(c)) for c in np.asarray(handle[ref]).flat)
                                       for ref in np.asarray(handle['chanlocs/labels']).flat]
        expected = load_projection_matrix(profile['projection_path'])
        if (self._backend.n_regions != profile['n_regions']
                or projection_hash(self._backend.projection_matrix) != projection_hash(expected)):
            raise ValueError('Backend projection does not match the configured atlas')
        return self._backend

    @staticmethod
    def _simulation_config(config, profile, *, background=False):
        preset = 'rest_control' if background else config.preset
        parameters = {**PRESETS[preset]['parameters'], **({} if background else config.model_overrides)}
        if background or preset == 'rest_control':
            active = []
        elif config.active_regions is not None:
            active = list(config.active_regions)
        else:
            if config.n_active_regions > profile['n_regions']:
                raise ValueError('n_active_regions exceeds the atlas size')
            rng = np.random.default_rng(np.random.SeedSequence([config.random_seed, 3]))
            active = sorted(rng.choice(profile['n_regions'], config.n_active_regions, replace=False).tolist())
        if not background and preset == 'rest_control' and config.active_regions:
            raise ValueError('rest_control has no configured high-excitability regions')
        TVBSourceGenerator._validate_regions(active, profile['n_regions'])
        offset = config.background_start_s if background else config.start_s
        duration = max(config.trajectory_duration_s or 0, offset + config.duration_s +
                       (config.baseline_duration_s if background else 0))
        return TVBSignalConfig(
            sample_id='rest_background' if background else config.sample_id,
            connectivity_path=str(profile['connectivity_path']), active_regions=active,
            simulation_length_ms=duration * 1000, warmup_ms=config.warmup_s * 1000,
            dt_ms=config.dt_ms, output_srate_hz=config.srate,
            random_seed=_seed(config.random_seed, config.background_seed, 1) if background else config.random_seed,
            reshape_to_994=profile['reshape_to_994'], normalize=False, **parameters)

    def generate_sample(self, config: NMMSimulationConfig, *, real_sensor_noise=None,
                        real_noise_srate=None, real_noise_channel_names=None, real_noise_recording_id=None):
        """Return component arrays plus a JSON-safe manifest (no hidden mutable result)."""
        _validate(config)
        profile = get_atlas_profile(config.atlas)
        bank_dir = Path(config.bank_dir) if config.bank_dir else ROOT / 'generated_simulation_dataset/NMM_unified' / config.atlas
        backend = self._get_backend(profile, bank_dir)
        sensor_seed = _seed(config.random_seed, config.sensor_seed, 2)
        real_noise_provenance = None
        if config.real_noise_registry is not None:
            if real_sensor_noise is not None or any(value is not None for value in
                    (real_noise_srate, real_noise_channel_names, real_noise_recording_id)):
                raise ValueError('Registered noise cannot be combined with bare real-noise arguments')
            if (self._channel_names is None or len(self._channel_names) != backend.n_sensors
                    or len(set(self._channel_names)) != len(self._channel_names)):
                raise ValueError('Registered noise requires forward-model channel names')
            read_noise = self.noise_reader.read if self.noise_reader is not None else load_noise_window
            real_sensor_noise, real_noise_provenance = read_noise(
                config.real_noise_registry, config.real_noise_record_id,
                expected_srate=config.srate, expected_channels=self._channel_names,
                expected_units=config.sensor_units if config.amplitude_mode == 'physical' else None,
                expected_reference='average' if config.average_reference else 'as_supplied',
                baseline_samples=baseline_sample_count(config.baseline_duration_s, config.srate),
                event_samples=round(config.duration_s * config.srate), seed=sensor_seed,
                split=config.real_noise_split)
            real_noise_provenance['identity_status'] = 'registered_verified'
            if config.amplitude_mode == 'relative' and len(set(real_noise_provenance['sensor_units'])) != 1:
                raise ValueError('Registered noise with mixed sensor units requires common-unit preprocessing or physical mode')
            if (config.source_snr_db is not None and real_noise_provenance['noise_role'] != 'empty_room'
                    and not config.allow_combined_real_source_background):
                raise ValueError('Real neural background plus source background requires allow_combined_real_source_background=True')
        if config.sensor_snr_db is not None and config.sensor_noise_type == 'real' and real_sensor_noise is None:
            raise ValueError('sensor_noise_type=real requires real_sensor_noise')
        if real_sensor_noise is not None and config.sensor_noise_type != 'real':
            raise ValueError('Use sensor_noise_type=real for supplied measurement noise')
        if real_sensor_noise is not None and config.real_noise_registry is None:
            if real_noise_srate != config.srate:
                raise ValueError('Real sensor noise must be supplied at the requested sampling rate')
            if self._channel_names is None or real_noise_channel_names is None:
                raise ValueError('Real sensor noise requires channel names to verify leadfield alignment')
            names = list(real_noise_channel_names)
            if len(names) != len(set(names)) or set(names) != set(self._channel_names):
                raise ValueError('Real noise channels must match the forward-model channels')
            real_sensor_noise = np.asarray(real_sensor_noise)
            if real_sensor_noise.ndim != 2 or real_sensor_noise.shape[0] != len(names):
                raise ValueError('Real noise must be channels x time and match its names')
            if np.iscomplexobj(real_sensor_noise) or not np.isfinite(real_sensor_noise).all():
                raise ValueError('Real sensor noise must contain finite real values')
            aligned = np.asarray(real_sensor_noise[[names.index(name) for name in self._channel_names]], dtype='<f8')
            event_count = int(round(config.duration_s * config.srate))
            baseline_count = baseline_sample_count(config.baseline_duration_s, config.srate)
            count = event_count + baseline_count
            if aligned.shape[1] < count:
                raise ValueError('Real noise must contain event plus baseline duration; it is not tiled')
            if real_noise_recording_id is not None and (not isinstance(real_noise_recording_id, str) or not real_noise_recording_id.strip()):
                raise ValueError('real_noise_recording_id must be a nonempty string')
            offset = int(np.random.default_rng(sensor_seed).integers(aligned.shape[1] - count + 1)) if aligned.shape[1] > count else 0
            real_sensor_noise = aligned[:, offset:offset + count].copy()
            digest = hashlib.sha256(aligned.tobytes()).hexdigest()
            group = hashlib.sha256(real_noise_recording_id.encode()).hexdigest() if real_noise_recording_id else digest
            real_noise_provenance = dict(input_sha256=digest, input_shape=list(aligned.shape),
                sampling_rate_hz=config.srate, aligned_channel_names=self._channel_names,
                start_sample=offset, n_samples=count,
                baseline_start_sample=offset, baseline_stop_sample=offset+baseline_count,
                event_start_sample=offset+baseline_count, event_stop_sample=offset+count,
                selected_sha256=hashlib.sha256(real_sensor_noise.tobytes()).hexdigest(),
                recording_id=real_noise_recording_id, parent_group_id='real_noise_' + group,
                grouping_scope='recording_id' if real_noise_recording_id else 'exact_aligned_input_content',
                identity_status='unverified_bare_array_reference_units_and_subject_not_registered')
        source_config = self._simulation_config(config, profile)
        if config.source_mode == 'selected':
            if config.active_regions is not None:
                backend._validate_selected_regions(config.active_regions)
            elif config.provider != 'bank' and config.nmm_sample_id is None and config.preset != 'rest_control':
                # A custom atlas can have overlapping columns; reject proposals before expensive simulation.
                rng = np.random.default_rng(np.random.SeedSequence([config.random_seed, 4]))
                for _ in range(64):
                    try:
                        backend._validate_selected_regions(source_config.active_regions)
                        break
                    except ValueError:
                        source_config.active_regions = sorted(rng.choice(
                            profile['n_regions'], config.n_active_regions, replace=False).tolist())
                else:
                    raise ValueError('Could not sample disjoint foreground regions; reduce n_active_regions or change the atlas')
        raw, times, entry = resolve_trajectory(source_config, provider=config.provider, atlas=config.atlas,
                                                bank_dir=bank_dir, sample_id=config.nmm_sample_id)
        if entry['source_unit'] != 'mV_PSP':
            raise ValueError('Unified JR source mixing requires unnormalized PSP recordings')
        foreground = prepare_bank_segment(raw, times, srate=config.srate,
                                          duration_s=config.duration_s, start_s=config.start_s)
        configured = list(entry['manifest']['active_regions'])
        selected = configured if config.active_regions is None else list(config.active_regions)
        TVBSourceGenerator._validate_regions(selected, profile['n_regions'])
        if config.source_mode == 'selected':
            if not selected:
                raise ValueError('selected mode needs region IDs; use network for the rest control')
            foreground[np.setdiff1d(np.arange(profile['n_regions']), selected)] = 0
        elif config.active_regions is not None and set(selected) != set(configured):
            raise ValueError('Region masking of a bank record requires source_mode=selected')
        foreground_vertex = None
        spatial_meta = {'spatial_sources': [], 'spatial_skip_reason': 'network preserves the original NMM trajectory'}
        if config.source_mode == 'selected':
            decay = config.spatial_decay_mm
            if decay is None and config.spatial.mode != 'static':
                decay = 12.
            centered = foreground.astype(np.float64) - foreground.mean(axis=1, dtype=np.float64, keepdims=True)
            projected, spatial_meta = backend.project_regions(
                centered, selected, srate=config.srate, time_window=(0., config.duration_s),
                decay_mm=decay, spatial=config.spatial)
            if config.spatial.mode != 'static' or (decay is not None and decay > 0):
                foreground_vertex = projected
        background, background_entry, baseline_region = None, None, None
        baseline_metadata = {}
        if config.source_snr_db is not None:
            background_config = self._simulation_config(config, profile, background=True)
            background_provider = 'bank' if config.background_sample_id else config.provider
            if background_provider == 'bank' and not config.background_sample_id:
                raise ValueError('Bank-only source SNR requires background_sample_id; cache can generate a resting draw')
            bg_raw, bg_times, background_entry = resolve_trajectory(background_config,
                provider=background_provider, atlas=config.atlas,
                bank_dir=Path(config.background_bank_dir) if config.background_bank_dir else bank_dir,
                sample_id=config.background_sample_id)
            bg_manifest = background_entry['manifest']
            model = bg_manifest.get('model', {})
            if (background_entry['source_unit'] != 'mV_PSP' or bg_manifest.get('signal_type') != 'resting_jr'
                    or bg_manifest.get('active_regions') or bg_manifest.get('normalize', True)
                    or model.get('active_A') is None or model.get('background_A') is None
                    or model.get('active_A') != model.get('background_A') or model.get('class') != 'JansenRit'):
                raise ValueError('Background must be a raw resting JR trajectory without perturbed regions')
            foreground_seed = entry['manifest'].get('random_seed')
            background_seed = bg_manifest.get('random_seed')
            if foreground_seed is None or background_seed is None or foreground_seed == background_seed:
                raise ValueError('Foreground and resting background require distinct recorded simulation seeds')
            if entry['simulation_group_id'] == background_entry['simulation_group_id']:
                raise ValueError('Foreground and background cannot share the same trajectory group')
            background = prepare_bank_segment(bg_raw, bg_times, srate=config.srate,
                                                duration_s=config.duration_s, start_s=config.background_start_s)
            # Establish disjointness before resampling, on the actual bank's native grid.
            native_rate = 1000 / (bg_times[1] - bg_times[0])
            baseline_count = baseline_sample_count(config.baseline_duration_s, native_rate)
            event_start = int(round(config.background_start_s * native_rate))
            event_stop = event_start + int(round(config.duration_s * native_rate))
            baseline_start = disjoint_baseline_start(bg_raw.shape[1], event_start, event_stop, baseline_count)
            baseline_region = prepare_bank_segment(bg_raw, bg_times, srate=config.srate,
                duration_s=config.baseline_duration_s, start_s=baseline_start/native_rate)
            baseline_metadata = dict(background_sample_id=background_entry['sample_id'],
                background_parent=background_entry['simulation_group_id'],
                background_start_sample=baseline_start, background_stop_sample=baseline_start+baseline_count,
                background_event_start_sample=event_start, background_event_stop_sample=event_stop,
                background_window_nonoverlapping=True, background_native_srate_hz=native_rate)
        arrays, source_meta = mix_sources(foreground, background, backend.projection_matrix,
            backend.vertex_area, config.source_snr_db, config.normalize_source,
            foreground_vertex=foreground_vertex, source_snr_metric=config.source_snr_metric)
        if config.amplitude_mode == 'physical':
            physical = validate_amplitude_settings(amplitude_mode='physical',
                foreground_peak_am=config.foreground_peak_am, leadfield_source_unit=config.leadfield_source_unit,
                sensor_units=config.sensor_units, sensor_noise_std=config.sensor_noise_std,
                sensor_snr_db=config.sensor_snr_db, normalize_source=config.normalize_source,
                n_channels=backend.n_sensors)
            apply_physical_source_amplitude(arrays, source_meta, config.foreground_peak_am)
            source_meta.update(physical,
                regional_output_unit='A*m_model_coefficients', native_temporal_model_unit=entry['source_unit'])
        sensor_arrays, sensor_meta = make_sensors(backend, arrays, srate=config.srate,
            sensor_noise_type=config.sensor_noise_type, sensor_snr_db=config.sensor_snr_db,
            sensor_snr_mode=config.sensor_snr_mode, sensor_snr_clip_quantile=config.sensor_snr_clip_quantile,
            sensor_seed=sensor_seed, average_reference=config.average_reference,
            real_sensor_noise=real_sensor_noise, baseline_duration_s=config.baseline_duration_s,
            baseline_source=None if baseline_region is None else project_baseline(
                baseline_region, backend.projection_matrix, source_meta), baseline_metadata=baseline_metadata,
            covariance_estimator=config.covariance_estimator, amplitude_mode=config.amplitude_mode,
            sensor_noise_std=config.sensor_noise_std, sensor_units=config.sensor_units,
            record_observability=config.record_observability, observability_settings=config.observability_settings)
        arrays.update(sensor_arrays)
        arrays['time_ms'] = np.arange(foreground.shape[1]) * 1000 / config.srate
        parents = [entry['simulation_group_id']]
        data_parents=set(parent_group_ids(entry))
        if background_entry:
            parents.append(background_entry['simulation_group_id'])
            data_parents.update(parent_group_ids(background_entry))
        if real_noise_provenance:
            data_parents.update(real_noise_provenance.get('parent_data_group_ids',[real_noise_provenance['parent_group_id']]))
        manifest = dict(generator_type='Unified_G2_G3_NMM', config=asdict(config),
            sample_schema='esi-sample-v1', supervision_target='source_foreground',
            source_total_definition='source_total = source_foreground + source_background; source is an alias of source_total',
            source_temporal_representation='per_vertex_event_mean_removed',
            sample_id=config.sample_id, atlas_id=profile['atlas_id'],
            region_names=entry['manifest'].get('region_names'),
            configured_regions=configured, projected_regions=selected if config.source_mode == 'selected' else list(range(profile['n_regions'])),
            requested_preset=config.preset,
            preset_used_for_generation=config.provider == 'online' or (config.provider == 'cache' and config.nmm_sample_id is None),
            requested_preset_validation_status=PRESETS[config.preset]['validation_status'],
            foreground_trajectory=entry, background_trajectory=background_entry,
            parent_simulation_group_ids=parents,
            parent_data_group_ids=sorted(data_parents),
            split_policy='Keep all samples sharing ANY foreground/background/real-noise parent in the same split',
            real_noise_provenance=real_noise_provenance,
            source_mode=config.source_mode, parameter_preset_is_activation_truth=False,
            target_definition='source_foreground is the localization target; source_total/source also include source_background',
            baseline_definition='separate nuisance-only sensor segment; no foreground target; same gains/reference/units as event',
            background_policy='independently_seeded_all_region_rest' if background_entry else 'none_added',
            channel_names=self._channel_names,
            source_unit='A*m' if config.amplitude_mode == 'physical' else 'normalized_model_state' if config.normalize_source else 'mV_PSP',
            **{**spatial_meta, **source_meta, **sensor_meta})
        with np.errstate(over='raise', invalid='raise'):
            arrays = {key: value if key in {'time_ms', *COVARIANCE_KEYS}
                      else value.astype(np.float32) for key, value in arrays.items()}
        if config.amplitude_mode == 'physical' and not np.isclose(
                np.max(abs(arrays['source_foreground'])), config.foreground_peak_am, rtol=5e-6, atol=0):
            raise ValueError('Physical foreground cannot be represented at export precision')
        arrays['source_total'] = arrays['source']
        manifest['array_units'] = dict(source_foreground=manifest['source_unit'], source_background=manifest['source_unit'],
            source_total=manifest['source_unit'], source=manifest['source_unit'],
            sensor=config.sensor_units if config.amplitude_mode == 'physical' else 'projected_relative_model_units',
            baseline_sensor=config.sensor_units if config.amplitude_mode == 'physical' else 'projected_relative_model_units')
        from Simulation_G1.quality import require_sample_quality
        sample={**arrays,'manifest':_json_safe(manifest)}
        supports=[item['vertex_indices'] for item in spatial_meta.get('spatial_sources',[])] or None
        require_sample_quality(sample,cortex_mask=np.asarray(backend.projection_matrix.getnnz(axis=1)>0),
                               supports=supports,full_source=False)
        return sample

    def generate_sensor(self, config, **kwargs):
        sample = self.generate_sample(config, **kwargs)
        # Keep the five-return API while making whitening data available to callers.
        metadata = {**sample['manifest'], **{key: sample[key].tolist()
                    for key in NOISE_ARRAY_KEYS}}
        return tuple(sample[key] for key in ('sensor', 'clean_sensor', 'sensor_noise', 'source')) + (metadata,)

    def save_sample(self, output_dir, config, *, resume=False, **kwargs):
        """Transactional save; resume regenerates/replays before verifying identity.

        Online generation is not skipped. This avoids claiming a pre-generation
        identity before the trajectory resolver creates or updates its bank.
        """
        from Simulation_G1.sample_io import save_sample_directory
        _validate(config)
        directory = Path(output_dir) / config.sample_id
        if (directory.exists() or directory.is_symlink()) and not resume:
            raise FileExistsError(f'Sample directory already exists: {directory}')
        sample = self.generate_sample(config, **kwargs)
        manifest = sample.pop('manifest')
        filename = config.sample_id + '_data.npz'

        def file_hash(path):
            path = Path(path)
            if not path.is_file():
                return None
            with path.open('rb') as stream:
                return hashlib.file_digest(stream, 'sha256').hexdigest()

        def array_hash(value):
            array = np.ascontiguousarray(value)
            digest = hashlib.sha256((array.dtype.str + str(array.shape)).encode())
            digest.update(array.tobytes())
            return digest.hexdigest()

        profile = get_atlas_profile(config.atlas)
        bank_dir = Path(config.bank_dir) if config.bank_dir else ROOT / 'generated_simulation_dataset/NMM_unified' / config.atlas
        resource_paths = [Path(profile['projection_path']), Path(profile['connectivity_path'])]
        # Bank indexes are appendable; bind only the selected immutable recordings.
        for key, trajectory_bank in (
                ('foreground_trajectory', bank_dir),
                ('background_trajectory', Path(config.background_bank_dir) if config.background_bank_dir else bank_dir)):
            entry = manifest.get(key)
            if entry and entry.get('source_file'):
                trajectory_path = (trajectory_bank / entry['source_file']).resolve()
                if (not trajectory_path.is_relative_to(trajectory_bank.resolve())
                        or file_hash(trajectory_path) != entry.get('source_sha256')):
                    raise ValueError('Selected NMM trajectory checksum changed before transactional save')
                resource_paths.append(trajectory_path)
        if config.real_noise_registry:
            resource_paths.append(Path(config.real_noise_registry))
            noise_provenance = manifest['real_noise_provenance']
            noise_path = Path(noise_provenance['input_path'])
            if file_hash(noise_path) != noise_provenance['file_sha256']:
                raise ValueError('Registered noise checksum changed before transactional save')
            resource_paths.append(noise_path)
        code_paths = [Path(__file__)] + [path for folder in ('Simulation_G1', 'Simulation_G2', 'Simulation_G3')
                                        for path in sorted((ROOT / folder).glob('*.py'))]
        request_identity = dict(config=_json_safe(asdict(config)),
            code={str(path.relative_to(ROOT)): file_hash(path) for path in code_paths},
            resources={str(path.resolve()): file_hash(path) for path in resource_paths},
            leadfield_sha256=array_hash(self._backend.leadfield), projection_sha256=projection_hash(self._backend.projection_matrix),
            vertex_areas_sha256=array_hash(self._backend.vertex_area),
            vertices_sha256=array_hash(self._backend.vertices), faces_sha256=array_hash(self._backend.faces),
            channel_names=_json_safe(self._channel_names),
            arguments={key: {'array_sha256': array_hash(value)} if isinstance(value, np.ndarray)
                       or (key == 'real_sensor_noise' and value is not None) else _json_safe(value)
                       for key, value in kwargs.items()},
            generated_manifest_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True, allow_nan=False).encode()).hexdigest(),
            arrays_sha256={key: array_hash(value) for key, value in sample.items()},
            verification_stage='after_generation_or_bank_replay')
        manifest.update(array_shapes={key: list(value.shape) for key, value in sample.items()}, data_file=filename,
                        resume_verification='post_generation; online simulation is not skipped')
        return save_sample_directory(output_dir, config.sample_id, sample, manifest,
                                     request_identity=request_identity, resume=resume)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='JSON NMMSimulationConfig or a list of configs')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true',
                        help='Verify existing samples after generation/bank replay; online simulation is not skipped')
    parser.add_argument('--baseline-duration-s', type=float,
                        help='Override baseline duration for every JSON recipe (recipe default: 2 s)')
    args = parser.parse_args(argv)
    configurations = json.loads(args.config.read_text(encoding='utf-8-sig'))
    if isinstance(configurations, dict):
        configurations = [configurations]
    override = {} if args.baseline_duration_s is None else {'baseline_duration_s': args.baseline_duration_s}
    configs = [NMMSimulationConfig(**{**values, **override}) for values in configurations]
    for config in configs:
        _validate(config)
    if len({c.sample_id.casefold() for c in configs}) != len(configs):
        raise ValueError('Batch sample IDs must be unique')
    if not args.resume and any((args.output_dir / c.sample_id).exists() for c in configs):
        raise FileExistsError('A batch output sample already exists')
    simulator = NMMSourceSimulator()
    for config in configs:
        result = simulator.save_sample(args.output_dir, config, resume=args.resume)
        print(json.dumps({key: result.get(key) for key in
            ('sample_id', 'atlas_id', 'realized_source_snr_db', 'realized_sensor_snr_db')}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
