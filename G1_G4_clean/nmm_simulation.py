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
from Simulation_G2.g2_simulation_core import G2SourceSimulator, load_projection_matrix
from Simulation_G2.nmm_bank import prepare_bank_segment, projection_hash
from Simulation_G2.nmm_mixing import mix_sources, make_sensors, disjoint_baseline_start, project_baseline
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
    baseline_duration_s: float = 1.0
    trajectory_duration_s: float | None = None
    start_s: float = 0.0
    warmup_s: float = 1.0
    dt_ms: float = 0.5
    srate: float = 500.0
    random_seed: int = 42
    model_overrides: dict = field(default_factory=dict)
    source_snr_db: float | None = None
    background_bank_dir: str | None = None
    background_sample_id: str | None = None
    background_seed: int | None = None
    background_start_s: float = 0.0
    sensor_noise_type: str = 'iid_white'
    sensor_snr_db: float | None = 10.0
    sensor_seed: int | None = None
    average_reference: bool = True
    normalize_source: bool = True


def _seed(root, supplied, stream):
    return int(supplied) if supplied is not None else int(np.random.SeedSequence([root, stream]).generate_state(1)[0])


def _validate(config):
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
    def __init__(self, backend=None, *, channel_names=None):
        self._backend = backend
        self._channel_names = channel_names
        self._loaded_atlas = None

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
        if config.sensor_snr_db is not None and config.sensor_noise_type == 'real' and real_sensor_noise is None:
            raise ValueError('sensor_noise_type=real requires real_sensor_noise')
        if real_sensor_noise is not None and config.sensor_noise_type != 'real':
            raise ValueError('Use sensor_noise_type=real for supplied measurement noise')
        if real_sensor_noise is not None:
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
                grouping_scope='recording_id' if real_noise_recording_id else 'exact_aligned_input_content')
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
            foreground_vertex=foreground_vertex)
        sensor_arrays, sensor_meta = make_sensors(backend, arrays, srate=config.srate,
            sensor_noise_type=config.sensor_noise_type, sensor_snr_db=config.sensor_snr_db,
            sensor_seed=sensor_seed, average_reference=config.average_reference,
            real_sensor_noise=real_sensor_noise, baseline_duration_s=config.baseline_duration_s,
            baseline_source=None if baseline_region is None else project_baseline(
                baseline_region, backend.projection_matrix, source_meta), baseline_metadata=baseline_metadata)
        arrays.update(sensor_arrays)
        arrays['time_ms'] = np.arange(foreground.shape[1]) * 1000 / config.srate
        parents = [entry['simulation_group_id']]
        if background_entry:
            parents.append(background_entry['simulation_group_id'])
        manifest = dict(generator_type='Unified_G2_G3_NMM', config=asdict(config),
            sample_id=config.sample_id, atlas_id=profile['atlas_id'],
            region_names=entry['manifest'].get('region_names'),
            configured_regions=configured, projected_regions=selected if config.source_mode == 'selected' else list(range(profile['n_regions'])),
            requested_preset=config.preset,
            preset_used_for_generation=config.provider == 'online' or (config.provider == 'cache' and config.nmm_sample_id is None),
            requested_preset_validation_status=PRESETS[config.preset]['validation_status'],
            foreground_trajectory=entry, background_trajectory=background_entry,
            parent_simulation_group_ids=parents,
            parent_data_group_ids=parents + ([real_noise_provenance['parent_group_id']] if real_noise_provenance else []),
            split_policy='Keep all samples sharing ANY foreground/background/real-noise parent in the same split',
            real_noise_provenance=real_noise_provenance,
            source_mode=config.source_mode, parameter_preset_is_activation_truth=False,
            background_policy='independently_seeded_all_region_rest' if background_entry else 'none_added',
            channel_names=self._channel_names,
            source_unit='normalized_model_state' if config.normalize_source else 'mV_PSP',
            **spatial_meta, **source_meta, **sensor_meta)
        with np.errstate(over='raise', invalid='raise'):
            arrays = {key: value if key in {'time_ms', *COVARIANCE_KEYS}
                      else value.astype(np.float32) for key, value in arrays.items()}
        return {**arrays, 'manifest': _json_safe(manifest)}

    def generate_sensor(self, config, **kwargs):
        sample = self.generate_sample(config, **kwargs)
        # Keep the five-return API while making whitening data available to callers.
        metadata = {**sample['manifest'], **{key: sample[key].tolist()
                    for key in NOISE_ARRAY_KEYS}}
        return tuple(sample[key] for key in ('sensor', 'clean_sensor', 'sensor_noise', 'source')) + (metadata,)

    def save_sample(self, output_dir, config, **kwargs):
        _validate(config)
        directory = Path(output_dir) / config.sample_id
        if directory.exists():
            raise FileExistsError(f'Sample directory already exists: {directory}')
        sample = self.generate_sample(config, **kwargs)
        directory.mkdir(parents=True)
        manifest = sample.pop('manifest')
        filename = config.sample_id + '_data.npz'
        np.savez_compressed(directory / filename, **sample)
        manifest.update(arrays={key: list(value.shape) for key, value in sample.items()}, data_file=filename)
        (directory / (config.sample_id + '_manifest.json')).write_text(
            json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
        return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='JSON NMMSimulationConfig or a list of configs')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--baseline-duration-s', type=float,
                        help='Override baseline duration for every JSON recipe (recipe default: 1 s)')
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
    if any((args.output_dir / c.sample_id).exists() for c in configs):
        raise FileExistsError('A batch output sample already exists')
    simulator = NMMSourceSimulator()
    for config in configs:
        result = simulator.save_sample(args.output_dir, config)
        print(json.dumps({key: result.get(key) for key in
            ('sample_id', 'atlas_id', 'realized_source_snr_db', 'realized_sensor_snr_db')}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
