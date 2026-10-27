"""Resolve complete NMM trajectories; cropping and noise belong to the caller.

Preset labels describe parameter recipes, not a guarantee about every waveform.
Explicit bank IDs replay the stored recipe rather than changing its dynamics.
"""
from dataclasses import asdict, replace
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from Simulation_G2.g2_simulation_core import load_projection_matrix
from Simulation_G2.nmm_bank import load_bank_sample, projection_hash, read_bank
from Simulation_G3.atlas_profiles import get_atlas_profile
from Simulation_G3.g3_tvb_core import TVBSignalConfig, TVBSourceGenerator
from Simulation_G3.generate_nmm_bank import generate_bank


PRESETS = {
    'spike_origin': {
        'parameters': dict(signal_type='spike_jr', jr_background_A=3.25, jr_active_A=3.6,
                           jr_mu=.087, jr_noise_scale=.2757716446627535, coupling_strength=1.),
        'validation_status': 'short-run spike morphology checked in legacy998 and aal78',
    },
    'rest_control': {
        'parameters': dict(signal_type='resting_jr', jr_background_A=3.25, jr_active_A=3.25,
                           jr_mu=.087, jr_noise_scale=.2757716446627535, coupling_strength=1.),
        'validation_status': 'short-run model control; not validated as empirical resting EEG',
    },
    'spike_variant': {
        'parameters': dict(signal_type='spike_jr', jr_background_A=3.25, jr_active_A=3.5,
                           jr_mu=.083, jr_noise_scale=.41365746699413, coupling_strength=1.),
        'validation_status': 'single-seed legacy998 spike morphology checked',
    },
    'alpha_candidate': {
        'parameters': dict(signal_type='alpha_jr', jr_background_A=3.25, jr_active_A=3.35,
                           jr_mu=.10, jr_noise_scale=.8, coupling_strength=.8),
        'validation_status': 'candidate; actual frequency and waveform require measurement',
    },
}


def _verified_replay(bank_dir, sample_id, profile):
    bank = read_bank(bank_dir)
    expected = dict(atlas_id=profile['atlas_id'], n_regions=profile['n_regions'],
                    projection_sha256=projection_hash(load_projection_matrix(profile['projection_path'])))
    if any(bank[key] != value for key, value in expected.items()):
        raise ValueError('NMM bank atlas/projection differs from the selected atlas')
    return load_bank_sample(bank_dir, sample_id)


def _trajectory_id(config, profile):
    """Full cache identity; physical split grouping remains the bank's responsibility."""
    values = asdict(config)
    values.pop('sample_id')
    try:
        tvb_version = version('tvb-library')
    except PackageNotFoundError:
        tvb_version = 'not-installed'
    identity = dict(config=values, tvb_version=tvb_version, atlas_id=profile['atlas_id'],
                    projection_sha256=projection_hash(load_projection_matrix(profile['projection_path'])))
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode())
    module_dir = Path(__file__).parent
    paths = [module_dir / name for name in ('g3_tvb_core.py', 'tvb_jr.py', 'atlas_profiles.py',
                                            'generate_nmm_bank.py')]
    connectivity = Path(config.connectivity_path)
    paths.append(connectivity)
    if (connectivity.parent / 'manifest.json').exists():
        paths.append(connectivity.parent / 'manifest.json')
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return 'nmm_' + digest.hexdigest()


def resolve_trajectory(config: TVBSignalConfig, *, provider: str, atlas: str,
                       bank_dir: Path, sample_id: str | None = None):
    """Return regional source, zero-based clock and verified trajectory metadata.

    bank: requires an explicit stored ID. cache: an ID replays; otherwise the
    full generation config selects an immutable record. online: always generates
    in temporary storage and returns no dangling file/checksum fields. The
    producer handles simulation validation, provenance and group IDs in all
    generation modes. A bank supports one writer at a time.
    """
    if provider not in {'bank', 'cache', 'online'}:
        raise ValueError('provider must be bank, cache or online')
    if provider == 'bank' and sample_id is None:
        raise ValueError('bank provider requires an explicit sample_id')
    if provider == 'online' and sample_id is not None:
        raise ValueError('online provider does not accept a stored sample_id')
    profile = get_atlas_profile(atlas)
    bank_dir = Path(bank_dir).resolve()
    if sample_id is not None:
        return _verified_replay(bank_dir, sample_id, profile)
    if config.normalize is not False:
        raise ValueError('Unified NMM generation requires normalize=False; scale components together later')
    if config.reshape_to_994 != profile['reshape_to_994']:
        raise ValueError('reshape_to_994 differs from the selected atlas')
    TVBSourceGenerator._validate_config(config)
    TVBSourceGenerator._validate_regions(config.active_regions, profile['n_regions'])
    if isinstance(config.random_seed, bool):
        raise ValueError('random_seed must be an integer, not a boolean')
    # JSON 1 and Python 1.0 describe identical physics and must also store the
    # same provenance, not merely resolve to the same cache filename.
    float_fields = ('simulation_length_ms', 'warmup_ms', 'dt_ms', 'output_srate_hz',
                    'discard_ms', 'jr_background_A', 'jr_active_A', 'jr_mu',
                    'jr_noise_scale', 'coupling_strength')
    config = replace(config, **{name: float(getattr(config, name)) for name in float_fields
                                if getattr(config, name) is not None})
    connectivity = Path(config.connectivity_path or profile['connectivity_path']).resolve()
    if connectivity.read_bytes() != Path(profile['connectivity_path']).read_bytes():
        raise ValueError('Connectivity differs from the selected named atlas')
    config = replace(config, connectivity_path=str(connectivity))
    identifier = _trajectory_id(config, profile)
    config = replace(config, sample_id=identifier)
    plan = dict(atlas=atlas, projection_path=str(Path(profile['projection_path']).resolve()),
                samples=[asdict(config)])
    # One shared producer owns all numerical/provenance checks; temporary online
    # banks also exercise exactly the same stored-format validation as cache.
    with TemporaryDirectory(prefix='nmm_trajectory_') as temporary:
        root = Path(temporary)
        plan_path = root / 'plan.json'
        plan_path.write_text(json.dumps(plan, allow_nan=False), encoding='utf-8')
        target = root / 'bank' if provider == 'online' else bank_dir
        generate_bank(plan_path, target)
        source, times, entry = _verified_replay(target, identifier, profile)
    if provider == 'online':
        entry = {key: value for key, value in entry.items() if key not in {'source_file', 'source_sha256'}}
    return source, times, entry
