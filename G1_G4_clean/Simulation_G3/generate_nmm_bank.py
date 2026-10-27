"""Generate explicit, resumable G3 trajectories for the G2 bank interface.

Signal types are parameter presets, not detected or certified waveform classes.
Each sample retains the full coupled network after the configured warm-up.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from zipfile import ZipFile

import numpy as np

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Simulation_G2.g2_simulation_core import load_projection_matrix
from Simulation_G2.nmm_bank import (
    initialize_bank, load_bank_sample, projection_hash, read_bank, save_bank_sample,
)
from Simulation_G3.g3_tvb_core import (
    ROOT, TVBSignalConfig, TVBSourceGenerator, find_default_connectivity,
)
from Simulation_G3.atlas_profiles import ATLAS_CHOICES, get_atlas_profile, validate_atlas_connectivity, load_connectivity_metadata


def _sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _connectivity_regions(path: Path) -> int:
    """Inspect the standard TVB ZIP without importing or configuring TVB."""
    with ZipFile(path) as archive:
        arrays = {}
        for name in ('weights', 'tract_lengths'):
            with archive.open(name + '.txt') as stream:
                array = np.loadtxt(stream, ndmin=2)
            if (array.shape[0] != array.shape[1] or not array.size
                    or not np.isfinite(array).all() or np.any(array < 0)):
                raise ValueError(f'{path}: {name} must be a finite nonnegative square matrix')
            arrays[name] = array
        if arrays['weights'].shape != arrays['tract_lengths'].shape:
            raise ValueError(f'{path}: connectivity matrix dimensions differ')
        with archive.open('centres.txt') as stream:
            centres = np.loadtxt(stream, dtype=str, ndmin=2)
        n_regions = arrays['weights'].shape[0]
        if centres.shape != (n_regions, 4) or not np.isfinite(centres[:, 1:].astype(float)).all():
            raise ValueError(f'{path}: centres must have a label and three coordinates per region')
    return n_regions


def generate_bank(plan_path: str | Path, output_dir: str | Path | None = None,
                  *, dry_run: bool = False, atlas: str | None = None) -> dict:
    """Validate the whole plan, verify existing samples, then generate missing ones."""
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text(encoding='utf-8-sig'))
    if not isinstance(plan, dict):
        raise ValueError('Plan must be a JSON object')
    unknown = set(plan) - {'atlas', 'atlas_id', 'projection_path', 'defaults', 'samples'}
    if unknown:
        raise ValueError(f'Unknown plan fields: {sorted(unknown)}')
    if atlas is not None and plan.get('atlas') not in (None, atlas):
        raise ValueError('CLI atlas conflicts with the plan atlas')
    atlas_name = atlas or plan.get('atlas')
    profile = get_atlas_profile(atlas_name) if atlas_name is not None else None
    atlas_id = plan.get('atlas_id', profile['atlas_id'] if profile else 'legacy998_export994')
    if profile and atlas_id != profile['atlas_id']:
        raise ValueError('atlas_id does not match the named atlas')
    if not isinstance(atlas_id, str) or not atlas_id.strip():
        raise ValueError('atlas_id must be a nonempty string')
    defaults, samples = plan.get('defaults', {}), plan.get('samples')
    if not isinstance(defaults, dict) or not isinstance(samples, list) or not samples:
        raise ValueError('Plan needs a defaults object and nonempty samples list')
    projection_path = Path(plan.get('projection_path', profile['projection_path'] if profile else ROOT / 'Anatomy/FS_994nmm_to_20k.mat'))
    if not projection_path.is_absolute():
        projection_path = plan_path.parent / projection_path
    projection = load_projection_matrix(projection_path.resolve())
    projection.sum_duplicates()
    projection.eliminate_zeros()
    if (min(projection.shape) < 1 or not np.isfinite(projection.data).all()
            or np.any(np.diff(projection.indptr) == 0)):
        raise ValueError('Projection must be finite with a nonempty column for every region')
    identity = {'atlas_id': atlas_id, 'n_regions': int(projection.shape[1]),
                'projection_sha256': projection_hash(projection)}
    if profile:
        expected_projection = load_projection_matrix(profile['projection_path'])
        if identity['projection_sha256'] != projection_hash(expected_projection):
            raise ValueError('Projection differs from the named atlas; use a custom atlas_id without atlas')
    if atlas_id == 'legacy998_export994' and identity['n_regions'] != 994:
        raise ValueError('A non-994 projection requires an explicit, appropriate atlas_id')
    generator_digest = hashlib.sha256()
    for filename in ('g3_tvb_core.py', 'tvb_jr.py', 'atlas_profiles.py'):
        generator_digest.update(filename.encode())
        generator_digest.update((Path(__file__).parent / filename).read_bytes())
    try:
        tvb_version = version('tvb-library')
    except PackageNotFoundError:
        tvb_version = 'not-installed'

    prepared, connectivity_info, sample_ids = [], {}, set()
    for sample in samples:
        if not isinstance(sample, dict) or 'sample_id' not in sample:
            raise ValueError('Each sample must be an object with its own sample_id')
        atlas_defaults = ({'connectivity_path': str(profile['connectivity_path']),
                           'reshape_to_994': profile['reshape_to_994']} if profile else {})
        config = TVBSignalConfig(**{'normalize': False, **atlas_defaults, **defaults, **sample})
        if (not isinstance(config.sample_id, str)
                or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,99}', config.sample_id) is None
                or re.fullmatch(r'(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])', config.sample_id)
                or config.sample_id.casefold() in sample_ids):
            raise ValueError(f'Unsafe or duplicate sample_id: {config.sample_id!r}')
        sample_ids.add(config.sample_id.casefold())
        if not isinstance(config.normalize, bool) or not isinstance(config.reshape_to_994, bool):
            raise ValueError('normalize and reshape_to_994 must be JSON booleans')
        if not isinstance(config.active_regions, list) or isinstance(config.random_seed, bool):
            raise ValueError('active_regions must be a list and random_seed an integer')
        if config.coupling_strength is None:
            raise ValueError('coupling_strength must be finite')
        TVBSourceGenerator._validate_config(config)
        if config.signal_type not in TVBSourceGenerator.SUPPORTED_SIGNAL_TYPES:
            raise ValueError(f'Unsupported signal_type: {config.signal_type}')
        output_samples = ((config.simulation_length_ms - config.discard_ms)
                          * config.output_srate_hz / 1000.)
        if output_samples < 2:
            raise ValueError('Bank samples require at least two output samples')
        path = Path(config.connectivity_path) if config.connectivity_path else find_default_connectivity()
        if not path.is_absolute():
            path = plan_path.parent / path
        path = path.resolve()
        config.connectivity_path = str(path)
        if path not in connectivity_info:
            connectivity_info[path] = (_connectivity_regions(path), _sha256(path), load_connectivity_metadata(path))
        n_raw, connectivity_digest, connectivity_metadata = connectivity_info[path]
        if profile:
            validate_atlas_connectivity(profile, path)
        n_output = len(TVBSourceGenerator._output_region_indices(n_raw, config.reshape_to_994))
        if n_output != identity['n_regions']:
            raise ValueError(f'{config.sample_id}: output regions do not match projection')
        TVBSourceGenerator._validate_regions(config.active_regions, n_output)
        provenance = {'config': asdict(config), 'connectivity_sha256': connectivity_digest,
                      'generator_sha256': generator_digest.hexdigest(), 'tvb_version': tvb_version}
        if connectivity_metadata:
            provenance['connectivity_metadata_sha256'] = connectivity_metadata['manifest_sha256']
        prepared.append((config, provenance))

    output_dir = (Path(output_dir) if output_dir is not None
                  else ROOT / 'generated_simulation_dataset/NMM_bank').resolve()
    bank = read_bank(output_dir) if (output_dir / 'bank.json').exists() else None
    if bank is not None:
        for key, value in identity.items():
            if bank.get(key) != value:
                raise ValueError(f'Existing bank {key} does not match plan')
    reused, pending = [], []
    for config, provenance in prepared:
        if bank is not None and any(name.casefold() == config.sample_id.casefold() and name != config.sample_id
                                    for name in bank['samples']):
            raise ValueError(f'{config.sample_id}: case-insensitive sample ID collision')
        if bank is not None and config.sample_id in bank['samples']:
            _, _, entry = load_bank_sample(output_dir, config.sample_id)
            if entry.get('provenance') != provenance:
                raise ValueError(f'{config.sample_id}: existing provenance does not match plan')
            reused.append(config.sample_id)
        else:
            pending.append((config, provenance))
    result = {'output_dir': str(output_dir), **identity, 'dry_run': dry_run,
              'reused': reused, 'pending': [config.sample_id for config, _ in pending],
              'generated': []}
    if dry_run:
        return result
    initialize_bank(output_dir, **identity)
    for config, provenance in pending:
        print(f'Generating {config.sample_id} ({config.signal_type})', flush=True)
        source, times, manifest = TVBSourceGenerator(config.connectivity_path).generate(config)
        save_bank_sample(output_dir, config.sample_id, source, times, manifest, provenance)
        result['generated'].append(config.sample_id)
    result['pending'] = []
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--dry-run', action='store_true', help='Validate and report without simulation or writes')
    parser.add_argument('--atlas', choices=ATLAS_CHOICES, default=None)
    args = parser.parse_args(argv)
    print(json.dumps(generate_bank(args.plan, args.output_dir, dry_run=args.dry_run, atlas=args.atlas), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
