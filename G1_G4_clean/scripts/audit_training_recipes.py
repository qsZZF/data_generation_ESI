"""Audit a fixed stratified recipe selection; failed samples are never replaced.

Example: python scripts/audit_training_recipes.py PATH --output-dir NEW_PATH --samples 240
Use --samples 0 for all recipes. Anatomy paths default to the frozen manifest.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Simulation_G1.training_recipes import RecipeDataset, _digest


def _category(stats, name, value):
    stats.setdefault(name, Counter())[str(value)] += 1


def _number(stats, name, value):
    if value is None or not np.isfinite(value):
        return
    values = stats.setdefault(name, dict(count=0, sum=0., min=float(value), max=float(value), histogram=Counter()))
    values['count'] += 1
    values['sum'] += float(value)
    values['min'] = min(values['min'], float(value))
    values['max'] = max(values['max'], float(value))
    # Fixed 5-unit bins: dB for SNR, cm2 for patch area. Bounds are in the keys.
    lower = float(5*np.floor(value/5))
    values['histogram'][f'[{lower:g},{lower+5:g})'] += 1


def _finalize(stats):
    for value in stats.values():
        if isinstance(value, dict) and 'sum' in value:
            value['mean'] = value.pop('sum') / value['count']
    return stats


def plan_statistics(dataset):
    """Stream metadata once; retain only compact support summaries and index strata."""
    manifest = dataset.manifest
    supports, identities, errors = {}, {}, []
    table = manifest['support_table']
    if _digest(dataset.path/table['file']) != table['sha256']:
        raise ValueError('Support table checksum mismatch')
    with (dataset.path/table['file']).open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['support_id'] in identities and identities[row['support_id']] != row['split']:
                errors.append(f"Support {row['support_id']} crosses splits")
            identities[row['support_id']] = row['split']
            supports[row['index']] = {key: row[key] for key in ('support_id', 'split', 'actual_area_mm2')}
    strata, stats, noise_groups = defaultdict(list), {}, {}
    n_rows = 0
    for shard in manifest['shards']:
        if _digest(dataset.path/shard['file']) != shard['sha256']:
            raise ValueError(f"Recipe shard checksum mismatch: {shard['file']}")
        shard_count = 0
        with (dataset.path/shard['file']).open(encoding='utf-8') as stream:
            for offset, line in enumerate(stream):
                row = json.loads(line)
                support = supports[row['support_index']]
                index = shard['start'] + offset
                if (support['support_id'], support['split']) != (row['support_id'], row['split']):
                    errors.append(f'Recipe {index} has inconsistent support/split identity')
                for group in row.get('real_noise', {}).get('parent_data_group_ids', []):
                    if group in noise_groups and noise_groups[group] != row['split']:
                        errors.append(f'Recipe {index} has noise ancestry crossing splits')
                    noise_groups[group] = row['split']
                config = {**manifest['base_config'], **row['config']}
                families = tuple(sorted({item[0] for item in config['waveform_params']}))
                key = (row['split'], row['difficulty'], row['spatial_profile'], row['waveform_arrangement'],
                       len(support['actual_area_mm2']), ','.join(families))
                strata[key].append(index)
                for name, value in zip(('split', 'difficulty', 'spatial_profile', 'waveform_arrangement',
                                        'source_count', 'waveform_family_combination'), key):
                    _category(stats, name, value)
                for family in families:
                    _category(stats, 'waveform_family', family)
                for name in ('source_background_type', 'sensor_noise_type', 'sensor_noise_peak_ratio'):
                    _category(stats, name, config.get(name))
                for name in ('source_snr_db', 'sensor_snr_db'):
                    _number(stats, 'requested_'+name, config.get(name))
                for area in support['actual_area_mm2']:
                    _number(stats, 'sample_patch_area_cm2', area/100.)
                n_rows += 1
                shard_count += 1
        if shard_count != shard['count']:
            errors.append(f"Shard {shard['file']} count differs from manifest")
    if n_rows != manifest['n_samples']:
        errors.append('Recipe count differs from manifest')
    stats['noise_ancestor_groups_by_split'] = dict(Counter(noise_groups.values()))
    return strata, _finalize(stats), errors


def stratified_indices(strata, count, seed=42):
    """Cover marginal labels first, then round-robin strata; never use a prefix."""
    total = sum(map(len, strata.values()))
    if count == 0 or count >= total:
        return sorted(index for indices in strata.values() for index in indices)
    if count < 0:
        raise ValueError('Sample count must be nonnegative; zero selects all')
    rng = np.random.default_rng(seed)
    keys = list(strata)
    rng.shuffle(keys)
    pools = {key: rng.permutation(strata[key]).tolist() for key in keys}
    labels = {key: set(enumerate(key)) for key in keys}
    uncovered = set().union(*labels.values()) if labels else set()
    chosen = []
    while uncovered and len(chosen) < count:
        candidates = [key for key in keys if pools[key]]
        key = max(candidates, key=lambda item: len(labels[item] & uncovered))
        chosen.append(pools[key].pop())
        uncovered -= labels[key]
    while len(chosen) < count:
        for key in keys:
            if pools[key]:
                chosen.append(pools[key].pop())
                if len(chosen) == count:
                    break
    return chosen


def audit_dataset(dataset, output_dir, *, count=240, seed=42):
    from Simulation_G1.quality import assess_sample
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(recipe_manifest_sha256=_digest(dataset.path/'dataset.json'),
                   requested_sample_count=count, selection_seed=seed,
                   selection_policy='fixed marginal-coverage then round-robin strata; no replacement after failure',
                   errors=[], results_count={}, selected_distributions={})
    try:
        strata, summary['plan_distributions'], summary['errors'] = plan_statistics(dataset)
        indices = stratified_indices(strata, count, seed)
        summary['selected_indices'] = indices
        counts, selected = Counter(), {}
        with (output/'samples.jsonl').open('x', encoding='utf-8') as log:
            for index in indices:
                result = dict(recipe_index=index)
                try:
                    config, row = dataset.recipe(index)
                    context = {key: row[key] for key in ('support_id', 'split', 'difficulty',
                               'spatial_profile', 'waveform_arrangement', 'variant_index')}
                    context.update(source_count=config.n_patch, source_background_type=config.source_background_type,
                                   sensor_noise_type=config.sensor_noise_type,
                                   requested_sensor_snr_db=config.sensor_snr_db, requested_source_snr_db=config.source_snr_db)
                    result['recipe'] = context
                    for name, value in context.items():
                        if name not in ('support_id', 'variant_index'):
                            if name.startswith('requested_'):
                                _number(selected, name, value)
                            else:
                                _category(selected, name, value)
                    for kind in {item[0] for item in config.waveform_params}:
                        _category(selected, 'waveform_family', kind)
                    for vertices in config.patch_vertices:
                        area = dataset.simulator.vertex_area_coord2[vertices].sum()*dataset.simulator.coord_to_mm**2/100.
                        _number(selected, 'sample_patch_area_cm2', float(area))
                    sample = dataset[index]
                    result['quality'] = assess_sample(sample, cortex_mask=dataset.simulator.cortex_mask,
                        supports=config.patch_vertices, full_covariance=True)
                    result['status'] = result['quality']['status']
                    meta = sample['manifest']
                    result['measurements'] = {key: meta.get(key) for key in (
                        'realized_source_snr_db', 'realized_sensor_snr_db', 'realized_sensor_snr_reference_db',
                        'effective_target_global_snr_db', 'effective_target_active_snr_db',
                        'noise_peak_limit_applied', 'noise_peak_limit_scale')}
                    for key, value in result['measurements'].items():
                        if key.endswith('_db'):
                            _number(selected, key, value)
                        else:
                            _category(selected, key, value)
                except Exception as error:
                    result.update(status='failed', exception_type=type(error).__name__,
                                  exception=str(error), traceback=traceback.format_exc())
                counts[result['status']] += 1
                log.write(json.dumps(result, allow_nan=False)+'\n')
                log.flush()
        summary.update(results_count=dict(counts), selected_distributions=_finalize(selected),
                       support_splits_disjoint=not summary['errors'])
        summary['status'] = ('failed' if summary['errors'] or counts['failed'] else
                             'warning' if counts['warning'] else 'passed')
    except Exception as error:
        summary.update(status='failed', fatal_exception_type=type(error).__name__,
                       fatal_exception=str(error), traceback=traceback.format_exc())
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recipes', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=240, help='0 audits every recipe')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cortex', type=Path)
    parser.add_argument('--leadfield', type=Path)
    args = parser.parse_args()
    from Simulation_G1.g1_simulation_core import G1SourceSimulator
    from Simulation_G2.g2_simulation_core import load_brainstorm_cortex, load_brainstorm_leadfield
    manifest = json.loads((args.recipes/'dataset.json').read_text(encoding='utf-8'))
    resources = manifest.get('resource_paths', {})
    cortex = load_brainstorm_cortex(args.cortex or resources.get('cortex', ROOT/'Anatomy/Cortex_FS_20k.mat'))
    leadfield = load_brainstorm_leadfield(args.leadfield or resources.get('leadfield', ROOT/'Anatomy/Gain_20k_FS.mat'),
                                        n_vertices=np.asarray(cortex['Vertices']).shape[-1])
    simulator = G1SourceSimulator.from_brainstorm_cortex(leadfield, cortex,
        channel_names=manifest['assets'].get('channel_names'))
    dataset = RecipeDataset(args.recipes, simulator, split=None)
    report = audit_dataset(dataset, args.output_dir, count=args.samples, seed=args.seed)
    print(json.dumps({key: report.get(key) for key in ('status', 'results_count', 'fatal_exception')}, indent=2))
    return 1 if report['status'] == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
