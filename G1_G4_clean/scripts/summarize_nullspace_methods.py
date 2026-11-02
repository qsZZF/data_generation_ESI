"""Validate and summarize every completed method/support cell without selection."""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ('modality', 'part', 'covariance', 'scene', 'nominal_full_array_snr_db',
          'n_channels', 'n_sources', 'support', 'method')
METRICS = ('source_nrmse', 'shape_nrmse', 'power_cosine', 'truth_null_fraction',
           'inferred_null_fraction', 'null_cosine', 'null_recovery',
           'support_approximation_nrmse', 'test_snr_db', 'test_sensor_residual')


def summarize(directory):
    rows = []
    for kind in ('meg', 'hdeeg'):
        for part in ('methods', 'supports'):
            for covariance in ('baseline', 'reference'):
                with (directory / f'{kind}_{part}_{covariance}.csv').open() as handle:
                    rows.extend(csv.DictReader(handle))
    groups = defaultdict(list)
    invariant = defaultdict(list)
    for row in rows:
        assert int(row['rank']) + int(row['nullity']) == int(row['n_sources'])
        if row['source_nrmse']:
            error = float(row['source_nrmse']) ** 2
            parts = float(row['observable_error_fraction']) + float(row['null_error_fraction'])
            assert abs(error - parts) <= 1e-8 * max(1, error)
        if row['method'] == 'sloreta':
            assert all(not row[key] for key in ('source_nrmse', 'null_recovery', 'null_cosine', 'inferred_null_fraction'))
        groups[tuple(row[key] for key in GROUPS)].append(row)
        invariant[tuple(row[key] for key in GROUPS[:-2]) + (row['seed'],)].append(float(row['truth_null_fraction']))
    assert all(np.ptp(values) < 1e-12 for values in invariant.values())
    aggregated = []
    for key, values in groups.items():
        result = dict(zip(GROUPS, key))
        result['replicates'] = len(values)
        for metric in METRICS:
            numbers = [float(row[metric]) for row in values if row[metric]]
            for name, function in (('mean', np.mean), ('min', np.min), ('max', np.max)):
                result[f'{metric}_{name}'] = float(function(numbers)) if numbers else None
        aggregated.append(result)
    with (directory / 'aggregate.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregated[0]))
        writer.writeheader()
        writer.writerows(aggregated)
    sbl = [row for row in rows if row['method'] == 'sbl']
    unconverged = [{k: row[k] for k in GROUPS + ('seed', 'iterations')} for row in sbl if row['converged'] == 'False']
    summary = dict(rows=len(rows), aggregated_cells=len(aggregated),
                   source_error_decomposition_verified=True,
                   truth_null_fraction_independent_of_method_and_support=True,
                   sloreta_current_metrics_excluded=True,
                   sbl_records=len(sbl), sbl_records_not_converged=len(unconverged),
                   nonconverged=unconverged,
                   replicate_definition='Two source/noise seeds for baseline; one seed for reference-covariance control',
                   group_columns=GROUPS)
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({key: value for key, value in summary.items() if key != 'nonconverged'}, indent=2))
    return aggregated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT / 'generated_simulation_dataset/nullspace_study/methods_supports')
    args = parser.parse_args()
    summarize(args.directory)


if __name__ == '__main__':
    main()
