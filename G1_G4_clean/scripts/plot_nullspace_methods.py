"""Report figures from validated aggregate CSV; all numbers remain in source CSV."""
from pathlib import Path
import argparse
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT / 'generated_simulation_dataset/nullspace_study/methods_supports')
    args = parser.parse_args()
    with (args.directory / 'aggregate.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 8,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    supports = ['points', 'patch32', 'green32', 'lbo32', 'patch_green64', 'hybrid64',
                'green_full400', 'local_multiscale1200', 'local_green1600', 'hybrid64_bands']
    labels = ['Points', 'Patch32', 'Green32', 'LBO32', 'Patch+Green64', 'LBO+Green64',
              'Full Green400', 'Local1200', 'Local+Green1600', 'Hybrid bands']
    scenes = ['local_patch', 'lbo_matched', 'lbo_mismatch', 'mixed', 'propagating']
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 7), layout='constrained')
    for i, kind in enumerate(('meg', 'hdeeg')):
        for j, density in enumerate(('full', 'reduced')):
            selected = [r for r in rows if r['part'] == 'supports' and r['covariance'] == 'baseline'
                        and r['modality'] == kind and r['method'] == 'sbl'
                        and float(r['nominal_full_array_snr_db']) == 20
                        and (int(r['n_channels']) > 200 if density == 'full' else int(r['n_channels']) < 100)]
            matrix = np.array([[float(next(r for r in selected if r['scene'] == scene and r['support'] == support)['null_recovery_mean'])
                                for support in supports] for scene in scenes])
            ax = axes[i, j]
            artist = ax.imshow(np.clip(matrix, -1, 1), cmap='RdBu', norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), aspect='auto')
            for y in range(len(scenes)):
                for x in range(len(supports)):
                    value = matrix[y, x]
                    ax.text(x, y, f'{value:.2f}', ha='center', va='center', fontsize=7,
                            color='white' if abs(value) > .65 else 'black')
            ax.set_xticks(range(len(supports)), labels, rotation=48, ha='right')
            ax.set_yticks(range(len(scenes)), scenes)
            ax.set_title(f'{chr(97 + 2*i+j)}  {kind.upper()}: {selected[0]["n_channels"]} channels, 400 sources', loc='left')
    fig.colorbar(artist, ax=axes, shrink=.7, label='Null error reduction (color clipped to [-1, 1])')
    fig.suptitle('Support-dependent null recovery: matched priors help; mismatch can increase error', fontsize=12)
    fig.supxlabel('SBL; 20 dB full-array training SNR; 2 s baseline; mean of two source/noise seeds. Labels show unclipped values.', fontsize=8)
    fig.savefig(args.directory / 'support_null_recovery.png', dpi=300)
    fig.savefig(args.directory / 'support_null_recovery.svg')
    fig.savefig(args.directory / 'support_null_recovery.pdf')
    plt.close(fig)

    methods = ['mne', 'wmne', 'sloreta', 'lcmv', 'sbl', 'sbl_lcmv', 'sbl_lcmv_active']
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), layout='constrained')
    for ax, kind in zip(axes, ('meg', 'hdeeg')):
        for scene, marker in zip(('sparse', 'synchronous', 'distributed'), ('o', 's', '^')):
            selected = [r for r in rows if r['part'] == 'methods' and r['covariance'] == 'baseline'
                        and r['modality'] == kind and r['scene'] == scene
                        and float(r['nominal_full_array_snr_db']) == 20 and int(r['n_channels']) < 100]
            values = [float(next(r for r in selected if r['method'] == method)['shape_nrmse_mean']) for method in methods]
            ax.plot(range(len(methods)), values, marker=marker, label=scene)
        ax.set_xticks(range(len(methods)), ['MNE', 'wMNE', 'sLORETA', 'LCMV', 'SBL', 'SBL-cov\nLCMV', 'SBL-active\nLCMV'], rotation=35, ha='right')
        ax.set(ylim=(0, 1.05), ylabel='Scale-aligned shape NRMSE', title=f'{kind.upper()}: reduced array, 400 sources')
    axes[0].legend(frameon=False, fontsize=8)
    fig.supxlabel('One global scalar fitted to test truth for shape scoring only; it is not physical amplitude recovery. Mean of two seeds.', fontsize=8)
    fig.savefig(args.directory / 'method_shapes.png', dpi=300)
    fig.savefig(args.directory / 'method_shapes.svg')
    fig.savefig(args.directory / 'method_shapes.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
