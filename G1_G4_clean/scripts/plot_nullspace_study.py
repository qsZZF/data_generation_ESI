"""Plot the saved numerical pilot; no simulated observations are added here."""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path,
                        default=ROOT / 'generated_simulation_dataset/nullspace_study')
    args = parser.parse_args()
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 8,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'legend.frameon': False})
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.1), layout='constrained')
    colors = {'null': '#336699', 'near': '#C58843',
              'pinv': '#555555', 'tsvd': '#558DA6', 'ridge': '#A06B98'}
    for row, kind in enumerate(('meg', 'hdeeg')):
        summary = json.loads((args.directory / f'{kind}_summary.json').read_text())
        title = 'MEG sample geometry' if kind == 'meg' else 'Template hdEEG geometry'
        source_path = summary['full_array']
        channel_path = summary['channel_path_n100']
        for column, (records, field, xlabel) in enumerate((
                (source_path, 'n_sources', 'Source count (full sensor array)'),
                (channel_path, 'n_channels', 'Channel count (100 sources)'))):
            ax = axes[row, column]
            x = [record[field] for record in records]
            ax.plot(x, [100 * r['null_energy'] for r in records], 'o-',
                    color=colors['null'], label='Numerical null')
            ax.plot(x, [100 * r['near_null_energy'] for r in records], 's--',
                    color=colors['near'], label='Near-null (relative cutoff 0.001)')
            ax.set(xlabel=xlabel, ylabel='Source energy (%)', ylim=(-2, 102))
            if column == 1:
                ax.invert_xaxis()
            ax.set_title(f'{chr(97 + row * 3 + column)}  {title}', loc='left', fontsize=9)
        ax = axes[row, 2]
        for method in ('pinv', 'tsvd', 'ridge'):
            values = np.array([r[f'{method}_nrmse'] for r in channel_path])
            if np.any(values <= 0):
                raise ValueError('Log error plot requires positive errors')
            ax.semilogy([r['n_channels'] for r in channel_path], values, 'o-',
                        color=colors[method], label=method)
        ax.axhline(1, color='0.75', lw=0.8)
        ax.invert_xaxis()
        ax.set(xlabel='Channel count (100 sources)', ylabel='Source NRMSE (log scale)')
        ax.set_title(f'{chr(99 + row * 3)}  Noise amplification', loc='left', fontsize=9)
    axes[0, 0].legend(fontsize=7, loc='upper left')
    axes[0, 2].legend(fontsize=7)
    fig.suptitle('Nested inverse models: null energy and reconstruction error behave differently', fontsize=11)
    fig.supxlabel('Fixed normal sources; 20 dB on full array; covariance from 2 s baseline; sensor ordering 1 of 3', fontsize=8)
    fig.savefig(args.directory / 'nullspace_overview.png', dpi=300)
    fig.savefig(args.directory / 'nullspace_overview.svg')
    fig.savefig(args.directory / 'nullspace_overview.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
