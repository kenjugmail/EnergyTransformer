"""Render measured kernel speedups from the benchmark artifact."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads((args.report/'benchmark.json').read_text())
    labels, plain, compiled = [], [], []
    for length, phase, description in [(64, 'forward', 'Forward'), (128, 'forward', 'Forward'),
                                      (64, 'forward_backward', 'Forward + backward'),
                                      (128, 'forward_backward', 'Forward + backward')]:
        row = data['workloads'][f'b24_t{length}_{phase}']
        labels.append(f'{description}\nB24 · {length} bytes')
        plain.append(row['krylov2']['speedup_over_reference'])
        compiled.append(row['compiled_krylov2']['speedup_over_reference'])
    for length in (16, 128):
        row = data['decode'][f'prompt{length}_continuation32']['timings']
        labels.append(f'Generation\nB1 · {length}-byte prompt + 32 decode calls')
        plain.append(row['krylov2_persistent']['total']['speedup_over_reference'])
        compiled.append(row['compiled_krylov2_persistent']['total']['speedup_over_reference'])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, ax = plt.subplots(figsize=(10.8, 6.7))
    fig.patch.set_facecolor('#f8fafc')
    ax.set_facecolor('#f8fafc')
    y = np.arange(len(labels))
    first = ax.barh(y-.17, plain, height=.30, color='#087f8c', label='Mode space')
    second = ax.barh(y+.17, compiled, height=.30, color='#6b4cbb', label='Compiled mode space')
    for bars in (first, second):
        ax.bar_label(bars, labels=[f'{b.get_width():.2f}×' for b in bars], padding=6, fontsize=10)
    ax.axvline(1, color='#677487', linestyle='--', linewidth=1.2)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 2.65)
    ax.set_xlabel('Speedup over the existing cached reference (CPU time)')
    ax.set_xticks([0, .5, 1, 1.5, 2, 2.5], ['0', '0.5×', '1×', '1.5×', '2×', '2.5×'])
    ax.grid(axis='x', color='#dce2e9', linewidth=.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='y', length=0, pad=9)
    ax.legend(loc='lower right', bbox_to_anchor=(1, 1.01), ncol=2,
              borderaxespad=0, frameon=False)
    fig.suptitle('REWT: faster execution of unchanged weights', x=.05, y=.98,
                 ha='left', fontsize=17, fontweight='bold', color='#182b46')
    fig.text(.05, .925, 'Float32 · one CPU thread · median of seven alternating repetitions',
             fontsize=11, color='#46576e')
    fig.text(.05, .025, 'Generation includes persistent caches. Compilation startup is excluded and reported separately.\n'
             'Forward + backward excludes optimizer and auxiliary losses. No GPU throughput or quality gain is claimed.',
             fontsize=9, color='#46576e')
    fig.subplots_adjust(left=.37, right=.96, top=.85, bottom=.15)
    fig.savefig(args.report/'kernel_speedups.png', dpi=180)
    fig.savefig(args.report/'kernel_speedups.pdf')


if __name__ == '__main__':
    main()
