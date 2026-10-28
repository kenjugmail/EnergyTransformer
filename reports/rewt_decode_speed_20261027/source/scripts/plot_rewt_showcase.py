"""Export presentation figures from saved recall and decoder measurements."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'course_project/presentation_20261028/streaming_showcase'


def main():
    study = json.loads((ROOT/'reports/rewt_streaming_20261026/summary.json').read_text())['summary']
    directory = ROOT/'reports/rewt_decode_speed_20261027'
    cpu = json.loads((directory/'cpu.json').read_text())['medians']
    gpu = json.loads((directory/'cuda.json').read_text())['medians']
    names = ['transformer', 'window_transformer', 'hybrid', 'hybrid_energy']
    labels = ['Full\nTransformer', 'Window\nTransformer', 'Hybrid', 'Hybrid\n+ energy']
    colors = ['#355f82', '#867593', '#168478', '#c58b2c']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'axes.spines.top': False, 'axes.spines.right': False})
    x = np.arange(4)
    fig, axes = plt.subplots(1, 3, figsize=(15, 7))
    fig.suptitle('Compact memory is the clearest result', x=.055, ha='left', y=.96,
                 fontsize=23, fontweight='bold', color='#172f42')
    fig.text(.055, .89, '~112K parameters • recall: three training seeds • timing: one checkpoint per model', color='#52616b')
    ax = axes[0]
    bars = ax.bar(x, [100*study[n]['long_recall'] for n in names], color=colors)
    ax.bar_label(bars, fmt='%.1f%%', padding=4, fontsize=12)
    ax.set_ylim(0, 104); ax.set_ylabel('Exact response accuracy (%) ↑')
    ax.set_title('Recall through distractions', loc='left', fontweight='bold', pad=16)
    ax.text(0, -.25, '32–128 distractor facts; up to 520 bytes\n72 test prompts × 3 seeds per model', transform=ax.transAxes, fontsize=11, color='#52616b')
    ax = axes[1]
    sizes = [cpu[f'{n}/compile/8192']['state_bytes']/1024 for n in names]
    bars = ax.bar(x, sizes, color=colors)
    ax.set_yscale('log'); ax.set_ylim(10, 30000)
    ax.bar_label(bars, labels=['8.03 MiB', '64.0 KiB', '36.5 KiB', '36.5 KiB'], padding=5, fontsize=11)
    ax.set_ylabel('Allocated cache/state (KiB, log scale) ↓')
    ax.set_title('Small carried state at 8K', loc='left', fontweight='bold', pad=16)
    ax.text(0, -.25, '8,192-token prefix + 32 output capacity\nExcludes weights and runtime workspaces', transform=ax.transAxes, fontsize=11, color='#52616b')
    ax = axes[2]
    bars = ax.bar(x, [cpu[f'{n}/compile/8192']['wall_ms'] for n in names], color=colors)
    ax.bar_label(bars, fmt='%.2f', padding=4, fontsize=12)
    ax.set_ylim(0, 20); ax.set_ylabel('Milliseconds / 32 generated tokens ↓')
    ax.set_title('Faster long-prefix CPU decode', loc='left', fontweight='bold', pad=16)
    ax.text(0, -.25, 'One CPU core; all arms compiled\nWarmed decode only; nine repetitions', transform=ax.transAxes, fontsize=11, color='#52616b')
    for ax in axes:
        ax.set_xticks(x, labels, fontsize=10)
        ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
    fig.text(.055, .055, 'Energy refinement has not improved recall. The 8K measurements test execution cost, not 8K recall accuracy.',
             fontsize=12, fontweight='bold', color='#684b23')
    fig.subplots_adjust(left=.065, right=.985, top=.77, bottom=.28, wspace=.38)
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ['png', 'pdf']:
        fig.savefig(OUT/f'memory_showcase.{suffix}', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.4))
    fig.suptitle('Same model, faster decoding', x=.07, ha='left', y=.95, fontsize=23, fontweight='bold')
    fig.text(.07, .87, 'Hybrid + energy • identical checkpoint and two refinement rounds • 8,192-token prefix', color='#52616b')
    for ax, result, backend, title in [(axes[0], cpu, 'compile', 'CPU: compilation'),
                                       (axes[1], gpu, 'graph', 'GB10 GPU: CUDA graph replay')]:
        values = [result[f'hybrid_energy/{b}/8192']['wall_ms'] for b in ['reference', backend]]
        bars = ax.bar([0, 1], values, color=['#98a5ae', '#168478'], width=.58)
        ax.bar_label(bars, fmt='%.2f ms', padding=6, fontsize=14)
        ax.set_xticks([0, 1], ['Previous sampler', 'Optimized sampler'])
        ax.set_ylim(0, max(values)*1.28); ax.set_ylabel('Milliseconds / 32 generated tokens ↓')
        ax.set_title(f'{title}\n{values[0]/values[1]:.2f}× faster', loc='left', fontsize=14, pad=14)
        ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
    fig.text(.07, .08, 'Medians of nine alternating runs; warmed generation including token return to CPU.\n'
             'Prefill, compilation and graph capture excluded. Shared GPU (3% → 44% utilization); no training-speed claim.', fontsize=11, color='#52616b')
    fig.subplots_adjust(left=.085, right=.97, top=.70, bottom=.22, wspace=.3)
    for suffix in ['png', 'pdf']:
        fig.savefig(OUT/f'decode_speed.{suffix}', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
