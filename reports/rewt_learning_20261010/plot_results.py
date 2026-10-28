"""Export measured, frozen task and language results as a research figure."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent


def main():
    tasks = json.loads((OUT / 'onehop_test/results.json').read_text())
    language = json.loads((OUT / 'language_test/results.json').read_text())
    uncertainty = json.loads((OUT / 'test_uncertainty.json').read_text())
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.9), gridspec_kw={'width_ratios': [1.45, 1, 1]})
    fig.suptitle('Frozen REWT test results · October 10, 2026', x=.055, ha='left', fontsize=15, fontweight='bold')
    policies = {r['rounds']: r['groups'] for r in tasks['results']}
    x = np.arange(3)
    labels = ['Addition\n0–31 inputs', 'One rule\nwith distractors', 'Recall\n4 distractors']
    for rounds, dx, color, name in [(0, -.19, '#a8b1bd', 'Same weights, 0 rounds'),
                                     (2, .19, '#2864b1', 'Primary, 2 rounds')]:
        values = [policies[rounds][f'known/{family}'] for family in ('arithmetic', 'rules', 'recall')]
        bars = axes[0].bar(x + dx, [r['complete_accuracy'] * 100 for r in values], width=.34, color=color, label=name)
        for bar, row in zip(bars, values):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                         f"{row['correct']}/{row['count']}", ha='center', fontsize=8)
    axes[0].set(title='Bounded one-step tasks', ylabel='Complete trace and answer correct (%)', xticks=x, xticklabels=labels, ylim=(0, 114))
    axes[0].legend(loc='upper left', bbox_to_anchor=(0, -.22), ncol=2,
                   columnspacing=1, fontsize=8, frameon=False)
    colors = ['#2864b1', '#c86820', '#a8b1bd']
    names = ['Fourier\nenergy', 'Neural\nenergy', 'Transformer']
    values = [r['test']['bpb'] for r in language]
    cis = [uncertainty['language']['models'][r['case']]['interval_95_percent'] for r in language]
    errors = np.array([[value-ci[0] for value, ci in zip(values, cis)],
                       [ci[1]-value for value, ci in zip(values, cis)]])
    bars = axes[1].bar(x, values, color=colors, yerr=errors, capsize=4)
    for bar, value, ci in zip(bars, values, cis):
        axes[1].text(bar.get_x()+bar.get_width()/2, ci[1]+.035, f'{value:.3f}', ha='center', fontsize=9)
    axes[1].set(title='Separate language checkpoints', ylabel='Bits per raw byte (lower is better)',
                xticks=x, xticklabels=names, ylim=(0, 2))
    costs = [json.loads((OUT / 'language_bpe_extended' / f"{r['case']}.json").read_text())['total_training_cpu_seconds']/1000 for r in language]
    bars = axes[2].bar(x, costs, color=colors)
    for bar, value in zip(bars, costs):
        axes[2].text(bar.get_x()+bar.get_width()/2, value+.08, f'{value:.2f}', ha='center', fontsize=9)
    axes[2].set(title='Equal language token budgets', ylabel='Measured training CPU seconds (×1,000)',
                xticks=x, xticklabels=names, ylim=(0, 4.6))
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=.2)
        ax.tick_params(axis='x', length=0)
    fig.text(.055, .035,
        'One initialization lineage per model; task and language models are separate. Arithmetic has 66 semantic pairs in two wordings.\n'
        'Language bars: 64 unseen story windows; intervals resample documents and do not represent training-seed variability.\n'
        'All 90 generated story continuations failed the exploratory usefulness rubric. Lower predictive loss does not establish coherent prose.',
        fontsize=9, va='bottom', color='#41464e')
    fig.subplots_adjust(left=.055, right=.99, top=.80, bottom=.31, wspace=.43)
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(OUT / f'results_figure.{extension}', dpi=180, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
