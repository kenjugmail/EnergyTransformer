"""Render the compact study results with matplotlib (optional reporting dependency)."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def render(root):
    report = json.loads((root / 'results.json').read_text())
    colors = {'bp': '#41556b', 'bp_ce': '#8398aa', 'pc': '#d68b3e', 'pcalm': '#299d8f', 'energy': '#7856b7'}
    method_labels = {'bp': 'BP', 'bp_ce': 'BP-CE', 'pc': 'PC', 'pcalm': 'PC-ALM', 'energy': 'Full EBM'}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.facecolor': 'white', 'axes.axisbelow': True})
    fresh = json.loads((root / 'byte_followup' / 'results.json').read_text())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), layout='constrained')
    for ax, task, label in zip(axes, ('fashion', 'graph', 'bytes'),
                               ('Fashion-MNIST · 2,000 images', 'Structured graph · 1,200 episodes', 'Fresh next byte · 1,000 windows')):
        rows = [r for r in (fresh if task == 'bytes' else report)['rows'] if r['task'] == task]
        if task == 'fashion':
            rows = [r for r in rows if r['depth'] == 32]
        xlabels = [method_labels[r['method']] + ('\n+ memory' if r['memory'] else '') for r in rows]
        for i, row in enumerate(rows):
            field = 'bits_per_target' if task == 'bytes' else 'accuracy'
            values = np.array([p[field] for p in row['per_seed']]) * (1 if task == 'bytes' else 100)
            ax.bar(i, values.mean(), color=colors[row['method']], alpha=.8, width=.7)
            ax.scatter(np.linspace(i - .15, i + .15, len(values)), values, c='#14202b', s=18, zorder=4)
        ax.set_xticks(range(len(rows)), xlabels, rotation=45, ha='right')
        ax.set_title(label, loc='left', fontweight='bold')
        ax.set_ylabel('Bits per byte · lower is better' if task == 'bytes' else 'Accuracy (%)')
        ax.grid(axis='y', alpha=.2)
        if task == 'bytes':
            ax.axhline(fresh['unigram_reference']['mean_bits_per_target'], color='#26394a', linestyle='--',
                       label='Unigram reference')
            ax.legend(fontsize=8)
            ax.set_xlabel('2,000 updates · 8-byte context')
        if task == 'fashion':
            ax.set_xlabel('32 hidden layers · 400 updates')
        if task != 'bytes': ax.set_ylim(0, 100)
    fig.suptitle('Models trained from scratch · dots show three seeds', fontsize=15, fontweight='bold')
    fig.savefig(root / 'model_comparison.png', dpi=180)
    fig.savefig(root / 'model_comparison.pdf')
    plt.close(fig)

    graph = [r for r in report['rows'] if r['task'] == 'graph']
    groups = list(graph[0]['per_seed'][0]['groups'])
    matrix = np.array([[np.mean([s['groups'][g]['accuracy'] for s in r['per_seed']]) for g in groups] for r in graph])
    labels = [method_labels[r['method']] + (' + memory' if r['memory'] else '') for r in graph]
    if 'discrete_graph' in report:
        matrix = np.concatenate([matrix, [[report['discrete_graph']['groups'][g] for g in groups]]])
        labels.append('Discrete supplied rules')
    if (root / 'discrete_results.json').exists():
        discrete = json.loads((root / 'discrete_results.json').read_text())
        extra = [np.mean([t['trained']['groups'][g] for t in discrete['tests']]) for g in groups]
        matrix = np.concatenate([matrix, [extra]])
        labels.append('Discrete learned factors')
    fig, ax = plt.subplots(figsize=(11, 5.4), layout='constrained')
    im = ax.imshow(matrix * 100, vmin=0, vmax=100, cmap='YlGnBu', aspect='auto')
    ax.set_xticks(range(len(groups)), [g.replace('_', '\n') for g in groups])
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(groups)):
            ax.text(j, i, f'{100 * matrix[i, j]:.1f}', ha='center', va='center',
                    color='white' if matrix[i, j] > .6 else '#182735', fontsize=10)
    fig.colorbar(im, ax=ax, label='Accuracy (%)', shrink=.8)
    ax.set_title('Generalization by graph task · supplied structured inputs', loc='left', fontweight='bold', pad=15)
    fig.savefig(root / 'graph_generalization.png', dpi=180)
    fig.savefig(root / 'graph_generalization.pdf')
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=Path(__file__).parent)
    render(p.parse_args().out)
