"""Plot the native reader adaptation; matplotlib is a reporting-only dependency."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def render(root=Path('reports/hybrid_integration')):
    report = json.loads((root / 'results.json').read_text())
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.axisbelow': True})
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.3), layout='constrained')
    colors = ['#187f86', '#5369ad', '#cc882e']
    for record, color in zip(report['records'], colors):
        x = [0] + [h['step'] for h in record['history']]
        y = [record['initial_validation']['accuracy']] + [h['valid_accuracy'] for h in record['history']]
        left.plot(x, np.array(y) * 100, marker='o', color=color, label=f"Seed {record['seed']}")
    left.set_title('Only the fact readers are trained', loc='left', fontweight='bold')
    left.set_xlabel('Reader updates')
    left.set_ylabel('Validation accuracy (%)')
    left.legend(frameon=False, loc='lower right')
    for i, (key, label, color) in enumerate(zip(('memory', 'memory_disabled', 'memory_shuffled'),
             ('Current facts', 'Readers disabled', 'Shuffled facts'), ('#187f86', '#8792a1', '#cc882e'))):
        values = [t[key]['accuracy'] * 100 for t in report['tests']]
        right.bar(i, np.mean(values), color=color, alpha=.8, width=.6)
        right.scatter(np.linspace(i - .12, i + .12, len(values)), values, color='#14202b', s=20, zorder=3)
        right.text(i, min(102, np.mean(values) + 3), f'{np.mean(values):.1f}%', ha='center', fontsize=10)
    right.set_xticks(range(3), ('Current\nfacts', 'Readers\ndisabled', 'Shuffled\nfacts'))
    right.set_title('Held-out memory usage · 600 queries', loc='left', fontweight='bold')
    right.set_ylabel('Test accuracy (%)')
    for ax in (left, right):
        ax.set_ylim(0, 106)
        ax.set_yticks(range(0, 101, 20))
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Native checkpoint + learned fact readers · original weights frozen', fontweight='bold', fontsize=13)
    fig.savefig(root / 'reader_adaptation.png', dpi=180)
    fig.savefig(root / 'reader_adaptation.pdf')
    plt.close(fig)


if __name__ == '__main__':
    render()
