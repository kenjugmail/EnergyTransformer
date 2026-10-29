"""Export the frozen whole-word comparison as a PNG and a vector PDF."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, default=Path(__file__).resolve().parents[1]/'reports/rewt_word_refinement_20261029')
    a = p.parse_args(); result = json.loads((a.report/'comparison.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(1,2,figsize=(11.8,4.9),gridspec_kw={'width_ratios':[1,1.25]})
    colors = {'transformer':'#d9772e','hybrid':'#7d6b91','refined':'#087f8c'}
    labels = {'transformer':'Transformer','hybrid':'REWT backbone\nwithout energy','refined':'REWT with\nrefinement'}
    arms = [arm for arm in colors if arm in result['aggregates']]
    for index, arm in enumerate(arms):
        rows = sorted([r for r in result['models'].values() if r['arm']==arm],key=lambda r:r['seed'])
        values = [r['score']['token_nll'] for r in rows]
        axes[0].scatter(index+np.linspace(-.065,.065,len(rows)),values,color=colors[arm],s=42,zorder=3)
        axes[0].plot([index-.16,index+.16],[np.mean(values)]*2,color=colors[arm],lw=3)
        axes[0].annotate(f'{np.mean(values):.4f}',(index,max(values)),xytext=(0,13),textcoords='offset points',ha='center',fontweight='bold')
    axes[0].set_xticks(range(len(arms)),[labels[arm] for arm in arms])
    axes[0].set_ylabel('Held-out token NLL (lower is better)')
    axes[0].set_title('Matched parameters and training tokens',loc='left',fontweight='bold',pad=18)
    axes[0].margins(x=.18,y=.35); axes[0].grid(axis='y',alpha=.18)
    depths = [0,1,2,4,8,16,32]; curve = result['refinement_curve']
    for seed_index in range(len(curve['0']['values'])):
        axes[1].plot(range(len(depths)),[curve[str(k)]['values'][seed_index] for k in depths],
                     color=colors['refined'],alpha=.25,lw=1)
    means = [curve[str(k)]['mean'] for k in depths]
    axes[1].plot(range(len(depths)),means,'o-',color=colors['refined'],lw=2.5,label='REWT: mean of 3 seeds')
    baseline = result['aggregates']['transformer']['token_nll']
    axes[1].axhline(baseline,color=colors['transformer'],ls='--',lw=1.6,label='Transformer mean')
    axes[1].set_xticks(range(len(depths)),depths); axes[1].set_xlabel('Inference refinement steps')
    axes[1].set_ylabel('Held-out token NLL'); axes[1].grid(axis='y',alpha=.18)
    axes[1].set_title('More refinement, same trained weights',loc='left',fontweight='bold',pad=18)
    axes[1].legend(frameon=False,fontsize=9,loc='best')
    fig.suptitle('Whole-word prediction on fresh TinyStories',x=.07,ha='left',fontweight='bold',fontsize=17)
    fig.text(.07,.895,'1,024 unseen stories · 65,536 target positions · 3 training seeds · about 1.40M parameters',fontsize=10,color='#555555')
    fig.text(.07,.035,'12.288M training-token exposures per fit. Checkpoint/depth selection uses validation.\nNLL includes punctuation and special tokens; training FLOPs are not matched.',fontsize=9,color='#555555')
    fig.subplots_adjust(left=.07,right=.98,top=.78,bottom=.23,wspace=.32)
    for extension in ('png','pdf'):
        fig.savefig(a.report/f'comparison.{extension}',dpi=190,bbox_inches='tight',facecolor='white')


if __name__ == '__main__': main()
