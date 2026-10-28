"""Research figure for the failed and binding-supervised memory experiments."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'reports/rewt_architecture_20261022'


def main():
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.4),sharey=True)
    x=np.array([7,47,95,191])
    for ax,folder,title in zip(axes,('long_memory','binding_memory'),
        ('Response loss only · 768 updates','Binding + write-gate supervision · 1,536 updates')):
        rows=json.loads((OUT/folder/'test_results.json').read_text())
        for family,color,label in [('local','#2563eb','Local attention + scan'),('delta','#d97706','Local attention + delta memory')]:
            subset=[r for r in rows if r['case']['name'].startswith(family+'_')]
            values=[]
            for r in subset:
                g=r['test']['groups'];values.append([np.mean([g[f'{n}/True'] for n in (16,32,64)]),*[g[f'{n}/False'] for n in (16,32,64)]])
            values=np.array(values)*100;mean=values.mean(0)
            ax.errorbar(x,mean,yerr=np.vstack((mean-values.min(0),values.max(0)-mean)),
                color=color,label=label,marker='o',capsize=3,lw=2,ms=5)
        ax.set_xscale('log');ax.set_xticks(x,[str(v) for v in x]);ax.set_ylim(-2,104)
        ax.set_title(title,fontsize=11,pad=14);ax.set_xlabel('Gap since the latest requested-key write (bytes)')
        ax.axvline(16,color='#9ca3af',linestyle=':',lw=1.2)
        ax.text(17,5,'16-byte local window',rotation=90,fontsize=8,color='#6b7280',va='bottom')
        ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Correct recall on fresh test prompts (%)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.025))
    fig.suptitle('Learned recall beyond a 16-byte attention window',fontsize=13,y=.98)
    fig.text(.5,.005,'Means and seed ranges across three cold starts. Same grammar, randomized bindings; farthest test gap exceeds the 71-byte training maximum.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.14,1,.94))
    fig.savefig(OUT/'memory_recall.png',dpi=180);fig.savefig(OUT/'memory_recall.pdf');plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,4.8))
    rows=json.loads((OUT/'memory_ablation/test_results.json').read_text())
    for family,color,style,label in (
        ('local_response','#64748b','--','Local · response loss'),
        ('local_binding','#2563eb','-','Local · binding supervision'),
        ('delta_response','#d97706','--','Delta · response loss'),
        ('delta_binding','#15803d','-','Delta · binding supervision')):
        subset=[r for r in rows if r['ablation_family']==family]
        values=np.array([[np.mean([r['test']['groups'][f'{n}/True'] for n in (16,32,64)]),
                          *[r['test']['groups'][f'{n}/False'] for n in (16,32,64)]] for r in subset])*100
        mean=values.mean(0)
        ax.errorbar(x,mean,yerr=np.vstack((mean-values.min(0),values.max(0)-mean)),
                    color=color,linestyle=style,label=label,marker='o',capsize=3,lw=2,ms=5)
    ax.set_xscale('log');ax.set_xticks(x,[str(v) for v in x]);ax.set_ylim(-2,104)
    ax.set_xlabel('Gap since the latest requested-key write (bytes)')
    ax.set_ylabel('Correct recall on fresh test prompts (%)')
    ax.axvline(16,color='#9ca3af',linestyle=':',lw=1.2)
    ax.text(17,37,'16-byte local window',rotation=90,fontsize=8,color='#6b7280')
    ax.axhline(10,color='#cbd5e1',linestyle=':',lw=1)
    ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,-.33),ncol=2,frameon=False,fontsize=9)
    fig.suptitle('Binding supervision improves learned delta-memory recall',fontsize=12)
    ax.set_title('Matched seeds, batches and 1,536 updates · three-seed means and ranges',fontsize=10)
    fig.tight_layout(rect=(0,.02,1,.97))
    fig.savefig(OUT/'memory_matched.png',dpi=180);fig.savefig(OUT/'memory_matched.pdf');plt.close(fig)

if __name__=='__main__':main()
