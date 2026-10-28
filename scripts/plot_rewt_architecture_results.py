"""Publication-style static figures from the completed architecture study."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'reports/rewt_architecture_20261022'


def main():
    result=json.loads((OUT/'summary.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(11.8,5.5))
    for ax,metric,title,scale in zip(axes,('task','language'),
        ('Generated reasoning · complete response','Language · bits per original byte'),(100,1)):
        groups=result[metric];names=list(groups);x=np.arange(len(names))
        means=np.array([groups[n]['test_mean'] for n in names])*scale
        low=np.array([groups[n]['test_min'] for n in names])*scale
        high=np.array([groups[n]['test_max'] for n in names])*scale
        colors=['#475569' if n=='control' else '#15803d' if n=='delta' else '#2563eb' for n in names]
        ax.barh(x,means,color=colors,alpha=.85,height=.67)
        ax.errorbar(means,x,xerr=np.vstack((means-low,high-means)),fmt='none',ecolor='#111827',capsize=2,lw=1)
        ax.set_yticks(x,names);ax.invert_yaxis();ax.set_title(title,fontsize=11,pad=12)
        ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
        ax.set_xlabel('Correct responses (%) · higher is better' if metric=='task' else 'Bits per byte · lower is better')
    axes[0].set_xlim(0,max(v['test_max'] for v in result['task'].values())*110)
    fig.suptitle('Architecture changes: measured quality at fixed update budgets',fontsize=13)
    fig.text(.5,.015,'Three cold starts per variant; bars are means, whiskers are seed ranges. Different parameter counts and compute. Separate task and language fits.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.045,1,.95));fig.savefig(OUT/'architecture_quality.png',dpi=180);fig.savefig(OUT/'architecture_quality.pdf');plt.close(fig)

    bench=json.loads((OUT/'benchmark_final.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.5),sharey=True)
    names=['reference','krylov2','direct_packed','precond4_structured','scan','feedback8','delta','patch4']
    for ax,backward in zip(axes,(False,True)):
        row=next(r for r in bench['cases'] if r['batch']==24 and r['length']==128 and r['backward']==backward)
        times=[row['arms'][n]['median_cpu_ms'] for n in names]
        ax.barh(np.arange(len(names)),times,color=['#475569']+['#2563eb']*2+['#d97706']+['#15803d']*4)
        ax.set_yticks(np.arange(len(names)),names);ax.set_title('Forward + backward' if backward else 'Inference forward',fontsize=11)
        ax.set_xlabel('CPU process time per batch (ms)');ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    axes[0].invert_yaxis()
    fig.suptitle('Execution cost at batch 24 × 128 positions',fontsize=13)
    fig.text(.5,.015,'One pinned CPU thread, seven alternating repetitions. Orange changes solver; green changes architecture. Timing does not imply equal quality.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.045,1,.94));fig.savefig(OUT/'architecture_execution.png',dpi=180);fig.savefig(OUT/'architecture_execution.pdf');plt.close(fig)

if __name__=='__main__':main()
