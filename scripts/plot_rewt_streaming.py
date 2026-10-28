"""Publication-style tradeoff figure from the closed streaming study."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,default=Path('reports/rewt_streaming_20261026'));a=p.parse_args()
    summary=json.loads((a.study/'summary.json').read_text())['summary'];bench=json.loads((a.study/'benchmark.json').read_text())['summary'];gpu=json.loads((a.study/'cuda_benchmark.json').read_text())['medians']
    names=['transformer','window_transformer','hybrid','hybrid_energy'];labels=['Full\nTransformer','Window\nTransformer','Hybrid','Hybrid\n+ energy'];colors=['#31567a','#8b708e','#168579','#d69336']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans'})
    fig,axes=plt.subplots(2,2,figsize=(12.5,8.2));x=np.arange(4)
    ax=axes[0,0]
    for i,name in enumerate(names):
        r=summary[name]
        ax.scatter(np.full(3,i-.13),r['language_seed_bpb'],color=colors[i],s=22,alpha=.5)
        ax.scatter(np.full(3,i+.13),r['compute_seed_bpb'],facecolors='none',edgecolors=colors[i],s=22,alpha=.6)
        ax.plot(i-.13,r['language_bpb'],'o',color=colors[i],markersize=8)
        ax.plot(i+.13,r['compute_bpb'],'s',mfc='none',mec=colors[i],markersize=8)
    ax.plot([],[],'o',color='#444444',label='Equal updates');ax.plot([],[],'s',mfc='none',mec='#444444',label='Equal CPU time')
    ax.set_xticks(x,labels);ax.set_ylabel('Bits per original byte ↓');ax.set_title('Story prediction: budget changes the winner',loc='left',fontweight='bold');ax.legend(frameon=False,loc='upper right',fontsize=9)
    ax.grid(axis='y',alpha=.15)
    ax=axes[0,1];width=.18
    for i,name in enumerate(names):
        vals=[100*summary[name]['task_complete'],100*summary[name]['long_recall']]
        bars=ax.bar(np.arange(2)+(i-1.5)*width,vals,width,color=colors[i],label=labels[i].replace('\n',' '))
        ax.bar_label(bars,fmt='%.1f',fontsize=8,padding=2)
    ax.set_xticks([0,1],['Complete synthetic tasks','Long recall']);ax.set_ylabel('Exact response accuracy (%) ↑');ax.set_ylim(0,108);ax.set_title('Task gains are largely retrieval',loc='left',fontweight='bold');ax.legend(frameon=False,fontsize=8,loc='upper left')
    ax=axes[1,0];lengths=[128,512,2048,8192]
    for name,color,label in zip(names,colors,labels):ax.plot(lengths,[1000*bench[name]['decode'][str(n)]['decode_cpu_seconds'] for n in lengths],marker='o',color=color,label=label.replace('\n',' '))
    ax.set_xscale('log',base=2);ax.set_xticks(lengths,[str(n) for n in lengths]);ax.set_xlabel('Prefix length (tokens)');ax.set_ylabel('CPU milliseconds / 32 decoded tokens ↓');ax.set_title('Bounded state helps at longer prefixes',loc='left',fontweight='bold');ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=8,loc='upper left')
    ax=axes[1,1];bars=ax.bar(x,[gpu[n]['gpu_ms'] for n in names],color=colors)
    ax.bar_label(bars,fmt='%.2f',padding=3);ax.set_xticks(x,labels);ax.set_ylim(0,9.5);ax.set_ylabel('GPU milliseconds / forward + backward ↓');ax.set_title('Eager GPU training still favors attention',loc='left',fontweight='bold');ax.grid(axis='y',alpha=.15)
    fig.suptitle('Streaming REWT: measured wins and remaining costs',fontsize=16,fontweight='bold',x=.055,ha='left')
    fig.text(.055,.925,'~112K parameters · three training seeds · fresh held-out tests · independent timing',fontsize=10,color='#555555')
    fig.text(.055,.025,'CPU: one pinned core, preallocated caches. GPU: shared GB10, float32, 24×128, eager execution. Small-model results; no general superiority claim.',fontsize=9,color='#555555')
    fig.subplots_adjust(left=.075,right=.98,top=.86,bottom=.12,hspace=.48,wspace=.28)
    fig.savefig(a.study/'comparison.png',dpi=180);fig.savefig(a.study/'comparison.pdf');plt.close(fig)


if __name__=='__main__':main()
