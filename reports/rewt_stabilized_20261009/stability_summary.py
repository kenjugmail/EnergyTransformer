"""Summarize actual reasoning gradients without changing weights."""
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent


def main():
    plan=json.loads((ROOT/'selected_protocol.json').read_text())
    assert all((Path(plan['runs'])/f'seed_{seed}'/'full_rewt'/'reasoning'/'training.json').exists()
               for seed in plan['settings']['seeds']), 'Complete every full reasoning fit first'
    records=[]
    traces={}
    for seed in plan['settings']['seeds']:
        for arm in ('legacy_fourier','neural_energy','fourier_rewt','full_rewt'):
            folder=Path(plan['runs'])/f'seed_{seed}'/arm/'reasoning'
            rows=[json.loads(x) for x in (folder/'trace.jsonl').read_text().splitlines()]
            assert len(rows)==plan['settings']['reason_updates']
            assert all(math.isfinite(r['raw_gradient_norm']) and math.isfinite(r['total_loss']) for r in rows)
            # Parent controls recorded float32 norms; their reductions differ
            # by up to 1.1e-5 at the clipping boundary. New full fits use double.
            norm_tolerance=1e-7 if arm=='full_rewt' else 2e-5
            assert all(r['applied_gradient_norm']<=1+norm_tolerance for r in rows)
            record=dict(variant=arm,seed=seed,steps=len(rows),
                max_raw_norm=max(r['raw_gradient_norm'] for r in rows),
                median_raw_norm=statistics.median(r['raw_gradient_norm'] for r in rows),
                clipped_fraction=sum(r['clipped'] for r in rows)/len(rows),
                spike_fraction=sum(r['spike'] for r in rows)/len(rows),
                max_total_loss=max(r['total_loss'] for r in rows),
                max_applied_norm=max(r['applied_gradient_norm'] for r in rows),
                applied_norm_rounding_tolerance=norm_tolerance,
                all_objectives_and_norms_finite=True)
            records.append(record);traces[(arm,seed)]=rows
    (ROOT/'stability_summary.json').write_text(json.dumps(records,indent=2)+'\n')
    output=['# Long-run reasoning stability','',
        'All recorded optimizer updates completed with finite objectives and gradient components. The new full fits use float64 norm reduction and clip the applied vector to approximately norm one. Reused controls retain their original float32 norm diagnostics, with rounding differences up to 1.1e-5 at that boundary. Completion alone does not establish smooth learning: a large raw norm or loss remains a material diagnostic, even if clipping allows the next update. The full model bounds credit through carried latent states to sixteen observed tokens during training; forward memory is preserved.','',
        '| Variant / seed | Maximum raw gradient norm | Median raw norm | Clipped updates % | Maximum total loss |','|---|---:|---:|---:|---:|']
    for r in records:
        output.append(f"| {r['variant']} / {r['seed']} | {r['max_raw_norm']:.3g} | {r['median_raw_norm']:.3g} | {r['clipped_fraction']*100:.1f} | {r['max_total_loss']:.3g} |")
    (ROOT/'STABILITY.md').write_text('\n'.join(output)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,2,figsize=(12,4))
    for seed in plan['settings']['seeds']:
        rows=traces[('full_rewt',seed)]
        axes[0].plot([r['step'] for r in rows],[r['raw_gradient_norm'] for r in rows],label=f'seed {seed}',alpha=.8)
        axes[1].plot([r['step'] for r in rows],[r['ce'] for r in rows],label=f'seed {seed}',alpha=.7)
    axes[0].set_yscale('log');axes[0].set_ylabel('Full reasoning raw gradient norm')
    axes[1].set_ylabel('Full reasoning training cross-entropy')
    for a in axes:
        a.set_xlabel('Optimizer update');a.grid(alpha=.2);a.legend()
        a.axvline(200,color='gray',linestyle='--',linewidth=.8);a.axvline(440,color='gray',linestyle='--',linewidth=.8)
    figure.tight_layout();figure.savefig(ROOT/'stability.png',dpi=160);plt.close(figure)
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
