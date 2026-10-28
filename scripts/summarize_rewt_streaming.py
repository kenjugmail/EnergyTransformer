"""Audit and report every frozen streaming case, including losses and controls."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from rewt.data import file_sha256
from rewt.experiment import save_json
from rewt.tasks import Task,score_response
from study_rewt_streaming import verify


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,default=ROOT/'reports/rewt_streaming_20261026');args=p.parse_args();out=args.study
    plan=json.loads((out/'plan.json').read_text());verify(plan)
    results=json.loads((out/'test_results.json').read_text());benchmark=json.loads((out/'benchmark.json').read_text())
    assert len(results)==len(plan['cases'])==len({r['case']['name'] for r in results})
    assert {r['case']['name'] for r in results}=={c['name'] for c in plan['cases']}
    tasks={s:[Task(**json.loads(l)) for l in (out/f'tasks_{s}.jsonl').read_text().splitlines()] for s in ('test','test_long')}
    audited=0;groups=defaultdict(list)
    for r in results:
        assert file_sha256(r['checkpoint'])==r['checkpoint_sha256']
        groups[r['case']['objective'],r['case']['variant']].append(r)
        if r['case']['objective']=='tasks':
            for split in ('test','test_long','zero_rounds'):
                if split not in r:continue
                expected=tasks['test_long' if split=='test_long' else 'test']
                assert len(r[split]['rows'])==len(expected)
                for row,task in zip(r[split]['rows'],expected):
                    assert row['id']==task.id and row['template']==task.template
                    check=score_response(task,row['response'])
                    for key in ('both_correct','answer_correct','steps_correct','valid_format'):assert row[key]==check[key]
                    audited+=1
    for objective in ('language','tasks'):
        for seed in sorted({c['seed'] for c in plan['cases']}):
            rows=[r for r in results if r['case']['objective']==objective and r['case']['seed']==seed]
            assert len({r['batch_sha256'] for r in rows})==1,(objective,seed)
    for rows in groups.values():rows.sort(key=lambda r:r['case']['seed'])
    summary={};names=['transformer','window_transformer','hybrid','hybrid_energy']
    for name in names:
        lr=groups['language',name];cr=groups['compute',name];tr=groups['tasks',name]
        summary[name]=dict(parameters=lr[0]['parameters'],active_parameters=lr[0]['active_parameters'],
            language_bpb=statistics.mean(r['test']['bpb'] for r in lr),language_seed_bpb=[r['test']['bpb'] for r in lr],
            long_language_bpb=statistics.mean(r['test_long']['bpb'] for r in lr),
            compute_bpb=statistics.mean(r['test']['bpb'] for r in cr),compute_seed_bpb=[r['test']['bpb'] for r in cr],
            compute_updates=[r['updates'] for r in cr],
            task_complete=statistics.mean(r['test']['complete_accuracy'] for r in tr),task_seed_complete=[r['test']['complete_accuracy'] for r in tr],
            task_groups={key:{v:statistics.mean(r['test']['groups'][key][v] for r in tr) for v in tr[0]['test']['groups'][key]} for key in ('family','difficulty','depth')},
            long_recall=statistics.mean(r['test_long']['complete_accuracy'] for r in tr),
            long_recall_depth={d:statistics.mean(r['test_long']['groups']['depth'][d] for r in tr) for d in tr[0]['test_long']['groups']['depth']})
    rng=np.random.default_rng(261777);intervals={}
    for objective in ('language','compute'):
        for name in ('hybrid','hybrid_energy'):
            a=groups[objective,name];b=groups[objective,'transformer'];n=len(a[0]['test']['rows'])
            aa=np.array([[x['nll'] for x in r['test']['rows']] for r in a]);bb=np.array([[x['nll'] for x in r['test']['rows']] for r in b])
            nb=np.array([x['bytes'] for x in a[0]['test']['rows']]);values=[]
            for _ in range(3000):
                ix=rng.integers(0,n,n);values.append(float(((aa[:,ix]-bb[:,ix]).sum(1)/nb[ix].sum()/np.log(2)).mean()))
            intervals[f'{objective}_{name}_minus_transformer_bpb']=dict(delta=summary[name]['language_bpb' if objective=='language' else 'compute_bpb']-summary['transformer']['language_bpb' if objective=='language' else 'compute_bpb'],ci95=np.quantile(values,[.025,.975]).tolist())
    by_id=defaultdict(list)
    for i,t in enumerate(tasks['test']):by_id[t.id].append(i)
    semantic=list(by_id);strata=defaultdict(list)
    for j,identity in enumerate(semantic):t=tasks['test'][by_id[identity][0]];strata[t.family,t.steps].append(j)
    for name in ('hybrid','hybrid_energy'):
        aa=np.array([[np.mean([r['test']['rows'][i]['both_correct'] for i in by_id[identity]]) for identity in semantic] for r in groups['tasks',name]])
        bb=np.array([[np.mean([r['test']['rows'][i]['both_correct'] for i in by_id[identity]]) for identity in semantic] for r in groups['tasks','transformer']])
        values=[]
        for _ in range(3000):
            ix=np.concatenate([rng.choice(group,len(group),replace=True) for group in strata.values()]);values.append(float((aa[:,ix]-bb[:,ix]).mean()))
        intervals[f'tasks_{name}_minus_transformer']=dict(delta=float((aa-bb).mean()),ci95=np.quantile(values,[.025,.975]).tolist())
    ablations={objective:dict(with_energy=statistics.mean(r['test']['bpb' if objective!='tasks' else 'complete_accuracy'] for r in groups[objective,'hybrid_energy']),
        disabled=statistics.mean(r['zero_rounds']['bpb' if objective!='tasks' else 'complete_accuracy'] for r in groups[objective,'hybrid_energy'])) for objective in ('language','compute','tasks')}
    report=dict(summary=summary,paired_intervals=intervals,frozen_energy_ablations=ablations,audited_responses=audited,
        completed_fits=len(results),training_cpu_seconds=sum(r['training_cpu_seconds'] for r in results),
        validation_cpu_seconds=sum(r['validation_cpu_seconds'] for r in results),
        uncertainty='Paired bootstrap over documents or family/depth-stratified semantic IDs; three observed seeds held fixed. These intervals do not quantify uncertainty over arbitrary future training seeds.',
        source_and_checkpoint_hashes_verified=True,equal_update_batch_hashes_verified=True)
    save_json(out/'summary.json',report)
    labels={'transformer':'Full Transformer','window_transformer':'Window Transformer','hybrid':'Hybrid, no refinement','hybrid_energy':'Hybrid + energy'}
    lines=['# Streaming REWT: matched Transformer comparison','',
        'This study completed **36 cold fits** across three seeds. It evaluates an opt-in autoregressive hybrid with gated delta memory, local attention and optional latent energy refinement. The comparison uses modern full-attention and window-attention controls, matched to within 0.1% of allocated parameters. **This is a bounded comparison, not evidence of superiority in every setting.**','',
        '## Language and generated task accuracy','',
        '| Model | Parameters | Story bpb, equal updates ↓ | Story bpb, equal CPU ↓ | Complete task accuracy ↑ | Long recall ↑ |',
        '|---|---:|---:|---:|---:|---:|']
    for name in names:
        r=summary[name];lines.append(f'| {labels[name]} | {r["parameters"]:,} | {r["language_bpb"]:.4f} | {r["compute_bpb"]:.4f} | {100*r["task_complete"]:.2f}% | {100*r["long_recall"]:.2f}% |')
    lines += ['', 'All cells average all three seeds. Lower bits per original byte is better. Language fits use 768 updates / 2,359,296 BPE target tokens. Equal-CPU fits use 90 measured training process seconds, with at most one-update overshoot. Task fits are separate cold models trained for 1,536 updates on raw-byte prompts; they do not inherit language pretraining or the earlier structured reasoner.','',
        'The task test has 576 prompts (288 semantic identities, each in familiar and withheld wording). Long recall adds 72 prompts with 32, 64 or 128 distractors. Inference receives only the prompt and generates the complete response autoregressively. No symbolic solver or task metadata participates in generation.','',
        '| Model | Arithmetic complete | Rule-chain complete | Recall complete | Familiar wording | New wording |',
        '|---|---:|---:|---:|---:|---:|']
    for name in names:
        r=summary[name]['task_groups'];f=r['family'];d=r['difficulty']
        lines.append(f'| {labels[name]} | {100*f["arithmetic"]:.2f}% | {100*f["rules"]:.2f}% | {100*f["recall"]:.2f}% | {100*d["familiar_format"]:.2f}% | {100*d["withheld_wording"]:.2f}% |')
    lines += ['', 'The family breakdown is essential: aggregate task accuracy must not be described as general reasoning if it is driven by recall. Greedy story samples also remain repetitive and sometimes incoherent; lower held-out loss alone does not establish high-quality prose. [Unselected samples](validation_samples.json) preserve a fixed prompt for all four variants.','',
        '| Model | Task seed 26301 | Task seed 26302 | Task seed 26303 |','|---|---:|---:|---:|']
    for name in names:
        values=summary[name]['task_seed_complete'];lines.append(f'| {labels[name]} | {100*values[0]:.2f}% | {100*values[1]:.2f}% | {100*values[2]:.2f}% |')
    lines += ['', 'Both attention controls have a weak third task seed. The hybrid beats the full Transformer in every paired task seed, but the mean gain is enlarged by that unstable baseline run. All arms used one common learning-rate recipe without architecture-specific tuning. These results do not establish performance against the best-tuned Transformer.','',
        '## Paired uncertainty','',report['uncertainty'],'', '| Comparison | Mean difference | Paired 95% interval |','|---|---:|---:|']
    for name,r in intervals.items():
        scale=100 if name.startswith('tasks') else 1;unit=' pp' if scale==100 else ' bpb'
        lines.append(f'| {name} | {scale*r["delta"]:+.4f}{unit} | [{scale*r["ci95"][0]:+.4f}, {scale*r["ci95"][1]:+.4f}]{unit} |')
    lines += ['', 'Per-seed values and all long-context results are in [summary.json](summary.json). Test outcomes were opened only after every declared run finished. Validation chose checkpoints; no test outcome chose a training configuration.','',
        '## Does energy refinement help?','', '| Frozen energy model | Refinement enabled | Same weights, refinement disabled |','|---|---:|---:|']
    for objective,r in ablations.items():
        scale=100 if objective=='tasks' else 1;unit='%' if objective=='tasks' else ' bpb'
        lines.append(f'| {objective} | {scale*r["with_energy"]:.4f}{unit} | {scale*r["disabled"]:.4f}{unit} |')
    lines += ['', 'The separately trained hybrid without energy is also reported above. A frozen intervention and a separately trained ablation answer different questions; neither should be omitted. Energy descent remains an internal optimization property rather than a correctness certificate.','',
        '## CPU execution and actual carried storage','',
        'The benchmark uses one pinned CPU core after study training finishes, alternating model order across five decode and seven forward/backward repetitions. Both Transformer controls use preallocated caches. Decode consumes the same 32 tokens for every arm, rather than timing different generated outputs. Prefill uses 128-token chunks. The machine is shared with other user work.','',
        '| Model | Forward + backward, 24×128 | Decode 32, prefix 128 | Decode 32, prefix 8192 | Allocated state at prefix 8192 |','|---|---:|---:|---:|---:|']
    for name in names:
        r=benchmark['summary'][name];short=r['decode']['128'];long=r['decode']['8192']
        lines.append(f'| {labels[name]} | {1000*r["training_cpu_seconds"]:.2f} ms | {1000*short["decode_cpu_seconds"]:.2f} ms | {1000*long["decode_cpu_seconds"]:.2f} ms | {int(long["allocated_state_bytes"]):,} bytes |')
    lines += ['', 'Training microbenchmarks include forward, cross entropy and backward, excluding optimizer work. Study CPU budgets include optimizer and preprocessing, excluding validation and checkpoint I/O. Memory values count allocated cache/state storage, not total process RAM or parameter storage. Full-attention capacity includes space for the 32 continuation tokens. Detailed intermediate lengths, prefill cost, CPU and wall samples are in [benchmark.json](benchmark.json).','',
        'Early equal-update timing was unpinned on a CPU with two core types and is descriptive only. Before the equal-CPU fits, workers were pinned to matching performance cores; [the change is recorded](worker_affinity.json). The separate fixed-core benchmark is the source for execution comparisons.','',
        '## Verification and reproducibility','',
        f'- Audited {audited:,} generated responses against the independent scorer; all checkpoint hashes and frozen source/data hashes match.',
        '- Exact per-seed batch hashes match across all equal-update arms.',
        '- Chunked delta recurrence and gradients match a separate sequential implementation; causal, chunked and token decoding parity pass.',
        '- Owned preallocated sessions match full forward calls, preserve earlier outputs, and reject changed weights or capacity overflow.',
        '- GB10 float32 CPU/GPU and cached-decoding checks pass with matching greedy predictions. This is numerical verification, not a GPU throughput claim.',
        '',f'Total full-study training CPU time was {report["training_cpu_seconds"]:.1f} seconds; validation consumed {report["validation_cpu_seconds"]:.1f} additional process seconds. This excludes the eight preliminary 128-update validation pilots, kernel tests, benchmark, data preparation and report work.','',
        'See [the architecture and exact math](DESIGN.md), [the frozen plan](plan.json), [all test results](test_results.json), and [CUDA verification](cuda_verification.json).','',
        '```bash','OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\',
        '  taskset -c 5,6,8 .venv/bin/python scripts/study_rewt_streaming.py \\',
        '  --output reports/rewt_streaming_replication --runs runs/rewt_streaming_replication \\',
        '  --prepare --train --workers 3',
        'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\',
        '  .venv/bin/python scripts/study_rewt_streaming.py \\',
        '  --output reports/rewt_streaming_replication --test --workers 3',
        '.venv/bin/python scripts/benchmark_rewt_streaming.py --study reports/rewt_streaming_replication',
        '.venv/bin/python scripts/summarize_rewt_streaming.py --study reports/rewt_streaming_replication','```','',
        'Select CPU IDs available on the reproduction machine. A new task preparation excludes previously recorded test identities, so its task test is a fresh replication. Story partitions are the fixed document-disjoint corpus from this study. Preserve partial or completed directories rather than overwriting them.','',
        'Sample a saved language checkpoint:','', '```bash',
        '.venv/bin/python scripts/sample_rewt_streaming.py \\',
        '  --checkpoint runs/rewt_streaming_20261026/language_hybrid_26301/best.pt \\',
        "  --prompt 'Once upon a time, ' --tokens 96",'```','',
        'The stored hybrid sampler above uses a fixed seed checkpoint, not one chosen for attractive generations. All models are small research prototypes; broad-domain quality, deeper reasoning, long-context retention and competitive GPU throughput remain open.','']
    if (out/'cuda_benchmark.json').exists():
        gpu=json.loads((out/'cuda_benchmark.json').read_text())
        lines += ['## Shared-device GPU timing','',
            'A separate NVIDIA GB10 snapshot used eager float32 forward, cross entropy and backward at batch 24 × 128, with TF32 disabled, 15 alternating repetitions and three warmups. It excludes optimizer work, compilation and mixed precision. Other GPU work remained active; these are scoped measurements rather than exclusive-device peak throughput.','',
            '| Model | Median CUDA-event interval | Median wall time |','|---|---:|---:|']
        for name in names:
            row=gpu['medians'][name];lines.append(f'| {labels[name]} | {row["gpu_ms"]:.3f} ms | {row["wall_ms"]:.3f} ms |')
        lines += ['',f'GPU utilization was {gpu["utilization_before"]} before and {gpu["utilization_after"]} after this short benchmark. Event intervals include idle gaps between host launches. The small delta solves and refinement operations have overhead; this implementation does not establish a GPU training-speed advantage. [Raw timings and numerical checks](cuda_benchmark.json).','']
    if (out/'comparison.png').exists():lines += ['## Comparison figure','', '![Measured quality and execution tradeoffs](comparison.png)','', '[PDF for presentation use](comparison.pdf)','']
    (out/'FINDINGS.md').write_text('\n'.join(lines))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
