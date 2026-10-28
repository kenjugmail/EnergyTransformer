"""Aggregate all preregistered seeds, paired heldout outcomes and timing evidence."""
from collections import defaultdict
import json,math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
OUT=ROOT/'reports/rewt_architecture_20261022'


def grouped_results(path,key):
    rows=json.loads(path.read_text());groups=defaultdict(list)
    for r in rows:groups[r.get('ablation_family',r['case'].get('variant',r['case']['name'].rsplit('_',1)[0]))].append(r)
    result={}
    for name,group in groups.items():
        group.sort(key=lambda r:r['case']['seed'])
        values=[r['test'][key] for r in group]
        result[name]=dict(seeds=[r['case']['seed'] for r in group],test_mean=float(np.mean(values)),
            test_min=min(values),test_max=max(values),test_values=values,parameters=group[0]['parameters'],
            training_cpu_mean=float(np.mean([r['training_cpu_seconds'] for r in group])),
            validation_mean=float(np.mean([r.get('validation_complete',r.get('validation_bpb',r.get('validation_accuracy'))) for r in group])))
    return result,rows


def paired_intervals(rows):
    by_variant=defaultdict(dict)
    for result in rows:
        records=[json.loads(l) for l in (OUT/f"{result['case']['name']}_test_outputs.jsonl").read_text().splitlines()]
        ids=defaultdict(list)
        for r in records:ids[r['id']].append(float(r['both_correct']))
        by_variant[result['case']['variant']][result['case']['seed']]={k:np.mean(v) for k,v in ids.items()}
    seeds=sorted(by_variant['control']);ids=sorted(by_variant['control'][seeds[0]])
    tasks={r['id']:r for r in [json.loads(l) for l in (OUT/'tasks_test.jsonl').read_text().splitlines()]}
    strata=defaultdict(list)
    for i,identity in enumerate(ids):strata[tasks[identity]['family'],tasks[identity]['steps']].append(i)
    rng=np.random.default_rng(22208)
    draws=np.concatenate([rng.choice(indices,(5000,len(indices)),replace=True) for indices in strata.values()],axis=1)
    result={}
    for name in by_variant:
        differences=np.array([[by_variant[name][seed][i]-by_variant['control'][seed][i] for i in ids] for seed in seeds])
        delta=differences.mean(0)
        interval=np.quantile(delta[draws].mean(-1),[.025,.975]).tolist()
        result[name]=dict(mean_delta=float(delta.mean()),ci95=interval,seed_deltas=differences.mean(-1).tolist(),
            method='5000 paired semantic-ID bootstrap samples stratified by family/depth, conditional on the three trained seeds; both wording variants stay together. Does not estimate training-seed population uncertainty.')
    return result


def table(groups,scale=100):
    lines=['| Variant | Parameters | Validation mean | Test mean | Test seed range | CPU training seconds, mean |',
           '|---|---:|---:|---:|---:|---:|']
    for n,r in groups.items():
        lines.append(f"| {n} | {r['parameters']:,} | {scale*r['validation_mean']:.3f} | {scale*r['test_mean']:.3f} | {scale*r['test_min']:.3f}–{scale*r['test_max']:.3f} | {r['training_cpu_mean']:.1f} |")
    return '\n'.join(lines)


def main():
    task,rows=grouped_results(OUT/'test_results.json','macro_complete')
    language,_=grouped_results(OUT/'language/test_results.json','bpb')
    memory,_=grouped_results(OUT/'long_memory/test_results.json','accuracy')
    binding,binding_rows=grouped_results(OUT/'binding_memory/test_results.json','accuracy')
    ablation,ablation_rows=grouped_results(OUT/'memory_ablation/test_results.json','accuracy')
    compute,compute_rows=grouped_results(OUT/'equal_compute/test_results.json','bpb')
    for name,group in compute.items():
        subset=[r for r in compute_rows if r['case']['variant']==name]
        group['mean_updates']=float(np.mean([r['updates'] for r in subset]))
        group['training_cpu_max']=max(r['training_cpu_seconds'] for r in subset)
    paired=paired_intervals(rows)
    selected=max(task,key=lambda n:task[n]['validation_mean'])
    selected_language=min(language,key=lambda n:language[n]['validation_mean'])
    memory_gaps={}
    for n in binding:
        subset=[r for r in binding_rows if r['case']['name'].startswith(n+'_')]
        memory_gaps[n]={key:float(np.mean([r['test']['groups'][key] for r in subset])) for key in subset[0]['test']['groups']}
    ablation_gaps={n:{key:float(np.mean([r['test']['groups'][key] for r in ablation_rows if r['ablation_family']==n]))
                     for key in ablation_rows[0]['test']['groups']} for n in ablation}
    result=dict(task=task,paired_task_deltas=paired,language=language,memory_initial=memory,memory_binding=binding,
        memory_ablation=ablation,memory_ablation_groups=ablation_gaps,equal_compute=compute,total_new_fits=96,
        memory_binding_groups=memory_gaps,selected_task_variant_by_validation=selected,
        selected_language_variant_by_validation=selected_language,
        caveats=['Cold small-model screening, not broad language or reasoning capability.',
                 'Changed architectures differ in parameter count and compute. All preregistered seeds included.',
                 'CPU training times reflect concurrent single-thread runs on the shared machine; use pinned alternating execution benchmarks separately.',
                 'The first memory follow-up changes supervision and update budget. The subsequent 2x2 ablation holds seeds, batches and 1536 updates fixed and uses a third fresh test set.',
                 'GPU measurements used a busy shared device; correctness established, uncontended speedups undetermined.'])
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    text='# Measured results\n\n'
    text+='## Reasoning: complete response accuracy (%)\n\n'+table(task)+'\n\n'
    text+=f'Variant selected by mean validation accuracy: **{selected}**. This choice does not use test outcomes.\n\n'
    text+='## Language: bits per original byte (lower is better)\n\n'+table(language,1)+'\n\n'
    text+=f'Variant selected by mean validation bits per byte: **{selected_language}**.\n\n'
    text+='## Initial learned memory recipe: accuracy (%)\n\n'+table(memory)+'\n\n'
    text+='## Binding-supervised memory follow-up: accuracy (%)\n\n'+table(binding)+'\n\n'
    text+='| Query placement | Local | Delta |\n|---|---:|---:|\n'
    for key in memory_gaps['local']:
        text+=f"| {key} | {100*memory_gaps['local'][key]:.2f}% | {100*memory_gaps['delta'][key]:.2f}% |\n"
    text+='\n`False` means a far query; `True` means a seven-byte gap. Record counts 16/32/64 correspond to far gaps of 47/95/191 bytes.\n'
    text+='\n## Matched memory supervision ablation: accuracy (%)\n\n'+table(ablation)+'\n\n'
    text+='Same three seeds, 1,536 updates, training batches and validation selection; a third fresh test set. The six binding-supervised checkpoints are reused without further training.\n\n'
    names=('local_response','local_binding','delta_response','delta_binding')
    text+='| Query placement | '+' | '.join(names)+' |\n|---|'+'---:|'*len(names)+'\n'
    for key in ablation_rows[0]['test']['groups']:
        text+='| '+key+' | '+' | '.join(f'{100*ablation_gaps[n][key]:.2f}%' for n in names)+' |\n'
    text+='\n## Equal measured CPU training budget: language bits/byte\n\n'+table(compute,1)+'\n\n'
    text+='150 training process seconds per fit, at most one update overshoot; validation and checkpoint I/O excluded. Same three new seeds and per-update batches, five solver paths, four validation checkpoints. Another 256-document test set excludes the earlier language test documents.\n\n'
    text+='| Solver | Mean updates | Maximum measured training CPU seconds |\n|---|---:|---:|\n'
    for name,r in compute.items():text+=f"| {name} | {r['mean_updates']:.1f} | {r['training_cpu_max']:.3f} |\n"
    (OUT/'RESULTS.md').write_text(text)
    print(json.dumps(dict(selected_task=selected,selected_language=selected_language,total_fits=96)))

if __name__=='__main__':main()
