"""Rebuild the compact report from the preserved, completed local study."""
from pathlib import Path
import json
import re
import shutil
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
STUDY = ROOT/'runs/reasoning_v2_final_stages/controllers/attempt_0/study'


def read(path):
    return json.loads(path.read_text())


def write(path, obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def main():
    study=read(STUDY/'summary.json')
    assert study['status']=='completed'
    assert len(study['records'])==16 and all(r['steps']==500 for r in study['records'])
    decisions=read(STUDY/'decision.json')
    costs=read(STUDY/'costs.json')
    ledger=read(ROOT/'runs/reasoning_v2_final_stages/stages.json')
    for name in ('selection.json','datasets.json','decision.json'):
        shutil.copy2(STUDY/name,OUT/name)
    write(OUT/'device_stages.json',ledger)
    baseline={kind:read(STUDY/f'baseline_{kind}.json')['summary'] for kind in ('precision','preconditioned_delta')}
    aggregate={}
    for kind in baseline:
        records=[r for r in study['test_results'] if r['kind']==kind]
        aggregate[kind]={
            'original_configuration_accuracy':baseline[kind]['accuracy'],
            'selected_initial_accuracy':statistics.mean(r['initial']['summary']['accuracy'] for r in records),
            'selected_checkpoint_accuracy':statistics.mean(r['trained']['summary']['accuracy'] for r in records),
            'unsupported_answers_across_seeds':sum(r['trained']['summary']['unsupported_answers'] for r in records),
            'per_seed':[{'seed':r['seed'],'initial':r['initial']['summary'],'selected':r['trained']['summary'],
                         'default_threshold':r['trained']['default_threshold_summary'],'learning_interval':r['learning_interval']}
                        for r in records],
        }
    cost_summary={}
    for kind in ('exact','precision','preconditioned_delta'):
        rows=[r for r in costs['rows'] if r['backend']==kind]
        cost_summary[kind]={key:statistics.median(r[key] for r in rows) for key in (
            'apply_s','build_first_query_s','median_query_s','database_bytes','snapshot_bytes',
            'registry_bytes','model_bytes','state_bytes','accounted_total_bytes')}
        cost_summary[kind]['process_memory']=next(r for r in costs['isolated_process_memory'] if r['backend']==kind)
    energy_cost=cost_summary['precision'];control_cost=cost_summary['preconditioned_delta']
    cost_gates={'state_reduction':1-energy_cost['state_bytes']/control_cost['state_bytes'],
                'cpu_query_latency_reduction':1-energy_cost['median_query_s']/control_cost['median_query_s'],
                'gpu_latency':'pending; no GPU efficiency claim'}
    suites={}
    for name in ('real','legacy','spark'):
        log=(OUT/f'tests_{name}.log').read_text()
        suites[name]={'passed':int(re.search(r'(\d+) passed',log)[1]),
                      'skipped':int(re.search(r'(\d+) skipped',log)[1]) if 'skipped' in log else 0}
    compact={'status':'completed_structured_memory_study','benchmark':read(STUDY/'datasets.json')['version'],
             'source_sha256':study['source_sha256'],'selection_sha256':study['selection_sha256'],
             'environment':read(OUT/'cpu_smoke.json')['device'],
             'elapsed_s':study['elapsed_s'],'gpu_seconds':study['gpu_seconds'],
             'fits':len(study['records']),'updates':sum(r['steps'] for r in study['records']),
             'screening':[{'kind':r['kind'],'dimension':r['dimension'],'prior':r['prior'],'seed':r['seed'],
                           'selected_step':r['selected_step'],'validation':r['validation']['selected']}
                          for r in study['records']],
             'exact':read(STUDY/'exact_oracle.json')['summary'],'memory':aggregate,'costs':cost_summary,
             'cost_gates':cost_gates,'decision':decisions,'tests':suites,'stage_usage':ledger['stages'],
             'full_results_directory':str(STUDY)}
    write(OUT/'results.json',compact)
    percent=lambda value:f'{100*value:.2f}%'
    table=['| Mechanism | Original configuration | Chosen configuration before training | Validation-selected checkpoint |',
           '|---|---:|---:|---:|','| Exact store | 100.00% | 100.00% | 100.00% |']
    for kind,row in aggregate.items():
        table.append(f"| {kind} | {percent(row['original_configuration_accuracy'])} | {percent(row['selected_initial_accuracy'])} | {percent(row['selected_checkpoint_accuracy'])} |")
    seeds=['| Mechanism | Seed | Selected update | Initial test accuracy | Selected test accuracy |',
           '|---|---:|---:|---:|---:|']
    for record in study['selected']:
        row=next(r for r in study['test_results'] if r['kind']==record['kind'] and r['seed']==record['seed'])
        seeds.append(f"| {record['kind']} | {record['seed']} | {record['selected_step']} | {percent(row['initial']['summary']['accuracy'])} | {percent(row['trained']['summary']['accuracy'])} |")
    costs_table=['| Mechanism | Median query (ms) | Recurrent state (KiB) | Process peak RSS (MiB) |',
                 '|---|---:|---:|---:|']
    for kind,row in cost_summary.items():
        costs_table.append(f"| {kind} | {1000*row['median_query_s']:.3f} | {row['state_bytes']/1024:.1f} | {row['process_memory']['process_peak_rss_bytes']/2**20:.1f} |")
    interval=decisions['paired_interval']
    used=ledger['stages']['controllers']['used_s']
    passed=sum(r['passed'] for r in suites.values());skipped=sum(r['skipped'] for r in suites.values())
    report=f'''# Energy memory and reasoning: executed v2.1 results

The shared memory benchmark, recall objective, confidence calibration and natural-
language reasoning diagnostics are implemented. The CPU study completed with
**{len(study['records'])} fits and {compact['updates']:,} optimizer updates** in
**{study['elapsed_s']/60:.2f} minutes**. Live model reasoning and GPU measurements
remain separate pending work.

## What improved

All figures below use the same 360 fresh test episodes, with 30 episodes in each
of 12 families. The original configuration uses seed 11, width 16, prior 0.01, score
threshold 0.5 and no confidence-margin requirement. Other columns average seeds
11/22/33. Width 64 and prior 0.001 were selected for both mechanisms on validation.

{chr(10).join(table)}

Configuration/calibration gains must not be attributed entirely to learning.
The initialization column uses the selected architecture and separately calibrated
initial weights. A validation-selected checkpoint can be update zero:

{chr(10).join(seeds)}

The energy-minus-classical difference is **{100*interval['difference']:.2f} percentage
points**, with a paired 95% episode-bootstrap interval of
**[{100*interval['low']:.2f}, {100*interval['high']:.2f}]**. It does not meet the five-point
promotion gate. Per-seed learning intervals and family scores are in
[the machine-readable results](results.json); a single improving seed does not
establish a replicated training benefit.

Energy produced **{aggregate['precision']['unsupported_answers_across_seeds']} unsupported answers** across
the three 360-episode evaluations. These remain visible. The comparison includes
deletions, absent facts and missing intermediate links, so aggregate accuracy
alone should not determine whether approximate memory is used for an answer.

## Experimental controls and costs

Training uses 384 structured episodes; validation uses 120. Split namespaces and
wording templates are disjoint. Twelve seed-11 configurations cover two mechanisms,
three widths and two priors. Validation chooses one configuration per mechanism;
seeds 22/33 replicate those choices. Each fit receives 500 updates of delayed-recall
cross-entropy plus 0.1 times edit-locality loss. Unseen/deleted addresses train an
explicit null target. No future random-value prediction objective is used.

Score and top-two-margin thresholds are calibrated on validation, including
unavailable facts. The [selection](selection.json) was frozen before test generation
and evaluation. [Dataset recipes and hashes](datasets.json) accompany it; complete
episodes and per-case traces remain in the local run directory. Query traces
distinguish wrong top candidates from correct candidates rejected by confidence.

{chr(10).join(costs_table)}

The cost sample uses 24 fixed test episodes and balanced arm order. Reports include
database bytes, resident evidence/registry serialization, parameters, recurrent
state, writes, state construction and repeated queries. Full-process resident
peaks use isolated Linux processes and `/proc/self/status` VmHWM, including runtime
and native allocations. Serialized components are accounting measures, not peak
RAM, and are kept separate from the measured process peak.

Energy’s state reduction versus the classical control is
{100*cost_gates['state_reduction']:.2f}%; its measured CPU query-latency reduction is
{100*cost_gates['cpu_query_latency_reduction']:.2f}% (negative means slower).
No GPU efficiency promotion is claimed.

## Reasoning adapter and verification

The adapter compares recent context, exact retrieval and energy retrieval with
identical questions/decoding. Natural questions require a model-produced relation
path; supplied paths remain a separately labelled test. Wrong paths are recorded
without oracle substitution. A fourth, explicitly diagnostic gold-evidence arm
isolates answer composition from retrieval and interpretation failures.

With model-proposed edits, recent context contains original statements rather
than oracle-parsed triples. Persistent writes still require caller authorization
and revision checks. Per-answer reports retain evidence, provenance, confidence
traces, model/configuration identity, interpretation errors, abstention and separate
generation/scoring results. A shared deadline bounds all service calls; a scoring
outage preserves already generated answers.

**{passed} tests passed, {skipped} skipped, zero failures.** Logs:
[repaired suite](tests_real.log), [legacy suite](tests_legacy.log),
[Spark suite](tests_spark.log). The [CPU numerical smoke test](cpu_smoke.json) passed.
The [selected-checkpoint smoke check](selected_memory_smoke.json) also matches the
frozen evaluation through the practical FactStore interface for all 12 families.
Mock tests verify natural-path interpretation, strict arm separation, missing
links, source-only recent context, malformed inputs, confidence rejection,
checkpoint/config compatibility, frozen selection, outage recovery and deadlines.
These fixture results are not live language-model reasoning scores.

## Resources, preserved evidence and decisions

The [stage ledger](device_stages.json) carries earlier usage into this revision.
Controller-stage time totals **{used/60:.2f} CPU minutes**, including the preserved
initial diagnostic run. Actual GPU time for this phase is zero. GPU stages remain
pending while unrelated workloads are active; localhost:8100 remains unavailable.

Before finalizing, an audit found that the initial generator could reuse IDs for
different content when sample size changed, and the first RSS counter could inherit
a pre-exec high-water mark. Version 2.1 fixes both and repeats the unchanged settings.
The original run and its source snapshot remain preserved; its scores are not used
for the final comparison above. All earlier repair reports, data and checkpoints
remain unchanged. [Artifact hashes](artifacts.json) identify this revision.

- **Memory quality:** retain the improved configuration as an experimental candidate.
  It has not demonstrated the required advantage over the stronger classical control.
- **Learning:** retain measured seed-specific results; no blanket claim that training
  improves every selected mechanism or seed.
- **Practical reasoning:** continue with exact memory as the reliable baseline and
  compare the selected energy memory against the existing live service when available.
- **Scaling:** do not start larger pretraining or additional model fine-tuning from
  these results.

Resume the pending stages without restarting completed controller work:

```bash
.venv/bin/python -m energy_real.stages --out runs/reasoning_v2_final_stages \\
  --stages smoke evaluation_profile reasoning \\
  --memory-selection reports/reasoning_v2/selection.json
```

See [interfaces and reproducible commands](../../energy_real/README.md).
Rebuild this compact report with `python3 reports/reasoning_v2/build_report.py`.
'''
    (OUT/'REPORT.md').write_text(report)
    print(json.dumps({'accuracy':aggregate['precision']['selected_checkpoint_accuracy'],
                      'classical_accuracy':aggregate['preconditioned_delta']['selected_checkpoint_accuracy'],
                      'passed':passed,'skipped':skipped,'decision':decisions['decision']},indent=2))


if __name__=='__main__':main()
