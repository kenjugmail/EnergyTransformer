"""Publish the pinned, compute-block confirmation from frozen artifacts."""
from pathlib import Path
import json
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from energy_real.runtime import atomic_json, sha256, source_identity

OUT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/compute_confirmation_v2'
LABELS = {'softmax': 'Full Transformer', 'local_softmax': 'Windowed Transformer', 'hybrid': 'Energy + local attention'}


def main():
    if (OUT / 'REPORT.md').exists():
        raise FileExistsError('Report already completed')
    plan = json.loads((RUN / 'plan.json').read_text())
    result = json.loads((RUN / 'results.json').read_text())
    selection = json.loads((RUN / 'selection.json').read_text())
    assert result['status'] == 'completed'
    assert result['source_sha256'] == plan['source_sha256'] == selection['source_sha256'] == source_identity()
    assert result['selection_sha256'] == sha256(RUN / 'selection.json')
    assert selection['plan_sha256'] == sha256(RUN / 'plan.json')
    assert sha256(ROOT / 'data/enwik8') == plan['corpus_sha256']
    verification = json.loads((OUT / 'compiled_verification.json').read_text())
    assert verification['source_sha256'] == source_identity()
    assert plan['cpu_affinity'] == [19] and plan['quantum_cpu_s'] == .5
    assert plan['compiled_training'] and plan['optimizer_foreach']
    assert len(selection['fits']) == 9
    for fit in selection['fits']:
        assert sha256(ROOT / fit['checkpoint']) == fit['checkpoint_sha256']
        assert 90 <= fit['cpu_s'] <= 90 + fit['max_update_cpu_s'] + 1e-7
    scheduler_checks = []
    for path in sorted(RUN.glob('*_scheduler.json')):
        rows = json.loads(path.read_text())
        target = 20 if path.name.startswith('pilot_') else 90
        assert len(rows) == round(target / .5)
        spread = max(max(r['cpu_s'].values()) - min(r['cpu_s'].values()) for r in rows)
        assert all(min(r['cpu_s'].values()) >= min(target, (r['round']+1)*.5)-1e-7 for r in rows)
        scheduler_checks.append({'file': path.name, 'rounds': len(rows), 'max_round_cpu_spread_s': spread})
    old = json.loads((ROOT / 'runs/stronger_preservation/before.json').read_text())['files']
    frozen = {p:h for p,h in old.items() if p.startswith('reports/') or p.endswith('.pt')}
    frozen.update(json.loads((ROOT / 'runs/stronger_preservation/phase1_files.json').read_text()))
    assert all(sha256(ROOT / p) == h for p,h in frozen.items())
    chosen = min((f for f in selection['fits'] if f['arm'] == 'hybrid'), key=lambda f:f['best']['valid_bpb'])
    destination = ROOT / 'runs/compute_confirmation_selected/hybrid.pt'
    destination.parent.mkdir(exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copyfile(ROOT / chosen['checkpoint'], destination)
    selected = {'rule': 'Lowest frozen validation loss among hybrid seeds; test scores not used',
                'fit': chosen, 'path': str(destination.relative_to(ROOT)), 'sha256': sha256(destination)}
    atomic_json(selected, OUT / 'selected_checkpoint.json')
    atomic_json({'schedulers': scheduler_checks, 'preserved_artifact_count': len(frozen)}, OUT / 'audit.json')
    quality = result['quality']
    table, seeds, intervals = [], [], []
    for arm, label in LABELS.items():
        fits = [f for f in selection['fits'] if f['arm'] == arm]
        table.append(f"| {label} | {quality['128'][arm]['mean_bpb']:.4f} | {quality['512'][arm]['mean_bpb']:.4f} | "
                     f"{statistics.mean(f['cpu_s'] for f in fits):.2f} | {statistics.mean(f['wall_s'] for f in fits):.2f} | "
                     f"{statistics.mean(f['training_bytes'] for f in fits):,.0f} |")
        seeds.append(f"| {label} | " + ' | '.join(f'{v:.4f}' for v in quality['128'][arm]['seed_bpb']) + ' |')
    for context in ('128','512'):
        for control in ('softmax','local_softmax'):
            d = quality[context]['hybrid_minus_' + control]
            intervals.append(f"| {context} | {LABELS[control]} | {d['mean']:+.4f} | [{d['interval95'][0]:+.4f}, {d['interval95'][1]:+.4f}] |")
    histories = [json.loads(p.read_text()) for p in RUN.glob('main_*_history.json')]
    loads = [r['host_load_1m'] for history in histories for r in history]
    pilots = json.loads((RUN / 'pilot_fits.json').read_text())
    setups = {arm:round(sum(f['compile_setup_wall_s'] for f in selection['fits'] + pilots if f['arm']==arm),3) for arm in LABELS}
    verdict = ('The hybrid passed the predeclared prediction-quality gate against both Transformer controls at equal steady-state CPU update budgets.'
               if result['quality_gate_passed'] else
               'The hybrid did not pass the predeclared prediction-quality gate against both Transformer controls at equal steady-state CPU update budgets.')
    report = f'''# Compute-controlled confirmation

{verdict} This is evidence about approximately 145,000-parameter byte models on one corpus. It is not evidence of general superiority over modern large Transformers.

![Confirmation quality and training curves](confirmation.png)

## What changed in the comparison

The [first follow-up](../stronger_comparison/REPORT.md) was effectively tied at the trained context length. Machine contention exposed a weakness in rotating one update at a time: the more expensive model finished its CPU allowance earlier and experienced a different period of host load.

This confirmation pins every arm to CPU 19, rotates **0.5-CPU-second blocks**, compiles every training objective, and uses foreach AdamW for every arm. The compiler check verified gradients for both controls and the hybrid; forward/backward CPU time fell from 27.2 to 20.5 ms for the hybrid, and by about 8–9% for the controls. No layer arrangement, parameter budget, memory reader, or language-model recurrence was changed.

Each arm gets three 20-CPU-second learning-rate pilot fits and three fresh 90-CPU-second main fits. Learning rates and checkpoints are selected on validation; all selections are frozen before fresh test-window evaluation. Training is float32, batch eight, 128 bytes, with the same minibatch-stream prefix within each seed. Faster models see more bytes.

## Held-out prediction

Mean bits per byte across three validation-selected seeds; lower is better. CPU and wall seconds cover updates only.

| Model | Test bpb, 128 | Test bpb, 512 | CPU seconds | Wall seconds | Training bytes |
|---|---:|---:|---:|---:|---:|
{chr(10).join(table)}

| Model | Seed 404, 128 bytes | Seed 505, 128 bytes | Seed 606, 128 bytes |
|---|---:|---:|---:|
{chr(10).join(seeds)}

| Context | Control | Hybrid minus control, bpb | Conditional 95% interval |
|---:|---|---:|---:|
{chr(10).join(intervals)}

Negative differences favor the hybrid. Bootstrap intervals resample windows after averaging paired differences across these three seeds. They do not quantify all seed uncertainty; random crops can overlap. The 512-byte evaluation extrapolates beyond the 128-byte training context. It is not a long-distance recall test.

## Compute and measurement limits

The largest CPU-progress spread at a completed main scheduling round was **{max(r['max_round_cpu_spread_s'] for r in scheduler_checks if r['file'].startswith('main_')):.4f} seconds**. Validation-point host load ranged from {min(loads):.1f} to {max(loads):.1f}. A fixed core and short blocks improve the comparison, but this is still a shared machine; CPU seconds are not FLOPs or GPU compute.

The equal budgets cover **steady-state updates**. Compilation, data loading, validation and checkpoint I/O are excluded equally. Recorded compiler warmup wall-time totals across pilot and main fits were {setups} seconds. Those warmups reused a compiler cache populated by the probes and an aborted pilot; they are not cold-start costs. The [compiler probe](../stronger_comparison/compile_probe.json) separately records its initial compile costs. No end-to-end training-cost win is claimed from the update-only comparison.

An initial confirmation pilot was stopped after discovering that contiguous dummy inputs did not match the real overlapping crop views, causing recompilation inside the timer. Warmup now matches the input strides and aliasing; the failed attempt and its log remain preserved. No test evaluation or checkpoint promotion occurred in that attempt.

## Implementation and inference

The CPU kernel now reduces the slot contraction explicitly and handles one-token recurrence directly. The preserved-weight timing experiment measured **1.75× faster updates at the same chunk size**, **2.05× with chunk size eight**, and **3.15× faster 32K-prefix decoding than the preallocated full Transformer**, with **1,363× less carried cache/state**.

Those inference measurements use the earlier preserved weights and eager CPU execution, separately from this compiled-training quality experiment. The preallocated full Transformer is faster at 4K; windowed attention is faster at all measured lengths. Small cache size does not establish better long-distance recall. Full timing rows, cache capacity accounting and scope are in [the implementation comparison](../stronger_comparison/REPORT.md).

## Artifacts

- [Frozen plan](plan.json), [results](results.json), [selection](selection.json), [full training log](training_v2.log), [audit](audit.json)
- [Selected native checkpoint record](selected_checkpoint.json): `runs/compute_confirmation_selected/hybrid.pt`, seed {chosen['seed']}
- All fit checkpoints, input windows, histories and scheduling rows: `runs/compute_confirmation_v2/`
- Frozen source: `runs/stronger_preservation/confirmation_v2_source.tar.gz`
- {len(frozen)} earlier report/checkpoint artifacts verified unchanged.

The selected checkpoint remains compatible with `energy_real.sample`. Earlier large checkpoints were preserved. GPU throughput, large-model quality, broad language tasks, reasoning and production serving remain unestablished.

Compiled/eager checks on real overlapping crop views and trained checkpoints passed, with maximum absolute gradient error {max(r['max_gradient_error'] for r in verification['rows']):.3g}. See [gradient verification](compiled_verification.json), [root tests](tests_root.log), and [Spark compatibility tests](tests_spark.log). The initial test-discovery attempt traversed archived source copies; the complete live suites were then run explicitly by directory.

An [isolated wheel installation](installation_verification.json) verifies all 53 package source files, the selected checkpoint, chunked sampling, exact CLI bytes and preallocated-cache parity outside the checkout.
'''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    colors = {'softmax':'#527fa0','local_softmax':'#b48242','hybrid':'#26836d'}
    for index, (arm,label) in enumerate(LABELS.items()):
        axes[0].bar(np.arange(2)+(index-1)*.24, [quality[n][arm]['mean_bpb'] for n in ('128','512')], .23, label=label,color=colors[arm])
        # Checkpoint-derived names avoid matching local_softmax with softmax's suffix.
        arm_histories = [json.loads(RUN.joinpath(Path(f['checkpoint']).stem + '_history.json').read_text())
                         for f in selection['fits'] if f['arm'] == arm]
        ys = np.array([[r['valid_bpb'] for r in history] for history in arm_histories])
        xs = [22.5,45,67.5,90]
        axes[1].plot(xs,ys.mean(0),marker='o',label=label,color=colors[arm])
        axes[1].fill_between(xs,ys.min(0),ys.max(0),alpha=.12,color=colors[arm])
    axes[0].set_xticks([0,1],['128 bytes','512 bytes'])
    axes[0].set_ylabel('Test bits per byte ↓')
    axes[0].set_title('Equal steady-state CPU budgets')
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel('CPU update seconds per fit')
    axes[1].set_ylabel('Validation bits per byte ↓')
    axes[1].set_title('Mean and range across three seeds')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
    fig.savefig(OUT/'confirmation.png',dpi=180)
    fig.savefig(OUT/'confirmation.pdf')
    plt.close(fig)
    (OUT/'REPORT.md').write_text(report)
    for name in ('plan.json','results.json','selection.json','rates.json','test_rows.json','pilot_fits.json'):
        shutil.copyfile(RUN/name,OUT/name)
    print(verdict)
    print(json.dumps(result['quality'],indent=2))


if __name__ == '__main__':
    main()
