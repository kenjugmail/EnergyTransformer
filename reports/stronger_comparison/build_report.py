"""Build the report from committed experiment and timing artifacts."""
from pathlib import Path
import json
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from energy_real.runtime import atomic_json, sha256, source_identity

OUT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/compute_matched_controlled'
NAMES = {'softmax': 'Full Transformer', 'local_softmax': 'Windowed Transformer', 'hybrid': 'Energy + local attention'}


def main():
    if (OUT / 'REPORT.md').exists():
        raise FileExistsError('Completed report already exists')
    result = json.loads((RUN / 'results.json').read_text())
    selection = json.loads((RUN / 'selection.json').read_text())
    benchmark = json.loads((OUT / 'benchmark.json').read_text())
    assert result['source_sha256'] == selection['source_sha256'] == benchmark['source_sha256'] == source_identity()
    assert result['selection_sha256'] == sha256(RUN / 'selection.json')
    for fit in selection['fits']:
        assert sha256(ROOT / fit['checkpoint']) == fit['checkpoint_sha256']
    before = json.loads((ROOT / 'runs/stronger_preservation/before.json').read_text())
    frozen = {p: h for p, h in before['files'].items() if p.startswith('reports/') or p.endswith('.pt')}
    assert all(sha256(ROOT / p) == h for p, h in frozen.items())
    chosen = min((r for r in selection['fits'] if r['arm'] == 'hybrid'), key=lambda r: r['best']['valid_bpb'])
    destination = ROOT / 'runs/stronger_native/selected_hybrid.pt'
    destination.parent.mkdir(exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copyfile(ROOT / chosen['checkpoint'], destination)
    atomic_json({'rule': 'Lowest recorded validation bpb among the frozen hybrid fits; no test-based selection',
                 'fit': chosen, 'checkpoint': str(destination.relative_to(ROOT)),
                 'sha256': sha256(destination)}, OUT / 'selected_checkpoint.json')
    quality = result['quality']
    rows = []
    for arm, name in NAMES.items():
        fits = [r for r in selection['fits'] if r['arm'] == arm]
        rows.append(f"| {name} | {quality['128'][arm]['mean_bpb']:.4f} | {quality['512'][arm]['mean_bpb']:.4f} | "
                    f"{statistics.mean(r['cpu_s'] for r in fits):.2f} | {statistics.mean(r['wall_s'] for r in fits):.2f} | "
                    f"{statistics.mean(r['training_bytes'] for r in fits):,.0f} |")
    intervals = []
    for length in ('128', '512'):
        for arm in ('softmax', 'local_softmax'):
            d = quality[length]['hybrid_minus_' + arm]
            intervals.append(f"| {length} | {NAMES[arm]} | {d['mean']:+.4f} | [{d['interval95'][0]:+.4f}, {d['interval95'][1]:+.4f}] |")
    long = benchmark['inference']['32768']
    inference = []
    for key, label in [('full_dynamic','Full Transformer, growing cache'), ('full_static','Full Transformer, preallocated cache'),
                       ('hybrid_old_kernel','Hybrid, preserved kernel'), ('hybrid','Hybrid, improved kernel'),
                       ('local_softmax','Windowed Transformer'), ('energy','Energy only')]:
        r = long[key]
        inference.append(f"| {label} | {r['decode_ms']:.3f} | {r['prefill_s']:.3f} | {r['allocated_state_bytes']:,.0f} |")
    old = benchmark['training']['old_chunk32']['wall_s']
    new = benchmark['training']['new_chunk32']['wall_s']
    small = benchmark['training']['new_chunk8']['wall_s']
    speedup = long['full_static']['decode_ms'] / long['hybrid']['decode_ms']
    state_ratio = long['full_static']['allocated_state_bytes'] / long['hybrid']['allocated_state_bytes']
    verdict = ('The hybrid passed the predeclared quality gate against both Transformer controls at equal CPU training budgets.'
               if result['quality_gate_passed'] else
               'The hybrid did not pass the predeclared quality gate against both Transformer controls at equal CPU training budgets.')
    report = f'''# Stronger native-model comparison

{verdict} This remains a small, one-corpus CPU experiment; it does not establish general superiority over Transformers or large language models.

The energy implementation is faster without changing its mathematical recurrence or weight format. At 32,768 bytes, hybrid decoding is **{speedup:.2f}× as fast as the preallocated full-Transformer baseline**, with **{state_ratio:,.1f}× less allocated cache/state**. The windowed Transformer remains a separate, bounded-memory control.

![Equal-compute quality and optimized inference](comparison.png)

## Equal-compute language prediction

Three freshly initialized seeds per arm, approximately 145,000 parameters, CPU float32, batch eight, 128-byte training sequences. Each architecture gets the same three-rate tuning budget and 90 process CPU seconds of training per main fit. Lower bits per byte are better.

| Model | Test bpb, 128 | Test bpb, 512 | CPU update seconds | Wall update seconds | Training bytes |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

The selected peak learning rates are {selection['rates']}. Each checkpoint is selected on validation only, and all main fits are frozen before test. Faster models receive more examples. This comparison matches measured CPU update time, not FLOPs, GPU compute, or training bytes. The 512-byte test extrapolates from 128-byte training.

Training used one thread without processor affinity on a shared host with different CPU core types. Other work caused high load during seed 303, reducing the number of updates completed within its CPU budget. Interleaved model updates reduce order bias but cannot remove memory-bandwidth contention or core-frequency differences. This limits the precision of the compute comparison.

| Context | Control | Hybrid minus control, bpb | Conditional 95% interval |
|---:|---|---:|---:|
{chr(10).join(intervals)}

Negative differences favor the hybrid. Intervals resample windows after averaging paired differences over the same three seeds; they do not capture all seed uncertainty. Crops can overlap. The equal-budget search is modest and does not establish optimal tuning for either architecture.

## Inference with a stronger Transformer cache

Identical preserved selected weights, 256-byte prompt chunks, batch one, 16 teacher-forced decoding tokens, six balanced rounds. Timings exclude sampling and serving. Full Transformer uses PyTorch scaled-dot-product attention and RoPE. The static version reserves the complete cache, avoids growing KV concatenations, and removes an unnecessary mask during unsegmented single-token decoding.

| Model / implementation | Decode ms per byte, 32K | Prefill seconds | Allocated cache/state bytes |
|---|---:|---:|---:|
{chr(10).join(inference)}

The old timing report used a growing cache. These measurements supersede that implementation comparison, without rewriting the historical result. Allocation counts cover carried tensors, including reserved static capacity; they are not peak process memory. Model weights and transient activations are excluded. Shorter contexts and CPU-time measurements are in [the raw summary](benchmark.json).

Inference CPU affinity was {benchmark['cpu_affinity']}; the host's one-minute load ranged from {benchmark['host_load_1m_range'][0]:.1f} to {benchmark['host_load_1m_range'][1]:.1f}. These shared-host timings are not isolated-machine throughput guarantees; all round values and ranges are retained.

## Kernel improvement

At the original chunk size of 32, median hybrid update time changed from **{old*1000:.2f} to {new*1000:.2f} ms** ({old/new:.2f}×). Chunk size eight measured **{small*1000:.2f} ms** ({old/small:.2f}× versus the preserved implementation); the new training experiment uses this setting. These are matched-weight timing probes, separate from the newly trained models.

Numerical and gradient comparisons use the preserved source and independent serial recurrences. Both energy reads, final states, document resets, carried states and causality are tested. The optional full-attention static cache also matches the original full and chunked outputs. The GPU was occupied; no new accelerator performance or large-model training claim is made.

## Artifacts and scope

- [Protocol and reproduction commands](METHOD.md)
- [Training log](training_controlled.log), [benchmark details](benchmark.json), and [selected checkpoint record](selected_checkpoint.json)
- Full frozen plan, selection, per-seed histories, datasets and test-window rows: `runs/compute_matched_controlled/`
- Validation-selected hybrid: `runs/stronger_native/selected_hybrid.pt` (seed {chosen['seed']})
- Frozen training source: `runs/stronger_preservation/training_source.tar.gz`
- {len(frozen)} earlier reports and checkpoint artifacts verified unchanged.

The earlier large checkpoints were preserved. This is a mixed energy-and-attention model, not an energy-only replacement for every Transformer component. Long-prefix timing does not measure how well either model uses distant information. One small corpus, short training contexts and native CPU timings cannot settle large-scale language quality, reasoning, GPU throughput or production serving performance.
'''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = ['#527fa0', '#b48242', '#26836d']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), constrained_layout=True)
    for i, (arm, label) in enumerate(NAMES.items()):
        axes[0].bar(np.arange(2)+(i-1)*.23, [quality[n][arm]['mean_bpb'] for n in ('128','512')], .22, label=label, color=colors[i])
    axes[0].set_xticks([0,1], ['128 bytes','512 bytes'])
    axes[0].set_ylabel('Test bits per byte ↓')
    axes[0].set_title('Equal CPU training budget')
    axes[0].legend(fontsize=8)
    for i, (arm, label) in enumerate([('full_static','Full / static'),('local_softmax','Windowed'),('hybrid','Hybrid')]):
        axes[1].bar(i, long[arm]['decode_ms'], color=colors[i])
        axes[2].bar(i, long[arm]['allocated_state_bytes'], color=colors[i])
    for ax in axes[1:]:
        ax.set_xticks(range(3), ['Full / static','Windowed','Hybrid'], rotation=15)
    axes[1].set_title('Decode after a 32K prefix')
    axes[1].set_ylabel('Milliseconds per byte ↓')
    axes[2].set_title('Allocated cache / state at 32K')
    axes[2].set_ylabel('Bytes, logarithmic scale ↓')
    axes[2].set_yscale('log')
    fig.savefig(OUT / 'comparison.png', dpi=180)
    fig.savefig(OUT / 'comparison.pdf')
    plt.close(fig)
    (OUT / 'REPORT.md').write_text(report)
    for name in ('plan.json','results.json','selection.json','rates.json','test_rows.json','pilot_fits.json'):
        shutil.copyfile(RUN/name, OUT/name)
    print(verdict)
    print(json.dumps({'training_speedup_same_chunk':old/new, 'training_speedup_chunk8':old/small,
                      'decode_speedup_vs_static_full':speedup,'state_ratio_vs_static_full':state_ratio}))


if __name__ == '__main__':
    main()
