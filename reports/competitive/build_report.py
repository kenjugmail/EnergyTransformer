"""Build the report from frozen primary and predeclared profiling results."""
from pathlib import Path
import json
import math
import shutil
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from energy_real.competitive_study import ARMS, summarize
from energy_real.runtime import atomic_json, sha256, source_identity


def main():
    report = Path(__file__).resolve().parent
    root = report.parents[1]
    run = root / 'runs/competitive_native'
    load = lambda name: json.loads((run/name).read_text())
    selected, result, plan = load('selection.json'), load('results.json'), load('plan.json')
    assert result['status']=='completed'
    assert source_identity()==selected['source_sha256']==result['source_sha256']
    assert sha256(run/'selection.json')==result['selection_sha256']
    assert load('long_context/result.json')['status']=='completed'
    fits, tests = load('fits.json'), load('test_rows.json')
    rows = load('profile_rows.json') + load('long_context/rows.json')
    summary = summarize(fits, tests, rows)
    atomic_json(summary, report/'results.json')
    for name in ('plan.json','selection.json','fits.json'):
        shutil.copy2(run/name, report/name)
    for arm in ARMS:
        seed = selected['selected_seeds'][arm]
        fit = next(f for f in fits if f['arm']==arm and f['seed']==seed)
        src = root / fit['checkpoint']
        assert sha256(src)==fit['checkpoint_sha256']
        shutil.copy2(src, run/f'selected_{arm}.pt')
    shutil.copy2(root/'runs/local_energy_budget.json', report/'budget.json')
    names = {'energy':'Energy only','softmax':'Full Transformer','local_softmax':'Windowed Transformer','hybrid':'Mixed energy + attention'}
    colors = {'energy':'#8B5CF6','softmax':'#64748B','local_softmax':'#D97706','hybrid':'#008B8B'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})
    figure, axes = plt.subplots(1,3,figsize=(14,4.3),layout='constrained')
    for arm in ARMS:
        values = summary['quality'][128][arm]['seed_bpb']
        i = list(ARMS).index(arm)
        axes[0].bar(i,statistics.mean(values),color=colors[arm],alpha=.85)
        axes[0].scatter(np.full(len(values),i),values,s=16,color='#172033',zorder=4)
    axes[0].set_xticks(range(4),['Energy','Full\nattention','Windowed\nattention','Mixed'])
    axes[0].set_ylabel('Test bits per byte · lower is better')
    axes[0].set_title('Language prediction · context 128')
    contexts = sorted(summary['efficiency'])
    for arm in ARMS:
        axes[1].plot(contexts,[summary['efficiency'][n][arm]['state_bytes']/1024 for n in contexts],
                     marker='o',color=colors[arm],label=names[arm])
        axes[2].plot(contexts,[summary['efficiency'][n][arm]['decode_ms'] for n in contexts],
                     marker='o',color=colors[arm],label=names[arm])
    for ax in axes[1:]:
        ax.set_xscale('log',base=2)
        ax.set_xlabel('Prompt bytes · batch 1')
        ax.grid(alpha=.18)
    axes[1].set_yscale('log',base=10)
    axes[1].set_ylabel('Carried tensor state · KiB')
    axes[1].set_title('Inference cache · after 16 tokens')
    axes[2].set_ylabel('CPU ms per decoded token')
    axes[2].set_title('Median of six balanced rounds')
    axes[2].legend(frameon=False,fontsize=8)
    figure.suptitle('Small native-model comparison · 3 seeds · matched parameters and training bytes',fontsize=13)
    figure.savefig(report/'comparison.png')
    figure.savefig(report/'comparison.pdf')
    plt.close(figure)
    fig, axes = plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for arm in ARMS:
        histories = [load(f'history_{seed}.json')[arm] for seed in plan['seeds']]
        bpb = np.mean([[r['valid_bpb'] for r in h] for h in histories],axis=0)
        steps = [r['step'] for r in histories[0]]
        seconds = np.mean([[r['train_s'] for r in h] for h in histories],axis=0)
        axes[0].plot(np.array(steps)*1024/1e6,bpb,label=names[arm],color=colors[arm])
        axes[1].plot(seconds,bpb,label=names[arm],color=colors[arm])
    axes[0].set_xlabel('Training bytes · millions')
    axes[1].set_xlabel('Measured optimizer-update time · CPU seconds')
    for ax in axes:
        ax.set_ylabel('Validation bits per byte')
        ax.grid(alpha=.18)
        ax.set_ylim(2.5,5.2)
    axes[1].legend(frameon=False,fontsize=8)
    fig.suptitle('Validation loss by training data and CPU time')
    fig.savefig(report/'training.png')
    fig.savefig(report/'training.pdf')
    plt.close(fig)
    q = summary['quality']
    e = summary['efficiency'][32768]
    reduction = e['softmax']['state_bytes']/e['hybrid']['state_bytes']
    ratio = e['softmax']['decode_ms']/e['hybrid']['decode_ms']
    quality_pass = all(q[n]['hybrid_minus_softmax']['interval95'][1] <= 0 for n in (128,512))
    window_pass = all(q[n]['hybrid_minus_local_softmax']['interval95'][1] <= 0 for n in (128,512))
    lines = [
        '# Native energy / Transformer comparison', '',
        ('The mixed model beats the full-attention Transformer on the measured test loss at both contexts, with conditional paired-window intervals below zero.'
         if quality_pass else 'The mixed model does not pass the predeclared language-quality gate against the full-attention Transformer at both tested contexts.'), '',
        f'At a 32,768-byte prompt it uses **{reduction:,.1f}× less carried tensor state** than full attention. '
        f'The full-attention / mixed median decode-latency ratio is **{ratio:.2f}×** on this CPU (above one favors mixed).', '',
        '**Scope:** three freshly trained seeds per arm, approximately 145,000 parameters, 1.024 million training bytes per fit, '
        'one preserved local enwik8 corpus, CPU float32. This is not evidence of general superiority over well-tuned or large Transformers. '
        'The existing 10.6-million-parameter checkpoints were preserved; they were not retrained or replaced.', '',
        '![Language quality, state and decoding measurements](comparison.png)', '',
        '## Test language prediction', '',
        'Each cell is mean bits per byte across all three validation-selected checkpoints; lower is better. '
        'Test windows reset model state. The 512-byte evaluation is a context-length extrapolation from 128-byte training.', '',
        '| Model | Parameters | Test, 128 bytes | Test, 512 bytes | Mean training seconds |',
        '|---|---:|---:|---:|---:|']
    for arm in ARMS:
        lines.append(f'| {names[arm]} | {summary["parameters"][arm]:,} | {q[128][arm]["mean_bpb"]:.4f} | '
                     f'{q[512][arm]["mean_bpb"]:.4f} | {summary["train_s"][arm]:.2f} |')
    lines += ['', 'Hybrid-minus-control differences in bits per byte (negative favors mixed):', '',
              '| Context | Control | Mean difference | Conditional 95% interval |','|---:|---|---:|---:|']
    for context in (128,512):
        for arm in ('energy','softmax','local_softmax'):
            delta = q[context]['hybrid_minus_'+arm]
            lo, hi = delta['interval95']
            lines.append(f'| {context} | {names[arm]} | {delta["mean"]:+.4f} | [{lo:+.4f}, {hi:+.4f}] |')
    lines += ['', 'Intervals resample test windows after averaging paired differences over the three seeds. '
              'They are conditional on these trained seeds and do not quantify all training-seed uncertainty; random corpus crops can overlap. '
              'Per-seed losses and window rows are preserved. Every arm sees identical minibatches within each seed. '
              'The common optimizer schedule was fixed without architecture-specific hyperparameter tuning.', '',
              f'Full-Transformer quality gate: **{"passed" if quality_pass else "not passed"}**. '
              f'Additional comparison with the windowed Transformer: **{"passed at both contexts" if window_pass else "not established at both contexts"}**.', '',
              '![Validation loss against both training bytes and update time](training.png)', '',
              'Compute is deliberately reported separately. Equal optimizer updates and tokens do not make these different mixers FLOP- or wall-time-matched. '
              'The timing includes forward/backward/optimizer work, excluding validation and checkpoint I/O. '
              'The data-matched test result must not be described as a compute-matched win.', '',
              '## Long-context inference', '',
              'Median of six rounds, batch one, 16 identical teacher-forced decoding tokens after prefill. '
              'Arm order rotates and reverses across rounds. These are native forward-pass timings, excluding token sampling and request serving. '
              'The GPU was occupied by an unrelated workload and was not used.', '',
              '| Model | State after 32,768 + 16 bytes | Decode ms/token | Prefill seconds |','|---|---:|---:|---:|']
    for arm in ARMS:
        row=e[arm]
        lines.append(f'| {names[arm]} | {int(row["state_bytes"]):,} B | {row["decode_ms"]:.3f} | {row["prefill_s"]:.3f} |')
    lines += ['', 'State counts include all returned keys, values, recurrent tensors, segment IDs, and absolute-position counters. '
              'They exclude parameters, activations, optimizer and process overhead. Parameter bytes are reported separately in the JSON. '
              'The windowed Transformer also has bounded state and is an essential control; cache reduction alone is not an energy-specific discovery. '
              'The full-attention implementation extends its cache with tensor concatenation; optimized static or paged caches and GPU kernels were not measured. '
              'There is no 32K language-quality result in this study.', '',
              'Full attention decodes faster than mixed at the measured 128-, 1024-, and 4096-byte prompts. '
              'Mixed is faster at 8192 and 32768 bytes. The windowed Transformer decodes faster than mixed at every measured length, '
              'while mixed has lower test loss and less carried state. This is a measured tradeoff, not dominance on every metric.', '',
              '## Implementation and verification', '',
              '- A configurable mixed layer schedule and bounded local attention are integrated into the native `LM` and resumable trainer.',
              '- The full attention and energy-only defaults retain their old layouts and bitwise checkpoint behavior.',
              '- Causality, independently recomputed attention outputs/gradients, mixed streaming/reset behavior, compact cache storage, activation checkpointing, fact-reader composition, and exact training resume are checked.',
              '- Root suites: 265 passed, 32 skipped. Spark: 83 passed, 2 skipped. Total: **348 passed, 34 skipped**.',
              '- A non-editable wheel imports outside the checkout, matches installed source bytes, and passes mixed-model streaming plus local-PC-ALM backward smoke checks.', '',
              'See [architecture and commands](ARCHITECTURE.md), [backward compatibility](backward_compatibility.json), '
              '[installation](installation.json), [root tests](tests_root.log), and [Spark tests](tests_spark.log).', '',
              'A [greedy-generation smoke check](sampling_smoke.json) on the first completed mixed checkpoint loads and streams correctly, '
              'but repeats short phrases. The lower byte-prediction loss should not be confused with assistant-quality generation.', '',
              '## Artifacts and selection', '',
              'All arms choose their lowest-validation-loss checkpoint, including initialization. The complete selection was frozen before test prediction. '
              'The selected seed for each arm is also chosen by validation, and is used for inference profiling. '
              'The long-context extension was declared during seed-11 training and does not change model selection.', '',
              '[Selected mixed checkpoint](../../runs/competitive_native/selected_hybrid.pt) · '
              '[Selected full Transformer](../../runs/competitive_native/selected_softmax.pt) · '
              '[Results](results.json) · [Frozen plan](plan.json) · [Selection](selection.json) · '
              '[Fit costs](fits.json) · [Per-window test scores](../../runs/competitive_native/test_rows.json) · '
              '[Profile rounds](../../runs/competitive_native/profile_rows.json) · '
              '[Long-context plan](long_context_plan.json) · [Long-context rounds](../../runs/competitive_native/long_context/rows.json)', '',
              '[Training source archive](../../runs/competitive_preservation/training_source.tar.gz) · '
              '[Source hashes](training_source.json) · [Shared cumulative time ledger](budget.json) · '
              '[Non-editable wheel](../../runs/competitive_install/wheels/energy_transformer_research-0.16.0-py3-none-any.whl)', '',
              'The [artifact manifest](artifacts.json) records the final files and preserved-checkpoint audit.', '',
              'Checkpoints, datasets and per-window scores remain local under `runs/`. Earlier source and reports were archived before editing; '
              'earlier report bytes and all three original model checkpoint hashes are preserved.', '']
    (report/'REPORT.md').write_text('\n'.join(lines))
    atomic_json({'quality_gate':quality_pass,'windowed_control_gate':window_pass,
                 'state_ratio_at_32768':reduction,'decode_ratio_at_32768':ratio,
                 'selected_hybrid_sha256':sha256(run/'selected_hybrid.pt')},report/'decision.json')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__': main()
