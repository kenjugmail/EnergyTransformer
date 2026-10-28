"""Render the completed investigation from its measured artifacts."""
from collections import defaultdict
import json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/rewt_architecture_20261022'


def read(name):return json.loads((OUT/name).read_text())


def main():
    s=read('summary.json');bench=read('benchmark_final.json');geometry=read('trained_geometry.json')['rows']
    inference=next(r['arms'] for r in bench['cases'] if r['batch']==24 and r['length']==128 and not r['backward'])
    training=next(r['arms'] for r in bench['cases'] if r['batch']==24 and r['length']==128 and r['backward'])
    execution='| Execution path | Forward CPU ms | Forward + backward CPU ms | Forward speed vs reference |\n|---|---:|---:|---:|\n'
    for n in ('reference','krylov2','direct_packed','certified_same_step','precond4','precond4_structured','scan','feedback8','delta','patch4'):
        ms=inference[n]['median_cpu_ms'];execution+=f"| {n} | {ms:.2f} | {training[n]['median_cpu_ms']:.2f} | {inference['reference']['median_cpu_ms']/ms:.2f}× |\n"
    prep=bench['preparation'];prep_ratios=[prep[f'dense/backward={b}']['median_cpu_ms']/prep[f'structured/backward={b}']['median_cpu_ms'] for b in (False,True)]
    decodetable='| Batch | Prefix | Dynamic cache CPU ms | Ring cache CPU ms |\n|---:|---:|---:|---:|\n'
    for r in bench['decode']:decodetable+=f"| {r['batch']} | {r['prefix']} | {r['arms']['krylov2/dynamic']['median_cpu_ms']:.2f} | {r['arms']['krylov2/ring']['median_cpu_ms']:.2f} |\n"
    taskrows=read('test_results.json');groups=defaultdict(list)
    for r in taskrows:groups[r['case']['variant']].append(r)
    tasktable='| Variant | Test complete | Arithmetic | Rules | Recall | Δ vs control, paired 95% interval |\n|---|---:|---:|---:|---:|---:|\n'
    for n,g in groups.items():
        mean=s['task'][n]['test_mean'];families=[statistics.mean(r['test']['summary']['by_family'][f]['both_correct'] for r in g) for f in ('arithmetic','rules','recall')]
        ci=s['paired_task_deltas'][n]
        tasktable+=f"| {n} | {100*mean:.2f}% | "+' | '.join(f'{100*v:.2f}%' for v in families)+f" | {100*ci['mean_delta']:+.2f} pp [{100*ci['ci95'][0]:+.2f}, {100*ci['ci95'][1]:+.2f}] |\n"
    contribution=read('refinement_contribution.json')['rows'];ablations=defaultdict(list)
    for r in contribution:ablations[r['case'].rsplit('_',1)[0]].append(r)
    ablationtable='| Trained variant | Normal validation accuracy | Zero-round accuracy | Changed generations |\n|---|---:|---:|---:|\n'
    for n,g in ablations.items():
        ablationtable+=f"| {n} | {100*statistics.mean(r['normal_complete'] for r in g):.2f}% | {100*statistics.mean(r['no_refinement_complete'] for r in g):.2f}% | {sum(r['changed_generations'] for r in g)}/486 |\n"
    gt=defaultdict(list)
    for r in geometry:gt[r['case']['variant']].append(r)
    geometrytable='| Variant | Frequency stable rank | Correction / anchor logit RMS | Prompt argmax changed |\n|---|---:|---:|---:|\n'
    for n in ('control','precond4','precond8','scan','delta','diversity','hybrid_modes','combined'):
        g=gt[n];geometrytable+=f"| {n} | {statistics.mean(r['frequency_stable_rank'] for r in g):.2f} | {statistics.mean(r['correction_logit_rms']/r['anchor_logit_rms'] for r in g):.5f} | {100*statistics.mean(r['prompt_argmax_changed_fraction'] for r in g):.3f}% |\n"
    skipped=statistics.mean(r['skipped_positions_above_anchor_fraction'] for r in gt['patch4'])
    doc=read('language/document_audit.json')['paired_deltas']['delta'];memory=s['memory_ablation'];mg=s['memory_ablation_groups']
    language=s['language'];selected=s['selected_task_variant_by_validation'];selectedlm=s['selected_language_variant_by_validation']
    compute=s['equal_compute']
    computetable='| Solver | Mean updates | Mean training CPU seconds | Validation bpb | Test bpb | Test seed range |\n|---|---:|---:|---:|---:|---:|\n'
    for n,r in compute.items():computetable+=f"| {n} | {r['mean_updates']:.1f} | {r['training_cpu_mean']:.3f} | {r['validation_mean']:.4f} | {r['test_mean']:.4f} | {r['test_min']:.4f}–{r['test_max']:.4f} |\n"
    text=f'''# REWT: implemented kernels and measured architecture improvements

The architecture now has dedicated execution paths and controlled learning
results, rather than only a list of proposed optimizations. This follow-up
completed **96 new fits**: 39 reasoning fits, 24 language fits at equal updates,
15 language fits at equal measured CPU budgets, and 18 memory fits. The final
memory ablation also reuses six frozen checkpoints. All declared
seeds and variants are included, including unsuccessful approaches.

The strongest demonstrated learning improvement is persistent delta memory
with explicit training of binding and write behavior. On the matched fresh
memory test it scores **{100*memory['delta_binding']['test_mean']:.2f}%**, versus
**{100*memory['local_binding']['test_mean']:.2f}%** for the supervised local control.
The language improvement is modest, and the reasoning results do not establish
general multistep reasoning. Kernel speed, energy descent, learned memory and
answer accuracy are evaluated separately below.

## What was implemented

| Layer | Delivered change | Contract |
|---|---|---|
| Conditioning | Packed affine projections and precomputed GRU input maps | Same weights and mathematical function; rounding may differ |
| General refinement | Quadratic directional evaluation of all line-search trials | Full energy, arbitrary rounds, full supporting model, masking and coarse proposals |
| Fixed-step refinement | Curvature-bound steps; two-step reduced lift | Same trajectory only when the accepted step matches; larger steps change policy |
| Preconditioning | Arbitrary rounds in M mode coefficients | Different optimization geometry; trained and evaluated as a separate policy |
| Preparation | Analytic output elimination and latent Schur factorization | Same preconditioner and gradients with a smaller factorization |
| Observation path | Parallel affine scan and optional chunk feedback | Different recurrent architecture |
| Memory | Chunked delta writes and persistent matrix state | Learned retrieval beyond local attention; additional parameters |
| Update scheduling | Fixed causal patches and reused corrections | Fewer refinements; skipped positions have no current-energy guarantee |
| Mode/head experiments | Diversity penalty, shared/block modes, factorized output heads | Independently trained variants; no assumed quality preservation |
| Decode | Fixed-size ring, owned outputs, cache invalidation and whole-step CUDA graph | Fixed-weight GRU core only |
| GPU kernel | Full-row and output-tiled Triton mode-space refinement | Float32 inference only; PyTorch remains the training implementation |

The [design and derivations](DESIGN.md) explain each implementation. The code is
in `rewt/advanced_kernels.py`, `architecture.py`, `structured_solver.py`,
`ring_decode.py` and `triton_refine.py`. New paths are opt-in; the eight audited
archived checkpoints retain bitwise-identical default outputs.

## Why these are architecture-specific kernel improvements

For the anchor-relative core, displacement z has conditional energy
`E(z) = 1/2 z H zᵀ + Σ a_i cos((z Fᵀ)_i + phase_i)`.
At a stationary point, `H z = Fᵀ[a sin(F z + phase)]`. Therefore its displacement
lies in the M-dimensional span of `F H⁻¹`, even though the joint state has D
coordinates. Preconditioning exposes this directly:

```
B = F H⁻¹; G = F H⁻¹ Fᵀ
c_next = (1 - eta) c + eta a sin(c G + phase)
state = anchor + c B
```

The implementation performs every round on M coefficients and lifts once.
Ordinary finite-step gradient descent can contain transient directions outside
this span, so switching a checkpoint's solver is not an exact drop-in change.
The original two-step Krylov path, packed projections and structured preparation
provide the exact execution alternatives.

The other major cost is serial conditioning. A faster refinement kernel cannot
remove the dependency created when the previous refined recurrent state feeds
the next token. The scan variant changes that dependency so whole chunks can
use batched attention, conditioning and refinement. Chunk feedback restores an
explicit dependency at eight-byte boundaries. Delta memory adds persistent
content state without requiring a growing attention cache.

This also exposes a modeling limit: refinement currently optimizes a fixed
prefix-conditioned landscape. Additional rounds do not perform another read
of the input or invent a new reasoning operation. Their value must be learned
and measured, rather than inferred from decreasing energy.

## CPU execution

The final benchmark uses one CPU thread pinned to core 5, seven alternating
repetitions, and batch 24 × 128 positions. Values below are median process time.
Full wall-time samples, smaller batches, decode and preparation are retained in
[benchmark_final.json](benchmark_final.json). This run follows the first 81 fits
and precedes the separate compute-budget fits, with no concurrent study training
workers. The machine remains shared with unrelated work.

{execution}

Most of the exact forward gain above comes from the previously implemented
Krylov reduction. Additional direct packing changes the large-batch forward
from {inference['krylov2']['median_cpu_ms']:.2f} to
{inference['direct_packed']['median_cpu_ms']:.2f} ms; it is not another 2× gain.

These are execution measurements, not equal-quality speedups. Scan, feedback,
delta and patches change the model. The reference already caches causal history;
the table does not compare against an artificially uncached baseline. Forward /
backward uses a logit-square objective, excluding optimizer and task auxiliary
losses. Preparation is included in sequence calls; decode measurements include
prefill and session setup.

The structured preconditioner factors **144 instead of 400 coordinates** at d=48.
Isolated preparation is **{prep_ratios[0]:.2f}×** faster forward and
**{prep_ratios[1]:.2f}×** faster including backward in this measurement. Its
forward values and every parameter gradient match the dense solve within the
recorded numerical tolerances, including nonuniform output precision.

This smaller factorization does not ensure a faster full backward pass: the
four-round structured model takes {training['precond4_structured']['median_cpu_ms']:.2f} ms
versus {training['precond4']['median_cpu_ms']:.2f} ms for its dense version in the
full sequence benchmark. Preparation and end-to-end performance are distinct.

The general directional solver is workload-dependent. In the separate
[directional benchmark](directional_benchmark.json), core inference is 1.44×
faster when all backtracking trials are evaluated, but 0.83× as fast when the
CPU reference accepts its first small trial immediately. Both paths remain
available. Packing also does not improve every training shape; consult the
complete table rather than treating an inference win as a backward guarantee.

For prefill plus 32 decoded bytes, the CPU ring cache is approximately tied
with the dynamic cache. Its main purpose is fixed storage for graph capture;
these CPU measurements do not establish a decode acceleration:

{decodetable}

![CPU execution](architecture_execution.png)

## GPU correctness and resource use

On NVIDIA GB10 with PyTorch 2.14 / CUDA 13, the full-row kernel passed 24
standalone checks and the ring/graph path passed ten decode comparisons,
including prefixes longer than the ring and 135 following bytes. Maximum
whole-step error was 3.44e-5 and audited greedy choices matched with TF32 disabled.
The tiled path passed **52 further GPU checks**, including whole-step graphs;
maximum error was below 3.6e-7 in that separate small-model matrix. The CPU
Triton interpreter also passed 64 formula/index checks.

At M=32, D=400 and four warps, the full-row kernel uses 158 compiled registers
per thread; the 64-coordinate tile uses 40. Tiling repeats the small recurrence
per output tile, so fewer registers do not automatically mean lower latency.
Compiler spill counts and shared memory are in [triton_resources.json](triton_resources.json)
and [triton_tiles.json](triton_tiles.json). These are not occupancy measurements.

The GPU was about 90% occupied by unrelated work and context allocation
occasionally failed under shared memory pressure. **No reliable GPU speedup is
claimed.** The delivered paths have numerical and graph-lifecycle validation;
uncontended latency tuning remains unmeasured. No custom backward, BF16/FP16,
dynamic-shape graph batching or other-GPU portability result is claimed.
See [PyTorch graph constraints](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-graphs)
and [Triton resource guidance](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html).

## Learned reasoning: all variants

Every arm has three cold starts, 1,024 updates, identical per-seed task batches
and supervision, 5,400 training identities and validation-selected checkpoints.
The test contains 162 semantic identities, each with familiar and withheld
wording: 324 prompts per model. These test identities exclude the 2,792 earlier
recorded test identities. The fresh training draw reuses 502 previous training
IDs and validation reuses three prior validation IDs; all new models start cold
and this study's own partitions are disjoint. See the [data audit](DESIGN.md).

The variant chosen by mean validation accuracy is **{selected}**. Test accuracy
requires both intermediate steps and the final answer to be correct.
It scores **{100*s['task'][selected]['test_mean']:.2f}%**, versus
**{100*s['task']['control']['test_mean']:.2f}%** for the control; its paired
interval includes zero improvement.

{tasktable}

Intervals use paired semantic-ID bootstrap samples, stratified by family/depth,
keeping both wordings together and averaging the three fixed trained seeds.
They do not estimate uncertainty over an unlimited population of training
seeds; exploratory multiple comparisons are not adjusted. Seed ranges,
parameters, training process times and all validation scores appear in
[RESULTS.md](RESULTS.md). Equal update counts are not equal compute or exact
parameter matching.

The family breakdown is essential: a one-third overall score can come mostly
from recall while arithmetic and multistep rules remain weak. These results do
not justify advertising more refinement rounds as stronger general reasoning.

![Quality across architectures](architecture_quality.png)

## Does refinement improve the learned answer?

A separate post-training intervention removes all refinement rounds from nine
frozen checkpoints and evaluates the existing validation set. It does not
retrain, select a new policy, or inspect another test set.

{ablationtable}

Refinement lowers response cross-entropy in these comparisons, but it does not
increase complete-answer accuracy. For the GRU models, removing refinement also
changes the later recurrent carry, so this is a full-policy intervention. The
delta variant changes only seven of 486 generations, supporting the narrower
finding that its learned observation/memory path dominates these outputs.

The validation-only geometry audit measures the frequency span and the output
correction relative to the anchor. Stable rank is sum(eigenvalues)/max(eigenvalue),
not the numerical matrix rank. Statistics below are three-seed means on the
same 18 validation prompts.

{geometrytable}

The diversity penalty raises stable rank from about 5.6 to 9.1, but its
complete-answer interval still overlaps the control. The correction is only a
small fraction of anchor-logit RMS in these trained models. Improving numerical
mode diversity alone does not make the residual responsible for correct answers.

The maximum audited round-energy increase across all 39 selected models is
**{max(r['largest_round_energy_increase'] for r in geometry):.3g}**. This tests
numerical descent, not answer correctness. At skipped patch positions,
**{100*skipped:.2f}%** of audited states have energy above the current anchor;
reusing a previous correction does not promise descent for a new token's energy.
See [trained_geometry.json](trained_geometry.json) and
[refinement_contribution.json](refinement_contribution.json).

## Language: a modest measured gain

The separate language screen trains eight variants across three seeds for
512 updates, or 1,572,864 BPE tokens per fit. Evaluation uses 256 distinct new
documents and divides negative log-likelihood by actual decoded target bytes.
The tokenizer is learned on the training partition only. Holdout source,
normalization, document exclusion and hashes are preserved.

The validation-selected variant is **{selectedlm}**. Delta memory averages
**{language['delta']['test_mean']:.4f} bits/byte**, versus
**{language['control']['test_mean']:.4f}** for the control. Its paired difference
is **{doc['delta_bpb']:.4f}**, with a document-bootstrap 95% interval
**[{doc['ci95'][0]:.4f}, {doc['ci95'][1]:.4f}]**, conditional on these three trained
seeds. All three seed differences favor delta memory. It has 257,537 parameters
versus the control's 256,096, so this is not an exact parameter-matched result.

Plain scan, chunk feedback and fixed patches are slightly worse on language.
Rank-16 output heads reduce parameters to 208,800 but worsen the score to
**{language['lowrank16']['test_mean']:.4f} bits/byte**. That capacity trade is not
supported at this width and budget. Generated samples remain repetitive and
grammatically weak. This screen is much shorter than the earlier language
study and does not establish fluent generation or superiority to Transformers.

All 144 language continuations are retained. `inference.pt` copies include the
verified tokenizer metadata; original `best.pt` hashes and weights are unchanged.
See [document-level uncertainty](language/document_audit.json) and the full
language table in [RESULTS.md](RESULTS.md).

## Equal measured training compute

A separate declared screen compares the existing two-round Krylov control with
two-, four- and eight-round preconditioning, plus four-round structured
preparation. Each arm uses the same 256,096 parameters and three new seeds.
Each fit receives **150 measured training process seconds**, excluding
initialization, validation and checkpoint I/O, with at most one update overshoot.
Learning rate and four validation points follow consumed CPU budget. Faster
paths can therefore perform more updates, instead of receiving an identical
update count. All workers use one CPU thread on pinned performance cores.

{computetable}

At this short budget, structured four-round preparation has the lowest mean
validation loss and scores {compute['precond4_structured']['test_mean']:.4f} test
bits/byte versus {compute['control']['test_mean']:.4f} for the control. It completes
{compute['precond4_structured']['mean_updates']:.1f} updates on average, versus
{compute['control']['mean_updates']:.1f}. Eight rounds complete only
{compute['precond8']['mean_updates']:.1f} updates and score
{compute['precond8']['test_mean']:.4f} bits/byte. This supports treating refinement
depth as a compute allocation choice, not an automatic accuracy improvement.

The compute test uses 256 additional documents, excluding the entire earlier
256-document language test set. The study was declared after the first screen;
its new holdout was not used for model selection. This is a short CPU-budget
screen rather than a GPU compute comparison or long-run scaling result.
Process CPU seconds include batch preparation and training; they are not a
FLOP count and can still reflect shared memory bandwidth and clock conditions.
Fits run concurrently in declared order, with four workers and three final
structured fits; this is not a randomized hardware-isolation experiment. Use
the alternating execution benchmark to assess preparation cost separately.
See [the declared compute plan](equal_compute/plan.json), cumulative CPU traces
and [complete results](equal_compute/test_results.json).

## Persistent memory: the successful architectural change

The memory task uses eight keys and ten digit values. The requested key is
written twice with different values; other records are distractors. The model
must return the latest value from bytes alone. Training-only binding, copy and
write-gate labels do not enter inference. The longest trained far gap is 71
bytes; the test includes gaps of 95 and 191 bytes with a 16-byte local window.

The first response-loss-only recipe failed: delta memory averaged 34.46%, below
the local control's 54.17%. The longer binding-supervised follow-up reached
96.01%, but it changed both supervision and update budget. To isolate supervision,
the final ablation fixes seeds, batches, 1,536 updates, architecture and optimizer,
and evaluates all four arms on a third fresh 384-prompt test set:

| Memory | Training supervision | Mean accuracy | Accuracy at 191-byte gap |
|---|---|---:|---:|
| Local | Response only | {100*memory['local_response']['test_mean']:.2f}% | {100*mg['local_response']['64/False']:.2f}% |
| Local | Binding + copy | {100*memory['local_binding']['test_mean']:.2f}% | {100*mg['local_binding']['64/False']:.2f}% |
| Delta | Response only | {100*memory['delta_response']['test_mean']:.2f}% | {100*mg['delta_response']['64/False']:.2f}% |
| Delta | Binding + copy + write gate | {100*memory['delta_binding']['test_mean']:.2f}% | {100*mg['delta_binding']['64/False']:.2f}% |

![Matched memory ablation](memory_matched.png)

This is evidence for learned persistent overwrite memory under a narrow grammar,
not open-ended recall or reasoning. Keys, values and syntax stay within the
training vocabulary; only bindings, distractors and lengths change. The delta
arm has 34,321 parameters versus 32,752 in the local arm. The experiment identifies
the combined training recipe, not the individual causal contribution of each
auxiliary loss. It also demonstrates that adding a theoretically capable memory
state without teaching its writes can make performance worse.

The recurrence follows the delta-rule family described in
[DeltaNet](https://arxiv.org/abs/2406.06484). The local implementation uses a
differentiable triangular chunk solve, not an imported optimized DeltaNet kernel.
The observation scan follows the affine-recurrence idea in
[Were RNNs All We Needed?](https://arxiv.org/abs/2410.01201); published results from
those papers are not attributed to this small model.

## Recommended defaults and remaining research

For existing checkpoints, retain their solver and use the validated Krylov,
packing and fixed-cache execution options where measured beneficial. Use the
structured factorization when choosing the separately trained preconditioned
policy. Keep the CPU early-exit reference available. GPU graphs and tiled Triton
are opt-in until uncontended end-to-end timing establishes their useful shapes.

For a new architecture, persistent memory with explicit write/read training has
the strongest evidence here. The scan exposes useful batching, but its language
tradeoff should remain visible. Keep full-rank output heads at this size. Treat
fixed patches as an experimental latency/quality trade; they do not retain a
per-token energy guarantee. Do not promote extra rounds or diversity penalties
as reasoning improvements without complete-answer gains.

The next modeling target is the contribution of refinement itself: richer
context-dependent residual geometry or another causal memory read per round,
paired with a training objective that rewards actual answer correction. Another
read changes the frozen conditional objective and therefore needs a new descent
contract; it is not a free extension of the current kernel. A learned patch
boundary likewise needs causal boundary supervision and separate quality tests.
These are research directions, not implemented or validated claims in this run.

## Verification and reproducibility

**198 REWT tests passed.** Exact-path checks cover values, every parameter
gradient, causality, awkward chunk boundaries, ring wraparound, graph-state
restoration, output ownership, stale-cache rejection, structured solves and
delta overwrites. Eight archived defaults remain bitwise equal; 486 prompt/path
pairs have identical generated bytes on the audited exact execution paths.
All 13 training variants replay outputs, gradients and updated weights bitwise
against the frozen source for three optimizer updates.

The delivered source archive, data/checkpoint checksums, all plans, optimizer
traces, predictions, failures and measurements are preserved. See
[reproduction instructions](REPRODUCING.md), [complete tables](RESULTS.md),
[machine-readable summary](summary.json), [artifact manifest](manifest.json),
and [source archive](final_source.tar.gz). No archived checkpoint was overwritten
or new architecture silently substituted into a legacy model.
'''
    (OUT/'FINDINGS.md').write_text(text)
    print(json.dumps(dict(report=str(OUT/'FINDINGS.md'),words=len(text.split()))))

if __name__=='__main__':main()
