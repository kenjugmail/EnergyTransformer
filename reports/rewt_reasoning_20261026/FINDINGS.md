# Learned rereading improves bounded rule composition

The new opt-in **structured rule head** completes every one of **1,440 fresh
test prompts in all three seeds**, with energy refinement enabled or disabled.
The fixed-query control averages **27.40%** on the 960-prompt main test and
26.11% on the 480-prompt distractor test. This establishes a useful learned
read loop in the specified grammar. It does **not** establish a benefit from
the Fourier energy term, or an improvement to general language generation.

The [editable Google Slides presentation for October 28](https://docs.google.com/presentation/d/11p2zXD7gTftP4EZXtRqXL948wSRPqkao5pTgoINKgVQ)
explains the mechanism and these limits. A [PowerPoint copy](../../course_project/presentation_20261028/REWT_class_presentation.pptx),
[PDF backup](../../course_project/presentation_20261028/REWT_class_presentation.pdf)
and [ten-minute speaker script](../../course_project/presentation_20261028/speaker_notes.txt)
are also available.

## What changed

The [earlier architecture study](../rewt_architecture_20261022/FINDINGS.md)
found that additional optimization of a fixed prefix-conditioned energy rarely
corrected whole answers. The new [implementation](../../rewt/reasoning.py)
lets each predicted node change the query used for the next memory read:

1. Encode the supplied prompt bytes. Learned heads predict fact keys, values,
   a fact gate, the starting query and the requested number of reads.
2. Match a soft query distribution to the learned keys and retrieve a value.
3. Optionally refine a joint query/read/output state under a positive quadratic
   plus learned cosine energy.
4. Use the resulting output distribution as the next query. Repeat for the
   configured maximum of three reads.
5. Decode intermediate nodes and a learned read count, and serialize the result
   as `steps=...;ans=...`.

There is **no runtime rule parser, graph traversal, reference answer, symbolic
solver or task metadata input**. All operations in the model are learned neural
extraction, soft copying, attention and state refinement. The explicit soft
lookup and repeated-read structure is a strong inductive bias for this task.
Training supplies fact-location, source/target, start-query, count and trace
labels; the training script's regex and solver construct those labels only.

This interface is a **specialized structured head**, not autoregressive byte
generation. Syntax is serialized around predicted nodes and a predicted length.
It consumes the complete supplied prompt; its query locator may inspect the
next already-observed byte. It is not a streaming language-model position.
The earlier language models, checkpoints and sampler retain their behavior.
Percentages from this study must not be directly compared with their older
free-generation scores.

Each new read changes the conditional energy. Backtracking guarantees numerical
descent **within a read**, with coefficients fixed during its optimization;
there is no single-energy descent claim across the entire reasoning chain.

## Frozen comparison

The [declared plan](plan.json) uses three cold seeds, 800 updates per fit,
batch size 64, AdamW, the same per-seed initial tensors and exactly the same
rendered training batches. Recorded batch hashes agree across arms. Nine fits
were completed before [the test plan](test_plan.json) was written. Checkpoints
were chosen by validation complete-answer accuracy, with answer cross-entropy
breaking ties. All arms are reported; no model was retrained after these tests.

| Arm | Main test: complete trace + answer | Test seed range | Eight distractor edges | Training CPU / fit |
|---|---:|---:|---:|---:|
| Fixed query + energy | 27.40% | 25.10–31.98% | 26.11% | 38.31 s |
| Repeated reads + energy | 100.00% | 100–100% | 100.00% | 31.51 s |
| Repeated reads, zero energy rounds | 100.00% | 100–100% | 100.00% | 22.50 s |

All models allocate 68,502 parameters. The zero-energy arm has 60,950 active
parameters; its 7,552 energy-only parameters do not affect predictions or
receive gradients. It is an ablation with the same allocated tensors, not an
exact active-parameter match. Equal examples and updates are not equal compute.
The fixed-query model is a mechanism ablation, not a strong general reasoning
baseline or Transformer comparison.

Training process time includes batch construction and optimization, excluding
initialization, validation and checkpoint I/O. The nine main fits consumed
276.95 measured training CPU seconds plus 4.63 validation seconds. Workers used
one CPU thread each on a shared host; these are not GPU throughput or FLOP
measurements. Preliminary validation-only smoke fits and debugging are additional
work, excluded from those totals.

The main-test repeated-read advantage over fixed queries is 72.60 percentage
points, with a paired semantic-ID bootstrap interval of 71.67–73.58 points.
The interval keeps both wordings together, stratifies by depth and averages
the three fixed seeds; it does not estimate uncertainty over all training
initializations. The observed energy-versus-zero-energy difference is zero.
A degenerate zero bootstrap interval on a perfect finite test is not proof
of universal equality or perfect population accuracy.

## What the test covers, and what it misses

- Training: 6,000 semantic identities, one to three hops, five wording templates,
  two disconnected distractor edges, node symbols A–X.
- Validation: 180 identities in familiar wording and two different withheld
  phrasings, totaling 360 prompts.
- Main test: 480 new identities, each in familiar wording and one of two further
  withheld phrasings, totaling 960 prompts.
- Distractor test: another 240 identities with eight distractor edges, each in
  two wordings, totaling 480 prompts. Eight distractor edges were not trained.
- The test excludes all 2,954 semantic test identities found in prior
  `tasks_test*.jsonl` artifacts. Training, validation and both new test sets are
  mutually disjoint. Wording and shuffled fact order do not change identity.

Every repeated-read seed completes all one-, two- and three-hop test groups in
both familiar and withheld test wording. But **wording robustness is not solved**:
the energy models average 100% on familiar validation wording and only **43.15%**
on its two withheld phrasings, for 71.57% overall. Those phrasings differ from
the test phrasings, which are easier for the learned query locator. Report this
alongside the perfect test result; “unseen wording” is not a uniform difficulty.

There is no new evidence here for arithmetic, natural-language reasoning,
unbounded depth, branching programs, contradictions, multi-character node names,
or coherent prose. The structured serialization also avoids the syntax-learning
problem faced by the earlier autoregressive models. These remain explicit limits.

## Which mechanism causes the gain?

Two interventions were declared before opening the test and applied to the
already-trained repeated-read energy checkpoints, without weight changes:

| Frozen inference policy | Main complete accuracy, each of three seeds |
|---|---:|
| Original repeated reads + energy | 100% |
| Disable energy refinement | 100% |
| Keep the initial query fixed for all reads | 33.33% |

Freezing the query preserves one-hop accuracy and breaks the longer traces.
Together with the independently trained zero-energy controls, this attributes
the demonstrated gain to learned extraction and query updates. It does not
justify crediting the waves. For this narrow task, the simpler zero-energy
read loop is sufficient and uses less measured training CPU time.

## Runnable example and audit

The demonstration checkpoint was chosen **on validation only**: seed 26103.
It is saved at
[`runs/rewt_reasoning_20261026/reread_energy_26103/best.pt`](../../runs/rewt_reasoning_20261026/reread_energy_26103/best.pt)
with SHA-256
`7fc41b47cc92b7c9465475ce0b904e1647e76236cbc028b514187e67e6fa37a6`.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/sample_rewt_reasoner.py \
  --checkpoint runs/rewt_reasoning_20261026/reread_energy_26103/best.pt \
  --prompt $'B>C;A>B;C>D;X>Y;from A go 3=?\n'
```

The [actual CLI output](cli_demo.json) is `steps=B,C,D;ans=D`. Add
`--energy-rounds 0` for the frozen zero-energy intervention, or `--fixed-query`
for the query ablation. This example illustrates use and is not included as an
extra benchmark success. Inputs outside the studied grammar have no guarantee.

**207 REWT tests passed in 9.53 seconds.** New checks cover analytic gradients,
per-read descent, finite gradients through every active parameter, padding and
prefix boundaries, actual query-update effects, and unchanged inference weights.
The [verification record](verification.json) contains source and test hashes.
All 18,720 saved outputs across main cases and frozen interventions were checked
again against their prompts and scoring rules. Inference weights and checkpoint
file hashes remain unchanged. Maximum audited within-read energy change for
the repeated-read energy models is negative; this is a numerical check only.

See [all fit and test results](test_results.json), [summary and uncertainty](summary.json),
[training results](training_results.json), and the
[reproducible study script](../../scripts/study_rewt_reasoning.py).

To reproduce, choose fresh output directories rather than overwriting this
closed study:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/study_rewt_reasoning.py \
  --output reports/rewt_reasoning_replication --runs runs/rewt_reasoning_replication \
  --prepare --train --workers 3
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/study_rewt_reasoning.py \
  --output reports/rewt_reasoning_replication --test --workers 3
.venv/bin/python scripts/summarize_rewt_reasoning.py \
  --study reports/rewt_reasoning_replication
```

The test generator excludes all previously recorded tests, so a new replication
will use different test identities. Existing partial runs are preserved rather
than silently restarted. Source and data hashes must match the saved plan.
