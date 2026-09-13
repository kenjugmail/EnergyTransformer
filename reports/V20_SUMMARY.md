# V20: a solved specialist memory task is not yet pretraining readiness

September 12, 2026. Twelve new CPU fits (four variants x three seeds) plus one validation pilot. No general-language model, GPU run, paid service, or production memory interface was changed.

## Result

A causal local encoder learns when to write, which address to edit, and which categorical value to store. Auxiliary semantic targets are supplied during training only. At inference the model receives token IDs and boundary resets, not labels or a gold key/value table. Final evaluation inputs end at the query key; the answer token is not present.

Mean exact answer accuracy across three seeds:

| Model | Six keys | 64 distractors | 24 keys | Twelve edits | All32 known keys | 2,048 distractors | Changed layout |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hard energy + write supervision | 100% | 100% | 100% | 100% | 100% | 100% | 10.29% |
| Soft energy + write supervision | 100% | 100% | 100% | 100% | 100% | 100% | 9.38% |
| Identical hard architecture, no auxiliary labels | 10.03% | 4.95% | 2.60% | 20.44% | 6.64% | 7.81% | 9.24% |
| Improved local-encoder softmax control | 98.83% | 51.82% | 81.38% | 78.65% | 56.64% | 7.29% | 16.67% |

There are1,600 independent final episodes:256 for each of six conditions and64 for the longest stress. All12 models reuse them, producing19,200 primary records. History interventions add1,536 records, for20,736 total. These are not20,736 independent tasks. At100% empirical success the recorded bootstrap interval collapses; it is not a guarantee of zero population error.

## Why the successful score is scope-limited

The model has32 supplied address classes,17 categorical value channels and a four-token local receptive field. This is task specialization, not the unchanged general v19 backbone passing its gate. Hard and soft routing both succeed; the experiment does not establish a unique benefit from hard routing. An exact parser/dictionary also solves every target. A post-hoc register implementation with the same trained router and decoder matches energy logits bit for bit over96 model-episode checks.

The changed-layout condition inserts four untrained FILL tokens before values. It jointly tests lexical novelty and longer local distance, not either in isolation. It fails badly; no training repair was performed on those final examples. Stronger attention reaches98.83% on the main task, so the earlier weak Transformer score must not be treated as proof that attention cannot solve it. The attention control is not trained to convergence, GPU-optimized, or resource-matched to a general energy LM.

## Actual computation

The long-term state is P[32], S[32,17] with prior lambda=.25. A three-embedding local shift register [3,64] is also retained and explicitly counted. It is a local activation buffer, even though it can be described as linear coefficients of a unit-precision energy.

For predicted address a, SET weight w and combined SET/DELETE erase gate e:

    P_new[i] = lambda + (1-e*a[i])*(P[i]-lambda) + w*a[i]
    S_new[i] = (1-e*a[i])*S[i] + w*a[i]*value

The read computes S/P. P remains at least lambda. Hard gates use biased straight-through surrogate gradients; soft gates use softmax. The hard model has79,157 parameters and3,072 raw state bytes including the local buffer. The attention control has112,693 parameters and a growing KV cache. Equal examples do not imply equal information, inductive bias or compute.

## Protocol and audits

Each fit uses1,000 updates of16 episodes, a common staged curriculum and identical per-seed input/target hashes. Validation chooses accuracy then weighted LM loss. Model selections were recorded before new final tests were generated. The hard model was prespecified, not the overall validation winner: soft routing has slightly lower mean selected validation loss.

Three trained-state restarts reproduce logits and tensors exactly in a fresh Python interpreter given only weights, saved state and new tokens. Erasing protected history reduces accuracy to10.16%; swapping read histories reduces it to8.98%. The full local archive passes160 tests, including16 new tests. The GitHub tree does not contain all unpublished v19 source/tests, so its total inherited test count differs. New tests require trained weights for one trained-checkpoint case; it skips when those weights are absent.

The new trainer retains final optimizer snapshots but has no tested mid-run resume CLI. Do not attribute v19's separate resumption tests to this trainer.

## Decision

**NO_GO_FOR_SERIOUS_PRETRAINING.** The95% primary and85% delayed specialist thresholds pass without being lowered. General language, flexible role/span binding, target-GPU acceptance and resource-matched modern baselines do not. Bounded language-integration research is a reasonable next stage; no substantial compute allocation is justified by this synthetic result alone.

Code, training and evaluation are source-published. Original checkpoint binaries, full v19+v20 worktree, raw predictions, complete audits and report are retained in the conversation experiment download; no binary upload is inferred from this source commit.

## Primary references

- Test-time regression: https://arxiv.org/abs/2501.12352
- End-To-End Memory Networks: https://arxiv.org/abs/1503.08895
- Mamba: https://arxiv.org/abs/2312.00752
- Gated Delta Networks: https://arxiv.org/abs/2412.06464
- Prerequisite toy tasks: https://arxiv.org/abs/1502.05698
- RULER: https://arxiv.org/abs/2404.06654
- MIRAS: https://arxiv.org/abs/2504.13173

No new theorem, fractal mechanism, independent replication or state-of-the-art language result is claimed.
