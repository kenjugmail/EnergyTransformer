# V14: text-to-protected-memory and bounded learned answers

Executed September 12, 2026. Three separately seeded parser/answer pipelines; CPU only. No old language-model weights were retrained. No user GPU, paid compute, or remote training service was used.

## What changed

V13 required operation/key/value records supplied by the caller and exposed explicit key recall. V14 accepts a sentence, predicts its operation and finite-namespace slots, edits protected energy coefficients, and answers a question with a separately trained byte-output head. This closes a bounded text interface, not general instruction tuning. The old news byte model is archived unchanged; its free-form output logits are not conditioned on these protected facts by a new training run.

There are 16 names, 16 locations, four operation classes, and a maximum of 40 input words. Sentence boundaries and this relation grammar are supplied. Strong synthetic labels supervise operation/entity/value extraction. A lexical guard checks that the predicted fixed-namespace words explicitly and uniquely occur; this validation is supplied, not learned open-world entity resolution.

## Model

The parser uses 40-dimensional word embeddings; three 40-channel convolutional branches with kernels 3,5,7; masked mean/max pooling; a 96-unit SiLU layer; and operation/key/value classifiers. A separate decoder uses the protected mode vector, evidence mass, a 16-dimensional position embedding, and a 64-unit MLP to predict 12 byte positions plus EOS. It learns only the 16 location strings and `unknown`.

Total static parser+decoder parameters: **73,741**. Protected memory: P[16], S[16,16], **1,088 raw float32 bytes**. Sentence embeddings, logits, queries, and answer activations are temporary. The original 291,684-parameter language runtime is a separate inherited artifact, not included in that new parameter count.

For key i, E_i(W)=0.5 P_i ||W||^2-S_i^T W and the transient read is S_i/P_i. A hold leaves the state unchanged. Replacement sets P_i=1.25 and S_i to a one-hot value; deletion sets P_i=0.25 and S_i=0. Unaddressed rows are unchanged. This is an explicit categorical specialization, not unrestricted continuous reasoning.

## Training and data

Seeds: 11,22,33. Each parser trains for 900 AdamW updates at batch 64, with peak learning rate 0.002, cosine decay, weight decay 0.01, and gradient clipping 1. Minimum validation cross entropy at 150-step intervals chooses the checkpoint; all selected step 900. Each answer head trains for 500 updates at batch 64 with learning rate 0.005 on synthetic memory features. The modules are trained separately, not end-to-end through long episodes.

This means 57,600 presented sentences and 32,000 presented answer examples per pipeline, resampled from a smaller finite generator. They are not independent real-world documents. Three pipelines comprise six module fits. No external corpus was downloaded or redistributed.

Exact-text hashes allocate training/validation/parser-IID-test buckets. The held-out wording condition uses complete templates absent from training. Its unknown words map to the ordinary unknown token; it measures both phrasing and lexical shift, not syntax alone. Validation contains 768 sampled records. Final parser evaluation has 2,048 sampled records per condition, reused across three seeds; repeated texts can occur.

The episodic task has 192 independently generated episodes: 32 per gap (0,64,1024) and wording condition. Each starts with four assignments, changes one, deletes another, then inserts distractors and asks four questions. In-domain episodes can reuse known sentences; it is the sequence/configuration that is newly generated, not necessarily each sentence. This distinction follows a recorded pre-evaluation setup correction. The exact-text-disjoint parser test is separate.

Five methods and three seeds produce **2,880 method-episode records**, or **11,520 answer evaluations**. These are not 11,520 independent tasks.

## Results

| Task | Result |
|---|---:|
| Known-grammar exact held-out parse | 100% at all three seeds |
| New-wording exact parse | 68.85%, 47.95%, 56.59%; mean **57.80%** |
| Known-grammar energy recall, all three gaps | **100%** |
| Learned parser + conventional dictionary, known grammar | **100%** |
| Supplied-record oracle + energy, all wording/gaps | **100%** |
| Last-event-only control, known grammar | **25%** |

The no-history control's 25% reflects the one deleted/unknown key among four questions; it is not meaningful reasoning performance.

For the confidence/lexically guarded automatic pipeline on held-out wording:

| Distractors | Correct answers |
|---:|---:|
| 0 | **19.79%** |
| 64 | **6.51%** |
| 1024 | **1.04%** |

The unguarded argmax policy reaches 47.66%, 11.20%, and 1.56% respectively. Rejecting uncertain input can avoid some edits but also reject necessary updates/queries. It does not solve this distribution shift. The ordinary dictionary with the same learned parser exactly matches the guarded energy outcome. The failure is predominantly the text interface, not evidence that energy memory decays under exact holds.

At 1024 new-wording distractors, the guarded pipeline makes 34,196 false edits across the 96 seed-episode records. Repeated instances and shared templates mean those are not independent error trials. A confident parser can repeatedly misinterpret harmless statements and overwrite memory. No failed episodes were removed or repaired by training on final test templates.

## Safety and deployment behavior

After this failure, the public `handle()` default was changed to prohibit writes unless the caller explicitly supplies `trusted=True`. This is a post-evaluation safety amendment. Scores above still evaluate automatically processed trusted sentences, including all mistakes. Neither the threshold nor the weights were retuned to improve them.

Caller trust does not prove linguistic correctness. Do not mark retrieved documents, quoted content, or arbitrary user text trusted automatically. The model is a constrained research demo, not an unattended memory extraction service. New names, multiple entities, unknown locations, and long paragraphs require rejection or a different interface.

## Audits

**20 tests passed** with the representative trained checkpoint. They cover mode-based answers for all 17 outcomes, writes/deletes, 8192 exact holds, locality, nonmutation, bad snapshots, parameter gradients, exact-text split separation, vocabulary checks, learned text round trips, blocked untrusted/default writes, batch-padding parity, and a fresh-process restart.

The restart process received only model weights, P/S, and a new question; it did not receive the preceding sentence. It returned the same answer. This test concerns the bounded bridge, not a new audit of every inherited language model.

No old benchmark or checkpoint was silently counted as a new training result. All three new checkpoints, training logs, final test sentences, raw parser/episode results, setup failure, amendments, and code hashes are retained in the full payload.

## Publication scope

The connected API publishes source and small reports/results to `kenjugmail/EnergyTransformer`. Direct Git transfer from this container fails because github.com cannot resolve here. The 13 historical archives total approximately 272.95 MiB; they and binary checkpoints are assembled in the full download but not claimed as uploaded by the source commit. `scripts/publish_all.py --push` performs the remaining bulk upload from a machine with normal Git access and existing authorization. It never force-pushes or changes repository visibility.

A filename and credential-pattern scan over the 13 original archives found no flagged items. This is limited screening, not a perfect privacy or licensing audit. Raw external corpora and unrelated conversation content are intentionally excluded.

## Interpretation

The useful implemented step is a learned sentence-to-memory-to-answer path within a finite grammar. It does not outperform a conventional dictionary with the same parser and fails badly on new phrasing. The next genuinely unresolved integration challenge is robust, correctly authorized fact extraction and broad memory-conditioned language training, not additional retention length on already parsed records.

Prior art: Sukhbaatar et al., *End-To-End Memory Networks*, https://arxiv.org/abs/1503.08895; Wang, Shi, Fox, *Test-time regression*, https://arxiv.org/abs/2501.12352. No novelty or superiority claim is inferred from the architecture name.
