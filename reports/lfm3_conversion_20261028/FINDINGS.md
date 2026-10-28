# LFM3-1.7B → streaming REWT: conversion and recovery

A standalone **1,784,445,792-parameter** streaming REWT model now runs from LFM3-derived weights. Its 13 original gated-convolution slots are replaced by gated delta memory, all 11 attention layers use a 256-token window, and two final latent-energy refinement steps precede the inherited vocabulary projection. The model has no donor forward path at inference.

**The architecture conversion is implemented; full capability recovery is not established.** This is one short conversion trajectory without independent training replications. Continuation loss improves sharply, while generated answers must be inspected separately. It is not a validated replacement for the original LFM3 assistant.

[Architecture, transfer map and equations](DESIGN.md) · [Complete raw comparison and samples](comparison.json) · [Environment](environment.json)

## Held-out continuation comparison

All rows below use the same 8,192 next-token positions: 32 fixed, document-contained 256-token crops sampled from the prepared official test split. NLL and KL are natural-log units per token; lower is better. Agreement is with the donor’s top next-token choice under the reference conversation prefix, **not answer accuracy**.

| Model | NLL ↓ | KL to donor ↓ | Top-token agreement ↑ |
|---|---:|---:|---:|
| Donor LFM3-1.7B | 2.3117 | 0.0000 | 100.00% |
| Converted, before recovery | 8.4520 | 6.7265 | 4.97% |
| After the 256-token pilot | 2.4939 | 0.9530 | 56.42% |
| Final recovered student | 2.3468 | 0.8280 | 59.22% |

The final student NLL is 1.52% above the donor on these crops. This is a loss comparison, not a percentage of capabilities retained. The pilot test and four sample prompts were inspected before deciding to continue recovery, so the table is an exploratory comparison, not a sealed final benchmark. None of the runs trains on validation or test conversations.

## What was trained

- Donor: `LiquidAI/LFM3-1.7B`, immutable revision `40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd`.
- The first run used 128 layer-alignment updates and 256 full-model distillation updates, two 256-token sequences per update. It processed 196,608 token exposures.
- The second run continued the selected pilot checkpoint for 512 distillation updates at length 512, two sequences per update, with a fresh optimizer and learning rate 5e-5. It processed 524,288 further token exposures.
- A final 128-update pass uses 512-token conversation prefixes (131,072 token exposures), learning rate 2e-5 and seed 281707. Its validation/test crops also begin at conversation boundaries. The final artifact selects prefix-pass update 128 by the lowest KL on 4,096 fixed validation positions. Earlier training seeds are 281702 and 281706; initialization seed 281701 is retained separately in checkpoint provenance.
- Embeddings, feedforward layers and FFN normalization stayed frozen. The mixers, operator norms and energy head supplied 389,739,360 trainable parameters. This is a backbone conversion with substantial weight training, not a LoRA wrapper around the donor.
- Objective: exact full-vocabulary teacher KL + 0.1 hard next-token NLL + 0.1 normalized hidden-state MSE. Validation KL selected each checkpoint; there was one seed per run.
- Measured loop/evaluation/checkpoint time: 395.1s, 1096.2s and 277.2s for the three stages, excluding initial model loading. Peak CUDA allocations were 13.15, 13.15 and 13.15 GiB on the shared GB10. These are not isolated performance benchmarks.

[Pilot plan](pilot_plan.json) · [Pilot training log](pilot_metrics.jsonl) · [Continuation plan](recovery_plan.json) · [Continuation training log](recovery_metrics.jsonl) · [Prefix-pass plan](prefix_plan.json) · [Prefix-pass training log](prefix_metrics.jsonl) · [Data manifest](corpus_manifest.json)

A crop audit found that only 12 of the first two stages’ 1,792 training crops began at conversation start. The final pass addresses that distribution gap by retaining the beginning of the conversation; it uses ordinary training conversations, never the four comparison prompts as targeted training examples. This is a motivated recipe adjustment, not proof that crop placement caused the earlier generation failures. [Exact crop coverage](crop_coverage.json).

The corpus has 8,192 selected training conversations (7,635,132 stored tokens), 128 validation conversations and 128 test conversations. Exact conversation hashes are disjoint. Crops never cross document boundaries. The longer run only samples documents with at least 513 tokens. Metrics include template and user tokens; no broad knowledge, code, tool-use, long-context-quality or instruction-following benchmark was completed.

## Unedited final responses

The final checkpoint says `hello` exactly and now extracts Maya/Boston into valid JSON. It computes 42 but ignores the arithmetic prompt’s “just the number” format requirement. Its leaf-color response mentions chlorophyll but contains inaccurate and confused wording. These remaining failures prevent a claim of complete capability recovery.

These four prompts were fixed before the first recovery comparison. Greedy generation uses the original non-thinking chat template and a 48-token output limit. Responses that exhaust that limit are truncated. All donor, initialized and recovered outputs and token IDs are retained in the raw comparison.

**Prompt:** Reply with exactly the word hello.

~~~~json
"hello"
~~~~

**Prompt:** Extract the name and city as JSON with keys name and city. Text: Maya lives in Boston. Return only JSON.

~~~~json
"{\n    \"name\": \"Maya\",\n    \"city\": \"Boston\"\n}"
~~~~

**Prompt:** What is 17 plus 25? Answer with just the number.

~~~~json
"17 + 25 = 42."
~~~~

**Prompt:** Explain in one sentence why leaves are green.

~~~~json
"Leaves are green because they are made from leaves of plants, which contain chlorophyll, and they are a natural pigment that gives them their color."
~~~~

## Numerical and state checks

- All 277 REWT tests pass, including 14 conversion/distillation checks. Coverage includes independent sequential recurrence values and gradients, causal isolation, chunked decode, cache ownership, mixed-precision training, checkpoint reload, exact full-vocabulary KL gradients and document-isolated/prefix sampling.
- Actual converted-model BF16 chunk audit: relative hidden-state L2 error 0.015241, maximum absolute error 0.250000 across a 513-token prefix. BF16 arithmetic depends on chunk shape; this is approximate consistency, not exact parity.
- Carried inference state is 19,695,616 bytes (18.783 MiB) per sequence after the window fills. It stayed fixed when extending from 513 to 1,024 tokens, and continuing a cache left the earlier cache unchanged. This excludes model weights and temporary tensors.
- Full-width FP32 audit with highest matmul precision: relative chunk error 1.4113917e-06, maximum absolute error 2.6226044e-05 at 513 tokens.

Disabling only the energy-refinement steps gives test NLL 2.346825 versus 2.346841 with refinement; KL is 0.827874 versus 0.828011. This small ablation does not establish a meaningful energy benefit.

## Run the converted model

The local inference bundle includes the checkpoint, tokenizer, retained donor license and standalone Python implementation:

`runs/lfm3_conversion_20261028/converted/`

```bash
.venv-lfm3/bin/python runs/lfm3_conversion_20261028/converted/sample.py \
  --prompt 'Reply with exactly the word hello.' --tokens 48
```

The bundle is a custom PyTorch checkpoint, not a Hugging Face `AutoModel` export. It can be copied with its code and tokenizer to another compatible environment; the donor checkpoint is not needed for inference. `requirements.txt` records its Python dependencies. The original donor license is preserved as `TEACHER_LICENSE.txt`.

## Reproduce or continue recovery

Use a CUDA-capable PyTorch environment, then install the project’s `lfm-conversion` optional dependencies. The tested environment is recorded above; `.venv-lfm3` is the isolated local environment used here.

```bash
.venv-lfm3/bin/python scripts/prepare_rewt_lfm3.py --download --prepare-data --initialize
# Preparation refuses to overwrite existing prepared data or initialization.
.venv-lfm3/bin/python scripts/distill_rewt_lfm3.py \
  --output runs/lfm3_conversion_new_pilot
.venv-lfm3/bin/python scripts/distill_rewt_lfm3.py \
  --initial runs/lfm3_conversion_20261028/converted/model.pt \
  --output runs/lfm3_conversion_next --align-steps 0 \
  --distill-steps 512 --length 512 --batch 2 --lr 0.00005 \
  --train-prefix-fraction 0.5 --eval-prefix-fraction 0.5
```

Continuation starts a fresh optimizer and preserves checkpoint precision. `--all-weights` additionally unfreezes the inherited weights and increases memory use. No longer run is left running after this report. Improving capability recovery requires a broader recovery corpus, more training and independent instruction/knowledge/code/tool evaluations; local attention also needs long-context quality testing.

## Retained artifacts

- [Initial tensor transfer](initial_transfer.json) and [initial checkpoint metadata](initial_metadata.json).
- [Original pilot comparison, including failures](pilot_comparison.json).
- [Final comparison and energy ablation](comparison.json).
- [Final verification record](completed_verification.json).
- Exact earlier training sources are archived in `pilot_source/` and `recovery_source/`; their source hashes remain verifiable after the loader and sampler updates.
