"""Write the conversion report from retained numerical results and raw samples."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/lfm3_conversion_20261028'


def main():
    comparison = json.loads((OUT/'comparison.json').read_text())
    pilot_comparison = json.loads((OUT/'pilot_comparison.json').read_text())
    pilot = json.loads((OUT/'pilot_summary.json').read_text())
    recovery = json.loads((OUT/'recovery_summary.json').read_text())
    prefix = json.loads((OUT/'prefix_summary.json').read_text())
    initial = comparison['models']['initial']['test']
    middle = pilot_comparison['models']['recovered']['test']
    result = comparison['models']['recovered']
    final = result['test']; audit = result['cached_decode_audit']
    nll_difference = 100*(final['nll']/final['teacher_nll']-1)
    nll_direction = 'above' if nll_difference >= 0 else 'below'
    rows = [('Donor LFM3-1.7B', final['teacher_nll'], 0., 1.),
            ('Converted, before recovery', initial['nll'], initial['kl'], initial['agreement']),
            ('After the 256-token pilot', middle['nll'], middle['kl'], middle['agreement']),
            ('Final recovered student', final['nll'], final['kl'], final['agreement'])]
    lines = [
        '# LFM3-1.7B → streaming REWT: conversion and recovery', '',
        'A standalone **1,784,445,792-parameter** streaming REWT model now runs from LFM3-derived weights. Its 13 original gated-convolution slots are replaced by gated delta memory, all 11 attention layers use a 256-token window, and two final latent-energy refinement steps precede the inherited vocabulary projection. The model has no donor forward path at inference.', '',
        '**The architecture conversion is implemented; full capability recovery is not established.** This is one short conversion trajectory without independent training replications. Continuation loss improves sharply, while generated answers must be inspected separately. It is not a validated replacement for the original LFM3 assistant.', '',
        '[Architecture, transfer map and equations](DESIGN.md) · [Complete raw comparison and samples](comparison.json) · [Environment](environment.json)', '',
        '## Held-out continuation comparison', '',
        'All rows below use the same 8,192 next-token positions: 32 fixed, document-contained 256-token crops sampled from the prepared official test split. NLL and KL are natural-log units per token; lower is better. Agreement is with the donor’s top next-token choice under the reference conversation prefix, **not answer accuracy**.', '',
        '| Model | NLL ↓ | KL to donor ↓ | Top-token agreement ↑ |',
        '|---|---:|---:|---:|']
    for label, nll, kl, agreement in rows:
        lines.append(f'| {label} | {nll:.4f} | {kl:.4f} | {100*agreement:.2f}% |')
    lines += ['', f'The final student NLL is {abs(nll_difference):.2f}% {nll_direction} the donor on these crops. This is a loss comparison, not a percentage of capabilities retained. The pilot test and four sample prompts were inspected before deciding to continue recovery, so the table is an exploratory comparison, not a sealed final benchmark. None of the runs trains on validation or test conversations.', '',
        '## What was trained', '',
        '- Donor: `LiquidAI/LFM3-1.7B`, immutable revision `40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd`.',
        '- The first run used 128 layer-alignment updates and 256 full-model distillation updates, two 256-token sequences per update. It processed 196,608 token exposures.',
        f'- The second run continued the selected pilot checkpoint for 512 distillation updates at length 512, two sequences per update, with a fresh optimizer and learning rate 5e-5. It processed {recovery["training_tokens"]:,} further token exposures.',
        f'- A final 128-update pass uses 512-token conversation prefixes (131,072 token exposures), learning rate 2e-5 and seed 281707. Its validation/test crops also begin at conversation boundaries. The final artifact selects prefix-pass update {prefix["best_step"]} by the lowest KL on 4,096 fixed validation positions. Earlier training seeds are 281702 and 281706; initialization seed 281701 is retained separately in checkpoint provenance.',
        '- Embeddings, feedforward layers and FFN normalization stayed frozen. The mixers, operator norms and energy head supplied 389,739,360 trainable parameters. This is a backbone conversion with substantial weight training, not a LoRA wrapper around the donor.',
        '- Objective: exact full-vocabulary teacher KL + 0.1 hard next-token NLL + 0.1 normalized hidden-state MSE. Validation KL selected each checkpoint; there was one seed per run.',
        f'- Measured loop/evaluation/checkpoint time: {pilot["elapsed_seconds"]:.1f}s, {recovery["elapsed_seconds"]:.1f}s and {prefix["elapsed_seconds"]:.1f}s for the three stages, excluding initial model loading. Peak CUDA allocations were {pilot["peak_cuda_allocated_gib"]:.2f}, {recovery["peak_cuda_allocated_gib"]:.2f} and {prefix["peak_cuda_allocated_gib"]:.2f} GiB on the shared GB10. These are not isolated performance benchmarks.', '',
        '[Pilot plan](pilot_plan.json) · [Pilot training log](pilot_metrics.jsonl) · [Continuation plan](recovery_plan.json) · [Continuation training log](recovery_metrics.jsonl) · [Prefix-pass plan](prefix_plan.json) · [Prefix-pass training log](prefix_metrics.jsonl) · [Data manifest](corpus_manifest.json)', '',
        'A crop audit found that only 12 of the first two stages’ 1,792 training crops began at conversation start. The final pass addresses that distribution gap by retaining the beginning of the conversation; it uses ordinary training conversations, never the four comparison prompts as targeted training examples. This is a motivated recipe adjustment, not proof that crop placement caused the earlier generation failures. [Exact crop coverage](crop_coverage.json).', '',
        'The corpus has 8,192 selected training conversations (7,635,132 stored tokens), 128 validation conversations and 128 test conversations. Exact conversation hashes are disjoint. Crops never cross document boundaries. The longer run only samples documents with at least 513 tokens. Metrics include template and user tokens; no broad knowledge, code, tool-use, long-context-quality or instruction-following benchmark was completed.', '',
        '## Unedited final responses', '',
        'The final checkpoint says `hello` exactly and now extracts Maya/Boston into valid JSON. It computes 42 but ignores the arithmetic prompt’s “just the number” format requirement. Its leaf-color response mentions chlorophyll but contains inaccurate and confused wording. These remaining failures prevent a claim of complete capability recovery.', '',
        'These four prompts were fixed before the first recovery comparison. Greedy generation uses the original non-thinking chat template and a 48-token output limit. Responses that exhaust that limit are truncated. All donor, initialized and recovered outputs and token IDs are retained in the raw comparison.', '']
    for row in result['generation']:
        # Render arbitrary generation safely as a JSON string in a fixed fence.
        lines += [f'**Prompt:** {row["prompt"]}', '', '~~~~json', json.dumps(row['response'], ensure_ascii=False), '~~~~', '']
    lines += ['## Numerical and state checks', '',
        '- All 277 REWT tests pass, including 14 conversion/distillation checks. Coverage includes independent sequential recurrence values and gradients, causal isolation, chunked decode, cache ownership, mixed-precision training, checkpoint reload, exact full-vocabulary KL gradients and document-isolated/prefix sampling.',
        f'- Actual converted-model BF16 chunk audit: relative hidden-state L2 error {audit["chunk_relative_l2"]:.6f}, maximum absolute error {audit["chunk_max_absolute_error"]:.6f} across a 513-token prefix. BF16 arithmetic depends on chunk shape; this is approximate consistency, not exact parity.',
        f'- Carried inference state is {audit["owned_state_bytes"]:,} bytes ({audit["owned_state_bytes"]/2**20:.3f} MiB) per sequence after the window fills. It stayed fixed when extending from 513 to 1,024 tokens, and continuing a cache left the earlier cache unchanged. This excludes model weights and temporary tensors.']
    if 'float32_chunk_audit' in result:
        full = result['float32_chunk_audit']
        lines += [f'- Full-width FP32 audit with highest matmul precision: relative chunk error {full["relative_l2"]:.8g}, maximum absolute error {full["max_absolute_error"]:.8g} at 513 tokens.']
    off = result['energy_off_test']
    lines += ['', f'Disabling only the energy-refinement steps gives test NLL {off["nll"]:.6f} versus {final["nll"]:.6f} with refinement; KL is {off["kl"]:.6f} versus {final["kl"]:.6f}. This small ablation does not establish a meaningful energy benefit.', '',
        '## Run the converted model', '',
        'The local inference bundle includes the checkpoint, tokenizer, retained donor license and standalone Python implementation:', '',
        '`runs/lfm3_conversion_20261028/converted/`', '',
        '```bash',
        '.venv-lfm3/bin/python runs/lfm3_conversion_20261028/converted/sample.py \\',
        "  --prompt 'Reply with exactly the word hello.' --tokens 48", '```', '',
        'The bundle is a custom PyTorch checkpoint, not a Hugging Face `AutoModel` export. It can be copied with its code and tokenizer to another compatible environment; the donor checkpoint is not needed for inference. `requirements.txt` records its Python dependencies. The original donor license is preserved as `TEACHER_LICENSE.txt`.', '',
        '## Reproduce or continue recovery', '',
        'Use a CUDA-capable PyTorch environment, then install the project’s `lfm-conversion` optional dependencies. The tested environment is recorded above; `.venv-lfm3` is the isolated local environment used here.', '',
        '```bash',
        '.venv-lfm3/bin/python scripts/prepare_rewt_lfm3.py --download --prepare-data --initialize',
        '# Preparation refuses to overwrite existing prepared data or initialization.',
        '.venv-lfm3/bin/python scripts/distill_rewt_lfm3.py \\',
        '  --output runs/lfm3_conversion_new_pilot',
        '.venv-lfm3/bin/python scripts/distill_rewt_lfm3.py \\',
        '  --initial runs/lfm3_conversion_20261028/converted/model.pt \\',
        '  --output runs/lfm3_conversion_next --align-steps 0 \\',
        '  --distill-steps 512 --length 512 --batch 2 --lr 0.00005 \\',
        '  --train-prefix-fraction 0.5 --eval-prefix-fraction 0.5',
        '```', '',
        'Continuation starts a fresh optimizer and preserves checkpoint precision. `--all-weights` additionally unfreezes the inherited weights and increases memory use. No longer run is left running after this report. Improving capability recovery requires a broader recovery corpus, more training and independent instruction/knowledge/code/tool evaluations; local attention also needs long-context quality testing.', '',
        '## Retained artifacts', '',
        '- [Initial tensor transfer](initial_transfer.json) and [initial checkpoint metadata](initial_metadata.json).',
        '- [Original pilot comparison, including failures](pilot_comparison.json).',
        '- [Final comparison and energy ablation](comparison.json).',
        '- [Final verification record](completed_verification.json).',
        '- Exact earlier training sources are archived in `pilot_source/` and `recovery_source/`; their source hashes remain verifiable after the loader and sampler updates.', '']
    (OUT/'FINDINGS.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
