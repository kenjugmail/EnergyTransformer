"""Compare frozen initial/recovered students with the donor and audit decoding."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from rewt.lfm_conversion import load_converted
from rewt.lfm_distill import ConversationTokens
from scripts.distill_rewt_lfm3 import evaluate
from scripts.prepare_rewt_lfm3 import TEACHER_REVISION, save_json, sha

PROMPTS = [
    'Reply with exactly the word hello.',
    'Extract the name and city as JSON with keys name and city. Text: Maya lives in Boston. Return only JSON.',
    'What is 17 plus 25? Answer with just the number.',
    'Explain in one sentence why leaves are green.',
]


def generate(student, tokenizer, text, args, *, rounds=None):
    tokens = tokenizer.apply_chat_template([dict(role='user', content=text)], tokenize=True,
              add_generation_prompt=True, enable_thinking=False, return_dict=False, return_tensors='pt').to(args.device)
    output = []; start = time.perf_counter()
    with torch.no_grad(), torch.autocast(args.device, dtype=torch.bfloat16):
        logits, state = student(tokens, return_state=True, last_only=True, rounds=rounds)
        for _ in range(args.generate_tokens):
            next_id = logits[:, -1].argmax(-1, keepdim=True); value = int(next_id.item()); output.append(value)
            if value == student.config.eos_token_id:
                break
            logits, state = student(next_id, state, return_state=True, last_only=True, rounds=rounds)
    return dict(prompt=text, response=tokenizer.decode(output, skip_special_tokens=True), token_ids=output,
                seconds=time.perf_counter()-start, state_bytes=state.allocated_bytes())


def cached_audit(student, corpus, args):
    long_data = ConversationTokens(corpus, 'test', 1024)
    tokens, _ = long_data.batch(1, np.random.default_rng(281705), args.device)
    with torch.no_grad(), torch.autocast(args.device, dtype=torch.bfloat16):
        full, _ = student.hidden(tokens[:, :513])
        state = None; chunks = []
        for first, last in [(0, 1), (1, 127), (127, 129), (129, 257), (257, 513)]:
            x, state = student.hidden(tokens[:, first:last], state, return_state=True)
            chunks.append(x)
        cached = torch.cat(chunks, 1)
        relative = float((full.float()-cached.float()).norm()/full.float().norm())
        maximum = float((full.float()-cached.float()).abs().max())
        size = state.allocated_bytes(); old = [(x.clone(), y.clone()) for x,y in state.layers]
        for first in range(513, 1024, 128):
            _, extended = student.hidden(tokens[:, first:first+128], state, return_state=True)
            if first == 513:
                for before, preserved in zip(state.layers, old):
                    for x, y in zip(before, preserved):
                        torch.testing.assert_close(x, y, atol=0, rtol=0)
            state = extended
            assert state.allocated_bytes() == size, 'carried state grew beyond its window'
        # BF16 matmuls can round differently when the sequence/chunk shape changes.
        # Strict double-precision algebra and gradients are covered by unit tests.
        assert relative < .05, f'excessive BF16 chunk drift: {relative}'
        assert torch.isfinite(full).all() and torch.isfinite(cached).all()
    return dict(prefix=513, extended_prefix=1024, chunk_relative_l2=relative,
                chunk_max_absolute_error=maximum, acceptance_relative_l2=.05,
                owned_state_bytes=size, state_unchanged_after_branch=True, bounded_after_extension=True,
                dtype='BF16 autocast with FP32 recurrent state; shape-dependent rounding is not exact parity')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher', type=Path, default=ROOT/'data/lfm3_1_7b_teacher'/TEACHER_REVISION)
    p.add_argument('--initial', type=Path, default=ROOT/'runs/lfm3_conversion_20261028/initial/initialized.pt')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--corpus', type=Path, default=ROOT/'data/lfm3_conversion_corpus')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--length', type=int, default=256)
    p.add_argument('--eval-batches', type=int, default=16)
    p.add_argument('--logit-chunk', type=int, default=32)
    p.add_argument('--seed', type=int, default=281702)
    p.add_argument('--generate-tokens', type=int, default=48)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    a = p.parse_args(); torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(a.initial.parent, local_files_only=True)
    teacher = AutoModelForCausalLM.from_pretrained(a.teacher, trust_remote_code=True, local_files_only=True,
              dtype=torch.bfloat16, device_map=a.device, attn_implementation='sdpa').eval().requires_grad_(False)
    report = dict(teacher_revision=TEACHER_REVISION, corpus_manifest_sha256=sha(a.corpus/'manifest.json'),
                  evaluation='Fixed disjoint test crops, selected checkpoint is frozen before this comparison.', models={})
    corpus = ConversationTokens(a.corpus, 'test', a.length)
    for name, path in [('initial', a.initial), ('recovered', a.checkpoint)]:
        student, metadata = load_converted(path, device=a.device); student.eval()
        metrics = evaluate(student, teacher, corpus, a, split='test')
        row = dict(path=str(path.resolve()), sha256=sha(path), metadata=metadata, test=metrics)
        row['generation'] = [generate(student, tokenizer, text, a) for text in PROMPTS]
        if name == 'recovered':
            row['cached_decode_audit'] = cached_audit(student, a.corpus, a)
            original_rounds = student.config.rounds
            # Energy ablation uses identical weights/data; no extra checkpoint selection.
            from dataclasses import replace
            student.config = replace(student.config, rounds=0)
            student.energy.config = student.config
            row['energy_off_test'] = evaluate(student, teacher, corpus, a, split='test')
            student.config = replace(student.config, rounds=original_rounds)
            student.energy.config = student.config
        report['models'][name] = row; save_json(a.output, report)
        print(json.dumps(dict(model=name, test=metrics, generation=row['generation'])), flush=True)
        del student
        if a.device == 'cuda':
            torch.cuda.empty_cache()
    donor_samples = []
    with torch.no_grad():
        for text in PROMPTS:
            tokens = tokenizer.apply_chat_template([dict(role='user', content=text)], tokenize=True,
                     add_generation_prompt=True, enable_thinking=False, return_dict=False, return_tensors='pt').to(a.device)
            generated = teacher.generate(tokens, max_new_tokens=a.generate_tokens, do_sample=False,
                          pad_token_id=teacher.config.pad_token_id)[0, tokens.shape[1]:].tolist()
            donor_samples.append(dict(prompt=text, token_ids=generated, response=tokenizer.decode(generated, skip_special_tokens=True)))
    report['teacher_generation'] = donor_samples
    save_json(a.output, report); print(json.dumps(dict(teacher_generation=donor_samples)), flush=True)


if __name__ == '__main__':
    main()
