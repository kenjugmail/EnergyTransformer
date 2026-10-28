"""Recover the standalone LFM3→REWT student with layer and logit distillation."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from transformers import AutoModelForCausalLM

from rewt.lfm_conversion import load_converted
from rewt.lfm_distill import ConversationTokens, TeacherOperators, relative_mse, vocabulary_losses
from scripts.prepare_rewt_lfm3 import TEACHER_REVISION, sha, save_json


def evaluate(student, teacher, corpus, args, split='valid', batches=None):
    student.eval(); rng = np.random.default_rng(args.seed+100 if split == 'valid' else args.seed+200)
    totals = dict(kl=0., nll=0., teacher_nll=0., agreement=0., hidden_relative_mse=0.)
    count = args.eval_batches if batches is None else batches
    with torch.no_grad(), torch.autocast(args.device, dtype=torch.bfloat16):
        for _ in range(count):
            tokens, targets = corpus.batch(args.batch, rng, args.device,
                                          prefix_fraction=getattr(args, 'eval_prefix_fraction', 0.))
            reference = teacher.model(tokens, use_cache=False).last_hidden_state
            hidden, _ = student.hidden(tokens)
            metrics = vocabulary_losses(hidden, reference, targets, student.embedding.weight,
                                        teacher.model.embed_tokens.weight, chunk=args.logit_chunk, checkpoint_chunks=False)
            metrics['hidden_relative_mse'] = relative_mse(hidden, reference)
            for name, value in metrics.items():
                totals[name] += float(value)/count
    totals['tokens'] = count*args.batch*args.length
    return totals


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher', type=Path, default=ROOT/'data/lfm3_1_7b_teacher'/TEACHER_REVISION)
    p.add_argument('--initial', type=Path, default=ROOT/'runs/lfm3_conversion_20261028/initial/initialized.pt')
    p.add_argument('--corpus', type=Path, default=ROOT/'data/lfm3_conversion_corpus')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--align-steps', type=int, default=128)
    p.add_argument('--distill-steps', type=int, default=256)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--length', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--align-lr', type=float, default=3e-4)
    p.add_argument('--hard-weight', type=float, default=.1)
    p.add_argument('--hidden-weight', type=float, default=.1)
    p.add_argument('--logit-chunk', type=int, default=32)
    p.add_argument('--eval-batches', type=int, default=4)
    p.add_argument('--eval-every', type=int, default=64)
    p.add_argument('--test-batches', type=int, default=16)
    p.add_argument('--seed', type=int, default=281702)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--all-weights', action='store_true', help='Unfreeze inherited FFNs/embeddings for end-to-end recovery (more memory).')
    p.add_argument('--train-prefix-fraction', type=float, default=0., help='Fraction of training crops starting at the conversation boundary.')
    p.add_argument('--eval-prefix-fraction', type=float, default=0., help='Same prefix sampling policy for fixed validation/test crops.')
    a = p.parse_args()
    if min(a.batch, a.length, a.eval_batches, a.eval_every, a.test_batches) < 1 or min(a.align_steps, a.distill_steps) < 0:
        p.error('invalid step, batch or length setting')
    if not 0 <= a.train_prefix_fraction <= 1 or not 0 <= a.eval_prefix_fraction <= 1:
        p.error('prefix fractions must be between zero and one')
    if a.output.exists():
        p.error('use a new output directory; checkpoints are never overwritten across runs')
    a.output.mkdir(parents=True)
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    train = ConversationTokens(a.corpus, 'train', a.length)
    valid = ConversationTokens(a.corpus, 'valid', a.length)
    test = ConversationTokens(a.corpus, 'test', a.length)
    source = ['rewt/lfm_conversion.py', 'rewt/lfm_distill.py', 'scripts/distill_rewt_lfm3.py']
    plan = dict(arguments={k:str(v) if isinstance(v, Path) else v for k,v in vars(a).items()},
                initial_sha256=sha(a.initial), corpus_manifest_sha256=sha(a.corpus/'manifest.json'),
                source_sha256={f:sha(ROOT/f) for f in source},
                selection='Lowest validation KL among initial, aligned and end-to-end checkpoints; test only after selection.')
    save_json(a.output/'plan.json', plan)
    teacher = AutoModelForCausalLM.from_pretrained(a.teacher, trust_remote_code=True, local_files_only=True,
                 dtype=torch.bfloat16, device_map=a.device, attn_implementation='sdpa').eval().requires_grad_(False)
    # Preserve mixed-precision learned weights when continuing a converted run.
    student, initial_metadata = load_converted(a.initial, device=a.device)
    parameters = student.trainable_conversion_parameters()
    for parameter in parameters:
        parameter.data = parameter.data.float()
    start = time.perf_counter(); history = []
    def log(record):
        record['elapsed_seconds'] = time.perf_counter()-start
        history.append(record)
        with (a.output/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        print(json.dumps(record), flush=True)
    def save(name, stage, step):
        student.save(a.output/name, metadata=dict(initial_metadata, stage=stage,
                      training_steps=initial_metadata.get('training_steps', 0)+step, run_training_steps=step,
                      run=str(a.output.resolve()), corpus_manifest_sha256=plan['corpus_manifest_sha256']))
    metrics = evaluate(student, teacher, valid, a)
    log(dict(stage='initial', validation=metrics))
    best = metrics['kl']; best_stage = 'initial'; best_step = 0
    save('best.pt', best_stage, 0)
    optimizer = torch.optim.AdamW(parameters, lr=a.align_lr, weight_decay=.01)
    hooks = TeacherOperators(teacher)
    for step in range(1, a.align_steps+1):
        student.train(); optimizer.zero_grad(set_to_none=True)
        tokens, _ = train.batch(a.batch, rng, a.device, prefix_fraction=a.train_prefix_fraction)
        with torch.no_grad(), torch.autocast(a.device, dtype=torch.bfloat16):
            teacher.model(tokens, use_cache=False)
        losses = []
        for index, block in enumerate(student.blocks):
            with torch.autocast(a.device, dtype=torch.bfloat16):
                prediction, _ = block.operator(hooks.inputs[index])
                loss = relative_mse(prediction, hooks.outputs[index])
            (loss/len(student.blocks)).backward(); losses.append(float(loss.detach()))
        hooks.clear()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 8 == 0 or step == a.align_steps:
            log(dict(stage='alignment', step=step, loss=float(np.mean(losses)), layer_losses=losses, grad_norm=float(norm)))
    hooks.close(); del optimizer
    if a.align_steps:
        metrics = evaluate(student, teacher, valid, a)
        log(dict(stage='aligned', validation=metrics))
        save('aligned.pt', 'aligned', a.align_steps)
        if metrics['kl'] < best:
            best = metrics['kl']; best_stage = 'aligned'; best_step = a.align_steps
            save('best.pt', best_stage, best_step)
    if a.all_weights:
        student.requires_grad_(True)
        for parameter in student.parameters():
            parameter.data = parameter.data.float()
        parameters = list(student.parameters())
    student.gradient_checkpointing = True
    optimizer = torch.optim.AdamW(parameters, lr=a.lr, weight_decay=.01)
    for step in range(1, a.distill_steps+1):
        student.train(); optimizer.zero_grad(set_to_none=True)
        tokens, targets = train.batch(a.batch, rng, a.device, prefix_fraction=a.train_prefix_fraction)
        with torch.no_grad(), torch.autocast(a.device, dtype=torch.bfloat16):
            reference = teacher.model(tokens, use_cache=False).last_hidden_state
        with torch.autocast(a.device, dtype=torch.bfloat16):
            hidden, _ = student.hidden(tokens)
            losses = vocabulary_losses(hidden, reference, targets, student.embedding.weight,
                        teacher.model.embed_tokens.weight, chunk=a.logit_chunk)
            hidden_loss = relative_mse(hidden, reference)
            loss = losses['kl']+a.hard_weight*losses['nll']+a.hidden_weight*hidden_loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('nonfinite distillation objective')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        # Short linear warmup, then cosine decay; this does not use validation.
        factor = min(1., step/10)*(.2+.8*.5*(1+np.cos(np.pi*step/max(1, a.distill_steps))))
        for group in optimizer.param_groups:
            group['lr'] = a.lr*factor
        optimizer.step()
        if step == 1 or step % 8 == 0 or step == a.distill_steps:
            log(dict(stage='distillation', step=step, loss=float(loss.detach()), grad_norm=float(norm),
                     hidden_relative_mse=float(hidden_loss.detach()), **{k:float(v.detach()) for k,v in losses.items()}))
        if step % a.eval_every == 0 or step == a.distill_steps:
            metrics = evaluate(student, teacher, valid, a)
            log(dict(stage='validation', step=step, validation=metrics))
            if metrics['kl'] < best:
                best = metrics['kl']; best_stage = 'distilled'; best_step = a.align_steps+step
                save('best.pt', best_stage, best_step)
    save('last.pt', 'distilled', a.align_steps+a.distill_steps)
    del optimizer, parameters, student
    if a.device == 'cuda':
        torch.cuda.empty_cache()
    student, _ = load_converted(a.output/'best.pt', device=a.device)
    metrics = evaluate(student, teacher, test, a, split='test', batches=a.test_batches)
    report = dict(best_stage=best_stage, best_step=best_step, best_validation_kl=best,
                  test=metrics, training_tokens=(a.align_steps+a.distill_steps)*a.batch*a.length,
                  trainable_parameters=sum(p.numel() for p in student.parameters()) if a.all_weights else sum(p.numel() for p in student.trainable_conversion_parameters()),
                  parameters=sum(p.numel() for p in student.parameters()), elapsed_seconds=time.perf_counter()-start,
                  best_sha256=sha(a.output/'best.pt'), peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if a.device == 'cuda' else None,
                  limitations='Short conversion pilot, not capability recovery certification. Evaluation uses cropped chat continuations; no long-context or broad benchmark claim.')
    save_json(a.output/'summary.json', report); print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
