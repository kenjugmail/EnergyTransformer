"""Resumable, budgeted byte-LM experiments. Historical weights are warm starts only."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import signal
import threading
import time
import numpy as np
import torch
import torch.nn.functional as F
from .model import LM, Config
from .data import Enwik8, LOCAL_CORPUS_SHA256
from .runtime import (atomic_json, atomic_checkpoint, environment, exclusive_run,
                      log_identity, reconcile_log, sha256, source_identity, sync)

FORMAT = 'energy_real_train_v1'


def matched_ff(width, heads, slots, energy_ff):
    """Match actual mixer parameter difference using the FF's 2*width+1 slope."""
    extra = (width + 1) * (3 * heads * slots - width + heads) + (2 * width + 1) * heads * slots + heads
    return max(1, energy_ff + round(extra / (2 * width + 1)))


@torch.no_grad()
def evaluate(model, corpus, length, max_bytes, device, batch=32, amp=True, deadline=None):
    training = model.training
    model.eval()
    nll, n = 0.0, 0
    status = 'completed'
    try:
        for x, y in corpus.sequential_windows(length, limit=max_bytes, batch=batch, device=device):
            if deadline is not None and time.perf_counter() >= deadline:
                status = 'budget_exhausted'
                break
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=amp and device.type == 'cuda'):
                logits, _ = model(x)
            loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten(), reduction='sum')
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite evaluation loss')
            nll += float(loss)
            n += y.numel()
    finally:
        model.train(training)
    if n == 0 and status == 'completed':
        raise ValueError('No evaluation targets')
    return {'status': status, 'bpb': nll / n / math.log(2) if n else None, 'nll_nats': nll, 'targets': n,
            'split': corpus.split, 'context_bytes': length, 'state_policy': 'reset_each_window'}


def learning_rate(step, steps, warmup, peak):
    return peak * min(1.0, step / max(1, warmup)) * (.1 + .9 * .5 * (1 + math.cos(math.pi * min(1., step / steps))))


def validate(args):
    for key in ('steps', 'batch', 'seq', 'accum', 'eval_every', 'eval_bytes', 'eval_batch', 'log_every', 'save_every'):
        if type(getattr(args, key)) is not int or getattr(args, key) < 1:
            raise ValueError(f'{key} must be a positive integer')
    if args.eval_bytes < 2 or args.warmup < 0 or args.seed < 0:
        raise ValueError('Invalid evaluation size, warmup, or seed')
    for key in ('lr', 'max_seconds', 'max_cuda_gib'):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f'{key} must be positive and finite')
    if args.weight_decay < 0 or not math.isfinite(args.weight_decay):
        raise ValueError('Invalid weight decay')
    if args.resume and args.warm_start:
        raise ValueError('Choose resume or warm start, not both')
    if args.stop_after is not None and not 0 <= args.stop_after <= args.steps:
        raise ValueError('stop_after must be an absolute step in the planned run')
    if args.max_tokens is not None and (args.max_tokens < 1 or type(args.max_tokens) is not int):
        raise ValueError('Positive integer token budget required')


def run(args, *, corpus_factory=Enwik8):
    validate(args)
    device = torch.device(args.device)
    if device.type == 'cuda':
        from .device import require_idle_gpu, set_memory_limit
        require_idle_gpu()
        set_memory_limit(device, args.max_cuda_gib)
    ff = matched_ff(args.width, args.heads, args.slots, args.ff_width) if args.kind == 'softmax' else args.ff_width
    cfg = Config(kind=args.kind, width=args.width, layers=args.layers, heads=args.heads,
                 slots=args.slots, ff_width=ff, chunk=args.chunk, backend=args.backend,
                 activation_checkpointing=args.activation_checkpointing)
    cfg.validate()
    train = corpus_factory(args.data, 'train', expected_sha256=args.data_sha256)
    valid = corpus_factory(args.data, 'valid')
    file_hash = train.file_identity()
    identity = {'config': asdict(cfg), 'data_sha256': file_hash, 'source_sha256': source_identity(),
                'runtime': environment(device), 'schedule': {k: getattr(args, k) for k in
                ('steps', 'batch', 'seq', 'accum', 'lr', 'warmup', 'seed', 'weight_decay',
                 'eval_every', 'eval_bytes', 'eval_batch', 'amp', 'max_tokens', 'max_seconds', 'max_cuda_gib')}}
    with exclusive_run(args.out, args.resume) as out:
        torch.manual_seed(args.seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(args.seed)
            torch.cuda.reset_peak_memory_stats(device)
        model = LM(cfg).to(device)
        decay = [p for p in model.parameters() if p.dim() >= 2]
        no_decay = [p for p in model.parameters() if p.dim() < 2]
        optimizer = torch.optim.AdamW([{'params': decay, 'weight_decay': args.weight_decay},
                                      {'params': no_decay, 'weight_decay': 0.}],
                                     lr=args.lr, betas=(.9, .95), eps=1e-8, foreach=False)
        rng = np.random.default_rng(args.seed)
        step = tokens = 0
        best = None
        best_step = None
        best_checkpoint = None
        elapsed = train_seconds = eval_seconds = 0.
        initial_hash = None
        log_path = out / 'steps.jsonl'
        if args.resume:
            old = torch.load(out / 'last.pt', weights_only=True, map_location='cpu')
            if old.get('format') != FORMAT or old['identity'] != identity:
                raise ValueError('Resume identity mismatch (source, data, config, schedule, or runtime)')
            reconcile_log(log_path, old['log'])
            model.load_state_dict(old['weights'], strict=True)
            optimizer.load_state_dict(old['optimizer'])
            torch.set_rng_state(old['torch_rng'])
            if device.type == 'cuda':
                torch.cuda.set_rng_state_all(old['cuda_rng'])
            rng.bit_generator.state = old['numpy_rng']
            step, tokens = old['step'], old['tokens']
            best, best_step = old['best_valid_bpb'], old['best_step']
            elapsed, train_seconds, eval_seconds = old['elapsed_s'], old['train_s'], old['eval_s']
            initial_hash = old['initial_sha256']
            best_checkpoint = old['best_checkpoint']
            if (out / 'best.pt').exists():
                candidate = torch.load(out / 'best.pt', map_location='cpu', weights_only=True)
                if candidate.get('step', -1) > step:
                    import uuid
                    (out / 'best.pt').rename(out / f'recovered_best.{uuid.uuid4().hex}.pt')
            if best_checkpoint is not None:
                atomic_checkpoint(best_checkpoint, out / 'best.pt')
            elif (out / 'best.pt').exists():
                import uuid
                (out / 'best.pt').rename(out / f'recovered_best.{uuid.uuid4().hex}.pt')
        elif args.warm_start:
            initial_hash = sha256(args.warm_start)
            old = torch.load(args.warm_start, weights_only=True, map_location='cpu')
            old_cfg = Config(**old['config'])
            # Kernel/backend choices may change for an explicitly labelled new run.
            architecture = ('kind', 'vocab', 'width', 'layers', 'heads', 'slots', 'ff_width', 'prior')
            if any(getattr(old_cfg, k) != getattr(cfg, k) for k in architecture):
                raise ValueError('Warm-start architecture mismatch')
            model.load_state_dict(old['weights'], strict=True)
        tokens_per_step = args.batch * args.seq * args.accum
        meta = {'format': FORMAT, 'identity': identity, 'parameters': model.parameters_count(),
                'planned_tokens': args.steps * tokens_per_step, 'tokens_per_step': tokens_per_step,
                'initial_sha256': initial_hash, 'initialization': 'warm_start' if initial_hash else 'random'}
        if not args.resume:
            atomic_json(meta, out / 'run.json')
        started = time.perf_counter()
        previous_elapsed = elapsed
        stop = {'requested': False}
        handlers = {}
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGTERM, signal.SIGINT):
                handlers[sig] = signal.signal(sig, lambda *_: stop.update(requested=True))
        last_saved = step
        status = 'running'
        with log_path.open('a') as log:
            def emit(row):
                log.write(json.dumps(row, allow_nan=False) + '\n')
                if row.get('event') != 'step' or row['step'] % args.log_every == 0:
                    print(json.dumps(row), flush=True)

            def save():
                nonlocal last_saved
                log.flush()
                import os
                os.fsync(log.fileno())
                sync(device)
                atomic_checkpoint({'format': FORMAT, 'identity': identity, 'config': asdict(cfg),
                                   'weights': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                   'torch_rng': torch.get_rng_state(),
                                   'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
                                   'numpy_rng': rng.bit_generator.state, 'step': step, 'tokens': tokens,
                                   'best_valid_bpb': best, 'best_step': best_step, 'best_checkpoint': best_checkpoint,
                                   'elapsed_s': previous_elapsed + time.perf_counter() - started,
                                   'train_s': train_seconds, 'eval_s': eval_seconds,
                                   'initial_sha256': initial_hash, 'log': log_identity(log_path)}, out / 'last.pt')
                last_saved = step

            try:
                if not args.resume:
                    save()
                while step < args.steps:
                    elapsed = previous_elapsed + time.perf_counter() - started
                    if stop['requested'] or (args.stop_after is not None and step >= args.stop_after):
                        status = 'interrupted'
                        break
                    if elapsed >= args.max_seconds or (args.max_tokens is not None and tokens + tokens_per_step > args.max_tokens):
                        status = 'budget_exhausted'
                        break
                    if device.type == 'cuda' and torch.cuda.max_memory_allocated(device) > args.max_cuda_gib * 2**30:
                        status = 'memory_budget_exhausted'
                        break
                    begin = time.perf_counter()
                    lr = learning_rate(step + 1, args.steps, args.warmup, args.lr)
                    for group in optimizer.param_groups:
                        group['lr'] = lr
                    optimizer.zero_grad(set_to_none=True)
                    loss_sum = 0.
                    for _ in range(args.accum):
                        x, y = train.batch(rng, args.batch, args.seq, device)
                        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                            enabled=args.amp == 'bfloat16' and device.type == 'cuda'):
                            logits, _ = model(x)
                            loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten()) / args.accum
                        loss.backward()
                        loss_sum += float(loss.detach())
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    if not math.isfinite(loss_sum):
                        raise FloatingPointError('Non-finite training loss')
                    optimizer.step()
                    sync(device)
                    duration = time.perf_counter() - begin
                    train_seconds += duration
                    step += 1
                    tokens += tokens_per_step
                    emit({'event': 'step', 'step': step, 'tokens': tokens, 'loss': loss_sum,
                          'lr': lr, 'grad_norm': float(norm), 'train_s': train_seconds,
                          'step_s': duration, 'tok_s': tokens_per_step / duration})
                    if step % args.eval_every == 0 or step == args.steps:
                        begin = time.perf_counter()
                        score = evaluate(model, valid, args.seq, args.eval_bytes, device, args.eval_batch,
                                         amp=args.amp == 'bfloat16')
                        sync(device)
                        eval_seconds += time.perf_counter() - begin
                        emit({'event': 'eval', 'step': step, 'tokens': tokens, 'valid_bpb': score['bpb'],
                              'targets': score['targets'], 'train_s': train_seconds, 'eval_s': eval_seconds})
                        if best is None or score['bpb'] < best:
                            best, best_step = score['bpb'], step
                            best_checkpoint = {'config': asdict(cfg),
                                               'weights': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                                               'step': step, 'tokens': tokens, 'valid_bpb': best, 'identity': identity}
                            atomic_checkpoint(best_checkpoint, out / 'best.pt')
                    if step % args.save_every == 0 or step % args.eval_every == 0:
                        save()
                if step == args.steps:
                    status = 'completed'
                save()
            except BaseException as exc:
                # Partial optimizer updates are never promoted as resumable state.
                atomic_json({'status': 'failed', 'error': f'{type(exc).__name__}: {exc}',
                             'last_checkpoint_step': last_saved, 'completed_steps': step,
                             'tokens': tokens}, out / 'result.json')
                raise
            finally:
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)
        final = {'status': status, 'kind': cfg.kind, 'backend': cfg.backend,
                 'parameters': model.parameters_count(), 'steps': step, 'tokens': tokens,
                 'planned_tokens': args.steps * tokens_per_step, 'best_valid_bpb': best,
                 'best_step': best_step, 'train_s': train_seconds, 'eval_s': eval_seconds,
                 'elapsed_s': previous_elapsed + time.perf_counter() - started,
                 'last_checkpoint_step': last_saved}
        atomic_json(final, out / 'result.json')
        print(json.dumps(final), flush=True)
        return final


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', default='energy', choices=['energy', 'softmax'])
    p.add_argument('--data', default='data/enwik8')
    p.add_argument('--data-sha256', default=LOCAL_CORPUS_SHA256)
    p.add_argument('--out', required=True)
    for name, default in [('width', 384), ('layers', 6), ('heads', 4), ('slots', 16), ('ff-width', 1536),
                          ('chunk', 64), ('seq', 512), ('batch', 16), ('accum', 1), ('steps', 5000),
                          ('warmup', 200), ('seed', 1234), ('eval-every', 500), ('eval-bytes', 1_000_000),
                          ('eval-batch', 32), ('log-every', 50), ('save-every', 100)]:
        p.add_argument('--' + name, type=int, default=default)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--weight-decay', type=float, default=.1)
    p.add_argument('--backend', choices=['dmat', 'matmul'], default='dmat')
    p.add_argument('--amp', choices=['none', 'bfloat16'], default='bfloat16')
    p.add_argument('--activation-checkpointing', action='store_true')
    p.add_argument('--device', default='cuda')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--warm-start', type=Path)
    p.add_argument('--stop-after', type=int)
    p.add_argument('--max-seconds', type=float, default=1800.)
    p.add_argument('--max-tokens', type=int)
    p.add_argument('--max-cuda-gib', type=float, default=48.)
    return p


def main():
    run(parser().parse_args())


if __name__ == '__main__':
    main()
