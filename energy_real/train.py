"""Resumable, budgeted byte-LM experiments. Historical weights are warm starts only."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
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
    if args.learning_rule == 'pcalm':
        if args.amp != 'none':
            raise ValueError('Experimental PC-ALM requires --amp none')
        if not 1 <= args.activity_steps <= 256 or any(not math.isfinite(x) or x <= 0 for x in
                (args.state_lr, args.dual_lr, args.rho)):
            raise ValueError('Invalid PC-ALM activity schedule')


def model_config(args):
    """Validate architecture before parameter matching or device/data access."""
    cfg = Config(kind=args.kind, width=args.width, layers=args.layers, heads=args.heads,
                 slots=args.slots, ff_width=args.ff_width, chunk=args.chunk, backend=args.backend,
                 activation_checkpointing=args.activation_checkpointing,
                 energy_projection_fp32=getattr(args, "energy_projection_fp32", False),
                 memory_dim=args.memory_dim, memory_capacity=args.memory_capacity,
                 attention_window=args.attention_window, attention_every=args.attention_every)
    cfg.validate()
    softmax_layers = (args.layers if args.kind == 'softmax' else
                      args.layers // args.attention_every if args.kind == 'hybrid' else 0)
    softmax_ff = matched_ff(args.width, args.heads, args.slots, args.ff_width)
    ff = args.ff_width + round((softmax_ff - args.ff_width) * softmax_layers / args.layers)
    cfg = replace(cfg, ff_width=ff)
    cfg.validate()
    return cfg


def run(args, *, corpus_factory=Enwik8):
    validate(args)
    cfg = model_config(args)
    study=getattr(args,'shared_hybrid_study',False)
    if study and getattr(args,'shared_gpu_diagnostic',False):raise ValueError('Choose one shared-device protocol')
    shared=getattr(args,'shared_gpu_diagnostic',False) or study
    if shared:
        if ((cfg.width,cfg.layers) not in ((160,3),(208,6)) or cfg.heads!=4 or cfg.slots!=16
            or cfg.memory_dim or cfg.backend!='dmat' or cfg.ff_width>4*cfg.width+128
            or args.batch!=1 or args.seq!=512 or args.accum!=16 or args.steps>(128 if study else 8)
            or args.max_cuda_gib>4 or args.max_seconds>(600 if study else 120) or args.compile or args.learning_rule!='bp'
            or (study and cfg.kind not in ('hybrid','softmax'))):
            raise ValueError('Shared diagnostic/study exceeds its small-model, update, time or 4 GiB bounds')
    device = torch.device(args.device)
    if device.type == 'cuda':
        from .device import require_idle_gpu, set_memory_limit, available_host_bytes
        if shared:
            if not torch.cuda.is_available() or available_host_bytes()<20*2**30:
                raise RuntimeError('Shared probe needs CUDA and 20 GiB available before a 4 GiB allocation')
        else:require_idle_gpu()
        set_memory_limit(device, args.max_cuda_gib)
    train = corpus_factory(args.data, 'train', expected_sha256=args.data_sha256)
    valid = corpus_factory(args.data, 'valid')
    file_hash = train.file_identity()
    identity = {'config': asdict(cfg), 'data_sha256': file_hash, 'source_sha256': source_identity(),
                'runtime': environment(device), 'shared_gpu_diagnostic':shared, 'shared_hybrid_study':study, 'compiled': getattr(args, 'compile', False), 'campaign': getattr(args, 'campaign_identity', None), 'schedule': {k: getattr(args, k) for k in
                ('steps', 'batch', 'seq', 'accum', 'lr', 'warmup', 'seed', 'weight_decay',
                 'eval_every', 'eval_bytes', 'eval_batch', 'amp', 'max_tokens', 'max_seconds', 'max_cuda_gib',
                 'learning_rule', 'activity_steps', 'state_lr', 'dual_lr', 'rho')}}
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
            if hasattr(train, 'restore_coverage'):
                train.restore_coverage(old.get('coverage', {}))
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
            architecture = ('kind', 'vocab', 'width', 'layers', 'heads', 'slots', 'ff_width', 'prior',
                            'memory_dim', 'memory_capacity', 'attention_window', 'attention_every')
            if any(getattr(old_cfg, k) != getattr(cfg, k) for k in architecture):
                raise ValueError('Warm-start architecture mismatch')
            model.load_state_dict(old['weights'], strict=True)
        def objective(x, y):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=args.amp == 'bfloat16' and device.type == 'cuda'):
                logits, _ = model(x)
                return F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
        training_objective = torch.compile(objective) if getattr(args, 'compile', False) else objective
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
                                   'initial_sha256': initial_hash, 'coverage': getattr(train, 'coverage', None), 'log': log_identity(log_path)}, out / 'last.pt')
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
                    if device.type == 'cuda' and torch.cuda.max_memory_reserved(device) > args.max_cuda_gib * 2**30:
                        status = 'memory_budget_exhausted'
                        break
                    if device.type == 'cuda':
                        from .device import available_host_bytes
                        if available_host_bytes() < 16 * 2**30:
                            status = 'host_memory_reserve_exhausted'
                            break
                    begin = time.perf_counter()
                    lr = learning_rate(step + 1, args.steps, args.warmup, args.lr)
                    for group in optimizer.param_groups:
                        group['lr'] = lr
                    optimizer.zero_grad(set_to_none=True)
                    loss_sum = 0.
                    local_residual = None
                    for _ in range(args.accum):
                        x, y = train.batch(rng, args.batch, args.seq, device)
                        if args.learning_rule == 'pcalm':
                            from .local_lm import pcalm_loss
                            def check_activity_budget():
                                if stop['requested'] or previous_elapsed + time.perf_counter() - started >= args.max_seconds:
                                    raise TimeoutError('Stopped during PC-ALM activity inference; resume from committed checkpoint')
                            local = pcalm_loss(model, x, y, steps=args.activity_steps, state_lr=args.state_lr,
                                               dual_lr=args.dual_lr, rho=args.rho, check_budget=check_activity_budget)
                            loss = local.loss / args.accum
                            local_residual = local.residual_rms
                        else:
                            loss = training_objective(x, y) / args.accum
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
                          'step_s': duration, 'tok_s': tokens_per_step / duration,
                          'unique_targets': getattr(train, 'unique_targets', None),
                          'learning_rule': args.learning_rule, 'local_residual_rms': local_residual})
                    if step % args.eval_every == 0 or step == args.steps:
                        begin = time.perf_counter()
                        score = evaluate(model, valid, args.seq, args.eval_bytes, device, args.eval_batch,
                                         amp=args.amp == 'bfloat16', deadline=started + args.max_seconds - previous_elapsed)
                        sync(device)
                        eval_seconds += time.perf_counter() - begin
                        emit({'event': 'eval', 'step': step, 'tokens': tokens, 'valid_bpb': score['bpb'],
                              'targets': score['targets'], 'train_s': train_seconds, 'eval_s': eval_seconds})
                        if score['status'] == 'completed' and (best is None or score['bpb'] < best):
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
                 'planned_tokens': args.steps * tokens_per_step, 'unique_targets': getattr(train, 'unique_targets', None), 'best_valid_bpb': best,
                 'best_step': best_step, 'train_s': train_seconds, 'eval_s': eval_seconds,
                 'elapsed_s': previous_elapsed + time.perf_counter() - started,
                 'last_checkpoint_step': last_saved,'shared_gpu_diagnostic':shared,
                 'peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                 'peak_reserved_bytes':torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None}
        atomic_json(final, out / 'result.json')
        print(json.dumps(final), flush=True)
        return final


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', default='energy', choices=['energy', 'softmax', 'hybrid'])
    p.add_argument('--attention-window', type=int, default=0)
    p.add_argument('--attention-every', type=int, default=3)
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
    p.add_argument('--energy-projection-fp32', action='store_true', help='Keep the first energy projection in FP32 under autocast; saved in model config')
    p.add_argument('--compile', action='store_true')
    p.add_argument('--activation-checkpointing', action='store_true')
    p.add_argument('--shared-gpu-diagnostic',action='store_true',help='Explicit small, bounded shared-device diagnostic; never a timing benchmark')
    p.add_argument('--shared-hybrid-study',action='store_true',help='Selected hybrid/attention shared study, <=128 updates, <=600 seconds, <=4 GiB')
    p.add_argument('--memory-dim', type=int, default=0)
    p.add_argument('--memory-capacity', type=int, default=256)
    p.add_argument('--learning-rule', choices=['bp', 'pcalm'], default='bp')
    p.add_argument('--activity-steps', type=int, default=8)
    p.add_argument('--state-lr', type=float, default=.02)
    p.add_argument('--dual-lr', type=float, default=.1)
    p.add_argument('--rho', type=float, default=1.)
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
