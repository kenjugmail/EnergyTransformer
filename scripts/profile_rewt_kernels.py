"""Attribute reference REWT execution without changing its learned weights."""
import argparse
from dataclasses import replace
from functools import wraps
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from rewt.position_memory import load_learning_checkpoint
from rewt.data import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=('reference', 'krylov2'), default='reference')
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, _ = load_learning_checkpoint(args.checkpoint)
    if args.backend != model.config.refinement_backend:
        adapted = type(model)(replace(model.config, refinement_backend=args.backend))
        adapted.load_state_dict(model.state_dict())
        model = adapted
    if args.backend == 'krylov2':
        from rewt.kernels import krylov2_refine
        def profiled_kernel(*a, **kw):
            with torch.profiler.record_function('rewt::krylov2'):
                return krylov2_refine(*a, **kw)
        model._compiled_refinement = profiled_kernel
    for name in ('condition', 'energy', 'gradient', 'retrieve', 'observed_bank'):
        original = getattr(model, name)
        def wrapped(*a, _name=name, _fn=original, **kw):
            with torch.profiler.record_function('rewt::' + _name):
                return _fn(*a, **kw)
        setattr(model, name, wraps(original)(wrapped))
    generator = torch.Generator().manual_seed(22101)
    result = {}
    for name, batch, length, training, decode in [
        ('prefill', 24, 16, False, False),
        ('training', 24, 16, True, False),
        ('decode', 1, 1, False, True),
    ]:
        model.train(training)
        tokens = torch.randint(0, 256, (batch, length), generator=generator)
        state = None
        if decode:
            with torch.inference_mode():
                state = model(torch.randint(0, 256, (1, 128), generator=generator)).state
        for _ in range(2):
            with torch.set_grad_enabled(training):
                warmup = model(tokens, state=state)
                if training:
                    warmup.logits.square().mean().backward()
            model.zero_grad(set_to_none=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            with torch.set_grad_enabled(training):
                with torch.profiler.record_function('rewt::forward'):
                    output = model(tokens, state=state)
                if training:
                    with torch.profiler.record_function('rewt::backward'):
                        output.logits.square().mean().backward()
        events = profile.key_averages()
        result[name] = dict(
            batch=batch, length=length,
            stages=[dict(name=x.key, calls=x.count, total_us=x.cpu_time_total,
                         self_us=x.self_cpu_time_total)
                    for x in events if x.key.startswith('rewt::')],
            operators=[dict(name=x.key, calls=x.count, self_us=x.self_cpu_time_total)
                       for x in sorted(events, key=lambda e: e.self_cpu_time_total, reverse=True)[:15]],
            cost=output.details['cost'])
        model.zero_grad(set_to_none=True)
    result['scope'] = 'CPU1 profiler attribution; timings include profiler overhead and are not benchmarks.'
    result['source_sha256'] = {p.name: file_sha256(p) for p in (ROOT/'rewt').glob('*.py')}
    result['script_sha256'] = file_sha256(Path(__file__))
    result['checkpoint_sha256'] = file_sha256(args.checkpoint)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v['stages'] for k, v in result.items() if isinstance(v, dict) and 'stages' in v}, indent=2))


if __name__ == '__main__':
    main()
