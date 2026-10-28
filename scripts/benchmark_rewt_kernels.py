"""Reproducible CPU1 execution comparisons for unchanged REWT weights."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import resource
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from rewt.data import file_sha256
from rewt.decode import EnergyDecodeSession
from rewt.position_memory import load_learning_checkpoint
from benchmark_rewt_memory import weight_hash


def compiler_cpu():
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return time.process_time() + usage.ru_utime + usage.ru_stime


def summarize(samples):
    return dict(cpu_seconds=[v[0] for v in samples], wall_seconds=[v[1] for v in samples],
                median_cpu_seconds=statistics.median(v[0] for v in samples),
                median_wall_seconds=statistics.median(v[1] for v in samples))


def forward(model, tokens, training):
    model.train(training)
    model.zero_grad(set_to_none=True)
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        out = model(tokens, trace_energy=False, state_gradient_interval=32)
        if training:
            out.logits.square().mean().backward()
    return out.logits.detach()


def alternating(functions, repeats):
    names = list(functions)
    samples = {name: [] for name in names}
    for repeat in range(repeats):
        # Rotate, then reverse alternate rounds to distribute ordering effects.
        order = names[repeat % len(names):] + names[:repeat % len(names)]
        if repeat % 2:
            order.reverse()
        for name in order:
            cpu, wall = time.process_time(), time.perf_counter()
            functions[name]()
            samples[name].append((time.process_time() - cpu, time.perf_counter() - wall))
    rows = {name: summarize(values) for name, values in samples.items()}
    baseline = rows['reference']['median_cpu_seconds']
    for row in rows.values():
        row['speedup_over_reference'] = baseline / row['median_cpu_seconds']
    return rows


@torch.inference_mode()
def decode(model, tokens, continuation=32, persistent=False):
    model.eval()
    session = EnergyDecodeSession(model) if persistent else None
    cpu, wall = time.process_time(), time.perf_counter()
    output = session.prefill(tokens) if session else model(tokens)
    prefill = (time.process_time() - cpu, time.perf_counter() - wall)
    generated = []
    cpu, wall = time.process_time(), time.perf_counter()
    for _ in range(continuation):
        token = output.logits[:, -1].argmax(-1)[:, None]
        generated.append(token)
        # Exactly continuation decode calls, including consumption of the last
        # emitted byte. This measures the decoder, not newline stopping/checkers.
        output = session.step(token) if session else model(token, output.state)
    return torch.cat(generated, 1), prefill, (time.process_time() - cpu, time.perf_counter() - wall)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--skip-compile', action='store_true')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    torch.set_num_threads(1)
    torch.manual_seed(22108)
    loaded, _ = load_learning_checkpoint(args.checkpoint)
    models = {}
    for name in ['reference', 'krylov2'] + ([] if args.skip_compile else ['compiled_krylov2']):
        model = type(loaded)(replace(loaded.config, cache_history=True,
                                    refinement_backend='reference' if name == 'reference' else 'krylov2'))
        model.load_state_dict(loaded.state_dict())
        if name == 'compiled_krylov2':
            model.compile_refinement()
        models[name] = model
    before = {name: weight_hash(model) for name, model in models.items()}
    record = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=file_sha256(args.checkpoint),
                  created=datetime.now(timezone.utc).isoformat(), torch=torch.__version__,
                  platform=platform.platform(), threads=torch.get_num_threads(), config=vars(loaded.config),
                  scope='CPU1 float32, preserved weights; training times include forward and backward but no optimizer/auxiliary losses. Compilation warmup excluded and recorded. No GPU claim.',
                  warmup=[], parity=[], workloads={}, decode={})
    # Full-precision and numerical correctness thresholds are fixed in tests;
    # this artifact also checks the actual trained checkpoint in float32.
    generator = torch.Generator().manual_seed(22109)
    for batch, length in [(1, 32), (24, 64), (24, 128)]:
        tokens = torch.randint(0, 256, (batch, length), generator=generator)
        for training in (False, True):
            case = f'b{batch}_t{length}_' + ('forward_backward' if training else 'forward')
            outputs, gradients = {}, {}
            for name, model in models.items():
                cpu, wall = compiler_cpu(), time.perf_counter()
                outputs[name] = forward(model, tokens, training)
                record['warmup'].append(dict(case=case, backend=name,
                                             wall_seconds=time.perf_counter() - wall,
                                             cpu_seconds_including_children=compiler_cpu() - cpu))
                if training:
                    gradients[name] = {key: p.grad.clone() for key, p in model.named_parameters()
                                       if p.grad is not None}
            for name in models:
                torch.testing.assert_close(outputs[name], outputs['reference'], rtol=3e-4, atol=3e-4)
                assert torch.equal(outputs[name].argmax(-1), outputs['reference'].argmax(-1))
                grad_max = 0.
                if training:
                    assert gradients[name].keys() == gradients['reference'].keys()
                    for key in gradients[name]:
                        a, b = gradients[name][key], gradients['reference'][key]
                        torch.testing.assert_close(a, b, rtol=3e-4, atol=3e-5, msg=key)
                        grad_max = max(grad_max, float((a - b).abs().max()))
                record['parity'].append(dict(case=case, backend=name, identical_argmax=True,
                                              max_logit_difference=float((outputs[name] - outputs['reference']).abs().max()),
                                              max_gradient_difference=grad_max if training else None))
            functions = {name: (lambda m=model: forward(m, tokens, training)) for name, model in models.items()}
            record['workloads'][case] = alternating(functions, args.repeats)
            print(json.dumps(dict(case=case, timings=record['workloads'][case])), flush=True)
    arms = {name: (model, False) for name, model in models.items()}
    arms.update({name + '_persistent': (model, True) for name, model in models.items()})
    for length in (16, 128):
        tokens = torch.randint(32, 127, (1, length), generator=generator)
        samples = {name: dict(prefill=[], continuation=[], total=[]) for name in arms}
        generated = {}
        for name, (model, persistent) in arms.items():
            cpu, wall = compiler_cpu(), time.perf_counter()
            generated[name], _, _ = decode(model, tokens, persistent=persistent)
            record['warmup'].append(dict(case=f'decode_prompt{length}', backend=name,
                                         wall_seconds=time.perf_counter()-wall,
                                         cpu_seconds_including_children=compiler_cpu()-cpu))
            assert torch.equal(generated[name], generated['reference'])
        names = list(arms)
        for repeat in range(args.repeats):
            order = names[repeat % len(names):] + names[:repeat % len(names)]
            if repeat % 2:
                order.reverse()
            for name in order:
                model, persistent = arms[name]
                result, prefill, continuation = decode(model, tokens, persistent=persistent)
                assert torch.equal(result, generated['reference'])
                samples[name]['prefill'].append(prefill)
                samples[name]['continuation'].append(continuation)
                samples[name]['total'].append(tuple(a+b for a, b in zip(prefill, continuation)))
        rows = {name: {phase: summarize(values) for phase, values in phases.items()} for name, phases in samples.items()}
        for name in rows:
            for phase in rows[name]:
                rows[name][phase]['speedup_over_reference'] = rows['reference'][phase]['median_cpu_seconds'] / rows[name][phase]['median_cpu_seconds']
        record['decode'][f'prompt{length}_continuation32'] = dict(timings=rows, identical_generated_bytes=True,
                                                                generated_hex=bytes(generated['reference'][0].tolist()).hex())
        print(json.dumps(dict(case=f'decode_prompt{length}', total_speedups={name: row['total']['speedup_over_reference'] for name, row in rows.items()})), flush=True)
    after = {name: weight_hash(model) for name, model in models.items()}
    assert before == after
    record.update(unchanged_weights=True, weight_hashes=before,
                  source_sha256={p.name: file_sha256(p) for p in (ROOT / 'rewt').glob('*.py')},
                  script_sha256=file_sha256(Path(__file__)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
