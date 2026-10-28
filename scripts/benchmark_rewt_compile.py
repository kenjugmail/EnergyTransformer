"""Probe compilation of the analytic gradient without modifying stored weights."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from rewt.data import file_sha256
from rewt.position_memory import load_learning_checkpoint
from benchmark_rewt_memory import timed, weight_hash


def cpu_total():
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return time.process_time()+usage.ru_utime+usage.ru_stime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=24)
    parser.add_argument('--context', type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(1)
    loaded, _ = load_learning_checkpoint(args.checkpoint)
    models = {name: type(loaded)(replace(loaded.config, cache_history=True))
              for name in ('eager', 'compiled_gradient')}
    for model in models.values():
        model.load_state_dict(loaded.state_dict())
    tokens = torch.randint(0, 128, (args.batch, args.context), generator=torch.Generator().manual_seed(4261))
    before = {name: weight_hash(model) for name, model in models.items()}
    models['compiled_gradient'].gradient = torch.compile(models['compiled_gradient'].gradient,
                                                        fullgraph=True, dynamic=False)
    begin, cpu = time.perf_counter(), cpu_total()
    compiled = models['compiled_gradient']
    compiled.train()
    output = compiled(tokens, trace_energy=False, state_gradient_interval=32)
    output.logits.square().mean().backward()
    warmup_wall, warmup_cpu = time.perf_counter()-begin, cpu_total()-cpu
    results, grads = {}, {}
    for name, model in models.items():
        model.zero_grad(set_to_none=True)
        output = model.train()(tokens, trace_energy=False, state_gradient_interval=32)
        results[name] = output.logits.detach()
        output.logits.square().mean().backward()
        grads[name] = {k: p.grad.detach().clone() for k, p in model.named_parameters() if p.grad is not None}
    torch.testing.assert_close(results['eager'], results['compiled_gradient'], rtol=2e-5, atol=2e-5)
    assert torch.equal(results['eager'].argmax(-1), results['compiled_gradient'].argmax(-1))
    grad_difference = 0.
    for key in grads['eager']:
        a, b = grads['eager'][key], grads['compiled_gradient'][key]
        torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-5, msg=key)
        grad_difference = max(grad_difference, float((a-b).abs().max()))
    timings = {name: timed(model, tokens, True, 3) for name, model in models.items()}
    after = {name: weight_hash(model) for name, model in models.items()}
    assert before == after
    record = dict(created=datetime.now(timezone.utc).isoformat(), checkpoint=str(args.checkpoint.resolve()),
                  checkpoint_sha256=file_sha256(args.checkpoint), source_sha256={p.name: file_sha256(p) for p in (ROOT / 'rewt').glob('*.py')},
                  script_sha256=file_sha256(Path(__file__)), batch=args.batch, context=args.context,
                  maximum_logit_difference=float((results['eager']-results['compiled_gradient']).abs().max()),
                  maximum_parameter_gradient_difference=grad_difference, identical_argmax=True,
                  unchanged_weights=True, warmup_wall_seconds=warmup_wall,
                  warmup_cpu_seconds_including_child_compilers=warmup_cpu, timings=timings,
                  speedup=timings['eager']['median_cpu_seconds']/timings['compiled_gradient']['median_cpu_seconds'],
                  scope='Single-thread CPU forward/backward, compiled analytic gradient only. Compile warmup reported separately; no optimizer updates or change to the training experiment.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True)+'\n')
    print(json.dumps({key: record[key] for key in ('maximum_logit_difference', 'maximum_parameter_gradient_difference', 'speedup', 'warmup_wall_seconds')}))


if __name__ == '__main__':
    main()
