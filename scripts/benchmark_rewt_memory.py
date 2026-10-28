"""Compare cached and uncached execution of preserved REWT weights."""
from dataclasses import replace
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from rewt.data import file_sha256
from rewt.evaluate import generate_batch
from rewt.model import JointEnergyLM
from rewt.position_memory import load_learning_checkpoint


def weight_hash(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def timed(model, tokens, training, repeats):
    times = []
    for repeat in range(repeats + 1):
        model.train(training)
        model.zero_grad(set_to_none=True)
        begin = time.process_time()
        if training:
            output = model(tokens, trace_energy=False, state_gradient_interval=64)
            output.logits.square().mean().backward()
        else:
            with torch.inference_mode():
                model(tokens, trace_energy=False)
        if repeat:
            times.append(time.process_time() - begin)
    return dict(cpu_seconds=times, median_cpu_seconds=statistics.median(times))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=24)
    parser.add_argument('--context', type=int, default=64)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if min(args.batch, args.context, args.repeats) < 1:
        parser.error('positive batch, context and repeat counts required')
    torch.set_num_threads(1)
    loaded, _ = load_learning_checkpoint(args.checkpoint)
    if not isinstance(loaded, JointEnergyLM):
        parser.error('an energy checkpoint is required')
    models = {name: type(loaded)(replace(loaded.config, cache_history=cache))
              for name, cache in [('uncached', False), ('cached', True)]}
    for model in models.values():
        model.load_state_dict(loaded.state_dict())
    tokens = torch.randint(0, 128, (args.batch, args.context),
                           generator=torch.Generator().manual_seed(4201))
    before = {name: weight_hash(model) for name, model in models.items()}
    with torch.inference_mode():
        results = {name: model.eval()(tokens, details=True) for name, model in models.items()}
    a, b = (results[name] for name in ('uncached', 'cached'))
    maximum_difference = float((a.logits - b.logits).abs().max())
    torch.testing.assert_close(a.logits, b.logits, rtol=2e-5, atol=2e-5)
    assert torch.equal(a.logits.argmax(-1), b.logits.argmax(-1))
    assert a.details['cost'] == b.details['cost']
    timings = {phase: {name: timed(model, tokens, phase == 'forward_backward', args.repeats)
                       for name, model in models.items()}
               for phase in ('forward', 'forward_backward')}
    after = {name: weight_hash(model) for name, model in models.items()}
    assert before == after
    record = dict(created=datetime.now(timezone.utc).isoformat(),
                  checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=file_sha256(args.checkpoint),
                  config=vars(loaded.config), batch=args.batch, context=args.context,
                  repeats=args.repeats, maximum_logit_difference=maximum_difference,
                  identical_argmax=True, identical_energy_call_counts=True,
                  unchanged_weights=before == after, weight_hashes=before, timings=timings,
                  speedup={phase: row['uncached']['median_cpu_seconds'] / row['cached']['median_cpu_seconds']
                           for phase, row in timings.items()},
                  script_sha256=file_sha256(Path(__file__)),
                  source_sha256={p.name: file_sha256(p) for p in (ROOT / 'rewt').glob('*.py')},
                  scope='CPU single-thread preserved-weight forward/backward; no optimizer updates, auxiliary losses or GPU speed claim.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: record[key] for key in ('maximum_logit_difference', 'speedup', 'unchanged_weights')}))


if __name__ == '__main__':
    main()
