"""Time preserved-weight cached/uncached greedy generation on a supplied prompt."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from rewt.data import file_sha256
from rewt.model import JointEnergyLM
from rewt.position_memory import load_learning_checkpoint
from benchmark_rewt_memory import weight_hash


@torch.inference_mode()
def generate(model, raw, maximum):
    begin = time.process_time()
    output = model(torch.tensor([list(raw)], dtype=torch.long), trace_energy=False)
    prefill = time.process_time() - begin
    begin = time.process_time()
    generated = []
    for step in range(maximum):
        token = int(output.logits[0, -1].argmax())
        generated.append(token)
        if token == 10 or step + 1 == maximum:
            break
        output = model(torch.tensor([[token]], dtype=torch.long), state=output.state, trace_energy=False)
    continuation = time.process_time() - begin
    return dict(prefill_cpu_seconds=prefill, continuation_cpu_seconds=continuation,
                total_cpu_seconds=prefill+continuation, generated_hex=bytes(generated).hex(),
                generated_bytes=len(generated))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--prompt')
    source.add_argument('--prompt-file', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-new', type=int, default=40)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    raw = args.prompt_file.read_bytes() if args.prompt_file else args.prompt.encode('ascii')
    if not raw or min(args.max_new, args.repeats) < 1 or len(raw)+args.max_new > 128:
        parser.error('nonempty prompt, positive generation/repeats, and prompt+generation<=128 required')
    torch.set_num_threads(1)
    loaded, saved = load_learning_checkpoint(args.checkpoint)
    if not isinstance(loaded, JointEnergyLM) or loaded.config.support or saved.get('tokenizer'):
        parser.error('a deterministic raw-byte energy checkpoint is required')
    models = {name: type(loaded)(replace(loaded.config, cache_history=cached)).eval()
              for name, cached in (('uncached', False), ('cached', True))}
    for model in models.values():
        model.load_state_dict(loaded.state_dict(), strict=True)
    before = {name: weight_hash(model) for name, model in models.items()}
    checkpoint_before = file_sha256(args.checkpoint)
    measurements = {name: [] for name in models}
    for repetition in range(args.repeats+1):
        # Reverse the order each time; discard the first pair as warmup.
        for name in (list(models) if repetition % 2 == 0 else list(models)[::-1]):
            row = generate(models[name], raw, args.max_new)
            if repetition:
                measurements[name].append(row)
    generations = {r['generated_hex'] for rows in measurements.values() for r in rows}
    assert len(generations) == 1
    assert before == {name: weight_hash(model) for name, model in models.items()}
    assert checkpoint_before == file_sha256(args.checkpoint)
    phases = ('prefill_cpu_seconds', 'continuation_cpu_seconds', 'total_cpu_seconds')
    medians = {name: {phase: statistics.median(r[phase] for r in rows) for phase in phases}
               for name, rows in measurements.items()}
    result = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_before,
                  prompt_hex=raw.hex(), measurements=measurements, median_cpu_seconds=medians,
                  speedup={phase: medians['uncached'][phase]/medians['cached'][phase] for phase in phases},
                  identical_generation=True, unchanged_weights=True, unchanged_checkpoint=True,
                  weight_hashes=before,
                  script_sha256=file_sha256(Path(__file__)),
                  source_sha256={p.name: file_sha256(p) for p in (ROOT/'rewt').glob('*.py')},
                  scope=f'{args.repeats} alternating measured repeats after warmup, one CPU thread. Prefill processes the complete prompt; continuation begins from its logits and excludes prefill. Both policies recompute observed encodings at each decode call; no persistent inference KV cache or GPU measurement.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps(result['speedup']))


if __name__ == '__main__':
    main()
