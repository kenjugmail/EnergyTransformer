"""Audit optimized execution against a frozen reference and validation prompts."""
import argparse
from collections import defaultdict
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from rewt.data import file_sha256
from rewt.decode import EnergyDecodeSession
from rewt.position_memory import load_learning_checkpoint
from benchmark_rewt_memory import weight_hash


def original_loader(directory):
    name = 'rewt_kernel_frozen_reference'
    spec = importlib.util.spec_from_file_location(name, directory / '__init__.py',
                                                 submodule_search_locations=[str(directory)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    return importlib.import_module(name + '.position_memory').load_learning_checkpoint


@torch.inference_mode()
def generate(model, prompts, persistent):
    model.eval()
    tokens = torch.tensor([list(p.encode('ascii')) for p in prompts])
    maximum = min(40, 128 - tokens.shape[1])
    session = EnergyDecodeSession(model) if persistent else None
    output = session.prefill(tokens) if session else model(tokens)
    rows, done = [[] for _ in prompts], [False] * len(prompts)
    for step in range(maximum):
        token = output.logits[:, -1].argmax(-1)
        for i, value in enumerate(token.tolist()):
            if not done[i]:
                rows[i].append(value)
                done[i] = value == 10
        if all(done) or step + 1 == maximum:
            break
        output = session.step(token[:, None]) if session else model(token[:, None], output.state)
    return [bytes(row) for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--reference-source', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    loader = original_loader(args.reference_source)
    legacy = json.loads((ROOT/'reports/rewt_polish_20261010_v2/legacy_parity.json').read_text())
    archive_rows = []
    checkpoints = [Path(row['checkpoint']) for row in legacy['rows']]
    checkpoints += [args.checkpoint, ROOT/'runs/rewt_learning_20261010/binding/aligned/best.pt']
    for path in checkpoints:
        old, _ = loader(path)
        new, _ = load_learning_checkpoint(path)
        old.eval(); new.eval()
        tokens = torch.randint(0, 256, (2, 24), generator=torch.Generator().manual_seed(22111))
        with torch.inference_mode():
            a, b = old(tokens), new(tokens)
        assert torch.equal(a.logits, b.logits)
        assert torch.equal(a.state.joint, b.state.joint)
        archive_rows.append(dict(checkpoint=str(path.resolve()), sha256=file_sha256(path), bitwise_equal=True))
    baseline, _ = loader(args.checkpoint)
    current, _ = load_learning_checkpoint(args.checkpoint)
    optimized = type(current)(replace(current.config, refinement_backend='krylov2'))
    optimized.load_state_dict(current.state_dict())
    hashes = {name: weight_hash(m) for name, m in [('baseline', baseline), ('optimized', optimized)]}
    tasks = [json.loads(line) for line in args.validation.read_text().splitlines()]
    assert all(row['split'] == 'valid' for row in tasks)
    grouped = defaultdict(list)
    for row in tasks:
        grouped[len(row['prompt'])].append(row)
    records = []
    for length, group in sorted(grouped.items()):
        for start in range(0, len(group), 8):
            selected = group[start:start + 8]
            prompts = [row['prompt'] for row in selected]
            a, b = generate(baseline, prompts, False), generate(optimized, prompts, True)
            for task, x, y in zip(selected, a, b):
                records.append(dict(id=task['id'], prompt=task['prompt'],
                                    reference_hex=x.hex(), optimized_hex=y.hex(), identical=x == y))
    assert all(row['identical'] for row in records)
    assert hashes == {name: weight_hash(m) for name, m in [('baseline', baseline), ('optimized', optimized)]}
    output = dict(archived_checkpoint_parity=archive_rows,
                  validation_file=str(args.validation.resolve()), validation_sha256=file_sha256(args.validation),
                  prompts_checked=len(records), generated_bytes_checked=sum(len(bytes.fromhex(r['reference_hex'])) for r in records),
                  identical_generation=True, unchanged_weights=True, weight_hashes=hashes, records=records,
                  reference_source_sha256={p.name: file_sha256(p) for p in args.reference_source.glob('*.py')},
                  source_sha256={p.name: file_sha256(p) for p in (ROOT/'rewt').glob('*.py')},
                  script_sha256=file_sha256(Path(__file__)),
                  scope='Numerical execution parity on previously available validation prompts, not a new task-quality evaluation. No training or closed-test model selection.')
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({k: output[k] for k in ('prompts_checked', 'generated_bytes_checked', 'identical_generation', 'unchanged_weights')}))


if __name__ == '__main__':
    main()
