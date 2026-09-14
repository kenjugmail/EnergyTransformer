"""Evaluate frozen checkpoints with explicit split and window protocol."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
import torch
from .data import Enwik8, LOCAL_CORPUS_SHA256
from . import data as dataset
from .model import LM, Config
from .train import evaluate
from .runtime import atomic_json, sha256, source_identity, environment


def run(args):
    Enwik8._validate_window(args.seq, args.batch)
    if args.max_bytes is not None and (type(args.max_bytes) is not int or args.max_bytes < 2):
        raise ValueError('max_bytes must contain at least two bytes')
    for name in ('max_seconds', 'max_cuda_gib'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if not args.checkpoints or len(set(map(str, args.checkpoints))) != len(args.checkpoints):
        raise ValueError('Provide distinct checkpoints')
    out = Path(args.out)
    if out.exists():
        raise FileExistsError('Use a new evaluation output; retain earlier measurements')
    if args.freeze_selection and out.with_suffix('.selection.json').exists():
        raise FileExistsError(out.with_suffix('.selection.json'))
    device = torch.device(args.device)
    if device.type == 'cuda':
        from .device import require_idle_gpu, set_memory_limit
        require_idle_gpu()
        set_memory_limit(device, args.max_cuda_gib)
    corpus = Enwik8(args.data, args.split, expected_sha256=args.data_sha256)
    hashes = {str(Path(p)): sha256(p) for p in args.checkpoints}
    protocol = {'context_bytes': args.seq, 'backend': args.backend, 'amp': args.amp,
                'state_policy': 'reset_each_window', 'data_sha256': corpus.file_identity(),
                'max_bytes': args.max_bytes, 'batch': args.batch,
                'split_boundaries': {key:list(value) for key,value in dataset.SPLITS.items()},
                'device_type': device.type, 'source_sha256': source_identity(),
                'torch': str(torch.__version__), 'matmul_precision': torch.get_float32_matmul_precision()}
    if args.split == 'test':
        if args.selection is None:
            raise ValueError('Test evaluation requires a frozen validation-selection manifest')
        selected = json.loads(Path(args.selection).read_text())
        validation = Path(args.selection).parent / selected.get('validation_report', '')
        if not validation.is_file() or sha256(validation) != selected.get('validation_report_sha256'):
            raise ValueError('Frozen validation report is missing or has changed')
        if selected['checkpoint_sha256'] != hashes or selected['protocol'] != protocol:
            raise ValueError('Checkpoint or evaluation protocol differs from frozen selection')
    if args.freeze_selection and args.split != 'valid':
        raise ValueError('Freeze selection using validation only')
    records = []
    started = time.perf_counter()
    for path in args.checkpoints:
        if time.perf_counter() - started >= args.max_seconds:
            records.append({'checkpoint': str(path), 'status': 'budget_exhausted'})
            continue
        saved = torch.load(path, map_location='cpu', weights_only=True)
        config = Config(**{**saved['config'], 'backend': args.backend})
        model = LM(config).to(device)
        model.load_state_dict(saved['weights'], strict=True)
        t0 = time.perf_counter()
        result = evaluate(model, corpus, args.seq, args.max_bytes, device, args.batch,
                          amp=args.amp == 'bfloat16', deadline=started+args.max_seconds)
        record = {'status': 'completed', 'checkpoint': str(path), 'checkpoint_sha256': hashes[str(Path(path))],
                  'config': asdict(config), 'checkpoint_step': saved.get('step'),
                  'training_tokens': saved.get('tokens'), 'planned_targets': (len(corpus) if args.max_bytes is None else min(len(corpus), args.max_bytes))-1, 'parameters': model.parameters_count(),
                  'seconds': time.perf_counter() - t0, **result}
        records.append(record)
        del model, saved
        if device.type == 'cuda': torch.cuda.empty_cache()
        atomic_json({'protocol': protocol, 'environment': environment(device), 'source_sha256': source_identity(),
                     'results': records, 'status': 'in_progress'}, out)
    result = {'protocol': protocol, 'environment': environment(device), 'source_sha256': source_identity(),
              'results': records, 'status': 'completed' if all(r['status']=='completed' for r in records) else 'incomplete',
              'comparison': 'Descriptive when checkpoints consumed different training budgets'}
    atomic_json(result, out)
    if args.freeze_selection:
        if result['status'] != 'completed': raise ValueError('Cannot freeze an incomplete validation evaluation')
        selection_path = out.with_suffix('.selection.json')
        if selection_path.exists(): raise FileExistsError(selection_path)
        atomic_json({'checkpoint_sha256': hashes, 'protocol': protocol, 'validation_report': out.name, 'validation_report_sha256': sha256(out)}, selection_path)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoints', nargs='+', type=Path)
    p.add_argument('--data', default='data/enwik8')
    p.add_argument('--data-sha256', default=LOCAL_CORPUS_SHA256)
    p.add_argument('--split', choices=['valid','test'], default='valid')
    p.add_argument('--seq', type=int, default=512)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--max-bytes', type=int)
    p.add_argument('--backend', choices=['dmat','matmul'], default='dmat')
    p.add_argument('--amp', choices=['none','bfloat16'], default='bfloat16')
    p.add_argument('--device', default='cuda')
    p.add_argument('--max-seconds', type=float, default=1800)
    p.add_argument('--max-cuda-gib', type=float, default=48)
    p.add_argument('--selection', type=Path)
    p.add_argument('--freeze-selection', action='store_true')
    p.add_argument('--out', required=True, type=Path)
    print(json.dumps(run(p.parse_args()), indent=2))


if __name__ == '__main__': main()
