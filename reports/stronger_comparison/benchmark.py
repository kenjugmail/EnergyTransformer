"""Benchmark old/new kernels and a preallocated Transformer baseline.

Run from the repository root with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1 .venv/bin/python reports/stronger_comparison/benchmark.py
"""
from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
import statistics
import sys
import tarfile
import time
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F
import energy_real.model as model_module
from energy_real.bench import balanced_order, tensor_bytes
from energy_real.local_energy_study import Budget
from energy_real.model import Config, LM
from energy_real.runtime import atomic_json, environment, sha256, source_identity
from energy_real.static_cache import allocated_state_bytes, with_static_cache

OUT = Path(__file__).resolve().parent
with tarfile.open(ROOT / 'runs/stronger_preservation/before.tar.gz') as archive:
    old_source = archive.extractfile('energy_real/kernel.py').read()
old_kernel = types.ModuleType('preserved_kernel')
exec(compile(old_source, 'preserved_kernel.py', 'exec'), old_kernel.__dict__)
new_functions = model_module.chunk_gla, model_module.read_again


@contextmanager
def kernel(old=False):
    model_module.chunk_gla, model_module.read_again = (
        (old_kernel.chunk_gla, old_kernel.read_again) if old else new_functions)
    try:
        yield
    finally:
        model_module.chunk_gla, model_module.read_again = new_functions


def load(arm, chunk=None):
    saved = torch.load(ROOT / f'runs/competitive_native/selected_{arm}.pt', weights_only=True)
    config = Config(**saved['config'])
    model = LM(config if chunk is None else replace(config, chunk=chunk)).eval()
    model.load_state_dict(saved['weights'])
    return model


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    budget = Budget(ROOT / 'runs/local_energy_budget.json', 1200,
                    ROOT / 'reports/publication/device_stages.json')
    status = 'failed'
    try:
        torch.manual_seed(92011)
        x, y = torch.randint(256, (2, 128)), torch.randint(256, (2, 128))
        parity = []
        for arm in ('energy', 'hybrid'):
            model = load(arm)
            outputs, gradients = [], []
            for old in (True, False):
                model.zero_grad(set_to_none=True)
                with kernel(old):
                    logits, _ = model(x)
                    F.cross_entropy(logits.flatten(0, 1), y.flatten()).backward()
                outputs.append(logits.detach())
                gradients.append(torch.cat([p.grad.flatten() for p in model.parameters()]))
            torch.testing.assert_close(outputs[0].softmax(-1), outputs[1].softmax(-1), atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(gradients[0], gradients[1], atol=3e-5, rtol=1e-3)
            parity.append({'arm': arm, 'max_probability_error': float((outputs[0].softmax(-1)-outputs[1].softmax(-1)).abs().max()),
                           'max_gradient_error': float((gradients[0]-gradients[1]).abs().max())})
        train_rows = []
        x, y = torch.randint(256, (8, 128)), torch.randint(256, (8, 128))
        labels = ('old_chunk32', 'new_chunk32', 'new_chunk8')
        for turn in range(6):
            for label in balanced_order(list(labels), turn):
                budget.check()
                host_load = os.getloadavg()[0]
                model = load('hybrid', 8 if label.endswith('8') else 32).train()
                optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, foreach=False)
                with kernel(label.startswith('old')):
                    for step in range(20):
                        start_cpu, start = time.process_time(), time.perf_counter()
                        optimizer.zero_grad(set_to_none=True)
                        logits, _ = model(x)
                        F.cross_entropy(logits.flatten(0, 1), y.flatten()).backward()
                        optimizer.step()
                        wall, cpu = time.perf_counter()-start, time.process_time()-start_cpu
                        if step >= 5:
                            train_rows.append({'variant': label, 'round': turn, 'step': step,
                                               'wall_s': wall, 'cpu_s': cpu, 'host_load_1m': host_load})
        atomic_json(train_rows, OUT / 'kernel_training_rows.json')
        profiles = []
        labels = ('full_dynamic', 'full_static', 'hybrid_old_kernel', 'hybrid', 'local_softmax', 'energy')
        for length in (128, 4096, 32768):
            tokens = torch.randint(256, (1, length+16))
            models = {name: load('softmax' if name.startswith('full') else 'hybrid' if name.startswith('hybrid') else name)
                      for name in labels}
            models['full_static'] = with_static_cache(models['full_static'], length+16)
            for turn in range(6):
                for name in balanced_order(list(labels), turn):
                    budget.check()
                    host_load = os.getloadavg()[0]
                    model = models[name]
                    with torch.no_grad(), kernel(name == 'hybrid_old_kernel'):
                        model(tokens[:, :64])
                        state = None
                        start_cpu, start = time.process_time(), time.perf_counter()
                        for lo in range(0, length, 256):
                            logits, state = model(tokens[:, lo:min(length, lo+256)], state)
                        prefill_s, prefill_cpu = time.perf_counter()-start, time.process_time()-start_cpu
                        start_cpu, start = time.process_time(), time.perf_counter()
                        for position in range(length, length+16):
                            logits, state = model(tokens[:, position:position+1], state)
                        decode_s, decode_cpu = time.perf_counter()-start, time.process_time()-start_cpu
                        if name == 'full_static' and turn == 0:
                            # Each arm consumes exactly the same teacher-forced sequence.
                            with torch.no_grad():
                                reference_state = None
                                for lo in range(0, length+16, 256):
                                    reference, reference_state = models['full_dynamic'](tokens[:, lo:lo+256], reference_state)
                            torch.testing.assert_close(logits[:, -1], reference[:, -1], atol=2e-4, rtol=1e-4)
                    profiles.append({'arm': name, 'context': length, 'round': turn,
                                     'host_load_1m': host_load,
                                     'prefill_s': prefill_s, 'prefill_cpu_s': prefill_cpu,
                                     'decode_ms': decode_s*1000/16, 'decode_cpu_ms': decode_cpu*1000/16,
                                     'used_state_bytes': tensor_bytes(state),
                                     'allocated_state_bytes': allocated_state_bytes(model, state)})
                atomic_json(profiles, OUT / 'inference_rows.json')
            print(json.dumps({'profile_context_completed': length}), flush=True)
        summary = {'source_sha256': source_identity(), 'environment': environment('cpu'),
                   'cpu_affinity': sorted(os.sched_getaffinity(0)),
                   'host_load_1m_range': [min(r['host_load_1m'] for r in profiles), max(r['host_load_1m'] for r in profiles)],
                   'kernel_reference_archive_sha256': sha256(ROOT / 'runs/stronger_preservation/before.tar.gz'),
                   'parity': parity, 'training': {}, 'inference': {},
                   'scope': 'One CPU thread including BLAS; float32; old selected checkpoints; 256-byte prefill chunks; 16 teacher-forced decode tokens; six balanced rounds. Static cache allocation counted in full.'}
        for label in ('old_chunk32', 'new_chunk32', 'new_chunk8'):
            summary['training'][label] = {key: statistics.median(r[key] for r in train_rows if r['variant'] == label)
                                          for key in ('wall_s', 'cpu_s')}
        for length in (128, 4096, 32768):
            summary['inference'][length] = {name: {key: statistics.median(r[key] for r in profiles if r['arm'] == name and r['context'] == length)
                for key in ('prefill_s', 'decode_ms', 'decode_cpu_ms', 'used_state_bytes', 'allocated_state_bytes')}
                for name in labels}
            for name in labels:
                values = [r['decode_ms'] for r in profiles if r['arm'] == name and r['context'] == length]
                summary['inference'][length][name]['decode_ms_range'] = [min(values), max(values)]
        atomic_json(summary, OUT / 'benchmark.json')
        print(json.dumps(summary, indent=2), flush=True)
        status = 'completed'
    finally:
        budget.finish(status)


if __name__ == '__main__':
    main()
