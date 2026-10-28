"""Exploratory CPU compiler check, with both Transformer controls included."""
from dataclasses import replace
import copy
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F
from energy_real.bench import balanced_order
from energy_real.competitive_study import configuration
from energy_real.local_energy_study import Budget
from energy_real.model import LM
from energy_real.runtime import atomic_json, source_identity


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = Path(__file__).with_name('compile_probe.json')
    if output.exists():
        raise FileExistsError(output)
    budget = Budget(ROOT / 'runs/local_energy_budget.json', 1800,
                    ROOT / 'reports/publication/device_stages.json')
    status = 'failed'
    rows = []
    try:
        for arm in ('softmax', 'local_softmax', 'hybrid'):
            budget.check()
            torch.manual_seed(927)
            model = LM(replace(configuration(arm), chunk=8)).train()
            x, y = torch.randint(256, (8,128)), torch.randint(256, (8,128))
            def objective(x, y):
                return F.cross_entropy(model(x)[0].flatten(0,1), y.flatten())
            eager = objective(x, y)
            eager.backward()
            gradients = [p.grad.clone() for p in model.parameters()]
            model.zero_grad(set_to_none=True)
            compiled = torch.compile(objective, fullgraph=True)
            begin = time.perf_counter()
            value = compiled(x, y)
            value.backward()
            compile_wall = time.perf_counter()-begin
            torch.testing.assert_close(value, eager, atol=2e-5, rtol=1e-5)
            for p, expected in zip(model.parameters(), gradients):
                torch.testing.assert_close(p.grad, expected, atol=3e-5, rtol=1e-3)
            timings = {'eager': [], 'compiled': []}
            for turn in range(6):
                for name in balanced_order(list(timings), turn):
                    function = compiled if name == 'compiled' else objective
                    for step in range(12):
                        budget.check()
                        model.zero_grad(set_to_none=True)
                        begin = time.process_time()
                        function(x, y).backward()
                        if step >= 2:
                            timings[name].append(time.process_time()-begin)
            rows.append({'arm':arm,'compile_and_first_backward_wall_s':compile_wall,
                         'median_forward_backward_cpu_s':{k:statistics.median(v) for k,v in timings.items()},
                         'gradient_parity':True,'rows':timings})
            atomic_json({'source_sha256':source_identity(),'rows':rows},output)
            print(json.dumps({k:v for k,v in rows[-1].items() if k != 'rows'}),flush=True)
        status = 'completed'
    finally:
        budget.finish(status)


if __name__ == '__main__':
    main()
