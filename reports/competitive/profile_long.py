"""Execute the predeclared long-context profile after frozen primary evaluation."""
from pathlib import Path
import json
import time

import torch

from energy_real.bench import balanced_order, tensor_bytes
from energy_real.local_energy_study import Budget
from energy_real.model import Config, LM
from energy_real.runtime import atomic_json, sha256, source_identity


def main():
    report = Path(__file__).resolve().parent
    root = report.parents[1]
    run = root / 'runs/competitive_native'
    out = run / 'long_context'
    if out.exists(): raise FileExistsError(out)
    plan = json.loads((report / 'long_context_plan.json').read_text())
    selected = json.loads((run / 'selection.json').read_text())
    primary = json.loads((run / 'results.json').read_text())
    assert primary['status'] == 'completed'
    assert source_identity() == selected['source_sha256'] == primary['source_sha256']
    assert primary['selection_sha256'] == sha256(run / 'selection.json')
    torch.set_num_threads(plan['threads'])
    models = {}
    for arm in plan['arms']:
        fit = next(f for f in selected['fits'] if f['arm']==arm and f['seed']==selected['selected_seeds'][arm])
        path = root / fit['checkpoint']
        assert sha256(path) == fit['checkpoint_sha256']
        saved = torch.load(path, weights_only=True, map_location='cpu')
        models[arm] = LM(Config(**saved['config'])).eval()
        models[arm].load_state_dict(saved['weights'])
    budget = Budget(root/'runs/local_energy_budget.json', plan['max_seconds'], root/'reports/publication/device_stages.json')
    status = 'failed'
    rows = []
    try:
        out.mkdir()
        atomic_json({'plan_sha256': sha256(report/'long_context_plan.json'),
                     'selection_sha256': sha256(run/'selection.json'), 'source_sha256': source_identity()}, out/'identity.json')
        generator = torch.Generator().manual_seed(plan['seed'])
        with torch.no_grad():
            for length in plan['contexts']:
                tokens = torch.randint(256, (1, length + plan['decode_tokens']), generator=generator)
                for turn in range(plan['rounds']):
                    for arm in balanced_order(plan['arms'], turn):
                        budget.check()
                        model = models[arm]
                        model(tokens[:, :64])
                        start = time.perf_counter()
                        _, state = model(tokens[:, :length])
                        prefill_s = time.perf_counter()-start
                        before = tensor_bytes(state)
                        start = time.perf_counter()
                        for i in range(length, length + plan['decode_tokens']):
                            _, state = model(tokens[:, i:i+1], state)
                        decode_s = time.perf_counter()-start
                        rows.append({'arm':arm, 'context':length, 'round':turn,
                                     'prefill_s':prefill_s, 'decode_ms':decode_s*1000/plan['decode_tokens'],
                                     'state_bytes':tensor_bytes(state), 'prefill_state_bytes':before,
                                     'parameter_bytes':sum(p.numel()*p.element_size() for p in model.parameters())})
                        atomic_json(rows, out/'rows.json')
                print(json.dumps({'profile_context_completed':length}), flush=True)
        status='completed'
        atomic_json({'status':status, 'rows':len(rows)}, out/'result.json')
    finally:
        budget.finish(status)


if __name__ == '__main__': main()
