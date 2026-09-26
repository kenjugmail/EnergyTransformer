"""Compare compiled/eager gradients on real crop views and trained weights."""
from pathlib import Path
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from energy_real.compute_study import objective_for
from energy_real.data import Enwik8
from energy_real.model import Config, LM
from energy_real.runtime import atomic_json, sha256, source_identity


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    x, y = Enwik8(ROOT/'data/enwik8','train').batch(np.random.default_rng(92109),8,128)
    assert x.stride() == y.stride() == (129,1)
    assert x.untyped_storage().data_ptr() == y.untyped_storage().data_ptr()
    fits = json.loads((ROOT/'runs/compute_confirmation_v2/main_404_fits.json').read_text())
    rows = []
    for fit in fits:
        path = ROOT/fit['checkpoint']
        assert sha256(path) == fit['checkpoint_sha256']
        saved = torch.load(path,weights_only=True)
        model = LM(Config(**saved['config'])).train()
        model.load_state_dict(saved['weights'])
        objective = objective_for(model)
        expected = objective(x,y)
        expected.backward()
        gradients = [p.grad.clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        begin = time.perf_counter()
        actual = torch.compile(objective,fullgraph=True)(x,y)
        actual.backward()
        elapsed = time.perf_counter()-begin
        torch.testing.assert_close(actual,expected,atol=2e-5,rtol=1e-5)
        for p,want in zip(model.parameters(),gradients):
            torch.testing.assert_close(p.grad,want,atol=3e-5,rtol=1e-3)
        rows.append({'arm':fit['arm'],'checkpoint_sha256':fit['checkpoint_sha256'],
                     'loss_error':abs(float(actual.detach()-expected.detach())),
                     'max_gradient_error':max(float((p.grad-want).abs().max()) for p,want in zip(model.parameters(),gradients)),
                     'compile_and_check_wall_s':elapsed})
    result = {'source_sha256':source_identity(),'data':'Training crops only, seed 92109',
              'input_stride':list(x.stride()),'target_stride':list(y.stride()),
              'overlapping_input_target_storage':True,'rows':rows}
    atomic_json(result,Path(__file__).with_name('compiled_verification.json'))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
