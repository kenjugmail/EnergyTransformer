"""Small CUDA correctness check; deliberately makes no throughput claim."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from rewt.data import file_sha256
from rewt.kernels import krylov2_refine, prepare_krylov2
from rewt.position_memory import load_learning_checkpoint
from benchmark_rewt_memory import weight_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    loaded, _ = load_learning_checkpoint(args.checkpoint)
    models = {name: type(loaded)(replace(loaded.config, refinement_backend='reference')).cuda()
              for name in ('reference', 'compiled')}
    for model in models.values():
        model.load_state_dict(loaded.state_dict())
    before = {name: weight_hash(model) for name, model in models.items()}
    fn = torch.compile(krylov2_refine, fullgraph=True, dynamic=False)
    results, gradients = {}, {}
    for name, model in models.items():
        constants = model.frequencies(), .15 + 1.85 * model.raw_precision.sigmoid()
        state = model.initial_state(4)
        cond, _ = model.condition(torch.tensor([65, 66, 67, 68], device='cuda'), state, constants=constants)
        if name == 'reference':
            result, _, _ = model.refine(cond.anchor, cond)
        else:
            result, _, _ = fn(cond.anchor, cond.amplitudes, model.phase, constants[0],
                              *prepare_krylov2(model, *constants),
                              output_start=model.blocks['output'].start, centered=True)
        results[name] = result.detach().cpu()
        result.square().mean().backward()
        gradients[name] = {key: p.grad.detach().cpu() for key, p in model.named_parameters() if p.grad is not None}
    torch.cuda.synchronize()
    torch.testing.assert_close(results['reference'], results['compiled'], rtol=3e-4, atol=3e-4)
    assert gradients['reference'].keys() == gradients['compiled'].keys()
    error = 0.
    for key, value in gradients['reference'].items():
        torch.testing.assert_close(value, gradients['compiled'][key], rtol=3e-4, atol=3e-5, msg=key)
        error = max(error, float((value-gradients['compiled'][key]).abs().max()))
    after = {name: weight_hash(model) for name, model in models.items()}
    assert before == after
    record = dict(device=torch.cuda.get_device_name(), torch=torch.__version__, batch=4,
                  max_state_difference=float((results['reference']-results['compiled']).abs().max()),
                  max_parameter_gradient_difference=error,
                  identical_argmax=torch.equal(results['reference'][:, model.blocks['output']].argmax(-1),
                                               results['compiled'][:, model.blocks['output']].argmax(-1)),
                  unchanged_weights=True, weight_hashes=before, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  checkpoint_sha256=file_sha256(args.checkpoint), script_sha256=file_sha256(Path(__file__)),
                  source_sha256={p.name:file_sha256(p) for p in (ROOT/'rewt').glob('*.py')},
                  scope='One four-row conditional refinement forward/backward, compiled fullgraph on CUDA. GPU was shared and busy; this is correctness coverage, not a speed benchmark.')
    assert record['identical_argmax']
    args.output.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
