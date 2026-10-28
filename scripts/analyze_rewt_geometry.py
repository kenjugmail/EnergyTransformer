"""Measure the core energy's curvature and exact reduced optimization spaces."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from rewt.data import file_sha256
from rewt.kernels import core_quadratic_action
from rewt.position_memory import load_learning_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, _ = load_learning_checkpoint(args.checkpoint)
    model.double().eval()
    assert not model.config.support
    assert model.config.compatibility_geometry == model.config.wave_geometry == 'anchor_relative'
    assert model.config.output_geometry == 'centered'
    with torch.inference_mode():
        start = time.process_time()
        frequency = model.frequencies()
        precision = .15 + 1.85 * model.raw_precision.sigmoid()
        hessian = core_quadratic_action(model, torch.eye(model.dimensions, dtype=torch.float64), precision)
        eigen = torch.linalg.eigvalsh(hessian)
        gram = frequency @ frequency.T
        spectrum = torch.linalg.eigvalsh(gram)
        amplitude_bound = .5 / math.sqrt(model.config.modes)
        wave_bound = amplitude_bound * frequency.T @ frequency
        upper = torch.linalg.eigvalsh(hessian + wave_bound)
        lower = torch.linalg.eigvalsh(hessian - wave_bound)
        gauge = torch.zeros(model.dimensions, dtype=torch.float64)
        gauge[model.blocks['output']] = 1 / math.sqrt(256)
        # The output common-offset gauge is null. Add its unit projector only
        # for the solve; all modes and meaningful displacements are orthogonal.
        inverse_basis = torch.linalg.solve(hessian + gauge[:, None] * gauge[None, :], frequency.T).T
        inverse_gram = frequency @ inverse_basis.T
        inverse_spectrum = torch.linalg.eigvalsh(inverse_gram)
        construction_cpu = time.process_time() - start
        # Demonstrate the preconditioned-subspace identity, not task accuracy.
        generator = torch.Generator().manual_seed(22110)
        amplitudes = torch.rand(8, model.config.modes, generator=generator, dtype=torch.float64) * (2 * amplitude_bound) - amplitude_bound
        coordinate = torch.zeros_like(amplitudes)
        displacement = torch.zeros(8, model.dimensions, dtype=torch.float64)
        max_state_error = max_energy_error = 0.
        eta = .25
        def energy(z):
            return .5 * ((z @ hessian) * z).sum(-1) + (amplitudes * (z @ frequency.T + model.phase).cos()).sum(-1)
        energy_path = [energy(displacement)]
        for _ in range(12):
            wave_coefficient = -amplitudes * (displacement @ frequency.T + model.phase).sin()
            displacement = displacement - eta * (displacement + wave_coefficient @ inverse_basis)
            reduced_wave = -amplitudes * (coordinate @ inverse_gram + model.phase).sin()
            coordinate = coordinate - eta * (coordinate + reduced_wave)
            lifted = coordinate @ inverse_basis
            reduced_energy = .5 * ((coordinate @ inverse_gram) * coordinate).sum(-1)
            reduced_energy += (amplitudes * (coordinate @ inverse_gram + model.phase).cos()).sum(-1)
            max_state_error = max(max_state_error, float((lifted - displacement).abs().max()))
            max_energy_error = max(max_energy_error, float((energy(lifted) - reduced_energy).abs().max()))
            energy_path.append(energy(displacement))
        energy_path = torch.stack(energy_path, -1)
        assert max_state_error < 1e-11 and max_energy_error < 1e-11
        assert (energy_path[:, 1:] <= energy_path[:, :-1] + 1e-12).all()
    record = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=file_sha256(args.checkpoint),
                  dimensions=model.dimensions, modes=model.config.modes,
                  quadratic_eigenvalues=eigen.tolist(), frequency_gram_eigenvalues=spectrum.tolist(),
                  frequency_rank=int(torch.linalg.matrix_rank(frequency)),
                  two_round_span_rank=int(torch.linalg.matrix_rank(torch.cat((frequency, frequency @ hessian)))),
                  frequency_stable_rank=float(spectrum.sum()/spectrum[-1]),
                  amplitude_absolute_bound=amplitude_bound,
                  global_curvature_upper=float(upper[-1]), global_curvature_lower=float(lower[0]),
                  original_step_size=model.config.step_size,
                  strict_descent_step_upper_bound=float(2 / upper[-1]),
                  geometry_preparation_cpu_seconds=construction_cpu,
                  preconditioned_curvature_upper=float(1 + amplitude_bound * inverse_spectrum[-1]),
                  preconditioned_strict_descent_step_upper_bound=float(2/(1 + amplitude_bound * inverse_spectrum[-1])),
                  preconditioned_identity_check=dict(rounds=12, batch=8,
                      maximum_state_error=max_state_error, maximum_energy_error=max_energy_error,
                      energy_nondecreasing_violations=int((energy_path[:, 1:] > energy_path[:, :-1]+1e-12).sum()),
                      energy_paths=energy_path.tolist()),
                  source_sha256={p.name: file_sha256(p) for p in (ROOT/'rewt').glob('*.py')},
                  script_sha256=file_sha256(Path(__file__)),
                  scope='Float64 CPU geometry audit. Curvature bound covers all amplitudes allowed by this fixed checkpoint. Preconditioned updates change the solver; no learned-quality or speed claim. Numerical eigenvalues are not an interval-arithmetic proof.')
    args.output.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({key: record[key] for key in ('global_curvature_upper', 'global_curvature_lower', 'strict_descent_step_upper_bound', 'frequency_stable_rank', 'geometry_preparation_cpu_seconds')}))


if __name__ == '__main__':
    main()
