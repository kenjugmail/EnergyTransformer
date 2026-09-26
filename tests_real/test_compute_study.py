import json
from types import SimpleNamespace

import torch

import energy_real.compute_study as study
from energy_real.model import Config, LM


def test_objectives_keep_independent_models():
    config = Config(kind='softmax', width=16, layers=1, heads=2, ff_width=32)
    first, second = LM(config), LM(config)
    objectives = [study.objective_for(first), study.objective_for(second)]
    tokens = torch.arange(8)[None]
    objectives[0](tokens, tokens).backward()
    assert any(p.grad is not None and bool(p.grad.ne(0).any()) for p in first.parameters())
    assert all(p.grad is None for p in second.parameters())


def test_cpu_quanta_keep_different_cost_models_at_matched_progress(monkeypatch, tmp_path):
    clock = SimpleNamespace(cpu=0.)

    class Toy(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.weight = torch.nn.Parameter(torch.ones(()))

        def parameters_count(self):
            return 1

    def objective_for(model):
        def objective(x, y):
            # A slow model consumes three times the CPU of a fast one.
            clock.cpu += .3 if model.config.kind == 'hybrid' else .1
            return model.weight.square()
        return objective

    monkeypatch.setattr(study, 'LM', Toy)
    monkeypatch.setattr(study, 'objective_for', objective_for)
    monkeypatch.setattr(study.time, 'process_time', lambda: clock.cpu)
    monkeypatch.setattr(study, 'score', lambda model, data, budget: {'bpb': float(model.weight.detach())})
    data = (torch.zeros(1, 1, dtype=torch.long),) * 2
    corpus = SimpleNamespace(batch=lambda *args: data)
    budget = SimpleNamespace(check=lambda: None)
    fits = study.fit_group([('softmax', 1, .002), ('hybrid', 1, .002)],
                           1., corpus, data, budget, tmp_path, 'test', quantum=.25)
    rows = json.loads((tmp_path / 'test_scheduler.json').read_text())
    assert len(rows) == 4
    for row in rows:
        values = list(row['cpu_s'].values())
        target = min(1., (row['round'] + 1) * .25)
        assert min(values) >= target - 1e-9
        assert max(values) <= target + .3 + 1e-9
        assert max(values) - min(values) <= .3 + 1e-9
    by_arm = {fit['arm']: fit for fit in fits}
    assert by_arm['softmax']['steps'] > by_arm['hybrid']['steps']
    assert all(fit['cpu_s'] >= 1. for fit in fits)
