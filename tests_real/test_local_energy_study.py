from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from energy_real.local_energy_data import graph_dataset
from energy_real.local_energy_study import (
    Budget, BudgetExpired, FitConfig, evaluate_fit, load_model, parser, run, train_fit,
)


class Unlimited:
    def check(self):
        pass


def test_interrupted_fit_resumes_exact_weights_and_selection(tmp_path):
    torch.set_num_threads(2)
    train, valid = graph_dataset('train', 32, 3), graph_dataset('valid', 12, 4)
    config = FitConfig('graph', 'pcalm', 11, 8, 2, True, 8, 8, 4)
    reference = train_fit(config, train, valid, tmp_path / 'reference', Unlimited(), 'test-source')
    interrupted = tmp_path / 'interrupted'
    class StopAfterCommit:
        def check(self):
            path = interrupted / 'progress.json'
            if path.exists() and json.loads(path.read_text())['step'] >= 2:
                raise BudgetExpired('simulated interruption')
    with pytest.raises(BudgetExpired):
        train_fit(config, train, valid, interrupted, StopAfterCommit(), 'test-source')
    # Simulate an uncommitted best.pt tail. Resume must repair it from last.pt.
    (interrupted / 'best.pt').write_bytes(b'not a committed checkpoint')
    resumed = train_fit(config, train, valid, interrupted, Unlimited(), 'test-source', resume=True)
    assert resumed['best_step'] == reference['best_step']
    assert resumed['selected_validation']['rows'] == reference['selected_validation']['rows']
    for name in ('best', 'last'):
        a = torch.load(tmp_path / 'reference' / f'{name}.pt', weights_only=False)
        b = torch.load(interrupted / f'{name}.pt', weights_only=False)
        if name == 'last':
            a, b = a['model'], b['model']
        for key in a['state_dict']:
            torch.testing.assert_close(a['state_dict'][key], b['state_dict'][key], rtol=0, atol=0)


def test_completed_fit_rejects_changed_source_or_checkpoint(tmp_path):
    train, valid = graph_dataset('train', 16, 3), graph_dataset('valid', 6, 4)
    config = FitConfig('graph', 'bp', 11, 8, 2, False, 2, 8, 4)
    out = tmp_path / 'fit'
    result = train_fit(config, train, valid, out, Unlimited(), 'source-a')
    with pytest.raises(ValueError, match='identity'):
        train_fit(config, train, valid, out, Unlimited(), 'source-b', resume=True)
    (out / 'best.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checkpoint'):
        train_fit(config, train, valid, out, Unlimited(), 'source-a', resume=True)
    with pytest.raises(ValueError, match='Checkpoint'):
        evaluate_fit(result, valid, Unlimited())


def test_budget_carries_actual_and_unclosed_reservations(tmp_path):
    prior = tmp_path / 'prior.json'
    prior.write_text(json.dumps({'stages': {'old': {'used_s': 100}}}))
    ledger = tmp_path / 'budget.json'
    budget = Budget(ledger, 10, prior)
    budget.finish('completed')
    payload = json.loads(ledger.read_text())
    assert 100 <= payload['total_used_s'] < 110
    payload['runs'].append({'status': 'running', 'reserved_s': 20, 'elapsed_s': 0})
    ledger.write_text(json.dumps(payload))
    resumed = Budget(ledger, 10, prior)
    assert resumed.data['runs'][1]['elapsed_s'] == 20
    assert resumed.data['runs'][1]['status'] == 'unclosed_charged_reservation'
    resumed.finish('completed')
    with pytest.raises(ValueError, match='remaining'):
        Budget(ledger, 28800, prior)


def test_tiny_complete_study_freezes_before_test_and_can_resume(tmp_path, monkeypatch):
    import energy_real.local_energy_study as study
    prior = tmp_path / 'prior.json'
    prior.write_text(json.dumps({'stages': {'old': {'used_s': 0}}}))
    args = parser().parse_args(['--out', str(tmp_path / 'study'), '--tasks', 'graph', '--seeds', '11',
                               '--graph-steps', '2', '--graph-depth', '2', '--graph-width', '6',
                               '--graph-train', '16', '--graph-valid', '6', '--graph-test', '6', '--batch', '8',
                               '--budget-ledger', str(tmp_path / 'budget.json'), '--prior-ledger', str(prior),
                               '--max-seconds', '120'])
    original = study.evaluate_fit
    def checked(record, data, budget):
        selected = json.loads((args.out / 'selection.json').read_text())
        assert selected['status'] == 'frozen_before_test'
        assert len(selected['records']) == 8
        return original(record, data, budget)
    monkeypatch.setattr(study, 'evaluate_fit', checked)
    first = run(args)
    assert first['status'] == 'completed'
    assert first['completed_fits'] == first['evaluated_fits'] == 8
    args.resume = True
    def forbidden(*args, **kwargs):
        raise AssertionError('A completed test evaluation was repeated')
    monkeypatch.setattr(study, 'evaluate_fit', forbidden)
    second = run(args)
    assert first['test_results'] == second['test_results']
