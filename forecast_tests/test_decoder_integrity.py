"""Regression fixtures for study identity, progress, publication, and evaluation.

These tiny random examples check execution contracts, not language-model quality.
"""
from pathlib import Path
import json
import math
import subprocess
import sys

import pytest
import torch

from energy_forecast.decoder_integrity import (
    atomic_json, exclusive_directory, same_json, selected_runs,
    validate_head_metadata, validate_schedule,
)
from energy_forecast.probe import checkpoint_hash
from energy_forecast.train_decoder import main, train_one
from energy_forecast.evaluate_decoder import main as evaluate


def data_fixture():
    rng = torch.Generator().manual_seed(27001)
    def part(n):
        return {'h': torch.randn(n, 16, generator=rng),
                'x': torch.randn(n, 16, 8, generator=rng),
                'y': torch.randint(0, 8, (n,), generator=rng),
                'group': torch.arange(n) // (n // 2), 'anchor': torch.arange(n)}
    return {'train': part(24), 'validation': part(12), 'test': part(20),
            'weight': torch.randn(8, 16, generator=rng),
            'metadata': {'width': 16, 'history': 16, 'variables': 8,
                         'horizons': [1, 4, 16, 64], 'teacher_sha256': 'fixture',
                         'normalization': {'center': torch.zeros(8), 'scale': torch.ones(8)}}}


def setup(tmp_path, *, seeds=('7',), kinds=('hidden',), steps=2):
    torch.set_num_threads(1)
    data = data_fixture()
    path = tmp_path / 'data.pt'
    torch.save(data, path)
    args = ['--data', str(path), '--out', str(tmp_path / 'fits'), '--seeds', *seeds,
            '--kinds', *kinds, '--steps', str(steps), '--batch', '6']
    return data, path, args


def inventory(path):
    return {str(p.relative_to(path)): checkpoint_hash(p) for p in path.rglob('*') if p.is_file()}


def test_completed_resume_is_read_only(tmp_path):
    _, _, args = setup(tmp_path)
    assert main(args) == 0
    before = inventory(tmp_path / 'fits')
    assert main(args + ['--resume']) == 0
    assert inventory(tmp_path / 'fits') == before


def test_changed_data_cannot_relabel_old_checkpoints(tmp_path):
    data, path, args = setup(tmp_path)
    assert main(args) == 0
    before = inventory(tmp_path / 'fits')
    data['weight'][0, 0] += 1
    torch.save(data, path)
    with pytest.raises(ValueError, match='Study identity'):
        main(args + ['--resume'])
    assert inventory(tmp_path / 'fits') == before


@pytest.mark.parametrize('flag,value', [('--batch', '7'), ('--steps', '3'), ('--seeds', '19'), ('--kinds', 'history')])
def test_completed_changed_schedule_rejected_before_write(tmp_path, flag, value):
    _, _, args = setup(tmp_path)
    assert main(args) == 0
    before = inventory(tmp_path / 'fits')
    args[args.index(flag) + 1] = value
    with pytest.raises(ValueError, match='Study identity'):
        main(args + ['--resume'])
    assert inventory(tmp_path / 'fits') == before


def test_zero_step_cli_snapshot_and_resume(tmp_path):
    _, _, args = setup(tmp_path)
    assert main(args + ['--stop-after', '0']) == 75
    assert not (tmp_path / 'fits/FROZEN.json').exists()
    old = torch.load(tmp_path / 'fits/hidden_seed7/last.pt', weights_only=True)
    assert old['step'] == 0
    assert main(args + ['--resume']) == 0
    frozen = json.loads((tmp_path / 'fits/FROZEN.json').read_text())
    assert frozen['seeds'] == [7]


def test_cli_process_partial_exit_75(tmp_path):
    _, _, args = setup(tmp_path)
    result = subprocess.run([sys.executable, '-m', 'energy_forecast.train_decoder', *args,
                             '--stop-after', '0'], capture_output=True, text=True)
    assert result.returncode == 75, result.stderr
    assert json.loads(result.stdout.splitlines()[-1])['status'] == 'partial'


def test_backward_resume_does_not_rewrite_any_artifact(tmp_path):
    _, _, args = setup(tmp_path, steps=4)
    assert main(args + ['--stop-after', '3']) == 75
    before = inventory(tmp_path / 'fits')
    with pytest.raises(ValueError, match='refusing to rewind'):
        main(args + ['--resume', '--stop-after', '2'])
    assert inventory(tmp_path / 'fits') == before


def test_all_arms_prevalidated_before_any_continuation(tmp_path):
    _, _, args = setup(tmp_path, seeds=('7', '19'), steps=4)
    assert main(args + ['--stop-after', '1']) == 75
    path = tmp_path / 'fits/hidden_seed19/last.pt'
    old = torch.load(path, weights_only=True)
    old['identity']['data_sha256'] = 'wrong'
    torch.save(old, path)
    before = inventory(tmp_path / 'fits')
    with pytest.raises(ValueError, match='resume identity'):
        main(args + ['--resume'])
    assert inventory(tmp_path / 'fits') == before


@pytest.mark.parametrize('name', ['best.pt', 'last.pt', 'training.json'])
def test_completed_artifact_modification_rejected(tmp_path, name):
    _, _, args = setup(tmp_path)
    assert main(args) == 0
    path = tmp_path / 'fits/hidden_seed7' / name
    if name.endswith('.json'):
        row = json.loads(path.read_text())
        row['selected_step'] += 1
        path.write_text(json.dumps(row))
    else:
        payload = torch.load(path, weights_only=True)
        next(iter(payload['weights'].values())).add_(.01)
        torch.save(payload, path)
    before = inventory(tmp_path / 'fits')
    with pytest.raises(ValueError):
        main(args + ['--resume'])
    assert inventory(tmp_path / 'fits') == before


@pytest.mark.parametrize('flag,values', [('--seeds', ['7', '7']), ('--kinds', ['hidden', 'hidden'])])
def test_duplicate_study_members_fail_before_creation(tmp_path, flag, values):
    _, _, args = setup(tmp_path)
    index = args.index(flag)
    args[index + 1:index + 2] = values
    with pytest.raises(ValueError, match='Repeated'):
        main(args)
    assert not (tmp_path / 'fits').exists()


def test_missing_forecast_heads_fail_before_creation(tmp_path):
    _, _, args = setup(tmp_path, kinds=('mean',))
    with pytest.raises(ValueError, match='require --heads'):
        main(args)
    assert not (tmp_path / 'fits').exists()


def test_nondefault_seeds_evaluate_exact_selection(tmp_path):
    _, path, args = setup(tmp_path, seeds=('7', '19'), kinds=('hidden', 'history'))
    assert main(args) == 0
    evaluate(['--data', str(path), '--fits', str(tmp_path / 'fits'), '--out', str(tmp_path / 'eval')])
    row = json.loads((tmp_path / 'eval/EVALUATION.json').read_text())
    assert row['seeds'] == [7, 19] and row['candidate'] is None
    import pandas as pd
    result = pd.read_csv(tmp_path / 'eval/runs.csv')
    assert set(result.query("kind != 'base'").seed) == {7, 19}


def test_nonfinite_initial_validation_has_no_model_publication(tmp_path):
    data = data_fixture()
    data['validation']['h'].fill_(float('nan'))
    with pytest.raises((FloatingPointError, ValueError)):
        train_one(data, 'fixture', {'train': None, 'validation': None}, None,
                  tmp_path / 'bad', 'hidden', steps=1, batch=6)
    assert not (tmp_path / 'bad/best.pt').exists()
    assert not (tmp_path / 'bad/last.pt').exists()


@pytest.mark.parametrize('steps,batch,seed,stop', [(True, 2, 1, None), (0, 2, 1, None),
    (2, 1.5, 1, None), (2, 2, -1, None), (2, 2, 1, True), (2, 2, 1, 3)])
def test_invalid_schedule(steps, batch, seed, stop):
    with pytest.raises(ValueError):
        validate_schedule('hidden', seed, steps, batch, .001, stop)


def test_head_binding_checks_coordinates_not_just_teacher():
    meta = data_fixture()['metadata']
    h = {'teacher_sha256': meta['teacher_sha256'],
         'config': {k: meta[k] for k in ('history', 'variables', 'horizons')},
         'normalization': {k: v.clone() for k, v in meta['normalization'].items()}}
    validate_head_metadata(h, meta)
    h['normalization']['center'][0] += .1
    with pytest.raises(ValueError, match='normalization differs'):
        validate_head_metadata(h, meta)
    h['normalization'] = meta['normalization']
    h['config']['horizons'] = [1, 2, 16, 64]
    with pytest.raises(ValueError, match='horizons'):
        validate_head_metadata(h, meta)


def test_local_competing_writer_and_lock_reuse(tmp_path):
    path = tmp_path / 'locked'
    with exclusive_directory(path, resume=False):
        code = ("from energy_forecast.decoder_integrity import exclusive_directory; "
                f"p={str(path)!r}; "
                "\nwith exclusive_directory(p,resume=True):\n print('incorrectly acquired')")
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        assert result.returncode != 0 and 'Another writer' in result.stderr
    with exclusive_directory(path, resume=True):
        pass
    assert (path / '.writer.lock').exists()


def test_process_exit_releases_lock(tmp_path):
    path = tmp_path / 'exited'
    code = ("import os;from energy_forecast.decoder_integrity import exclusive_directory;"
            f"\nwith exclusive_directory({str(path)!r},resume=False):\n os._exit(23)")
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 23, result.stderr
    with exclusive_directory(path, resume=True):
        pass


def test_atomic_json_failed_replace_preserves_old(tmp_path, monkeypatch):
    import energy_forecast.decoder_integrity as m
    dest = tmp_path / 'receipt.json'
    atomic_json({'old': True}, dest)
    original = dest.read_bytes()
    def fail(*args):
        raise OSError('injected replace failure')
    monkeypatch.setattr(m.os, 'replace', fail)
    with pytest.raises(OSError, match='injected'):
        atomic_json({'new': True}, dest)
    assert dest.read_bytes() == original
    assert not list(tmp_path.glob('*.tmp'))


def test_atomic_json_nan_rejected_before_write(tmp_path):
    dest = tmp_path / 'receipt.json'
    with pytest.raises(ValueError):
        atomic_json({'bad': math.nan}, dest)
    assert not list(tmp_path.iterdir())


def test_selected_legacy_seed_inference():
    row = {'validation_means': {'hidden': 1., 'history': 1.},
           'files': {f'{k}_seed{s}/best.pt': 'a' * 64 for k in ('hidden', 'history') for s in (7, 19)}}
    assert selected_runs(row)[1] == [7, 19]
    del row['files']['hidden_seed7/best.pt']
    with pytest.raises(ValueError):
        selected_runs(row)


@pytest.mark.parametrize('path', ['../hidden_seed7/best.pt', '/hidden_seed7/best.pt',
                                   'hidden_seed-1/best.pt', 'unknown_seed7/best.pt'])
def test_selected_paths_fail_closed(path):
    with pytest.raises(ValueError):
        selected_runs({'validation_means': {'hidden': 1.}, 'files': {path: 'a' * 64}})
