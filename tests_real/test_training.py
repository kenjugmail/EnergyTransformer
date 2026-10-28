from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
from unittest.mock import patch
import numpy as np
import pytest
import torch
from energy_real import data
from energy_real.data import Enwik8
from energy_real.runtime import atomic_checkpoint, sha256
from energy_real.train import run, parser


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(data, 'EXPECTED_BYTES', 192)
    monkeypatch.setattr(data, 'SPLITS', {'train': (0, 128), 'valid': (128, 160), 'test': (160, 192)})
    path = tmp_path / 'bytes.bin'
    path.write_bytes(bytes(range(192)))
    return path


def args_for(corpus, out, extra=()):
    return parser().parse_args(['--data', str(corpus), '--data-sha256', sha256(corpus), '--out', str(out),
        '--device', 'cpu', '--width', '16', '--layers', '1', '--heads', '2', '--slots', '4',
        '--ff-width', '32', '--seq', '8', '--batch', '2', '--steps', '6', '--warmup', '1',
        '--eval-every', '2', '--eval-bytes', '32', '--eval-batch', '2', '--save-every', '2', *extra])


def equal(a, b):
    if isinstance(a, torch.Tensor): return torch.equal(a, b)
    if isinstance(a, dict): return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)): return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


@pytest.mark.parametrize('extra', [
    ['--layers', '0'], ['--width', '0'], ['--heads', '0'],
    ['--kind', 'hybrid', '--attention-window', '4', '--attention-every', '0'],
    ['--kind', 'hybrid', '--attention-window', '4', '--layers', '2'],
    ['--kind', 'hybrid', '--layers', '3', '--attention-window', '-1'],
])
def test_invalid_architecture_fails_before_gpu_or_filesystem(corpus, tmp_path, extra):
    out = tmp_path / 'must_not_exist'
    args = args_for(corpus, out, [*extra, '--device', 'cuda'])
    with patch('energy_real.device.require_idle_gpu') as gpu, patch('energy_real.train.Enwik8') as data:
        with pytest.raises(ValueError):
            run(args, corpus_factory=data)
        gpu.assert_not_called()
        data.assert_not_called()
    assert not out.exists()


def test_exact_resume_and_crash_tail(corpus, tmp_path):
    whole, resumed = tmp_path/'whole', tmp_path/'resume'
    run(args_for(corpus, whole))
    run(args_for(corpus, resumed, ['--stop-after', '3']))
    with (resumed/'steps.jsonl').open('a') as f: f.write('{"partial":')
    run(args_for(corpus, resumed, ['--resume']))
    a = torch.load(whole/'last.pt', weights_only=True)
    b = torch.load(resumed/'last.pt', weights_only=True)
    for key in ['weights', 'optimizer', 'torch_rng', 'numpy_rng', 'step', 'tokens', 'best_valid_bpb', 'best_step']:
        assert equal(a[key], b[key]), key
    assert list(resumed.glob('recovered_steps.*.jsonl'))[0].read_text() == '{"partial":'
    assert b['tokens'] == 96 and b['step'] == 6
    with pytest.raises(FileExistsError): run(args_for(corpus, resumed))
    with pytest.raises(ValueError, match='identity'):
        run(args_for(corpus, resumed, ['--resume', '--lr', '.1']))


def test_pcalm_native_training_exact_resume(corpus, tmp_path):
    extra = ['--learning-rule', 'pcalm', '--amp', 'none', '--activity-steps', '3', '--layers', '2']
    whole, split = tmp_path / 'pcalm_whole', tmp_path / 'pcalm_split'
    run(args_for(corpus, whole, extra))
    run(args_for(corpus, split, [*extra, '--stop-after', '3']))
    run(args_for(corpus, split, [*extra, '--resume']))
    a, b = [torch.load(path / 'last.pt', weights_only=True) for path in (whole, split)]
    for key in ('weights', 'optimizer', 'torch_rng', 'numpy_rng', 'step', 'tokens', 'best_valid_bpb'):
        assert equal(a[key], b[key]), key
    with pytest.raises(ValueError, match='identity'):
        run(args_for(corpus, split, [*extra, '--resume', '--state-lr', '.01']))


def test_window_hybrid_training_exact_resume(corpus, tmp_path):
    extra = ['--kind', 'hybrid', '--layers', '3', '--attention-window', '4']
    whole, split = tmp_path / 'window_whole', tmp_path / 'window_split'
    run(args_for(corpus, whole, extra))
    run(args_for(corpus, split, [*extra, '--stop-after', '3']))
    run(args_for(corpus, split, [*extra, '--resume']))
    a, b = [torch.load(path / 'last.pt', weights_only=True) for path in (whole, split)]
    for key in ('weights', 'optimizer', 'torch_rng', 'numpy_rng', 'step', 'tokens', 'best_valid_bpb'):
        assert equal(a[key], b[key]), key
    with pytest.raises(ValueError, match='identity'):
        run(args_for(corpus, split, [*extra, '--resume', '--attention-window', '5']))


def test_fresh_process_resume(corpus, tmp_path):
    script = '''from energy_real import data
from energy_real.train import main
data.EXPECTED_BYTES=192
data.SPLITS={'train':(0,128),'valid':(128,160),'test':(160,192)}
main()
'''
    common = ['--data', str(corpus), '--data-sha256', sha256(corpus), '--device', 'cpu', '--width', '16',
              '--layers', '1', '--heads', '2', '--slots', '4', '--ff-width', '32', '--seq', '8',
              '--batch', '2', '--steps', '4', '--eval-every', '2', '--eval-bytes', '32', '--save-every', '2']
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    for name, extra in [('whole', []), ('split', ['--stop-after', '2']), ('split', ['--resume'])]:
        subprocess.run([sys.executable, '-c', script, *common, '--out', str(tmp_path/name), *extra],
                       env=env, check=True, capture_output=True, timeout=60)
    a, b = [torch.load(tmp_path/name/'last.pt', weights_only=True) for name in ('whole', 'split')]
    for key in ['weights', 'optimizer', 'torch_rng', 'numpy_rng', 'step', 'tokens']:
        assert equal(a[key], b[key]), key


def test_bad_log_preserved_and_checkpoint_atomic(corpus, tmp_path):
    out = tmp_path/'run'
    run(args_for(corpus, out, ['--stop-after', '2']))
    path = out/'last.pt'; before = sha256(path)
    with patch('energy_real.runtime.os.replace', side_effect=OSError('injected')):
        with pytest.raises(OSError): atomic_checkpoint({'weights': {}}, path)
    assert sha256(path) == before
    log = out/'steps.jsonl'; log.write_text('corrupt\n')
    with pytest.raises(ValueError, match='committed'):
        run(args_for(corpus, out, ['--resume']))
    assert log.read_text() == 'corrupt\n'


def test_weight_only_checkpoint_is_warm_start_not_resume(corpus, tmp_path):
    out = tmp_path/'first'; run(args_for(corpus, out))
    next_out = tmp_path/'warm'
    run(args_for(corpus, next_out, ['--warm-start', str(out/'best.pt'), '--stop-after', '0']))
    saved = torch.load(next_out/'last.pt', weights_only=True)
    assert saved['step'] == 0 and saved['tokens'] == 0 and not saved['optimizer']['state']
    assert saved['initial_sha256'] == sha256(out/'best.pt')
    with pytest.raises(FileNotFoundError):
        run(args_for(corpus, tmp_path/'missing', ['--resume']))


def test_last_crop_and_exact_target_coverage(corpus):
    c = Enwik8(corpus, 'train')
    class Last:
        def integers(self, low, high, size):
            assert high == 128 - 8
            return np.full(size, high - 1)
    x, y = c.batch(Last(), 1, 8)
    assert y[0, -1] == 127
    for length in [1, 7, 31, 32, 127, 200]:
        ys = [y.flatten() for _, y in c.sequential_windows(length, batch=3)]
        assert torch.equal(torch.cat(ys), torch.arange(1, 128))
    for split, lo, hi in [('train', 0, 128), ('valid', 128, 160), ('test', 160, 192)]:
        assert bytes(Enwik8(corpus, split).data) == bytes(range(lo, hi))
    with pytest.raises(ValueError, match='checksum'): Enwik8(corpus, expected_sha256='0'*64)
    with pytest.raises(ValueError): c.batch(np.random.default_rng(), 1, 128)


def test_token_budget_stops_at_completed_step(corpus, tmp_path):
    result = run(args_for(corpus, tmp_path/'budget', ['--max-tokens', '35']))
    assert result['status'] == 'budget_exhausted' and result['tokens'] == 32 and result['steps'] == 2


def test_uncommitted_best_checkpoint_is_preserved_and_restored(corpus,tmp_path):
    out=tmp_path/'run';run(args_for(corpus,out,['--stop-after','2']))
    committed=torch.load(out/'best.pt',weights_only=True)
    future={**committed,'step':999}
    atomic_checkpoint(future,out/'best.pt')
    run(args_for(corpus,out,['--resume','--stop-after','2']))
    assert torch.load(out/'best.pt',weights_only=True)['step']==2
    recovered=list(out.glob('recovered_best.*.pt'))
    assert len(recovered)==1 and torch.load(recovered[0],weights_only=True)['step']==999


def test_sigterm_saves_completed_update(corpus,tmp_path):
    import signal,time
    script='''from energy_real import data
from energy_real.train import main
data.EXPECTED_BYTES=192
data.SPLITS={'train':(0,128),'valid':(128,160),'test':(160,192)}
main()
'''
    out=tmp_path/'signal'
    command=[sys.executable,'-c',script,'--data',str(corpus),'--data-sha256',sha256(corpus),
             '--out',str(out),'--device','cpu','--width','16','--layers','1','--heads','2','--slots','4',
             '--ff-width','32','--seq','8','--batch','2','--steps','10000','--eval-every','100',
             '--eval-bytes','32','--save-every','2']
    process=subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,
                             env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    try:
        deadline=time.monotonic()+20
        while not (out/'last.pt').exists() and process.poll() is None and time.monotonic()<deadline:time.sleep(.02)
        assert (out/'last.pt').exists()
        process.send_signal(signal.SIGTERM)
        _,stderr=process.communicate(timeout=20)
        assert process.returncode==0,stderr.decode()
        result=json.loads((out/'result.json').read_text())
        saved=torch.load(out/'last.pt',weights_only=True)
        assert result['status']=='interrupted' and saved['tokens']==saved['step']*16
    finally:
        if process.poll() is None:process.kill();process.wait()
