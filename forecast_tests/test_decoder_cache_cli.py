"""Publication regression tests; no private checkpoint or GPU is needed."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
from energy_forecast import decoder_cache as cache


@pytest.fixture
def cli_args(tmp_path):
    # Sparse fixed-size fixture; the mocked digest avoids reading 100 MB per test.
    corpus = tmp_path / 'enwik8'
    with corpus.open('wb') as stream:
        stream.truncate(100_000_000)
    return ['--teacher', str(tmp_path / 'unused.pt'), '--corpus', str(corpus),
            '--data-sha256', 'fixture-hash', '--shards', str(tmp_path / 'shards'),
            '--out', str(tmp_path / 'cache.pt')]


@pytest.mark.parametrize('status,code', [('completed', 0), ('partial', 75)])
def test_cli_reports_distinct_completion_status(cli_args, monkeypatch, capsys, status, code):
    monkeypatch.setattr(cache, 'file_hash', lambda _: 'fixture-hash')
    monkeypatch.setattr(cache, 'build', lambda *a, **kw: {'status': status})
    assert cache.main(cli_args) == code
    assert json.loads(capsys.readouterr().out)['status'] == status


def test_cli_unknown_completion_status_is_error(cli_args, monkeypatch):
    monkeypatch.setattr(cache, 'file_hash', lambda _: 'fixture-hash')
    monkeypatch.setattr(cache, 'build', lambda *a, **kw: {'status': 'unexpected'})
    with pytest.raises(RuntimeError, match='unknown status'):
        cache.main(cli_args)


def test_partial_status_reaches_process_exit(cli_args):
    script = (
        'from energy_forecast import decoder_cache as c; '
        "c.file_hash=lambda _: 'fixture-hash'; "
        "c.build=lambda *a,**kw: {'status':'partial'}; "
        f'raise SystemExit(c.main({cli_args!r}))'
    )
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert result.returncode == 75, result.stderr
    assert json.loads(result.stdout)['status'] == 'partial'


@pytest.mark.parametrize('changes', [
    {'batch': True}, {'batch': 1.5}, {'batch': 0},
    {'stride': True}, {'stride': 1.5}, {'stride': 0},
    {'max_seconds': True}, {'max_seconds': float('nan')},
    {'max_seconds': float('inf')}, {'max_seconds': '900'},
])
def test_invalid_limits_rejected_before_loading(tmp_path, changes):
    with pytest.raises(ValueError, match='Invalid extraction limits'):
        cache.build('missing-teacher', 'missing-corpus', [], tmp_path/'shards',
                    tmp_path/'data.pt', **changes)


@pytest.mark.parametrize('blocks,total', [
    (None, 96), ([], True), ([], 0),
    ([None], 96), ([{'split': 'train'}], 96),
    ([{'split': 'unrecognized', 'offset': 0, 'bytes': 32}], 96),
])
def test_invalid_block_schema(blocks, total):
    with pytest.raises(ValueError):
        cache.validate_blocks(blocks, total)


def test_existing_dataset_rejected_before_teacher_or_shard_mutation(tmp_path, monkeypatch):
    corpus = tmp_path/'bytes'; corpus.write_bytes(bytes(range(96)))
    blocks = [{'split': s, 'offset': i*32, 'bytes': 32}
              for i,s in enumerate(('train','validation','test'))]
    out = tmp_path/'data.pt'; out.write_bytes(b'keep existing data')
    def must_not_load(*a, **kw):
        raise AssertionError('Teacher should not be loaded for a conflicting output')
    monkeypatch.setattr(cache, 'load_energy_model', must_not_load)
    with pytest.raises(FileExistsError):
        cache.build('unused', corpus, blocks, tmp_path/'shards', out)
    assert out.read_bytes() == b'keep existing data'
    assert not (tmp_path/'shards').exists()
