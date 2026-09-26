"""Fault injection and concurrent publication for complete local artifacts."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import pytest
import torch
from energy_forecast.experiment import atomic_save


def test_snapshot_roundtrip_and_nested_directory(tmp_path):
    path = tmp_path/'nested'/'snapshot.pt'
    atomic_save({'value':torch.arange(7)}, path)
    assert torch.equal(torch.load(path, weights_only=True)['value'], torch.arange(7))
    assert not list(path.parent.glob('*.tmp'))


def test_failed_serialization_preserves_old_file_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path/'snapshot.pt'; atomic_save({'value':torch.ones(2)}, path)
    before = path.read_bytes()
    def fail(data, file): file.write(b'incomplete'); raise OSError('simulated failed write')
    monkeypatch.setattr(torch, 'save', fail)
    with pytest.raises(OSError): atomic_save({'value':2}, path)
    assert path.read_bytes() == before and not list(tmp_path.glob('*.tmp'))


def test_failed_replace_preserves_old_file(tmp_path, monkeypatch):
    import energy_forecast.experiment as module
    path=tmp_path/'snapshot.pt'; atomic_save({'value':1},path);before=path.read_bytes()
    def fail(*args): raise OSError('simulated replace failure')
    monkeypatch.setattr(module.os, 'replace', fail)
    with pytest.raises(OSError): atomic_save({'value':2},path)
    assert path.read_bytes()==before and not list(tmp_path.glob('*.tmp'))


def test_predictable_old_temp_name_untouched(tmp_path):
    target=tmp_path/'data.pt';sentinel=tmp_path/'data.pt.tmp';sentinel.write_bytes(b'unrelated')
    atomic_save({'value':1},target)
    assert sentinel.read_bytes()==b'unrelated'


def test_existing_dataset_not_overwritten(tmp_path):
    path=tmp_path/'data.pt';atomic_save({'value':1},path,overwrite=False);before=path.read_bytes()
    with pytest.raises(FileExistsError):atomic_save({'value':2},path,overwrite=False)
    assert path.read_bytes()==before and not list(tmp_path.glob('*.tmp'))


def test_racing_create_only_writers_publish_exactly_once(tmp_path):
    path=tmp_path/'data.pt'; barrier=threading.Barrier(2)
    def publish(value):
        barrier.wait()
        try: atomic_save({'value':value},path,overwrite=False); return ('published',value)
        except FileExistsError: return ('exists',value)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(publish,[1,2]))
    assert sorted(r[0] for r in results)==['exists','published']
    expected=next(v for status,v in results if status=='published')
    assert torch.load(path,weights_only=True)['value']==expected
    assert not list(tmp_path.glob('*.tmp'))
