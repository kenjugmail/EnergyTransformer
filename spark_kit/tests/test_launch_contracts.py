"""Additional launch and resource contracts, without fabricating a CUDA device."""
import json,time
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
from energy_spark.common import ROOT,atomic_json,source_identity,environment,sha256
from energy_spark.train import require_acceptance,FIT_FIELDS,validate_config
from energy_spark.accept import run

def test_explicit_cuda_not_run_on_cpu_host(tmp_path):
    if torch.cuda.is_available():pytest.skip('CPU-specific negative test')
    r=run(tmp_path,'cuda')
    assert r['status']=='NOT_RUN_CUDA_UNAVAILABLE' and r['checks']=={}

def test_changed_runtime_not_accepted(tmp_path):
    c=json.loads((ROOT/'configs/v22/cpu_test.json').read_text());e=environment('cpu');e['torch']='different-build'
    r={'status':'PASS_FOR_BOUNDED_DEVICE_PILOT','source_sha256':source_identity(),'checkpoint_sha256':None,'environment':e,'completed_unix':time.time(),'fit_configurations':[{k:c[k] for k in FIT_FIELDS}]}
    p=tmp_path/'receipt.json';atomic_json(r,p)
    with pytest.raises(ValueError,match='runtime'):require_acceptance(p,'cpu',c)

def test_unmeasured_shape_not_accepted(tmp_path):
    c=json.loads((ROOT/'configs/v22/cpu_test.json').read_text())
    r={'status':'PASS_FOR_BOUNDED_DEVICE_PILOT','source_sha256':source_identity(),'checkpoint_sha256':None,'environment':environment('cpu'),'completed_unix':time.time(),'fit_configurations':[]}
    p=tmp_path/'receipt.json';atomic_json(r,p)
    with pytest.raises(ValueError,match='fit-tested'):require_acceptance(p,'cpu',c)

@pytest.mark.parametrize('name',['continuation_973k','pilot_3m','pilot_3m_softmax','pilot_10m','pilot_10m_softmax','cpu_test'])
def test_each_config_respects_declared_cap(name):
    c=validate_config(json.loads((ROOT/f'configs/v22/{name}.json').read_text()))
    assert c['steps']*c['sequence_length']*c['microbatch']*c['accumulation']<=c['max_tokens']

def test_launcher_does_not_install_or_upload():
    s=(ROOT/'spark/run.sh').read_text()
    assert '--network none' in s and '--read-only' in s and '--cap-drop ALL' in s
    assert 'pip install' not in s and 'sudo ' not in s and 'drop_caches' not in s and 'ssh ' not in s
    assert 'docker.sock' not in s and 'gh release' not in s


def test_numerical_fixture_independent_of_missing_or_corrupt_original(tmp_path,monkeypatch):
    import energy_spark.accept as accept
    monkeypatch.setattr(accept,'ROOT',tmp_path)
    a=accept.load_numerical_fixture()
    (tmp_path/'checkpoints').mkdir()
    (tmp_path/'checkpoints/v21_energy_973k.pt').write_bytes(b'not an original')
    b=accept.load_numerical_fixture()
    assert all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict())
    with pytest.raises(ValueError,match='checksum'):accept.load_native(require_original=True)


def test_continuation_requires_original_identity(tmp_path):
    c=json.loads((ROOT/'configs/v22/cpu_test.json').read_text())
    c['init_checkpoint']='checkpoints/not_installed.pt'
    r={'status':'PASS_FOR_BOUNDED_DEVICE_PILOT','source_sha256':source_identity(),
       'checkpoint_sha256':None,'environment':environment('cpu'),'completed_unix':time.time(),
       'fit_configurations':[{k:c[k] for k in FIT_FIELDS}]}
    path=tmp_path/'receipt.json';atomic_json(r,path)
    with pytest.raises(ValueError,match='Continuation'):require_acceptance(path,'cpu',c)
