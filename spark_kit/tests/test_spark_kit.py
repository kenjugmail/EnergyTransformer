from __future__ import annotations
import copy, json, math, os, subprocess, sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from energy_spark import model
from energy_spark.model import ScaleConfig,ScaleLM,state_bytes,transported_scan
from energy_spark import reference_model as ref
from energy_spark.common import ROOT,atomic_json,atomic_checkpoint,sha256,json_hash,Budget,BudgetStop,exclusive_run
from energy_spark.data import fixtures,ByteCorpus,prepare
from energy_spark.train import train,validate_config,make_model,initialize,require_acceptance
from energy_spark.accept import load_native,checkpoint_parity,scan_audit

torch.set_num_threads(1)

def config(**updates):
    c=json.loads((ROOT/'configs/v22/cpu_test.json').read_text());c.update(updates);return c

def tensors_equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return set(a)==set(b) and all(tensors_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(tensors_equal(x,y) for x,y in zip(a,b))
    return a==b

@pytest.mark.parametrize('kind',['energy','softmax'])
@pytest.mark.parametrize('reset',[False,True])
def test_inherited_parity(kind,reset):
    torch.manual_seed(22051)
    c=ScaleConfig(kind=kind,vocab=256,width=32,heads=2,slots=8,layers=2,ff_width=64,chunk=64,cpu_scan=False)
    a=ScaleLM(c);b=ref.ScaleLM(ref.ScaleConfig(**vars(c)));b.load_state_dict(a.state_dict());x=torch.randint(256,(2,33))
    r=torch.zeros_like(x,dtype=torch.bool) if reset else None
    if r is not None:r[1,18]=True
    ya,sa=a(x,reset=r);yb,sb=b(x,reset=r)
    torch.testing.assert_close(ya,yb,rtol=2e-5,atol=2e-5)
    for la,lb in zip(sa,sb):
        for ta,tb in zip(la,lb):torch.testing.assert_close(ta,tb,atol=2e-6,rtol=2e-5)

@pytest.mark.integration
@pytest.mark.skipif(not (ROOT/'checkpoints/v21_energy_973k.pt').is_file(), reason='Original artifact not installed')
def test_actual_inherited_checkpoint():
    a=load_native(require_original=True);d=torch.load(ROOT/'checkpoints/v21_energy_973k.pt',map_location='cpu',weights_only=True)
    b=ref.ScaleLM(ref.ScaleConfig(kind='energy',vocab=256,width=160,layers=4,heads=4,slots=16,ff_width=320,cpu_scan=False));b.load_state_dict(d['weights'])
    x=torch.randint(256,(1,31))
    with torch.no_grad():ya,sa=a(x);yb,sb=b(x)
    assert torch.equal(ya,yb)
    assert a.parameters_count()==972896
    assert state_bytes(sa)==41984

def test_attention_fastpath_and_reset_fallback():
    c=ScaleConfig(kind='softmax',vocab=256,width=32,layers=1,heads=2,ff_width=64,chunk=32,cpu_scan=False);m=ScaleLM(c)
    orig=F.scaled_dot_product_attention;calls=[]
    def record(*args,**kwargs):calls.append(kwargs);return orig(*args,**kwargs)
    x=torch.randint(256,(2,16))
    with patch.object(F,'scaled_dot_product_attention',record):m(x)
    assert calls[0].get('is_causal') is True and 'attn_mask' not in calls[0]
    calls.clear();r=torch.zeros_like(x,dtype=torch.bool);r[:,8]=True
    with patch.object(F,'scaled_dot_product_attention',record):m(x,reset=r)
    assert 'attn_mask' in calls[0]

@pytest.mark.parametrize('kind',['energy','softmax'])
def test_streaming_reset_causality(kind):
    torch.manual_seed(22052);c=ScaleConfig(kind=kind,width=32,vocab=256,layers=2,heads=2,slots=8,ff_width=64,chunk=16,cpu_scan=False);m=ScaleLM(c).eval()
    x=torch.randint(256,(2,39));r=torch.zeros_like(x,dtype=torch.bool);r[:,19]=True
    with torch.no_grad():
        a,state=m(x,reset=r);b,_=m(x[:,19:]);chunks=[];s=None
        for i in range(39):y,s=m(x[:,i:i+1],s,r[:,i:i+1]);chunks.append(y)
        torch.testing.assert_close(a,torch.cat(chunks,1),atol=4e-5,rtol=4e-5)
        torch.testing.assert_close(a[:,19:],b,atol=4e-5,rtol=4e-5)
        xx=x.clone();xx[:,30:]=torch.randint(256,xx[:,30:].shape);out,_=m(xx,reset=r)
        assert torch.equal(a[:,:30],out[:,:30])

def test_activation_checkpoint_gradient():assert checkpoint_parity(torch.device('cpu'))['activation_recompute_gradient_relative']==0

def test_scan_gradient_matches_serial():assert scan_audit(torch.device('cpu'))['gradient_max_abs']<3e-5

def test_scan_gradcheck():
    f=(torch.rand(1,5,2,3,dtype=torch.double)*.8).requires_grad_();b=torch.randn(1,5,2,3,2,dtype=torch.double,requires_grad=True);i=torch.randn(1,2,3,2,dtype=torch.double,requires_grad=True)
    assert torch.autograd.gradcheck(lambda f,b,i:transported_scan(f,b,i,torch.tensor([0,1])),(f,b,i),fast_mode=True)

def test_accumulation_matches_full_batch():
    torch.manual_seed(22053);a=make_model(config());b=copy.deepcopy(a);x=torch.randint(256,(4,19));y=torch.randint(256,x.shape)
    o,_=a(x);F.cross_entropy(o.flatten(0,1),y.flatten()).backward()
    for xx,yy in zip(x.chunk(2),y.chunk(2)):
        oo,_=b(xx);(F.cross_entropy(oo.flatten(0,1),yy.flatten())/2).backward()
    for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-4)

def test_fp32_state_under_cpu_bf16():
    m=make_model(config());x=torch.randint(256,(2,16))
    with torch.autocast('cpu',dtype=torch.bfloat16):y,s=m(x)
    y.float().sum().backward();assert all(t.dtype==torch.float32 for layer in s for t in layer)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())

@pytest.mark.parametrize('field,value',[('steps',3000),('max_tokens',1),('allocator_fraction',.95),('sequence_length',2048),('prior',float('nan')),('microbatch',True),('learning_rate',0),('init_checkpoint','../escape.pt')])
def test_invalid_budgets(field,value):
    c=config(**{field:value})
    with pytest.raises(ValueError):validate_config(c)

def test_data_identity_and_pairing(tmp_path):
    root=fixtures(tmp_path);a=ByteCorpus(root,'train');b=ByteCorpus(root,'train')
    xa,ya=a.batch(3,5,0,4,32);xb,yb=b.batch(3,5,0,4,32)
    assert torch.equal(xa,xb) and torch.equal(ya,yb)
    assert torch.equal(xa[:,1:],ya[:,:-1])
    rows=a.manifest['documents'];sets={split:{r['content_sha256'] for r in rows if r['split']==split} for split in ('train','validation','test')}
    assert not sets['train']&sets['test'] and not sets['train']&sets['validation']
    assert sum(y.numel() for x,y in b.validation_windows(64,129))==129

def test_corrupted_shard_rejected(tmp_path):
    root=fixtures(tmp_path);p=root/'train.bin';b=p.read_bytes();p.write_bytes(b[:-1]+bytes([b[-1]^1]))
    with pytest.raises(ValueError):ByteCorpus(root,'train')

def test_data_metadata_required(tmp_path):
    p=tmp_path/'bad.jsonl';p.write_text(json.dumps({'text':'abc','document_id':'x'})+'\n')
    with pytest.raises(ValueError):prepare(p,tmp_path/'dest',min_bytes=2)

def test_prepare_refuses_overwrite(tmp_path):
    root=fixtures(tmp_path)
    with pytest.raises(FileExistsError):prepare(tmp_path/'fixture.jsonl',root)

def test_atomic_checkpoint_failure_preserves_original(tmp_path):
    p=tmp_path/'x.pt';atomic_checkpoint({'a':torch.ones(2)},p);before=sha256(p)
    with patch('energy_spark.common.os.replace',side_effect=OSError('injected failure')):
        with pytest.raises(OSError):atomic_checkpoint({'a':torch.zeros(2)},p)
    assert sha256(p)==before and not list(tmp_path.glob('*.tmp'))

def test_cpu_receipt_cannot_authorize_gpu_pilot(tmp_path):
    p=tmp_path/'a.json';atomic_json({'status':'CPU_VALIDATION_ONLY'},p)
    with pytest.raises(ValueError):require_acceptance(p,'cpu',config())

def test_reject_distributed():
    from energy_spark.common import configure
    with patch.dict(os.environ,{'WORLD_SIZE':'2'}):
        with pytest.raises(RuntimeError):configure('cpu')

def test_expired_receipt_rejected(tmp_path):
    from energy_spark.common import environment,source_identity
    p=tmp_path/'a.json';atomic_json({'status':'PASS_FOR_BOUNDED_DEVICE_PILOT','source_sha256':source_identity(),'environment':environment('cpu'),'checkpoint_sha256':None,'completed_unix':0},p)
    with pytest.raises(ValueError):require_acceptance(p,'cpu')

def test_budget_clock():
    b=Budget('cpu',seconds=-1)
    with pytest.raises(BudgetStop):b.check()

def test_model_expansion_not_silent(tmp_path, monkeypatch):
    import energy_spark.train as trainer
    monkeypatch.setattr(trainer, 'ROOT', tmp_path)
    (tmp_path/'checkpoints').mkdir()
    torch.save({'format':'wiki_pilot_v21', 'kind':'energy', 'width':999}, tmp_path/'checkpoints/v21_energy_973k.pt')
    c=config(init_checkpoint='checkpoints/v21_energy_973k.pt')
    with pytest.raises(ValueError):initialize(make_model(c),c)

def test_trainer_fresh_process_resume(tmp_path):
    data=fixtures(tmp_path/'data');c=config();cp=tmp_path/'cfg.json';atomic_json(c,cp)
    # A fresh child interpreter drives all paths, rather than merely reloading in-process.
    env={**os.environ,'PYTHONPATH':str(ROOT),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    common=[sys.executable,'-m','energy_spark.train','--config',str(cp),'--data',str(data),'--device','cpu','--cpu-diagnostic']
    def run(args):subprocess.run(common+args,cwd=ROOT,env=env,check=True,capture_output=True,timeout=120)
    run(['--out',str(tmp_path/'whole')]);run(['--out',str(tmp_path/'resume'),'--stop-after','3']);run(['--out',str(tmp_path/'resume'),'--resume'])
    a=torch.load(tmp_path/'whole/last.pt',weights_only=True);b=torch.load(tmp_path/'resume/last.pt',weights_only=True)
    for key in ('weights','optimizer','rng','tokens','stream','step','best'):assert tensors_equal(a[key],b[key]),key
    assert a['step']==6 and a['tokens']==768
    # Wrong budget/config changes may not silently alter the resumed schedule.
    wrong={**c,'learning_rate':.001}
    with pytest.raises(ValueError):train(wrong,data,tmp_path/'resume',resume=True,cpu_diagnostic=True)

def test_signal_or_requested_stop_saves_complete_update(tmp_path):
    data=fixtures(tmp_path/'data');out=tmp_path/'run';r=train(config(),data,out,stop_after=1,cpu_diagnostic=True)
    assert r['step']==1 and r['status']=='STOPPED_AT_UPDATE_BOUNDARY'
    a=torch.load(out/'last.pt',weights_only=True);assert a['tokens']==128

def test_existing_run_not_overwritten(tmp_path):
    data=fixtures(tmp_path/'data');out=tmp_path/'run';train(config(),data,out,stop_after=1,cpu_diagnostic=True)
    old=sha256(out/'last.pt')
    with pytest.raises(FileExistsError):train(config(),data,out,cpu_diagnostic=True)
    assert sha256(out/'last.pt')==old

def test_softmax_fast_path_gradient_equivalence():
    torch.manual_seed(22054)
    c=ScaleConfig(kind='softmax',vocab=256,width=32,layers=2,heads=2,slots=8,ff_width=64,chunk=32,cpu_scan=False)
    a=ScaleLM(c);b=ref.ScaleLM(ref.ScaleConfig(**vars(c)));b.load_state_dict(a.state_dict())
    x=torch.randint(256,(2,23));y=torch.randint(256,x.shape)
    for m in (a,b):
        logits,_=m(x);F.cross_entropy(logits.flatten(0,1),y.flatten()).backward()
    for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-4)
