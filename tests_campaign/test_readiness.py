import json
import signal
import subprocess
from pathlib import Path
import pytest
import torch
from energy_campaign import runner
from energy_campaign.spec import DATA,MODEL,source_identity
from energy_real.runtime import sha256


def test_resource_monitor_ignores_only_own_process_group(monkeypatch):
    monkeypatch.setattr(runner,'available_host_bytes',lambda:20*2**30)
    monkeypatch.setattr(runner,'gpu_processes',lambda:[{'pid':11},{'pid':12}])
    monkeypatch.setattr(runner.os,'getpgid',lambda pid:99)
    assert runner.resource_violation(99) is None
    monkeypatch.setattr(runner.os,'getpgid',lambda pid:99 if pid==11 else 100)
    assert runner.resource_violation(99)=='foreign_gpu_process_detected'
    monkeypatch.setattr(runner,'available_host_bytes',lambda:15*2**30)
    assert runner.resource_violation(99)=='host_memory_reserve_exhausted'


@pytest.mark.parametrize('reason',['host_memory_reserve_exhausted','foreign_gpu_process_detected'])
def test_supervisor_stops_only_our_group(monkeypatch,reason):
    signals=[]
    class Child:
        pid=991;calls=0
        def wait(self,timeout=None):
            self.calls+=1
            if self.calls==1:raise subprocess.TimeoutExpired('fixture',timeout)
            return 0
        def poll(self):return None
    monkeypatch.setattr(runner,'resource_violation',lambda pid:reason)
    monkeypatch.setattr(runner.os,'killpg',lambda pid,sig:signals.append((pid,sig)))
    result=runner.supervise(Child(),60,runner.time.monotonic())
    assert result==('interrupted',reason) and signals==[(991,signal.SIGTERM)]


def test_gpu_identity_never_initializes_cuda(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Supervisor initialized CUDA')
    monkeypatch.setattr(torch.cuda,'get_device_name',forbidden)
    monkeypatch.setattr(runner.subprocess,'run',lambda *a,**k:subprocess.CompletedProcess(a,0,'NVIDIA GB10, UUID, 580\n',''))
    assert runner.gpu_identity()=={'name':'NVIDIA GB10','uuid':'UUID','driver':'580'}


def test_profile_matches_accumulation_and_rejects_wrong_compilation(monkeypatch):
    from energy_campaign.profile import profile_one
    from energy_real.model import Config
    cfg=Config(width=16,heads=2,slots=4,layers=1,ff_width=32,chunk=8)
    row=profile_one(cfg,steps=1,batch=1,length=8,effective_targets=16)
    assert row['accumulation']==2 and row['effective_targets']==16 and row['parity']['status']=='passed'
    assert row['seconds_per_target']==row['update_seconds'][0]/16
    monkeypatch.setattr(torch,'compile',lambda fn:lambda x,y:fn(x,y)*2)
    with pytest.raises(AssertionError,match='parity'):profile_one(cfg,compile_model=True,steps=1,length=8,effective_targets=16)


def test_followon_is_independent_complete_and_unfunded():
    from energy_campaign.design import design
    p=design()
    assert len(p['scaling_runs'])==96 and len(p['slot_capacity_runs'])==27
    assert not p['execution_authorized'] and p['cost_estimate'] is None
    assert all(r['targets'] is None for r in p['scaling_runs'])
    rows=p['slot_capacity_runs'];counts=[r['parameters'] for r in rows]
    assert max(counts)/min(counts)-1<.005
    assert {r['config']['slots'] for r in rows if r['family']=='energy'}=={8,16,32,64}
    fixed=design(2**18,'hybrid')
    assert {r['targets'] for r in fixed['scaling_runs']}=={2**16,2**17,2**18,2**19}
    assert all(r['independent_run'] for r in fixed['scaling_runs'])


def preparation(root):
    def put(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    put(root/'spec.json',{'source_sha256':source_identity()})
    put(root/'cpu_verification.json',{'source_sha256':source_identity(),'status':'passed'})
    m={'source':DATA,'bytes':3,'documents':{},'shards':{}}
    for split,raw in zip(('train','valid','test'),(b'a',b'b',b'c')):
        p=root/'fineweb'/f'{split}.bin';p.parent.mkdir(exist_ok=True);p.write_bytes(raw)
        m['shards'][split]=sha256(p);m['documents'][split]=[{'offset':0,'bytes':1,'sha256':sha256(p),'group':split}]
    put(root/'fineweb/manifest.json',m)
    cfg={'num_hidden_layers':30,'hidden_size':576,'num_attention_heads':9,'num_key_value_heads':3,'max_position_embeddings':8192}
    for name in ('config.json','model.safetensors','tokenizer.json','tokenizer_config.json'):put(root/'donor'/name,cfg if name=='config.json' else {})
    put(root/'donor_manifest.json',{'model':MODEL,'files':{p.name:{'sha256':sha256(p),'bytes':p.stat().st_size} for p in (root/'donor').iterdir()}})


def test_readiness_rejects_changed_donor_and_cross_split_group(tmp_path):
    from energy_campaign.readiness import audit
    preparation(tmp_path);assert audit(tmp_path)['status']=='passed'
    (tmp_path/'donor/model.safetensors').write_bytes(b'tampered')
    assert audit(tmp_path)['checks'][2]['status']=='failed'
    p=tmp_path/'fineweb/manifest.json';m=json.loads(p.read_text());m['documents']['valid'][0]['group']='train';p.write_text(json.dumps(m))
    assert audit(tmp_path)['checks'][1]['status']=='failed'


def test_scheduler_rejects_unmatched_or_unverified_profiles(tmp_path):
    from energy_campaign.schedule import profiles
    from energy_campaign.spec import FAMILIES
    rows=[{'family':f,'size':i,'effective_targets':8192,'batch':1,'length':512,
           'parity':{'status':'passed'},'seconds_per_target':.01} for f in FAMILIES for i in range(4)]
    p=tmp_path/'profile.json'
    def save():p.write_text(json.dumps({'source_sha256':source_identity(),'runtime':{'device':'cuda'},'rows':rows}))
    save();assert len(profiles(p))==16
    rows[0]['effective_targets']=512;save()
    with pytest.raises(ValueError,match='Complete'):profiles(p)
    rows[0]['effective_targets']=8192;rows[0]['parity']['status']='failed';save()
    with pytest.raises(ValueError,match='Complete'):profiles(p)


def test_scaling_fit_rejects_mixed_architectures_and_reused_trajectories():
    from energy_campaign.analysis import fit_scaling
    row={'metric':'nll','independent_run':True,'family':'energy','parameters':1000,'targets':1000,'seed':1,'loss':2.,'trajectory_id':'a'}
    with pytest.raises(ValueError,match='architecture'):fit_scaling([row,{**row,'family':'softmax'}])
    with pytest.raises(ValueError,match='Duplicate'):fit_scaling([row,row])
    with pytest.raises(ValueError,match='trajectory'):fit_scaling([row,{**row,'targets':2000}])
    assert fit_scaling([row])['status']=='unresolved'


def test_changed_driver_invalidates_gpu_receipt_without_cuda_init(tmp_path,monkeypatch):
    import platform
    identity={'name':'NVIDIA GB10','uuid':'fixed','driver':'old'}
    receipt={'status':'NATIVE_GPU_ACCEPTANCE_PASS','source_sha256':source_identity(),
             'runtime':{'torch':str(torch.__version__),'cuda':torch.version.cuda,'machine':platform.machine(),'gpu':'NVIDIA GB10'},
             'gpu_identity':identity,'compiled':[{'status':'passed'}],'restart':{'bitwise_weights_optimizer_rng_samples':True}}
    (tmp_path/'accept.json').write_text(json.dumps(receipt))
    monkeypatch.setattr(runner,'gpu_identity',lambda:{**identity,'driver':'new'})
    def forbidden(*args,**kwargs):raise AssertionError('Initialized CUDA')
    monkeypatch.setattr(torch.cuda,'get_device_name',forbidden)
    with pytest.raises(ValueError,match='acceptance'):runner.validate_acceptance(tmp_path,{'action':'train'})
