import json
from types import SimpleNamespace
import pytest
import torch
from energy_real.bench import balanced_order,run as benchmark,parser as bench_parser
from energy_real.memory_research import batch,objective,read_logits
from energy_transformer.geometric_memory import EditMemory
from energy_real.evaluation import run as evaluation
from energy_real.runtime import atomic_checkpoint,sha256
from energy_real.model import LM,Config
from energy_real import data


def test_balanced_round_positions():
    arms=['e','s','r']
    rows=[balanced_order(arms,i) for i in range(6)]
    for position in range(3):
        assert all(sum(row[position]==arm for row in rows)==2 for arm in arms)


def test_cpu_benchmark_records_real_next_token_work(tmp_path):
    args=bench_parser().parse_args(['--out',str(tmp_path/'bench.json'),'--device','cpu','--width','16',
        '--layers','1','--heads','2','--slots','4','--ff-width','32','--contexts','8','--batches','1',
        '--rounds','2','--steps','1','--warmup','1','--new-tokens','2'])
    result=benchmark(args)
    assert result['status']=='completed' and len(result['rounds'])==12
    for row in result['rounds']:
        assert row['status']=='completed' and row['tokens_per_second']>0
        assert row['tokens']==(2 if row['mode']=='decode' else 8)
        assert row['peak_allocated_bytes'] is None


def test_controller_objectives_and_projected_equivalence():
    torch.manual_seed(7)
    a=EditMemory('projected',16,16)
    b=EditMemory('preconditioned_delta',16,16);b.load_state_dict(a.state_dict())
    inputs=batch(19,batch_size=2)
    ya,_,_=a(inputs.keys,inputs.values,inputs.operations,inputs.queries)
    yb,_,_=b(inputs.keys,inputs.values,inputs.operations,inputs.queries)
    torch.testing.assert_close(ya,yb,atol=1e-3,rtol=1e-3)
    loss,parts=objective(a,inputs,True)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in a.parameters())
    assert parts['delayed_ce']>0 and parts['locality_mse']>=0
    assert torch.equal(read_logits(torch.zeros(2,16)).argmax(-1),torch.zeros(2,dtype=torch.long))


def test_test_split_requires_matching_frozen_selection(tmp_path,monkeypatch):
    monkeypatch.setattr(data,'EXPECTED_BYTES',128)
    monkeypatch.setattr(data,'SPLITS',{'train':(0,64),'valid':(64,96),'test':(96,128)})
    corpus=tmp_path/'corpus';corpus.write_bytes(bytes(range(128)))
    model=LM(Config(width=16,layers=1,heads=2,slots=4,ff_width=32))
    checkpoint=tmp_path/'model.pt'
    atomic_checkpoint({'config':vars(model.config),'weights':model.state_dict(),'step':2,'tokens':16},checkpoint)
    args=SimpleNamespace(out=tmp_path/'valid.json',device='cpu',data=corpus,split='valid',
        data_sha256=sha256(corpus),checkpoints=[checkpoint],seq=8,backend='dmat',amp='none',
        selection=None,freeze_selection=True,max_seconds=10,max_bytes=None,batch=2,max_cuda_gib=1)
    first=evaluation(args)
    assert first['results'][0]['targets']==31
    args.out=tmp_path/'test.json';args.split='test';args.freeze_selection=False
    with pytest.raises(ValueError,match='frozen'):evaluation(args)
    args.selection=tmp_path/'valid.selection.json'
    result=evaluation(args)
    assert result['results'][0]['split']=='test' and result['results'][0]['targets']==31
    assert result['results'][0]['planned_targets']==31
    for field,value in [('seq',4),('batch',1),('max_bytes',16)]:
        args.out=tmp_path/(field+'_bad.json')
        previous=getattr(args,field);setattr(args,field,value)
        with pytest.raises(ValueError,match='protocol'):evaluation(args)
        assert not args.out.exists()
        setattr(args,field,previous)
    (tmp_path/'valid.json').write_text('{}')
    args.out=tmp_path/'tampered.json'
    with pytest.raises(ValueError,match='validation report'):evaluation(args)
    assert not args.out.exists()
