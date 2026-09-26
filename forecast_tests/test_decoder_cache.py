from dataclasses import asdict
from pathlib import Path
import pytest,torch
from energy_real.model import LM,Config
from energy_forecast.decoder_cache import build,validate_blocks

def fixture(tmp_path):
    torch.manual_seed(181);c=Config(width=16,layers=1,heads=2,slots=4,ff_width=32,chunk=8);m=LM(c)
    model=tmp_path/'model.pt';torch.save({'config':asdict(c),'weights':m.state_dict()},model)
    corpus=tmp_path/'bytes';corpus.write_bytes(bytes(range(96)))
    blocks=[{'split':split,'offset':32*i,'bytes':32} for i,split in enumerate(('train','validation','test'))]
    return model,corpus,blocks

def test_resumed_shards_equal_uninterrupted(tmp_path):
    teacher,corpus,blocks=fixture(tmp_path)
    partial=build(teacher,corpus,blocks,tmp_path/'a',tmp_path/'a.pt',batch=1,stop_after_shards=1)
    assert partial['status']=='partial' and not (tmp_path/'a.pt').exists()
    resumed=build(teacher,corpus,blocks,tmp_path/'a',tmp_path/'a.pt',batch=1,resume=True)
    assert resumed['reused_shards']==1
    build(teacher,corpus,blocks,tmp_path/'b',tmp_path/'b.pt',batch=1)
    a=torch.load(tmp_path/'a.pt',weights_only=True);b=torch.load(tmp_path/'b.pt',weights_only=True)
    for split in ('train','validation','test'):
        for key in a[split]:assert torch.equal(a[split][key],b[split][key])

def test_altered_shard_rejected(tmp_path):
    teacher,corpus,blocks=fixture(tmp_path)
    build(teacher,corpus,blocks,tmp_path/'a',tmp_path/'a.pt',batch=1,stop_after_shards=1)
    path=tmp_path/'a/000000.pt';d=torch.load(path,weights_only=True);d['records'][0]['h'][0,0]+=1;torch.save(d,path)
    with pytest.raises(ValueError):build(teacher,corpus,blocks,tmp_path/'a',tmp_path/'a.pt',batch=1,resume=True)

def test_changed_batch_rejected(tmp_path):
    teacher,corpus,blocks=fixture(tmp_path)
    build(teacher,corpus,blocks,tmp_path/'a',tmp_path/'a.pt',batch=1,stop_after_shards=1)
    with pytest.raises(ValueError):build(teacher,corpus,blocks,tmp_path/'a',tmp_path/'a.pt',batch=2,resume=True)

def test_overlapping_ranges_rejected():
    blocks=[{'split':s,'offset':i,'bytes':30} for i,s in enumerate(('train','validation','test'))]
    with pytest.raises(ValueError):validate_blocks(blocks,96)
