import hashlib,json,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from energy_spark.common import atomic_json,source_identity,environment,sha256,ROOT
from energy_spark.data import prepare,fixtures,ByteCorpus
from energy_spark.train import reconcile_step_log,require_acceptance,validate_config
from energy_spark.doctor import check


def test_stream_import_matches_original_order_and_bytes(tmp_path):
    rows=[]
    for i in range(20):
        rows.append({'text':('Document '+str(i)+' αβ🙂\n')*128,'document_id':str(i),'source':'test','license':'generated'})
    rows.append({**rows[3],'document_id':'duplicate-content'})
    src=tmp_path/'in.jsonl';src.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    out=tmp_path/'corpus';m=prepare(src,out,min_bytes=64)
    unique={}
    for r in rows:
        raw=r['text'].encode();unique.setdefault(hashlib.sha256(raw).hexdigest(),raw)
    keys=sorted(unique,key=lambda h:hashlib.sha256(f'22001:{h}'.encode()).digest())
    expected={'train':keys[:16],'validation':keys[16:18],'test':keys[18:]}
    for name,hashes in expected.items():
        assert (out/f'{name}.bin').read_bytes()==b''.join(unique[h] for h in hashes)
        assert np.array_equal(np.load(out/f'{name}_offsets.npy'),np.cumsum([0]+[len(unique[h]) for h in hashes]))
    assert m['source_sha256']==sha256(src)
    assert not list(out.glob('_prepare*'))


def test_import_failure_leaves_no_partial_dataset(tmp_path):
    src=tmp_path/'broken.jsonl';src.write_text('{"bad": true}\n')
    with pytest.raises(ValueError):prepare(src,tmp_path/'corpus')
    assert not (tmp_path/'corpus').exists() and not list(tmp_path.glob('corpus.*'))


def test_duplicate_short_id_rejected(tmp_path):
    row={'text':'x','document_id':'x','source':'test','license':'generated'}
    src=tmp_path/'in.jsonl';src.write_text((json.dumps(row)+'\n')*2)
    with pytest.raises(ValueError,match='Repeated'):prepare(src,tmp_path/'out')


@pytest.mark.parametrize('value',[0,1,-1,True,1.5])
def test_invalid_minimum(value,tmp_path):
    with pytest.raises(ValueError):prepare(tmp_path/'missing',tmp_path/'out',min_bytes=value)


def test_resume_log_trims_and_retains_crash_tail(tmp_path):
    lines=[json.dumps({'step':i,'loss':1/i})+'\n' for i in range(1,5)]
    (tmp_path/'steps.jsonl').write_text(''.join(lines)+'{"step":5')
    result=reconcile_step_log(tmp_path,2)
    assert result=={'retained':2,'recovered_tail':True}
    assert (tmp_path/'steps.jsonl').read_text()==''.join(lines[:2])
    backup=list(tmp_path.glob('recovered_steps.*.jsonl'))
    assert len(backup)==1 and backup[0].read_text()==''.join(lines[2:])+'{"step":5'
    assert reconcile_step_log(tmp_path,2)['recovered_tail'] is False


def test_committed_log_corruption_is_not_hidden(tmp_path):
    p=tmp_path/'steps.jsonl';p.write_text('{"step":1}\nnot json\n{"step":3}\n')
    with pytest.raises(ValueError,match='committed'):reconcile_step_log(tmp_path,3)
    assert 'not json' in p.read_text()


@pytest.mark.parametrize('timestamp',[float('nan'),float('inf'),True,None,'tomorrow',time.time()+3600,time.time()-8*86400])
def test_bad_receipt_time_rejected(timestamp,tmp_path):
    r={'status':'PASS_FOR_BOUNDED_DEVICE_PILOT','source_sha256':source_identity(),'checkpoint_sha256':None,
       'environment':environment('cpu'),'completed_unix':timestamp,'fit_configurations':[]}
    p=tmp_path/'receipt.json';p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='timestamp|future-dated'):require_acceptance(p,'cpu')


@pytest.mark.parametrize('field,value',[('heads',3),('slots',10000),('ff_width',10000),('microbatch',17),('accumulation',65),('chunk',1025)])
def test_dimensions_bounded_before_allocation(field,value):
    c=json.loads((ROOT/'configs/v22/cpu_test.json').read_text());c[field]=value
    with pytest.raises(ValueError):validate_config(c)


def test_shard_symlink_rejected(tmp_path):
    root=fixtures(tmp_path/'fixture');shard=root/'train.bin';other=tmp_path/'data.bin'
    shard.replace(other);shard.symlink_to(other)
    with pytest.raises(ValueError,match='Unsafe'):ByteCorpus(root,'train')


def test_doctor_does_not_require_torch_or_mutate(tmp_path):
    with patch('energy_spark.doctor.shutil.which',return_value=None):r=check(tmp_path)
    assert r['status']=='HOST_ACTION_NEEDED'
    assert 'missing_or_mismatched_original_checkpoint' in r['artifact_issues']
    assert not list(tmp_path.iterdir())
