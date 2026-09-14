from dataclasses import replace
import json
import pytest
from energy_real.episodic import FactEdit,FactStore,Query,EnergyRetriever
from energy_real.episodes import generate,oracle
from energy_real.memory_eval import run,paired_interval


def edit(subject='parcel',obj='locker',op='set'):
    return FactEdit(op,subject,'located_in',obj,f'Reviewed {op}: {subject} -> {obj}')


def test_review_revision_retry_correction_deletion_and_persistence(tmp_path):
    path=tmp_path/'facts.sqlite'
    store=FactStore(path)
    edits=[edit(),edit('locker','lobby'),edit('friend','park')]
    with pytest.raises(PermissionError):store.apply(edits,expected_revision=0,request_id='one')
    receipt=store.apply(edits,expected_revision=0,request_id='one',authorized=True)
    again=store.apply(edits,expected_revision=0,request_id='one',authorized=True)
    assert receipt['revision']==again['revision']==1 and again['replayed']
    with pytest.raises(ValueError,match='reused'):
        store.apply([edit('locker','office')],expected_revision=0,request_id='one',authorized=True)
    reopened=FactStore(path)
    assert reopened.query('Follow located_in/located_in from parcel.')['answer']=='lobby'
    with pytest.raises(ValueError,match='Stale'):
        reopened.apply([edit('locker','office')],expected_revision=0,request_id='two',authorized=True)
    reopened.apply([edit('locker','office')],expected_revision=1,request_id='two',authorized=True)
    result=store.query(Query('parcel',('located_in','located_in')))
    assert result['answer']=='office' and result['facts'][-1]['source_text'].endswith('office')
    assert store.query(Query('friend',('located_in',)))['answer']=='park'
    store.apply([edit('locker',None,'delete')],expected_revision=2,request_id='three',authorized=True)
    assert reopened.query('Follow located_in/located_in from parcel.')['abstained']
    assert reopened.query(Query('friend',('located_in',)))['answer']=='park'


def test_invalid_batch_applies_nothing(tmp_path):
    store=FactStore(tmp_path/'f.sqlite')
    with pytest.raises(ValueError):
        store.apply([edit(),edit('bad',None)],expected_revision=0,request_id='bad',authorized=True)
    assert store.revision==0 and store.snapshot()[2]==[]


def test_oracle_families_and_split_disjointness():
    train,valid,test=[generate(11,1,split) for split in ('train','valid','test')]
    assert not {e.query.subject for e in train}&{e.query.subject for e in test}
    for episode in test:
        answer,support=oracle(episode)
        if episode.family in ('delete','absent'):assert answer is None
        if episode.family=='two_hop':assert len(support)==2
        if episode.family=='three_hop':assert len(support)==3
    results=run(test,backends=('exact',))
    assert results['summary']['exact']['accuracy']==1
    assert results['summary']['exact']['support_accuracy']==1
    assert results['summary']['exact']['stale_facts']==0


def test_approximate_result_is_not_repaired_from_exact_store(tmp_path,monkeypatch):
    store=FactStore(tmp_path/'e.sqlite',backend='precision')
    store.apply([edit()],expected_revision=0,request_id='one',authorized=True)
    store.apply([edit('parcel','office')],expected_revision=1,request_id='two',authorized=True)
    old=store.snapshot()[2][0]
    monkeypatch.setattr(EnergyRetriever,'read',lambda *args:old)
    answer=store.query(Query('parcel',('located_in',)))
    assert answer['answer']=='locker' and answer['facts'][0]['revision']==1
    # The exact record still says office; no correction/filter hides retrieval error.
    assert store.snapshot()[1][0]['object']=='office'


def test_paired_interval_keeps_pairs():
    assert paired_interval([1,0,1],[1,0,1])['high']==0
    result=paired_interval([1]*20,[0]*20)
    assert result['difference']==result['low']==result['high']==1
