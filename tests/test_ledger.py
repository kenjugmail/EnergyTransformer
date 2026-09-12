import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import pytest
import torch
from energy_transformer.ledger import EnergyLedger, pack
from energy_transformer.review import Edit

MODEL = 'a' * 64


def test_authorization_and_revision(tmp_path):
    s = EnergyLedger(tmp_path / 's.db', MODEL)
    with pytest.raises(PermissionError):
        s.commit([Edit('set','alice','garden')], expected_revision=0, request_id='x')
    s.commit([Edit('set','alice','garden')], expected_revision=0, request_id='x', authorized=True)
    with pytest.raises(ValueError, match='Stale'):
        s.commit([Edit('set','alice','office')], expected_revision=0, request_id='y', authorized=True)
    assert s.read()[1] == 1
    assert s.audit()['valid']


def test_retry_and_collision(tmp_path):
    s = EnergyLedger(tmp_path / 's.db', MODEL)
    kwargs = dict(expected_revision=0, request_id='same', authorized=True)
    first = s.commit([Edit('set','alice','garden')], **kwargs)
    second = s.commit([Edit('set','alice','garden')], **kwargs)
    assert not first['replayed'] and second['replayed']
    assert first['revision'] == second['revision'] == 1
    with pytest.raises(ValueError, match='different content'):
        s.commit([Edit('set','alice','office')], **kwargs)
    assert s.audit()['events'] == 1


def test_atomic_invalid_batch(tmp_path):
    s = EnergyLedger(tmp_path/'s.db', MODEL)
    before = pack(s.read()[0])
    with pytest.raises(ValueError):
        s.commit([Edit('set','alice','garden'),Edit('set','bob','invalid')],
                 expected_revision=0,request_id='x',authorized=True)
    assert pack(s.read()[0]) == before
    assert s.audit()['events'] == 0


def test_reopen_is_identical(tmp_path):
    path = tmp_path/'s.db';s=EnergyLedger(path, MODEL)
    s.commit([Edit('set','alice','garden'),Edit('set','bob','office')],expected_revision=0,request_id='x',authorized=True)
    restored=EnergyLedger(path, MODEL)
    assert pack(restored.read()[0]) == pack(s.read()[0])
    assert restored.audit() == s.audit()
    with pytest.raises(ValueError, match='mismatch'):
        EnergyLedger(path, 'b'*64)


def test_delete_and_nonmutating_copy(tmp_path):
    s=EnergyLedger(tmp_path/'s.db',MODEL)
    s.commit([Edit('set','alice','garden'),Edit('set','bob','office')],expected_revision=0,request_id='a',authorized=True)
    a,_=s.read(); original_bob=a.information[1].clone();a.information.zero_()
    assert s.read()[0].information.sum()==2
    s.commit([Edit('delete','alice')],expected_revision=1,request_id='b',authorized=True)
    a,_=s.read(); assert a.precision[0]==.25 and a.information[0].sum()==0
    assert torch.equal(a.information[1],original_bob)
    assert s.audit()['events']==2


def test_concurrent_compare_and_swap(tmp_path):
    path=tmp_path/'s.db';EnergyLedger(path,MODEL)
    def worker(i):
        store=EnergyLedger(path,MODEL)
        try:
            store.commit([Edit('set','alice','garden')],expected_revision=0,request_id=str(i),authorized=True)
            return 'committed'
        except ValueError as exc:
            assert 'Stale' in str(exc);return 'stale'
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results=list(pool.map(worker,range(8)))
    assert results.count('committed')==1 and results.count('stale')==7
    assert EnergyLedger(path,MODEL).audit()['events']==1


def test_journal_corruption_detected(tmp_path):
    path=tmp_path/'s.db';s=EnergyLedger(path,MODEL)
    s.commit([Edit('set','alice','garden')],expected_revision=0,request_id='x',authorized=True)
    with sqlite3.connect(path) as c:c.execute("UPDATE events SET hash=?",('0'*64,))
    with pytest.raises(ValueError, match='chain'):s.audit()


def test_state_corruption_detected(tmp_path):
    path=tmp_path/'s.db';s=EnergyLedger(path,MODEL)
    with sqlite3.connect(path) as c:c.execute("UPDATE state SET sha=?",('0'*64,))
    with pytest.raises(ValueError, match='integrity'):s.read()


def test_process_exit_rolls_back(tmp_path):
    path=tmp_path/'s.db';s=EnergyLedger(path,MODEL)
    code="import sqlite3,sys,os;c=sqlite3.connect(sys.argv[1]);c.execute('BEGIN IMMEDIATE');c.execute('UPDATE state SET revision=99');os._exit(17)"
    completed=subprocess.run([sys.executable,'-c',code,str(path)])
    assert completed.returncode==17
    assert s.audit()['revision']==0


def test_fresh_process_reads_state(tmp_path):
    path=tmp_path/'s.db';s=EnergyLedger(path,MODEL)
    s.commit([Edit('set','alice','garden')],expected_revision=0,request_id='x',authorized=True)
    code="from energy_transformer.ledger import EnergyLedger;import sys,json;print(json.dumps(EnergyLedger(sys.argv[1],sys.argv[2]).audit()))"
    p=subprocess.run([sys.executable,'-c',code,str(path),MODEL],capture_output=True,text=True,check=True)
    assert json.loads(p.stdout)==s.audit()


@pytest.mark.parametrize('request_id',['','a/b','a'*129])
def test_bad_request_identifiers(tmp_path,request_id):
    s=EnergyLedger(tmp_path/'s.db',MODEL)
    with pytest.raises(ValueError):s.commit([Edit('delete','alice')],expected_revision=0,request_id=request_id,authorized=True)


def test_multiple_processes_one_revision_winner(tmp_path):
    path=tmp_path/'s.db';EnergyLedger(path,MODEL)
    code="""from energy_transformer.ledger import EnergyLedger
from energy_transformer.review import Edit
import sys
s=EnergyLedger(sys.argv[1],sys.argv[2])
try:
 s.commit([Edit('set','alice','garden')],expected_revision=0,request_id=sys.argv[3],authorized=True)
 print('committed')
except ValueError as e:
 if 'Stale' not in str(e): raise
 print('stale')
"""
    processes=[subprocess.Popen([sys.executable,'-c',code,str(path),MODEL,str(i)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for i in range(4)]
    outcomes=[]
    for p in processes:
        out,err=p.communicate(timeout=30);assert p.returncode==0,err;outcomes.append(out.strip())
    assert outcomes.count('committed')==1 and outcomes.count('stale')==3
    assert EnergyLedger(path,MODEL).audit()['revision']==1
