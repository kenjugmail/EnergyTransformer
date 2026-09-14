from pathlib import Path
from types import SimpleNamespace
import sys
from energy_real import stages


def args(tmp_path):
    return SimpleNamespace(out=tmp_path/'stages',stages=['smoke'],wait_idle=False,max_wait_seconds=0)


def test_pending_gpu_retries_without_launching_over_busy_job(tmp_path,monkeypatch):
    monkeypatch.setattr(stages,'source_identity',lambda:'fixture')
    def busy():raise RuntimeError('existing job')
    monkeypatch.setattr(stages,'require_idle_gpu',busy)
    a=args(tmp_path)
    first=stages.run(a)
    assert first['stages']['smoke']['status']=='pending_gpu'
    assert first['stages']['smoke']['used_s']==0
    monkeypatch.setattr(stages,'require_idle_gpu',lambda:None)
    script='import pathlib,sys;pathlib.Path(sys.argv[-1]).write_text(\'{"status":"passed"}\')'
    monkeypatch.setattr(stages,'commands',lambda name,out,remaining:[[sys.executable,'-c',script,'--out',str(out/'fixture.json')]])
    result=stages.run(a)
    assert result['stages']['smoke']['status']=='completed'
    used=result['stages']['smoke']['used_s']
    assert stages.run(a)['stages']['smoke']['used_s']==used


def test_stage_time_ceiling_stops_only_its_child(tmp_path,monkeypatch):
    monkeypatch.setattr(stages,'source_identity',lambda:'fixture')
    monkeypatch.setattr(stages,'require_idle_gpu',lambda:None)
    monkeypatch.setitem(stages.LIMITS,'smoke',.05)
    monkeypatch.setattr(stages,'commands',lambda name,out,remaining:[[sys.executable,'-c','import time;time.sleep(5)','--out',str(out/'never.json')]])
    result=stages.run(args(tmp_path))['stages']['smoke']
    assert result['status']=='budget_exhausted' and result['returncode']==124
    assert not list(tmp_path.rglob('never.json'))


import json
import os
import signal
import subprocess
import time
import pytest


@pytest.mark.parametrize('body', [None, 'not JSON', '[]', '{}', '{"status": 1}'])
def test_success_exit_requires_a_valid_report(tmp_path, monkeypatch, body):
    monkeypatch.setattr(stages, 'source_identity', lambda: 'fixture')
    monkeypatch.setattr(stages, 'require_idle_gpu', lambda: None)
    script = 'pass' if body is None else f'import pathlib,sys;pathlib.Path(sys.argv[-1]).write_text({body!r})'
    monkeypatch.setattr(stages, 'commands', lambda name,out,remaining: [
        [sys.executable, '-c', script, '--out', str(out/'report.json')]])
    result = stages.run(args(tmp_path))['stages']['smoke']
    assert result['status'] == 'failed'
    assert result['completed_commands'] == []


def test_existing_report_cannot_supply_false_success(tmp_path, monkeypatch):
    monkeypatch.setattr(stages, 'source_identity', lambda: 'fixture')
    monkeypatch.setattr(stages, 'require_idle_gpu', lambda: None)
    a = args(tmp_path)
    output = a.out/'smoke/attempt_0/report.json'
    output.parent.mkdir(parents=True)
    output.write_text('{"status":"passed"}')
    monkeypatch.setattr(stages, 'commands', lambda name,out,remaining: [
        [sys.executable, '-c', 'raise AssertionError("should not launch")', '--out', str(output)]])
    result = stages.run(a)['stages']['smoke']
    assert result['status'] == 'failed' and result['used_s'] == 0
    assert 'pre-existing' in result['reason']


def test_launch_error_charges_budget_and_records_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(stages, 'source_identity', lambda: 'fixture')
    monkeypatch.setattr(stages, 'require_idle_gpu', lambda: None)
    monkeypatch.setattr(stages, 'commands', lambda name,out,remaining: [
        [str(tmp_path/'missing-executable'), '--out', str(out/'report.json')]])
    a = args(tmp_path)
    with pytest.raises(FileNotFoundError):
        stages.run(a)
    record = json.loads((a.out/'stages.json').read_text())['stages']['smoke']
    assert record['status'] == 'failed' and record['used_s'] > 0
    assert 'pid' not in record and 'started_unix' not in record


def test_sigterm_reaps_owned_child_and_persists_interruption(tmp_path):
    output = tmp_path/'stages'
    child_script = 'import pathlib,os,sys,time;pathlib.Path(sys.argv[-1]+".pid").write_text(str(os.getpid()));time.sleep(30)'
    wrapper = f'''
from types import SimpleNamespace
from pathlib import Path
import sys
from energy_real import stages
stages.require_idle_gpu = lambda: None
stages.commands = lambda name,out,remaining: [[sys.executable,'-c',{child_script!r},'--out',str(out/'report.json')]]
stages.run(SimpleNamespace(out=Path({str(output)!r}),stages=['smoke'],wait_idle=False,max_wait_seconds=0))
'''
    process = subprocess.Popen([sys.executable, '-c', wrapper], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    marker = output/'smoke/attempt_0/report.json.pid'
    child_pid = None
    try:
        deadline = time.monotonic()+15
        while not marker.exists() and process.poll() is None and time.monotonic()<deadline:
            time.sleep(.02)
        assert marker.exists(), process.communicate(timeout=1)[1].decode()
        child_pid = int(marker.read_text())
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=8)
        assert process.returncode != 0
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        record = json.loads((output/'stages.json').read_text())['stages']['smoke']
        assert record['status'] == 'interrupted' and record['used_s'] > 0
        assert record['completed_commands'] == [] and 'pid' not in record
    finally:
        if process.poll() is None:
            process.kill(); process.communicate()
        if child_pid is not None:
            try: os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError: pass


def test_reasoning_stage_uses_configured_service_url(tmp_path,monkeypatch):
    from energy_real import service
    seen=[]
    class FixtureModel:
        def __init__(self,base_url,timeout):seen.append(base_url)
        def health(self):return {'model':'fixture'}
    monkeypatch.setattr(service,'LocalModel',FixtureModel)
    monkeypatch.setattr(stages,'source_identity',lambda:'fixture')
    monkeypatch.setattr(stages,'gpu_processes',lambda:[])
    script='import json,pathlib,sys;pathlib.Path(sys.argv[sys.argv.index("--out")+1]).write_text(json.dumps({"status":"completed","url":sys.argv[sys.argv.index("--base-url")+1]}))'
    monkeypatch.setattr(stages,'commands',lambda name,out,remaining:[
        [sys.executable,'-c',script,'--out',str(out/'reasoning_fixture.json')]])
    a=args(tmp_path);a.stages=['reasoning'];a.base_url='http://127.0.0.1:9999'
    result=stages.run(a)['stages']['reasoning']
    assert result['status']=='completed' and seen==[a.base_url]
    report=json.loads((a.out/'reasoning/attempt_0/reasoning_fixture.json').read_text())
    assert report['url']==a.base_url


def test_new_source_ledger_inherits_previous_usage(tmp_path,monkeypatch):
    monkeypatch.setattr(stages,'source_identity',lambda:'new_source')
    monkeypatch.setattr(stages,'require_idle_gpu',lambda:None)
    previous=tmp_path/'old.json'
    previous.write_text(json.dumps({'source_sha256':'old_source','stages':{
        'smoke':{'used_s':1800,'status':'completed'}}}))
    a=args(tmp_path);a.prior_ledger=[previous,previous]
    result=stages.run(a)
    record=result['stages']['smoke']
    assert record['used_s']==record['inherited_used_s']==1800
    assert record['status']=='budget_exhausted' and record['completed_commands']==[]
    assert len(result['prior_ledgers'])==1
