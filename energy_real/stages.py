"""Run the approved local stages with persistent time ceilings and idle-GPU checks.

Pending stages are resumable. This command never stops an unrelated process,
starts cloud work, or expands a stage budget. Use --wait-idle to wait explicitly.
"""
from __future__ import annotations
import argparse
import json
import math
from contextlib import contextmanager
import threading
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .device import gpu_processes,require_idle_gpu
from .runtime import atomic_json,source_identity

LIMITS={'smoke':1800,'evaluation_profile':5400,'controllers':14400,'reasoning':7200}
ROOT=Path(__file__).resolve().parents[1]


def commands(name,out,remaining):
    python=sys.executable
    if name=='smoke':
        return [[python,'-m','energy_real.device_smoke','--out',str(out/'smoke.json'),'--max-seconds',str(remaining)]]
    if name=='evaluation_profile':
        checkpoints=[str(ROOT/'runs/enwik8_energy_10m/best.pt'),str(ROOT/'runs/enwik8_softmax_10m/best.pt')]
        return [[python,'-m','energy_real.evaluation',*checkpoints,'--data',str(ROOT/'data/enwik8'),
                 '--out',str(out/'validation.json'),'--freeze-selection','--max-seconds',str(min(1200,remaining))],
                [python,'-m','energy_real.evaluation',*checkpoints,'--data',str(ROOT/'data/enwik8'),
                 '--out',str(out/'test.json'),'--split','test','--selection',str(out/'validation.selection.json'),
                 '--max-seconds',str(min(1200,remaining))],
                [python,'-m','energy_real.bench','--out',str(out/'bench.json'),'--include-reference','--profile',
                 '--max-seconds',str(min(3000,remaining))]]
    if name=='controllers':
        return [[python,'-m','energy_real.recall_study','--out',str(out/'study'),
                 '--steps','500','--max-seconds',str(remaining)]]
    return [[python,'-m','energy_real.reasoning','evaluate','--out',str(out/f'reasoning_{mode}_{question}.json'),
             '--benchmark','v2','--question-mode',question,'--input-mode',mode,'--gold-evidence',
             '--score','--max-seconds',str(min(1800,remaining))]
            for mode in ('oracle_structured','model_proposals') for question in ('supplied_path','natural')]


class StageInterrupted(KeyboardInterrupt):
    pass


@contextmanager
def interruption_handler():
    """Let the child cleanup path handle termination before restoring the caller."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    def stop(signum, frame):
        raise StageInterrupted(f'Stage runner received signal {signum}')
    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def stop_child(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def report_status(path):
    if path.is_dir():
        path = path / 'summary.json'
    try:
        report = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f'Missing or invalid stage report: {path}') from exc
    if not isinstance(report, dict) or not isinstance(report.get('status'), str):
        raise ValueError(f'Stage report has no status: {path}')
    return report['status']


def run(args):
    if not math.isfinite(args.max_wait_seconds) or args.max_wait_seconds < 0:
        raise ValueError('max_wait_seconds must be finite and nonnegative')
    args.out.mkdir(parents=True,exist_ok=True)
    ledger=args.out/'stages.json'
    # Separate nonblocking lock; repeated invocations may own pending stages.
    import fcntl
    with (args.out/'.stage_lock').open('a') as lock, interruption_handler():
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if ledger.exists():
            existing=json.loads(ledger.read_text())
        else:
            existing={'source_sha256':source_identity(),'stages':{},'prior_ledgers':[]}
            for prior in dict.fromkeys(Path(p).resolve() for p in getattr(args,'prior_ledger',[])):
                from .runtime import sha256
                previous=json.loads(prior.read_text())
                existing['prior_ledgers'].append({'path':str(prior),'sha256':sha256(prior)})
                for name,old in previous['stages'].items():
                    if name not in LIMITS:continue
                    used=old['used_s']
                    if not math.isfinite(used) or used<0:raise ValueError('Invalid prior stage usage')
                    record=existing['stages'].setdefault(name,{'limit_s':LIMITS[name],'used_s':0.,'completed_commands':[],'inherited_used_s':0.})
                    record['used_s']+=used;record['inherited_used_s']+=used
        if existing['source_sha256']!=source_identity():raise ValueError('Source changed; use a new stage directory')
        for name in args.stages:
            record=existing['stages'].setdefault(name,{'limit_s':LIMITS[name],'used_s':0.,'completed_commands':[]})
            if record.get('status')=='running':
                pid=record.get('pid')
                if pid is not None:
                    try:os.kill(pid,0)
                    except ProcessLookupError:pass
                    else:raise RuntimeError(f'Stage child {pid} is still running; do not launch a duplicate')
                record['used_s']=min(record['limit_s'],record['used_s']+max(0,time.time()-record.get('started_unix',time.time())))
                record.update(status='failed',reason='Previous stage owner interrupted; elapsed budget conservatively charged')
                atomic_json(existing,ledger)
            if record.get('status') not in (None,'pending_gpu','pending_service','ready'):continue
            previous_status=record.get('status')
            if name=='reasoning':
                from .service import LocalModel,ServiceUnavailable
                try:LocalModel(getattr(args,'base_url','http://127.0.0.1:8100'),timeout=3).health()
                except ServiceUnavailable as exc:
                    record.update(status='pending_service',reason=str(exc))
                    atomic_json(existing,ledger);continue
            if name not in ('controllers','reasoning'):
                waited=0.
                while True:
                    try:
                        require_idle_gpu()
                        record['status']='ready'
                        break
                    except (RuntimeError,subprocess.SubprocessError,OSError) as exc:
                        if not args.wait_idle or waited>=args.max_wait_seconds:
                            record.update(status='pending_gpu',reason=str(exc))
                            atomic_json(existing,ledger)
                            print(json.dumps({'stage':name,**record}),flush=True)
                            break
                        print(f'Waiting for the GPU; existing workload remains running ({name}).',flush=True)
                        duration=min(15,args.max_wait_seconds-waited);time.sleep(duration);waited+=duration
                if record.get('status')=='pending_gpu':continue
            if name=='reasoning':
                unrelated=[]
                for process in gpu_processes():
                    try:cmd=Path(f"/proc/{process['pid']}/cmdline").read_bytes().decode(errors='replace')
                    except OSError:cmd=''
                    if 'serve_looped.py' not in cmd:unrelated.append(process)
                if unrelated:
                    record.update(status='pending_gpu',reason=f'Unrelated GPU workload: {unrelated}')
                    atomic_json(existing,ledger);continue
            remaining=record['limit_s']-record['used_s']
            if remaining<=0:
                record['status']='budget_exhausted';atomic_json(existing,ledger);continue
            record.pop('reason',None);record['status']='running'
            stage_out=args.out/name/f"attempt_{record.get('attempt',0)}";stage_out.mkdir(parents=True,exist_ok=True)
            for index,command in enumerate(commands(name,stage_out,remaining)):
                if index in record['completed_commands']:continue
                if name=='reasoning':
                    command=[*command,'--base-url',getattr(args,'base_url','http://127.0.0.1:8100')]
                    if getattr(args,'memory_selection',None):
                        command += ['--memory-selection',str(args.memory_selection)]
                    if getattr(args,'artifact_root',None):
                        command += ['--artifact-root',str(args.artifact_root)]
                remaining=record['limit_s']-record['used_s']
                if remaining<=0:record['status']='budget_exhausted';break
                report_path=Path(command[command.index('--out')+1])
                if report_path.exists():
                    if name=='reasoning' and previous_status=='pending_service':
                        saved=json.loads(report_path.read_text())
                        command += ['--resume']
                        command[command.index('--max-seconds')+1]=str(saved['max_seconds'])
                    else:
                        record.update(status='failed',reason=f'Refusing a pre-existing command output: {report_path}')
                        break
                start=time.perf_counter()
                record['started_unix']=time.time();record['last_command']=command
                atomic_json(existing,ledger)
                process=None
                try:
                    with (stage_out/f'command_{index}.log').open('a') as log:
                        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                                                 env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
                        record['pid']=process.pid;atomic_json(existing,ledger)
                        try:
                            code=process.wait(timeout=remaining)
                        except subprocess.TimeoutExpired:
                            stop_child(process)
                            code=124
                except BaseException as exc:
                    if process is not None:
                        stop_child(process)
                    record.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                                  reason=f'{type(exc).__name__}: {exc}')
                    raise
                finally:
                    record['used_s']+=time.perf_counter()-start
                    record.pop('started_unix',None);record.pop('pid',None)
                    atomic_json(existing,ledger)
                if code!=0:
                    record['status']='budget_exhausted' if code==124 else 'failed'
                    record['returncode']=code;break
                try:
                    status=report_status(report_path)
                except ValueError as exc:
                    record.update(status='failed',reason=str(exc));break
                if status not in ('passed','completed'):
                    record.update(status=status,reason='Stage command produced an incomplete report')
                    break
                record['completed_commands'].append(index)
                atomic_json(existing,ledger)
            else:record['status']='completed'
            if name=='reasoning' and record['status']=='completed':
                reports=[json.loads(p.read_text()) for p in stage_out.glob('reasoning_*.json')]
                if any(r['status']!='completed' for r in reports):record['status']='pending_service'
            atomic_json(existing,ledger)
        return existing


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--stages',choices=list(LIMITS),nargs='+',default=['smoke','evaluation_profile'])
    p.add_argument('--base-url',default='http://127.0.0.1:8100')
    p.add_argument('--memory-selection',type=Path)
    p.add_argument('--artifact-root',type=Path)
    p.add_argument('--prior-ledger',type=Path,action='append',default=[],help='Conservatively carry previous stage usage into a new source revision')
    p.add_argument('--wait-idle',action='store_true')
    p.add_argument('--max-wait-seconds',type=float,default=300)
    a=p.parse_args();print(json.dumps(run(a),indent=2))
