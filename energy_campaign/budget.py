"""Crash-conservative POSIX campaign ledger. Reservations count until settled."""
from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path
import time
import uuid
from energy_real.runtime import atomic_json
from .spec import STAGES


class Ledger:
    def __init__(self,path,limits=None):
        self.path=Path(path); self.limits=dict(STAGES if limits is None else limits)
        self.path.parent.mkdir(parents=True,exist_ok=True)

    @contextmanager
    def locked(self):
        with self.path.with_suffix('.lock').open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            d=json.loads(self.path.read_text()) if self.path.exists() else {'limits':self.limits,'entries':[]}
            if d['limits']!=self.limits:raise ValueError('Budget limits changed')
            yield d

    def remaining(self,stage):
        with self.locked() as d:return self.limits[stage]-sum(e['charged'] for e in d['entries'] if e['stage']==stage)

    def reserve(self,stage,seconds,identity):
        if stage not in self.limits or not math.isfinite(seconds) or seconds<=0:raise ValueError('Invalid reservation')
        with self.locked() as d:
            # One supervisor owns GPU campaign execution. Unsettled entries need
            # explicit audit, not automatic refunds after a crash.
            if any(e['status']=='running' for e in d['entries']):raise RuntimeError('Unsettled campaign reservation')
            phase=identity.get('job',{}).get('phase') if isinstance(identity.get('job'),dict) else None
            caps={'calibration':3600,'matched_data':32400,'matched_time':7200,
                  'heads':3600,'joint':9000,'forecast_evaluate':1800,
                  'transfer':3600,'recovery':3600,'lora':3600,'conversion_evaluate':3600}
            if phase in caps:
                charged=sum(e['charged'] for e in d['entries'] if e['identity'].get('job',{}).get('phase')==phase)
                if charged+seconds>caps[phase]:raise RuntimeError('Substage budget exhausted')
            used=sum(e['charged'] for e in d['entries'] if e['stage']==stage)
            if used+seconds>self.limits[stage]:raise RuntimeError('Stage budget exhausted')
            key=uuid.uuid4().hex
            d['entries'].append({'id':key,'stage':stage,'reserved':seconds,'charged':seconds,'status':'running',
                                 'identity':identity,'started_unix':time.time()})
            atomic_json(d,self.path)
            return key

    def settle(self,key,elapsed,status):
        if not math.isfinite(elapsed) or elapsed<0 or status not in ('completed','failed','interrupted','timeout'):
            raise ValueError('Invalid settlement')
        with self.locked() as d:
            e=next(e for e in d['entries'] if e['id']==key)
            if e['status']!='running':raise ValueError('Reservation already settled')
            e.update(charged=elapsed,status=status,finished_unix=time.time(),overrun=elapsed>e['reserved'])
            atomic_json(d,self.path)
