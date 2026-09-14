"""Durable successful service calls and strict evaluation identity checks."""
import hashlib
import json
import time
from .runtime import atomic_json


def model_identity(health):
    identity = {k:health[k] for k in ('model','model_id','model_sha256','checkpoint_sha256','revision','decode_mode','adapters','passes','checkpoint','weights_sha256','dtype') if k in health}
    if not identity:
        raise ValueError('Service health must expose a stable model identity for resumable evaluation')
    return identity


class RecordedService:
    def __init__(self, service, state, path, started, used):
        self.service, self.state, self.path = service, state, path
        self.last_elapsed_s = 0.
        self.started, self.used = started, used
        self.scope = None

    def __getattr__(self, method):
        def call(*args, **kwargs):
            key = hashlib.sha256(json.dumps([self.scope,method,args,kwargs],sort_keys=True,allow_nan=False).encode()).hexdigest()
            calls = self.state['service_calls']
            if key not in calls:
                start = time.perf_counter()
                try:
                    value = getattr(self.service,method)(*args,**kwargs)
                    calls[key] = {'value':value, 'elapsed_s':time.perf_counter()-start}
                except ValueError as exc:
                    calls[key] = {'error':str(exc), 'elapsed_s':time.perf_counter()-start}
                finally:
                    self.state['elapsed_s']=self.used+time.perf_counter()-self.started
                    atomic_json(self.state,self.path)
            self.last_elapsed_s = calls[key]['elapsed_s']
            if 'error' in calls[key]:raise ValueError(calls[key]['error'])
            return calls[key]['value']
        return call
