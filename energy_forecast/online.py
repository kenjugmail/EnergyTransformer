"""One bounded, fixed-origin forecast updated ONLY by observed teacher states.

Forecast parameters are frozen at issue time. Original target timestamps remain
fixed; at t+16 a target at t+64 has 48 bytes remaining, not another 64 bytes.
The optional gate produces an empirical mixture pool, not an exact posterior.
This session never injects predictions into the language model or skips tokens.
Single-stream/single-thread research API, not an authenticated evidence service.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import torch
from .conditioning import GaussianMixture, blend
from .session import ForecastSession
from .experiment import atomic_save


def _payload_digest(payload):
    h=hashlib.sha256(b'observed-forecast-session-v1\0')
    def visit(x):
        if isinstance(x,torch.Tensor):
            t=x.detach().cpu().contiguous();h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.numpy().tobytes())
        elif isinstance(x,dict):
            for key in sorted(x):h.update(key.encode());visit(x[key])
        elif isinstance(x,(tuple,list)):
            h.update(str(len(x)).encode())
            for v in x:visit(v)
        else:h.update(repr(x).encode())
    visit(payload);return h.hexdigest()


class ObservedForecastSession:
    def __init__(self,teacher,head,*,gate_path=None,device='cpu'):
        self._base=ForecastSession(teacher,head,device)
        self._ticket=None
        self._gates={}
        self._gate_hash=None
        if gate_path is not None:
            raw=Path(gate_path).read_bytes();gate=json.loads(raw)
            if gate.get('format')!='forecast_conditioning_gate_v1':raise ValueError('Unknown gate format')
            if self._base.head_hash not in gate.get('heads',{}).values() or gate.get('teacher_sha256')!=self._base.teacher_hash:
                raise ValueError('Gate belongs to different model weights')
            for key,value in gate.get('gates',{}).items():
                if str(int(key))!=key or int(key) not in self._base.head.config.horizons[:-1]:raise ValueError('Unsupported conditioning horizon')
                if type(value) not in (float,int) or not 0<=value<=1:raise ValueError('Invalid conditioning weight')
            self._gates={int(k):float(v) for k,v in gate['gates'].items()}
            self._gate_hash=hashlib.sha256(raw).hexdigest()

    @property
    def steps(self):return self._base.steps

    def reset(self):
        self._base.reset();self._ticket=None

    @torch.no_grad()
    def issue(self,*,replace=False):
        if self._ticket is not None and not replace:raise ValueError('One forecast is already tracked; explicitly replace or clear it')
        f=GaussianMixture.from_landscape(self._base.forecast()).cloned()
        self._ticket={'origin':self.steps,'distribution':f,'observations':{}}
        return self.status()

    def clear(self):self._ticket=None

    @torch.no_grad()
    def observe(self,tokens):
        # Validate the entire supplied batch before applying the first token.
        if not isinstance(tokens,torch.Tensor) or tokens.dtype!=torch.long or tokens.ndim!=2 or tokens.shape[0]!=1 or tokens.shape[1]<1:
            raise ValueError('Expected nonempty integer tokens [1,T]')
        if (tokens<0).any() or (tokens>=self._base.model.config.vocab).any():raise ValueError('Token outside vocabulary')
        logits=[]
        for i in range(tokens.shape[1]):
            logits.append(self._base.observe(tokens[:,i:i+1]))
            if self._ticket is not None:
                elapsed=self.steps-self._ticket['origin']
                if elapsed in self._base.head.config.horizons:
                    value=(self._base.history[:,-1]-self._base.center)/self._base.scale
                    self._ticket['observations'][elapsed]=value.double().clone()
        return torch.cat(logits,1)

    def status(self):
        if self._ticket is None:return {'status':'no_tracked_forecast','steps':self.steps}
        origin=self._ticket['origin'];seen=sorted(self._ticket['observations'])
        pending=[h for h in self._base.head.config.horizons if h>self.steps-origin]
        return {'status':'active' if pending else 'complete','origin_step':origin,'current_step':self.steps,
                'observed_horizons':seen,'pending':[{'origin_horizon':h,'absolute_target_step':origin+h,'bytes_remaining':origin+h-self.steps} for h in pending],
                'gate_weight':self._gates.get(max(seen),0.) if seen and pending else 0.,
                'gate_status':'validated_on_development_documents' if self._gate_hash else 'disabled_no_validation_gate',
                'warning':'Only arriving observations update evidence. No complete-state or token forecast, truth guarantee, or token skipping.'}

    @torch.no_grad()
    def distribution(self,*,policy='guarded'):
        if policy not in ('guarded','exact_conditional','original'):raise ValueError('Unknown forecast policy')
        status=self.status()
        if status['status']!='active':raise ValueError('No unobserved tracked targets remain')
        t=self._ticket;v=self._base.head.config.variables;hs=self._base.head.config.horizons
        observed=sorted(t['observations']);idx=[hs.index(h)*v+j for h in observed for j in range(v)]
        keep=[i for i in range(t['distribution'].dimension) if i not in idx]
        prior=t['distribution'].marginal(keep)
        if not observed or policy=='original':return prior.cloned()
        values=torch.cat([t['observations'][h] for h in observed],1)
        weight=status['gate_weight']
        if policy=='guarded' and weight==0:return prior.cloned()
        posterior=t['distribution'].condition(idx,values).distribution
        return (posterior if policy=='exact_conditional' else blend(prior,posterior,weight)).cloned()

    def forecast(self):
        """A NEW forecast from the current prefix, separate from the tracked origin."""
        return self._base.forecast()

    def raw_state_bytes(self):
        def size(x):
            if isinstance(x,torch.Tensor):return x.numel()*x.element_size()
            if isinstance(x,dict):return sum(size(v) for v in x.values())
            if isinstance(x,(list,tuple)):return sum(size(v) for v in x)
            return 0
        result={'language_coefficients':size(self._base.state),'history':size(self._base.history),'frozen_forecast':0,'observed_evidence':0}
        if self._ticket:
            result['frozen_forecast']=size(self._ticket['distribution'].payload())
            result['observed_evidence']=size(self._ticket['observations'])
        result['total']=sum(result.values());return result

    def save(self,path):
        base={'format':'forecast_session_v1','teacher_sha':self._base.teacher_hash,'head_sha':self._base.head_hash,
              'state':self._base.state,'history':self._base.history,'steps':self.steps}
        ticket=None
        if self._ticket:
            ticket={'origin':self._ticket['origin'],'distribution':self._ticket['distribution'].payload(),
                    'observations':{str(k):v.clone() for k,v in self._ticket['observations'].items()}}
        payload={'format':'observed_forecast_session_v1','base':base,'ticket':ticket,'gate_sha256':self._gate_hash}
        atomic_save({'payload':payload,'sha256':_payload_digest(payload)},path)

    def restore(self,path):
        saved=torch.load(path,map_location=self._base.device,weights_only=True)
        if not isinstance(saved,dict) or set(saved)!={'payload','sha256'}:raise ValueError('Invalid online snapshot')
        d=saved['payload']
        if _payload_digest(d)!=saved['sha256']:raise ValueError('Online snapshot integrity mismatch')
        if not isinstance(d,dict) or set(d)!={'format','base','ticket','gate_sha256'} or d['format']!='observed_forecast_session_v1':raise ValueError('Invalid online snapshot fields')
        if d['gate_sha256']!=self._gate_hash:raise ValueError('Gate changed since snapshot')
        state,history,steps=self._base.validated_snapshot(d['base'])
        ticket=d['ticket'];validated=None
        if ticket is not None:
            if not isinstance(ticket,dict) or set(ticket)!={'origin','distribution','observations'}:raise ValueError('Invalid tracked forecast')
            origin=ticket['origin'];hs=self._base.head.config.horizons;v=self._base.head.config.variables
            if type(origin) is not int or origin<self._base.head.config.history or origin>steps:raise ValueError('Invalid forecast origin')
            dist=GaussianMixture.restore(ticket['distribution'])
            if dist.means.shape!=(1,self._base.head.config.components,len(hs)*v) or dist.means.dtype!=torch.float64:raise ValueError('Invalid saved function shape/type')
            obs=ticket['observations'];expected={str(h) for h in hs if origin+h<=steps}
            if not isinstance(obs,dict) or set(obs)!=expected:raise ValueError('Missing, duplicate, or future observations in forecast snapshot')
            values={}
            for key,value in obs.items():
                if not isinstance(value,torch.Tensor) or value.shape!=(1,v) or value.dtype!=torch.float64 or not torch.isfinite(value).all():raise ValueError('Invalid evidence tensor')
                values[int(key)]=value.detach().clone()
            validated={'origin':origin,'distribution':dist,'observations':values}
        # Only mutate live state after every part was checked.
        self._base.state=state;self._base.history=history;self._base.steps=steps;self._ticket=validated
