"""Optional past-only forecasting sidecar; original token outputs stay unchanged.

It retains a bounded summary history in addition to the LM's original memory.
It cannot replace P/S with a guessed future, skip decoding, or authorize edits.
"""
from pathlib import Path
import torch
from .probe import load_energy_model,state_sketch,checkpoint_hash
from .experiment import load_head,atomic_save

class ForecastSession:
    def __init__(self,teacher,head,device='cpu'):
        self.teacher_hash=checkpoint_hash(teacher);self.head_hash=checkpoint_hash(head)
        self.model=load_energy_model(teacher,device);self.head,self.metadata=load_head(head,device)
        if self.metadata['teacher_sha256']!=self.teacher_hash:
            raise ValueError('Forecast head belongs to another teacher; prepare and retrain on this checkpoint')
        self.center=self.metadata['normalization']['center'].to(device)
        self.scale=self.metadata['normalization']['scale'].to(device)
        v=self.head.config.variables
        if self.center.shape!=(v,) or self.scale.shape!=(v,) or not torch.isfinite(self.center).all() or not torch.isfinite(self.scale).all() or (self.scale<=0).any():
            raise ValueError('Invalid forecast normalization')
        self.device=device;self.reset()
    def reset(self):
        self.state=None;self.history=None;self.steps=0
    @torch.no_grad()
    def observe(self,tokens):
        if not isinstance(tokens,torch.Tensor) or tokens.dtype!=torch.long or tokens.ndim!=2 or tokens.shape[0]!=1 or tokens.shape[1]<1:
            raise ValueError('Session accepts one nonempty stream [1,time]')
        if (tokens<0).any() or (tokens>=self.model.config.vocab).any():raise ValueError('Token outside vocabulary')
        logits=[]
        for t in range(tokens.shape[1]):
            out,self.state=self.model(tokens[:,t:t+1].to(self.device),self.state)
            feature=state_sketch(self.state)[:,None]
            self.history=feature if self.history is None else torch.cat((self.history,feature),1)
            self.history=self.history[:,-self.head.config.history:].clone(memory_format=torch.contiguous_format)
            self.steps+=1;logits.append(out)
        return torch.cat(logits,1)
    @torch.no_grad()
    def forecast(self):
        if self.history is None or self.history.shape[1]<self.head.config.history:
            raise ValueError('More real observations are needed before forecasting')
        return self.head((self.history-self.center)/self.scale)
    def save(self,path):
        atomic_save({'format':'forecast_session_v1','teacher_sha':self.teacher_hash,'head_sha':self.head_hash,
            'state':self.state,'history':self.history,'steps':self.steps},path)
    def restore(self,path):
        d=torch.load(path,map_location=self.device,weights_only=True)
        self.state,self.history,self.steps=self.validated_snapshot(d)

    def validated_snapshot(self,d):
        if not isinstance(d,dict) or set(d)!={'format','teacher_sha','head_sha','state','history','steps'} or d['format']!='forecast_session_v1':
            raise ValueError('Invalid session snapshot')
        if (d['teacher_sha'],d['head_sha'])!=(self.teacher_hash,self.head_hash):raise ValueError('Snapshot model mismatch')
        if type(d['steps']) is not int or d['steps']<0:raise ValueError('Invalid step count')
        h=d['history']; state=d['state']; steps=d['steps']
        if steps==0:
            if h is not None or state is not None:raise ValueError('Empty snapshot contains history')
        else:
            if not isinstance(h,torch.Tensor) or state is None:raise ValueError('Missing observed state/history')
            if (h.ndim!=3 or h.shape!=(1,min(steps,self.head.config.history),self.head.config.variables)
                or h.dtype!=torch.float32 or not torch.isfinite(h).all()):
                raise ValueError('Invalid observed history')
            c=self.model.config
            if not isinstance(state,(tuple,list)) or len(state)!=c.layers:raise ValueError('Invalid layer count')
            for layer in state:
                if not isinstance(layer,(tuple,list)) or len(layer)!=2:raise ValueError('Invalid layer schema')
                p,s=layer
                if not isinstance(p,torch.Tensor) or not isinstance(s,torch.Tensor):raise ValueError('Coefficients must be tensors')
                if p.dtype!=torch.float32 or s.dtype!=torch.float32:raise ValueError('Expected float32 energy coefficients')
                if p.shape!=(1,c.heads,c.slots) or s.shape!=(1,c.heads,c.slots,c.width//c.heads):raise ValueError('Invalid energy shapes')
                if not torch.isfinite(p).all() or not torch.isfinite(s).all() or (p<c.prior).any():raise ValueError('Invalid coefficients')
            feature=state_sketch(state)
            if not torch.equal(feature,h[:,-1]):raise ValueError('Latest summary disagrees with saved observed state')
        return state,h,steps

    def point_forecast_status(self):
        """Validation evidence, not a probability of correctness or authorization."""
        entries=self.metadata.get('selection',{}).get('horizons')
        if entries is None:
            return {'status':'no_horizon_gate','meaning':'No calibration support assessment is recorded.'}
        return {'status':'validation_assessed','horizons':[
            {'horizon':entry['horizon'],
             'point_mean':'context_linear' if entry['use_context'] else 'training_target_mean',
             'context_gain':float(self.head.anchor.gain[i]),
             'warning':'Empirical development evidence, not guaranteed future accuracy.'}
            for i,entry in enumerate(entries)]}
