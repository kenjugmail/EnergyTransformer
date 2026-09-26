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
        self.device=device;self.reset()
    def reset(self):
        self.state=None;self.history=None;self.steps=0
    @torch.no_grad()
    def observe(self,tokens):
        if tokens.ndim!=2 or tokens.shape[0]!=1 or tokens.shape[1]<1:
            raise ValueError('Session accepts one nonempty stream [1,time]')
        logits=[]
        for t in range(tokens.shape[1]):
            out,self.state=self.model(tokens[:,t:t+1].to(self.device),self.state)
            feature=state_sketch(self.state)[:,None]
            self.history=feature if self.history is None else torch.cat((self.history,feature),1)
            self.history=self.history[:,-self.head.config.history:].contiguous()
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
        if set(d)!={'format','teacher_sha','head_sha','state','history','steps'} or d['format']!='forecast_session_v1':
            raise ValueError('Invalid session snapshot')
        if (d['teacher_sha'],d['head_sha'])!=(self.teacher_hash,self.head_hash):raise ValueError('Snapshot model mismatch')
        if type(d['steps']) is not int or d['steps']<0:raise ValueError('Invalid step count')
        h=d['history']
        if h is not None and (h.ndim!=3 or h.shape[0]!=1 or h.shape[-1]!=self.head.config.variables or h.shape[1]>self.head.config.history or not torch.isfinite(h).all()):
            raise ValueError('Invalid history snapshot')
        # Shape/finite validation is local; hashes are identities, not signatures.
        if d['state'] is not None:
            c=self.model.config
            if len(d['state'])!=c.layers:raise ValueError('Invalid layer count')
            for p,s in d['state']:
                if p.shape!=(1,c.heads,c.slots) or s.shape!=(1,c.heads,c.slots,c.width//c.heads):raise ValueError('Invalid energy shapes')
                if not torch.isfinite(p).all() or not torch.isfinite(s).all() or (p<c.prior).any():raise ValueError('Invalid coefficients')
        self.state=d['state'];self.history=h;self.steps=d['steps']
