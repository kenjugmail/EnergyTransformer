"""Optional live candidate refinement; only the original teacher P/S persists.

Use only a checkpoint trained for this exact teacher and memory read. No latent
candidate, assumed answer, or optimized prediction is installed in teacher state.
The first 15 bytes use the unmodified teacher, matching cached-study warmup.
"""
from pathlib import Path
import argparse,json,math
import torch
from .probe import checkpoint_hash,load_energy_model
from .refine_model import load_refinement,compact_energy_read
from .experiment import atomic_save

class RefinementSession:
    def __init__(self,teacher,adapter,*,steps=4,tolerance=None,device='cpu'):
        if type(steps)is not int or not 0<=steps<=64:raise ValueError('Invalid inference depth')
        if tolerance is not None and (not math.isfinite(tolerance) or tolerance<0):raise ValueError('Invalid tolerance')
        self.device=torch.device(device);self.teacher_hash=checkpoint_hash(teacher);self.adapter_hash=checkpoint_hash(adapter)
        self.adapter,self.metadata=load_refinement(adapter,device=device)
        if self.metadata['teacher_sha256']!=self.teacher_hash:raise ValueError('Teacher identity mismatch')
        self.teacher=load_energy_model(teacher,device)
        if self.adapter.config.width!=self.teacher.config.width or self.adapter.config.memory_width!=16*self.teacher.config.heads:
            raise ValueError('Teacher/read shape mismatch')
        if tolerance is not None and (not self.adapter.config.energy or steps<1):raise ValueError('Adaptive policy needs energy steps')
        self.depth,self.tolerance=steps,tolerance;self.state=None;self.steps=0;self._h=None
        self._hook=self.teacher.norm.register_forward_hook(self._capture)
    def _capture(self,module,args,out):self._h=out.detach()
    def close(self):self._hook.remove()
    @torch.no_grad()
    def observe(self,tokens):
        if not isinstance(tokens,torch.Tensor) or tokens.dtype!=torch.long or tokens.ndim!=2 or tokens.shape[0]!=1 or tokens.shape[1]<1:
            raise ValueError('Expected nonempty single byte stream')
        if (tokens<0).any() or (tokens>=256).any():raise ValueError('Byte outside vocabulary')
        logits=[]
        for i in range(tokens.shape[1]):
            base,new=self.teacher(tokens[:,i:i+1].to(self.device),self.state)
            h=self._h[:,0];memory=compact_energy_read(new)
            if self.steps+1<16:current=base
            elif self.tolerance is None:current=self.adapter(h,memory,self.teacher.head.weight,steps=self.depth)[:,None]
            else:current=self.adapter.solve(h,memory,self.teacher.head.weight,max_steps=self.depth,tolerance=self.tolerance)['logits'][:,None]
            if not torch.isfinite(current).all():raise FloatingPointError('Nonfinite prediction; this byte not committed')
            self.state=new;self.steps+=1;logits.append(current)
        self._h=None
        return torch.cat(logits,1)
    def save(self,path):
        atomic_save({'format':'refinement_session_v1','teacher':self.teacher_hash,'adapter':self.adapter_hash,
            'depth':self.depth,'tolerance':self.tolerance,'steps':self.steps,'state':self.state},path)
    def restore(self,path):
        d=torch.load(path,map_location=self.device,weights_only=True)
        if set(d)!={'format','teacher','adapter','depth','tolerance','steps','state'} or d['format']!='refinement_session_v1':raise ValueError('Malformed session')
        if (d['teacher'],d['adapter'],d['depth'],d['tolerance'])!=(self.teacher_hash,self.adapter_hash,self.depth,self.tolerance):raise ValueError('Session identity mismatch')
        n=d['steps'];s=d['state'];c=self.teacher.config
        if type(n)is not int or n<0:raise ValueError('Invalid steps')
        if n==0:
            if s is not None:raise ValueError('Unexpected initial memory')
        else:
            if not isinstance(s,(tuple,list)) or len(s)!=c.layers:raise ValueError('Missing layer states')
            for layer in s:
                if not isinstance(layer,(tuple,list)) or len(layer)!=2:raise ValueError('Malformed layer')
                p,a=layer
                if not isinstance(p,torch.Tensor) or not isinstance(a,torch.Tensor):raise ValueError('Non-tensor state')
                if p.shape!=(1,c.heads,c.slots) or a.shape!=(1,c.heads,c.slots,c.width//c.heads):raise ValueError('Wrong state shape')
                if p.dtype!=torch.float32 or a.dtype!=torch.float32 or not torch.isfinite(p).all() or not torch.isfinite(a).all() or (p<c.prior).any():raise ValueError('Invalid energy coefficients')
            s=tuple(tuple(t.detach().clone() for t in layer) for layer in s)
        self.state=s;self.steps=n;self._h=None

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--teacher',default='checkpoints/teacher_973k.pt')
    p.add_argument('--adapter',required=True);p.add_argument('--steps',type=int,default=4);p.add_argument('--tolerance',type=float)
    p.add_argument('--prompt',default='The ');p.add_argument('--length',type=int,default=80);p.add_argument('--seed',type=int,default=123)
    args=p.parse_args()
    if not args.prompt or not 0<=args.length<=4096:raise ValueError('Nonempty prompt and bounded length required')
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    s=RefinementSession(args.teacher,args.adapter,steps=args.steps,tolerance=args.tolerance)
    raw=bytearray(args.prompt.encode());logits=s.observe(torch.tensor([list(raw)]))
    for _ in range(args.length):
        token=torch.multinomial((logits[:,-1]/.8).softmax(-1),1);raw.extend(token[0].tolist());logits=s.observe(token)
    print(raw.decode('utf8',errors='replace'));s.close()
if __name__=='__main__':main()
