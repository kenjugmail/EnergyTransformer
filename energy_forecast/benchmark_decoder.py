"""Interleaved end-to-end serial CPU decoding of fixed real inputs, no training."""
from pathlib import Path
import argparse,json,time,random,statistics
import torch
from .decoder_bridge import DecoderBridgeSession,KINDS
from .probe import load_energy_model

class TeacherSession:
    def __init__(self,path):self.model=load_energy_model(path);self.reset()
    def reset(self):self.state=None
    @torch.no_grad()
    def observe(self,x):
        out=[]
        for t in range(x.shape[1]):
            y,self.state=self.model(x[:,t:t+1],self.state);out.append(y)
        return torch.cat(out,1)

def state_bytes(x):
    if x is None:return 0
    if isinstance(x,torch.Tensor):return x.numel()*x.element_size()
    return sum(state_bytes(a) for a in x)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='.');p.add_argument('--out',required=True);p.add_argument('--repetitions',type=int,default=5);a=p.parse_args();r=Path(a.root)
    if Path(a.out).exists():raise FileExistsError(a.out)
    torch.set_num_threads(1);teacher=r/'checkpoints/teacher_973k.pt';head=r/'results/repair/fits/anchored_seed11.pt'
    sessions={'base':TeacherSession(teacher)}
    for kind in KINDS:sessions[kind]=DecoderBridgeSession(teacher,r/f'results/decoder_bridge/fits/{kind}_seed11/best.pt',head)
    sessions['folded_mean']=DecoderBridgeSession(teacher,r/'results/decoder_bridge/folded_mean_seed11.pt')
    text=(b'Energy memory stores constraints and reads them using observed context. '*32)
    stream=torch.tensor(list(text),dtype=torch.long)[None]
    def clone(x):
        if x is None:return None
        if isinstance(x,torch.Tensor):return x.clone()
        return tuple(clone(v) for v in x)
    records=[];rng=random.Random(21000)
    for prefix in (64,512):
        snapshots={}
        for name,s in sessions.items():
            s.reset();s.observe(stream[:,:prefix])
            snapshots[name]=(clone(s.state),clone(getattr(s,'history',None)),getattr(s,'steps',None))
            s.observe(stream[:,prefix:prefix+8])
        for rep in range(a.repetitions):
            names=list(sessions);rng.shuffle(names)
            for name in names:
                s=sessions[name];state,history,steps=snapshots[name]
                s.state=clone(state)
                if hasattr(s,'history'):s.history=clone(history);s.steps=steps
                start=time.perf_counter();s.observe(stream[:,prefix:prefix+64]);elapsed=time.perf_counter()-start
                records.append({'kind':name,'prefix':prefix,'rep':rep,'decode_ms_per_byte':elapsed*1000/64,
                    'state_bytes':state_bytes(s.state)+state_bytes(getattr(s,'history',None))})
    summary=[]
    for prefix in (64,512):
        for name in sessions:
            rows=[x for x in records if x['kind']==name and x['prefix']==prefix]
            summary.append({'kind':name,'prefix':prefix,'median_ms_per_byte':statistics.median(x['decode_ms_per_byte'] for x in rows),'state_bytes':rows[0]['state_bytes']})
    out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps({'torch':str(torch.__version__),'device':'cpu','threads':1,'repetitions':a.repetitions,'timed_continuation_bytes':64,'summary':summary,'records':records,
        'scope':'Prefix computed once; snapshots cloned outside timer. End-to-end fixed input serial decode after untimed prefix; CPU only; representative seed11; input is synthetic English-like text, not a quality benchmark.'},indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
