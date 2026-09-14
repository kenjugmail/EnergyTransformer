"""Small actual-device forward/backward, reset, stream and causality checks."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import torch
from .device import require_idle_gpu,set_memory_limit
from .model import LM,Config
from .kernel import chunk_gla
from .runtime import atomic_json,environment,source_identity


def run(out,device='cuda',max_seconds=1800):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    dev=torch.device(device)
    if dev.type=='cuda':require_idle_gpu();set_memory_limit(dev,16)
    started=time.perf_counter();checks=[]
    for kind in ('energy','softmax'):
        if time.perf_counter()-started>=max_seconds:raise TimeoutError('Smoke budget exhausted')
        torch.manual_seed(14)
        cfg=Config(kind=kind,width=32,layers=2,heads=2,slots=8,ff_width=64,chunk=8)
        cpu=LM(cfg).eval();model=LM(cfg).to(dev).eval();model.load_state_dict(cpu.state_dict())
        tokens=torch.randint(256,(2,25));x=tokens.to(dev)
        reset=torch.zeros_like(x,dtype=torch.bool);reset[0,9]=True;reset[1,13]=True
        with torch.no_grad():
            full,state=model(x,reset=reset)
            first,s=model(x[:,:11],reset=reset[:,:11])
            second,_=model(x[:,11:],s,reset[:,11:])
            torch.testing.assert_close(full,torch.cat((first,second),1),atol=3e-4,rtol=3e-4)
            changed=x.clone();changed[:,18:]=(changed[:,18:]+1)%256
            future,_=model(changed,reset=reset)
            if not torch.equal(full[:,:18],future[:,:18]):raise AssertionError('Future inputs altered prefix')
            reference,_=cpu(tokens,reset=reset.cpu())
            torch.testing.assert_close(full.cpu(),reference,atol=2e-3,rtol=2e-3)
        model.train();model.zero_grad(set_to_none=True)
        with torch.autocast(device_type=dev.type,dtype=torch.bfloat16,enabled=dev.type=='cuda'):
            y,_=model(x,reset=reset)
            loss=y.float().square().mean()
        loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise AssertionError('Nonfinite model gradient')
        # Include learned forget-gate derivatives in extreme decay backward.
        u=torch.randn(1,16,2,4,device=dev,requires_grad=True)
        wk=torch.randn_like(u,requires_grad=True);v=torch.randn(1,16,2,3,device=dev,requires_grad=True)
        lf=torch.full_like(u,-9.,requires_grad=True);s0=torch.zeros(1,2,4,3,device=dev,requires_grad=True)
        output,final,_=chunk_gla(u,wk,v,lf,s0,8)
        (output.square().mean()+final.square().mean()).backward()
        if not all(torch.isfinite(t.grad).all() for t in (u,wk,v,lf,s0)):raise AssertionError('Nonfinite kernel gradient')
        checks.append({'kind':kind,'config':asdict(cfg),'status':'passed',
                       'cpu_device_max_abs':float((full.cpu()-reference).abs().max())})
    result={'status':'passed','device':environment(dev),'source_sha256':source_identity(),
            'checks':checks,'elapsed_s':time.perf_counter()-started,
            'meaning':'Numerical smoke test only; restart independently covered by fresh-process trainer tests.'}
    atomic_json(result,out);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--device',default='cuda');p.add_argument('--max-seconds',type=float,default=1800)
    a=p.parse_args();print(json.dumps(run(a.out,a.device,a.max_seconds),indent=2))
