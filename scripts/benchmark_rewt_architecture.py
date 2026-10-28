"""Alternating-arm CPU benchmark with warmup, numerical checks and setup costs.

Prefer running after training workers finish. Wall and CPU process timing are
reported separately; external system load is recorded, not assumed absent.
"""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from rewt.position_memory import load_learning_checkpoint,PositionMemoryEnergyLM
from rewt.architecture import ArchitectureConfig,ArchitectureEnergyLM
from rewt.advanced_kernels import prepare_solver
from rewt.structured_solver import prepare_structured
from rewt.decode import EnergyDecodeSession
from rewt.ring_decode import RingDecodeSession
from rewt.data import file_sha256
from study_rewt_architecture import VARIANTS
OUT=ROOT/'reports/rewt_architecture_20261022'


def stats(rows):
    return dict(median_cpu_ms=statistics.median(r[0] for r in rows),median_wall_ms=statistics.median(r[1] for r in rows),samples=rows)


def measure(fn):
    wall,cpu=time.perf_counter(),time.process_time();fn()
    return (time.process_time()-cpu)*1000,(time.perf_counter()-wall)*1000


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cpu',type=int);parser.add_argument('--repeats',type=int,default=7);parser.add_argument('--output',default='benchmark.json');args=parser.parse_args()
    torch.set_num_threads(1);torch.manual_seed(22193)
    if args.cpu is not None:os.sched_setaffinity(0,{args.cpu})
    path=ROOT/'runs/rewt_polish_20261010_v2/termination/warm_heads2/best.pt'
    parent,saved=load_learning_checkpoint(path)
    models={}
    for name,backend in [('reference','reference'),('krylov2','krylov2'),('direct_packed','krylov2')]:
        model=PositionMemoryEnergyLM(replace(parent.config,refinement_backend=backend,packed_projections=name=='direct_packed'));model.load_state_dict(parent.state_dict());models[name]=model
    for name,opts in [('packed',{}),('certified_same_step',dict(solver='certified')),('certified_075',dict(solver='certified',step_size=.75)),
                      ('precond2',dict(solver='preconditioned',step_size=.75)),('precond4',dict(solver='preconditioned',step_size=.75,rounds=4)),
                      ('precond8',dict(solver='preconditioned',step_size=.75,rounds=8)),('precond4_structured',dict(solver='preconditioned',step_size=.75,rounds=4,preconditioner_factorization='structured')),('scan',dict(backbone='scan')),('feedback8',dict(backbone='scan',feedback_stride=8)),
                      ('delta',dict(backbone='scan',delta_size=16)),('patch4',dict(backbone='scan',patch_stride=4))]:
        model=ArchitectureEnergyLM(ArchitectureConfig(**dict(saved['config'],**opts)))
        weights=model.state_dict();weights.update({k:v for k,v in parent.state_dict().items() if k in weights and weights[k].shape==v.shape});model.load_state_dict(weights)
        models[name]=model
    rows=[];checks=[]
    for batch,length in ((1,32),(24,64),(24,128)):
        tokens=torch.randint(0,256,(batch,length))
        with torch.no_grad():
            reference=models['krylov2'].eval()(tokens).logits
            for name in ('direct_packed','packed','certified_same_step'):
                actual=models[name].eval()(tokens).logits
                torch.testing.assert_close(actual,reference,atol=3e-4,rtol=3e-4)
                checks.append(dict(batch=batch,length=length,arm=name,max_logit_abs=float((actual-reference).abs().max())))
        for backward in (False,True):
            calls={}
            for name,model in models.items():
                model.train(backward)
                def run(model=model):
                    model.zero_grad(set_to_none=True)
                    with torch.set_grad_enabled(backward):
                        output=model(tokens)
                        if backward:output.logits.square().mean().backward()
                calls[name]=run
                for _ in range(2):run()
            samples={name:[] for name in calls}
            names=list(calls)
            for i in range(args.repeats):
                order=names[i%len(names):]+names[:i%len(names)]
                if i%2:order=order[::-1]
                for name in order:samples[name].append(measure(calls[name]))
            rows.append(dict(batch=batch,length=length,backward=backward,arms={n:stats(s) for n,s in samples.items()}))
    decode=[]
    for batch,prefix in ((1,16),(1,128),(8,128)):
        tokens=torch.randint(0,256,(batch,prefix+32));calls={}
        with torch.inference_mode():
            for name in ('krylov2','packed','certified_same_step','precond4'):
                model=models[name].eval()
                for ring in (False,True):
                    if not ring and isinstance(model,ArchitectureEnergyLM):continue
                    def run(model=model,ring=ring):
                        session=RingDecodeSession(model) if ring else EnergyDecodeSession(model)
                        session.prefill(tokens[:,:prefix])
                        for t in range(prefix,tokens.shape[1]):session.step(tokens[:,t:t+1])
                    calls[name+('/ring' if ring else '/dynamic')]=run
                    run()
            samples={name:[] for name in calls}
            for i in range(args.repeats):
                for name in (list(calls) if i%2==0 else list(calls)[::-1]):samples[name].append(measure(calls[name]))
            decode.append(dict(batch=batch,prefix=prefix,new_bytes=32,arms={n:stats(s) for n,s in samples.items()}))
    preparation={};model=models['precond4'].eval()
    for grad in (False,True):
        for method,fn in [('dense',prepare_solver),('structured',prepare_structured)]:
            def run():
                model.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(grad):
                    f,p=model.frequencies(),.15+1.85*model.raw_precision.sigmoid()
                    ctx=fn(model,f,p,'preconditioned') if method=='dense' else fn(model,f,p)
                    if grad:ctx['lift'].square().mean().backward()
            run();preparation[f'{method}/backward={grad}']=stats([measure(run) for _ in range(args.repeats)])
    result=dict(torch=torch.__version__,threads=1,cpu_affinity=sorted(os.sched_getaffinity(0)),load_average=os.getloadavg(),checkpoint_sha256=file_sha256(path),
        note='Single-thread CPU process-time and wall-time medians. Architecture timings are execution comparisons, not equal-quality claims. Changed scan architectures initialized with matching compatible tensors for timing only; learned quality comes from separate cold training.',
        cases=rows,decode=decode,preparation=preparation,parity=checks,
        source_sha256={p.name:file_sha256(p) for p in (ROOT/'rewt').glob('*.py')})
    (OUT/args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(complete=True,cases=len(rows)),indent=2))

if __name__=='__main__':main()
