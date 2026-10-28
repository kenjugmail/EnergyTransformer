"""Small CUDA correctness and shared-device latency checks; no exclusive-GPU claim."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import subprocess
import time
import statistics
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from rewt.position_memory import load_learning_checkpoint, PositionMemoryEnergyLM
from rewt.architecture import ArchitectureConfig, ArchitectureEnergyLM
from rewt.advanced_kernels import prepare_solver, reduced_refine
from rewt.ring_decode import RingDecodeSession
from rewt.triton_refine import triton_preconditioned


def utilization():
    return subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader'],text=True).strip()


def timed(fn, repeat=9):
    samples=[]
    for _ in range(repeat):
        torch.cuda.synchronize();start=time.perf_counter();fn();torch.cuda.synchronize()
        samples.append((time.perf_counter()-start)*1000)
    return dict(median_ms=statistics.median(samples), samples_ms=samples)


@torch.no_grad()
def main():
    torch.set_num_threads(1); torch.manual_seed(22173)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device='cuda'; utilization_before=utilization()
    parent,saved=load_learning_checkpoint(ROOT/'runs/rewt_polish_20261010_v2/termination/warm_heads2/best.pt')
    base=PositionMemoryEnergyLM(replace(parent.config,refinement_backend='krylov2')).to(device).eval()
    base.load_state_dict(parent.state_dict())
    config=ArchitectureConfig(**dict(saved['config'],solver='preconditioned',rounds=4,step_size=.75))
    model=ArchitectureEnergyLM(config).to(device).eval();model.load_state_dict(parent.state_dict())
    f,p=model.frequencies(),.15+1.85*model.raw_precision.sigmoid()
    prepared={k:v.contiguous() for k,v in prepare_solver(model,f,p,'preconditioned').items()}
    kernel_rows=[]
    for batch in (1,8,32,256):
        anchor=model.canonical_state(torch.randn(batch,model.dimensions,device=device))
        amps=.5*torch.randn(batch,config.modes,device=device).tanh()/config.modes**.5
        for rounds in (0,1,2,4,8,12):
            ref=lambda:reduced_refine(model,anchor,amps,f,p,prepared,'preconditioned',rounds)[0]
            fused=lambda:triton_preconditioned(anchor,amps,model.phase,prepared,rounds=rounds,output_start=model.blocks['output'].start)
            a,b=ref(),fused();torch.testing.assert_close(a,b,atol=3e-5,rtol=3e-5)
            row=dict(batch=batch,rounds=rounds,max_abs=float((a-b).abs().max()))
            if rounds in (2,4,8):
                row.update(pytorch=timed(ref),triton=timed(fused))
            kernel_rows.append(row)
    (ROOT/'reports/rewt_architecture_20261022/cuda_kernel_partial.json').write_text(json.dumps(dict(kernel=kernel_rows,utilization_before=utilization_before,tf32=False),indent=2)+'\n')
    graph_rows=[]
    for selected in (base, model):
        for batch,prefix in ((1,3),(4,129)):
            tokens=torch.randint(0,256,(batch,prefix+135),device=device)
            ordinary=selected(tokens).logits
            for graph,triton in ((False,False),(True,False),*(([(True,True)]) if selected is model else [])):
                session=RingDecodeSession(selected,cuda_graph=graph,triton=triton)
                start=time.perf_counter();output=session.prefill(tokens[:,:prefix]);torch.cuda.synchronize()
                preparation=time.perf_counter()-start
                collected=[output.logits]
                for t in range(prefix,tokens.shape[1]):
                    collected.append(session.step(tokens[:,t:t+1],validate_values=False))
                actual=torch.cat(collected,1)
                print('decode comparison', session.solver, batch, prefix, graph, triton, float((actual-ordinary).abs().max()), flush=True)
                torch.testing.assert_close(actual,ordinary,atol=5e-4,rtol=3e-4)
                assert torch.equal(actual.argmax(-1),ordinary.argmax(-1))
                fixed=tokens[:,-1:]
                latency=timed(lambda:session.step(fixed,validate_values=False),repeat=15)
                graph_rows.append(dict(solver=session.solver,batch=batch,prefix=prefix,graph=graph,triton=triton,
                    preparation_seconds=preparation,max_abs=float((actual-ordinary).abs().max()),argmax_equal=True,latency=latency))
    result=dict(tf32=False,device=torch.cuda.get_device_name(),torch=torch.__version__,utilization_before=utilization_before,
        utilization_after=utilization(),note='Shared busy GPU, descriptive latency only. End-to-end step includes input copy, graph replay, and owned output clone. Compilation/preparation excluded from warm latency.',
        kernel=kernel_rows,decode=graph_rows,peak_allocated_bytes=torch.cuda.max_memory_allocated())
    (ROOT/'reports/rewt_architecture_20261022/cuda.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('kernel','decode')},indent=2))

if __name__=='__main__':main()
