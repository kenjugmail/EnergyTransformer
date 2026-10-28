"""Numerical and compiler-resource audit of tiled versus full-row mode kernels."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,triton
from rewt.architecture import ArchitectureConfig,ArchitectureEnergyLM
from rewt.advanced_kernels import prepare_solver,reduced_refine
from rewt.triton_refine import triton_preconditioned,_preconditioned_tiled
from rewt.ring_decode import RingDecodeSession


@torch.no_grad()
def main():
    torch.set_num_threads(1);torch.manual_seed(22203)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    checks=[];resources=[]
    for modes,width in ((5,8),(16,48),(32,48),(64,48)):
        c=ArchitectureConfig(width=width,hidden=2*width,modes=modes,history=8,position_period=16,
            context_kernel=3,recurrent_type='gru',compatibility_geometry='anchor_relative',wave_geometry='anchor_relative',
            initial_carry_scale=0.,output_geometry='centered',solver='preconditioned',step_size=.75,rounds=4)
        model=ArchitectureEnergyLM(c).cuda().eval();f,p=model.frequencies(),.15+1.85*model.raw_precision.sigmoid()
        prepared={k:v.contiguous() for k,v in prepare_solver(model,f,p,'preconditioned').items()}
        prepared['output_mean']=prepared['lift'][:,model.blocks['output'].start:].mean(-1).contiguous()
        anchor=model.canonical_state(torch.randn(3,model.dimensions,device='cuda'));amp=.5*torch.randn(3,modes,device='cuda').tanh()/modes**.5
        for rounds in (0,1,4,12):
            reference=reduced_refine(model,anchor,amp,f,p,prepared,'preconditioned',rounds)[0]
            for tile in (64,128,256):
                result=triton_preconditioned(anchor,amp,model.phase,prepared,rounds=rounds,output_start=model.blocks['output'].start,tile_size=tile)
                torch.testing.assert_close(result,reference,atol=3e-5,rtol=3e-5)
                checks.append(dict(modes=modes,dimensions=model.dimensions,rounds=rounds,tile=tile,max_abs=float((result-reference).abs().max())))
                if rounds==4:
                    kernel=_preconditioned_tiled.warmup(anchor,amp,model.phase,prepared['gram'],prepared['lift'],prepared['output_mean'],prepared['step'],torch.empty_like(anchor),
                        modes,model.dimensions,triton.next_power_of_2(modes),tile,rounds,model.blocks['output'].start,True,
                        num_warps=4,enable_fp_fusion=False,grid=(3,triton.cdiv(model.dimensions,tile)))
                    kernel._init_handles();resources.append(dict(modes=modes,tile=tile,registers=kernel.n_regs,spills=kernel.n_spills,shared_bytes=kernel.metadata.shared))
        tokens=torch.randint(0,256,(3,37),device='cuda');reference=model(tokens).logits
        session=RingDecodeSession(model,cuda_graph=True,triton=True,triton_tile=128)
        first=session.prefill(tokens[:,:3]);parts=[first.logits]
        for t in range(3,37):parts.append(session.step(tokens[:,t:t+1],validate_values=False))
        actual=torch.cat(parts,1);torch.testing.assert_close(actual,reference,atol=3e-5,rtol=3e-5)
        checks.append(dict(modes=modes,tile=128,whole_graph=True,max_abs=float((actual-reference).abs().max())))
    result=dict(checks=checks,resources=resources,tf32=False,note='Correctness and resource audit; no speed ranking on the shared busy GPU.')
    (ROOT/'reports/rewt_architecture_20261022/triton_tiles.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(checks=len(checks),max_abs=max(r['max_abs'] for r in checks),resources=resources),indent=2))

if __name__=='__main__':main()
