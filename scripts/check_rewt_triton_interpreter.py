"""CPU interpreter audit of Triton indexing and formulas, independent of GPU JIT."""
import json,os,sys
from pathlib import Path
if os.environ.get('TRITON_INTERPRET')!='1':raise RuntimeError('set TRITON_INTERPRET=1 before importing Triton kernels')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,triton
from rewt.architecture import ArchitectureConfig,ArchitectureEnergyLM
from rewt.advanced_kernels import prepare_solver,reduced_refine
from rewt.triton_refine import _preconditioned,_preconditioned_tiled


@torch.no_grad()
def main():
    torch.set_num_threads(1);torch.manual_seed(22215);rows=[]
    for modes,width in ((5,8),(16,48),(32,48),(64,48)):
        model=ArchitectureEnergyLM(ArchitectureConfig(width=width,hidden=32,modes=modes,history=8,
            recurrent_type='gru',compatibility_geometry='anchor_relative',wave_geometry='anchor_relative',initial_carry_scale=0.,
            output_geometry='centered',solver='preconditioned',step_size=.75)).eval()
        f,p=model.frequencies(),.15+1.85*model.raw_precision.sigmoid();ctx={k:v.contiguous() for k,v in prepare_solver(model,f,p,'preconditioned').items()}
        cut=model.blocks['output'].start;mean=ctx['lift'][:,cut:].mean(-1).contiguous();d=model.dimensions
        anchor=torch.randn(2,d);amp=.5*torch.randn(2,modes).tanh()/modes**.5
        for rounds in (0,1,4,12):
            expected=reduced_refine(model,anchor,amp,f,p,ctx,'preconditioned',rounds)[0]
            for tile in (0,64,128,256):
                output=torch.empty_like(anchor)
                if tile:
                    _preconditioned_tiled[(2,triton.cdiv(d,tile))](anchor,amp,model.phase,ctx['gram'],ctx['lift'],mean,ctx['step'],output,
                        modes,d,triton.next_power_of_2(modes),tile,rounds,cut,True,num_warps=4,enable_fp_fusion=False)
                else:
                    _preconditioned[(2,)](anchor,amp,model.phase,ctx['gram'],ctx['lift'],ctx['step'],output,
                        modes,d,triton.next_power_of_2(modes),triton.next_power_of_2(d),rounds,cut,True,num_warps=4,enable_fp_fusion=False)
                torch.testing.assert_close(output,expected,atol=3e-5,rtol=3e-5)
                rows.append(dict(modes=modes,dimensions=d,rounds=rounds,tile=tile,max_abs=float((output-expected).abs().max())))
    result=dict(rows=rows,mode='Triton CPU interpreter',scope='Indexing and formula verification only. GPU compiler, execution and CUDA graph tests are separate.')
    (ROOT/'reports/rewt_architecture_20261022/triton_interpreter.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(cases=len(rows),max_abs=max(r['max_abs'] for r in rows))))

if __name__=='__main__':main()
