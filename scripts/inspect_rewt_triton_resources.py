"""Compile/resource audit, separate from contended GPU latency measurements."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,triton
from triton.runtime import driver
from rewt.triton_refine import _preconditioned


def main():
    torch.set_num_threads(1)
    props=driver.active.utils.get_device_properties(torch.cuda.current_device())
    rows=[]
    for m in (16,32,64):
        d=400;anchor=torch.zeros(1,d,device='cuda');amp=torch.zeros(1,m,device='cuda')
        phase=torch.zeros(m,device='cuda');gram=torch.eye(m,device='cuda');lift=torch.zeros(m,d,device='cuda');eta=torch.tensor(.75,device='cuda');out=torch.empty_like(anchor)
        for warps in (2,4,8):
            kernel=_preconditioned.warmup(anchor,amp,phase,gram,lift,eta,out,m,d,triton.next_power_of_2(m),triton.next_power_of_2(d),4,144,True,
                num_warps=warps,enable_fp_fusion=False,grid=(1,))
            kernel._init_handles()
            registers=kernel.n_regs;shared=kernel.metadata.shared
            resident_by_registers=props['max_num_regs']//(registers*32*warps)
            rows.append(dict(modes=m,dimensions=d,warps=warps,registers_per_thread=registers,
                spills=getattr(kernel,'n_spills',None),shared_bytes=shared,
                blocks_per_sm_register_limit=resident_by_registers))
    result=dict(device=torch.cuda.get_device_name(),properties={k:v for k,v in props.items() if isinstance(v,(str,int,float))},
        variants=rows,note='Resource bounds, not measured occupancy. One program per batch row; batch=1 activates at most one SM regardless of theoretical occupancy. Default is four warps, pending uncontended latency tuning.')
    (ROOT/'reports/rewt_architecture_20261022/triton_resources.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
