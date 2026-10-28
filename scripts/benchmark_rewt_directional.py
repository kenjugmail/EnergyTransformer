"""Full core/support-model directional execution, including backward parity."""
from dataclasses import replace
import json,os,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from rewt.model import EnergyConfig,JointEnergyLM
OUT=ROOT/'reports/rewt_architecture_20261022'


def main():
    torch.set_num_threads(1);os.sched_setaffinity(0,{5});rows=[]
    for support in (False,True):
        for step in (.25,4.):
            for early in (False,True):
                torch.manual_seed(22212)
                cfg=EnergyConfig(width=16,hidden=64,modes=8,history=16,rounds=3,step_size=step,
                    support=support,early_backtrack=early,cache_history=True,
                    compatibility_geometry='anchor_relative',wave_geometry='anchor_relative',
                    initial_carry_scale=0.,output_geometry='centered',recurrent_type='gru',context_kernel=3,
                    diffusion_samples=1,diffusion_levels=1)
                models={'reference':JointEnergyLM(cfg),'directional':JointEnergyLM(replace(cfg,refinement_backend='directional'))}
                models['directional'].load_state_dict(models['reference'].state_dict())
                tokens=torch.randint(0,256,(4,16))
                outputs={}
                for name,model in models.items():
                    output=model(tokens);outputs[name]=output;output.logits.square().mean().backward()
                torch.testing.assert_close(outputs['reference'].logits,outputs['directional'].logits,atol=2e-5,rtol=2e-4)
                max_gradient=0.
                for name,p in models['reference'].named_parameters():
                    q=dict(models['directional'].named_parameters())[name]
                    assert (p.grad is None)==(q.grad is None)
                    if p.grad is not None:
                        torch.testing.assert_close(p.grad,q.grad,atol=3e-6,rtol=3e-4)
                        max_gradient=max(max_gradient,float((p.grad-q.grad).abs().max()))
                for backward in (False,True):
                    calls={};samples={name:[] for name in models}
                    for name,model in models.items():
                        model.train(backward)
                        def run(model=model):
                            model.zero_grad(set_to_none=True)
                            with torch.set_grad_enabled(backward):
                                output=model(tokens)
                                if backward:output.logits.square().mean().backward()
                        calls[name]=run
                        for _ in range(2):run()
                    for repeat in range(9):
                        for name in (list(calls) if repeat%2 else list(calls)[::-1]):
                            start=time.process_time();calls[name]();samples[name].append((time.process_time()-start)*1000)
                    rows.append(dict(support=support,step_size=step,early_backtrack=early,backward=backward,
                        median_cpu_ms={n:statistics.median(v) for n,v in samples.items()},samples_cpu_ms=samples,
                        max_logit_abs=float((outputs['reference'].logits-outputs['directional'].logits).detach().abs().max()),max_parameter_gradient_abs=max_gradient))
    result=dict(rows=rows,batch=4,length=16,rounds=3,cpu_affinity=sorted(os.sched_getaffinity(0)),torch=torch.__version__,
        note='Alternating warmed arms, one pinned CPU thread on a shared host. Full model forward or forward/backward, including conditioning and support proposals; synthetic initialized weights, not learned-quality evidence.')
    (OUT/'directional_benchmark.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(cases=len(rows),all_parity_passed=True)))

if __name__=='__main__':main()
