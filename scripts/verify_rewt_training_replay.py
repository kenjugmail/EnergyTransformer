"""Replay frozen training code against final opt-in execution additions."""
import importlib,importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from rewt.architecture import ArchitectureConfig,ArchitectureEnergyLM
from rewt.data import file_sha256
OUT=ROOT/'reports/rewt_architecture_20261022'


def main():
    torch.set_num_threads(1)
    directory=OUT/'training_source';name='rewt_architecture_training_frozen'
    spec=importlib.util.spec_from_file_location(name,directory/'__init__.py',submodule_search_locations=[str(directory)])
    package=importlib.util.module_from_spec(spec);sys.modules[name]=package;spec.loader.exec_module(package)
    module=importlib.import_module(name+'.architecture')
    plan=json.loads((OUT/'plan.json').read_text());records=[]
    for case in plan['cases'][::3]:
        config=dict(plan['config'],**case['overrides'])
        torch.manual_seed(case['seed']);old=module.ArchitectureEnergyLM(module.ArchitectureConfig(**config))
        torch.manual_seed(case['seed']);new=ArchitectureEnergyLM(ArchitectureConfig(**config))
        assert all(torch.equal(t,new.state_dict()[n]) for n,t in old.state_dict().items())
        optimizers=[torch.optim.AdamW(m.parameters(),lr=.001) for m in (old,new)]
        for step in range(3):
            tokens=torch.randint(0,256,(2,19),generator=torch.Generator().manual_seed(22210+step))
            outputs=[]
            for model,optimizer in zip((old,new),optimizers):
                optimizer.zero_grad();output=model(tokens,state_gradient_interval=8)
                loss=output.logits.square().mean()+.03*model.diversity_loss();loss.backward();outputs.append(output)
            assert torch.equal(outputs[0].logits,outputs[1].logits),case['name']
            for n,p in old.named_parameters():
                q=dict(new.named_parameters())[n]
                assert (p.grad is None)==(q.grad is None)
                if p.grad is not None:assert torch.equal(p.grad,q.grad),(case['name'],n)
            for optimizer in optimizers:optimizer.step()
            assert all(torch.equal(t,new.state_dict()[n]) for n,t in old.state_dict().items()),case['name']
        records.append(dict(variant=case['variant'],optimizer_steps=3,outputs_gradients_and_weights_bitwise_equal=True))
    result=dict(records=records,note='All 13 variants replayed for three optimizer updates with final defaults. Added direct packing, directional robustness, structured opt-in and GPU variants do not change the frozen training paths.',
        frozen_source_sha256={p.name:file_sha256(p) for p in directory.glob('*.py')},final_source_sha256={p.name:file_sha256(p) for p in (ROOT/'rewt').glob('*.py')})
    (OUT/'training_replay.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(variants=len(records),bitwise_equal=True)))

if __name__=='__main__':main()
