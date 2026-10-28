"""Archive, generation, gradient and certification checks for new execution paths."""
from collections import defaultdict
from dataclasses import replace
import importlib.util,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from rewt.architecture import ArchitectureConfig,ArchitectureEnergyLM
from rewt.position_memory import load_learning_checkpoint
from rewt.ring_decode import RingDecodeSession
from rewt.data import file_sha256
from verify_rewt_kernels import original_loader
from benchmark_rewt_memory import weight_hash
OUT=ROOT/'reports/rewt_architecture_20261022'


@torch.inference_mode()
def generate(model,prompts,ring=False):
    x=torch.tensor([list(p.encode()) for p in prompts]);model.eval()
    if isinstance(model,ArchitectureEnergyLM):model.prepare_inference()
    session=RingDecodeSession(model) if ring else None
    output=session.prefill(x) if ring else model(x)
    state,logits=output.state,output.logits[:,-1];rows=[[] for _ in prompts];done=[False]*len(prompts)
    for t in range(min(40,128-x.shape[1])):
        token=logits.argmax(-1)
        for i,n in enumerate(token.tolist()):
            if not done[i]:rows[i].append(n);done[i]=n==10
        if all(done):break
        if ring:logits=session.step(token[:,None])[:,-1]
        else:
            output=model(token[:,None],state);state=output.state;logits=output.logits[:,-1]
    return [bytes(r) for r in rows]


def main():
    torch.set_num_threads(1);torch.manual_seed(22195)
    loader=original_loader(OUT/'baseline_source')
    previous=json.loads((ROOT/'reports/rewt_kernels_20261022/verification.json').read_text())
    archive=[]
    for row in previous['archived_checkpoint_parity']:
        path=Path(row['checkpoint']);a,_=loader(path);b,_=load_learning_checkpoint(path)
        tokens=torch.randint(0,256,(2,21))
        with torch.no_grad():x,y=a.eval()(tokens),b.eval()(tokens)
        assert torch.equal(x.logits,y.logits) and torch.equal(x.state.joint,y.state.joint)
        archive.append(dict(checkpoint=str(path),sha256=file_sha256(path),bitwise_equal=True))
    path=ROOT/'runs/rewt_polish_20261010_v2/termination/warm_heads2/best.pt'
    base,saved=load_learning_checkpoint(path);base.eval()
    models={}
    for solver in ('krylov2','certified'):
        model=ArchitectureEnergyLM(ArchitectureConfig(**dict(saved['config'],solver=solver))).eval()
        model.load_state_dict(base.state_dict());models[solver]=model
    direct=type(base)(replace(base.config,refinement_backend='krylov2',packed_projections=True))
    direct.load_state_dict(base.state_dict());models['direct_packed']=direct
    groups=defaultdict(list)
    for line in (OUT/'tasks_valid.jsonl').read_text().splitlines():
        row=json.loads(line);groups[len(row['prompt'])].append(row['prompt'])
    records=[]
    hashes={name:weight_hash(model) for name,model in models.items()}
    for _,prompts in sorted(groups.items()):
        for first in range(0,len(prompts),8):
            subset=prompts[first:first+8];expected=generate(base,subset)
            for name,model in models.items():
                actual=generate(model,subset,True)
                records.extend(dict(variant=name,prompt=p,reference_hex=a.hex(),candidate_hex=b.hex(),equal=a==b)
                               for p,a,b in zip(subset,expected,actual))
    assert all(r['equal'] for r in records)
    assert hashes=={name:weight_hash(model) for name,model in models.items()}
    # Trainable parameter gradient parity, including packed weights and GRU
    # precomputation, checked independently on the saved trained model.
    tokens=torch.randint(0,256,(3,27));base.train();base.zero_grad();reference=base(tokens);reference.logits.square().mean().backward()
    gradients=[]
    for name,model in models.items():
        model.train();model.zero_grad();output=model(tokens);output.logits.square().mean().backward()
        differences={}
        for n,p in base.named_parameters():
            q=dict(model.named_parameters())[n]
            assert (p.grad is None)==(q.grad is None)
            if p.grad is not None:
                torch.testing.assert_close(p.grad,q.grad,atol=3e-5,rtol=5e-4)
                differences[n]=float((p.grad-q.grad).abs().max())
        gradients.append(dict(variant=name,max_parameter_gradient_abs=max(differences.values()),per_parameter=differences,
                              max_logit_abs=float((reference.logits-output.logits).detach().abs().max())))
    result=dict(archived=archive,records=records,generations_equal=True,weights_unchanged=True,gradients=gradients,
        source_sha256={p.name:file_sha256(p) for p in (ROOT/'rewt').glob('*.py')})
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(archives=len(archive),prompt_variant_pairs=len(records),all_equal=True)))

if __name__=='__main__':main()
