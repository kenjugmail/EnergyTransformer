"""CPU square-root audit against a preserved covariance reference.

No fitting, checkpoint selection, GPU work or decoder changes. The optional
trained-data check uses all available test rows for numerical parity only.
The generated fixtures are mathematical controls, not language benchmarks.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import time
import torch
from .conditioning import GaussianMixture
from .conditioning_reference import dense_condition, dense_marginal
from .experiment import load_head


def max_error(a, b):
    return float((a-b).abs().max())


def constructed_regressions():
    factor=torch.tensor([[[[1.,0.,0.],[1.,1e-4,0.],[0.,0.,1.]]]])
    f=GaussianMixture(torch.zeros(1,1),torch.zeros(1,1,3),factor)
    records=[]
    cases=[('marginal_leading',lambda:dense_marginal(f,[0,1]),lambda:f.marginal([0,1])),
           ('marginal_reordered',lambda:dense_marginal(f,[1,0]),lambda:f.marginal([1,0])),
           ('condition_reordered',lambda:dense_condition(f,[1,0],torch.zeros(1,2)),lambda:f.condition([1,0],torch.zeros(1,2)).distribution),
           ('condition_nonprefix',lambda:dense_condition(f,[1],torch.zeros(1,1)),lambda:f.condition([1],torch.zeros(1,1)).distribution)]
    for name,old,new in cases:
        try: old();before='completed'
        except torch.linalg.LinAlgError:before='factorization_failed'
        actual=new()
        records.append({'case':name,'covariance_reference':before,'square_root':'completed',
                        'energy_at_zero':float(actual.energy(torch.zeros(1,actual.dimension))),
                        'variance':actual.variance.tolist()})
    return records


def numerical_parity(function, targets):
    d=function.dimension;count=max(1,d//4)
    cases={'leading':list(range(count)), 'reversed_leading':list(reversed(range(count))),
           'nonleading_even':list(range(1,2*count,2))}
    out=[]
    for name,given in cases.items():
        observed=targets[:,given]
        a=function.condition(given,observed);b=dense_condition(function,given,observed)
        remaining=list(a.remaining_indices);m=function.marginal(remaining);n=dense_marginal(function,remaining)
        out.append({'case':name,'origins':len(targets),
                    'mean_max_abs':max_error(a.distribution.mean,b.distribution.mean),
                    'covariance_max_abs':max_error(a.distribution.component_covariance,b.distribution.component_covariance),
                    'log_evidence_max_abs':max_error(a.log_evidence,b.log_evidence),
                    'conditional_energy_max_abs':max_error(a.distribution.energy(targets[:,remaining]),b.distribution.energy(targets[:,remaining])),
                    'marginal_energy_max_abs':max_error(m.energy(targets[:,remaining]),n.energy(targets[:,remaining]))})
    return out


def timing(function,targets,repeats=9,calls=12):
    d=function.dimension;count=d//4;indices=list(range(1,2*count,2));observed=targets[:,indices]
    prefix=list(range(count));pobs=targets[:,:count];remaining=list(range(count,d))
    # For exact-posterior policy the old online session needlessly made a prior
    # marginal before using its already-optimized prefix condition.
    def old_online_exact():
        dense_marginal(function,remaining)
        return function.condition_prefix(count,pobs).distribution.cloned()
    methods={'general_covariance':lambda:dense_condition(function,indices,observed),
             'general_square_root':lambda:function.condition(indices,observed),
             'marginal_covariance':lambda:dense_marginal(function,remaining),
             'marginal_square_root':lambda:function.marginal(remaining),
             'online_exact_previous_work':old_online_exact,
             'online_exact_required_work':lambda:function.condition_prefix(count,pobs).distribution.cloned()}
    rng=random.Random(261409);records=[]
    for f in methods.values():
        for _ in range(3):f()
    for rep in range(repeats):
        order=list(methods);rng.shuffle(order)
        for name in order:
            start=time.perf_counter()
            for _ in range(calls):methods[name]()
            records.append({'repeat':rep,'method':name,'milliseconds':(time.perf_counter()-start)*1000/calls})
    return {'batch':len(targets),'records':records,'medians':{name:statistics.median(r['milliseconds'] for r in records if r['method']==name) for name in methods}}


def run(head=None,data=None):
    torch.set_num_threads(1)
    result={'torch':str(torch.__version__),'device':'cpu','cuda_executed':False,
            'scope':'Numerical parity and component timing, not new forecasting quality or token acceleration',
            'constructed_regressions':constructed_regressions()}
    with torch.no_grad():
        if head is not None:
            model,_=load_head(head);part=torch.load(data,map_location='cpu',weights_only=True)['test']
            records=[]
            for start in range(0,len(part['x']),96):
                fun=GaussianMixture.from_landscape(model(part['x'][start:start+96]))
                records.extend(numerical_parity(fun,part['y'][start:start+96].flatten(1).double()))
            result['teacher_head_sha256']=hashlib.sha256(Path(head).read_bytes()).hexdigest()
            result['forecast_data_sha256']=hashlib.sha256(Path(data).read_bytes()).hexdigest()
            result['test_origins']=len(part['x']);result['parity_records']=records
            result['timing']=[timing(GaussianMixture.from_landscape(model(part['x'][:b])),part['y'][:b].flatten(1).double()) for b in (1,64)]
        else:
            gen=torch.Generator().manual_seed(261410);means=torch.randn(4,3,6,generator=gen,dtype=torch.float64)
            raw=torch.randn(4,3,6,6,generator=gen,dtype=torch.float64)*.1
            fun=GaussianMixture(torch.zeros(4,3,dtype=torch.float64),means,torch.tril(raw,-1)+torch.eye(6)*.8)
            result['fixture_parity']=numerical_parity(fun,torch.zeros(4,6,dtype=torch.float64))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--head');p.add_argument('--data');p.add_argument('--out',required=True)
    a=p.parse_args()
    if bool(a.head)!=bool(a.data):p.error('--head and --data must be supplied together')
    out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    result=run(a.head,a.data);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'output':str(out),'regressions':result['constructed_regressions'],
                      'timing':[x['medians'] for x in result.get('timing',[])]},indent=2))


if __name__=='__main__':main()
