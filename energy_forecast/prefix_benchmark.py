"""Paired CPU benchmark of exact ordered versus general dense conditioning.

Dense uses reversed observation order to exercise the unmodified generic path.
This does not change which information is supplied or the remaining coordinates.
Times exclude the language model and forecast-head construction.
"""
from pathlib import Path
import argparse,json,random,time,statistics
import torch
from .conditioning import GaussianMixture
from .experiment import load_head


def run(head,data,out):
    torch.set_num_threads(1);m,_=load_head(head);part=torch.load(data,weights_only=True)['validation']
    records=[];summary=[];rng=random.Random(20260926)
    with torch.no_grad():
      for b in (1,64):
       for count in (8,16,24):
        f=GaussianMixture.from_landscape(m(part['x'][:b]));y=part['y'][:b].flatten(1).double()[:,:count]
        indices=list(reversed(range(count)));rev=y.flip(-1)
        calls={'ordered':lambda:f.condition_prefix(count,y),'dense':lambda:f.condition(indices,rev)}
        a=calls['ordered']();z=calls['dense']()
        errors={'mean_max':float((a.distribution.mean-z.distribution.mean).abs().max()),
                'covariance_max':float((a.distribution.component_covariance-z.distribution.component_covariance).abs().max()),
                'log_evidence_max':float((a.log_evidence-z.log_evidence).abs().max())}
        for fn in calls.values():
            for _ in range(4):fn()
        for rep in range(11):
            order=list(calls);rng.shuffle(order)
            for method in order:
                t=time.perf_counter()
                for _ in range(20):calls[method]()
                records.append({'batch':b,'observed_coordinates':count,'repeat':rep,'method':method,
                                'ms_per_call':(time.perf_counter()-t)*1000/20})
        for method in calls:
            vals=[r['ms_per_call'] for r in records if r['batch']==b and r['observed_coordinates']==count and r['method']==method]
            summary.append({'batch':b,'observed_coordinates':count,'method':method,'median_ms':statistics.median(vals),**errors})
    result={'scope':'one-thread CPU float64, same trained head and observations; not generation speed',
            'torch':str(torch.__version__),'summary':summary,'records':records}
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('head','data','out'):p.add_argument('--'+k,required=True)
    a=p.parse_args();run(a.head,a.data,a.out)
if __name__=='__main__':main()
