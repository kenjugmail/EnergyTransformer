"""Demonstrate fixed-target forecasting from genuine arriving byte observations.

Uses the bundled old 973K teacher unless paths are supplied. Never substitutes
forecasts for observations or skips bytes; no corpus is needed for this demo.
"""
from pathlib import Path
import argparse,json
import torch
from .online import ObservedForecastSession


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher',default='checkpoints/teacher_973k.pt')
    p.add_argument('--head',default='results/repair/fits/anchored_seed11.pt')
    p.add_argument('--gate',default='results/online_gate/gate.json')
    a=p.parse_args();torch.set_num_threads(1)
    s=ObservedForecastSession(a.teacher,a.head,gate_path=a.gate)
    prefix=b'The next observations may change a forecast.'
    continuation=b' Each new byte is processed by the actual frozen model before it can be used as evidence.'
    s.observe(torch.tensor([list(prefix)],dtype=torch.long));s.issue()
    previous=0
    for age in (0,1,4,16,64):
        if age>previous:s.observe(torch.tensor([list(continuation[previous:age])],dtype=torch.long))
        status=s.status();record={'age_bytes':age,**status,'raw_state_bytes':s.raw_state_bytes()}
        if status['status']=='active':
            fun=s.distribution();record['mean_shape']=list(fun.mean.shape)
            record['mean_of_pending_means']=float(fun.mean.mean())
            record['energy_at_mean']=float(fun.energy(fun.mean))
        print(json.dumps(record,sort_keys=True));previous=age
    print('Complete: 64 real bytes processed, no forecasting-based token skipping, no weights updated.')
if __name__=='__main__':main()
