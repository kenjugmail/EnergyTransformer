"""Oracle-input memory comparisons, including source selection and stale facts."""
from __future__ import annotations
import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import tempfile
import time
import numpy as np
from .episodes import generate,oracle
from .episodic import FactStore
from .runtime import atomic_json,source_identity


def paired_interval(a,b,seed=144,draws=2000):
    """Paired bootstrap over episodes; run-seed uncertainty is reported separately."""
    if len(a)!=len(b) or not a:raise ValueError('Paired nonempty samples required')
    d=np.asarray(a,dtype=float)-np.asarray(b,dtype=float)
    rng=np.random.default_rng(seed)
    samples=d[rng.integers(0,len(d),(draws,len(d)))].mean(1)
    return {'difference':float(d.mean()),'low':float(np.quantile(samples,.025)),
            'high':float(np.quantile(samples,.975)),'unit':'episode','draws':draws}


def run(episodes,*,backends=('exact','diagonal','precision','delta','preconditioned_delta','projected'),
        controllers=None,dimension=16,capacity=256,max_seconds=300):
    controllers=controllers or {};rows=[];start=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='energy-memory-eval-') as directory:
        for episode in episodes:
            expected,support=oracle(episode)
            for backend in backends:
                if time.perf_counter()-start>=max_seconds:
                    return summarize(rows,status='budget_exhausted')
                path=Path(directory)/f'{episode.id}_{backend}.sqlite'
                store=FactStore(path,backend=backend,controller=controllers.get(backend),dimension=dimension,capacity=capacity)
                store.apply(episode.edits,expected_revision=0,request_id='episode',authorized=True)
                # Build approximate state before timing query, and report build separately.
                t0=time.perf_counter();first=store.query(episode.query);build=time.perf_counter()-t0
                t0=time.perf_counter();got=store.query(episode.query);latency=time.perf_counter()-t0
                _,active,history=store.snapshot()
                current_ids={f['fact_id'] for f in active}
                expected_ids=[history[i]['fact_id'] for i in support]
                retrieved_ids=[f['fact_id'] for f in got['facts']]
                state_bytes=(store._cached[1].state_bytes if store._cached else
                             len(json.dumps(active).encode()))
                rows.append({'episode_id':episode.id,'family':episode.family,'backend':backend,
                             'expected':expected,'answer':got['answer'],'correct':got['answer']==expected,
                             'expected_abstention':expected is None,'abstained':got['abstained'],
                             'support_correct':retrieved_ids==expected_ids,
                             'stale_facts':sum(i not in current_ids for i in retrieved_ids),
                             'latency_s':latency,'build_and_first_query_s':build,'state_bytes':state_bytes,
                             'retrieved_fact_ids':retrieved_ids,'expected_fact_ids':expected_ids})
    return summarize(rows)


def summarize(rows,status='completed'):
    groups=defaultdict(list)
    for r in rows:groups[r['backend']].append(r)
    summary={}
    for arm,records in groups.items():
        summary[arm]={'episodes':len(records),'accuracy':statistics.mean(r['correct'] for r in records),
                      'support_accuracy':statistics.mean(r['support_correct'] for r in records),
                      'stale_facts':sum(r['stale_facts'] for r in records),
                      'median_query_ms':statistics.median(r['latency_s'] for r in records)*1000,
                      'median_state_bytes':statistics.median(r['state_bytes'] for r in records),
                      'state_accounting':'raw tensors' if arm!='exact' else 'serialized active facts; excludes database/history',
                      'families':{family:statistics.mean(r['correct'] for r in records if r['family']==family)
                                  for family in sorted({r['family'] for r in records})}}
    return {'status':status,'input_mode':'oracle_structured_edits','source_sha256':source_identity(),
            'summary':summary,'rows':rows,
            'limitations':['Registered finite episodes; not general-language reasoning.',
                          'Approximate state excludes provenance registry, model weights and event history; not total system memory.',
                          'Latency is CPU Python implementation timing, not a fused-kernel comparison.',
                          'No promotion from a single controller seed or unmatched storage accounting.']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=1001)
    p.add_argument('--per-family',type=int,default=10)
    p.add_argument('--split',choices=['train','valid','test'],default='test')
    p.add_argument('--max-seconds',type=float,default=300)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    episodes=generate(args.seed,args.per_family,args.split)
    result=run(episodes,max_seconds=args.max_seconds)
    result['episode_manifest']=[e.payload() for e in episodes]
    atomic_json(result,args.out)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','episode_manifest')},indent=2))


if __name__=='__main__':main()
