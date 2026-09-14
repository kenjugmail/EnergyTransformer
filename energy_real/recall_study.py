"""Bounded, validation-selected recall learning on the shared relation benchmark.

CPU screening uses twelve configurations, followed by two additional seeds for
each validation winner. All selection/calibration precedes fresh held-out tests.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import os
import statistics
import subprocess
import sys
import tempfile
import time
import torch
import torch.nn.functional as F
from energy_transformer.geometric_memory import EditMemory
from . import diagnostics as data
from .episodes import oracle
from .episodic import EnergyRetriever, FactStore, Query, canonical, key_vector
from .memory_eval import paired_interval
from .retrieval import rank_candidates, accepts
from .runtime import atomic_checkpoint, atomic_json, sha256, source_identity

KINDS = ('precision','preconditioned_delta')
DIMENSIONS = (16,32,64)
PRIORS = (.01,.001)
THRESHOLDS = tuple((score,margin) for score in (.05,.15,.3,.5,.7) for margin in (0.,.02,.1,.2))
CAPACITY = 256
FORMAT = 'relation_recall_controller_v2'


class BudgetExpired(RuntimeError):
    pass


def check_time(deadline):
    if time.perf_counter() >= deadline:
        raise BudgetExpired('Recall study reached its wall-time ceiling')


def new_model(kind, dimension, prior, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return EditMemory(kind,dimension,CAPACITY,prior=prior)


def tensors(episode, dimension):
    events = data.events_for(episode)
    addresses = list(dict.fromkeys((e['subject'],e['relation']) for e in events))
    # An unseen key is always queried, including in otherwise answerable episodes.
    addresses += [(episode.query.subject+'_unseen',episode.query.relations[0])]
    if (episode.query.subject,episode.query.relations[0]) not in addresses:
        addresses.append((episode.query.subject,episode.query.relations[0]))
    ids = {key:i for i,key in enumerate(addresses)}
    keys = torch.stack([key_vector(e['subject'],e['relation'],dimension) for e in events])
    values = torch.eye(CAPACITY)[:len(events)]
    operations = torch.tensor([{'set':0,'delete':1,'hold':2}[e['operation']] for e in events])
    queries = torch.stack([key_vector(*key,dimension) for key in addresses])
    active = data.active_records(events)
    positions = {e['fact_id']:i+1 for i,e in enumerate(events)}
    targets = torch.tensor([positions[active[key]['fact_id']] if key in active else 0 for key in addresses])
    edited_ids = torch.tensor([ids[e['subject'],e['relation']] for e in events])
    return keys,values,operations,queries,targets,edited_ids


def recall_loss(model, episode):
    keys,values,operations,queries,targets,edited_ids = tensors(episode,model.dimension)
    state = model.initial(1)
    queries = queries[None]
    locality = []
    for index in range(len(keys)):
        before = model.read(queries,state)
        state = model.step(keys[index:index+1],values[index:index+1],operations[index:index+1],state)
        if operations[index] != 2:
            after = model.read(queries,state)
            unchanged = torch.arange(queries.shape[1]) != edited_ids[index]
            locality.append((after-before.detach()).square().mean(-1)[0,unchanged].mean())
    scores = model.read(queries,state)[0]
    # Null has a fixed score; unavailable/deleted addresses explicitly train abstention.
    logits = torch.cat((torch.full_like(scores[:,:1],.5),scores),-1)*4.
    delayed = F.cross_entropy(logits,targets)
    local = torch.stack(locality).mean() if locality else delayed.new_zeros(())
    return delayed+.1*local, {'delayed_ce':float(delayed.detach()),'locality_mse':float(local.detach())}


class Readout:
    """One immutable state; raw predictions cached across confidence calibrations.

    Cache keys depend only on the query, never on expected answers or active facts.
    """
    def __init__(self, episode, model):
        self.episode=episode;self.events=data.events_for(episode);self.model=model
        self.records={i:e for i,e in enumerate(self.events) if e['operation']=='set'}
        self.state=model.initial(1);self.cache={}
        start=time.perf_counter()
        with torch.no_grad():
            for index,event in enumerate(self.events):
                value=torch.zeros(1,CAPACITY)
                if event['operation']=='set':value[0,index]=1.
                key=key_vector(event['subject'],event['relation'],model.dimension)[None]
                op=torch.tensor([{'set':0,'delete':1,'hold':2}[event['operation']]])
                self.state=model.step(key,value,op,self.state)
        self.build_s=time.perf_counter()-start

    @torch.no_grad()
    def candidates(self,subject,relation):
        key=(subject,relation)
        if key not in self.cache:
            query=key_vector(subject,relation,self.model.dimension)[None,None]
            scores=self.model.read(query,self.state)[0,0]
            value=rank_candidates(scores,self.records)
            self.cache[key]=value
        return self.cache[key]

    def query(self,threshold):
        subject=self.episode.query.subject;facts=[];trace=[]
        score_limit,margin_limit=threshold
        for relation in self.episode.query.relations:
            row=self.candidates(subject,relation)
            accepted=accepts(row,score_limit,margin_limit)
            trace.append({'hop':len(trace)+1,'subject':subject,'relation':relation,
                          'score':row['score'],'margin':row['margin'],'accepted':accepted,'finite':row['finite'],
                          'candidate_fact_ids':[f['fact_id'] for f in row['candidates']],
                          'selected_fact_id':row['candidates'][0]['fact_id'] if accepted else None})
            if not accepted:return {'answer':None,'abstained':True,'facts':facts,'trace':trace}
            fact=row['candidates'][0];facts.append(fact);subject=fact['object']
        return {'answer':subject,'abstained':False,'facts':facts,'trace':trace}


def measure(episode,answer,events):
    expected,support=oracle(episode)
    wanted=[events[i]['fact_id'] for i in support]
    actual=[f['fact_id'] for f in answer['facts']]
    active={e['fact_id'] for e in data.active_records(events).values()}
    stale=sum(i not in active for i in actual)
    confidence_rejections=wrong_candidates=0
    for hop,trace in enumerate(answer['trace']):
        if hop>=len(wanted) or 'candidate_fact_ids' not in trace:continue
        candidates=trace['candidate_fact_ids']
        right_top=bool(candidates) and candidates[0]==wanted[hop]
        confidence_rejections+=int(right_top and not trace['accepted'])
        wrong_candidates+=int(not right_top)
    return {'episode_id':episode.id,'family':episode.family,'expected':expected,'answer':answer['answer'],
            'correct':answer['answer']==expected,'support_correct':actual==wanted,
            'abstained':answer['abstained'],'expected_abstention':expected is None,
            'unsupported_answer':expected is None and not answer['abstained'],
            'false_abstention':expected is not None and answer['abstained'],
            'retrieval_failure':actual!=wanted,'stale_facts':stale,
            'confidence_rejections':confidence_rejections,'wrong_top_candidates':wrong_candidates,
            'expected_fact_ids':wanted,'retrieved_fact_ids':actual,'trace':answer['trace']}


def summary(rows):
    if not rows:return {'episodes':0}
    return {'episodes':len(rows),'accuracy':statistics.mean(r['correct'] for r in rows),
            'support_accuracy':statistics.mean(r['support_correct'] for r in rows),
            'unsupported_answers':sum(r['unsupported_answer'] for r in rows),
            'false_abstentions':sum(r['false_abstention'] for r in rows),
            'stale_facts':sum(r['stale_facts'] for r in rows),
            'confidence_rejections':sum(r['confidence_rejections'] for r in rows),
            'wrong_top_candidates':sum(r['wrong_top_candidates'] for r in rows),
            'families':{family:statistics.mean(r['correct'] for r in rows if r['family']==family)
                        for family in sorted({r['family'] for r in rows})}}


def calibrate(model, episodes, deadline):
    if any(e.split!='valid' for e in episodes):raise ValueError('Confidence calibration requires validation episodes')
    accum={threshold:[] for threshold in THRESHOLDS}
    for episode in episodes:
        check_time(deadline)
        readout=Readout(episode,model)
        for threshold in THRESHOLDS:
            accum[threshold].append(measure(episode,readout.query(threshold),readout.events))
    records=[{'score_threshold':t[0],'margin_threshold':t[1],**summary(rows)} for t,rows in accum.items()]
    # Accuracy includes unavailable examples; ties favor fewer unsupported answers,
    # then stricter abstention. Gold labels never enter model reads.
    best=max(records,key=lambda r:(r['accuracy'],-r['unsupported_answers'],r['support_accuracy'],r['score_threshold'],r['margin_threshold']))
    return {'selected':best,'grid':records}


def save_controller(model,path,seed,selected_step):
    atomic_checkpoint({'format':FORMAT,'kind':model.kind,'dimension':model.dimension,'value_dim':CAPACITY,
                       'prior':model.prior,'seed':seed,'selected_step':selected_step,
                       'objective':'delayed_ce_plus_0.1_locality','weights':model.state_dict()},path)


def fit(kind,dimension,prior,seed,steps,out,train,valid,deadline):
    out.mkdir(parents=True,exist_ok=False)
    model=new_model(kind,dimension,prior,seed)
    initial=calibrate(model,valid,deadline)
    best=initial;best_step=0;weights={k:v.detach().clone() for k,v in model.state_dict().items()}
    save_controller(model,out/'initial.pt',seed,0)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.001)
    rng=random.Random(seed)
    families={family:[e for e in train if e.family==family] for family in data.FAMILIES}
    history=[];started=time.perf_counter()
    for step in range(1,steps+1):
        check_time(deadline)
        episode=rng.choice(families[data.FAMILIES[(step-1)%len(data.FAMILIES)]])
        optimizer.zero_grad();loss,parts=recall_loss(model,episode)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite recall loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if step%100==0 or step==steps:
            score=calibrate(model,valid,deadline)
            history.append({'step':step,'loss':float(loss.detach()),**parts,'validation':score['selected']})
            key=lambda r:(r['selected']['accuracy'],-r['selected']['unsupported_answers'],r['selected']['support_accuracy'])
            if key(score)>key(best):
                best=score;best_step=step;weights={k:v.detach().clone() for k,v in model.state_dict().items()}
            atomic_json({'status':'in_progress','step':step,'history':history},out/'progress.json')
    model.load_state_dict(weights)
    save_controller(model,out/'controller.pt',seed,best_step)
    record={'status':'completed','kind':kind,'dimension':dimension,'prior':prior,'capacity':CAPACITY,'seed':seed,
            'steps':steps,'selected_step':best_step,'initial_validation':initial,'validation':best,'history':history,
            'controller':str(out/'controller.pt'),'initial_controller':str(out/'initial.pt'),
            'controller_sha256':sha256(out/'controller.pt'),'initial_sha256':sha256(out/'initial.pt'),
            'seconds':time.perf_counter()-started,'objective':'delayed_ce_plus_0.1_locality'}
    atomic_json(record,out/'result.json')
    print(json.dumps({k:record[k] for k in ('kind','dimension','prior','seed','selected_step','seconds')},sort_keys=True),flush=True)
    return record


def evaluate_config(record,episodes,deadline,*,initial=False):
    checkpoint=record['initial_controller'] if initial else record['controller']
    model=new_model(record['kind'],record['dimension'],record['prior'],record['seed'])
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(saved['weights']);model.eval()
    calibration=record['initial_validation' if initial else 'validation']['selected']
    threshold=(calibration['score_threshold'],calibration['margin_threshold'])
    rows=[]
    for episode in episodes:
        check_time(deadline)
        readout=Readout(episode,model)
        row=measure(episode,readout.query(threshold),readout.events)
        row['default_threshold']=measure(episode,readout.query((.5,0.)),readout.events)
        rows.append(row)
    return {'summary':summary(rows),'rows':rows,'calibration':calibration,
            'default_threshold_summary':summary([r['default_threshold'] for r in rows])}


def cost_comparison(records,episodes,deadline):
    """Use the same durable store and identical retained evidence for every arm.

    Storage counts include DB, serialized resident snapshot/registry, parameter and
    state tensor bytes. Python allocator overhead is explicitly not measured.
    """
    rows=[]
    with tempfile.TemporaryDirectory(prefix='recall-cost-') as directory:
        for index,episode in enumerate(episodes):
            arms=[('exact',None),*[(r['kind'],r) for r in records]]
            offset=index%len(arms);arms=arms[offset:]+arms[:offset]
            for name,record in arms:
                check_time(deadline)
                config={} if record is None else {
                    'controller':record['controller'],'dimension':record['dimension'],'prior':record['prior'],
                    **{key:record['validation']['selected'][key] for key in ('score_threshold','margin_threshold')}}
                store=FactStore(Path(directory)/f'{episode.id}_{name}.sqlite',backend=name,**config)
                start=time.perf_counter()
                store.apply(episode.edits,expected_revision=0,request_id='episode',authorized=True)
                apply_s=time.perf_counter()-start
                start=time.perf_counter();answer=store.query(episode.query);first_s=time.perf_counter()-start
                times=[]
                for _ in range(3):
                    start=time.perf_counter();store.query(episode.query);times.append(time.perf_counter()-start)
                _,active,events=store.snapshot()
                snapshot_bytes=len(canonical({'active':active,'history':events}).encode())
                state_bytes=model_bytes=registry_bytes=0
                if store._cached:
                    retriever=store._cached[1]
                    state_bytes=retriever.state_bytes
                    model_bytes=sum(p.numel()*p.element_size() for p in retriever.model.parameters())
                    registry_bytes=len(canonical(retriever.records).encode())
                disk_bytes=sum(p.stat().st_size for p in store.path.parent.glob(store.path.name+'*'))
                rows.append({'episode_id':episode.id,'backend':name,'apply_s':apply_s,'build_first_query_s':first_s,
                             'median_query_s':statistics.median(times),'database_bytes':disk_bytes,
                             'snapshot_bytes':snapshot_bytes,'registry_bytes':registry_bytes,
                             'model_bytes':model_bytes,'state_bytes':state_bytes,
                             'accounted_total_bytes':disk_bytes+snapshot_bytes+registry_bytes+model_bytes+state_bytes,
                             **measure(episode,answer,events)})
    return {'rows':rows,'accounting':'Database disk bytes plus serialized resident evidence, registry, model and recurrent tensors.',
            'limitations':['Disk and resident bytes are separate components, not peak RAM.',
                           'Python/container/SQLite allocator overhead and process RSS are not measured.',
                           'Shared model parameters counted per independent arm.',
                           'CPU timings only; no GPU efficiency promotion.']}


def peak_worker(record,episodes):
    """Isolated-process resident peak includes Python, native code, DB and tensors."""
    def peak_rss():
        # VmHWM belongs to this exec's address space. getrusage().ru_maxrss
        # may retain the much larger pre-exec parent's high-water mark.
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmHWM:'):return int(line.split()[1])*1024
        raise RuntimeError('Linux process high-water RSS is unavailable')
    baseline=peak_rss()
    # Keep only one store alive at a time, matching query service usage.
    with tempfile.TemporaryDirectory(prefix='recall-rss-') as directory:
        for episode in episodes:
            config={} if record is None else {
                'controller':record['controller'],'dimension':record['dimension'],'prior':record['prior'],
                **{key:record['validation']['selected'][key] for key in ('score_threshold','margin_threshold')}}
            store=FactStore(Path(directory)/f'{episode.id}.sqlite',backend='exact' if record is None else record['kind'],**config)
            store.apply(episode.edits,expected_revision=0,request_id='episode',authorized=True)
            store.query(episode.query)
            del store
    return {'backend':'exact' if record is None else record['kind'],
            'process_peak_rss_bytes':peak_rss(),
            'import_peak_rss_bytes':baseline,'scope':'Linux /proc/self/status VmHWM after exec; includes runtime, native allocations and DB.'}


def measure_peaks(records,out,deadline):
    rows=[]
    for index,record in enumerate([None,*records]):
        check_time(deadline)
        payload=out/f'peak_input_{index}.json'
        atomic_json({'record':record},payload)
        try:
            process=subprocess.run([sys.executable,'-m','energy_real.recall_study','--peak-worker',str(payload)],
                                   capture_output=True,text=True,check=True,timeout=max(.01,deadline-time.perf_counter()),
                                   env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
        except subprocess.TimeoutExpired as exc:
            raise BudgetExpired('Process memory measurement reached the study ceiling') from exc
        rows.append(json.loads(process.stdout))
    return rows


def evaluate_exact(episodes,deadline):
    rows=[]
    with tempfile.TemporaryDirectory(prefix='recall-exact-') as directory:
        for episode in episodes:
            check_time(deadline)
            store=FactStore(Path(directory)/f'{episode.id}.sqlite')
            store.apply(episode.edits,expected_revision=0,request_id='episode',authorized=True)
            rows.append(measure(episode,store.query(episode.query),store.snapshot()[2]))
    return {'summary':summary(rows),'rows':rows}


def evaluate_baseline(kind,episodes,deadline):
    model=new_model(kind,16,.01,11);rows=[]
    for episode in episodes:
        check_time(deadline)
        readout=Readout(episode,model)
        rows.append(measure(episode,readout.query((.5,0.)),readout.events))
    return {'kind':kind,'dimension':16,'prior':.01,'seed':11,'score_threshold':.5,'margin_threshold':0.,
            'summary':summary(rows),'rows':rows,'label':'Original configuration on the fresh v2 test set; untrained.'}


def selected_memory(path,kind='precision',seed=11,artifact_root=None):
    """Resolve only a frozen, hashed validation selection for practical evaluation."""
    path=Path(path)
    frozen=json.loads(path.read_text())
    if frozen.get('status')!='frozen_for_test':raise ValueError('A frozen memory selection is required')
    if sha256(path.parent/'datasets.json')!=frozen['dataset_sha256']:
        raise ValueError('Validation dataset manifest checksum changed')
    selected=[r for r in frozen['selected'] if r['kind']==kind and r['seed']==seed]
    if len(selected)!=1:raise ValueError('Selected memory kind/seed is missing or ambiguous')
    record=selected[0]
    from .artifacts import resolve_artifact
    controller=resolve_artifact(record['controller'],record['controller_sha256'],path,artifact_root)
    return {'controller':controller,'dimension':record['dimension'],'capacity':record['capacity'],
            'prior':record['prior'],**{k:record['validation']['selected'][k] for k in ('score_threshold','margin_threshold')}}


def run(args):
    args.out=args.out.resolve()
    if args.out.exists():raise FileExistsError(args.out)
    if not 1<=args.steps<=500 or args.max_seconds<=0 or not math.isfinite(args.max_seconds):
        raise ValueError('Use 1–500 steps and a finite positive time limit')
    if args.max_seconds>14400:raise ValueError('Controller stage ceiling is four hours')
    if min(args.train_per_family,args.valid_per_family,args.test_per_family,args.threads)<1:
        raise ValueError('Positive dataset sizes and thread count required')
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True)
    started=time.perf_counter();deadline=started+args.max_seconds
    identity=source_identity();records=[];selected=[];tests=[];status='running';selection_sha256=None
    train=data.generate('train',args.train_per_family)
    valid=data.generate('valid',args.valid_per_family)
    # Record the split recipe before selection; test episodes are instantiated later.
    train_manifest=data.manifest(train);valid_manifest=data.manifest(valid)
    atomic_json(train_manifest,args.out/'train_episodes.json')
    atomic_json(valid_manifest,args.out/'valid_episodes.json')
    atomic_json({'version':data.VERSION,
                 'train':{'sha256':train_manifest['sha256'],'episodes':len(train),'per_family':args.train_per_family,'seed':data.SPLIT_SEEDS['train']},
                 'valid':{'sha256':valid_manifest['sha256'],'episodes':len(valid),'per_family':args.valid_per_family,'seed':data.SPLIT_SEEDS['valid']},
                 'test_recipe':{'split':'test','per_family':args.test_per_family,'seed':data.SPLIT_SEEDS['test']}},args.out/'datasets.json')
    def progress():
        atomic_json({'status':status,'source_sha256':identity,'records':records,'selected':selected,
                     'test_results':tests,'elapsed_s':time.perf_counter()-started,
                     'max_seconds':args.max_seconds,'gpu_seconds':0,'selection_sha256':selection_sha256},args.out/'summary.json')
    progress()
    try:
        for kind in KINDS:
            for dimension in DIMENSIONS:
                for prior in PRIORS:
                    record=fit(kind,dimension,prior,11,args.steps,args.out/f'{kind}_d{dimension}_p{prior}_seed11',train,valid,deadline)
                    records.append(record);progress()
        winners={}
        for kind in KINDS:
            winner=max((r for r in records if r['kind']==kind),
                       key=lambda r:(r['validation']['selected']['accuracy'],-r['validation']['selected']['unsupported_answers'],
                                     r['validation']['selected']['support_accuracy'],-r['dimension'],r['prior']))
            winners[kind]=winner
            selected.append(winner)
        atomic_json({'status':'frozen_configurations','source_sha256':identity,
                     'selected':{kind:{k:r[k] for k in ('kind','dimension','prior','capacity')} for kind,r in winners.items()}},
                    args.out/'configuration_selection.json')
        for seed in (22,33):
            for kind in KINDS:
                winner=winners[kind]
                record=fit(kind,winner['dimension'],winner['prior'],seed,args.steps,
                           args.out/f'{kind}_d{winner["dimension"]}_p{winner["prior"]}_seed{seed}',train,valid,deadline)
                records.append(record);selected.append(record);progress()
        frozen={'status':'frozen_for_test','source_sha256':identity,'dataset_sha256':sha256(args.out/'datasets.json'),
                'selected':selected,'selection_rule':'Validation accuracy, unsupported answers, support accuracy, smaller dimension.',
                'test_recipe':{'split':'test','per_family':args.test_per_family,'seed':data.SPLIT_SEEDS['test']}}
        atomic_json(frozen,args.out/'selection.json')
        selection_sha256=sha256(args.out/'selection.json')
        test=data.generate('test',args.test_per_family)
        atomic_json(data.manifest(test),args.out/'test_episodes.json')
        atomic_json(evaluate_exact(test,deadline),args.out/'exact_oracle.json')
        for kind in KINDS:
            atomic_json(evaluate_baseline(kind,test,deadline),args.out/f'baseline_{kind}.json')
        for record in selected:
            tested={'kind':record['kind'],'seed':record['seed'],
                    'trained':evaluate_config(record,test,deadline),
                    'initial':evaluate_config(record,test,deadline,initial=True)}
            tested['learning_interval']=paired_interval([r['correct'] for r in tested['trained']['rows']],
                                                        [r['correct'] for r in tested['initial']['rows']])
            tests.append(tested);progress()
        # Chosen seed 11 representatives; never choose the best test seed.
        cost=cost_comparison([winners[k] for k in KINDS],data.generate('test',2),deadline)
        cost['isolated_process_memory']=measure_peaks([winners[k] for k in KINDS],args.out,deadline)
        atomic_json(cost,args.out/'costs.json')
        arrays={kind:torch.tensor([[r['correct'] for r in t['trained']['rows']] for t in tests if t['kind']==kind],dtype=torch.float64)
                for kind in KINDS}
        interval=paired_interval(arrays['precision'].mean(0).tolist(),arrays['preconditioned_delta'].mean(0).tolist())
        gains=(arrays['precision'].mean(1)-arrays['preconditioned_delta'].mean(1)).tolist()
        promoted=interval['difference']>=.05 and interval['low']>0 and min(gains)>0
        decision={'decision':'candidate_for_replication' if promoted else 'retain_bounded_result',
                  'paired_interval':interval,'per_seed_differences':gains,
                  'learning_claims':{f'{t["kind"]}_{t["seed"]}':t['learning_interval']['low']>0 for t in tests},
                  'efficiency_gate':'not established; CPU component accounting is not measured end-to-end peak memory or GPU latency'}
        atomic_json(decision,args.out/'decision.json')
        if source_identity()!=identity:raise ValueError('Source changed during the study; retain results as incomplete')
        status='completed'
    except BudgetExpired as exc:
        status='budget_exhausted';atomic_json({'status':status,'reason':str(exc)},args.out/'incomplete.json')
    except BaseException as exc:
        status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed'
        atomic_json({'status':status,'reason':f'{type(exc).__name__}: {exc}'},args.out/'incomplete.json')
        raise
    finally:
        progress()
    return json.loads((args.out/'summary.json').read_text())


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=500)
    p.add_argument('--max-seconds',type=float,default=14400)
    p.add_argument('--train-per-family',type=int,default=32)
    p.add_argument('--valid-per-family',type=int,default=10)
    p.add_argument('--test-per-family',type=int,default=30)
    p.add_argument('--threads',type=int,default=2)
    return p


if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--peak-worker':
        torch.set_num_threads(2)
        payload=json.loads(Path(sys.argv[2]).read_text())
        print(json.dumps(peak_worker(payload['record'],data.generate('test',2))))
    else:
        from .stages import interruption_handler
        with interruption_handler():
            result=run(parser().parse_args())
        print(json.dumps({k:v for k,v in result.items() if k not in ('records','selected','test_results')},indent=2))
