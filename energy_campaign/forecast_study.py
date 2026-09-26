"""Checkpoint-bound state forecasting and fixed-teacher joint comparisons."""
from dataclasses import asdict
from pathlib import Path
import copy
import json
import math
import time
import numpy as np
import torch
from energy_real.runtime import atomic_checkpoint,atomic_json,sha256
from energy_real.model import LM,Config
from energy_real.summaries import energy_layer
from energy_forecast.probe import trace_states,aligned_windows,load_energy_model
from energy_forecast.model import ForecastConfig,ForecastEnergy
from energy_forecast.experiment import train,load_head,measure
from .data import Documents
from .joint import JointForecastLM,frozen_targets
from .spec import source_identity


def extract(checkpoint,data,out,device='cpu',counts=(128,32,32),length=384):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    model=load_energy_model(checkpoint,device);result={};rng=np.random.default_rng(8721)
    for split,count in zip(('train','valid','test'),counts):
        corpus=Documents(data,split);docs=corpus.docs
        chosen=rng.choice(len(docs),min(count,len(docs)),replace=False);parts=[]
        for group,i in enumerate(chosen):
            d=docs[int(i)];raw=corpus.data[d['offset']:d['offset']+min(length,d['bytes'])]
            tokens=torch.tensor(np.asarray(raw,dtype=np.int64)[None],device=device)
            trace=trace_states(model,tokens)[0].cpu();windows=aligned_windows(trace,16,(1,4,16,64))
            if windows is None:continue
            x,y,anchors=windows;parts.append((x,y,torch.full((len(x),),group),anchors))
        if not parts:raise ValueError('Empty extraction split')
        result['validation' if split=='valid' else split]={key:torch.cat([r[i] for r in parts]) for i,key in enumerate(('x','y','group','anchor'))}
    center=result['train']['x'].flatten(0,1).mean(0);scale=result['train']['x'].flatten(0,1).std(0).clamp_min(.001)
    for part in result.values():part['x']=(part['x']-center)/scale;part['y']=(part['y']-center)/scale
    result['metadata']={'teacher_sha256':sha256(checkpoint),'source_sha256':source_identity(),'summary_layer':energy_layer(model.config),
                        'corpus_sha256':Documents(data).file_identity(),'history':16,'horizons':[1,4,16,64],
                        'variables':len(center),'center':center,'scale':scale,'model_config':asdict(model.config)}
    from energy_forecast.experiment import atomic_save
    atomic_save(result,out,overwrite=False);return result


def head_study(job):
    out=Path(job['out']);out.mkdir(parents=True,exist_ok=True);dataset=out/'data.pt'
    if not dataset.exists():extract(job['checkpoint'],job['data'],dataset,job.get('device','cuda'))
    from energy_forecast.repair import features,ridge_weights
    data=torch.load(dataset,weights_only=True,map_location='cpu');meta=data['metadata']
    if (meta['teacher_sha256']!=sha256(job['checkpoint']) or meta['source_sha256']!=source_identity()
        or meta['corpus_sha256']!=Documents(job['data']).file_identity()):raise ValueError('Stale teacher/source/corpus extraction')
    for kind in ('mixture','spectral'):
        for seed in (11,22,33):
            directory=out/f'{kind}_{seed}'
            if (directory/'training.json').exists():continue
            train(dataset,directory,kind,seed,steps=job.get('steps',600),device=job.get('device','cuda'),resume=(directory/'last.pt').exists())
    x=features(data['train']['x']);y=data['train']['y'].flatten(1).double();vx=features(data['validation']['x']);vy=data['validation']['y'].flatten(1).double()
    fits=[(float(((vx@ridge_weights(x,y,a)-vy)**2).mean()),a) for a in (.1,1.,10.,100.,1000.,10000.)]
    alpha=min(fits)[1];ridge=ridge_weights(x,y,alpha)
    selection={'alpha':alpha,'teacher_sha256':meta['teacher_sha256'],'data_sha256':sha256(dataset)}
    atomic_json(selection,out/'selection.json')
    tx=features(data['test']['x']);ty=data['test']['y'];rows=[]
    means={'constant':data['train']['y'].mean(0).expand_as(ty),'persistence':data['test']['x'][:,-1,None,:].expand_as(ty),
           'ridge':(tx@ridge).float().reshape_as(ty)}
    for name,pred in means.items():rows.append({'kind':name,'mse_by_horizon':(pred-ty).square().mean((0,2)).tolist()})
    for kind in ('mixture','spectral'):
        for seed in (11,22,33):
            model,_=load_head(out/f'{kind}_{seed}/best.pt','cpu')
            with torch.no_grad():
                density=model(data['test']['x']);sample=density.sample(512,generator=torch.Generator().manual_seed(77))
                lo,hi=sample.quantile(.05,dim=1),sample.quantile(.95,dim=1)
                rows.append({'kind':kind,'seed':seed,**measure(model,data['test']),
                             'mse_by_horizon':(density.mean.reshape_as(ty)-ty).square().mean((0,2)).tolist(),
                             'marginal_90pct_coverage':float(((ty.flatten(1)>=lo)&(ty.flatten(1)<=hi)).float().mean())})
    atomic_json({'selection':selection,'rows':rows,'scope':'summary forecasting only; no LM promotion'},out/'result.json');return rows


def joint_train(job):
    """One paired arm; all target traces come from an immutable teacher snapshot."""
    out=Path(job['out']);out.mkdir(parents=True,exist_ok=True);device=job.get('device','cuda');mode=job['mode']
    if mode not in ('continuation','residual','forecast'):raise ValueError('Invalid joint arm')
    source=torch.load(job['checkpoint'],map_location='cpu',weights_only=True);teacher=load_energy_model(job['checkpoint'],device)
    meta=torch.load(job['forecast_data'],map_location='cpu',weights_only=True)['metadata']
    if meta['teacher_sha256']!=sha256(job['checkpoint']):raise ValueError('Targets belong to another teacher')
    torch.manual_seed(job['seed']);backbone=LM(Config(**source['config'])).to(device);backbone.load_state_dict(source['weights'])
    model=JointForecastLM(backbone,meta['center'].to(device),meta['scale'].to(device),layer=meta['summary_layer']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,betas=(.9,.95),weight_decay=.1)
    rng=np.random.default_rng(job['seed']);corpus=Documents(job['data']);validation=Documents(job['data'],'valid')
    if meta['source_sha256']!=source_identity() or meta['corpus_sha256']!=corpus.file_identity():raise ValueError('Stale target extraction source/corpus')
    identity={'source':source_identity(),'teacher':sha256(job['checkpoint']),'data':corpus.file_identity(),
              'forecast_data':sha256(job['forecast_data']),'job':job}
    # Normalization, selected summary layer and source are bound by the extraction hash.
    cache=out.parent/'frozen_targets'/str(job['seed'])/identity['forecast_data'];cache.mkdir(parents=True,exist_ok=True)
    start=0;best=None;history=[]
    if (out/'last.pt').exists():
        d=torch.load(out/'last.pt',map_location='cpu',weights_only=True)
        if d['identity']!=identity:raise ValueError('Joint resume identity mismatch')
        model.load_state_dict(d['weights']);optimizer.load_state_dict(d['optimizer']);rng.bit_generator.state=d['rng'];start=d['step'];history=d['history'];best=d['best']
        torch.set_rng_state(d['torch_rng'])
        if device=='cuda':torch.cuda.set_rng_state_all(d['cuda_rng'])
    began=time.monotonic();steps=job.get('steps',64)
    for step in range(start,steps):
        if time.monotonic()-began>job['seconds']-20:break
        model.train();x,y=corpus.batch(rng,1,128,device);optimizer.zero_grad(set_to_none=True)
        key=__import__('hashlib').sha256(x.cpu().numpy().tobytes()+identity['teacher'].encode()).hexdigest();target_path=cache/(key+'.pt')
        if target_path.exists():targets=torch.load(target_path,map_location=device,weights_only=True)
        else:
            targets=frozen_targets(teacher,x,model.center,model.scale,model.layer);atomic_checkpoint(targets.cpu(),target_path)
        if mode=='continuation':logits,_=backbone(x);aux=logits.sum()*0
        else:logits,_,aux=model(x,future_targets=targets)
        ce=torch.nn.functional.cross_entropy(logits.flatten(0,1),y.flatten());loss=ce+(.1*aux if mode=='forecast' else 0)
        if not torch.isfinite(loss):raise FloatingPointError('Joint loss')
        auxiliary_norm=0.
        if mode=='forecast':
            grads=torch.autograd.grad(.1*aux,tuple(backbone.parameters()),retain_graph=True,allow_unused=True)
            auxiliary_norm=float(sum(g.float().square().sum() for g in grads if g is not None).sqrt())
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        history.append({'step':step+1,'ce':float(ce.detach()),'forecast_nll':float(aux.detach()),'auxiliary_backbone_gradient_norm':auxiliary_norm,'total_gradient_norm':float(norm)})
        if (step+1)%8==0 or step+1==steps:
            model.eval();total=0.;count=0
            with torch.no_grad():
                for vx,vy in validation.sequential_windows(128,limit=1024,device=device):
                    ly=backbone(vx)[0] if mode=='continuation' else model(vx)[0]
                    total+=float(torch.nn.functional.cross_entropy(ly.flatten(0,1),vy.flatten(),reduction='sum'));count+=vy.numel()
            score=total/count/math.log(2)
            if best is None or score<best:
                best=score;atomic_checkpoint({'weights':model.state_dict(),'config':source['config'],'metadata':meta,'mode':mode,'valid_bpb':score,'identity':identity},out/'best.pt')
            atomic_checkpoint({'identity':identity,'weights':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':rng.bit_generator.state,
                               'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else [],
                               'step':step+1,'history':history,'best':best},out/'last.pt')
    atomic_json({'mode':mode,'steps':len(history),'planned_steps':steps,'status':'completed' if len(history)==steps else 'interrupted','best_valid_bpb':best,'history':history},out/'result.json')


def profile_joint(job):
    device=job.get('device','cuda');source=torch.load(job['checkpoint'],map_location='cpu',weights_only=True)
    teacher=load_energy_model(job['checkpoint'],device);meta=torch.load(job['forecast_data'],map_location='cpu',weights_only=True)['metadata']
    backbone=LM(Config(**source['config'])).to(device);backbone.load_state_dict(source['weights'])
    model=JointForecastLM(backbone,meta['center'].to(device),meta['scale'].to(device),layer=meta['summary_layer']).to(device)
    corpus=Documents(job['data']);rng=np.random.default_rng(908);optimizer=torch.optim.AdamW(model.parameters(),lr=.0003);times=[]
    for _ in range(3):
        start=time.monotonic();x,y=corpus.batch(rng,1,128,device);targets=frozen_targets(teacher,x,model.center,model.scale,model.layer)
        optimizer.zero_grad(set_to_none=True);logits,_,aux=model(x,future_targets=targets)
        ce=torch.nn.functional.cross_entropy(logits.flatten(0,1),y.flatten());(ce+.1*aux).backward();optimizer.step()
        if device=='cuda':torch.cuda.synchronize()
        times.append(time.monotonic()-start)
    result={'source_sha256':source_identity(),'device':device,'seconds_per_update':max(times),'samples':times,'includes_frozen_target_extraction':True}
    atomic_json(result,job['out']);return result


def evaluate_joint(job):
    root=Path(job['root']);rows=[];device=job.get('device','cuda');corpus=Documents(job['data'],'test')
    for seed in (404,505,606):
        for mode in ('continuation','residual','forecast'):
            saved=torch.load(root/f'joint/{seed}_{mode}/best.pt',map_location='cpu',weights_only=True);meta=saved['metadata']
            backbone=LM(Config(**saved['config'])).to(device)
            model=JointForecastLM(backbone,meta['center'].to(device),meta['scale'].to(device),layer=meta['summary_layer']).to(device)
            model.load_state_dict(saved['weights']);model.eval();total=0.;count=0;start=time.monotonic()
            with torch.no_grad():
                for x,y in corpus.sequential_windows(128,limit=4096,device=device):
                    logits=backbone(x)[0] if mode=='continuation' else model(x)[0]
                    total+=float(torch.nn.functional.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='sum'));count+=y.numel()
            rows.append({'seed':seed,'mode':mode,'test_bpb':total/count/math.log(2),'targets':count,'seconds':time.monotonic()-start})
    result={'rows':rows,'promotion':False,'scope':'small paired joint pilot; recall and broader replication still required'}
    atomic_json(result,job['out']);return result
