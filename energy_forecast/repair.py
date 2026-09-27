"""Validation-gated follow-up. Fit/calibrate first, freeze hashes, test separately.

The original test domain has already been inspected. These are development results.
CPU only by default; no downloads, remote jobs, LM updates, or automatic writes.
"""
from __future__ import annotations
import argparse, hashlib, json, math, time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from .model import ForecastConfig, EnergyLandscape
from .experiment import load_head, atomic_save, digest
from .calibrated import HorizonAnchor, AnchoredForecast, RecenteredForecast

ALPHAS = (.1, 1., 10., 100., 1000., 10000., 100000., 1000000.)


def split_validation(groups):
    unique = groups.unique().tolist()
    ordered = sorted(unique, key=lambda g: hashlib.sha256(f'forecast-repair:{g}'.encode()).hexdigest())
    if len(ordered) < 4: raise ValueError('At least four independent validation groups required')
    select = set(ordered[:len(ordered)//2]); mask = torch.tensor([g in select for g in groups.tolist()])
    return mask, ~mask


def subset(part, mask):
    return {k: v[mask] for k, v in part.items() if isinstance(v, torch.Tensor)}


def features(x):
    x = x.flatten(1).double()
    return torch.cat((x, torch.ones(len(x), 1, dtype=torch.float64)), -1)


def ridge_weights(x, y, alpha):
    regularizer = torch.eye(x.shape[1], dtype=x.dtype)
    regularizer[-1, -1] = 0
    return torch.linalg.solve(x.T @ x + alpha * regularizer, x.T @ y)


def group_interval(delta, groups, seed=92621, draws=4000):
    delta, groups = np.asarray(delta), np.asarray(groups)
    unique = np.unique(groups)
    totals = np.array([delta[groups==g].sum() for g in unique])
    counts = np.array([(groups==g).sum() for g in unique])
    index = np.random.default_rng(seed).integers(len(unique), size=(draws, len(unique)))
    boot = totals[index].sum(1) / counts[index].sum(1)
    return {'mean': float(delta.mean()), 'low': float(np.quantile(boot, .025)),
            'high': float(np.quantile(boot, .975)), 'groups':len(unique)}


def fit_anchor(train, validation_select, meta):
    """Group-crossfit ridge, then test fixed OOF shrink gains on validation.

Hyperparameters share training data across folds: not nested CV. Entire documents
are withheld for the fold-specific linear fits; no test target is inspected.
"""
    x, y = features(train['x']), train['y'].flatten(1).double()
    h, v, n = len(meta['horizons']), meta['variables'], len(x)
    group = train['group']; unique = group.unique()
    if len(unique) < 5: raise ValueError('At least five training groups required')
    permutation = unique[torch.randperm(len(unique), generator=torch.Generator().manual_seed(92621))]
    fold_map = {int(g):i%5 for i,g in enumerate(permutation)}
    fold = torch.tensor([fold_map[int(g)] for g in group])
    oof = torch.zeros(len(ALPHAS), n, h*v, dtype=torch.float64)
    constants = torch.zeros(n, h*v, dtype=torch.float64)
    for f in range(5):
        held = fold==f; keep = ~held
        constants[held] = y[keep].mean(0)
        for j,a in enumerate(ALPHAS):
            oof[j,held] = x[held] @ ridge_weights(x[keep], y[keep], a)
    cv = (oof-y).square().reshape(len(ALPHAS),n,h,v).mean((1,3))
    best = cv.argmin(0)
    weights = torch.zeros(x.shape[-1], h*v, dtype=torch.float64)
    raw_oof = torch.zeros_like(y)
    for i in range(h):
        j=int(best[i]); sl=slice(i*v,(i+1)*v)
        weights[:,sl]=ridge_weights(x,y,ALPHAS[j])[:,sl]
        raw_oof[:,sl]=oof[j,:,sl]
    diff=(raw_oof-constants).reshape(n,h,v); target=(y-constants).reshape(n,h,v)
    gain=((diff*target).sum((0,2))/diff.square().sum((0,2)).clamp_min(1e-12)).clamp(0,1)
    constant=y.mean(0)
    vx=features(validation_select['x']); vy=validation_select['y'].double()
    raw=(vx@weights).reshape(-1,h,v)
    shrunk=constant.reshape(h,v)+gain[None,:,None]*(raw-constant.reshape(h,v))
    advantage=(vy-constant.reshape(h,v)).square().mean(-1)-(vy-shrunk).square().mean(-1)
    gates=[];details=[]
    for i, horizon in enumerate(meta['horizons']):
        interval=group_interval(advantage[:,i].numpy(), validation_select['group'].numpy())
        accepted=interval['low']>0
        gates.append(accepted)
        details.append({'horizon':horizon,'alpha':ALPHAS[int(best[i])], 'crossfit_gain':float(gain[i]),
                        'use_context':bool(accepted), 'validation_advantage':interval})
    model=HorizonAnchor(meta['history'],v,tuple(meta['horizons']))
    model.weight.copy_(weights.float());model.constant.copy_(constant.float())
    model.gain.copy_((gain*torch.tensor(gates)).float())
    final_oof=constants+model.gain.double().repeat_interleave(v)*(raw_oof-constants)
    return model, final_oof.float(), {'horizons':details,'oof_mse_by_alpha_horizon':cv.tolist(),
                                    'alphas':ALPHAS,'fold_by_group':fold_map}, raw_oof.float()


@torch.no_grad()
def measure(model, part):
    nll, mse = [], []
    for s in range(0,len(part['x']),128):
        f=model(part['x'][s:s+128]); y=part['y'][s:s+128].flatten(1)
        nll.append(f.energy(y)/y.shape[-1]); mse.append((f.mean-y).square().mean(-1))
    return {'nll':float(torch.cat(nll).mean()),'mse':float(torch.cat(mse).mean())}


def calibrate(model, part, steps=150):
    """Fit four positive affine scale factors only on calibration documents."""
    fixed=[]
    with torch.no_grad():
        for start in range(0,len(part['x']),128):
            fixed.append(model(part['x'][start:start+128]).detached())
        fun=EnergyLandscape(torch.cat([f.logits for f in fixed]),torch.cat([f.means for f in fixed]),
                            torch.cat([f.scales for f in fixed]),fixed[0].correlation_cholesky)
        center=fun.mean.detach(); before=float(fun.energy(part['y'].flatten(1)).mean()/model.d)
    logt=torch.zeros(len(model.config.horizons),requires_grad=True)
    optimizer=torch.optim.Adam([logt],lr=.02)
    from .calibrated import affine_landscape
    history=[]
    for step in range(steps):
        temperature=logt.exp().repeat_interleave(model.config.variables)
        loss=affine_landscape(fun,center,temperature).energy(part['y'].flatten(1)).mean()/model.d
        optimizer.zero_grad();loss.backward();optimizer.step()
        with torch.no_grad():logt.clamp_(math.log(.5),math.log(2.))
        history.append(float(loss.detach()))
    model.temperature.copy_(logt.detach().exp())
    after=measure(model,part)
    return {'before_nll':before,'after':after,'temperature':model.temperature.tolist(),'losses':history}


def save_model(model, path, variant, meta, identity, selection):
    atomic_save({'format':'calibrated_trajectory_v2','variant':variant,'config':asdict(model.config),
        'weights':model.state_dict(),'normalization':{k:meta[k] for k in ('center','scale')},
        'teacher_sha256':meta.get('teacher_sha256'),'identity':identity,'selection':selection,
        'summary_only':True},path)


def fit(data_path, runs_path, out, steps=600):
    torch.set_num_threads(1)
    if type(steps) is not int or steps<1 or steps>5000: raise ValueError('Invalid bounded fit budget')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    data=torch.load(data_path,weights_only=True,map_location='cpu')
    # The serialized container includes test data for legacy compatibility. Delete
    # that view immediately: fitting has no API access to its values or groups.
    del data['test']
    meta=data['metadata'];train=data['train'];valid=data['validation']
    selection_mask,cal_mask=split_validation(valid['group'])
    select,cal=subset(valid,selection_mask),subset(valid,cal_mask)
    anchor, oof, info, raw_oof=fit_anchor(train,select,meta)
    atomic_save({'weights':anchor.state_dict(),'oof_mean':oof,'oof_raw':raw_oof,'details':info},out/'anchor.pt')
    selection_info={'selection_groups':select['group'].unique().tolist(),'calibration_groups':cal['group'].unique().tolist(),**info}
    (out/'mean_selection.json').write_text(json.dumps(selection_info,indent=2))
    print('MEAN_SELECTION',json.dumps(info['horizons']),flush=True)
    # A full Gaussian regression baseline: OOF residual covariance, not optimistic
    # training residuals. Shrinkage is selected using selection documents only.
    residual=train['y'].flatten(1)-oof
    covariance=residual.T@residual/len(residual)
    candidates=[]
    for shrink in (0.,.1,.5,1.):
        cov=(1-shrink)*covariance+shrink*torch.diag(covariance.diag())+1e-4*torch.eye(covariance.shape[0])
        chol=torch.linalg.cholesky(cov)
        dist=torch.distributions.MultivariateNormal(anchor(select['x']),scale_tril=chol)
        loss=float(-dist.log_prob(select['y'].flatten(1)).mean()/cov.shape[0])
        candidates.append((loss,shrink,chol))
    score,shrink,chol=min(candidates,key=lambda v:v[0])
    atomic_save({'anchor':anchor.state_dict(),'chol':chol,'shrinkage':shrink,'validation_nll':score,
                 'candidates':[(v,s) for v,s,_ in candidates],'metadata':meta},out/'ridge_gaussian.pt')
    records=[]
    for seed in (11,22,33):
        original,old=load_head(Path(runs_path)/f'spectral_seed{seed}'/'best.pt')
        if old['identity']['data_sha256']!=digest(data_path): raise ValueError('Wrong parent head/data pair')
        identity={'data_sha256':digest(data_path),'teacher_sha256':meta.get('teacher_sha256'),
                  'seed':seed,'steps':steps,'parent_head_sha256':digest(Path(runs_path)/f'spectral_seed{seed}'/'best.pt')}
        recentered=RecenteredForecast(original.config)
        recentered.base.load_state_dict(original.state_dict());recentered.anchor.load_state_dict(anchor.state_dict())
        save_model(recentered,out/f'recentered_uncal_seed{seed}.pt','recentered',meta,identity,selection_info)
        calibration=calibrate(recentered,cal)
        save_model(recentered,out/f'recentered_seed{seed}.pt','recentered',meta,identity,selection_info)
        records.append({'variant':'recentered','seed':seed,'new_neural_steps':0,'calibration':calibration})
        torch.manual_seed(seed);model=AnchoredForecast(original.config);model.anchor.load_state_dict(anchor.state_dict())
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
        gen=torch.Generator().manual_seed(seed+20260926)
        best=float('inf');log=[];batch_hash=bytes(32);t0=time.perf_counter()
        for step in range(steps+1):
            if step>0:
                model.train();idx=torch.randint(len(train['x']),(64,),generator=gen)
                batch_hash=hashlib.sha256(batch_hash+idx.numpy().tobytes()).digest()
                fun=model.landscape(train['x'][idx],oof[idx]);y=train['y'][idx].flatten(1)
                loss=fun.energy(y).mean()/model.d+1e-4*torch.tril(model.raw_correlation,-1).square().mean()
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                for g in optimizer.param_groups:g['lr']=.001*(.2+.8*.5*(1+math.cos(math.pi*step/steps)))
                optimizer.step()
            if step%100==0 or step==steps:
                model.eval();score=measure(model,select);log.append({'step':step,**score})
                if score['nll']<best:
                    best=score['nll'];chosen=step
                    save_model(model,out/f'anchored_uncal_seed{seed}.pt','anchored',meta,identity,{**selection_info,'selected_step':step})
                print('ANCHOR_FIT',seed,step,score,flush=True)
        selected=torch.load(out/f'anchored_uncal_seed{seed}.pt',weights_only=True)
        atomic_save({'weights':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':torch.get_rng_state(),
                     'generator':gen.get_state(),'step':steps,'identity':identity,'batch_sha256':batch_hash.hex()},out/f'training_final_seed{seed}.pt')
        model.load_state_dict(selected['weights']);calibration=calibrate(model,cal)
        save_model(model,out/f'anchored_seed{seed}.pt','anchored',meta,identity,{**selection_info,'selected_step':chosen})
        records.append({'variant':'anchored','seed':seed,'new_neural_steps':steps,'presented_windows':steps*64,
                        'selected_step':chosen,'fit_seconds':time.perf_counter()-t0,'batch_sha256':batch_hash.hex(),
                        'validation_history':log,'calibration':calibration})
    (out/'training.json').write_text(json.dumps(records,indent=2))
    frozen={'data_sha256':digest(data_path),'selection_only':True,
            'files':{p.name:digest(p) for p in sorted(out.glob('*.pt'))},
            'teacher_unchanged':True,'note':'Frozen before new final-test evaluation; final dataset was already observed in parent study.'}
    (out/'FROZEN.json').write_text(json.dumps(frozen,indent=2))
    print('FROZEN_COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--parent-runs',required=True);p.add_argument('--out',required=True)
    p.add_argument('--steps',type=int,default=600);args=p.parse_args()
    fit(args.data,args.parent_runs,args.out,args.steps)

if __name__=='__main__':main()
