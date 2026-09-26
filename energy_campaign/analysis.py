"""Scaling fits require independent crossed runs, not correlated checkpoints."""
import numpy as np
from scipy.optimize import least_squares


def fit_scaling(rows):
    if any(r.get('metric')!='nll' or not r.get('independent_run') for r in rows):
        raise ValueError('Only independently trained next-token NLL endpoints supported')
    if rows:
        families={r.get('architecture',r.get('family')) for r in rows}
        if None in families or len(families)!=1:raise ValueError('Fit exactly one explicitly identified architecture at a time')
        cells=[(r['parameters'],r['targets'],r['seed']) for r in rows]
        if len(set(cells))!=len(cells):raise ValueError('Duplicate model/data/seed endpoint; checkpoints are not independent runs')
        trajectories=[r.get('trajectory_id',r.get('run_id')) for r in rows]
        known=[v for v in trajectories if v is not None]
        if len(set(known))!=len(known):raise ValueError('Repeated training trajectory is not independent evidence')
        if any(not np.isfinite(r[k]) or r[k]<=0 for r in rows for k in ('parameters','targets','loss')):
            raise ValueError('Positive finite model sizes, target budgets and losses required')
    sizes=sorted({r['parameters'] for r in rows});budgets=sorted({r['targets'] for r in rows})
    if len(sizes)<4 or len(budgets)<4:return {'status':'unresolved','reason':'Need at least four independent sizes and data budgets'}
    for n in sizes:
        for d in budgets:
            if len({r['seed'] for r in rows if r['parameters']==n and r['targets']==d})<3:
                return {'status':'unresolved','reason':'Crossed grid needs three seeds in every cell'}
    x=np.array([[r['parameters'],r['targets']] for r in rows],dtype=float);y=np.array([r['loss'] for r in rows])
    xn=x/x.min(0)
    def predict(p,a):return p[0]+np.exp(p[1])*a[:,0]**(-p[3])+np.exp(p[2])*a[:,1]**(-p[4])
    def fit(a,b):return least_squares(lambda p:predict(p,a)-b,[y.min()*.8,-1,-1,.2,.2],bounds=([0,-20,-20,.001,.001],[y.min(),20,20,3,3]),max_nfev=5000)
    result=fit(xn,y);condition=float(np.linalg.cond(result.jac));held=[]
    for axis in (0,1):
        for value in np.unique(xn[:,axis]):
            mask=xn[:,axis]==value;other=fit(xn[~mask],y[~mask]);held.append(float(np.sqrt(np.mean((predict(other.x,xn[mask])-y[mask])**2))))
    if not result.success or not np.isfinite(condition) or condition>1e8:
        return {'status':'unresolved','reason':'Fit not identifiable','jacobian_condition':condition if np.isfinite(condition) else None}
    # Cluster bootstrap by seed: a repeated seed contributes its complete grid.
    seeds=sorted({r['seed'] for r in rows});rng=np.random.default_rng(773);samples=[]
    for _ in range(200):
        chosen=rng.choice(seeds,len(seeds),replace=True);idx=[i for s in chosen for i,r in enumerate(rows) if r['seed']==s]
        estimate=fit(xn[idx],y[idx])
        if estimate.success:samples.append(estimate.x)
    if len(samples)<180:return {'status':'unresolved','reason':'Bootstrap fit instability'}
    ci=np.quantile(samples,[.025,.975],axis=0)
    return {'status':'provisional','parameters':result.x.tolist(),'normalization':x.min(0).tolist(),'parameter_95pct':ci.tolist(),
            'jacobian_condition':condition,'leave_size_or_budget_out_rmse':held,
            'limits':'Seed-cluster intervals only; document-resampled loss records needed for document uncertainty. Next-size validation required before extrapolation.'}


def paired_interval(rows,draws=2000):
    """Hierarchical seed and document bootstrap of paired per-group loss means."""
    seeds=sorted({r['seed'] for r in rows});groups=sorted({r['group'] for r in rows});matrix=np.empty((len(seeds),len(groups)))
    for i,s in enumerate(seeds):
        for j,g in enumerate(groups):
            values=[r['difference'] for r in rows if r['seed']==s and r['group']==g]
            if len(values)!=1:raise ValueError('Complete paired seed/document matrix required')
            matrix[i,j]=values[0]
    rng=np.random.default_rng(994);boot=[]
    for _ in range(draws):boot.append(matrix[np.ix_(rng.integers(len(seeds),size=len(seeds)),rng.integers(len(groups),size=len(groups)))].mean())
    return {'mean':float(matrix.mean()),'low':float(np.quantile(boot,.025)),'high':float(np.quantile(boot,.975)),
            'seeds':len(seeds),'groups':len(groups),'weighting':'equal document groups'}
