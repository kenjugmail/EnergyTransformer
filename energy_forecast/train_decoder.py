"""Matched trainable-budget decoder adapters on frozen causal teacher features.

CPU head-only study. Cached test/secondary labels are removed before fitting.
Step zero is eligible; all selected hashes are frozen before final evaluation.
"""
from __future__ import annotations
from pathlib import Path
import argparse, hashlib, json, math, time
from dataclasses import asdict
import torch
from torch.nn import functional as F
from .decoder_bridge import DecoderAdapter, DecoderConfig, KINDS
from .model import EnergyLandscape
from .experiment import load_head, atomic_save
from .probe import checkpoint_hash
from .decoder_integrity import (atomic_json, exclusive_directory, run_identity,
    same_json, validate_head_metadata, validate_schedule)

@torch.no_grad()
def forecast_cache(head, part, batch=512):
    rows={k:[] for k in ('mean','means','scales','logits')}; lower=None
    for start in range(0,len(part['h']),batch):
        x=part['x'][start:start+batch];fun=head(x)
        rows['mean'].append(head.anchor(x).detach())
        for k in ('means','scales','logits'):rows[k].append(getattr(fun,k).detach())
        lower=fun.correlation_cholesky.detach()
    return {**{k:torch.cat(v) for k,v in rows.items()},'lower':lower}

def select(cache, idx):
    if cache is None:return None,None
    return EnergyLandscape(cache['logits'][idx],cache['means'][idx],cache['scales'][idx],cache['lower']),cache['mean'][idx]

@torch.no_grad()
def feature_normalization(adapter, part, cache):
    if adapter.config.kind=='hidden':return
    total=[]
    for start in range(0,len(part['h']),512):
        sl=slice(start,start+512);fun,mean=select(cache,sl)
        total.append(adapter.raw_features(part['h'][sl],part['x'][sl],fun,mean))
    feature=torch.cat(total)
    adapter.feature_center.copy_(feature.mean(0))
    adapter.feature_scale.copy_(feature.std(0).clamp_min(.05))

@torch.no_grad()
def losses(adapter, part, weight, cache=None, *, batch=512, erase_side=False, permutation=None):
    rows=[]
    for start in range(0,len(part['h']),batch):
        idx=torch.arange(start,min(start+batch,len(part['h'])))
        side_idx=idx if permutation is None else permutation[idx]
        fun,mean=select(cache,side_idx)
        if adapter is None:logits=F.linear(part['h'][idx],weight)
        else:logits=adapter(part['h'][idx],part['x'][side_idx],weight,fun,mean,erase_side=erase_side)
        rows.append(F.cross_entropy(logits,part['y'][idx],reduction='none')/math.log(2))
    return torch.cat(rows)

def _resume_state(out, identity, finish):
    old = torch.load(out / 'last.pt', map_location='cpu', weights_only=True)
    if not same_json(old['identity'], identity):
        raise ValueError('Changed resume identity')
    step = old['step']
    if type(step) is not int or not 0 <= step <= identity['steps']:
        raise ValueError('Invalid saved training step')
    if finish < step:
        raise ValueError('stop_after precedes the saved step; refusing to rewind')
    if (not same_json(old['best_payload']['identity'], identity)
            or old['best_payload']['selected_step'] != old['best_step']
            or not 0 <= old['best_step'] <= step or not math.isfinite(old['best'])):
        raise ValueError('Inconsistent saved model selection')
    return old


def _completed_record(out, identity, old):
    """A completed run is reusable only with bound, unchanged checkpoint files."""
    path = out / 'training.json'
    if old['step'] != identity['steps'] or not path.exists():
        return None
    record = json.loads(path.read_text())
    if record.get('steps_completed') != old['step']:
        return None  # Last checkpoint committed after an older report: reconstruct it.
    expected = {'identity': identity, 'status': 'completed',
                'selected_step': old['best_step'], 'validation_bpb': old['best'],
                'sample_hash': old['sample_hash'], 'presented_targets': old['step'] * identity['batch']}
    if any(not same_json(record.get(k), v) for k, v in expected.items()):
        raise ValueError('Completed training record disagrees with checkpoint')
    for name in ('best', 'last'):
        if checkpoint_hash(out / (name + '.pt')) != record.get(name + '_sha256'):
            raise ValueError('Completed checkpoint changed')
    return record


def train_one(data, data_hash, cache, head_hash, out, kind, seed=11, steps=600, batch=128,
              peak_lr=.001, stop_after=None, resume=False):
    validate_schedule(kind, seed, steps, batch, peak_lr, stop_after)
    identity = run_identity(data, data_hash, head_hash, kind, seed, steps, batch, peak_lr)
    with exclusive_directory(out, resume=resume) as directory:
        return _train_locked(data, cache, head_hash, directory, identity, stop_after, resume)


def _train_locked(data, cache, head_hash, out, identity, stop_after, resume):
    kind, seed, steps, batch, peak_lr = (identity[k] for k in ('config', 'seed', 'steps', 'batch', 'peak_lr'))
    cfg = DecoderConfig(**kind)
    kind = cfg.kind
    finish = steps if stop_after is None else stop_after
    old = _resume_state(out, identity, finish) if resume else None
    if old is not None:
        completed = _completed_record(out, identity, old)
        if completed is not None:
            return completed
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    meta = data['metadata']
    model = DecoderAdapter(cfg)
    feature_normalization(model, data['train'], cache['train'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=.01, foreach=False)
    generator = torch.Generator().manual_seed(19000 + seed)
    step = 0
    history = []
    sample_hash = bytes(32)
    best_step = 0

    def export(selected_step):
        return {'format': 'decoder_bridge_v1', 'config': asdict(cfg), 'weights': model.state_dict(),
                'teacher_sha256': meta['teacher_sha256'],
                'head_sha256': head_hash if kind in ('mean', 'moments', 'energy_read') else None,
                'normalization': meta['normalization'], 'identity': identity, 'selected_step': selected_step}

    if old is not None:
        model.load_state_dict(old['weights'])
        optimizer.load_state_dict(old['optimizer'])
        generator.set_state(old['generator'])
        torch.set_rng_state(old['rng'])
        step, best, best_step = old['step'], old['best'], old['best_step']
        history, sample_hash = old['history'], bytes.fromhex(old['sample_hash'])
        best_payload = old['best_payload']
        atomic_save(best_payload, out / 'best.pt')
    else:
        best = float(losses(model, data['validation'], data['weight'], cache['validation']).mean())
        if not math.isfinite(best):
            raise FloatingPointError('Nonfinite initial validation loss')
        history = [{'step': 0, 'validation_bpb': best}]
        best_payload = export(0)
        best_payload['weights'] = {k: v.detach().clone() for k, v in best_payload['weights'].items()}
        atomic_save(best_payload, out / 'best.pt')
        atomic_save(best_payload, out / 'initial.pt')

    def save():
        atomic_save({'identity': identity, 'weights': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'generator': generator.get_state(), 'rng': torch.get_rng_state(), 'step': step,
                     'best': best, 'best_step': best_step, 'history': history,
                     'sample_hash': sample_hash.hex(), 'best_payload': best_payload}, out / 'last.pt')

    if old is None:
        save()  # A zero-step stop is now resumable, with the original validation opportunity.
    start = step
    t0 = time.perf_counter()
    for step in range(start + 1, finish + 1):
        model.train()
        idx = torch.randint(len(data['train']['h']), (batch,), generator=generator)
        sample_hash = hashlib.sha256(sample_hash + idx.numpy().tobytes()).digest()
        fun, mean = select(cache['train'], idx)
        optimizer.zero_grad(set_to_none=True)
        logits = model(data['train']['h'][idx], data['train']['x'][idx], data['weight'], fun, mean)
        loss = F.cross_entropy(logits, data['train']['y'][idx])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss')
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        for group in optimizer.param_groups:
            group['lr'] = peak_lr * (.2 + .8 * .5 * (1 + math.cos(math.pi * step / steps)))
        optimizer.step()
        if step % 100 == 0 or step == steps:
            model.eval()
            valid = float(losses(model, data['validation'], data['weight'], cache['validation']).mean())
            if not math.isfinite(valid):
                raise FloatingPointError('Nonfinite validation loss')
            history.append({'step': step, 'validation_bpb': valid,
                            'train_nats': float(loss.detach()), 'gradient_norm': float(grad)})
            if valid < best:
                best, best_step = valid, step
                best_payload = export(step)
                best_payload['weights'] = {k: v.detach().clone() for k, v in best_payload['weights'].items()}
                atomic_save(best_payload, out / 'best.pt')
            print(kind, seed, history[-1], flush=True)
        if step % 100 == 0 or step == finish:
            save()
    record = {'status': 'completed' if step == steps else 'partial', 'kind': kind, 'seed': seed,
              'steps_completed': step, 'selected_step': best_step, 'validation_bpb': best,
              'parameters': sum(p.numel() for p in model.parameters()), 'sample_hash': sample_hash.hex(),
              'presented_targets': step * batch, 'seconds_this_invocation': time.perf_counter() - t0,
              'history': history, 'identity': identity,
              'best_sha256': checkpoint_hash(out / 'best.pt'), 'last_sha256': checkpoint_hash(out / 'last.pt')}
    atomic_json(record, out / 'training.json')
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--heads')
    p.add_argument('--out', required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[11, 22, 33])
    p.add_argument('--kinds', nargs='+', choices=KINDS, default=list(KINDS))
    p.add_argument('--steps', type=int, default=600)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stop-after', type=int)
    a = p.parse_args(argv)
    if len(set(a.seeds)) != len(a.seeds) or len(set(a.kinds)) != len(a.kinds):
        raise ValueError('Repeated seeds or kinds are not independent runs')
    for kind in a.kinds:
        for seed in a.seeds:
            validate_schedule(kind, seed, a.steps, a.batch, .001, a.stop_after)
    if not a.heads and any(k not in ('hidden', 'history') for k in a.kinds):
        raise ValueError('Forecast arms require --heads; hidden/history do not')
    torch.set_num_threads(1)
    data = torch.load(a.data, weights_only=True, map_location='cpu')
    data.pop('test', None)
    data.pop('secondary', None)
    data_hash = checkpoint_hash(a.data)
    heads = {}
    head_hashes = {}
    for seed in a.seeds:
        if a.heads:
            hp = Path(a.heads) / f'anchored_seed{seed}.pt'
            head, hm = load_head(hp)
            validate_head_metadata(hm, data['metadata'])
            for par in head.parameters():
                par.requires_grad_(False)
            heads[seed], head_hashes[seed] = head, checkpoint_hash(hp)
        else:
            heads[seed], head_hashes[seed] = None, None
    identities = {f'{kind}_seed{seed}': run_identity(data, data_hash, head_hashes[seed],
                   kind, seed, a.steps, a.batch, .001) for seed in a.seeds for kind in a.kinds}
    study = {'format': 'decoder_study_v2', 'runs': identities}
    records = []
    with exclusive_directory(a.out, resume=a.resume) as out:
        manifest = out / 'STUDY.json'
        if a.resume:
            if not manifest.exists() or not same_json(json.loads(manifest.read_text()), study):
                raise ValueError('Study identity changed or missing; start a separately labelled run')
            # Validate ALL existing arms before allowing any arm to make new progress.
            for name, identity in identities.items():
                rd = out / name
                if rd.exists():
                    old = _resume_state(rd, identity, a.steps if a.stop_after is None else a.stop_after)
                    _completed_record(rd, identity, old)
        else:
            atomic_json(study, manifest)
        for seed in a.seeds:
            head = heads[seed]
            cache = {name: forecast_cache(head, data[name]) if head is not None else None
                     for name in ('train', 'validation')}
            for kind in a.kinds:
                rd = out / f'{kind}_seed{seed}'
                records.append(train_one(data, data_hash, cache, head_hashes[seed], rd, kind, seed,
                                         a.steps, a.batch, stop_after=a.stop_after,
                                         resume=a.resume and rd.exists()))
        complete = all(r['status'] == 'completed' for r in records)
        if complete:
            files = {f"{r['kind']}_seed{r['seed']}/best.pt": r['best_sha256'] for r in records}
            means = {kind: sum(r['validation_bpb'] for r in records if r['kind'] == kind) / len(a.seeds)
                     for kind in a.kinds}
            frozen = {'format': 'decoder_bridge_selection_v1', 'selection': 'validation before final test',
                      'data_sha256': data_hash, 'files': files, 'seeds': sorted(a.seeds),
                      'study_sha256': checkpoint_hash(manifest), 'validation_means': means,
                      'validation_winner': min(means, key=means.get)}
            destination = out / 'FROZEN.json'
            if destination.exists():
                if not same_json(json.loads(destination.read_text()), frozen):
                    raise ValueError('Frozen selection changed; refusing to replace it')
            else:
                atomic_json(frozen, destination)
        print(json.dumps({'status': 'completed' if complete else 'partial', 'runs': len(records)}), flush=True)
        return 0 if complete else 75


if __name__ == '__main__':
    raise SystemExit(main())
