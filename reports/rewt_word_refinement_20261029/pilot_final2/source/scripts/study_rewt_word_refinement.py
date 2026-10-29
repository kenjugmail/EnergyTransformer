"""Matched whole-word experiments; development runs never evaluate test labels.

Train selects checkpoint and inference depth on a fixed validation panel. Run
the separate evaluate command only after freezing the comparison protocol.
"""
import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F

from rewt.data import file_sha256
from rewt.experiment import save_checkpoint, save_json
from rewt.refinement_training import refinement_loss
from rewt.streaming import StreamingConfig, StreamingLM
from rewt.word_data import WordCorpus
from rewt.word_lm import load_word_model, predict_words

DEPTHS = (0, 1, 2, 4, 8, 16, 32)
SOURCE = ['rewt/streaming.py', 'rewt/refinement_training.py', 'rewt/word_lm.py',
          'rewt/word_data.py', 'rewt/story_data.py', 'rewt/words.py',
          'scripts/study_rewt_word_refinement.py']


@torch.inference_mode()
def evaluate_depths(model, corpus, split, offsets, context, depths, batch_size=16):
    model.eval(); device = model.embedding.weight.device
    rows = {r: [] for r in depths}
    for start in range(0, len(offsets), batch_size):
        selected = offsets[start:start + batch_size]
        batch = corpus.batch(split, None, len(selected), context, starts=selected).to(device)
        states = (model.refinement_states(batch.tokens, max(depths)) if model.energy_head is not None
                  else [model.encode(batch.tokens)[0]])
        words = torch.tensor(np.stack([corpus.word_masks[split][s+1:s+context+1] for s in selected]),
                             dtype=torch.bool, device=device)
        known = batch.targets != corpus.tokenizer.unk_id
        for r in depths:
            logits = F.linear(states[r], model.embedding.weight)
            nll = F.cross_entropy(logits.flatten(0, 1), batch.targets.flatten(), reduction='none').view_as(batch.targets)
            prediction = logits.argmax(-1)
            correct = (prediction == batch.targets) & known
            top5 = (logits.topk(5, -1).indices == batch.targets[..., None]).any(-1) & known
            values = torch.stack((nll.mean(-1), words.sum(-1), (words & known).sum(-1),
                                  (correct & words).sum(-1), (top5 & words).sum(-1),
                                  (prediction == batch.targets).sum(-1), (~known).sum(-1)), -1).cpu().tolist()
            rows[r].extend(values)
    result = {}
    for r, values in rows.items():
        array = np.asarray(values); totals = array.sum(0); nll = float(array[:, 0].mean())
        result[str(r)] = dict(token_nll=nll, token_perplexity=math.exp(nll),
            next_word_accuracy=float(totals[3]/totals[1]), next_word_top5_accuracy=float(totals[4]/totals[1]),
            word_vocabulary_coverage=float(totals[2]/totals[1]),
            token_accuracy_with_unk=float(totals[5]/(len(offsets)*context)),
            evaluated_tokens=len(offsets)*context, evaluated_word_positions=int(totals[1]),
            unknown_targets=int(totals[6]), per_document=values)
    return result


def compact(scores):
    return {k: {name: value for name, value in score.items() if name != 'per_document'}
            for k, score in scores.items()}


def train(a):
    if a.output.exists() or a.report.exists():
        raise ValueError('Use new output/report directories')
    if min(a.steps, a.batch, a.context, a.eval_every, a.eval_documents) < 1 or not math.isfinite(a.lr) or a.lr <= 0:
        raise ValueError('Positive finite training settings required')
    corpus = WordCorpus(a.data)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    hybrid = a.profile != 'transformer'
    cfg = StreamingConfig(width=a.width, hidden=a.hidden, layers=a.layers, heads=4,
                          hybrid=hybrid, window=64 if hybrid else 0,
                          rounds=a.rounds, modes=a.modes, vocab_size=len(corpus.tokenizer.vocabulary))
    model = StreamingLM(cfg).to(a.device)
    depths = DEPTHS if hybrid else (0,)
    valid = corpus.offsets('valid', a.eval_documents, a.context, 291901)
    test = corpus.offsets('test', 1024, a.context, 291902)
    a.output.mkdir(parents=True); a.report.mkdir(parents=True)
    for name in SOURCE:
        dest = a.report / 'source' / name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, dest)
    plan = dict(profile=a.profile, config=asdict(cfg), seed=a.seed, steps=a.steps, batch=a.batch,
                context=a.context, learning_rate=a.lr, weight_decay=.01, gradient_clip=1.,
                supervise_steps=a.supervise_steps, monotonic_weight=a.monotonic_weight,
                device=a.device, eval_every=a.eval_every, depths=depths, valid_offsets=valid,
                test_offsets=test, parameters=sum(p.numel() for p in model.parameters()),
                corpus_manifest_sha256=corpus.manifest_sha256, tokenizer_identity=corpus.tokenizer.identity(),
                source_sha256={name: file_sha256(ROOT/name) for name in SOURCE},
                selection='Lowest validation token NLL across checkpoints and inference depths; no test evaluation during training.',
                batch_protocol='Same seed produces the same document-contained training crops across arms.',
                torch_version=str(torch.__version__))
    save_json(a.report/'plan.json', plan)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=.01)
    best = math.inf; train_seconds = 0.; wall_start = time.perf_counter()
    with (a.report/'metrics.jsonl').open('w') as log:
        for step in range(1, a.steps+1):
            model.train()
            if a.device == 'cuda': torch.cuda.synchronize()
            start = time.perf_counter()
            batch = corpus.batch('train', rng, a.batch, a.context).to(a.device)
            lr = a.lr * min(1., step/100) * (.15 + .85*.5*(1+math.cos(math.pi*step/a.steps)))
            for group in optimizer.param_groups: group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            if a.profile == 'trajectory':
                loss, detail = refinement_loss(model, batch.tokens, batch.targets,
                    steps=a.supervise_steps, monotonic_weight=a.monotonic_weight)
            else:
                logits = model(batch.tokens, rounds=a.rounds if hybrid else 0)[0]
                loss = F.cross_entropy(logits.flatten(0, 1), batch.targets.flatten()); detail = None
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if a.device == 'cuda': torch.cuda.synchronize()
            train_seconds += time.perf_counter()-start
            row = dict(step=step, loss=float(loss.detach()), gradient_norm=float(norm), learning_rate=lr,
                       training_seconds=train_seconds)
            if detail is not None:
                row['depth_nll'] = {k:float(v) for k,v in detail['depth_nll'].items()}
                row['monotonic_penalty'] = float(detail['monotonic_penalty'])
            if step % a.eval_every == 0 or step == a.steps:
                scores = evaluate_depths(model, corpus, 'valid', valid, a.context, depths)
                row['validation'] = compact(scores)
                selected_rounds = min(depths, key=lambda r: scores[str(r)]['token_nll'])
                score = scores[str(selected_rounds)]
                if score['token_nll'] < best:
                    best = score['token_nll']
                    save_checkpoint(a.output/'best.pt', dict(arm='streaming_lm',
                        config=asdict(replace(cfg, rounds=selected_rounds)),
                        weights={k:v.detach().cpu() for k,v in model.state_dict().items()},
                        tokenizer_spec=corpus.tokenizer.to_dict(), tokenizer_identity=corpus.tokenizer.identity(),
                        seed=a.seed, step=step, selected_rounds=selected_rounds, validation=compact(scores),
                        training_token_exposures=step*a.batch*a.context, training_seconds=train_seconds, plan=plan))
                print(json.dumps(dict(profile=a.profile, seed=a.seed, step=step, selected_rounds=selected_rounds,
                    validation_nll=score['token_nll'], word_accuracy=score['next_word_accuracy'],
                    training_seconds=round(train_seconds, 2))), flush=True)
            log.write(json.dumps(row)+'\n'); log.flush()
    model, tokenizer, saved = load_word_model(a.output/'best.pt'); model.to(a.device)
    summary = dict(selected_step=saved['step'], selected_rounds=saved['selected_rounds'],
        validation=saved['validation'], parameters=plan['parameters'], training_seconds=train_seconds,
        wall_seconds=time.perf_counter()-wall_start, training_token_exposures=a.steps*a.batch*a.context,
        checkpoint=str((a.output/'best.pt').resolve()), checkpoint_sha256=file_sha256(a.output/'best.pt'),
        test_evaluated=False, plan=plan)
    save_json(a.report/'summary.json', summary)


def evaluate(a):
    if a.output.exists(): raise ValueError('Use a new evaluation output file')
    model, tokenizer, saved = load_word_model(a.checkpoint); model.to(a.device).eval()
    corpus = WordCorpus(a.data); plan = saved['plan']
    if corpus.manifest_sha256 != plan['corpus_manifest_sha256']:
        raise ValueError('Corpus differs from the training plan')
    depths = DEPTHS if model.energy_head is not None else (0,)
    offsets = plan[f'{a.split}_offsets']
    scores = evaluate_depths(model, corpus, a.split, offsets, plan['context'], depths)
    result = dict(checkpoint=str(a.checkpoint.resolve()), checkpoint_sha256=file_sha256(a.checkpoint),
                  split=a.split, offsets=offsets, selected_rounds=saved['selected_rounds'],
                  selection='Use the depth selected on validation before inspecting test.', scores=scores)
    save_json(a.output, result)
    print(json.dumps(dict(selected_rounds=saved['selected_rounds'], scores=compact(scores)), indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--data', type=Path, default=ROOT/'data/rewt_words_20261028')
    sub = p.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--output', type=Path, required=True); t.add_argument('--report', type=Path, required=True)
    t.add_argument('--profile', choices=['transformer', 'final', 'trajectory'], required=True)
    for name, default in [('steps',2400), ('batch',16), ('context',64), ('width',128), ('hidden',256),
                          ('layers',2), ('modes',16), ('rounds',8), ('seed',291101),
                          ('eval-every',600), ('eval-documents',512)]:
        t.add_argument('--'+name, type=int, default=default)
    t.add_argument('--lr', type=float, default=.003)
    t.add_argument('--supervise-steps', type=int, nargs='+', default=[0,2,4,8])
    t.add_argument('--monotonic-weight', type=float, default=.1)
    e = sub.add_parser('evaluate'); e.add_argument('--checkpoint', type=Path, required=True)
    e.add_argument('--output', type=Path, required=True); e.add_argument('--split', choices=['valid','test'], default='test')
    a = p.parse_args(); torch.set_num_threads(1)
    train(a) if a.command == 'train' else evaluate(a)


if __name__ == '__main__':
    main()
