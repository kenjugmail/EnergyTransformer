"""Fresh whole-word training with train-only vocabulary and document isolation.

Existing byte/subword checkpoints and historical comparisons are not relabeled.
The default is the current streaming energy model. --architecture joint trains
the original joint-energy model; transformer controls use the same tokenizer.
"""
import argparse
from dataclasses import asdict
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
from rewt.model import EnergyConfig, JointEnergyLM
from rewt.streaming import StreamingConfig, StreamingLM
from rewt.word_data import WordCorpus, prepare_word_corpus
from rewt.word_lm import forward, evaluate_words, predict_words, load_word_model

PROMPTS = ['Once upon a', 'The little girl went to the', 'The dog was very',
           'She opened the door and', 'They lived happily ever']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT / 'data/tinystories_rewt_20261010')
    p.add_argument('--data', type=Path, default=ROOT / 'data/rewt_words_20261028')
    p.add_argument('--output', type=Path, default=ROOT / 'runs/rewt_words_20261028')
    p.add_argument('--report', type=Path, default=ROOT / 'reports/rewt_words_20261028')
    p.add_argument('--architecture', choices=['streaming', 'joint', 'transformer', 'window-transformer'], default='streaming')
    p.add_argument('--vocab-size', type=int, default=8192)
    p.add_argument('--min-count', type=int, default=2)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--context', type=int, default=64)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--width', type=int, default=128)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--rounds', type=int, default=2)
    p.add_argument('--lr', type=float, default=.003)
    p.add_argument('--seed', type=int, default=281801)
    p.add_argument('--eval-documents', type=int, default=96)
    p.add_argument('--eval-every', type=int, default=200)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--threads', type=int, default=1)
    a = p.parse_args()
    if (min(a.steps, a.context, a.batch, a.eval_documents, a.eval_every, a.threads) < 1
            or not math.isfinite(a.lr) or a.lr <= 0):
        p.error('Positive training and evaluation settings required')
    if a.output.exists() or a.report.exists():
        p.error('Use new output and report directories')
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    corpus = WordCorpus(a.data) if a.data.exists() else prepare_word_corpus(a.source, a.data, vocab_size=a.vocab_size, min_count=a.min_count)
    if corpus.manifest['requested_vocab_size'] != a.vocab_size or corpus.manifest['min_count'] != a.min_count:
        p.error('Existing corpus tokenizer settings differ; use a new --data directory')
    tokenizer = corpus.tokenizer
    vocabulary = len(tokenizer.vocabulary)
    if a.architecture == 'joint':
        config = EnergyConfig(width=a.width, hidden=a.hidden, history=32, rounds=a.rounds,
                              vocab_size=vocabulary, output_geometry='centered', cache_history=True)
        model, arm = JointEnergyLM(config), 'fourier_rewt'
    else:
        config = StreamingConfig(width=a.width, hidden=a.hidden, rounds=a.rounds,
                                 vocab_size=vocabulary, hybrid=a.architecture == 'streaming',
                                 window=0 if a.architecture == 'transformer' else 64)
        model, arm = StreamingLM(config), 'streaming_lm'
    model.to(a.device)
    valid = corpus.offsets('valid', a.eval_documents, a.context, a.seed + 1)
    test = corpus.offsets('test', a.eval_documents, a.context, a.seed + 2)
    a.output.mkdir(parents=True)
    a.report.mkdir(parents=True)
    source_files = ['rewt/words.py', 'rewt/word_data.py', 'rewt/word_lm.py', 'rewt/story_data.py',
                    'rewt/bpe.py', 'rewt/streaming.py', 'rewt/model.py',
                    'scripts/train_rewt_words.py', 'scripts/predict_rewt_words.py']
    snapshot = a.report / 'source'
    for name in source_files:
        dest = snapshot / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    plan = dict(architecture=a.architecture, arm=arm, config=asdict(config), seed=a.seed,
                steps=a.steps, context=a.context, batch=a.batch, learning_rate=a.lr, device=a.device,
                evaluation_every=a.eval_every, valid_offsets=valid, test_offsets=test, sample_prompts=PROMPTS,
                parameters=sum(x.numel() for x in model.parameters()),
                corpus_manifest_sha256=corpus.manifest_sha256, tokenizer_identity=tokenizer.identity(),
                source_sha256={s:file_sha256(ROOT / s) for s in source_files},
                selection='Minimum validation token NLL; evaluate test after training. One seed.',
                scope='New whole-word fit on an existing story subset. Existing holdouts were inspected in earlier byte/subword work. '
                      'Exploratory language experiment, not a fresh independent benchmark. '
                      'Punctuation is separate; unknown words never fall back to bytes.',
                torch_version=str(torch.__version__))
    save_json(a.report / 'plan.json', plan)
    shutil.copy2(a.data / 'manifest.json', a.report / 'corpus_manifest.json')
    shutil.copy2(a.data / 'vocabulary.json', a.report / 'vocabulary.json')
    unigram = torch.tensor(np.bincount(corpus.data['train'], minlength=vocabulary), dtype=torch.float32) + 1
    unigram /= unigram.sum()
    baseline = evaluate_words(model, corpus, 'valid', valid, a.context, unigram=unigram)
    initial = evaluate_words(model, corpus, 'valid', valid, a.context)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=.01)
    best = float('inf')
    training_seconds = 0.
    with (a.report / 'metrics.jsonl').open('w') as log:
        for step in range(1, a.steps + 1):
            model.train()
            if a.device == 'cuda': torch.cuda.synchronize()
            start = time.perf_counter()
            batch = corpus.batch('train', rng, a.batch, a.context).to(a.device)
            lr = a.lr * min(1., step / min(100, a.steps)) * (.15 + .85 * .5 * (1 + math.cos(math.pi * step / a.steps)))
            for group in optimizer.param_groups: group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            logits, _ = forward(model, batch.tokens)
            loss = F.cross_entropy(logits.flatten(0, 1), batch.targets.flatten())
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if a.device == 'cuda': torch.cuda.synchronize()
            training_seconds += time.perf_counter() - start
            row = dict(step=step, loss=float(loss.detach()), gradient_norm=float(norm), learning_rate=lr)
            if step % a.eval_every == 0 or step == a.steps:
                score = evaluate_words(model, corpus, 'valid', valid, a.context)
                row['validation'] = score
                if score['token_nll'] < best:
                    best = score['token_nll']
                    save_checkpoint(a.output / 'best.pt', dict(arm=arm, config=asdict(config), seed=a.seed,
                        weights={k:v.detach().cpu() for k,v in model.state_dict().items()},
                        tokenizer_spec=tokenizer.to_dict(), tokenizer_identity=tokenizer.identity(),
                        step=step, training_token_exposures=step * a.batch * a.context,
                        selection=score, plan=plan))
                print(json.dumps(dict(step=step, train_loss=row['loss'], validation_nll=score['token_nll'],
                                      word_accuracy=score['next_word_accuracy'], training_seconds=training_seconds)), flush=True)
            log.write(json.dumps(row) + '\n'); log.flush()
    selected, _, saved = load_word_model(a.output / 'best.pt')
    selected.to(a.device).eval()
    result = dict(selected_step=saved['step'], parameters=plan['parameters'], vocabulary_size=vocabulary,
                  training_token_exposures=a.steps * a.batch * a.context, training_seconds=training_seconds,
                  initial_validation=initial, unigram_validation=baseline, validation=saved['selection'],
                  test=evaluate_words(selected, corpus, 'test', test, a.context),
                  unigram_test=evaluate_words(selected, corpus, 'test', test, a.context, unigram=unigram),
                  samples=[predict_words(selected, tokenizer, prompt, top=5, generate=24) for prompt in PROMPTS],
                  checkpoint=str((a.output / 'best.pt').resolve()), checkpoint_sha256=file_sha256(a.output / 'best.pt'))
    if a.architecture in ('streaming', 'joint'):
        result['zero_rounds_test'] = evaluate_words(selected, corpus, 'test', test, a.context, rounds=0)
    save_json(a.report / 'results.json', result)
    print(json.dumps({k:result[k] for k in ['selected_step', 'parameters', 'vocabulary_size', 'test', 'checkpoint']}, indent=2))


if __name__ == '__main__':
    main()
