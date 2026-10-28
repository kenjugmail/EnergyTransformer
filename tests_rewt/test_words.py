from dataclasses import asdict
import hashlib
import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from rewt.bpe import checkpoint_tokenizer, load_story_tokenizer
from rewt.fast_decode import FastDecodeSession
from rewt.model import EnergyConfig, JointEnergyLM, TransformerConfig, TransformerLM
from rewt.streaming import StreamingConfig, StreamingLM, load_streaming
from rewt.streaming_decode import StreamingSession
from rewt.word_data import WordCorpus, prepare_word_corpus
from rewt.word_lm import evaluate_words, load_word_model, predict_words
from rewt.words import WordTokenizer


def vocabulary():
    return WordTokenizer.learn(' '.join(f'word{i}' for i in range(400)) + " café can't well-known 123, hello!", vocab_size=512, min_count=1)


def test_whole_words_numbers_unicode_and_unknowns_never_split(tmp_path):
    tokenizer = vocabulary()
    text = "café can't well-known 123, neverseen!"
    tokens = tokenizer.encode(text)
    assert len(tokens) == 7
    assert tokens[-2] == tokenizer.unk_id
    assert tokenizer.decode(tokens).decode() == "café can't well-known 123, <unk>!"
    assert tokenizer.decode_continuation(tokenizer.encode('hello!'), 'Say') == b' hello!'
    assert max(tokenizer.encode('word399')) > 255
    before = tokenizer.identity()
    tokenizer.encode('heldout-only-word')
    assert tokenizer.identity() == before
    path = tmp_path / 'vocabulary.json'
    tokenizer.save(path, source={'split': 'train'})
    restored = load_story_tokenizer(path)
    assert restored.identity() == before
    assert restored.encode(text) == tokens
    with pytest.raises(ValueError):
        restored.decode([-1])


def raw_corpus(path):
    path.mkdir()
    manifest = dict(dataset='fixture', revision='v1', splits={})
    sha = lambda raw: hashlib.sha256(raw).hexdigest()
    for split in ('train', 'valid', 'test'):
        rows, raw = [], bytearray()
        for i in range(4):
            words = ' '.join(f'word{j}' for j in range(400))
            text = f'{split}{i} {words}. ' + ('heldout_only' if split != 'train' else 'hello world')
            encoded = text.encode()
            start = len(raw); raw.extend(encoded)
            rows.append(dict(id=sha(encoded), start=start, end=len(raw)))
        data, index = path / f'{split}.bin', path / f'{split}.json'
        data.write_bytes(raw); index.write_text(json.dumps(rows))
        manifest['splits'][split] = dict(path=str(data), index_path=str(index), documents=len(rows),
                                        sha256=sha(raw), index_sha256=sha(index.read_bytes()))
    (path / 'manifest.json').write_text(json.dumps(manifest))
    return path


def test_word_corpus_uses_wide_ids_document_boundaries_and_train_only_vocabulary(tmp_path):
    corpus = prepare_word_corpus(raw_corpus(tmp_path / 'raw'), tmp_path / 'words', vocab_size=512, min_count=1)
    assert len(corpus.tokenizer.vocabulary) > 256
    assert 'heldout' not in corpus.tokenizer.ids and 'valid0' not in corpus.tokenizer.ids
    starts = corpus.offsets('test', 4, 32, 181)
    batch = corpus.batch('test', None, 4, 32, starts=starts)
    assert int(batch.tokens.max()) > 255
    torch.testing.assert_close(batch.tokens[:, 1:], batch.targets[:, :-1])
    for split in corpus.data:
        for start, end in corpus.documents[split]:
            assert corpus.data[split][start] == corpus.tokenizer.bos_id
            assert corpus.data[split][end - 1] == corpus.tokenizer.eos_id
    with pytest.raises(ValueError, match='boundary'):
        corpus.batch('test', None, 1, 8, starts=[int(corpus.documents['test'][0, 1]) - 4])
    changed = tmp_path / 'words/test_words.bin'
    changed.write_bytes(changed.read_bytes() + b'\0')
    with pytest.raises(ValueError, match='mask changed'):
        WordCorpus(tmp_path / 'words')


@pytest.mark.parametrize('kind', ['streaming', 'joint', 'transformer'])
def test_large_vocabulary_forward_backward_and_incremental_causality(kind):
    torch.set_num_threads(1); torch.manual_seed(918)
    if kind == 'streaming':
        model = StreamingLM(StreamingConfig(width=16, heads=2, hidden=24, modes=4, vocab_size=513))
    elif kind == 'joint':
        model = JointEnergyLM(EnergyConfig(width=8, hidden=16, modes=4, vocab_size=513, rounds=1))
    else:
        model = TransformerLM(TransformerConfig(width=16, heads=2, feedforward=24, context=32, vocab_size=513))
    x = torch.tensor([[1, 300, 512, 278, 2]])
    result = model(x)
    logits = result[0] if kind == 'streaming' else result.logits
    assert logits.shape == (1, 5, 513)
    F.cross_entropy(logits[:, :-1].flatten(0, 1), x[:, 1:].flatten()).backward()
    assert torch.isfinite(model.embedding.weight.grad).all()
    model.eval()
    with torch.inference_mode():
        full = model(x)
        part = model(x[:, :3], return_state=True) if kind == 'streaming' else model(x[:, :3])
        state = part[1] if kind == 'streaming' else part.state
        tail = model(x[:, 3:], state)
        if kind == 'streaming':
            torch.testing.assert_close(tail[0], full[0][:, 3:], atol=3e-6, rtol=3e-5)
        else:
            torch.testing.assert_close(tail.logits, full.logits[:, 3:], atol=3e-6, rtol=3e-5)
    with pytest.raises(ValueError, match='vocabulary'):
        model(torch.tensor([[513]]))


def test_wide_word_ids_match_reference_and_optimized_streaming_decode():
    model = StreamingLM(StreamingConfig(width=16, heads=2, hidden=24, modes=4, window=8, vocab_size=513)).eval()
    x = torch.randint(260, 513, (1, 19))
    reference = StreamingSession(model, capacity=32)
    fast = FastDecodeSession(model, capacity=32, backend='eager')
    torch.testing.assert_close(reference.consume(x[:, :11]), fast.consume(x[:, :11]))
    fast.prepare()
    torch.testing.assert_close(reference.consume(x[:, 11:]), fast.consume(x[:, 11:]), atol=4e-6, rtol=4e-5)
    assert fast.generate(2, temperature=.7, top_k=500).shape == (1, 2)


def test_portable_word_checkpoint_and_old_byte_checkpoint(tmp_path):
    tokenizer = vocabulary()
    model = StreamingLM(StreamingConfig(width=16, heads=2, hidden=24, modes=4, vocab_size=len(tokenizer.pieces))).eval()
    path = tmp_path / 'words.pt'
    saved = dict(arm='streaming_lm', config=asdict(model.config), weights=model.state_dict(), tokenizer_spec=tokenizer.to_dict())
    torch.save(saved, path)
    restored, vocabulary2, _ = load_word_model(path)
    result = predict_words(restored, vocabulary2, 'hello unknown', top=5, generate=4)
    assert len(result['candidates']) == 5
    assert result['unknown_prompt_tokens'] == ['unknown']
    assert all(row['token_id'] in tokenizer.word_ids for row in result['candidates'])
    assert sum(row['probability'] for row in result['candidates']) <= 1
    byte_model = StreamingLM(StreamingConfig(width=16, heads=2, hidden=24, modes=4)).eval()
    config = asdict(byte_model.config); del config['vocab_size']
    path = tmp_path / 'legacy.pt'
    torch.save(dict(arm='streaming_lm', config=config, weights=byte_model.state_dict()), path)
    loaded, _ = load_streaming(path)
    x = torch.tensor([[0, 65, 255]])
    torch.testing.assert_close(loaded(x)[0], byte_model(x)[0])
    with pytest.raises(ValueError, match='include its tokenizer'):
        checkpoint_tokenizer({'config': {'vocab_size': 500}})
    with pytest.raises(ValueError, match='vocabulary has changed'):
        checkpoint_tokenizer({**saved, 'tokenizer_identity': 'wrong'})


def test_unknown_word_predictions_never_count_as_correct_words(tmp_path):
    corpus = prepare_word_corpus(raw_corpus(tmp_path / 'raw'), tmp_path / 'words', vocab_size=512, min_count=1)
    model = StreamingLM(StreamingConfig(width=16, heads=2, hidden=24, modes=4, vocab_size=len(corpus.tokenizer.pieces))).eval()
    # Force the unigram evaluator to predict <unk> at every position, including
    # the unknown first word in each test story. Those are still word errors.
    probability = torch.full((model.config.vocab_size,), 1e-8)
    probability[corpus.tokenizer.unk_id] = 1.
    probability /= probability.sum()
    result = evaluate_words(model, corpus, 'test', [0], 8, unigram=probability)
    assert result['unknown_target_tokens'] == 1
    assert result['next_word_accuracy'] == 0
    assert result['token_accuracy_with_unk'] == 1 / 8


@pytest.mark.parametrize('value', [0, 1, -1, True, 3.5])
@pytest.mark.parametrize('config', [EnergyConfig, TransformerConfig, StreamingConfig])
def test_invalid_vocabulary_sizes(value, config):
    with pytest.raises(ValueError):
        config(vocab_size=value).validate()


@pytest.mark.parametrize('size', [17, 513])
def test_joint_energy_large_output_ranking_and_cached_decode(size):
    from rewt.decode import EnergyDecodeSession
    cfg = EnergyConfig(width=8, hidden=16, modes=4, history=4, vocab_size=size,
                       rounds=1, cache_history=True, output_geometry='centered')
    model = JointEnergyLM(cfg)
    tokens = torch.randint(size, (2, 6)); targets = torch.randint(size, (2, 6))
    output = model(tokens, details=True)
    loss = model.ranking_loss(tokens, targets, output, torch.ones_like(tokens, dtype=torch.bool), hard_negatives=True)
    loss.backward()
    assert torch.isfinite(model.raw_frequency.grad).all()
    model.eval()
    session = EnergyDecodeSession(model)
    with torch.inference_mode():
        prefix = session.prefill(tokens[:, :4])
        for t in range(4, 6):
            reference = model(tokens[:, t:t+1], state=prefix.state)
            actual = session.step(tokens[:, t:t+1])
            torch.testing.assert_close(actual.logits, reference.logits)
            prefix = reference


def test_word_sized_architecture_structured_solver_matches_dense_and_ring():
    from dataclasses import replace
    from rewt.architecture import ArchitectureConfig, ArchitectureEnergyLM
    from rewt.ring_decode import RingDecodeSession
    cfg = ArchitectureConfig(width=8, hidden=19, modes=5, history=8, position_period=16,
        recurrent_type='gru', context_kernel=3, compatibility_geometry='anchor_relative',
        wave_geometry='anchor_relative', initial_carry_scale=0., output_geometry='centered',
        solver='preconditioned', rounds=2, vocab_size=333, output_rank=4)
    dense = ArchitectureEnergyLM(cfg).double().eval()
    structured = ArchitectureEnergyLM(replace(cfg, preconditioner_factorization='structured')).double().eval()
    structured.load_state_dict(dense.state_dict())
    tokens = torch.randint(260, 333, (1, 12))
    with torch.inference_mode():
        expected = dense(tokens)
        actual = structured(tokens)
        torch.testing.assert_close(actual.logits, expected.logits, atol=1e-11, rtol=1e-10)
        session = RingDecodeSession(structured)
        session.prefill(tokens[:, :10])
        for t in range(10, 12):
            torch.testing.assert_close(session.step(tokens[:, t:t+1]), expected.logits[:, t:t+1], atol=1e-11, rtol=1e-10)
