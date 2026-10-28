import hashlib
import json

import numpy as np
import pytest
import torch

from rewt.evaluate import language_metrics
from rewt.model import EnergyConfig, JointEnergyLM
from rewt.story_data import StoryCorpus


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def story_fixture(tmp_path):
    manifest = dict(dataset='test', revision='test', splits={})
    for split, letter in [('train', 'A'), ('valid', 'B'), ('test', 'C')]:
        stories = [(letter * 170 + str(i)).encode() for i in range(3)]
        raw = bytearray(); records = []
        for story in stories:
            start = len(raw); raw.extend(story); end = len(raw); raw.extend(b'\n\n')
            records.append(dict(id=digest(story), start=start, end=end))
        path = tmp_path / f'{split}.bin'; path.write_bytes(raw)
        index = tmp_path / f'{split}.json'; index.write_text(json.dumps(records))
        manifest['splits'][split] = dict(path=str(path), index_path=str(index), sha256=digest(raw),
            index_sha256=digest(index.read_bytes()), documents=len(records))
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    return StoryCorpus(tmp_path)


def test_batches_and_offsets_stay_inside_distinct_documents(tmp_path):
    corpus = story_fixture(tmp_path)
    starts = corpus.offsets('test', 3, 128, 983)
    indices = [int(np.searchsorted(corpus.documents['test'][:, 0], s, side='right')) - 1 for s in starts]
    assert len(set(indices)) == 3
    batch = corpus.batch('test', None, 3, 128, starts=starts)
    assert batch.tokens.shape == (3, 128)
    torch.testing.assert_close(batch.tokens[:, 1:], batch.targets[:, :-1], rtol=0, atol=0)
    with pytest.raises(ValueError):
        corpus.batch('test', None, 1, 8, starts=[170])
    prompts = corpus.text_prompts(count=3)
    assert len({p['document_index'] for p in prompts}) == 3
    assert all(len(bytes.fromhex(p['prompt_hex'])) == 32 for p in prompts)


def test_story_evaluation_uses_declared_split_even_with_small_file_offsets(tmp_path):
    torch.set_num_threads(1)
    corpus = story_fixture(tmp_path)
    visited = []
    original = corpus.batch
    def tracked(split, *args, **kwargs):
        visited.append(split)
        return original(split, *args, **kwargs)
    corpus.batch = tracked
    model = JointEnergyLM(EnergyConfig(width=4, hidden=8, modes=2, history=4, rounds=1))
    metrics = language_metrics(model, corpus, [0], context=16, split='test')
    assert visited == ['test'] and metrics['evaluated_bytes'] == 16


def test_modified_story_file_is_rejected(tmp_path):
    story_fixture(tmp_path)
    with (tmp_path / 'test.bin').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='changed'):
        StoryCorpus(tmp_path)
