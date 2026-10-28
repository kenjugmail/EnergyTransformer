"""Document-bounded batches from the pinned, deduplicated story subset."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .data import Batch, file_sha256


class StoryCorpus:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        manifest_path = self.directory / 'manifest.json'
        self.manifest = json.loads(manifest_path.read_text())
        dtype = self.manifest.get('dtype', 'uint8')
        if dtype not in ('uint8', '<u4'):
            raise ValueError('Unsupported story token dtype')
        self.dtype = np.dtype(dtype)
        self.manifest_sha256 = file_sha256(manifest_path)
        self.data, self.documents = {}, {}
        identities = {}
        for split in ('train', 'valid', 'test'):
            info = self.manifest['splits'][split]
            path, index = Path(info['path']), Path(info['index_path'])
            if file_sha256(path) != info['sha256'] or file_sha256(index) != info['index_sha256']:
                raise ValueError('Story data or document index changed')
            records = json.loads(index.read_text())
            if len(records) != info['documents'] or not records:
                raise ValueError('Invalid story document count')
            spans = np.array([[r['start'], r['end']] for r in records], dtype=np.int64)
            if path.stat().st_size % self.dtype.itemsize:
                raise ValueError('Incomplete story token storage')
            if ((spans[:, 0] < 0).any() or (spans[:, 1] <= spans[:, 0]).any()
                    or spans[-1, 1] > path.stat().st_size // self.dtype.itemsize
                    or (spans[1:, 0] < spans[:-1, 1]).any()):
                raise ValueError('Invalid story boundaries')
            identities[split] = {r['id'] for r in records}
            if len(identities[split]) != len(records):
                raise ValueError('Duplicate story IDs')
            self.documents[split] = spans
            self.data[split] = np.memmap(path, dtype=self.dtype, mode='r')
        if any(identities[a] & identities[b] for a, b in
               [('train', 'valid'), ('train', 'test'), ('valid', 'test')]):
            raise ValueError('Story splits overlap')

    def batch(self, split, rng, batch, context, *, starts=None):
        if split not in self.data or type(batch) is not int or batch < 1 or type(context) is not int or context < 1:
            raise ValueError('Known split and positive batch/context required')
        spans = self.documents[split]
        if starts is None:
            eligible = np.flatnonzero(spans[:, 1] - spans[:, 0] > context)
            if not len(eligible):
                raise ValueError('No story is long enough for this context')
            selected = rng.choice(eligible, batch)
            starts = [int(rng.integers(spans[i, 0], spans[i, 1] - context)) for i in selected]
        if len(starts) != batch:
            raise ValueError('Window count differs from batch')
        for start in starts:
            index = int(np.searchsorted(spans[:, 0], start, side='right')) - 1
            if index < 0 or start + context >= spans[index, 1]:
                raise ValueError('Window crosses a story boundary')
        array = np.stack([np.array(self.data[split][start:start + context + 1], dtype=np.int64)
                          for start in starts])
        tensor = torch.from_numpy(array)
        return Batch(tensor[:, :-1], tensor[:, 1:])

    def offsets(self, split, count, context, seed):
        rng = np.random.default_rng(seed)
        spans = self.documents[split]
        eligible = np.flatnonzero(spans[:, 1] - spans[:, 0] > context)
        if count > len(eligible):
            raise ValueError('Distinct evaluation documents required')
        selected = rng.choice(eligible, count, replace=False)
        return [int(rng.integers(spans[i, 0], spans[i, 1] - context)) for i in selected]

    def text_prompts(self, count=30, seed=985, prefix=32, continuation=96):
        if self.dtype != np.dtype('uint8'):
            raise ValueError('Use the word tokenizer to decode word-token prompts')
        rng = np.random.default_rng(seed)
        spans = self.documents['test']
        eligible = np.flatnonzero(spans[:, 1] - spans[:, 0] >= prefix + continuation)
        if count > len(eligible):
            raise ValueError('Not enough distinct held-out stories')
        selected = rng.choice(eligible, count, replace=False)
        records = []
        for index in selected:
            start = int(spans[index, 0])
            raw = bytes(self.data['test'][start:start + prefix])
            records.append(dict(id=f'story_{len(records):03d}', document_index=int(index),
                offset=start, prompt=raw.decode('ascii'), prompt_hex=raw.hex(),
                reference_hex=bytes(self.data['test'][start + prefix:start + prefix + continuation]).hex()))
        return records

    def identity(self):
        return dict(dataset=self.manifest['dataset'], revision=self.manifest['revision'],
            directory=str(self.directory), manifest_sha256=self.manifest_sha256,
            splits={split:{k:v for k,v in info.items() if k!='document_ids'}
                    for split,info in self.manifest['splits'].items()})
