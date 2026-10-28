"""Auditable inputs for the local-learning and explicit-energy experiments.

The graph task supplies structured events and queries. A causal, fixed memory
writer processes all events before the model sees a query. Labels are computed
separately by a dictionary interpreter. The network gets no intermediate gold
path or answer. All neural arms see the same complete active fact table.
"""
from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import random
import struct
from urllib.request import urlopen

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .runtime import atomic_json, sha256


VERSION = 'local_energy_data_v1'
FASHION_MD5 = {
    'train-images-idx3-ubyte.gz': '8d4fb7e6c68d591d4c3dfef9ec88bf0d',
    'train-labels-idx1-ubyte.gz': '25c81989df183df01b3e8a0aad5dffbe',
    't10k-images-idx3-ubyte.gz': 'bef4ecab320f06d8554ea6380940ec79',
    't10k-labels-idx1-ubyte.gz': 'bb300cfdad3c16e7a12a480ee83cd310',
}
SEEN_PATHS = ((0,), (1,), (0, 0), (0, 1), (1, 0))


@dataclass
class Dataset:
    x: Tensor
    y: Tensor
    ids: list[str]
    groups: list[str]
    memory: Tensor | None = None
    episodes: list[dict] | None = None

    def take(self, index):
        return self.x[index], self.y[index], None if self.memory is None else self.memory[index]

    def manifest(self):
        digest = hashlib.sha256()
        for name, tensor in (('x', self.x), ('y', self.y), ('memory', self.memory)):
            digest.update(name.encode())
            if tensor is not None:
                digest.update(str(tuple(tensor.shape)).encode())
                digest.update(str(tensor.dtype).encode())
                digest.update(tensor.contiguous().numpy().tobytes())
        digest.update(json.dumps([self.ids, self.groups, self.episodes], sort_keys=True).encode())
        return {'version': VERSION, 'count': len(self.y), 'sha256': digest.hexdigest(),
                'x_shape': list(self.x.shape), 'groups': {g: self.groups.count(g) for g in sorted(set(self.groups))}}


class QuadraticEventMemory:
    """Explicit E(A)=1/2 tr(A^T P A)-tr(A^T S), with analytic mode P^-1 S.

    Addresses are orthogonal supplied symbols. SET replaces the old row factor,
    DELETE stores the explicit unknown value, and HOLD is a byte-exact no-op.
    These operations equal the existing full-precision congruence update for
    unit basis keys, erase=1, write=1. The writer is fixed, not learned here.
    """
    def __init__(self, addresses: int, values: int, prior: float = .001):
        if addresses < 1 or values < 2 or not 0 < prior < float('inf'):
            raise ValueError('Invalid quadratic memory dimensions/prior')
        self.addresses, self.values, self.prior = addresses, values, prior
        self.precision = prior * torch.eye(addresses)
        self.information = torch.zeros(addresses, values)

    def apply(self, operation: str, address: int, value: int | None):
        if operation not in ('set', 'delete', 'hold') or not 0 <= address < self.addresses:
            raise ValueError('Invalid event')
        if operation == 'hold':
            return
        if operation == 'delete':
            value = self.values - 1
        if value is None or not 0 <= value < self.values:
            raise ValueError('Invalid SET value')
        self.precision[address, address] = self.prior + 1
        self.information[address].zero_()
        self.information[address, value] = 1

    def readout(self):
        mode = torch.linalg.solve(self.precision, self.information)
        # The unwritten / prior mass is explicitly represented as unknown.
        mode[:, -1] += 1 - mode.sum(-1)
        return mode

    @property
    def state_bytes(self):
        return sum(t.numel() * t.element_size() for t in (self.precision, self.information))


def execute(events, start: int, path: tuple[int, ...] | list[int], entities: int):
    """Independent dictionary reference; an unknown intermediate stays unknown."""
    active = {}
    for operation, subject, relation, obj in events:
        if operation == 'set':
            active[(subject, relation)] = obj
        elif operation == 'delete':
            active.pop((subject, relation), None)
        elif operation != 'hold':
            raise ValueError('Unknown operation')
    trace = [start]
    for relation in path:
        start = active.get((start, relation), entities) if start != entities else entities
        trace.append(start)
    return start, trace, active


def graph_dataset(split='train', count=4096, seed=481001, *, entities=6, group=None):
    if split not in ('train', 'valid', 'test') or count < 1 or entities < 2:
        raise ValueError('Invalid graph dataset recipe')
    if group is not None and split != 'test':
        raise ValueError('Special evaluation groups are test-only')
    test_groups = ('seen', 'heldout_pair', 'longer_path', 'long_delay', 'conflicting_updates', 'missing')
    if group is not None and group not in test_groups:
        raise ValueError('Unknown graph group')
    xs, ys, memories, ids, groups, episodes = [], [], [], [], [], []
    for index in range(count):
        family = group or ('seen' if split != 'test' else test_groups[index % len(test_groups)])
        identifier = hashlib.sha256(f'{VERSION}:{split}:{seed}:{family}:{index}'.encode()).hexdigest()[:24]
        rng = random.Random(int(identifier, 16))
        edges = [(s, r) for s in range(entities) for r in range(2)]
        rng.shuffle(edges)
        events = [('set', s, r, rng.randrange(entities)) for s, r in edges]
        start = rng.randrange(entities)
        if family == 'heldout_pair':
            path = (1, 1)
        elif family == 'longer_path':
            path = tuple(rng.randrange(2) for _ in range(3))
        else:
            path = SEEN_PATHS[rng.randrange(len(SEEN_PATHS))]
        # Ordinary training also includes conflicting edits and missing facts.
        for _ in range(12 if family == 'conflicting_updates' else rng.randrange(5)):
            s, r = rng.choice(edges)
            events.append(('set', s, r, rng.randrange(entities)))
        if family == 'missing' or rng.random() < .15:
            events.append(('delete', start, path[0], None))
        delay = 128 if family == 'long_delay' else rng.randrange(9)
        for _ in range(delay):
            s, r = rng.choice(edges)
            events.append(('hold', s, r, rng.randrange(entities)))
        answer, _, active = execute(events, start, path, entities)
        writer = QuadraticEventMemory(2 * entities, entities + 1)
        for operation, subject, relation, obj in events:
            writer.apply(operation, subject * 2 + relation, obj)
        # All arms get identical complete facts in their ordinary input.
        table = torch.full((entities * 2,), entities, dtype=torch.long)
        for (subject, relation), obj in active.items():
            table[subject * 2 + relation] = obj
        path_padded = torch.tensor([*path, *([2] * (3 - len(path)))])
        x = torch.cat((F.one_hot(table, entities + 1).flatten(), F.one_hot(torch.tensor(start), entities),
                       F.one_hot(path_padded, 3).flatten())).float()
        xs.append(x); ys.append(answer); memories.append(writer.readout()); ids.append(identifier); groups.append(family)
        episodes.append({'id': identifier, 'events': events, 'start': start, 'path': list(path),
                         'group': family, 'entities': entities, 'delay': delay})
    return Dataset(torch.stack(xs), torch.tensor(ys), ids, groups, torch.stack(memories), episodes)


def discrete_graph_energy(episode):
    """Exact min-sum dynamic programming over discrete path states (no embeddings).

    Factors cost 0 for following a current relation, 1 for violating it. Unknown
    is an absorbing symbol. This is a supplied-rule control, not a trained AI.
    """
    entities = episode['entities']
    _, _, active = execute(episode['events'], episode['start'], (), entities)
    costs = [float('inf')] * (entities + 1)
    costs[episode['start']] = 0.
    backpointers = []
    for relation in episode['path']:
        following, back = [], []
        for destination in range(entities + 1):
            possibilities = [costs[source] + float(destination != active.get((source, relation), entities))
                             for source in range(entities + 1)]
            source = min(range(entities + 1), key=lambda i: (possibilities[i], i))
            following.append(possibilities[source]); back.append(source)
        costs = following
        backpointers.append(back)
    answer = min(range(entities + 1), key=lambda i: (costs[i], i))
    path = [answer]
    for back in reversed(backpointers):
        path.append(back[path[-1]])
    return {'answer': answer, 'energy': costs[answer], 'candidate_energies': costs,
            'path': list(reversed(path)), 'stored_edges': len(active)}


def download_fashion(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, expected in FASHION_MD5.items():
        path = root / name
        if not path.exists():
            url = f'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/{name}'
            with urlopen(url, timeout=60) as response:
                data = response.read()
            if hashlib.md5(data).hexdigest() != expected:
                raise ValueError(f'Fashion-MNIST checksum mismatch: {name}')
            path.write_bytes(data)
        if hashlib.md5(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Fashion-MNIST checksum mismatch: {path}')
        files[name] = {'sha256': sha256(path), 'md5': expected}
    atomic_json({'source': 'https://github.com/zalandoresearch/fashion-mnist', 'files': files}, root / 'manifest.json')
    return files


def _idx(path):
    raw = gzip.decompress(path.read_bytes())
    magic = struct.unpack('>I', raw[:4])[0]
    if magic not in (2049, 2051):
        raise ValueError('Unexpected IDX format')
    dims = magic & 255
    shape = struct.unpack('>' + 'I' * dims, raw[4:4 + 4 * dims])
    array = np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * dims).copy()
    return torch.from_numpy(array.reshape(shape))


def fashion_datasets(root: Path, *, train_count=6000, valid_count=1000, test_count=2000, seed=482001):
    if min(train_count, valid_count, test_count) < 1 or train_count + valid_count > 60000 or test_count > 10000:
        raise ValueError('Invalid Fashion-MNIST subset sizes')
    train_x = _idx(root / 'train-images-idx3-ubyte.gz').flatten(1).float() / 255
    train_y = _idx(root / 'train-labels-idx1-ubyte.gz').long()
    test_x = _idx(root / 't10k-images-idx3-ubyte.gz').flatten(1).float() / 255
    test_y = _idx(root / 't10k-labels-idx1-ubyte.gz').long()
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(train_y), generator=generator)
    selected = {'train': order[:train_count], 'valid': order[train_count:train_count + valid_count],
                'test': torch.randperm(len(test_y), generator=generator)[:test_count]}
    # Scalar normalization uses only the selected training subset.
    mean, std = train_x[selected['train']].mean(), train_x[selected['train']].std().clamp_min(1e-6)
    output = {}
    for split, indices in selected.items():
        x, y = (test_x, test_y) if split == 'test' else (train_x, train_y)
        namespace = 'official_test' if split == 'test' else 'official_train'
        output[split] = Dataset((x[indices] - mean) / std, y[indices],
                                [f'fashion:{namespace}:{int(i)}' for i in indices], ['fashion'] * len(indices))
    return output


def byte_datasets(path: Path, *, context=8, train_count=6000, valid_count=1000, test_count=1000, seed=483001):
    """Nonoverlapping byte windows; raw binary features avoid a learned encoder.

    Train/validation/test occupy disjoint contiguous sections of enwik8, with
    a context-sized guard between sections. All 256 byte values remain labels.
    """
    raw = path.read_bytes()
    if len(raw) < 300000 or context < 1:
        raise ValueError('A corpus of at least 300,000 bytes and positive context are required')
    # Deliberately small corpus region; no historical LM checkpoint is loaded.
    bounds = {'train': (0, 100000), 'valid': (100000 + context, 200000),
              'test': (200000 + context, 300000)}
    result = {}
    for i, (split, count) in enumerate(zip(bounds, (train_count, valid_count, test_count))):
        lo, hi = bounds[split]
        starts = list(range(lo, hi - context, context + 1))
        rng = random.Random(seed + i)
        rng.shuffle(starts)
        if count < 1 or count > len(starts):
            raise ValueError('Too many byte windows requested')
        starts = starts[:count]
        windows = torch.tensor([list(raw[s:s + context]) for s in starts], dtype=torch.long)
        bits = ((windows[:, :, None] >> torch.arange(8)) & 1).float() * 2 - 1
        labels = torch.tensor([raw[s + context] for s in starts])
        result[split] = Dataset(bits.flatten(1), labels, [f'enwik8:{s}:{context}' for s in starts], ['bytes'] * count)
    return result
