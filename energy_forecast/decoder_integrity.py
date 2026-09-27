"""Fail-closed identities for bounded CPU decoder studies.

These hashes detect accidental mismatches, not malicious edits. Locks are local
POSIX advisory locks: all writers must cooperate; never unlink the lock file.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import itertools
import json
import math
import os
from pathlib import Path
import re
import tempfile

import torch

from .decoder_bridge import DecoderConfig, KINDS
from .probe import checkpoint_hash


def validate_schedule(kind, seed, steps, batch, peak_lr, stop_after):
    if kind not in KINDS:
        raise ValueError('Unknown decoder kind')
    if any(type(v) is not int or v < 1 for v in (steps, batch)):
        raise ValueError('Positive integer steps and batch required')
    if type(seed) is not int or not 0 <= seed < 2**63 - 19000:
        raise ValueError('Invalid seed')
    if type(peak_lr) not in (int, float) or not math.isfinite(peak_lr) or peak_lr <= 0:
        raise ValueError('Positive finite learning rate required')
    if stop_after is not None and (type(stop_after) is not int or not 0 <= stop_after <= steps):
        raise ValueError('stop_after must be an absolute step in the planned run')


def run_identity(data, data_hash, head_hash, kind, seed, steps, batch, peak_lr):
    meta = data['metadata']
    cfg = DecoderConfig(kind=kind, width=meta['width'], history=meta['history'],
                        variables=meta['variables'], horizons=tuple(meta['horizons']))
    cfg.validate()
    sources = ('train_decoder.py', 'decoder_bridge.py', 'decoder_integrity.py',
               'experiment.py', 'model.py', 'calibrated.py')
    return {'data_sha256': data_hash, 'head_sha256': head_hash,
            'teacher_sha256': meta['teacher_sha256'], 'config': asdict(cfg),
            'seed': seed, 'steps': steps, 'batch': batch, 'peak_lr': peak_lr,
            'torch': str(torch.__version__),
            'sources': {name: checkpoint_hash(Path(__file__).with_name(name)) for name in sources}}


def same_json(a, b):
    # JSON round trips turn config tuples into lists.
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def validate_head_metadata(head_meta, data_meta):
    if head_meta.get('teacher_sha256') != data_meta['teacher_sha256']:
        raise ValueError('Teacher mismatch')
    c = head_meta.get('config', {})
    for field in ('history', 'variables', 'horizons'):
        if not same_json(c.get(field), data_meta[field]):
            raise ValueError(f'Forecast {field} differs from cached features')
    norm = head_meta.get('normalization', {})
    expected = data_meta['normalization']
    if set(norm) != {'center', 'scale'} or set(expected) != set(norm):
        raise ValueError('Forecast normalization schema mismatch')
    for name in norm:
        a, b = norm[name], expected[name]
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            raise ValueError('Normalization must contain tensors')
        if a.dtype != b.dtype or not torch.equal(a.cpu(), b.cpu()):
            raise ValueError('Forecast normalization differs from cached features')


def atomic_json(value, destination):
    """Serialize before opening a temporary file; flush then atomically replace."""
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    path = Path(destination)
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def exclusive_directory(path, *, resume):
    """Fail rather than waiting when a cooperating local writer already owns it."""
    try:
        import fcntl
    except ImportError as exc:
        raise RuntimeError('Decoder training requires local POSIX file locking') from exc
    path = Path(path)
    if resume:
        if not path.is_dir():
            raise FileNotFoundError('Resume requires an existing run directory')
    else:
        path.mkdir(parents=True, exist_ok=False)
    with (path / '.writer.lock').open('a+b') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another writer owns this decoder run') from exc
        try:
            yield path
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def selected_runs(frozen):
    """Read exact kind/seed pairs from selected files, including legacy manifests."""
    kinds = tuple(frozen['validation_means'])
    if not kinds or any(k not in KINDS for k in kinds):
        raise ValueError('Invalid selected decoder kinds')
    rows = []
    for path, digest in frozen['files'].items():
        match = re.fullmatch(r'([a-z_]+)_seed([0-9]+)/best\.pt', path)
        if match is None or match[1] not in kinds or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError('Invalid selected checkpoint path or digest')
        rows.append((match[1], int(match[2]), path, digest))
    seeds = sorted({r[1] for r in rows})
    expected = set(itertools.product(kinds, seeds))
    if not rows or len(rows) != len(expected) or {(r[0], r[1]) for r in rows} != expected:
        raise ValueError('Selection must cover every declared kind and seed')
    if 'seeds' in frozen and frozen['seeds'] != seeds:
        raise ValueError('Selected seed metadata disagrees with checkpoint paths')
    return kinds, seeds, rows
