"""Atomic research artifacts, run ownership, and reproducible identities."""
from __future__ import annotations
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import torch


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def source_identity():
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for folder in ('energy_real', 'energy_transformer', 'spark_kit/energy_spark'):
        for p in sorted((root / folder).glob('*.py')):
            h.update(str(p.relative_to(root)).encode() + b'\0' + p.read_bytes())
    return h.hexdigest()


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_json(value, path):
    raw = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    atomic_write(path, lambda f: f.write(raw))


def atomic_checkpoint(value, path):
    atomic_write(path, lambda f: torch.save(value, f))


@contextmanager
def exclusive_run(path, resume=False):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with (path / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another process owns this run') from exc
        if not resume and any(p.name != '.lock' for p in path.iterdir()):
            raise FileExistsError('Run directory is not empty; use a new path or --resume')
        if resume and not (path / 'last.pt').is_file():
            raise FileNotFoundError('Exact resume requires last.pt, including optimizer and RNG state')
        yield path


def log_identity(path):
    raw = Path(path).read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def reconcile_log(path, committed):
    """A checkpoint commits an exact log prefix; preserve anything beyond it."""
    path = Path(path)
    raw = path.read_bytes() if path.exists() else b''
    n = committed['bytes']
    if len(raw) < n or hashlib.sha256(raw[:n]).hexdigest() != committed['sha256']:
        raise ValueError('Corrupt committed training log; refusing resume')
    tail = raw[n:]
    if tail:
        fd, backup = tempfile.mkstemp(prefix='recovered_steps.', suffix='.jsonl', dir=path.parent)
        with os.fdopen(fd, 'wb') as f:
            f.write(tail)
            f.flush()
            os.fsync(f.fileno())
        atomic_write(path, lambda f: f.write(raw[:n]))
    return bool(tail)


def sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def environment(device):
    import platform
    device = torch.device(device)
    return {'torch': str(torch.__version__), 'machine': platform.machine(),
            'device': str(device), 'cuda': torch.version.cuda,
            'threads': torch.get_num_threads(),
            'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None}
