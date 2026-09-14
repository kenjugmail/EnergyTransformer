"""enwik8 with fixed 90M/5M/5M byte splits and explicit file identity.

Scores reset context at each window. Sharing split boundaries with other work
alone does not make protocols or bits-per-byte numbers directly comparable.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import torch

SPLITS = {'train': (0, 90_000_000), 'valid': (90_000_000, 95_000_000), 'test': (95_000_000, 100_000_000)}
EXPECTED_BYTES = 100_000_000
# Fingerprint of the preserved local corpus, not a claim about its provenance.
LOCAL_CORPUS_SHA256 = '2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8'


class Enwik8:
    def __init__(self, path: str | Path, split: str = 'train', *, expected_sha256: str | None = None):
        path = Path(path)
        if split not in SPLITS:
            raise ValueError(f'split must be one of {sorted(SPLITS)}')
        size = path.stat().st_size
        if size != EXPECTED_BYTES:
            raise ValueError(f'{path} is {size} bytes; enwik8 must be exactly {EXPECTED_BYTES}')
        lo, hi = SPLITS[split]
        self.split = split
        self.data = np.memmap(path, dtype=np.uint8, mode='r')[lo:hi]
        self.path = path
        if expected_sha256 is not None and self.file_identity() != expected_sha256:
            raise ValueError('Corpus checksum mismatch')

    def __len__(self):
        return len(self.data)

    def batch(self, rng: np.random.Generator, batch: int, length: int, device=None):
        """Random crops. Returns (inputs, targets), each [batch, length] int64."""
        self._validate_window(length, batch)
        if length >= len(self.data):
            raise ValueError('Sequence must leave at least one next-byte target')
        starts = rng.integers(0, len(self.data) - length, size=batch)
        window = np.stack([np.asarray(self.data[s:s + length + 1], dtype=np.int64) for s in starts])
        tok = torch.from_numpy(window)
        x, y = tok[:, :-1], tok[:, 1:]
        if device is not None:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        return x, y

    def sequential_windows(self, length: int, limit: int | None = None, batch: int = 1, device=None):
        """Windows covering consecutive targets in split order.

        Adjacent windows share one boundary byte as input. Every byte after the
        first byte of the evaluated region is a target exactly once.
        Windows are grouped into batches of `batch`; a ragged final group is
        yielded as-is rather than padded, so no target is invented or dropped.
        """
        self._validate_window(length, batch)
        if limit is not None and (type(limit) is not int or limit < 2):
            raise ValueError('Evaluation limit must contain at least two bytes')
        n = len(self.data) if limit is None else min(limit, len(self.data))
        starts = [s for s in range(0, n - 1, length) if min(s + length + 1, n) - s >= 2]
        for i in range(0, len(starts), batch):
            group = starts[i:i + batch]
            widths = [min(s + length + 1, n) - s for s in group]
            # keep a batch rectangular: only group windows of equal width
            full = [s for s, w in zip(group, widths) if w == length + 1]
            short = [(s, w) for s, w in zip(group, widths) if w != length + 1]
            if full:
                arr = np.stack([np.asarray(self.data[s:s + length + 1], dtype=np.int64) for s in full])
                tok = torch.from_numpy(arr)
                if device is not None:
                    tok = tok.to(device, non_blocking=True)
                yield tok[:, :-1], tok[:, 1:]
            for s, w in short:
                tok = torch.from_numpy(np.asarray(self.data[s:s + w], dtype=np.int64))[None]
                if device is not None:
                    tok = tok.to(device, non_blocking=True)
                yield tok[:, :-1], tok[:, 1:]

    @staticmethod
    def _validate_window(length, batch):
        if any(type(v) is not int or v < 1 for v in (length, batch)):
            raise ValueError('Positive integer length and batch required')

    def file_identity(self):
        from .runtime import sha256
        return sha256(self.path)

    def identity(self):
        h = hashlib.sha256()
        h.update(f'enwik8:{self.split}:'.encode())
        step = 1 << 20
        for i in range(0, len(self.data), step):
            h.update(np.asarray(self.data[i:i + step]).tobytes())
        return h.hexdigest()


def bits_per_byte(total_nll_nats: float, n_targets: int) -> float:
    import math
    return total_nll_nats / n_targets / math.log(2)
