"""Fixed enwik8 splits and answer-only synthetic-task batches."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import torch


SPLITS = {"train": (0, 90_000_000), "valid": (90_000_000, 95_000_000),
          "test": (95_000_000, 100_000_000)}
CORPUS_SHA256 = "2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Batch:
    tokens: torch.Tensor
    targets: torch.Tensor
    process_negatives: torch.Tensor | None = None

    @property
    def valid(self):
        return self.targets >= 0

    def to(self, device):
        return Batch(self.tokens.to(device), self.targets.to(device),
                     self.process_negatives.to(device) if self.process_negatives is not None else None)


class Corpus:
    def __init__(self, path, *, expected_sha256=CORPUS_SHA256):
        self.path = Path(path).resolve()
        if self.path.stat().st_size != 100_000_000:
            raise ValueError("enwik8 must contain exactly 100,000,000 bytes")
        self.sha256 = file_sha256(self.path)
        if expected_sha256 is not None and self.sha256 != expected_sha256:
            raise ValueError("enwik8 checksum differs from the predeclared corpus")
        self.data = np.memmap(self.path, dtype=np.uint8, mode="r")

    def batch(self, split, rng, batch, context, *, starts=None):
        lo, hi = SPLITS[split]
        if not (type(context) is int and context >= 1 and type(batch) is int and batch >= 1):
            raise ValueError("Positive batch and context required")
        if starts is None:
            starts = rng.integers(lo, hi - context, size=batch)
        if any(not lo <= int(s) < hi - context for s in starts):
            raise ValueError("Window crosses its declared split")
        array = np.stack([np.array(self.data[int(s):int(s) + context + 1], dtype=np.int64)
                          for s in starts])
        tokens = torch.from_numpy(array)
        return Batch(tokens[:, :-1], tokens[:, 1:])

    def offsets(self, split, count, context, seed):
        lo, hi = SPLITS[split]
        return np.random.default_rng(seed).integers(lo, hi - context, count).tolist()

    def text_prompts(self, count=30, seed=719, prefix=32, continuation=96):
        """Model-independent prefix selection; never filters on model outputs.

        Eligibility depends on the prefix alone: printable ASCII, sufficient
        letters/words, and no XML brackets. Reference continuation stays hidden
        from the generation API and is retained only for the output artifact.
        """
        lo, hi = SPLITS["test"]
        rng = np.random.default_rng(seed)
        selected, used = [], set()
        for _ in range(100_000):
            start = int(rng.integers(lo, hi - prefix - continuation))
            if start in used:
                continue
            raw = bytes(self.data[start:start + prefix])
            text = raw.decode("ascii", errors="ignore")
            if (len(text) != prefix or any(not 32 <= c < 127 for c in raw)
                    or sum(chr(c).isalpha() for c in raw) < prefix // 2
                    or len(text.split()) < 3 or "<" in text or ">" in text):
                continue
            used.add(start)
            selected.append(dict(id=f"text_{len(selected):03d}", offset=start,
                                 prompt=text, prompt_hex=raw.hex(),
                                 reference_hex=bytes(self.data[start + prefix:start + prefix + continuation]).hex()))
            if len(selected) == count:
                return selected
        raise RuntimeError("Could not find enough eligible held-out prefixes")

    def identity(self):
        return dict(path=str(self.path), sha256=self.sha256, bytes=len(self.data),
                    split_boundaries={key: list(value) for key, value in SPLITS.items()})


def task_batch(tasks, indices, *, context=128, process_supervision=False, process_seed=0):
    selected = [tasks[int(i)] for i in indices]
    encoded = [(task.prompt + task.response).encode("ascii") for task in selected]
    width = max(len(text) - 1 for text in encoded)
    if width >= context:
        raise ValueError("Prompt plus answer must fit the shared context")
    x = torch.zeros(len(selected), width, dtype=torch.long)
    y = torch.full_like(x, -100)
    negatives = torch.full_like(x, -100) if process_supervision else None
    for row, (task, raw) in enumerate(zip(selected, encoded)):
        x[row, :len(raw) - 1] = torch.tensor(list(raw[:-1]))
        first = len(task.prompt.encode("ascii")) - 1
        y[row, first:len(raw) - 1] = torch.tensor(list(raw[first + 1:]))
        if process_supervision:
            from .tasks import corrupt_trace
            wrong = corrupt_trace(task, seed=process_seed)
            position = len(task.prompt) + wrong["response_byte_offset"] - 1
            negatives[row, position] = wrong["wrong_byte"]
            assert negatives[row, position] != y[row, position]
    return Batch(x, y, negatives)
