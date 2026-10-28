"""Document-isolated sampling and memory-bounded full-vocabulary distillation."""
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class ConversationTokens:
    def __init__(self, directory, split, length):
        if split not in ('train', 'valid', 'test') or length < 1:
            raise ValueError('invalid corpus split or sequence length')
        directory = Path(directory)
        self.tokens = np.load(directory/f'{split}_tokens.npy', mmap_mode='r')
        self.offsets = np.load(directory/f'{split}_offsets.npy')
        self.length = length
        self.eligible = np.flatnonzero(np.diff(self.offsets) >= length+1)
        if not len(self.eligible):
            raise ValueError('no documents long enough for requested sequence length')

    def batch(self, batch_size, rng, device, *, prefix_fraction=0.):
        if not 0 <= prefix_fraction <= 1:
            raise ValueError('prefix fraction must be between zero and one')
        rows = []
        for index in rng.choice(self.eligible, size=batch_size):
            first, last = self.offsets[index:index+2]
            # Preserve the original random-crop RNG stream when fraction is 0.
            prefix = prefix_fraction == 1 or (prefix_fraction > 0 and rng.random() < prefix_fraction)
            start = int(first) if prefix else int(rng.integers(first, last-self.length))
            rows.append(np.array(self.tokens[start:start+self.length+1], dtype=np.int64))
        sequence = torch.from_numpy(np.stack(rows)).to(device)
        return sequence[:, :-1], sequence[:, 1:]


def relative_mse(student, teacher):
    return (student.float()-teacher.float()).square().mean()/teacher.float().square().mean().clamp_min(1e-6)


def vocabulary_losses(student_hidden, teacher_hidden, targets, student_weight, teacher_weight,
                      *, chunk=32, temperature=1., checkpoint_chunks=True, token_weights=None):
    """Exact KL(teacher || student), hard NLL and teacher NLL, averaged per token.

    Projection and softmax use all vocabulary entries. Checkpoint each token
    chunk so backward recomputes large logits rather than retaining them all.
    Frozen teacher tensors must be created under no_grad, not inference_mode.
    """
    if chunk < 1 or temperature <= 0:
        raise ValueError('invalid loss chunk or temperature')
    sh = student_hidden.reshape(-1, student_hidden.shape[-1])
    th = teacher_hidden.reshape(-1, teacher_hidden.shape[-1])
    labels = targets.reshape(-1)
    if len(sh) != len(th) or len(sh) != len(labels):
        raise ValueError('hidden and target token counts differ')
    weights = torch.ones_like(labels, dtype=torch.float32) if token_weights is None else token_weights.reshape(-1).float()
    if weights.shape != labels.shape or not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()) or not bool(weights.sum() > 0):
        raise ValueError('token weights must be finite, nonnegative, aligned and have positive total')
    if token_weights is not None:
        # Right-padding has zero loss and gradient; do not project it to the
        # 128k vocabulary. Boolean indexing preserves gradients on real tokens.
        active = weights > 0
        sh, th, labels, weights = sh[active], th[active], labels[active], weights[active]
    denominator = weights.sum()

    def calculate(s, t, y, w):
        logits = F.linear(s, student_weight).float()
        with torch.no_grad():
            reference = F.linear(t, teacher_weight).float()
            reference_logp = F.log_softmax(reference/temperature, -1)
            reference_p = reference_logp.exp()
            teacher_nll = (F.cross_entropy(reference, y, reduction='none')*w).sum()
        logp = F.log_softmax(logits/temperature, -1)
        kl = ((reference_p*(reference_logp-logp)).sum(-1)*w).sum()*temperature**2
        nll = (F.cross_entropy(logits, y, reduction='none')*w).sum()
        agreement = ((logits.argmax(-1) == reference.argmax(-1)).float()*w).sum()
        return torch.stack((kl, nll, teacher_nll, agreement))

    total = sh.new_zeros(4, dtype=torch.float32)
    for start in range(0, len(sh), chunk):
        stop = min(start+chunk, len(sh))
        values = (sh[start:stop], th[start:stop], labels[start:stop], weights[start:stop])
        result = checkpoint(calculate, *values, use_reentrant=False) if checkpoint_chunks and torch.is_grad_enabled() else calculate(*values)
        total = total+result/denominator
    return dict(zip(('kl', 'nll', 'teacher_nll', 'agreement'), total.unbind()))


class RecoveryTokens:
    """Include short conversations with right padding; loss never sees padding.

    The networks are causal, so right padding cannot affect real-token states.
    Each row is a single document; positions and memory reset between examples.
    """
    def __init__(self, directory, split, length, pad_id, prompt_weight=.1):
        directory = Path(directory)
        self.tokens = np.load(directory/f'{split}_tokens.npy', mmap_mode='r')
        self.masks = np.load(directory/f'{split}_assistant.npy', mmap_mode='r')
        self.offsets = np.load(directory/f'{split}_offsets.npy')
        self.length, self.pad_id, self.prompt_weight = length, pad_id, prompt_weight
        if length < 1 or not 0 <= prompt_weight <= 1 or len(self.tokens) != len(self.masks):
            raise ValueError('invalid recovery corpus or settings')
        self.eligible = np.flatnonzero(np.diff(self.offsets) >= 2)
        if not len(self.eligible):
            raise ValueError('empty recovery corpus')

    def batch(self, batch_size, rng, device, *, prefix_fraction=.75):
        if not 0 <= prefix_fraction <= 1:
            raise ValueError('invalid prefix fraction')
        rows, masks = [], []
        for index in rng.choice(self.eligible, size=batch_size):
            first, last = map(int, self.offsets[index:index+2])
            start = first
            if last-first > self.length+1 and rng.random() >= prefix_fraction:
                start = int(rng.integers(first, last-self.length))
            stop = min(start+self.length+1, last)
            rows.append(np.array(self.tokens[start:stop], dtype=np.int64))
            masks.append(np.array(self.masks[start+1:stop], dtype=np.float32))
        width = max(map(len, rows))
        tokens = np.full((batch_size, width), self.pad_id, dtype=np.int64)
        weights = np.zeros((batch_size, width-1), dtype=np.float32)
        active = np.zeros_like(weights)
        for i, (row, mask) in enumerate(zip(rows, masks)):
            tokens[i, :len(row)] = row
            weights[i, :len(row)-1] = self.prompt_weight+(1-self.prompt_weight)*mask
            active[i, :len(row)-1] = 1
        sequence = torch.from_numpy(tokens).to(device)
        return sequence[:, :-1], sequence[:, 1:], torch.from_numpy(weights).to(device), torch.from_numpy(active).to(device)


class TeacherOperators:
    """Capture donor inputs and post-normalized operator outputs, excluding FFN."""
    def __init__(self, teacher):
        self.inputs = {}; self.outputs = {}; self.handles = []
        for i, block in enumerate(teacher.model.layers):
            self.handles.append(block.operator_norm.register_forward_pre_hook(
                lambda module, args, index=i: self._input(index, args[0])))
            self.handles.append(block.post_operator_norm.register_forward_hook(
                lambda module, args, value, index=i: self._output(index, value)))

    def _input(self, index, value):
        self.inputs[index] = value.detach()

    def _output(self, index, value):
        self.outputs[index] = value.detach()

    def clear(self):
        self.inputs.clear(); self.outputs.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.clear()
