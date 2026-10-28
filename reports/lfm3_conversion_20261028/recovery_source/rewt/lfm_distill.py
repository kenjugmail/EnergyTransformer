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

    def batch(self, batch_size, rng, device):
        rows = []
        for index in rng.choice(self.eligible, size=batch_size):
            first, last = self.offsets[index:index+2]
            start = int(rng.integers(first, last-self.length))
            rows.append(np.array(self.tokens[start:start+self.length+1], dtype=np.int64))
        sequence = torch.from_numpy(np.stack(rows)).to(device)
        return sequence[:, :-1], sequence[:, 1:]


def relative_mse(student, teacher):
    return (student.float()-teacher.float()).square().mean()/teacher.float().square().mean().clamp_min(1e-6)


def vocabulary_losses(student_hidden, teacher_hidden, targets, student_weight, teacher_weight,
                      *, chunk=32, temperature=1., checkpoint_chunks=True):
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

    def calculate(s, t, y):
        logits = F.linear(s, student_weight).float()
        with torch.no_grad():
            reference = F.linear(t, teacher_weight).float()
            reference_logp = F.log_softmax(reference/temperature, -1)
            reference_p = reference_logp.exp()
            teacher_nll = F.cross_entropy(reference, y)
        logp = F.log_softmax(logits/temperature, -1)
        kl = (reference_p*(reference_logp-logp)).sum(-1).mean()*temperature**2
        nll = F.cross_entropy(logits, y)
        agreement = (logits.argmax(-1) == reference.argmax(-1)).float().mean()
        return torch.stack((kl, nll, teacher_nll, agreement))

    total = sh.new_zeros(4, dtype=torch.float32)
    for start in range(0, len(sh), chunk):
        stop = min(start+chunk, len(sh))
        values = (sh[start:stop], th[start:stop], labels[start:stop])
        result = checkpoint(calculate, *values, use_reentrant=False) if checkpoint_chunks and torch.is_grad_enabled() else calculate(*values)
        total = total+result*((stop-start)/len(sh))
    return dict(zip(('kl', 'nll', 'teacher_nll', 'agreement'), total.unbind()))


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
