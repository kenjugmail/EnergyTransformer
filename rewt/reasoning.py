"""Learned read/revise workspace for bounded rule chains.

This is an opt-in structured reasoning head, not a replacement language model.
Only bytes and their lengths enter inference. Fact extraction, initial query,
and read count are learned. Annotation/parsing lives in the training script.
Every read defines a NEW conditional quadratic/Fourier energy. Backtracking
guarantees descent within a read only; energies across reads are incomparable.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ReasoningConfig:
    width: int = 48
    modes: int = 16
    max_hops: int = 3
    energy_rounds: int = 2
    step_size: float = .5
    backtracks: int = 4
    reread: bool = True

    def validate(self):
        for name in ('width', 'modes', 'max_hops', 'backtracks'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if type(self.energy_rounds) is not int or self.energy_rounds < 0:
            raise ValueError('energy_rounds must be nonnegative')
        if not math.isfinite(self.step_size) or self.step_size <= 0:
            raise ValueError('step_size must be finite and positive')
        if type(self.reread) is not bool:
            raise ValueError('reread must be boolean')


class ReadReviseReasoner(nn.Module):
    def __init__(self, config=ReasoningConfig()):
        super().__init__()
        config.validate()
        self.config = config
        d = config.width
        self.embedding = nn.Embedding(256, d)
        self.local = nn.Conv1d(d, d, 5)
        self.context = nn.Conv1d(d, d, 5)
        self.norm = nn.LayerNorm(d)
        self.key = nn.Linear(d, 256)
        self.value = nn.Linear(d, 256)
        self.fact = nn.Linear(d, 1)
        self.start = nn.Linear(2*d, 1)
        self.count_attention = nn.Linear(d, 1)
        self.count = nn.Linear(d, config.max_hops)
        self.amplitude = nn.Linear(2 * d, config.modes)
        self.raw_frequency = nn.Parameter(torch.randn(config.modes, 2*d+256) / math.sqrt(2*d+256))
        self.phase = nn.Parameter(torch.rand(config.modes) * 2*math.pi - math.pi)
        self.raw_precision = nn.Parameter(torch.zeros(2*d+256))

    def encode(self, tokens):
        x = self.embedding(tokens).transpose(1, 2)
        x = F.silu(self.local(F.pad(x, (4, 0))))
        x = self.norm((x + F.silu(self.context(F.pad(x, (4, 0))))).transpose(1, 2))
        return x

    def frequencies(self):
        f = self.raw_frequency
        d = 2*self.config.width
        f = torch.cat((f[:, :d], f[:, d:]-f[:, d:].mean(-1, keepdim=True)), -1)
        norm = f.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return 2*f*norm.tanh()/norm

    @staticmethod
    def energy(displacement, amplitude, frequency, phase, precision):
        return .5*(precision*displacement.square()).sum(-1) + (amplitude*F.linear(displacement, frequency, phase).cos()).sum(-1)

    @staticmethod
    def gradient(displacement, amplitude, frequency, phase, precision):
        return precision*displacement - (amplitude*F.linear(displacement, frequency, phase).sin()) @ frequency

    def refine(self, anchor, amplitude, *, rounds=None, trace=False):
        rounds = self.config.energy_rounds if rounds is None else rounds
        if type(rounds) is not int or rounds < 0:
            raise ValueError('rounds must be nonnegative')
        frequency, precision = self.frequencies(), .15 + 1.85*self.raw_precision.sigmoid()
        delta = torch.zeros_like(anchor)
        energies = [self.energy(delta, amplitude, frequency, self.phase, precision).detach()] if trace else []
        for _ in range(rounds):
            before = self.energy(delta, amplitude, frequency, self.phase, precision)
            gradient = self.gradient(delta, amplitude, frequency, self.phase, precision)
            chosen, accepted = delta, torch.zeros_like(before, dtype=torch.bool)
            for k in range(self.config.backtracks):
                candidate = delta - self.config.step_size * .5**k * gradient
                after = self.energy(candidate, amplitude, frequency, self.phase, precision)
                ok = torch.isfinite(after) & (after < before) & ~accepted
                chosen = torch.where(ok[:, None], candidate, chosen)
                accepted = accepted | ok
            delta = chosen
            if trace:
                energies.append(self.energy(delta, amplitude, frequency, self.phase, precision).detach())
        return anchor + delta, torch.stack(energies, -1) if trace else None

    def forward(self, tokens, lengths=None, *, details=False, energy_rounds=None, reread=None):
        if tokens.ndim != 2 or tokens.dtype != torch.long or not all(tokens.shape):
            raise ValueError('tokens must be nonempty [batch,time] int64')
        if bool(((tokens < 0) | (tokens > 255)).any()):
            raise ValueError('tokens must be bytes')
        if lengths is None:
            lengths = tokens.new_full((len(tokens),), tokens.shape[1])
        if (lengths.shape != (len(tokens),) or lengths.dtype != torch.long
                or lengths.device != tokens.device or bool(((lengths < 1) | (lengths > tokens.shape[1])).any())):
            raise ValueError('lengths must describe nonempty supplied prefixes')
        reread = self.config.reread if reread is None else reread
        if type(reread) is not bool:
            raise ValueError('reread must be boolean')
        mask = torch.arange(tokens.shape[1], device=tokens.device)[None] < lengths[:, None]
        encoded = self.encode(tokens)
        key_logits, value_logits = self.key(encoded), self.value(encoded)
        keys, values = key_logits.softmax(-1), value_logits.softmax(-1)
        fact_logits = self.fact(encoded).squeeze(-1)
        # This head consumes a supplied complete prompt, not streaming LM
        # positions. One following OBSERVED position helps distinguish a query
        # letter from a fact's source. Mask it at each real prefix boundary.
        following = torch.cat((encoded[:, 1:], torch.zeros_like(encoded[:, :1])), 1)
        following_mask = torch.cat((mask[:, 1:], torch.zeros_like(mask[:, :1])), 1)
        start_logits = self.start(torch.cat((encoded, following*following_mask[..., None]), -1)).squeeze(-1).masked_fill(~mask, -torch.inf)
        start_attention = start_logits.softmax(-1)
        # Differentiable byte copy; no position is selected by a parser.
        initial = encoded.new_zeros(len(tokens), 256).scatter_add(1, tokens, start_attention)
        observed = self.embedding(tokens)
        count_attention = self.count_attention(observed).squeeze(-1).masked_fill(~mask, -torch.inf).softmax(-1)
        count_logits = self.count((count_attention[..., None]*observed).sum(1))
        query = initial
        step_logits, pointer_logits, energy_traces = [], [], []
        for _ in range(self.config.max_hops):
            match = (keys*query[:, None]).sum(-1).clamp_min(1e-12)
            scores = (match.log() + F.logsigmoid(fact_logits)).masked_fill(~mask, -torch.inf)
            attention = scores.softmax(-1)
            value = (attention[..., None]*values).sum(1)
            current = query @ self.embedding.weight
            read = (attention[..., None]*encoded).sum(1)
            anchor_logits = value.clamp_min(1e-12).log()
            anchor_logits = anchor_logits - anchor_logits.mean(-1, keepdim=True)
            anchor = torch.cat((current, read, anchor_logits), -1)
            amplitude = .5*self.amplitude(torch.cat((current, read), -1)).tanh()/math.sqrt(self.config.modes)
            joint, trajectory = self.refine(anchor, amplitude, rounds=energy_rounds, trace=details)
            logits = joint[:, 2*self.config.width:]
            query = logits.softmax(-1) if reread else initial
            step_logits.append(logits); pointer_logits.append(scores)
            if details:
                energy_traces.append(trajectory)
        steps = torch.stack(step_logits, 1)
        answer_probability = (count_logits.softmax(-1)[..., None]*steps.softmax(-1)).sum(1)
        result = dict(answer_logits=answer_probability.clamp_min(1e-12).log(), step_logits=steps,
                      count_logits=count_logits)
        if details or self.training:
            result.update(key_logits=key_logits, value_logits=value_logits, fact_logits=fact_logits,
                          start_logits=start_logits, pointer_logits=torch.stack(pointer_logits, 1), mask=mask)
        if details:
            result['energy_trace'] = torch.stack(energy_traces, 1)
            result['energy_contract'] = 'nonincreasing within each read; different energy between reads'
        return result

    @torch.inference_mode()
    def predict(self, prompts, **kwargs):
        self.eval()
        encoded = [p.encode('utf-8') if isinstance(p, str) else bytes(p) for p in prompts]
        if not encoded or any(not p for p in encoded):
            raise ValueError('nonempty prompts required')
        device = self.embedding.weight.device
        lengths = torch.tensor([len(p) for p in encoded], device=device)
        tokens = torch.zeros(len(encoded), int(lengths.max()), dtype=torch.long, device=device)
        for i, raw in enumerate(encoded):
            tokens[i, :len(raw)] = torch.tensor(list(raw), device=device)
        output = self(tokens, lengths, **kwargs)
        rows = []
        for i, n in enumerate(output['count_logits'].argmax(-1).tolist()):
            trace = output['step_logits'][i, :n+1].argmax(-1).tolist()
            answer = int(output['answer_logits'][i].argmax())
            rows.append(dict(trace_bytes=trace, answer_byte=answer, predicted_hops=n+1,
                             response='steps='+','.join(bytes([v]).decode('latin1') for v in trace)
                                      +';ans='+bytes([answer]).decode('latin1')+'\n'))
        return rows


def load_reasoner(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('arm') != 'read_revise_reasoner':
        raise ValueError('not a read/revise checkpoint')
    model = ReadReviseReasoner(ReasoningConfig(**saved['config']))
    model.load_state_dict(saved['weights'], strict=True)
    return model, saved
