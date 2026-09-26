"""Causal, differentiable energy summaries with an explicit layer identity."""
import torch


def energy_layer(config, layer=None):
    candidates = [i for i in range(config.layers) if config.mixer_kind(i) == 'energy']
    if not candidates:
        raise ValueError('Model contains no energy layer')
    selected = candidates[-1] if layer is None else layer
    if type(selected) is not int or selected not in candidates:
        raise ValueError('Summary layer must identify an energy mixer')
    return selected


def state_sketch(state, layer=None):
    # Legacy pure-energy callers default to the last coefficient pair. Hybrid
    # callers pass the layer selected from the model configuration explicitly.
    if layer is None:
        candidates = [i for i, s in enumerate(state) if s is not None and len(s) == 2 and s[0].ndim == 3]
        if not candidates:
            raise ValueError('No energy coefficients in state')
        layer = candidates[-1]
    p, s = state[layer]
    if p.ndim != 3 or s.ndim != 4 or s.shape[:3] != p.shape or not torch.isfinite(p).all() or not torch.isfinite(s).all() or (p <= 0).any():
        raise ValueError('Expected finite positive precision and compatible information')
    size = s.shape[-2] * s.shape[-1]
    i = torch.arange(size, device=s.device, dtype=s.dtype)
    basis = torch.cos((i + .5) * 1.618033988749895)
    basis = basis / basis.norm()
    mode = (s / p[..., None]).flatten(-2)
    return torch.stack((p.log().mean(-1), (mode * basis).sum(-1)), -1).flatten(1)


def forward_with_summaries(model, tokens, state=None, reset=None, *, layer=None, memory=None):
    """Reference streaming trace; gradients retained, no full-state history saved.

    Ordinary model.forward stays unchanged. This intentionally favors a verified
    causal reference over an unvalidated fast trace kernel.
    """
    layer = energy_layer(model.config, layer)
    if tokens.ndim != 2 or tokens.shape[1] == 0:
        raise ValueError('Nonempty [batch,time] input required')
    logits, summaries = [], []
    for t in range(tokens.shape[1]):
        r = None if reset is None else reset[:, t:t+1]
        y, state = model(tokens[:, t:t+1], state, r, memory=memory)
        logits.append(y)
        summaries.append(state_sketch(state, layer))
    return torch.cat(logits, 1), state, torch.stack(summaries, 1)
