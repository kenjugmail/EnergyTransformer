"""Lossless tensor-only distribution for the bounded v14 parser/answer model.

No AutoModel integration, hosted inference, or new training is implied. Original
.pt archives remain available. JSON hashes detect inconsistency, not forgery.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import torch
from .bridge import EnergyTextSession, TextCodec, TextParser, AnswerDecoder, ProtectedState

FORMAT = 'energy_bridge_safetensors_v16'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def export_portable(checkpoint: str | Path, directory: str | Path) -> dict:
    from safetensors.torch import save_file
    checkpoint, directory = Path(checkpoint), Path(directory)
    if directory.exists():
        raise FileExistsError('Refusing to overwrite an export directory')
    engine = EnergyTextSession(checkpoint)
    if not math.isfinite(engine.threshold):
        raise ValueError('Invalid threshold')
    directory.mkdir(parents=True)
    tensors = {}
    for prefix, module in [('parser', engine.parser), ('decoder', engine.decoder)]:
        for name, value in module.state_dict().items():
            tensors[prefix + '.' + name] = value.detach().cpu().contiguous().clone()
    weight_file = directory / 'model.safetensors'
    save_file(tensors, str(weight_file), metadata={'format': FORMAT})
    config = {'format': FORMAT, 'vocabulary': engine.codec.vocabulary,
              'threshold': engine.threshold, 'source_checkpoint_sha256': engine.signature,
              'weights_sha256': sha256(weight_file),
              'scope': '16-name, 16-location bounded parser and byte answer head; not a general LLM'}
    (directory / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    return config


class PortableEnergyTextSession(EnergyTextSession):
    def __init__(self, directory: str | Path):
        from safetensors.torch import load_file
        directory = Path(directory)
        config_path, weights_path = directory / 'config.json', directory / 'model.safetensors'
        config = json.loads(config_path.read_text())
        required = {'format', 'vocabulary', 'threshold', 'source_checkpoint_sha256', 'weights_sha256', 'scope'}
        if set(config) != required or config['format'] != FORMAT:
            raise ValueError('Unsupported portable configuration')
        if sha256(weights_path) != config['weights_sha256']:
            raise ValueError('Weight-file checksum mismatch')
        tensors = load_file(str(weights_path), device='cpu')
        if any(not torch.isfinite(t).all() for t in tensors.values()):
            raise ValueError('Nonfinite checkpoint tensor')
        self.codec = TextCodec(config['vocabulary'])
        self.parser = TextParser(len(self.codec.vocabulary))
        self.decoder = AnswerDecoder()
        expected = {p + '.' + k for p, m in [('parser', self.parser), ('decoder', self.decoder)] for k in m.state_dict()}
        if set(tensors) != expected:
            raise ValueError('Unexpected or missing weight tensors')
        for prefix, module in [('parser', self.parser), ('decoder', self.decoder)]:
            module.load_state_dict({k[len(prefix)+1:]: v for k, v in tensors.items() if k.startswith(prefix + '.')}, strict=True)
            module.eval()
        self.threshold = float(config['threshold'])
        if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError('Invalid threshold')
        self.signature = hashlib.sha256((FORMAT + sha256(config_path) + config['weights_sha256']).encode()).hexdigest()
        self.state = ProtectedState.initial()
