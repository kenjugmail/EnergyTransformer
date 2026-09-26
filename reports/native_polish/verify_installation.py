"""Isolated wheel verification: run with the workspace Python and -I from /tmp."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile

root = Path(__file__).resolve().parents[2]
site = root/'runs/native_polish_install/site'
sys.path.insert(0, str(site))

import torch
import energy_real
from energy_real.bench import tensor_bytes
from energy_real.model import Config, LM
from energy_real.runtime import atomic_json, sha256, source_identity
from energy_real.sample import main, sample_bytes
from energy_real.train import model_config, parser


def main_check():
    torch.set_num_threads(2)
    assert str(site) in energy_real.__file__
    files = []
    for folder in ('energy_real', 'energy_transformer'):
        for path in (root/folder).glob('*.py'):
            assert path.read_bytes() == (site/path.relative_to(root)).read_bytes()
            files.append(str(path.relative_to(root)))
    before = json.loads((root/'runs/native_polish_preservation/before.json').read_text())
    frozen = {p:h for p,h in before['files'].items() if p.startswith('reports/') or p.endswith('.pt')}
    assert all(sha256(root/p)==h for p,h in frozen.items())
    assert sha256(root/'energy_real/model.py') == before['files']['energy_real/model.py']
    cfg_checks = []
    for arm in ('energy', 'softmax', 'local_softmax', 'hybrid'):
        saved = torch.load(root/f'runs/competitive_native/selected_{arm}.pt', weights_only=True)
        options = parser().parse_args(['--out','unused', '--kind','softmax' if arm=='local_softmax' else arm,
            '--width','64','--layers','3','--heads','2','--slots','8','--ff-width','192','--chunk','32',
            '--attention-window','64' if arm in ('local_softmax','hybrid') else '0'])
        assert model_config(options) == Config(**saved['config'])
        cfg_checks.append(arm)
    checkpoint = root/'runs/competitive_native/selected_hybrid.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = LM(Config(**saved['config'])).eval()
    model.load_state_dict(saved['weights'], strict=True)
    prompt = (b'The city and its history. '*100)[:2048]
    with torch.no_grad():
        output, state = model(torch.tensor([list(prompt)]))
        expected = bytearray(prompt)
        for i in range(16):
            token = output[:, -1].argmax(-1, keepdim=True)
            expected.append(token.item())
            if i < 15: output, state = model(token, state)
    actual = sample_bytes(model, prompt, tokens=16, prefill_chunk=128)
    assert actual == bytes(expected)
    observed = []
    handle = model.register_forward_hook(lambda module,args,result: observed.append({
        'input_bytes':args[0].shape[1], 'logit_elements':result[0].numel(), 'state_bytes':tensor_bytes(result[1])}))
    try:
        long_sample = sample_bytes(model, (b'The world. '*3000)[:32768], tokens=16, prefill_chunk=256)
    finally:
        handle.remove()
    assert len(long_sample)==32784
    assert max(r['input_bytes'] for r in observed)==256
    assert max(r['state_bytes'] for r in observed)==37512
    with tempfile.TemporaryDirectory() as tmp:
        output_path = Path(tmp)/'sample.bin'
        output_text = io.StringIO()
        with redirect_stdout(output_text):
            main(['--checkpoint',str(checkpoint),'--prompt','café','--tokens','32','--temperature','.8',
                  '--top-k','40','--seed','42','--prefill-chunk','2','--output',str(output_path)])
        expected = sample_bytes(model, 'café'.encode(), tokens=32, temperature=.8, top_k=40, seed=42, prefill_chunk=2)
        assert output_path.read_bytes()==expected
        assert output_text.getvalue()==expected.decode('utf-8',errors='replace')+'\n'
    report = {'installed_package':energy_real.__file__, 'installed_source_sha256':source_identity(),
              'source_files_identical':len(files), 'files':files,
              'model_code_unchanged':True, 'preserved_reports_and_checkpoints':len(frozen),
              'training_config_matches_frozen_study':cfg_checks,
              'selected_checkpoint_sha256':sha256(checkpoint), 'greedy_parity_prompt_bytes':2048,
              'chunked_prompt_bytes':32768, 'max_forward_bytes':max(r['input_bytes'] for r in observed),
              'max_logit_bytes':max(r['logit_elements'] for r in observed)*4,
              'max_state_bytes':max(r['state_bytes'] for r in observed),
              'exact_byte_cli_output':True}
    atomic_json(report,root/'reports/native_polish/verification.json')
    print(json.dumps(report,indent=2))


if __name__ == '__main__': main_check()
