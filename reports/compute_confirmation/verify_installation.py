"""Run with the workspace Python, -I, and a working directory outside the repo."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT/'runs/compute_confirmation_install/site'
sys.path.insert(0,str(SITE))
import torch
import energy_real
from energy_real.model import Config, LM
from energy_real.runtime import atomic_json, sha256
from energy_real.sample import main as sample_main, sample_bytes
from energy_real.static_cache import with_static_cache


def main():
    torch.set_num_threads(1)
    assert str(SITE) in energy_real.__file__
    files = [p for folder in ('energy_real','energy_transformer') for p in (ROOT/folder).glob('*.py')]
    assert all(p.read_bytes() == (SITE/p.relative_to(ROOT)).read_bytes() for p in files)
    selected = json.loads((ROOT/'reports/compute_confirmation/selected_checkpoint.json').read_text())
    path = ROOT/selected['path']
    assert sha256(path) == selected['sha256']
    saved = torch.load(path,weights_only=True)
    model = LM(Config(**saved['config'])).eval()
    model.load_state_dict(saved['weights'])
    prompt = ('The city and its history. '*50).encode()[:1024]
    reference = sample_bytes(model,prompt,tokens=32,prefill_chunk=128)
    assert sample_bytes(model,prompt,tokens=32,prefill_chunk=256) == reference
    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory)/'sample.bin'
        console = io.StringIO()
        with redirect_stdout(console):
            sample_main(['--checkpoint',str(path),'--prompt','The ','--tokens','32','--output',str(destination)])
        expected = sample_bytes(model,b'The ',tokens=32)
        assert destination.read_bytes() == expected
        assert console.getvalue() == expected.decode('utf-8',errors='replace')+'\n'
    full = LM(Config(kind='softmax',width=16,layers=2,heads=2,ff_width=32)).eval()
    cached = with_static_cache(full,33)
    tokens = torch.arange(33)[None]
    with torch.inference_mode():
        expected,_ = full(tokens)
        prefix,state = cached(tokens[:,:32])
        last,state = cached(tokens[:,32:],state)
    torch.testing.assert_close(torch.cat((prefix,last),1),expected,atol=3e-6,rtol=3e-6)
    wheel = next((ROOT/'runs/compute_confirmation_install/wheels').glob('*.whl'))
    result = {'package_path':energy_real.__file__,'source_files_identical':len(files),
              'wheel_sha256':sha256(wheel),'checkpoint_sha256':sha256(path),
              'prompt_bytes':len(prompt),'generated_bytes':32,'prefill_chunks_match':[128,256],
              'standalone_cli_exact_byte_parity':True,'installed_static_cache_parity':True}
    atomic_json(result,Path(__file__).with_name('installation_verification.json'))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
