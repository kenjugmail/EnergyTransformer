"""Package a converted checkpoint, tokenizer, license and independent inference code."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_rewt_lfm3 import sha, save_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, default=ROOT/'runs/lfm3_conversion_20261028/initial')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--report', type=Path, default=Path('reports/lfm3_conversion_20261028/FINDINGS.md'))
    a = p.parse_args()
    if a.output.exists():
        p.error('use a new bundle directory')
    a.output.mkdir(parents=True)
    try:
        os.link(a.checkpoint, a.output/'model.pt')
    except OSError:
        shutil.copy2(a.checkpoint, a.output/'model.pt')
    for name in ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                 'chat_template.jinja', 'TEACHER_LICENSE.txt']:
        shutil.copy2(a.tokenizer/name, a.output/name)
    (a.output/'rewt').mkdir()
    for name in ['lfm_conversion.py', 'streaming.py']:
        shutil.copy2(ROOT/'rewt'/name, a.output/'rewt'/name)
    (a.output/'rewt/__init__.py').write_text('"""Standalone converted REWT inference implementation."""\n')
    sampler = (ROOT/'scripts/sample_rewt_lfm3.py').read_text()
    sampler = sampler.replace('ROOT = Path(__file__).resolve().parents[1]', 'ROOT = Path(__file__).resolve().parent')
    sampler = sampler.replace("p.add_argument('--checkpoint', type=Path, required=True)",
                              "p.add_argument('--checkpoint', type=Path, default=ROOT/'model.pt')")
    sampler = sampler.replace("default=ROOT/'runs/lfm3_conversion_20261028/initial'", 'default=ROOT')
    (a.output/'sample.py').write_text(sampler)
    (a.output/'requirements.txt').write_text('torch>=2.5\ntransformers==5.21.0\n')
    (a.output/'README.txt').write_text(f'''LFM3-1.7B → streaming REWT experimental recovered model

This directory contains the recovered 1.784B-parameter checkpoint, inherited
tokenizer, donor license, and Python inference implementation. The original
donor checkpoint and LFM3 model code are not required for inference. It is a custom PyTorch
checkpoint, not a Hugging Face AutoModel checkpoint.

With the requirements installed in a CUDA-capable environment, run:
  python sample.py --prompt 'Reply with exactly the word hello.' --tokens 48
Use --device cpu for CPU execution. Greedy generation is the default.

The checkpoint is an experimental recovery artifact. General instruction following,
knowledge, tools, coding and long-context quality have not
been certified as equivalent to the donor. See the repository report at
{a.report} for complete measurements and
unedited sample outputs. TEACHER_LICENSE.txt retains the donor's license;
the converted weights are not relicensed by this package.
''')
    save_json(a.output/'manifest.json', dict(format='rewt_lfm_conversion_bundle_v1',
        checkpoint_source=str(a.checkpoint.resolve()),
        files={str(f.relative_to(a.output)):dict(bytes=f.stat().st_size, sha256=sha(f))
               for f in a.output.rglob('*') if f.is_file()}))
    print(a.output.resolve())


if __name__ == '__main__':
    main()
