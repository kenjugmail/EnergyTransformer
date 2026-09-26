"""Verify preserved evidence, links, source and new artifact hashes."""
from pathlib import Path
import json
import re

from energy_real.runtime import atomic_json, sha256, source_identity


def main():
    report = Path(__file__).resolve().parent
    root = report.parents[1]
    before = json.loads((root/'runs/competitive_preservation/before.json').read_text())
    preserved = {p:h for p,h in before.items() if p.startswith('reports/') or p.endswith('.pt')}
    assert all(sha256(root/p)==h for p,h in preserved.items())
    plan = json.loads((report/'plan.json').read_text())
    assert source_identity()==plan['source_sha256']
    broken = []
    checked = 0
    for path in report.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if target.startswith(('https:', 'http:', '#')):
                continue
            checked += 1
            if not (path.parent/target.split('#')[0]).exists():
                broken.append([str(path),target])
    assert not broken,broken
    files = []
    for folder in ('energy_real','energy_transformer','spark_kit/energy_spark'):
        files.extend((root/folder).glob('*.py'))
    files.extend(p for p in report.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.name!='artifacts.json')
    files.extend(p for p in (root/'runs/competitive_native').rglob('*') if p.is_file())
    files.extend(root/p for p in ('README.md','ARCHITECTURE.md','FINDINGS.md','energy_real/README.md',
        'scripts/sample_native.py','tests_real/test_window_hybrid.py','tests_real/test_training.py',
        'runs/competitive_native.log','runs/competitive_preservation/before.tar.gz',
        'runs/competitive_preservation/before.json','runs/competitive_preservation/training_source.tar.gz',
        'runs/competitive_preservation/training_source.json',
        'runs/competitive_install/wheels/energy_transformer_research-0.16.0-py3-none-any.whl'))
    files.extend(root/p for p in preserved if p.endswith('.pt'))
    manifest = {'source_sha256':source_identity(), 'preserved_files':len(preserved), 'local_links_checked':checked,
                'files':{str(p.relative_to(root)):sha256(p) for p in sorted(set(files))}}
    atomic_json(manifest,report/'artifacts.json')
    assert all(sha256(root/p)==h for p,h in manifest['files'].items())
    print(json.dumps({'artifacts_verified':len(manifest['files']), 'preserved_files':len(preserved),
                      'local_links_checked':checked}))


if __name__ == '__main__': main()
