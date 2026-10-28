"""Check the final artifact manifest without modifying any experiment."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from energy_real.runtime import sha256, source_identity


def main():
    manifest = json.loads(Path(__file__).with_name('artifacts.json').read_text())
    assert source_identity() == manifest['source_sha256']
    mismatches = [name for name, digest in manifest['files'].items()
                  if not (ROOT/name).is_file() or sha256(ROOT/name) != digest]
    if mismatches:
        raise RuntimeError(f'Changed artifacts: {mismatches}')
    for relative, digest in manifest['preserved_files'].items():
        assert sha256(ROOT/relative) == digest, relative
    print(json.dumps({'verified_files':len(manifest['files']),
                      'preserved_files':len(manifest['preserved_files']),
                      'source_sha256':manifest['source_sha256']},indent=2))


if __name__ == '__main__':
    main()
