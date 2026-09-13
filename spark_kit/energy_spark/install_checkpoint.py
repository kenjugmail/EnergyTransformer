"""Install only the hash-verified original native checkpoint from a local file/kit.

No network calls, credentials, training, or arbitrary archive extraction.
"""
from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import zipfile
from .doctor import EXPECTED, ROOT

NAME = 'v21_energy_973k.pt'
EXPECTED_BYTES = 3916246


def install(source: Path, root: Path = ROOT) -> dict:
    source, root = Path(source), Path(root)
    if not source.is_file() or source.is_symlink():
        raise ValueError('Supply a regular local checkpoint or standalone kit ZIP')
    target = root / 'checkpoints' / NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise ValueError('Refusing a symlink destination')
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != EXPECTED:
            raise FileExistsError('A different checkpoint is already present; nothing was changed')
        return {'status': 'ALREADY_VERIFIED', 'sha256': EXPECTED}
    fd, temporary = tempfile.mkstemp(prefix=NAME + '.', suffix='.tmp', dir=target.parent)
    try:
        with os.fdopen(fd, 'wb') as output:
            if source.suffix.lower() == '.zip':
                with zipfile.ZipFile(source) as archive:
                    entries = [i for i in archive.infolist()
                               if i.filename.endswith('/checkpoints/' + NAME)
                               or i.filename == 'checkpoints/' + NAME]
                    if len(entries) != 1 or entries[0].file_size != EXPECTED_BYTES:
                        raise ValueError('Expected exactly one original checkpoint in the kit')
                    with archive.open(entries[0]) as incoming:
                        shutil.copyfileobj(incoming, output, length=1024 * 1024)
            else:
                if source.stat().st_size != EXPECTED_BYTES:
                    raise ValueError('Unexpected original checkpoint size')
                with source.open('rb') as incoming:
                    shutil.copyfileobj(incoming, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        temporary = Path(temporary)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != EXPECTED:
            raise ValueError('Original checkpoint checksum mismatch')
        # Hard-link insertion is atomic and refuses a destination created by a race.
        os.link(temporary, target)
        return {'status': 'INSTALLED_VERIFIED_ORIGINAL', 'sha256': EXPECTED,
                'bytes': target.stat().st_size}
    finally:
        Path(temporary).unlink(missing_ok=True)


if __name__ == '__main__':
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    print(json.dumps(install(args.source), indent=2))
