#!/usr/bin/env python3
"""Upload a locally available full archive to a private-repository release.

Uses the user's existing gh authentication; never reads a token, force-pushes,
changes repository visibility, or deletes/replaces an existing release asset.
Default is local validation only. --publish authorizes external writes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import zipfile

REPO = 'kenjugmail/EnergyTransformer'
TAG = 'research-v1-v23'
MARKER = 'EnergyTransformer archive publisher v23'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check_archive(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ValueError('Archive must be a regular local file')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+\.zip', path.name):
        raise ValueError('Use a simple .zip filename without gh label syntax')
    if path.stat().st_size >= 2 * 1024**3:
        raise ValueError('Split archives of 2 GiB or larger before publishing')
    with zipfile.ZipFile(path) as z:
        bad = z.testzip()
        if bad is not None:
            raise ValueError('Corrupt archive member: ' + bad)
        seen = set()
        for info in z.infolist():
            name = info.filename
            if (name in seen or name.startswith(('/', '\\')) or '\\' in name
                    or '..' in Path(name).parts or ((info.external_attr >> 16) & 0o170000) == 0o120000):
                raise ValueError('Unsafe or duplicate archive member: ' + name)
            seen.add(name)
    return sha256(path)


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(args, capture_output=True, text=True)
    if check and result.returncode:
        raise RuntimeError('Command failed: ' + ' '.join(args) + '\n' + result.stderr)
    return result


def release_metadata(tag: str, runner=run) -> dict | None:
    result = runner('gh', 'api', f'repos/{REPO}/releases/tags/{tag}', check=False)
    if result.returncode:
        if 'HTTP 404' in result.stderr:
            return None
        raise RuntimeError('Cannot inspect release; refusing to guess: ' + result.stderr)
    return json.loads(result.stdout)


def verify_remote(tag: str, source: Path, runner=run) -> None:
    """Download and hash, rather than treating upload success as proof of parity."""
    with tempfile.TemporaryDirectory(prefix='energy-asset-check-') as tmp:
        target = Path(tmp) / source.name
        runner('gh', 'release', 'download', tag, '--repo', REPO,
               '--pattern', source.name, '--output', str(target))
        if not target.is_file() or sha256(target) != sha256(source):
            raise RuntimeError('Remote asset bytes do not match: ' + source.name)


def upload_archive(path: Path, *, publish: bool = False, tag: str = TAG,
                   target: str = 'main', runner=run) -> dict:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,99}', tag):
        raise ValueError('Unsafe release tag')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_./-]{0,199}', target):
        raise ValueError('Unsafe target ref')
    digest = check_archive(path)
    receipt = {'repository': REPO, 'tag': tag, 'archive': path.name,
               'bytes': path.stat().st_size, 'sha256': digest, 'published': False}
    if not publish:
        return receipt
    runner('gh', 'auth', 'status', '--hostname', 'github.com')
    repo = json.loads(runner('gh', 'api', f'repos/{REPO}').stdout)
    if repo.get('full_name') != REPO or repo.get('private') is not True:
        raise RuntimeError('Expected the explicitly authorized PRIVATE repository; stopped')
    metadata = release_metadata(tag, runner)
    if metadata is None:
        runner('gh', 'release', 'create', tag, '--repo', REPO, '--target', target,
               '--title', 'EnergyTransformer complete research archive',
               '--notes', MARKER + '\nOriginal research artifacts and source; exploratory results, not a general LLM.',
               '--draft', '--prerelease', '--latest=false')
        metadata = release_metadata(tag, runner)
    if metadata is None or MARKER not in (metadata.get('body') or ''):
        raise RuntimeError('Existing release is not owned by this publisher; stopped')
    if metadata.get('immutable'):
        raise RuntimeError('Release is immutable; choose a new tag')
    existing = {asset['name'] for asset in metadata.get('assets', [])}
    # The sidecar is placed outside the original package. Neither file is overwritten remotely.
    with tempfile.TemporaryDirectory(prefix='energy-publish-') as tmp:
        checksum = Path(tmp) / (path.name + '.sha256')
        checksum.write_text(digest + '  ' + path.name + '\n', encoding='utf-8')
        for asset in (path, checksum):
            if asset.name in existing:
                verify_remote(tag, asset, runner)
            else:
                runner('gh', 'release', 'upload', tag, str(asset), '--repo', REPO)
                verify_remote(tag, asset, runner)
    # Publication happens only after both uploaded byte streams have been verified.
    if metadata.get('draft'):
        runner('gh', 'release', 'edit', tag, '--repo', REPO, '--draft=false', '--latest=false')
    final = release_metadata(tag, runner)
    if final is None or final.get('draft'):
        raise RuntimeError('Could not verify published release status')
    receipt.update(published=True, url=final['html_url'])
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--tag', default=TAG)
    parser.add_argument('--target', default='main')
    args = parser.parse_args()
    if args.publish and shutil.which('gh') is None:
        raise SystemExit('GitHub CLI is required. Install gh and authenticate with gh auth login; never paste tokens here.')
    result = upload_archive(args.archive.resolve(), publish=args.publish, tag=args.tag, target=args.target)
    print(json.dumps(result, indent=2))
    if not args.publish:
        print('Local validation only. Add --publish to upload and verify the release assets.')

if __name__ == '__main__':
    main()
