#!/usr/bin/env python3
"""Optional upload to an EXISTING PRIVATE Hugging Face model repository.

No account/repository is created, made public, or silently selected. Default is
local archive validation only. Log in with `hf auth login` on your own machine;
do not paste access tokens into chat. Original bytes are downloaded and hashed
at the returned immutable commit before the receipt says verified.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parent))
from publish_release import check_archive, sha256


def publish_archive(path: Path, repo_id: str, *, publish=False, api=None,
                    download=None, add_operation=None) -> dict:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*', repo_id):
        raise ValueError('An explicit owner/repository is required')
    checksum = check_archive(path)
    receipt = {'provider': 'huggingface', 'repository': repo_id, 'archive': path.name,
               'bytes': path.stat().st_size, 'sha256': checksum, 'verified': False}
    if not publish:
        return receipt
    if api is None:
        from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd
        api, download, add_operation = HfApi(), hf_hub_download, CommitOperationAdd
    info = api.repo_info(repo_id=repo_id, repo_type='model')
    if info.private is not True or info.id != repo_id:
        raise RuntimeError('Expected the explicitly named existing private repository')
    prefix = f'research/{checksum}'
    remote_archive = f'{prefix}/{path.name}'
    remote_checksum = remote_archive + '.sha256'
    existing = set(api.list_repo_files(repo_id=repo_id, repo_type='model', revision=info.sha))
    with tempfile.TemporaryDirectory(prefix='energy-hf-') as tmp:
        sidecar = Path(tmp)/(path.name+'.sha256')
        sidecar.write_text(checksum+'  '+path.name+'\n', encoding='utf-8')
        uploads = [(path, remote_archive),(sidecar, remote_checksum)]
        operations = []
        for local, remote in uploads:
            if remote in existing:
                received = Path(download(repo_id=repo_id, repo_type='model', filename=remote,
                                         revision=info.sha, local_dir=str(Path(tmp)/'existing'), force_download=True))
                if sha256(received) != sha256(local):
                    raise RuntimeError('Existing remote file differs; refusing overwrite')
            else:
                operations.append(add_operation(path_in_repo=remote, path_or_fileobj=str(local)))
        revision = info.sha
        if operations:
            # Optimistic concurrency: reject if the repository changes after inspection.
            commit = api.create_commit(repo_id=repo_id, repo_type='model',
                         operations=operations, parent_commit=info.sha,
                         commit_message='Add verified EnergyTransformer research archive')
            revision = commit.oid
        for local, remote in uploads:
            received = Path(download(repo_id=repo_id, repo_type='model', filename=remote,
                                     revision=revision, local_dir=str(Path(tmp)/'verify'), force_download=True))
            if sha256(received) != sha256(local):
                raise RuntimeError('Uploaded bytes failed verification; no success receipt')
        if api.repo_info(repo_id=repo_id, repo_type='model').private is not True:
            raise RuntimeError('Repository visibility changed during upload')
    receipt.update(verified=True, commit=revision, path=remote_archive)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--repo-id', required=True)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    print(json.dumps(publish_archive(args.archive.resolve(), args.repo_id, publish=args.publish), indent=2))
