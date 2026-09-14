"""Portable, checksum-verified experiment bundles (binaries stay outside Git)."""
import argparse
import copy
import json
from pathlib import Path
import shutil
from .runtime import atomic_json, sha256


def resolve_artifact(value, expected, manifest, artifact_root=None):
    path = Path(value)
    if artifact_root is not None:
        root = Path(artifact_root).resolve()
        if path.is_absolute():
            raise ValueError('Artifact root requires relative bundle paths; export the legacy selection first')
        path = (root / path).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Artifact path escapes artifact root')
    elif not path.is_absolute():
        root = Path(manifest).resolve().parent
        path = (root / path).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Artifact path escapes bundle directory')
    if not path.is_file():
        raise FileNotFoundError(f'Missing artifact {path}; restore the bundle or pass --artifact-root to its relocated root')
    if sha256(path) != expected:
        raise ValueError(f'Artifact checksum changed: {path}')
    return path


def export_bundle(selection, out, artifact_root=None):
    selection, out = Path(selection), Path(out)
    frozen = json.loads(selection.read_text())
    if frozen.get('status') != 'frozen_for_test':
        raise ValueError('A frozen selection is required')
    dataset = resolve_artifact('datasets.json', frozen['dataset_sha256'], selection)
    copies = []
    result = copy.deepcopy(frozen)
    # Nonselected screening records are historical metadata, not bundle dependencies.
    result.pop("all_records", None)
    for record in result['selected']:
        for field, hash_field in [('controller','controller_sha256'), ('initial_controller','initial_sha256')]:
            source = resolve_artifact(record[field], record[hash_field], selection, artifact_root)
            relative = f"artifacts/{record[hash_field]}.pt"
            copies.append((source, relative))
            record[field] = relative
    # Validate every input before creating output; never overwrite an earlier bundle.
    out.mkdir(parents=True, exist_ok=False)
    (out/'artifacts').mkdir()
    for source, relative in copies:
        shutil.copyfile(source, out/relative)
    shutil.copyfile(dataset, out/'datasets.json')
    result['bundle_format'] = 'energy_experiment_bundle_v1'
    result['origin_selection_sha256'] = sha256(selection)
    atomic_json(result, out/'selection.json')
    atomic_json({'format':result['bundle_format'], 'files':{
        str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob('*')) if p.is_file()}}, out/'bundle.json')
    return out/'selection.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--artifact-root', type=Path)
    args = parser.parse_args()
    print(export_bundle(args.selection, args.out, args.artifact_root))


if __name__ == '__main__':
    main()
