"""Pin the LFM3 donor, prepare disjoint chat data, and write an initialized student."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TEACHER_REPO = 'LiquidAI/LFM3-1.7B'
TEACHER_REVISION = '40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd'
DATA_REPO = 'HuggingFaceTB/smol-smoltalk'
DATA_REVISION = 'f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def fetch(teacher, corpus):
    from huggingface_hub import hf_hub_download
    for repo, revision, out, repo_type, files in [
        (TEACHER_REPO, TEACHER_REVISION, teacher, 'model',
         ['LICENSE', 'README.md', 'config.json', 'configuration_lfm3.py', 'modeling_lfm3.py',
          'lfm3_conv.py', 'generation_config.json', 'tokenizer_config.json', 'special_tokens_map.json',
          'chat_template.jinja', 'tokenizer.json', 'model.safetensors']),
        (DATA_REPO, DATA_REVISION, corpus/'source', 'dataset',
         ['README.md', 'data/train-00000-of-00004.parquet', 'data/test-00000-of-00001.parquet'])]:
        records = {}
        for name in files:
            path = Path(hf_hub_download(repo, name, revision=revision, repo_type=repo_type, local_dir=out))
            records[name] = dict(bytes=path.stat().st_size, sha256=sha(path))
        save_json(out/('snapshot_metadata.json' if repo_type == 'model' else 'manifest.json'),
                  dict(repo=repo, revision=revision, files=records))


def prepare_corpus(args):
    import numpy as np
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    out = args.corpus
    if (out/'manifest.json').exists():
        raise FileExistsError('prepared corpus already exists')
    # The pinned local configuration source is inspected alongside the donor.
    tokenizer = AutoTokenizer.from_pretrained(args.teacher, local_files_only=True, trust_remote_code=True)
    groups = {split: [] for split in ('train', 'valid', 'test')}; used = set()
    limits = dict(train=args.train_docs, valid=args.valid_docs, test=args.test_docs)
    sources = Counter(); skipped = Counter()
    for official_split, filename in [('train', 'train-00000-of-00004.parquet'), ('test', 'test-00000-of-00001.parquet')]:
        source = pq.ParquetFile(out/'source/data'/filename)
        for batch in source.iter_batches(batch_size=256):
            for row in batch.to_pylist():
                messages = row['messages']
                if not messages or any(m['role'] not in ('system', 'user', 'assistant') or not isinstance(m['content'], str) for m in messages):
                    skipped['unsupported_message'] += 1; continue
                identity = hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                split = ('valid' if int(identity[:8], 16) % 32 == 0 else 'train') if official_split == 'train' else 'test'
                if len(groups[split]) >= limits[split]:
                    continue
                if identity in used:
                    skipped['duplicate'] += 1; continue
                ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                                                    enable_thinking=False, return_dict=False)
                if len(ids) < args.min_length:
                    skipped['short'] += 1; continue
                ids = ids[:args.max_length]
                if any(i < 0 or i >= 128000 for i in ids):
                    raise ValueError('tokenizer vocabulary mismatch')
                used.add(identity)
                groups[split].append(dict(id=identity, source=row['source'], ids=ids))
                sources[f'{split}/{row["source"]}'] += 1
            wanted = ('train', 'valid') if official_split == 'train' else ('test',)
            if all(len(groups[s]) >= limits[s] for s in wanted):
                break
    if any(len(groups[s]) != limits[s] for s in groups):
        raise ValueError(f'insufficient source documents: { {s:len(v) for s,v in groups.items()} }')
    records = {}
    for split, documents in groups.items():
        offsets = np.cumsum([0]+[len(row['ids']) for row in documents], dtype=np.int64)
        tokens = np.asarray([token for row in documents for token in row['ids']], dtype=np.uint32)
        token_path = out/f'{split}_tokens.npy'; offset_path = out/f'{split}_offsets.npy'
        np.save(token_path, tokens); np.save(offset_path, offsets)
        index_path = out/f'{split}_documents.json'
        save_json(index_path, [dict(id=row['id'], source=row['source'], tokens=len(row['ids'])) for row in documents])
        records[split] = dict(documents=len(documents), tokens=len(tokens),
                              tokens_path=str(token_path.resolve()), tokens_sha256=sha(token_path),
                              offsets_path=str(offset_path.resolve()), offsets_sha256=sha(offset_path),
                              index_path=str(index_path.resolve()), index_sha256=sha(index_path))
    save_json(out/'manifest.json', dict(dataset=DATA_REPO, revision=DATA_REVISION, splits=records,
              tokenizer_repo=TEACHER_REPO, tokenizer_revision=TEACHER_REVISION,
              tokenizer_sha256=sha(args.teacher/'tokenizer.json'), template_sha256=sha(args.teacher/'chat_template.jinja'),
              sources=dict(sources), skipped=dict(skipped), max_document_tokens=args.max_length,
              selection='First eligible unique conversations in the first pinned official train shard; SHA256 bucket 0 mod 32 is validation. Test is a separate official test shard, exact conversation duplicates removed. Documents never span examples during training.',
              limits='General English chat recovery pilot. Not a comprehensive knowledge/tool-use/code or long-context recovery corpus. No claim of semantic deduplication or exclusion from donor pretraining.'))
    print(json.dumps(records, indent=2), flush=True)


def initialize(args):
    import torch
    from safetensors import safe_open
    from rewt.lfm_conversion import ConversionConfig, ConvertedLM
    out = args.output
    if out.exists():
        raise FileExistsError('use a new student output directory')
    torch.manual_seed(args.seed); torch.set_num_threads(4)
    config = ConversionConfig.from_teacher(json.loads((args.teacher/'config.json').read_text()),
                                           modes=args.modes, window=args.window)
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        model = ConvertedLM(config)
    finally:
        torch.set_default_dtype(original_dtype)
    with safe_open(args.teacher/'model.safetensors', framework='pt', device='cpu') as donor:
        mapping = model.transfer(donor)
    out.mkdir(parents=True)
    metadata = dict(teacher_repo=TEACHER_REPO, teacher_revision=TEACHER_REVISION,
                    teacher_sha256=sha(args.teacher/'model.safetensors'), seed=args.seed,
                    training_steps=0, stage='initialized_only',
                    parameters=sum(p.numel() for p in model.parameters()), config=asdict(config))
    model.save(out/'initialized.pt', metadata=metadata)
    save_json(out/'transfer.json', mapping); save_json(out/'metadata.json', metadata)
    shutil.copy2(args.teacher/'LICENSE', out/'TEACHER_LICENSE.txt')
    for name in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'chat_template.jinja'):
        shutil.copy2(args.teacher/name, out/name)
    print(json.dumps(metadata, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher', type=Path, default=ROOT/'data/lfm3_1_7b_teacher'/TEACHER_REVISION)
    parser.add_argument('--corpus', type=Path, default=ROOT/'data/lfm3_conversion_corpus')
    parser.add_argument('--output', type=Path, default=ROOT/'runs/lfm3_conversion_20261028/initial')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--prepare-data', action='store_true')
    parser.add_argument('--initialize', action='store_true')
    parser.add_argument('--train-docs', type=int, default=8192)
    parser.add_argument('--valid-docs', type=int, default=128)
    parser.add_argument('--test-docs', type=int, default=128)
    parser.add_argument('--min-length', type=int, default=257)
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--window', type=int, default=256)
    parser.add_argument('--modes', type=int, default=64)
    parser.add_argument('--seed', type=int, default=281701)
    args = parser.parse_args()
    if args.min_length < 2 or args.max_length < args.min_length or min(args.train_docs, args.valid_docs, args.test_docs) < 1:
        parser.error('invalid data sizes')
    if args.download:
        fetch(args.teacher, args.corpus)
    if args.prepare_data:
        prepare_corpus(args)
    if args.initialize:
        initialize(args)


if __name__ == '__main__':
    main()
