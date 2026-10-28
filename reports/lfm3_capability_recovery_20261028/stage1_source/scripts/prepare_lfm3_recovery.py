"""Prepare a source-balanced, assistant-masked recovery corpus from pinned data."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from scripts.prepare_rewt_lfm3 import TEACHER_REVISION, DATA_REPO, DATA_REVISION, sha, save_json


def main():
    out = ROOT/'data/lfm3_capability_recovery_v1'
    if (out/'manifest.json').exists():
        raise FileExistsError(out)
    out.mkdir(exist_ok=True)
    old = ROOT/'data/lfm3_conversion_corpus'
    teacher = ROOT/'data/lfm3_1_7b_teacher'/TEACHER_REVISION
    tokenizer = AutoTokenizer.from_pretrained(teacher, local_files_only=True, trust_remote_code=True)
    old_ids = {s:{r['id'] for r in json.loads((old/f'{s}_documents.json').read_text())} for s in ('train','valid','test')}
    quotas = {'smol-magpie-ultra-short':6000, 'self-oss-instruct':2500, 'openhermes-50k':2500,
              'smol-contraints':2500, 'smollm-rewrite-30k':1000, 'smol-summarize-20k':750,
              'smol-summarize-5k':250, 'everyday-conversations':300, 'longalign':300,
              'explore-instruct-rewrite':300}
    buckets = defaultdict(list)
    source = old/'source/data/train-00000-of-00004.parquet'
    seen = set()
    for batch in pq.ParquetFile(source).iter_batches(batch_size=512):
        for row in batch.to_pylist():
            messages = row['messages']
            if not messages or any(m['role'] not in ('system','user','assistant') or not isinstance(m['content'],str) for m in messages):
                continue
            identity = hashlib.sha256(json.dumps(messages,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
            if identity in seen or identity in old_ids['test'] or identity in old_ids['valid']:
                continue
            seen.add(identity)
            # Keep the original validation bucket out of training. New validation
            # also cannot contain any previously trained conversation.
            split = 'valid' if int(identity[:8],16)%32 == 0 else 'train'
            if split == 'valid' and identity in old_ids['train']:
                continue
            buckets[split,row['source']].append((identity,messages))
    counts = Counter(); records = {}; used = set()
    for split in ('train','valid'):
        tokens=[]; masks=[]; offsets=[0]; index=[]
        for name, quota in quotas.items():
            limit = quota if split=='train' else max(8,round(quota/64))
            accepted=0
            for identity,messages in sorted(buckets[split,name]):
                encoded=tokenizer.apply_chat_template(messages,tokenize=True,enable_thinking=False,
                    add_generation_prompt=False,return_dict=True,return_assistant_tokens_mask=True)
                ids=encoded['input_ids'][:2049]; mask=encoded['assistant_masks'][:2049]
                if len(ids)<16 or not any(mask[1:]):
                    continue
                assert identity not in used
                used.add(identity); tokens.extend(ids); masks.extend(mask); offsets.append(len(tokens))
                index.append(dict(id=identity,source=name,tokens=len(ids),assistant_tokens=sum(mask)))
                accepted+=1; counts[f'{split}/{name}']+=1
                if accepted>=limit:break
        paths={}
        for suffix, values, dtype in [('tokens',tokens,np.uint32),('assistant',masks,np.uint8),('offsets',offsets,np.int64)]:
            path=out/f'{split}_{suffix}.npy'; np.save(path,np.asarray(values,dtype=dtype))
            paths[suffix]=dict(sha256=sha(path),path=str(path))
        save_json(out/f'{split}_documents.json',index)
        records[split]=dict(documents=len(index),tokens=len(tokens),assistant_tokens=sum(masks),files=paths,
                            short_under_513=sum(r['tokens']<513 for r in index))
        print(json.dumps(dict(split=split,**records[split])),flush=True)
    save_json(out/'manifest.json',dict(dataset=DATA_REPO,revision=DATA_REVISION,source_sha256=sha(source),
        tokenizer_revision=TEACHER_REVISION,tokenizer_sha256=sha(teacher/'tokenizer.json'),
        template_sha256=sha(teacher/'chat_template.jinja'),splits=records,source_counts=dict(counts),
        selection='Hash-sorted per-source quotas from pinned official training shard; original SHA256 validation bucket excluded from training. Old validation/test IDs excluded. New validation excludes all old training IDs.',
        evaluation='No capability evaluation prompts or official test rows used in this recovery corpus.',
        limits='Exact document deduplication only; no guarantee against semantic overlap or donor pretraining contamination.'))

if __name__=='__main__':main()
