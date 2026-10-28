"""Cache a pinned, capped public TinyStories subset; keep document splits disjoint."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT / 'data/tinystories_rewt_20261010'
REVISION = 'f54c09fd23315a6f9c86f9dc80f725de7d8f9c64'
BASE = f'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/{REVISION}/'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fetch(filename, cap):
    target=DATA/filename
    if target.exists():
        raw=target.read_bytes()
    else:
        request=urllib.request.Request(BASE+filename,headers={'Range':f'bytes=0-{cap-1}'})
        chunks=[]; remaining=cap
        with urllib.request.urlopen(request,timeout=30) as response:
            while remaining:
                block=response.read(min(1<<20,remaining))
                if not block: break
                chunks.append(block);remaining-=len(block)
        raw=b''.join(chunks); target.write_bytes(raw)
    assert len(raw)<=cap
    return raw,dict(url=BASE+filename,cached_path=str(target),cached_bytes=len(raw),
                    cap_bytes=cap,cached_sha256=sha(raw),scope='Capped leading range, not the entire remote file')


def stories(raw):
    # Discard the final potentially partial UTF-8/story fragment.
    text=raw.decode('utf-8',errors='ignore')
    result=[]
    for story in text.split('<|endoftext|>')[:-1]:
        normalized=story.strip().replace('\r','').translate(str.maketrans(
            {'“':'"','”':'"','‘':"'",'’':"'",'—':'-','–':'-','\u00a0':' '}))
        encoded=normalized.encode('ascii',errors='replace')
        if len(encoded)>=161:
            result.append(encoded)
    return result


def write_split(split, values):
    raw=bytearray(); records=[]
    for story in values:
        start=len(raw);raw.extend(story);end=len(raw);raw.extend(b'\n\n')
        records.append(dict(id=sha(story),start=start,end=end,bytes=len(story)))
    path=DATA/f'{split}.bin';path.write_bytes(raw)
    index=DATA/f'{split}_documents.json';index.write_text(json.dumps(records,indent=2)+'\n')
    return dict(path=str(path),index_path=str(index),sha256=sha(raw),index_sha256=sha(index.read_bytes()),
                bytes=len(raw),documents=len(records),document_ids=[r['id'] for r in records])


def main():
    DATA.mkdir(parents=True,exist_ok=True)
    train_raw,train_source=fetch('TinyStories-train.txt',32<<20)
    valid_raw,valid_source=fetch('TinyStories-valid.txt',8<<20)
    seen=set(); training=[]
    for story in stories(train_raw):
        identity=sha(story)
        if identity not in seen:seen.add(identity);training.append(story)
    validation=[];test=[];holdout_ids=set();duplicates=0
    for story in stories(valid_raw):
        identity=sha(story)
        if identity in seen or identity in holdout_ids:
            duplicates+=1;continue
        holdout_ids.add(identity)
        (validation if int(identity[:12],16)%2==0 else test).append(story)
    assert seen.isdisjoint(holdout_ids) and len(validation)>100 and len(test)>100
    card=urllib.request.urlopen(BASE+'README.md',timeout=20).read()
    (DATA/'UPSTREAM_README.md').write_bytes(card)
    manifest=dict(dataset='roneneldan/TinyStories',revision=REVISION,
        dataset_card='https://huggingface.co/datasets/roneneldan/TinyStories',
        paper='https://arxiv.org/abs/2305.07759',license='cdla-sharing-1.0',
        sources=[train_source,valid_source],splits={name:write_split(name,values) for name,values in
                [('train',training),('valid',validation),('test',test)]},
        holdout_duplicates_removed=duplicates,
        preprocessing='Complete stories only; normalize typographic punctuation, ASCII replacement for other non-ASCII, require >=161 bytes, deduplicate normalized story SHA256.',
        split_rule='Official training file stays train; official validation stories split by normalized story SHA256 parity into tuning validation and held-out test. Cross-split duplicates removed.',
        model_initialization='No TinyStories model weights are downloaded.')
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    (OUT/'story_data.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps({name:{key:value for key,value in split.items() if key!='document_ids'}
                      for name,split in manifest['splits'].items()},indent=2))


if __name__=='__main__':main()
