"""Encode the already pinned, document-disjoint stories with a training vocabulary."""
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from rewt.data import file_sha256
from rewt.experiment import save_json
from rewt.story_data import StoryCorpus
from rewt.bpe import StoryBPE

SOURCE=ROOT/'data/tinystories_rewt_20261010'
OUT=ROOT/'data/tinystories_bpe_20261010'


def main():
    start=time.process_time();OUT.mkdir(parents=True,exist_ok=True)
    corpus=StoryCorpus(SOURCE)
    tokenizer=StoryBPE.learn(bytes(corpus.data['train']))
    vocabulary=OUT/'vocabulary.json'
    tokenizer.save(vocabulary,source=corpus.identity())
    manifest=dict(dataset=corpus.manifest['dataset'],revision=corpus.manifest['revision'],
        parent_data=corpus.identity(),tokenizer=dict(path=str(vocabulary),sha256=file_sha256(vocabulary),
            identity=tokenizer.identity(),vocabulary_size=256,learned_from='train only'),splits={})
    for split in ('train','valid','test'):
        original=json.loads(Path(corpus.manifest['splits'][split]['index_path']).read_text())
        path=OUT/f'{split}.bin';index=OUT/f'{split}_documents.json';rows=[];offset=0
        with path.open('wb') as stream:
            for doc,(a,b) in zip(original,corpus.documents[split]):
                raw=bytes(corpus.data[split][a:b]);encoded=tokenizer.encode(raw)
                assert tokenizer.decode(encoded)==raw
                stream.write(encoded)
                rows.append(dict(id=doc['id'],start=offset,end=offset+len(encoded),raw_bytes=len(raw)))
                offset+=len(encoded)
        save_json(index,rows)
        manifest['splits'][split]=dict(path=str(path),index_path=str(index),sha256=file_sha256(path),
            index_sha256=file_sha256(index),bytes=path.stat().st_size,documents=len(rows),
            raw_bytes=sum(r['raw_bytes'] for r in rows),document_ids=[r['id'] for r in rows])
        print(json.dumps(dict(split=split,tokens=offset,raw_bytes=manifest['splits'][split]['raw_bytes'])),flush=True)
    manifest['preparation_cpu_seconds']=time.process_time()-start
    save_json(OUT/'manifest.json',manifest)
    StoryCorpus(OUT)


if __name__=='__main__':main()
