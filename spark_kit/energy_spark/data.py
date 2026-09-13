"""Local UTF-8 JSONL -> document-isolated byte shards. No downloads or uploads.

Every record needs text, document_id, source and license. Declarations do not
replace a rights review. Exact duplicate contents are grouped; near-duplicate passages and semantic
overlap require an additional corpus-specific audit. Raw text never enters receipts.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, sqlite3, tempfile
from pathlib import Path
import numpy as np
import torch
from .common import atomic_json, sha256, json_hash

SPLITS=('train','validation','test')

def prepare(path:Path,out:Path,*,seed=22001,min_bytes=1025):
    """Spool raw documents to disk; keep only per-document metadata in memory.

    Memory is O(largest input record + metadata), not O(total corpus text).
    Output bytes and stable split ordering match the v22 importer. Exact
    duplicate content keeps the first declared provenance, as before.
    """
    path,out=Path(path),Path(out)
    if type(min_bytes) is not int or min_bytes<2:raise ValueError('min_bytes must be an integer >= 2')
    if type(seed) is not int or seed<0:raise ValueError('seed must be a nonnegative integer')
    if out.exists():raise FileExistsError('Choose a new dataset directory; existing datasets are never overwritten')
    out.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix=out.name+'.',dir=out.parent))
    try:
        db_path=stage/'_prepare.sqlite';spool_path=stage/'_prepare.bin'
        with sqlite3.connect(db_path) as db, spool_path.open('w+b') as spool:
            db.execute('CREATE TABLE ids (id TEXT PRIMARY KEY)')
            db.execute('CREATE TABLE docs (digest TEXT PRIMARY KEY, sortkey TEXT, offset INTEGER, length INTEGER, id_hash TEXT, source TEXT, license TEXT)')
            source_hash=hashlib.sha256()
            with path.open('rb') as source:
                for line_no,line in enumerate(source,1):
                    source_hash.update(line)
                    if not line.strip():continue
                    r=json.loads(line.decode('utf-8'))
                    if not isinstance(r,dict) or any(not isinstance(r.get(k),str) or not r[k].strip() for k in ('text','document_id','source','license')):
                        raise ValueError(f'Line {line_no}: nonempty text, document_id, source, license are required')
                    try:db.execute('INSERT INTO ids VALUES (?)',(r['document_id'],))
                    except sqlite3.IntegrityError as exc:raise ValueError('Repeated document_id') from exc
                    raw=r['text'].encode('utf-8');h=hashlib.sha256(raw).hexdigest()
                    if len(raw)<min_bytes:continue
                    if db.execute('SELECT 1 FROM docs WHERE digest=?',(h,)).fetchone():continue
                    offset=spool.tell();spool.write(raw)
                    db.execute('INSERT INTO docs VALUES (?,?,?,?,?,?,?)',(h,hashlib.sha256(f'{seed}:{h}'.encode()).hexdigest(),offset,len(raw),hashlib.sha256(r['document_id'].encode()).hexdigest(),r['source'],r['license']))
            spool.flush();n=db.execute('SELECT COUNT(*) FROM docs').fetchone()[0]
            if n<10:raise ValueError('At least 10 unique sufficiently long documents are required (20+ recommended)')
            nv=max(1,int(.1*n));ntrain=n-2*nv;parts={};metadata=[];row_index=0
            cursor=iter(db.execute('SELECT digest,offset,length,id_hash,source,license FROM docs ORDER BY sortkey,digest'))
            for split,count in zip(SPLITS,(ntrain,nv,nv)):
                offsets=[0]
                with (stage/f'{split}.bin').open('wb') as target:
                    for _ in range(count):
                        h,offset,length,id_hash,origin,license_decl=next(cursor)
                        spool.seek(offset);remaining=length
                        while remaining:
                            chunk=spool.read(min(1024**2,remaining))
                            if not chunk:raise IOError('Truncated preparation spool')
                            target.write(chunk);remaining-=len(chunk)
                        offsets.append(offsets[-1]+length)
                        metadata.append({'id_sha256':id_hash,'source':origin,'license':license_decl,'content_sha256':h,'bytes':length,'split':split})
                        row_index+=1
                np.save(stage/f'{split}_offsets.npy',np.array(offsets,dtype=np.int64),allow_pickle=False)
                parts[split]={'documents':count,'bytes':offsets[-1],'file':f'{split}.bin','sha256':sha256(stage/f'{split}.bin'),'offsets':f'{split}_offsets.npy','offsets_sha256':sha256(stage/f'{split}_offsets.npy')}
            assert row_index==n
        db_path.unlink();spool_path.unlink()
        manifest={'format':'energy_byte_corpus_v22','seed':seed,'source_sha256':source_hash.hexdigest(),'min_bytes':min_bytes,
                  'vocabulary':256,'encoding':'UTF-8 bytes','grouping':'exact content before splitting; stable hash order',
                  'rights':'supplied declarations, not independent legal clearance','partitions':parts,'documents':metadata,
                  'limitations':'No semantic contamination guarantee; small/biased corpora do not justify scaling.'}
        manifest['identity']=json_hash(manifest)
        atomic_json(manifest,stage/'manifest.json')
        # Refuse a destination created concurrently. os.rename does not merge files.
        if out.exists():raise FileExistsError('Destination appeared during preparation')
        os.rename(stage,out)
        return manifest
    finally:
        if stage.exists():shutil.rmtree(stage)

class ByteCorpus:
    def __init__(self,root:Path,split:str):
        self.root=Path(root);self.manifest=json.loads((self.root/'manifest.json').read_text())
        if self.manifest.get('format')!='energy_byte_corpus_v22' or split not in SPLITS:raise ValueError('Invalid corpus format/split')
        supplied=self.manifest['identity'];m={k:v for k,v in self.manifest.items() if k!='identity'}
        if json_hash(m)!=supplied:raise ValueError('Manifest integrity mismatch')
        self.identity=supplied;part=self.manifest['partitions'][split]
        for name,check in ((part['file'],part['sha256']),(part['offsets'],part['offsets_sha256'])):
            if Path(name).name!=name or (self.root/name).is_symlink() or sha256(self.root/name)!=check:raise ValueError('Unsafe path or shard checksum mismatch')
        self.data=np.memmap(self.root/part['file'],dtype=np.uint8,mode='r')
        self.offsets=np.load(self.root/part['offsets'],allow_pickle=False)
        if self.offsets.dtype!=np.int64 or self.offsets.ndim!=1 or len(self.offsets)<2 or self.offsets[0]!=0 or self.offsets[-1]!=len(self.data) or np.any(np.diff(self.offsets)<2):raise ValueError('Invalid document offsets')
        self.lengths=np.diff(self.offsets)
    def batch(self,seed:int,step:int,micro:int,batch:int,length:int):
        if any(type(v) is not int for v in (seed,step,micro,batch,length)) or min(seed,step,micro)<0 or min(batch,length)<1:raise ValueError('Invalid sampling arguments')
        legal=self.lengths-length
        valid=np.flatnonzero(legal>0)
        if len(valid)==0:raise ValueError('No document is long enough for this context')
        rng=np.random.default_rng(np.random.SeedSequence([seed,step,micro,22022]))
        probs=legal[valid].astype(float);probs/=probs.sum();items=[]
        for i in rng.choice(valid,size=batch,p=probs):
            start=int(self.offsets[i])+int(rng.integers(0,legal[i]))
            items.append(np.array(self.data[start:start+length+1],dtype=np.int64))
        tokens=torch.from_numpy(np.stack(items));return tokens[:,:-1],tokens[:,1:]
    def validation_windows(self,length=128,max_targets=32768):
        count=0
        for left,right in zip(self.offsets[:-1],self.offsets[1:]):
            for start in range(int(left),int(right)-1,length):
                n=min(length,int(right)-1-start,max_targets-count)
                if n<1:return
                a=torch.from_numpy(np.array(self.data[start:start+n+1],dtype=np.int64))[None]
                yield a[:,:-1],a[:,1:]
                count+=n
                if count>=max_targets:return

def fixtures(out:Path):
    """Deterministic synthetic executable-test data, never a language benchmark."""
    path=out/'fixture.jsonl';out.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        with path.open('w') as f:
            for i in range(24):
                text=(f'Record {i:02d}. Red follows blue; green follows red. '+''.join(chr(65+(j+i)%26) for j in range(100))+'\n')*16
                f.write(json.dumps({'text':text,'document_id':f'fixture-{i}','source':'generated code fixture','license':'generated for this research test'})+'\n')
    dest=out/'corpus'
    if not dest.exists():prepare(path,dest,min_bytes=512)
    return dest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--jsonl',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--min-bytes',type=int,default=1025)
    a=p.parse_args();r=prepare(a.jsonl,a.out,min_bytes=a.min_bytes);print(json.dumps({'identity':r['identity'],'partitions':r['partitions']},indent=2))
