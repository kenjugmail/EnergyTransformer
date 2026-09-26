"""Document-contained byte batches and deterministic public-corpus preparation."""
from pathlib import Path
import hashlib
import json
import os
import random
import shutil
import tempfile
import numpy as np
import torch
from energy_real.runtime import atomic_json, sha256
from .spec import DATA


def digest(raw):return hashlib.sha256(raw).hexdigest()


def prepare(records,out,limit=256*2**20,source=None):
    """Exact dedup plus deterministic MinHash/LSH near-duplicate grouping.

    LSH detection is approximate, never a semantic-independence guarantee.
    Documents are never cut to achieve the byte cap.
    """
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(tempfile.mkdtemp(prefix=out.name+'.',dir=out.parent))
    docs=[]; signatures=[]; seen=set(); buckets={}; parent=[]; total=0
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    try:
        spool=tmp/'spool'
        with spool.open('wb') as f:
            for ordinal,record in enumerate(records):
                text=record['text']; raw=text.encode('utf-8'); h=digest(raw)
                if len(raw)<1025 or h in seen:continue
                if total+len(raw)>limit:break
                words=text.lower().split(); shingles={digest(' '.join(words[i:i+5]).encode())[:16] for i in range(max(1,len(words)-4))}
                # Stable 32-coordinate minhash; four rows per band, eight bands.
                vals=[int(s,16) for s in shingles]; prime=2**61-1
                sig=tuple(min(((2*j+1)*v+104729*j)%prime for v in vals) for j in range(32))
                idx=len(docs);parent.append(idx); candidates=set()
                for band in range(8):candidates.update(buckets.get((band,sig[4*band:4*band+4]),[]))
                for other in candidates:
                    if sum(a==b for a,b in zip(sig,signatures[other]))/32 >= .8:parent[root(idx)]=root(other)
                for band in range(8):buckets.setdefault((band,sig[4*band:4*band+4]),[]).append(idx)
                signatures.append(sig);seen.add(h)
                docs.append({'id':record.get('id',h),'url':record.get('url'),'sha256':h,'offset':total,'bytes':len(raw),'ordinal':ordinal})
                f.write(raw);total+=len(raw)
        if not docs:raise ValueError('No suitable documents')
        names=('train','valid','test');handles={n:(tmp/(n+'.bin')).open('wb') for n in names};offsets={n:0 for n in names};index={n:[] for n in names}
        group_hash={}
        for i,d in enumerate(docs):group_hash[root(i)]=min(group_hash.get(root(i),d['sha256']),d['sha256'])
        with spool.open('rb') as f:
            for i,d in enumerate(docs):
                group=group_hash[root(i)];b=int(group[:16],16)%100;split='train' if b<90 else 'valid' if b<95 else 'test'
                f.seek(d['offset']);raw=f.read(d['bytes']);handles[split].write(raw)
                index[split].append({**d,'offset':offsets[split],'group':group});offsets[split]+=len(raw)
        for f in handles.values():f.flush();os.fsync(f.fileno());f.close()
        if any(not index[n] for n in names):raise ValueError('Empty split; more documents needed')
        spool.unlink()
        manifest={'format':'energy_documents_v1','source':source,'bytes':total,'requested_bytes':limit,
                  'split_policy':'group_sha_mod100_90_5_5','near_duplicates':'32-minhash_8x4_LSH_estimated_Jaccard>=0.8',
                  'documents':index,'shards':{n:sha256(tmp/(n+'.bin')) for n in names},
                  'tokenizer':{'kind':'utf8_bytes','vocab':256},'contamination':'Not audited against donor pretraining'}
        atomic_json(manifest,tmp/'manifest.json')
        # mkdir claims publication; a competing writer cannot overwrite it.
        out.mkdir()
        for p in tmp.iterdir():os.replace(p,out/p.name)
        tmp.rmdir()
        return manifest
    except BaseException:
        shutil.rmtree(tmp,ignore_errors=True);raise


def download(out):
    from datasets import load_dataset
    rows=load_dataset(DATA['id'],name=DATA['config'],revision=DATA['revision'],split='train',streaming=True)
    return prepare(rows,out,DATA['bytes'],{**DATA,'license':'ODC-By-1.0; CommonCrawl terms','order':'pinned dataset streaming order'})


class Documents:
    def __init__(self,path,split='train',expected_sha256=None):
        self.path=Path(path);self.split=split;self.manifest=json.loads((self.path/'manifest.json').read_text())
        self.docs=self.manifest['documents'][split];self.data=np.memmap(self.path/(split+'.bin'),dtype=np.uint8,mode='r')
        if sha256(self.path/(split+'.bin'))!=self.manifest['shards'][split]:raise ValueError('Shard changed')
        if expected_sha256 is not None and self.file_identity()!=expected_sha256:raise ValueError('Manifest changed')
        self.last_ranges=[];self.coverage={};self._windows={}
    @property
    def unique_targets(self):return sum(b-a for spans in self.coverage.values() for a,b in spans)
    def restore_coverage(self,value):self.coverage={} if value is None else value
    def file_identity(self):return sha256(self.path/'manifest.json')
    def batch(self,rng,batch,length,device=None):
        if length not in self._windows:
            eligible=[d for d in self.docs if d['bytes']>length]
            if not eligible:raise ValueError('No document long enough')
            cumulative=np.array([d['bytes']-length for d in eligible],dtype=np.int64).cumsum()
            self._windows[length]=(eligible,cumulative)
        eligible,cumulative=self._windows[length]
        positions=rng.integers(0,int(cumulative[-1]),size=batch);rows=[];self.last_ranges=[]
        for p in positions:
            i=int(np.searchsorted(cumulative,p,side='right'));d=eligible[i];start=int(p-(cumulative[i-1] if i else 0));off=d['offset']+start
            rows.append(np.asarray(self.data[off:off+length+1],dtype=np.int64));self.last_ranges.append((d['sha256'],start+1,start+length+1))
        for key,a,b in self.last_ranges:
            spans=sorted(self.coverage.get(key,[])+[(a,b)]);merged=[]
            for lo,hi in spans:
                if merged and lo<=merged[-1][1]:merged[-1]=(merged[-1][0],max(hi,merged[-1][1]))
                else:merged.append((lo,hi))
            self.coverage[key]=merged
        tok=torch.from_numpy(np.stack(rows)).to(device);return tok[:,:-1],tok[:,1:]
    def sequential_windows(self,length,limit=None,batch=1,device=None):
        used=0
        for d in self.docs:
            for start in range(0,d['bytes']-1,length):
                width=min(length,d['bytes']-1-start)
                if limit is not None:width=min(width,limit-used)
                if width<=0:return
                off=d['offset']+start;tok=torch.tensor(np.asarray(self.data[off:off+width+1],dtype=np.int64)[None],device=device)
                yield tok[:,:-1],tok[:,1:];used+=width
