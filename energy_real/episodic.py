"""Experimental relation memory. Separate from the legacy 16-name protected ledger.

Models only propose changes. apply() requires explicit caller authorization,
revision matching and retry identity. Evidence returned by approximate retrieval
is never corrected using the exact store.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import torch
from energy_transformer.geometric_memory import EditMemory


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class FactEdit:
    operation: str
    subject: str
    relation: str
    object: str | None
    source_text: str

    def validate(self):
        if self.operation not in ('set','delete','hold'): raise ValueError('Unsupported operation')
        for x in (self.subject,self.relation):
            if not isinstance(x,str) or not x.strip() or len(x)>128: raise ValueError('Invalid subject or relation')
        if self.operation=='set' and (not isinstance(self.object,str) or not self.object.strip() or len(self.object)>128):
            raise ValueError('SET requires an object')
        if self.object is not None and not isinstance(self.object,str):raise ValueError('Object must be text or null')
        if self.operation=='delete' and self.object is not None: raise ValueError('DELETE must have no object')
        if not isinstance(self.source_text,str) or not self.source_text or len(self.source_text)>16384:
            raise ValueError('Source text required')

    @property
    def source_id(self): return digest(self.source_text)


@dataclass(frozen=True)
class Query:
    subject: str
    relations: tuple[str, ...]

    def validate(self):
        if not isinstance(self.subject,str) or not self.subject or len(self.subject)>128: raise ValueError('Invalid query subject')
        if not isinstance(self.relations,(list,tuple)) or not 1<=len(self.relations)<=3 or any(not isinstance(x,str) or not x or len(x)>128 for x in self.relations):
            raise ValueError('Query requires one to three relations')


def parse_query(question, service=None):
    if isinstance(question,Query): q=question
    elif isinstance(question,dict):
        if set(question)!={'subject','relations'} or not isinstance(question['relations'],(list,tuple)):
            raise ValueError('Query must have a subject and a relation list')
        q=Query(question['subject'],tuple(question['relations']))
    elif isinstance(question,str):
        match=re.fullmatch(r'Follow ([\w/]+) from ([\w-]+)\.?',question.strip())
        if match:q=Query(match[2],tuple(match[1].split('/')))
        elif service:
            obj=service.structured('Interpret a fact query as {"subject":string,"relations":[string]}. Use at most three relations.',question)
            return parse_query(obj)
        else:raise ValueError('Use Query, a structured query, or "Follow relation/relation from subject"')
    else:raise TypeError('Invalid query')
    q.validate();return q


def key_vector(subject,relation,dimension=16):
    # Explicitly near-colliding keys are a diagnostic input family, not a claim
    # about natural-language embedding similarity.
    base,sep,suffix=subject.partition('_near_')
    seed=int(digest(base+'\0'+relation)[:15],16)
    vector=torch.randn(dimension,generator=torch.Generator().manual_seed(seed))
    if sep:
        vector=vector+.04*torch.randn(dimension,generator=torch.Generator().manual_seed(int(digest(suffix)[:15],16)))
    return torch.nn.functional.normalize(vector,dim=0)


class EnergyRetriever:
    def __init__(self, events, *, kind='precision', dimension=16, capacity=256, controller=None,
                 prior=.01, score_threshold=.5, margin_threshold=0.):
        if len(events)>capacity: raise ValueError('Episode exceeds declared record capacity')
        if any(not math.isfinite(x) for x in (prior,score_threshold,margin_threshold)) or prior<=0 or margin_threshold<0:
            raise ValueError('Invalid precision prior or confidence thresholds')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(11)
            self.model=EditMemory(kind,dimension,capacity,prior=prior).eval()
        if controller:
            saved=torch.load(controller,map_location='cpu',weights_only=True)
            if (saved['kind']!=kind or saved['dimension']!=dimension or saved['value_dim']!=capacity
                    or saved.get('prior',.01)!=prior):
                raise ValueError('Controller configuration mismatch (kind, dimension, value dimension, or prior)')
            self.model.load_state_dict(saved['weights'],strict=True)
        self.score_threshold=score_threshold;self.margin_threshold=margin_threshold
        self.last_diagnostic=None
        self.state=self.model.initial(1)
        self.records={}
        with torch.no_grad():
            for index,event in enumerate(events):
                value=torch.zeros(1,capacity)
                if event['operation']=='set':
                    value[0,index]=1.
                    self.records[index]=event
                op={'set':0,'delete':1,'hold':2}[event['operation']]
                key=key_vector(event['subject'],event['relation'],dimension)[None]
                self.state=self.model.step(key,value,torch.tensor([op]),self.state)
        self.dimension=dimension

    @torch.no_grad()
    def read(self,subject,relation):
        key=key_vector(subject,relation,self.dimension)[None,None]
        scores=self.model.read(key,self.state)[0,0]
        finite=bool(torch.isfinite(scores).all())
        candidates=sorted(self.records,key=lambda i:float(scores[i]),reverse=True)[:2] if finite else []
        index=candidates[0] if candidates else None
        score=float(scores[index]) if index is not None else None
        runner_up=float(scores[candidates[1]]) if len(candidates)>1 else 0.
        margin=score-runner_up if score is not None else None
        accepted=score is not None and score>self.score_threshold and margin>=self.margin_threshold
        self.last_diagnostic={'subject':subject,'relation':relation,'score':score,'margin':margin,
                              'score_threshold':self.score_threshold,'margin_threshold':self.margin_threshold,
                              'accepted':accepted,'finite':finite,
                              'candidate_fact_ids':[self.records[i].get('fact_id') for i in candidates],
                              'selected_fact_id':self.records[index].get('fact_id') if accepted else None}
        if not accepted:
            return None
        # Do not check the selected record against the current exact state.
        return {**self.records[index], 'retrieval_score':float(scores[index])}

    @property
    def state_bytes(self):
        return sum(t.numel()*t.element_size() for t in (self.state.precision,self.state.information))


class FactStore:
    def __init__(self,path,*,service=None,backend='exact',controller=None,dimension=16,capacity=256,
                 prior=.01,score_threshold=.5,margin_threshold=0.):
        if backend not in ('exact','diagonal','precision','delta','preconditioned_delta','projected'):
            raise ValueError('Unknown memory backend')
        self.path=Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.service=service;self.backend=backend;self.controller=controller
        self.dimension=dimension;self.capacity=capacity
        self.prior=prior;self.score_threshold=score_threshold;self.margin_threshold=margin_threshold
        self._cached=None
        with self._connect() as db:
            db.executescript('''CREATE TABLE IF NOT EXISTS meta(id INTEGER PRIMARY KEY CHECK(id=1),revision INTEGER);
                INSERT OR IGNORE INTO meta VALUES(1,0);
                CREATE TABLE IF NOT EXISTS requests(id TEXT PRIMARY KEY,payload TEXT,receipt TEXT);
                CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY,event TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS facts(subject TEXT,relation TEXT,event TEXT,PRIMARY KEY(subject,relation));''')

    @contextmanager
    def _connect(self):
        db=sqlite3.connect(self.path,timeout=10)
        try:
            db.execute('PRAGMA synchronous=FULL')
            with db:
                yield db
        finally:
            db.close()

    @property
    def revision(self):
        with self._connect() as db:return db.execute('SELECT revision FROM meta WHERE id=1').fetchone()[0]

    def propose(self,text):
        if not isinstance(text,str) or not text or len(text)>16384:raise ValueError('Nonempty bounded text required')
        if self.service is None:raise ValueError('Text proposals require a configured model service')
        obj=self.service.structured(
            'Propose factual edits as {"edits":[{"operation":"set|delete|hold","subject":string,"relation":string,"object":string|null}]}. '
            'Quoted, hypothetical, negated commands are HOLD. DELETE has null object. Never authorize or apply edits.',text)
        if set(obj)!={'edits'} or not isinstance(obj['edits'],list) or len(obj['edits'])>32:raise ValueError('Malformed proposal')
        edits=[]
        for row in obj['edits']:
            if not isinstance(row,dict) or set(row)!={'operation','subject','relation','object'}:raise ValueError('Unexpected proposal fields')
            edit=FactEdit(**row,source_text=text);edit.validate();edits.append(asdict(edit))
        return {'status':'needs_review','expected_revision':self.revision,'source_id':digest(text),'edits':edits}

    def apply(self,reviewed_edits,*,expected_revision,request_id,authorized=False):
        if authorized is not True:raise PermissionError('Caller must explicitly authorize reviewed edits')
        if type(expected_revision) is not int or expected_revision<0:raise ValueError('Invalid revision')
        if not isinstance(request_id,str) or not re.fullmatch(r'[\w.:-]{1,128}',request_id):raise ValueError('Invalid request ID')
        if not isinstance(reviewed_edits,list) or not 1<=len(reviewed_edits)<=64:raise ValueError('Supply 1 to 64 reviewed edits')
        for edit in reviewed_edits:
            if not isinstance(edit,FactEdit):raise TypeError('Expected reviewed FactEdit objects')
            edit.validate()
        payload=canonical({'revision':expected_revision,'edits':[asdict(e) for e in reviewed_edits]})
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            old=db.execute('SELECT payload,receipt FROM requests WHERE id=?',(request_id,)).fetchone()
            if old:
                if old[0]!=payload:raise ValueError('Request ID reused for different content')
                return {**json.loads(old[1]),'replayed':True}
            revision=db.execute('SELECT revision FROM meta WHERE id=1').fetchone()[0]
            if revision!=expected_revision:raise ValueError('Stale revision; review again')
            revision+=1
            next_id=db.execute('SELECT COALESCE(MAX(id),0)+1 FROM events').fetchone()[0]
            for offset,edit in enumerate(reviewed_edits):
                event={**asdict(edit),'source_id':edit.source_id,'revision':revision,
                       'fact_id':digest(f'{revision}:{offset}:{canonical(asdict(edit))}')}
                encoded=canonical(event)
                db.execute('INSERT INTO events VALUES(?,?)',(next_id+offset,encoded))
                if edit.operation=='set':
                    db.execute('INSERT OR REPLACE INTO facts VALUES(?,?,?)',(edit.subject,edit.relation,encoded))
                elif edit.operation=='delete':
                    db.execute('DELETE FROM facts WHERE subject=? AND relation=?',(edit.subject,edit.relation))
            db.execute('UPDATE meta SET revision=? WHERE id=1',(revision,))
            receipt={'revision':revision,'replayed':False,'request_id':request_id}
            db.execute('INSERT INTO requests VALUES(?,?,?)',(request_id,payload,canonical(receipt)))
        self._cached=None
        return receipt

    def snapshot(self):
        with self._connect() as db:
            db.execute('BEGIN')
            revision=db.execute('SELECT revision FROM meta WHERE id=1').fetchone()[0]
            facts=[json.loads(row[0]) for row in db.execute('SELECT event FROM facts ORDER BY subject,relation')]
            events=[json.loads(row[0]) for row in db.execute('SELECT event FROM events ORDER BY id')]
        return revision,facts,events

    def query(self,question):
        q=parse_query(question,self.service)
        revision,facts,events=self.snapshot()
        exact={(f['subject'],f['relation']):f for f in facts}
        if self.backend=='exact':read=lambda subject,relation:exact.get((subject,relation))
        else:
            cached=self._cached
            if cached is None or cached[0]!=revision:
                cached=(revision,EnergyRetriever(events,kind=self.backend,dimension=self.dimension,
                                                 capacity=self.capacity,controller=self.controller,prior=self.prior,
                                                 score_threshold=self.score_threshold,margin_threshold=self.margin_threshold))
                self._cached=cached
            read=cached[1].read
        current=q.subject;evidence=[];trace=[]
        for relation in q.relations:
            fact=read(current,relation)
            diagnostic=(dict(cached[1].last_diagnostic) if self.backend!='exact' and cached[1].last_diagnostic else
                        {'subject':current,'relation':relation,'score':None,'margin':None,
                         'accepted':fact is not None,'selected_fact_id':fact.get('fact_id') if fact else None})
            trace.append({'hop':len(trace)+1,**diagnostic})
            if fact is None:
                return {'answer':None,'abstained':True,'reason':'missing_or_uncertain_fact',
                        'facts':evidence,'revision':revision,'backend':self.backend,'trace':trace}
            evidence.append(fact);current=fact['object']
        return {'answer':current,'abstained':False,'facts':evidence,'revision':revision,'backend':self.backend,'trace':trace}
