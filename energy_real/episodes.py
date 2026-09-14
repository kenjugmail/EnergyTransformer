"""Deterministic held-out relation episodes with exact, inspectable semantics."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import random
from .episodic import FactEdit, Query, canonical

FAMILIES=('insert','replace','delete','absent','similar','distractors','overload','two_hop','three_hop','paraphrase')


@dataclass
class Episode:
    id: str
    family: str
    edits: list[FactEdit]
    query: Query
    question: str

    def payload(self):
        return {'id':self.id,'family':self.family,'edits':[asdict(e) for e in self.edits],
                'query':asdict(self.query),'question':self.question}


def generate(seed=1001, per_family=4, split='test'):
    if split not in ('train','valid','test') or per_family<1:raise ValueError('Invalid episode split or count')
    rng=random.Random(seed)
    result=[]
    for family in FAMILIES:
        for i in range(per_family):
            prefix=f'{split}_{rng.randrange(10**9)}'
            names=[f'{prefix}_node{n}' for n in range(64)]
            a,b,c,d,e=names[:5]
            edits=[]
            def add(subject,obj,op='set'):
                if op=='delete':text=f'Remove the known location of {subject}.'
                elif op=='hold':text=f'Hypothetically, {subject} could be located in {obj}; do not update the record.'
                elif split=='train':text=f'{subject} is located in {obj}.'
                elif split=='valid':text=f'Current location: {subject} -> {obj}.'
                else:text=f'The record now places {subject} within {obj}.'
                edits.append(FactEdit(op,subject,'located_in',obj,text))
            hops=1
            if family=='absent':add(b,c)
            elif family=='overload':
                for j in range(48):add(names[j],names[(j+7)%64])
                a=names[rng.randrange(12)]
            else:
                add(a,b)
                if family=='replace':
                    add(e,c)
                    for destination in [c,d,b,e]:add(a,destination)
                elif family=='delete':add(a,None,'delete');add(e,c)
                elif family=='similar':
                    for j in range(8):add(a+f'_near_{j}',names[j+2])
                elif family=='distractors':
                    for j in range(20):add(a,names[2+j],'hold')
                    add(e,d)
                elif family in ('two_hop','three_hop'):
                    add(b,c);hops=2
                    if family=='three_hop':add(c,d);add(c,e);hops=3
                    else:add(b,d)
                elif family=='paraphrase':add(a,c)
            query=Query(a,('located_in',)*hops)
            # Natural-language wording is independent of the structured query
            # supplied to oracle memory experiments.
            question=f'Follow {"/".join(query.relations)} from {a}.'
            ident=hashlib.sha256(canonical([split,seed,family,i]).encode()).hexdigest()[:16]
            result.append(Episode(ident,family,edits,query,question))
    return result


def oracle(episode):
    facts={}
    for i,edit in enumerate(episode.edits):
        key=(edit.subject,edit.relation)
        if edit.operation=='set':facts[key]=(edit.object,i)
        elif edit.operation=='delete':facts.pop(key,None)
    current=episode.query.subject;support=[]
    for relation in episode.query.relations:
        if (current,relation) not in facts:return None,support
        current,index=facts[(current,relation)];support.append(index)
    return current,support
