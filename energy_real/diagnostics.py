"""Versioned relation episodes shared by memory training and language diagnostics.

Entity namespaces and language templates are disjoint across splits. Structured
paths are evaluator labels; natural-language interpretation must not receive them.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import random
from .episodes import Episode, oracle
from .episodic import FactEdit, Query, canonical

VERSION = 'relation_diagnostics_v2.1'
FAMILIES = ('insert', 'replace', 'delete', 'absent', 'similar', 'distractors',
            'overload', 'two_hop', 'three_hop', 'paraphrase', 'missing_two_hop', 'missing_three_hop')
SPLIT_SEEDS = {'train': 271001, 'valid': 271002, 'test': 271003}
TEMPLATES = {
    'train': {
        'located_in': '{s} is located in {o}.', 'owned_by': '{s} belongs to {o}.',
        'works_at': '{s} works at {o}.', 'delete': 'Forget the {r} fact about {s}.',
        'hold': 'Imagine {s} had {r} {o}; this is hypothetical.',
        'one': 'What is the {r} of {s}?',
        'two': 'Where does the owner of {s} work?',
        'three': 'Where is the workplace of the owner of {s} located?',
    },
    'valid': {
        'located_in': 'The location recorded for {s} is {o}.',
        'owned_by': 'The owner recorded for {s} is {o}.',
        'works_at': 'The employer recorded for {s} is {o}.',
        'delete': 'Erase the recorded {r} for {s}.',
        'hold': 'Suppose {s} had {r} {o}; retain the actual record.',
        'one': 'Retrieve the recorded {r} for {s}.',
        'two': 'Identify the workplace of whoever owns {s}.',
        'three': 'Identify the location of the workplace of whoever owns {s}.',
    },
    'test': {
        'located_in': 'According to the latest record, {o} contains {s}.',
        'owned_by': 'According to the latest record, {o} owns {s}.',
        'works_at': 'According to the latest record, {o} employs {s}.',
        'delete': 'The {r} information for {s} is withdrawn and now unknown.',
        'hold': 'A fictional account says {s} has {r} {o}; no factual change is asserted.',
        'one': 'Using current information, tell me the {r} associated with {s}.',
        'two': 'Which organization employs the person who owns {s}?',
        'three': 'In which place is the organization employing the person who owns {s}?',
    },
}
RELATION_NAMES = {'located_in':'location', 'owned_by':'owner', 'works_at':'workplace'}


@dataclass
class DiagnosticEpisode(Episode):
    natural_question: str
    split: str
    template_family: str

    def payload(self):
        return {**super().payload(), 'natural_question':self.natural_question,
                'split':self.split, 'template_family':self.template_family, 'version':VERSION}


def generate(split='test', per_family=10, seed=None):
    if split not in SPLIT_SEEDS or type(per_family) is not int or per_family<1:
        raise ValueError('Known split and positive episode count required')
    seed = SPLIT_SEEDS[split] if seed is None else seed
    result = []
    templates = TEMPLATES[split]
    for family in FAMILIES:
        for i in range(per_family):
            identifier = hashlib.sha256(canonical([VERSION,split,seed,family,i]).encode()).hexdigest()[:20]
            # Per-episode randomness keeps IDs and content stable when a caller
            # requests a smaller/larger subset of the same split.
            rng = random.Random(int(identifier,16))
            prefix = f'v2_{split}_{seed}_{family}_{i}_{rng.randrange(10**9)}'
            names = [f'{prefix}_n{j}' for j in range(72)]
            a,b,c,d,e = names[:5]
            edits = []
            def add(subject, relation, obj, operation='set'):
                template = templates[relation if operation=='set' else operation]
                text = template.format(s=subject, o=obj, r=RELATION_NAMES[relation])
                edits.append(FactEdit(operation,subject,relation,obj,text))
            path = ('located_in',)
            if family in ('two_hop','three_hop','missing_two_hop','missing_three_hop'):
                path = ('owned_by','works_at')
                add(a,'owned_by',b)
                add(b,'works_at',c)
                add(e,'works_at',d)  # Valid but irrelevant alternate path.
                if 'three' in family:
                    path += ('located_in',)
                    add(c,'located_in',d)
                    add(c,'located_in',names[6])
                if family=='two_hop':add(b,'works_at',names[7])
                if family.startswith('missing'):
                    missing_relation='located_in' if 'three' in family else 'works_at'
                    missing_subject=c if 'three' in family else b
                    add(missing_subject,missing_relation,None,'delete')
            elif family=='absent':
                add(b,'located_in',c)
            elif family=='overload':
                for j in range(48):add(names[j],'located_in',names[(j+13)%72])
                a=names[rng.randrange(32)]
                add(names[40],'located_in',None,'delete')
            else:
                add(a,'located_in',b)
                if family in ('replace','paraphrase'):
                    add(e,'located_in',d)
                    for obj in (c,d,b,e):add(a,'located_in',obj)
                elif family=='delete':
                    add(a,'located_in',None,'delete');add(e,'located_in',d)
                elif family=='similar':
                    for j in range(8):add(a+f'_near_{j}','located_in',names[j+2])
                elif family=='distractors':
                    for j in range(20):add(a,'located_in',names[j+2],'hold')
                    add(e,'located_in',d)
            query = Query(a,path)
            depth = ('one','two','three')[len(path)-1]
            natural = templates[depth].format(s=a,r=RELATION_NAMES[path[0]])
            result.append(DiagnosticEpisode(identifier,family,edits,query,
                          f'Follow {"/".join(path)} from {a}.',natural,split,f'{split}:{depth}'))
    return result


def manifest(episodes):
    payload = [episode.payload() for episode in episodes]
    return {'version':VERSION, 'sha256':hashlib.sha256(canonical(payload).encode()).hexdigest(),
            'episodes':payload}


def events_for(episode):
    """Same event identity as one reviewed batch in FactStore, without disk I/O."""
    return [{**asdict(edit),'source_id':edit.source_id,'revision':1,
             'fact_id':hashlib.sha256(f'1:{i}:{canonical(asdict(edit))}'.encode()).hexdigest()}
            for i,edit in enumerate(episode.edits)]


def active_records(events):
    active = {}
    for event in events:
        key = (event['subject'],event['relation'])
        if event['operation']=='set':active[key]=event
        elif event['operation']=='delete':active.pop(key,None)
    return active
