"""v17 compositional synthetic benchmark. No real-world truth labels are claimed.

A registered single entity/value is supplied lexical infrastructure. Reported,
hypothetical, quoted, or negated instructions do not authorize edits in this task.
The old wording test is never used for vocabulary, training, or selection.
"""
from __future__ import annotations
import hashlib
import random
from .bridge import NAMES, VALUES, HOLD, SET, DELETE, ASK, words
from .data import TRAIN_TEMPLATES, HELDOUT_TEMPLATES

PREFIXES = ('', 'for this record ', 'please note that ', 'in this update ', 'today ')
SUFFIXES = ('', ' thank you', ' for the record', ' right now')
WRAPPERS = (
    'do not execute this instruction: {body}',
    'someone said that {body}',
    'this is a fictional statement: {body}',
    'maybe {body}',
    'a story says that {body}',
    'do not record the following: {body}',
)
ALIASES = 'nora oscar quinn rachel sam tara ursula victor wendy xavier yara zane bella cyrus diana elena'.split()
ALIAS_VALUES = 'hangar chapel canteen pavilion orchard museum bakery theater observatory corridor conservatory station clinic warehouse classroom laboratory'.split()


def text_bucket(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16) % 10


def pair_heldout(k: int, v: int) -> bool:
    return (k + 3 * v) % 5 == 0


def catalog(expanded: bool):
    rows = []
    for op, templates in TRAIN_TEMPLATES.items():
        for ci, core in enumerate(templates):
            if not expanded:
                rows.append((op, core, f'basic-{op}-{ci}', False))
                continue
            for pi, prefix in enumerate(PREFIXES):
                for si, suffix in enumerate(SUFFIXES):
                    # Always expose each core. Hold out combinations, not vocabulary.
                    held = pi > 0 and si > 0 and (ci + 2*pi + si + op) % 4 == 0
                    template = prefix + core.rstrip('.?!') + suffix + ('?' if op == ASK else '.')
                    rows.append((op, template, f'compose-{op}-{ci}-{pi}-{si}', held))
            if op in (SET, DELETE):
                for wi, wrapper in enumerate(WRAPPERS):
                    # Each wrapper and each core occurs in training; selected pairs do not.
                    held = (ci + wi + op) % 4 == 0
                    template = wrapper.format(body=core)
                    rows.append((HOLD, template, f'scope-{op}-{ci}-{wi}', held))
    # Legacy wording templates remain excluded as whole templates. Their atoms may
    # occur in other compositions; this is a reused diagnostic, not a pristine test.
    def normalized(t):
        return ' '.join(words(t.format(name='entitytoken', value='valuetoken')))
    forbidden = {normalized(t) for ts in HELDOUT_TEMPLATES.values() for t in ts}
    return [r for r in rows if normalized(r[1]) not in forbidden]


def sample_rows(n: int, seed: int, split: str, expanded: bool = True):
    rng = random.Random(seed)
    if split == 'legacy_wording':
        rows = [(op, t, f'legacy-{op}-{i}', True)
                for op, ts in HELDOUT_TEMPLATES.items() for i,t in enumerate(ts)]
    else:
        rows = [r for r in catalog(expanded)
                if r[3] == (split == 'composition')]
    by_op = {op:[r for r in rows if r[0]==op] for op in range(4)}
    answer, seen = [], set()
    for attempt in range(max(100000, n * 1000)):
        op = SET if split == "pairs" else len(answer) % 4
        o, template, tid, held = rng.choice(by_op[op])
        k,v = rng.randrange(16),rng.randrange(16)
        if op == SET and pair_heldout(k,v) != (split == 'pairs'):
            continue
        text = template.format(name=NAMES[k],value=VALUES[v])
        bucket = text_bucket(text)
        if split=='train' and bucket>=8: continue
        if split=='validation' and bucket!=8:continue
        if split=='iid' and bucket!=9:continue
        if text in seen and split not in ("train", "legacy_wording"):continue
        seen.add(text)
        answer.append({'text':text,'op':o,'key':k,'value':v,'template_id':tid,'template':template})
        if len(answer)==n:return answer
    raise RuntimeError(f'Not enough unique examples for {split}: {len(answer)}/{n}')


def alias_rows(rows):
    return [{**r, 'text':r['template'].format(name=ALIASES[r['key']], value=ALIAS_VALUES[r['value']])}
            for r in rows]


def main():
    import json
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]/'results/v17/data'
    root.mkdir(parents=True,exist_ok=True)
    datasets={
       'train_basic':sample_rows(8192,17001,'train',False),
       'train_expanded':sample_rows(8192,17002,'train',True),
       'validation':sample_rows(512,17003,'validation',True),
       'iid':sample_rows(512,17101,'iid',True),
       'composition':sample_rows(1024,17102,'composition',True),
       'pairs':sample_rows(1024,17103,'pairs',True),
       'legacy_wording':sample_rows(1024,17104,'legacy_wording',True),
    }
    datasets['aliases']=alias_rows(datasets['iid'])
    training={r['text'] for k in ('train_basic','train_expanded') for r in datasets[k]}
    validation={r['text'] for r in datasets['validation']}
    manifest={}
    for key,rows in datasets.items():
        if key not in ('train_basic','train_expanded'):
            assert not training.intersection(r['text'] for r in rows),key
        if key not in ('train_basic','train_expanded','validation'):
            assert not validation.intersection(r['text'] for r in rows),key
        data=''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows).encode()
        (root/f'{key}.jsonl').write_bytes(data)
        manifest[key]={'rows':len(rows),'sha256':hashlib.sha256(data).hexdigest(),
                       'templates':len({r['template_id'] for r in rows}), 'unique_texts':len({r['text'] for r in rows})}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
