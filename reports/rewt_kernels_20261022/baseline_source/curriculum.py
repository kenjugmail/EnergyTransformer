"""Broader synthetic training prompts and explicitly training-only annotations.

Semantic IDs exclude wording, fact order and the position of the requested key.
Every paraphrase therefore remains in its original train/validation/test split.
Neither prompt construction nor answer annotations are used by model inference.
"""
from dataclasses import dataclass
import random
import re

from .tasks import Task, semantic_identity, semantic_partition, solve


@dataclass(frozen=True)
class TraceBoundary:
    query: int
    end: int
    keys: tuple[int, ...]
    source: int
    numeric: int | None
    remaining: int


def render_task(payload, split, template, *, seed=0):
    """Render a semantic problem; labels stay outside the inference prompt."""
    if not 0 <= template <= 8:
        raise ValueError('template must be in 0..8')
    identity = semantic_identity(payload)
    if semantic_partition(identity) != split:
        raise ValueError('semantic identity belongs to a different split')
    rng = random.Random(seed)
    family = payload['family']
    if family == 'arithmetic':
        start, deltas = payload['start'], payload['deltas']
        steps = len(deltas)
        if template == 0:
            names = 'ABCDE'
            prompt = f'A={start};' + ';'.join(f'{names[i+1]}={names[i]}+{v}' for i, v in enumerate(deltas))
            prompt += f';{names[steps]}=?'
        elif template == 1:
            prompt = f'start {start};' + ';'.join(f'add {v}' for v in deltas) + ';result?'
        elif template == 2:
            prompt = 'sum ' + '+'.join(map(str, [start, *deltas])) + '=?'
        elif template == 3:
            prompt = f'begin {start};' + ';'.join(f'plus {v}' for v in deltas) + ';total?'
        elif template == 4:
            prompt = f'start={start};increments=' + ','.join(map(str, deltas)) + ';total=?'
        elif template == 5:
            prompt = f'initial {start};' + ';'.join(f'increase {v}' for v in deltas) + ';final?'
        elif template == 6:
            prompt = 'compute ' + ' plus '.join(map(str, [start, *deltas])) + '=?'
        elif template == 7:
            prompt = f'value {start};' + ';'.join(f'append +{v}' for v in deltas) + ';answer?'
        else:
            prompt = 'calculate:' + '+'.join(map(str, [start, *deltas])) + ';give total?'
    elif family == 'rules':
        edges = list(payload['edges'])
        rng.shuffle(edges)
        facts = ';'.join(f'{a}>{b}' for a, b in edges)
        start, steps = payload['start'], payload['hops']
        prompts = [f'{facts};from {start} go {steps}=?',
                   f'walk {start} {steps}:{facts}=?',
                   f'map:{facts};{start} after {steps} links?',
                   f'trace {steps} from {start}:{facts}=?',
                   f'follow {start}/{steps};{facts}=?',
                   f'rules={facts};apply {steps} to {start}=?',
                   f'hops {steps}:{start}:{facts};result?',
                   f'{facts};move {steps} times from {start}=?',
                   f'edges:{facts};find {start} in {steps} jumps?']
        prompt = prompts[template]
    elif family == 'recall':
        steps = len(payload['distractors'])
        facts = [f'X={v}' for v in payload['distractors']]
        facts.insert(rng.randrange(len(facts) + 1), f'K={payload["value"]}')
        facts = ';'.join(facts)
        prompts = [f'{facts};K=?', f'remember:{facts};stored K=?',
                   f'keys:{facts};key K?', f'read:{facts};K?',
                   f'lookup K:{facts};K=?', f'retrieve K in:{facts};?',
                   f'which K?{facts}=?', f'{facts};what was K?',
                   f'values:{facts};return K=?']
        prompt = prompts[template]
    else:
        raise ValueError('unknown task family')
    prompt += '\n'
    trace, answer = solve(payload)
    response = 'steps=' + ','.join(trace) + ';ans=' + answer + '\n'
    if len((prompt + response).encode('ascii')) > 128:
        raise ValueError('task exceeds the shared 128-byte window')
    return Task(identity, family, split, template, steps, prompt, response, payload,
                'familiar_format' if template < 5 else 'withheld_wording')


def generate_curriculum(split, per_group, *, seed=2029, exclude_ids=(), test=False):
    """Balanced semantic groups, with fresh test wording never used in validation."""
    if split not in ('train', 'valid', 'test') or per_group < 1:
        raise ValueError('known split and positive group size required')
    rng = random.Random(seed + {'train': 0, 'valid': 10000, 'test': 20000}[split])
    used = set(exclude_ids)
    tasks = []
    groups = [('arithmetic', n) for n in (1, 2, 3)] + [('rules', n) for n in (1, 2, 3)]
    groups += [('recall', n) for n in (4, 8, 16)]
    for family, depth in groups:
        accepted = 0
        for _ in range(per_group * 10000):
            if family == 'arithmetic':
                payload = dict(family=family, start=rng.randrange(64),
                               deltas=[rng.randrange(1, 64) for _ in range(depth)])
                # All original one-addition test identities were already opened.
                # The new test measures the extended numerical range explicitly.
                if split == 'test' and depth == 1 and max(payload['start'], *payload['deltas']) < 32:
                    continue
            elif family == 'rules':
                nodes = rng.sample(list('ABCDEFGHIJKLMNOPQRSTUVWX'), depth + 4)
                edges = [[nodes[i], nodes[i+1]] for i in range(depth)]
                edges += [[nodes[-3], nodes[-2]], [nodes[-2], nodes[-1]]]
                payload = dict(family=family, start=nodes[0], hops=depth, edges=sorted(edges))
            else:
                payload = dict(family=family, value=rng.randrange(10),
                               distractors=[rng.randrange(10) for _ in range(depth)])
            identity = semantic_identity(payload)
            if identity in used or semantic_partition(identity) != split:
                continue
            templates = [rng.randrange(5)]
            if split != 'train':
                templates.append(rng.choice((7, 8) if split == 'test' else (5, 6)))
            try:
                rendered = [render_task(payload, split, t, seed=rng.randrange(2**32)) for t in templates]
            except ValueError:
                continue
            tasks.extend(rendered)
            used.add(identity)
            accepted += 1
            if accepted == per_group:
                break
        if accepted != per_group:
            raise RuntimeError(f'not enough disjoint {family}/{depth} identities')
    return tasks


def training_annotations(task):
    """Ground-truth supervision used only by a teacher-forced training loss.

    Each operand/fact key and source result must already have been observed
    when that output position predicts the next trace byte.
    """
    trace, _ = solve(task.payload)
    cursor = len(task.prompt) + len('steps=') - 1
    numbers = [(m.end()-1, int(m[0])) for m in re.finditer(r'\d+', task.prompt)] if task.family == 'arithmetic' else []
    facts = {m[1]: m.end()-1 for m in re.finditer(r'([A-Z])>([A-Z])', task.prompt)}
    recall = re.search(r'K=(\d)', task.prompt)
    boundaries = []
    previous = None
    for index, value in enumerate(trace):
        if task.family == 'arithmetic':
            keys = (numbers[0][0] if index == 0 else previous, numbers[index+1][0])
            source = task.payload['start'] if index == 0 else int(trace[index-1])
        elif task.family == 'rules':
            source = ord(task.payload['start'] if index == 0 else trace[index-1])
            keys = (facts[chr(source)],)
        else:
            if recall is None:
                raise ValueError('missing observed K fact')
            keys = (recall.end()-1,)
            source = 48 + task.payload['value']
        if any(key is None or not 0 <= key <= cursor for key in keys):
            raise ValueError('a training fact lies outside the observed prefix')
        boundaries.append(TraceBoundary(cursor, cursor + len(value), keys, source,
                                        int(value) if task.family == 'arithmetic' else None,
                                        len(trace)-index))
        previous = cursor + len(value)
        if task.family == 'arithmetic':
            numbers.append((previous, int(value)))
        cursor += len(value) + 1
    return boundaries, numbers
