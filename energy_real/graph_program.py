"""Bounded typed graph programs over one exact, provenance-bearing snapshot.

Only named operations are interpreted. There is no eval, generated code, or
model-authorized write. A missing premise makes the whole answer unknown.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Instruction:
    operation: str
    arguments: tuple[str, ...]


@dataclass(frozen=True)
class GraphProgram:
    steps: tuple[Instruction, ...]

    def validate(self):
        if not isinstance(self.steps, tuple) or not 1 <= len(self.steps) <= 16:
            raise ValueError('A graph program requires 1–16 instructions')
        for index, step in enumerate(self.steps):
            if not isinstance(step, Instruction) or not isinstance(step.arguments, tuple):
                raise ValueError('Expected typed graph instructions')
            if any(not isinstance(a, str) or not a or len(a) > 128 for a in step.arguments):
                raise ValueError('Invalid instruction arguments')
            if step.operation == 'entity':
                if len(step.arguments) != 1:
                    raise ValueError('entity requires one literal entity ID')
            elif step.operation == 'follow':
                if len(step.arguments) != 2:
                    raise ValueError('follow requires a previous result and relation')
            elif step.operation in ('intersect', 'union'):
                if len(step.arguments) != 2:
                    raise ValueError('Set operations require two previous results')
            else:
                raise ValueError('Unknown graph operation')
            refs = step.arguments[:1] if step.operation == 'follow' else step.arguments if step.operation in ('intersect', 'union') else ()
            for ref in refs:
                if not ref.isascii() or not ref.isdecimal() or str(int(ref)) != ref or not 0 <= int(ref) < index:
                    raise ValueError('Instructions may reference only earlier results')
        return self

    @classmethod
    def path(cls, subject, relations):
        return cls((Instruction('entity', (subject,)), *(
            Instruction('follow', (str(i), relation)) for i, relation in enumerate(relations)))).validate()

    @classmethod
    def parse(cls, value):
        if not isinstance(value, dict) or set(value) != {'steps'} or not isinstance(value['steps'], list):
            raise ValueError('Expected a steps list')
        steps = []
        for row in value['steps']:
            if not isinstance(row, dict) or set(row) != {'operation', 'arguments'} or not isinstance(row['arguments'], list):
                raise ValueError('Malformed graph instruction')
            steps.append(Instruction(row['operation'], tuple(row['arguments'])))
        return cls(tuple(steps)).validate()


def execute(program, facts, revision, *, max_bindings=256):
    program.validate()
    if type(max_bindings) is not int or max_bindings < 1:
        raise ValueError('Positive binding budget required')
    lookup = {(f['subject'], f['relation']): f for f in facts}
    if len(lookup) != len(facts):
        raise ValueError('Expected one current fact per address')
    states, trace, evidence = [], [], {}
    for index, step in enumerate(program.steps):
        if step.operation == 'entity':
            value = {step.arguments[0]}
        elif step.operation == 'follow':
            value = set()
            for subject in sorted(states[int(step.arguments[0])]):
                fact = lookup.get((subject, step.arguments[1]))
                if fact is None:
                    return {'answers': [], 'abstained': True, 'reason': 'missing_premise', 'revision': revision,
                            'backend': 'exact_graph', 'facts': list(evidence.values()), 'trace': trace,
                            'missing': {'step': index, 'subject': subject, 'relation': step.arguments[1]}}
                value.add(fact['object'])
                evidence[fact['fact_id']] = fact
        else:
            a, b = (states[int(ref)] for ref in step.arguments)
            value = a & b if step.operation == 'intersect' else a | b
        if len(value) > max_bindings:
            raise ValueError('Program exceeded binding budget')
        states.append(value)
        trace.append({'step': index, 'operation': step.operation, 'values': sorted(value)})
    return {'answers': sorted(states[-1]), 'abstained': False, 'revision': revision,
            'backend': 'exact_graph', 'facts': list(evidence.values()), 'trace': trace}
