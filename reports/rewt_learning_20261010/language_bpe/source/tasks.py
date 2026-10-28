"""Deterministic arithmetic, rule chaining, and distractor-recall tasks.

No evaluator or symbolic solver is called by the model. The solver is confined
to dataset construction and scoring after autoregressive output is generated.
Semantic hash partitioning prevents a paraphrase of a training problem from
appearing in validation/test. Template 2 and four-hop chains are held out.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import random
import re


@dataclass(frozen=True)
class Task:
    id: str
    family: str
    split: str
    template: int
    steps: int
    prompt: str
    response: str
    payload: dict
    difficulty: str

    def to_dict(self):
        return asdict(self)


def solve(payload):
    if payload["family"] == "arithmetic":
        value = payload["start"]
        trace = []
        for delta in payload["deltas"]:
            value += delta
            trace.append(str(value))
    elif payload["family"] == "rules":
        edges = dict(payload["edges"])
        value = payload["start"]
        trace = []
        for _ in range(payload["hops"]):
            value = edges[value]
            trace.append(value)
    elif payload["family"] == "recall":
        value = payload["value"]
        trace = [str(value)]
    else:
        raise ValueError("Unknown task family")
    return trace, str(value)


def semantic_identity(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def semantic_partition(identity):
    bucket = int(identity[:12], 16) % 10
    return "train" if bucket < 8 else "valid" if bucket == 8 else "test"


def generate_tasks(split, count, seed=811, *, number_range=10, exclude_ids=()):
    if split not in ("train", "valid", "test") or type(count) is not int or count < 1:
        raise ValueError("A known split and positive task count are required")
    if type(number_range) is not int or not 2 <= number_range <= 100:
        raise ValueError("number_range must be an integer in 2..100")
    rng = random.Random(seed + {"train": 0, "valid": 10_000, "test": 20_000}[split])
    tasks, used = [], set(exclude_ids)
    for attempt in range(count * 1000):
        family = ("arithmetic", "rules", "recall")[len(tasks) % 3]
        template = rng.choice((0, 1) if split == "train" else (0, 1, 2))
        steps = rng.choice((1, 2, 3) if split == "train" else (1, 2, 3, 4))
        difficulty = "heldout_template" if template == 2 else "heldout_instance"
        if steps == 4 and family != "recall":
            difficulty += "+longer_chain"
        if family == "arithmetic":
            start = rng.randrange(number_range)
            deltas = [rng.randrange(1, number_range) for _ in range(steps)]
            payload = dict(family=family, start=start, deltas=deltas)
            if template == 0:
                names = list("ABCDE")
                prompt = f"A={start};" + ";".join(
                    f"{names[i+1]}={names[i]}+{delta}" for i, delta in enumerate(deltas))
                prompt += f";{names[steps]}=?\n"
            elif template == 1:
                prompt = f"start {start};" + ";".join(f"add {delta}" for delta in deltas) + ";result?\n"
            else:
                prompt = "sum " + "+".join(map(str, [start, *deltas])) + "=?\n"
        elif family == "rules":
            nodes = rng.sample(list("ABCDEFGHIJKLMNOPQRSTUVWX"), steps + 4)
            edges = [[nodes[i], nodes[i+1]] for i in range(steps)]
            edges += [[nodes[-3], nodes[-2]], [nodes[-2], nodes[-1]]]
            # Ordering is part of the presented instance; sorting provides a
            # semantic identity independent of how its facts are paraphrased.
            payload = dict(family=family, start=nodes[0], hops=steps, edges=sorted(edges))
            rng.shuffle(edges)
            facts = ";".join(f"{a}>{b}" for a, b in edges)
            if template == 0:
                prompt = f"{facts};from {nodes[0]} go {steps}=?\n"
            elif template == 1:
                prompt = f"walk {nodes[0]} {steps}:" + facts + "=?\n"
            else:
                prompt = f"map:{facts};{nodes[0]} after {steps} links?\n"
        else:
            delay = rng.choice((1, 4, 8, 16))
            value = rng.randrange(10)
            distractors = [rng.randrange(10) for _ in range(delay)]
            payload = dict(family=family, value=value, distractors=distractors)
            facts = ";".join(f"X={number}" for number in distractors)
            if template == 0:
                prompt = f"K={value};{facts};K=?\n"
            elif template == 1:
                prompt = f"remember {value};{facts};remembered=?\n"
            else:
                prompt = f"key {value};{facts};key?\n"
            steps = delay
        identity = semantic_identity(payload)
        if identity in used or semantic_partition(identity) != split:
            continue
        trace, answer = solve(payload)
        response = "steps=" + ",".join(trace) + ";ans=" + answer + "\n"
        if len((prompt + response).encode("ascii")) > 128:
            continue
        used.add(identity)
        tasks.append(Task(identity, family, split, template, steps, prompt, response, payload, difficulty))
        if len(tasks) == count:
            return tasks
    raise RuntimeError("Task count exceeds this finite generator's available instances")


def score_response(task: Task, response: str):
    match = re.fullmatch(r"steps=([A-Z0-9,-]+);ans=([A-Z]|-?\d+)\s*", response)
    expected_steps, expected_answer = solve(task.payload)
    if match is None:
        return dict(valid_format=False, answer_correct=False, steps_correct=False,
                    both_correct=False, expected=task.response, response=response)
    trace = match[1].split(",")
    answer_correct = match[2] == expected_answer
    steps_correct = trace == expected_steps
    return dict(valid_format=True, answer_correct=answer_correct, steps_correct=steps_correct,
                both_correct=answer_correct and steps_correct, expected=task.response, response=response)


def process_labels(task: Task, response: str):
    """Training annotations stop at the first invalid intermediate step."""
    match = re.fullmatch(r"steps=([A-Z0-9,-]+);ans=([A-Z]|-?\d+)\s*", response)
    if match is None:
        return dict(valid_format=False, first_error=0, labels=[])
    expected, _ = solve(task.payload)
    actual = match[1].split(",")
    labels, first_error = [], None
    for index, value in enumerate(expected):
        correct = index < len(actual) and actual[index] == value
        labels.append(1 if correct else -1)
        if not correct:
            first_error = index
            break
    if first_error is None and len(actual) != len(expected):
        first_error = len(expected)
        labels.append(-1)
    return dict(valid_format=True, first_error=first_error, labels=labels)


def corrupt_trace(task: Task, *, seed=0):
    """Construct one parseable wrong step with an unchanged preceding prefix.

    Only dataset construction and losses consume these labels. Keeping the
    replacement width fixed makes the corresponding first-error byte align
    with the original teacher-forced batch without padding or future leakage.
    """
    rng = random.Random(seed ^ int(task.id[:12], 16))
    trace, answer = solve(task.payload)
    index = rng.randrange(len(trace))
    original = trace[index]
    first = original[0]
    if first.isdigit():
        replacement = str((int(first) + 1) % 10)
        if len(original) > 1 and replacement == "0":
            replacement = "8"
    else:
        replacement = chr((ord(first) - ord("A") + 1) % 26 + ord("A"))
    trace[index] = replacement + original[1:]
    response = "steps=" + ",".join(trace) + ";ans=" + answer + "\n"
    labels = process_labels(task, response)
    assert labels["first_error"] == index
    offset = len("steps=") + sum(len(value) + 1 for value in trace[:index])
    return dict(response=response, response_byte_offset=offset,
                wrong_byte=ord(replacement), **labels)


def task_manifest(tasks):
    encoded = json.dumps([task.to_dict() for task in tasks], sort_keys=True).encode()
    return dict(count=len(tasks), sha256=hashlib.sha256(encoded).hexdigest(),
                families={family: sum(task.family == family for task in tasks)
                          for family in ("arithmetic", "rules", "recall")},
                semantic_ids=[task.id for task in tasks])
