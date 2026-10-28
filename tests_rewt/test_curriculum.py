import pytest

from rewt.curriculum import generate_curriculum, render_task, training_annotations
from rewt.tasks import semantic_identity, semantic_partition, score_response


def test_paraphrases_remain_in_the_same_semantic_partition():
    tasks = {split: generate_curriculum(split, 6, seed=4601) for split in ('train', 'valid', 'test')}
    identities = {s: {t.id for t in rows} for s, rows in tasks.items()}
    assert identities['train'].isdisjoint(identities['valid'] | identities['test'])
    assert identities['valid'].isdisjoint(identities['test'])
    for split, rows in tasks.items():
        for task in rows:
            assert semantic_partition(task.id) == split
            assert score_response(task, task.response)['both_correct']
            assert len((task.prompt + task.response).encode('ascii')) <= 128
            assert task.template in (range(5) if split == 'train' else range(7) if split == 'valid' else [0, 1, 2, 3, 4, 7, 8])
            for template in range(9):
                other = render_task(task.payload, split, template, seed=341)
                assert other.id == task.id
    assert {t.steps for t in tasks['train'] if t.family == 'arithmetic'} == {1, 2, 3}


def test_annotations_reference_only_observed_operands_and_facts():
    tasks = generate_curriculum('train', 12, seed=4602)
    for task in tasks:
        for template in range(9):
            task = render_task(task.payload, 'train', template, seed=345 + template)
            boundaries, numbers = training_annotations(task)
            raw = task.prompt + task.response
            for b in boundaries:
                assert all(key <= b.query for key in b.keys)
                assert raw[b.query] in '=,'
                assert raw[b.end] in '0123456789ABCDEFGHIJKLMNOPQRSTUVWX'
                if task.family == 'rules':
                    assert ord(raw[b.keys[0]-2]) == b.source
                elif task.family == 'recall':
                    assert raw[b.keys[0]-2:b.keys[0]] == 'K='
                    assert ord(raw[b.keys[0]]) == b.source
            for position, number in numbers:
                digits = str(number)
                assert raw[position-len(digits)+1:position+1] == digits


def test_requested_recall_key_occurs_at_multiple_observed_positions():
    task = next(t for t in generate_curriculum('train', 1, seed=4603) if t.family == 'recall' and t.steps == 16)
    positions = {render_task(task.payload, 'train', 0, seed=seed).prompt.index('K=') for seed in range(30)}
    assert len(positions) > 8
    assert task.id == semantic_identity(task.payload)
    with pytest.raises(ValueError, match='different split'):
        render_task(task.payload, 'test', 0)
