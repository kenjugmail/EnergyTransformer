import itertools

import pytest
import torch
from torch.nn import functional as F

from energy_real.local_energy_data import (
    QuadraticEventMemory, SEEN_PATHS, byte_datasets, discrete_graph_energy, execute, graph_dataset,
)
from energy_transformer.geometric_memory import congruence_edit


def test_explicit_memory_matches_existing_precision_energy_edits():
    memory = QuadraticEventMemory(4, 3)
    p, s = memory.precision[None].clone(), memory.information[None].clone()
    for operation, address, value in [('set', 0, 1), ('set', 2, 0), ('set', 0, 0), ('delete', 2, None)]:
        memory.apply(operation, address, value)
        key = F.one_hot(torch.tensor([address]), 4).float()
        target = F.one_hot(torch.tensor([2 if operation == 'delete' else value]), 3).float()
        p, s = congruence_edit(p, s, key, target, torch.ones(1), torch.ones(1), prior=.001, geometry=True)
        torch.testing.assert_close(memory.precision, p[0])
        torch.testing.assert_close(memory.information, s[0])
    before = memory.precision.clone(), memory.information.clone()
    memory.apply('hold', 0, 2)
    assert torch.equal(before[0], memory.precision)
    assert torch.equal(before[1], memory.information)
    assert memory.readout().argmax(-1).tolist() == [0, 2, 2, 2]
    assert torch.linalg.eigvalsh(memory.precision).min() > 0


def test_graph_dataset_disjointness_holdout_and_stable_prefix():
    train = graph_dataset('train', 24, 13)
    valid = graph_dataset('valid', 24, 13)
    test = graph_dataset('test', 36, 13)
    assert not set(train.ids) & set(valid.ids)
    assert not set(train.ids) & set(test.ids)
    small = graph_dataset('test', 12, 13)
    torch.testing.assert_close(small.x, test.x[:12], rtol=0, atol=0)
    assert small.episodes == test.episodes[:12]
    assert all(tuple(e['path']) in SEEN_PATHS for e in train.episodes + valid.episodes)
    assert all(e['path'] == [1, 1] for e in test.episodes if e['group'] == 'heldout_pair')
    assert all(len(e['path']) == 3 for e in test.episodes if e['group'] == 'longer_path')
    for data in (train, valid, test):
        for i, episode in enumerate(data.episodes):
            expected, _, active = execute(episode['events'], episode['start'], episode['path'], episode['entities'])
            assert expected == int(data.y[i])
            # Both input facts and optional memory encode the same active data.
            table = data.x[i, :84].reshape(12, 7).argmax(-1)
            torch.testing.assert_close(table, data.memory[i].argmax(-1))
    with pytest.raises(ValueError):
        graph_dataset('valid', 12, group='heldout_pair')


def test_discrete_min_sum_matches_exhaustive_energy_and_interpreter():
    data = graph_dataset('test', 24, 90, entities=3)
    for episode, expected in zip(data.episodes, data.y):
        result = discrete_graph_energy(episode)
        _, _, active = execute(episode['events'], episode['start'], (), 3)
        best = []
        for candidate in range(4):
            energies = []
            for middle in itertools.product(range(4), repeat=len(episode['path']) - 1):
                path = (episode['start'], *middle, candidate)
                energies.append(sum(b != active.get((a, relation), 3)
                                    for a, b, relation in zip(path, path[1:], episode['path'])))
            best.append(min(energies))
        assert result['candidate_energies'] == best
        assert result['answer'] == int(expected)
        assert result['energy'] == 0
        assert result['path'][0] == episode['start']


def test_byte_windows_have_no_shared_train_validation_test_bytes(tmp_path):
    path = tmp_path / 'bytes'
    path.write_bytes(bytes(range(256)) * 1200)
    splits = byte_datasets(path, train_count=20, valid_count=20, test_count=20)
    occupied = []
    for data in splits.values():
        ranges = []
        for identifier in data.ids:
            _, start, context = identifier.split(':')
            ranges.append(set(range(int(start), int(start) + int(context) + 1)))
        assert sum(map(len, ranges)) == len(set.union(*ranges))
        occupied.append(set.union(*ranges))
    assert not occupied[0] & occupied[1]
    assert not occupied[0] & occupied[2]
    assert not occupied[1] & occupied[2]
    assert splits['train'].x.shape == (20, 64)
