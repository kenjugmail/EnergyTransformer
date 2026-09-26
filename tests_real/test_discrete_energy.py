import copy
import itertools

import torch

from energy_real.discrete_energy import DiscreteEnergyModel, encode_factors
from energy_real.local_energy_data import graph_dataset


def test_sum_product_matches_exhaustive_discrete_paths_and_has_finite_gradients():
    torch.manual_seed(10)
    data = graph_dataset('test', 6, 17, entities=3)
    model = DiscreteEnergyModel().double()
    features, steps, starts = encode_factors(data.episodes)
    features = features.double()
    logits = model(features, steps, starts)
    edge_scores = features @ model.weights
    for b, episode in enumerate(data.episodes):
        for answer in range(4):
            alternatives = []
            for middle in itertools.product(range(4), repeat=len(episode['path']) - 1):
                path = (episode['start'], *middle, answer)
                alternatives.append(sum(edge_scores[b, h, a, c] for h, (a, c) in enumerate(zip(path, path[1:]))))
            expected = torch.logsumexp(torch.stack(alternatives), 0)
            torch.testing.assert_close(logits[b, answer], expected)
    torch.nn.functional.cross_entropy(logits, data.y).backward()
    assert torch.isfinite(model.weights.grad).all()
    assert model.weights.numel() == 7


def test_discrete_model_is_equivariant_to_entity_renaming():
    torch.manual_seed(11)
    data = graph_dataset('test', 12, 41, entities=3)
    renamed = copy.deepcopy(data.episodes)
    permutation = [2, 0, 1, 3]
    for episode in renamed:
        episode['start'] = permutation[episode['start']]
        episode['events'] = [(op, permutation[s], r, None if obj is None else permutation[obj])
                             for op, s, r, obj in episode['events']]
    model = DiscreteEnergyModel()
    a, b = model(*encode_factors(data.episodes)), model(*encode_factors(renamed))
    torch.testing.assert_close(a.log_softmax(-1), b[:, permutation].log_softmax(-1))


def test_factor_encoding_does_not_use_answer_and_separates_stale_and_hypothetical():
    episode = {'entities': 3, 'start': 0, 'path': [0], 'events':
               [('set', 0, 0, 1), ('set', 0, 0, 2), ('hold', 0, 0, 0)]}
    features, _, _ = encode_factors([episode])
    assert features[0, 0, 0, 2, 0] == 1  # Current observation.
    assert features[0, 0, 0, 1, 2] == 1  # Overwritten observation.
    assert features[0, 0, 0, 0, 3] == 1  # Hypothetical assertion.
    episode['answer'] = 999
    changed, _, _ = encode_factors([episode])
    assert torch.equal(features, changed)
