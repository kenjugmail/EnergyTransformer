"""Resume evaluation only after native training/selection exhausts its deadline."""
import json
from pathlib import Path

import torch

from energy_real.diagnostics import active_records, events_for, generate
from energy_real.episodes import oracle
from energy_real.fact_memory import FactMemory
from energy_real.graph_program import GraphProgram, execute
from energy_real.hybrid_study import examples, pack, score
from energy_real.local_energy_study import Budget
from energy_real.model import Config, LM
from energy_real.runtime import atomic_json, environment, sha256, source_identity


def finish(study=Path('runs/hybrid_native')):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if (study / 'results.json').exists(): raise FileExistsError(study / 'results.json')
    selection = json.loads((study / 'selection.json').read_text())
    plan = json.loads((study / 'plan.json').read_text())
    identity = {'source_sha256': source_identity(), 'selection_sha256': sha256(study / 'selection.json'),
                'test_sha256': sha256(study / 'test_data.json')}
    assert selection['source_sha256'] == plan['source_sha256'] == identity['source_sha256']
    assert sha256(study / 'data.json') == plan['data_sha256']
    testing = json.loads((study / 'test_data.json').read_text())
    assert testing == examples(600, 591003)
    config = Config(**plan['config'])
    data = pack(testing, config.memory_dim, config.memory_capacity)
    budget = Budget(Path('runs/local_energy_budget.json'), 300, Path('reports/publication/device_stages.json'))
    budget.entry['phase'] = 'frozen_native_evaluation_continuation'
    status, tests = 'failed', []
    try:
        for record in selection['records']:
            output = study / f"evaluation_seed{record['seed']}.json"
            checkpoint = study / f"seed{record['seed']}.pt"
            assert sha256(checkpoint) == record['checkpoint_sha256']
            if output.exists():
                saved = json.loads(output.read_text())
                assert saved['identity'] == identity
                tests.append(saved['test'])
                continue
            saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
            assert saved['source_sha256'] == identity['source_sha256'] and saved['parent_sha256'] == plan['parent_sha256']
            model = LM(config).eval(); model.load_state_dict(saved['weights'], strict=True)
            active = score(model, data, budget)
            disabled = score(model, data, budget, use_memory=False)
            changed = (data[0], data[1], FactMemory(data[2].keys.roll(1, 0), data[2].values.roll(1, 0), data[2].mask.roll(1, 0)))
            shuffled = score(model, changed, budget)
            groups = {family: sum(v for v, row in zip(active['correct'], testing) if row['family'] == family) /
                      sum(row['family'] == family for row in testing) for family in ('set', 'replace', 'delete', 'hold')}
            test = {'seed': record['seed'], 'memory': active, 'memory_disabled': disabled,
                    'memory_shuffled': shuffled, 'groups': groups}
            atomic_json({'identity': identity, 'test': test}, output)
            tests.append(test)
            print(json.dumps({'seed': record['seed'], 'memory': active['accuracy'], 'disabled': disabled['accuracy'],
                              'shuffled': shuffled['accuracy']}), flush=True)
        exact = []
        for episode in generate('test', 30, seed=591004):
            budget.check()
            facts = active_records(events_for(episode))
            answer = execute(GraphProgram.path(episode.query.subject, episode.query.relations), list(facts.values()), 1)
            expected, _ = oracle(episode)
            correct = answer['abstained'] if expected is None else not answer['abstained'] and answer['answers'] == [expected]
            exact.append({'id': episode.id, 'family': episode.family, 'correct': correct,
                          'abstained': answer['abstained'], 'evidence': [f['fact_id'] for f in answer['facts']]})
        assert identity['source_sha256'] == source_identity()
        assert identity['selection_sha256'] == sha256(study / 'selection.json')
        assert identity['test_sha256'] == sha256(study / 'test_data.json')
        result = {'status': 'completed', 'source_sha256': identity['source_sha256'], 'environment': environment(torch.device('cpu')),
                  'parent_sha256': plan['parent_sha256'], 'selected_seed': selection['selected_seed'], 'records': selection['records'],
                  'tests': tests, 'graph_program_test': {'accuracy': sum(r['correct'] for r in exact) / len(exact), 'rows': exact},
                  'gpu_seconds': 0, 'scope': 'Reader integration and one-hop symbol binding; no general language quality claim.',
                  'evaluation_continuation': {**identity, 'reason': 'The original 1,800-second run completed training and selection, then expired during evaluation.',
                                              'training_updates_repeated': 0, 'selection_changes': 0, 'evaluation_deadline_s': 300}}
        atomic_json(result, study / 'results.json')
        status = 'completed'
        return result
    finally:
        budget.finish(status)


if __name__ == '__main__':
    finish()
