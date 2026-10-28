"""Package and audit a completed native-reader integration study."""
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import statistics
import time

import torch

from energy_real.episodic import FactEdit, FactStore
from energy_real.graph_program import GraphProgram
from energy_real.hybrid import HybridSession
from energy_real.memory_eval import paired_interval
from energy_real.model import Config, LM
from energy_real.runtime import atomic_json, sha256, source_identity


def build():
    torch.set_num_threads(2)
    study, out = Path('runs/hybrid_native'), Path('reports/hybrid_integration')
    result = json.loads((study / 'results.json').read_text())
    assert result['status'] == 'completed' and result['source_sha256'] == source_identity()
    parent = Path('runs/enwik8_energy_10m/best.pt')
    assert sha256(parent) == result['parent_sha256']
    selection = json.loads((study / 'selection.json').read_text())
    assert selection['selected_seed'] == min(result['records'], key=lambda r: (r['validation']['nll'], r['seed']))['seed']
    record = next(r for r in result['records'] if r['seed'] == selection['selected_seed'])
    selected = study / f"seed{selection['selected_seed']}.pt"
    assert sha256(selected) == record['checkpoint_sha256']
    destination = study / 'selected.pt'
    if destination.exists():
        assert sha256(destination) == sha256(selected)
    else:
        shutil.copy2(selected, destination)

    original = torch.load(parent, map_location='cpu', weights_only=True)
    base = LM(Config(**original['config'])).eval()
    base.load_state_dict(original['weights'], strict=True)
    saved = torch.load(destination, map_location='cpu', weights_only=True)
    model = LM(Config(**saved['config'])).eval()
    model.load_state_dict(saved['weights'], strict=True)
    for name, value in base.state_dict().items():
        assert torch.equal(value, model.state_dict()[name]), name
    tokens = torch.randint(256, (1, 17), generator=torch.Generator().manual_seed(801))
    with torch.no_grad():
        expected = base(tokens)[0]
        actual = model(tokens)[0]
    assert torch.equal(expected, actual)

    # A real persistent session on the selected native checkpoint, including
    # edits between cached chunks. These writes are local verification fixtures.
    store_path = study / 'verified_session.sqlite'
    if store_path.exists():
        raise FileExistsError('Use a fresh fixture database for session verification')
    store = FactStore(store_path)
    session = HybridSession(model, store)
    query = GraphProgram.path('a', ('r',))
    transitions = []
    for index, (operation, value, answer) in enumerate((('set', 'C', 'C'), ('set', 'D', 'D'),
                                                       ('hold', 'F', 'D'), ('delete', None, None))):
        store.apply([FactEdit(operation, 'a', 'r', value, f'Fixture: {operation} a.r={value}')],
                    expected_revision=store.revision, request_id=f'fixture-{index}', authorized=True)
        prompt = torch.tensor([list(b'ar?')])
        logits, info = session.forward(prompt)
        with torch.no_grad():
            fresh = model(prompt, memory=session.memory)[0]
        assert torch.equal(logits, fresh) and info['state_reset']
        exact = session.answer(query)
        assert exact['abstained'] if answer is None else exact['answers'] == [answer]
        transitions.append({'operation': operation, 'requested_object': value, 'exact': exact,
                            'neural_next_byte': int(logits[:, -1].argmax()), 'session': info,
                            'fresh_state_logits_equal': True})
        session.forward(torch.tensor([[32]]))
    atomic_json({'parent_unchanged': True, 'all_original_weights_equal': True,
                 'memory_disabled_logits_bitwise_equal': True, 'transitions': transitions}, out / 'session_verification.json')

    compact = {k: v for k, v in result.items() if k not in ('tests', 'graph_program_test')}
    compact['tests'] = []
    for test in result['tests']:
        compact['tests'].append({k: ({f: x for f, x in v.items() if f not in ('correct', 'prediction')}
                                     if k in ('memory', 'memory_disabled', 'memory_shuffled') else v)
                                 for k, v in test.items()})
    families = sorted({r['family'] for r in result['graph_program_test']['rows']})
    compact['graph_program_test'] = {'accuracy': result['graph_program_test']['accuracy'],
                                    'episodes': len(result['graph_program_test']['rows']),
                                    'families': {f: statistics.mean(r['correct'] for r in result['graph_program_test']['rows'] if r['family'] == f) for f in families}}
    means = {key: statistics.mean(t[key]['accuracy'] for t in result['tests'])
             for key in ('memory', 'memory_disabled', 'memory_shuffled')}
    a = [statistics.mean(t['memory']['correct'][i] for t in result['tests']) for i in range(600)]
    b = [statistics.mean(t['memory_disabled']['correct'][i] for t in result['tests']) for i in range(600)]
    compact['mean_accuracy'] = means
    datasets = json.loads((study / 'data.json').read_text())
    test_data = json.loads((study / 'test_data.json').read_text())
    def input_signature(row):
        return (row['prompt'], tuple(sorted((f['subject'], f['relation'], f['object']) for f in row['facts'])))
    seen_inputs = {input_signature(row) for row in datasets['train']}
    overlap = sum(input_signature(row) in seen_inputs for row in test_data)
    unseen_indices = [i for i, row in enumerate(test_data) if input_signature(row) not in seen_inputs]
    unseen_accuracy = statistics.mean(t['memory']['correct'][i] for t in result['tests'] for i in unseen_indices)
    compact['split_overlap'] = {'test_inputs_matching_training': overlap, 'test_count': len(test_data),
                               'mean_accuracy_on_unseen_test_inputs': unseen_accuracy,
                               'scope': 'Post hoc exact prompt/current-fact overlap audit, excluding provenance IDs. Independent sampling can repeat combinations; no checkpoint changes.'}
    compact['memory_minus_disabled'] = paired_interval(a, b)
    compact['full_results_sha256'] = sha256(study / 'results.json')
    atomic_json(compact, out / 'results.json')
    atomic_json({'checkpoint': str(destination), 'checkpoint_sha256': sha256(destination),
                 'source_sha256': source_identity(), 'parent_checkpoint': str(parent),
                 'parent_sha256': sha256(parent), 'config': asdict(model.config),
                 'selected_seed': selection['selected_seed'], 'scope': saved['scope']}, out / 'runtime.json')
    for source, name in ((study / 'selection.json', 'selection.json'), (study / 'plan.json', 'plan.json'),
                         (Path('runs/hybrid_install/smoke.json'), 'installation.json'),
                         (Path('runs/hybrid_install/activation_checkpointing.json'), 'activation_checkpointing.json'),
                         (Path('runs/hybrid_full_tests.log'), 'tests_root_real.log'),
                         (Path('runs/hybrid_spark_tests.log'), 'tests_spark.log'),
                         (Path('runs/local_energy_budget.json'), 'budget.json')):
        shutil.copy2(source, out / name)

    original_count = base.parameters_count()
    known_singletons = [t for t in transitions if not t['exact']['abstained']]
    correct_singletons = sum(chr(t['neural_next_byte']) in t['exact']['answers'] for t in known_singletons)
    lines = ['# Native hybrid integration', '',
             'The current native model now has exact fact storage, a learned reader in every block, '
             'bounded graph programs, and an optional PC-ALM training rule. BP remains the default. '
             'The full candidate-energy decoder was not promoted.', '',
             f'Adapted the existing **{original_count:,}-parameter checkpoint**, adding **{record["trainable_parameters"]:,} reader parameters**. '
             f'All original weights stayed bitwise unchanged. Three minibatch seeds each ran {record["updates"]} reader-only updates. '
             f'Validation selected seed **{selection["selected_seed"]}**, step **{record["best_step"]}**, before test evaluation.', '',
             'The original 30-minute run completed all training and froze selection, then reached its deadline during evaluation. '
             'A separate bounded evaluation continuation used the same source, checkpoints, and test data. '
             'It repeated no training updates and changed no selection decisions.', '',
             '## Measured integration', '',
             '| Native model condition | Mean test accuracy |', '|---|---:|',
             f'| Learned readers with current facts | {means["memory"]:.2%} |',
             f'| Same checkpoint, fact readers disabled | {means["memory_disabled"]:.2%} |',
             f'| Same checkpoint, facts shuffled across examples | {means["memory_shuffled"]:.2%} |', '',
             '![Reader training and memory-use ablation](reader_adaptation.png)', '',
             '**This is an access/usage ablation, not a fair algorithm comparison:** the disabled condition has no fact input. '
             'The task supplies the grammar `ar?` / `as?` / `br?` / `bs?`, six answer symbols, and event semantics. '
             'Examples independently sample assignments of objects to addresses. Test uses 600 independently generated examples; '
             'the fixed symbol vocabulary is shared across splits. These results establish learned use of the memory hook, '
             'not natural-language understanding or broad reasoning. '
             f'{overlap} of 600 test inputs exactly match a training prompt/current-fact combination; '
             'this is not a strictly disjoint-combination or novel-vocabulary benchmark. '
             f'A post hoc audit of the remaining {len(unseen_indices)} combinations gives {unseen_accuracy:.2%} mean accuracy, '
             'without changing any selected checkpoint.', '',
             '| Training seed | Selected step | Memory enabled | Disabled | Shuffled |', '|---|---:|---:|---:|---:|']
    for test, rec in zip(result['tests'], result['records']):
        lines.append(f'| {test["seed"]} | {rec["best_step"]} | {test["memory"]["accuracy"]:.2%} | {test["memory_disabled"]["accuracy"]:.2%} | {test["memory_shuffled"]["accuracy"]:.2%} |')
    lines += ['', '| Fact event | Mean accuracy with memory |', '|---|---:|']
    for family in ('set', 'replace', 'delete', 'hold'):
        lines.append(f'| {family} | {statistics.mean(t["groups"][family] for t in result["tests"]):.2%} |')
    lines += ['', f'The exact graph-program path scored **{compact["graph_program_test"]["accuracy"]:.2%}** on '
              f'**{compact["graph_program_test"]["episodes"]}** fresh diagnostics across 12 families, including missing multi-hop evidence, '
              'replacement, deletion, and distractors. It uses supplied typed relations and path programs. '
              'Branching intersection/union semantics, missing-premise abstention, and provenance are covered by unit tests.', '',
              '## Compatibility and local training', '',
              'Existing checkpoints retain their original layout with `memory_dim=0`. Explicit augmentation creates a separate '
              'weight-only checkpoint with closed reader gates; numerical tests verify original outputs exactly. '
              'After reader training, disabling memory still produces bitwise identical logits to the parent model. '
              'This does not guarantee unchanged text generation when the trained readers are enabled.', '',
              'The native PC-ALM option cuts credit propagation between blocks and keeps autograd inside each causal mixer. '
              'Activity iterations are detached and targets are used only during training. A linear equilibrium check matches '
              'the BP gradient; native nonlinear tests verify finite gradients and exact interrupted/resumed optimizer state. '
              'No new large native-LM comparison establishes a language-quality gain for PC-ALM.', '',
              '## Verification', '',
              '- **334 tests passed, 33 skipped:** 250 root/repaired tests plus 84 Spark tests.',
              '- A noneditable wheel imported outside the checkout; all 35 installed native-package source files matched.',
              '- The installed package passed graph execution, a native session forward pass, and PC-ALM backward.',
              '- Active-reader outputs and every parameter gradient matched exactly with and without activation checkpointing.',
              '- The selected full-size checkpoint passed persistent SET/replace/HOLD/DELETE session checks. Every revision reset matched a fresh forward pass.',
              '- Original checkpoint bytes, original weights, and earlier report bytes were preserved.',
              '- CPU only; reader training used the existing cumulative research ledger. No GPU training or efficiency claim.', '',
              'The readers add about **1.4%** to the original parameter count. Native recurrent state remains '
              '**148,992 bytes** per conversation; fact tensors add **1,028 bytes** for four facts or **65,792 bytes** '
              'at capacity 256. This is tensor accounting, excluding SQLite/provenance, activations, optimizer, and process overhead. '
              'See [footprint.json](footprint.json).', '',
              '## Use the integrated model', '',
              '[Selected native checkpoint](../../runs/hybrid_native/selected.pt) · '
              '[Runtime manifest](runtime.json) · [Architecture](ARCHITECTURE.md) · '
              '[Commands](../../energy_real/README.md#native-hybrid-architecture)', '',
              'Use `HybridSession.load(checkpoint, FactStore(path))` for memory-aware native inference and '
              '`session.answer(GraphProgram.path(subject, relations))` for exact graph answers. '
              'Existing explicit write authorization and revision checks remain in force. '
              'A committed revision discards old recurrent state, so applications may need to prefill current context again.', '',
              'The selected reader checkpoint is adapted to a narrow symbol-binding task. Use graph programs for evidence-backed '
              'fact answers; its free-form generated text is not guaranteed to follow the facts. '
              'Plain native byte training without aligned fact snapshots cannot train the new readers.', '',
              f'**Observed transfer limit:** in the one-fact persistent-session fixture, the neural decoder matched '
              f'**{correct_singletons}/{len(known_singletons)} known answers**, although graph execution returned all facts correctly. '
              'The reader was trained on three- and four-fact snapshots; its high test score does not cover this size change. '
              'A separate [supported-size session fixture](supported_session.json) records the neural and exact outputs '
              'for SET, replacement, HOLD, and DELETE with background facts retained. Both fixtures are preserved.', '',
              '[Machine results](results.json), [frozen selection](selection.json), '
              '[session evidence](session_verification.json), [installation evidence](installation.json), '
              '[artifact checksums](artifacts.json). Raw histories, predictions, and all three checkpoints remain in `runs/hybrid_native/`.', '',
              f'Core source: `{source_identity()}`.', '']
    (out / 'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'selected_seed': selection['selected_seed'], 'mean_accuracy': means,
                      'graph_accuracy': compact['graph_program_test']['accuracy'], 'total_parameters': model.parameters_count()}, indent=2))


if __name__ == '__main__':
    build()
