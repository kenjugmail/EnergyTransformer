# V17: compositional proposals and energy-native edit research

Executed September 12, 2026. Twelve new CPU parser fits (2 architectures x 2 curricula x 3 seeds), 150 aggregate parser records, 2,160 episodic records, and 89 passing local tests. No old language-model weights, answer decoder, default parser, reviewed authorization or ledger behavior were changed.

## Controlled sentence results

Mean exact semantic interpretation accuracy across three seeds:

| Method | Novel compositions | Withheld entity/value pairs | Registered new spellings | Legacy unfamiliar wording |
|---|---:|---:|---:|---:|
| Joint parser, basic templates | 68.52% | 77.02% | 23.57% | 56.71% |
| Binding parser, basic templates | 75.85% | 88.31% | 86.78% | 71.65% |
| Joint parser, broader training | 99.28% | 99.90% | 28.78% | 67.12% |
| Binding parser, broader training | 99.64% | 100.00% | 100.00% | 62.63% |

Most of the composition improvement comes from richer training coverage. The binding model uses a supplied registry to replace the one entity/value with canonical symbols and learns the operation. It has 50,804 parameters versus 55,108 for the broader joint model. Registry renaming makes operation inputs identical; it is not learned open-world entity recognition. The namespace remains 16 slots. The old answer decoder is not trained to emit the alias words.

Training: 900 updates of 64 examples per model, or 57,600 presented examples sampled from 8,192 training rows. Basic and expanded datasets contain 2,296 and 6,150 unique strings and 38/726 observed templates. Batch indices are paired within curriculum and seed. Validation selects checkpoints; both expanded families reach 100% mean validation accuracy. Their differently defined training losses are not cross-model quality comparisons. No final test tuning was performed.

Primary composition test: 1,024 unique records across 139 excluded template combinations. Other conditions: 512 IID, 1,024 withheld-pair SET, 512 registered-renaming records, and 1,024 legacy wording samples (530 unique strings, 12 templates). The legacy test has been used in prior project versions. Synthetic grammar coverage is not unrestricted language generalization.

## Failure and review boundaries

The unfamiliar-wording result remains poor. Strict unknown-vocabulary rejection accepts only 36.49% of legacy records and still has 15.19% mean error among accepted interpretations. It is not a correctness certificate. BindingPredictor NEVER mutates or authorizes memory. Existing reviewed edits remain mandatory.

Automatic application is simulated only in isolated evaluations. After 1,024 novel-composition distractors, the proposed parser preserves 75% of queried facts, versus 63.54% for the equally broadened joint parser and 6.25% for the basic joint control. Only 66.67% of proposed-model seed-episode cases preserve all four facts. The proposed model produces 116 wrong edits in these 24 seed-episode cases. A conventional dictionary with exactly the same predictions gives identical answers in every evaluation.

The episodic harness has 24 base episodes (8 each for IID/composition/legacy) at 3 exact prefix lengths, yielding 72 cases. Initial facts are supplied structured records and final keys are explicit queries; this is not end-to-end question answering. An initial unpaired-prefix diagnostic was corrected after inspection; all policies were replayed, and both runs are retained in the downloadable research archive. No model weights or primary parser scores changed.

## Separate energy primitive

quadratic_edit.py adds a separately tested full-quadratic update:

    A = I - alpha e e^T
    P_new = lambda I + A^T(P-lambda I)A + beta k k^T
    S_new = A^T S + beta k v^T

Here e and k can differ. Positive-semidefinite congruence and a nonnegative write keep P_new >= lambda I when P >= lambda I. This is an explicit energy construction, not EDA or Gated DeltaNet-2 reproduction. It is not language-trained or connected to the deployed recurrence, requires dense precision storage, and does not guarantee unrelated inferred modes remain unchanged for correlated keys.

## Reproduction

```
pip install -r requirements-research.txt
python -m energy_transformer.binding_data
mkdir -p checkpoints/v17
for seed in 11 22 33; do
  for curriculum in basic expanded; do
    for kind in joint bound; do
      python scripts/train_binding.py --kind "$kind" --curriculum "$curriculum" --seed "$seed"
    done
  done
done
python scripts/demo_binding.py
python scripts/evaluate_binding.py
python scripts/evaluate_binding_episodes.py
python -m pytest -q tests
```

The evaluation scripts also require the three inherited v14 checkpoints from the existing private release. The downloadable v17 archive includes these plus all twelve original newly trained checkpoints and raw results. Git source publication does not itself upload those binary artifacts.

See RESEARCH_v17.md for primary-source research and the next matched experiments. Proposed priorities are compositional span/role handling, content-directed energy edits, delayed-query training, and causal write interventions. No Transformer-superiority, fractal-reasoning, or general-assistant claim is supported by this release.
