# Local learning, energy modules, and full energy predictors

Completed 12 fits on CPU; all models trained from scratch. This is a bounded prototype study, not a reproduction of the Sakana results or a general-purpose language model.

All checkpoint and temperature selections used validation only and were frozen before test evaluation. Training seeds are paired within each architecture. Initializations remain eligible for selection.

## bytes

| Method | Hidden layers | Memory | Initial accuracy | Trained accuracy | Bits/target | Parameters | CPU ms/example |
|---|---:|---|---:|---:|---:|---:|---:|
| bp | 3 | no | 0.23% | 15.30% | 5.928 | 12,640 | 0.091 |
| bp_ce | 3 | no | 0.23% | 18.90% | 4.472 | 12,640 | 0.088 |
| pcalm | 3 | no | 0.23% | 13.93% | 6.623 | 12,640 | 0.079 |
| energy | 3 | no | 0.10% | 18.60% | 4.682 | 12,640 | 25.108 |

## Interpretation limits

- BP, PC, and PC-ALM use the same MSE objective and architecture. BP-CE is an additional byte likelihood control.
- The full EBM uses candidate energy minimization and a contrastive cross-entropy objective plus 0.1 positive energy. Its objective differs from PC-ALM.
- Finite EBM solves use approximate envelope gradients; stationarity is measured. They do not backpropagate through the solver.
- The graph writer and event semantics are supplied. Memory readers are learned. All neural arms receive the complete current fact table.
- The discrete graph control has supplied rules, no trained perception, and no learned hidden vector state.
- The byte model predicts one of all 256 bytes from eight preceding bytes. Its tiny corpus windows and test count do not support broad language claims.
- Reported intervals resample paired examples after averaging three seeds. Per-seed effects are reported separately; this is not a population estimate over many training seeds.
- Latencies are CPU implementation measurements. Different inference budgets and candidate enumeration are charged explicitly; no hardware efficiency promotion is made.
