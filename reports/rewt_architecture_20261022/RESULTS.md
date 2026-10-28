# Measured results

## Reasoning: complete response accuracy (%)

| Variant | Parameters | Validation mean | Test mean | Test seed range | CPU training seconds, mean |
|---|---:|---:|---:|---:|---:|
| control | 256,096 | 34.156 | 34.568 | 33.951–35.802 | 804.8 |
| certified | 256,096 | 33.539 | 33.128 | 31.173–34.877 | 724.8 |
| precond2 | 256,096 | 34.362 | 34.979 | 33.025–37.346 | 681.8 |
| precond4 | 256,096 | 34.362 | 34.568 | 33.951–35.802 | 719.2 |
| precond8 | 256,096 | 34.156 | 34.362 | 33.951–34.568 | 860.1 |
| scan | 246,688 | 34.568 | 34.465 | 33.333–35.185 | 166.3 |
| feedback8 | 249,040 | 33.745 | 35.185 | 34.568–36.111 | 213.3 |
| delta | 257,537 | 35.391 | 35.700 | 34.877–36.420 | 183.1 |
| patch4 | 246,688 | 34.979 | 35.802 | 35.185–36.420 | 151.7 |
| diversity | 256,096 | 34.156 | 35.391 | 34.877–36.111 | 657.6 |
| hybrid_modes | 256,096 | 34.156 | 33.951 | 33.333–34.877 | 814.0 |
| lowrank16 | 208,800 | 35.185 | 33.951 | 33.642–34.259 | 201.5 |
| combined | 259,889 | 36.008 | 35.288 | 34.877–35.802 | 247.2 |

Variant selected by mean validation accuracy: **combined**. This choice does not use test outcomes.

## Language: bits per original byte (lower is better)

| Variant | Parameters | Validation mean | Test mean | Test seed range | CPU training seconds, mean |
|---|---:|---:|---:|---:|---:|
| control | 256,096 | 2.088 | 2.082 | 2.080–2.086 | 695.2 |
| precond4 | 256,096 | 2.087 | 2.082 | 2.079–2.086 | 695.8 |
| scan | 246,688 | 2.101 | 2.093 | 2.084–2.098 | 92.9 |
| feedback8 | 249,040 | 2.101 | 2.093 | 2.084–2.098 | 125.3 |
| delta | 257,537 | 2.073 | 2.065 | 2.060–2.069 | 112.3 |
| patch4 | 246,688 | 2.102 | 2.094 | 2.084–2.099 | 91.2 |
| lowrank16 | 208,800 | 2.288 | 2.278 | 2.264–2.288 | 92.8 |
| combined | 259,889 | 2.075 | 2.067 | 2.057–2.076 | 166.0 |

Variant selected by mean validation bits per byte: **delta**.

## Initial learned memory recipe: accuracy (%)

| Variant | Parameters | Validation mean | Test mean | Test seed range | CPU training seconds, mean |
|---|---:|---:|---:|---:|---:|
| local | 32,752 | 52.604 | 54.167 | 51.042–56.250 | 29.4 |
| delta | 34,321 | 43.490 | 34.462 | 27.604–40.365 | 38.2 |

## Binding-supervised memory follow-up: accuracy (%)

| Variant | Parameters | Validation mean | Test mean | Test seed range | CPU training seconds, mean |
|---|---:|---:|---:|---:|---:|
| local | 32,752 | 55.122 | 54.774 | 54.427–55.208 | 54.6 |
| delta | 34,321 | 99.479 | 96.007 | 94.792–97.135 | 73.8 |

| Query placement | Local | Delta |
|---|---:|---:|
| 16/False | 10.42% | 98.44% |
| 16/True | 100.00% | 100.00% |
| 32/False | 7.29% | 94.79% |
| 32/True | 100.00% | 100.00% |
| 64/False | 10.94% | 82.81% |
| 64/True | 100.00% | 100.00% |

`False` means a far query; `True` means a seven-byte gap. Record counts 16/32/64 correspond to far gaps of 47/95/191 bytes.

## Matched memory supervision ablation: accuracy (%)

| Variant | Parameters | Validation mean | Test mean | Test seed range | CPU training seconds, mean |
|---|---:|---:|---:|---:|---:|
| local_response | 32,752 | 54.861 | 54.688 | 53.906–55.208 | 48.8 |
| delta_response | 34,321 | 59.809 | 37.674 | 34.115–44.792 | 63.6 |
| local_binding | 32,752 | 55.122 | 54.948 | 53.906–56.250 | 54.6 |
| delta_binding | 34,321 | 99.479 | 95.747 | 94.271–97.917 | 73.8 |

Same three seeds, 1,536 updates, training batches and validation selection; a third fresh test set. The six binding-supervised checkpoints are reused without further training.

| Query placement | local_response | local_binding | delta_response | delta_binding |
|---|---:|---:|---:|---:|
| 16/False | 12.50% | 7.29% | 16.15% | 97.40% |
| 16/True | 97.92% | 100.00% | 76.04% | 100.00% |
| 32/False | 9.38% | 13.02% | 12.50% | 93.23% |
| 32/True | 98.96% | 100.00% | 59.38% | 100.00% |
| 64/False | 9.38% | 9.38% | 12.50% | 83.85% |
| 64/True | 100.00% | 100.00% | 49.48% | 100.00% |

## Equal measured CPU training budget: language bits/byte

| Variant | Parameters | Validation mean | Test mean | Test seed range | CPU training seconds, mean |
|---|---:|---:|---:|---:|---:|
| precond2 | 256,096 | 2.867 | 2.886 | 2.856–2.910 | 150.6 |
| control | 256,096 | 2.888 | 2.907 | 2.889–2.931 | 150.4 |
| precond4 | 256,096 | 2.896 | 2.914 | 2.906–2.929 | 150.4 |
| precond8 | 256,096 | 2.936 | 2.954 | 2.936–2.974 | 150.5 |
| precond4_structured | 256,096 | 2.856 | 2.874 | 2.861–2.894 | 150.6 |

150 training process seconds per fit, at most one update overshoot; validation and checkpoint I/O excluded. Same three new seeds and per-update batches, five solver paths, four validation checkpoints. Another 256-document test set excludes the earlier language test documents.

| Solver | Mean updates | Maximum measured training CPU seconds |
|---|---:|---:|
| precond2 | 117.0 | 150.979 |
| control | 112.7 | 150.653 |
| precond4 | 111.7 | 150.667 |
| precond8 | 104.7 | 150.959 |
| precond4_structured | 118.3 | 150.607 |
