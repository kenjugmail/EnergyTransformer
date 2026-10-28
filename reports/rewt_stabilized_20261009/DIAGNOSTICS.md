# Follow-up diagnostics

These summaries consume saved outputs after the experiment completes. They do not change model weights or select training settings. The current source hashes and snapshot match the frozen restart protocol. All twenty-seven reused checkpoint bytes and original training records also match their separate parent provenance. Language fits use the same declared byte budget; evaluation windows/prefixes and synthetic identities exclude both prior runs.

Local refinement freezes each original prefix's conditioning. Negative bpb change means local refinement improves prediction. Common-gradient share is a share of squared norm. Energy monotonicity is checked separately from prediction quality.

| Variant | Initial common gradient share % | Two-round bpb change | Eight-round bpb change | Gradient / CE cosine | First-order CE descent % |
|---|---:|---:|---:|---:|---:|
| Raw REWT control | 58.543261 | -0.000461 | -0.001335 | -0.0036 | 39.2 |
| Revised neural energy | 0.000000 | -0.000762 | -0.001273 | -0.0034 | 42.8 |
| Revised Fourier REWT | 0.000000 | -0.000788 | -0.001291 | -0.0036 | 43.2 |
| Revised full REWT | 0.000000 | -0.002180 | -0.003628 | +0.0012 | 50.1 |

Reasoning breakdowns below average identical task groups across seeds. Short subgroups have limited precision; withheld wording and longer chains test different forms of generalization. Recall is reported separately from arithmetic and rules.

| Variant | Family | Steps/delay | Withheld wording | Tasks per seed | Exact answer % | Complete % |
|---|---|---:|---|---:|---:|---:|
| Transformer | arithmetic | 1 | False | 6 | 0.0 | 0.0 |
| Transformer | arithmetic | 1 | True | 3 | 0.0 | 0.0 |
| Transformer | arithmetic | 2 | False | 19 | 7.0 | 0.0 |
| Transformer | arithmetic | 2 | True | 11 | 0.0 | 0.0 |
| Transformer | arithmetic | 3 | False | 22 | 0.0 | 0.0 |
| Transformer | arithmetic | 3 | True | 11 | 3.0 | 0.0 |
| Transformer | arithmetic | 4 | False | 19 | 0.0 | 0.0 |
| Transformer | arithmetic | 4 | True | 9 | 0.0 | 0.0 |
| Transformer | recall | 1 | True | 1 | 0.0 | 0.0 |
| Transformer | recall | 4 | False | 18 | 100.0 | 98.1 |
| Transformer | recall | 4 | True | 14 | 2.4 | 0.0 |
| Transformer | recall | 8 | False | 19 | 100.0 | 94.7 |
| Transformer | recall | 8 | True | 13 | 12.8 | 5.1 |
| Transformer | recall | 16 | False | 21 | 82.5 | 73.0 |
| Transformer | recall | 16 | True | 14 | 7.1 | 2.4 |
| Transformer | rules | 1 | False | 18 | 7.4 | 7.4 |
| Transformer | rules | 1 | True | 6 | 0.0 | 0.0 |
| Transformer | rules | 2 | False | 18 | 5.6 | 0.0 |
| Transformer | rules | 2 | True | 6 | 0.0 | 0.0 |
| Transformer | rules | 3 | False | 21 | 4.8 | 0.0 |
| Transformer | rules | 3 | True | 7 | 0.0 | 0.0 |
| Transformer | rules | 4 | False | 17 | 5.9 | 0.0 |
| Transformer | rules | 4 | True | 7 | 0.0 | 0.0 |
| Raw REWT control | arithmetic | 1 | False | 6 | 0.0 | 0.0 |
| Raw REWT control | arithmetic | 1 | True | 3 | 0.0 | 0.0 |
| Raw REWT control | arithmetic | 2 | False | 19 | 0.0 | 0.0 |
| Raw REWT control | arithmetic | 2 | True | 11 | 0.0 | 0.0 |
| Raw REWT control | arithmetic | 3 | False | 22 | 4.5 | 0.0 |
| Raw REWT control | arithmetic | 3 | True | 11 | 0.0 | 0.0 |
| Raw REWT control | arithmetic | 4 | False | 19 | 0.0 | 0.0 |
| Raw REWT control | arithmetic | 4 | True | 9 | 0.0 | 0.0 |
| Raw REWT control | recall | 1 | True | 1 | 0.0 | 0.0 |
| Raw REWT control | recall | 4 | False | 18 | 5.6 | 5.6 |
| Raw REWT control | recall | 4 | True | 14 | 4.8 | 4.8 |
| Raw REWT control | recall | 8 | False | 19 | 7.0 | 7.0 |
| Raw REWT control | recall | 8 | True | 13 | 2.6 | 2.6 |
| Raw REWT control | recall | 16 | False | 21 | 11.1 | 11.1 |
| Raw REWT control | recall | 16 | True | 14 | 7.1 | 7.1 |
| Raw REWT control | rules | 1 | False | 18 | 1.9 | 1.9 |
| Raw REWT control | rules | 1 | True | 6 | 5.6 | 0.0 |
| Raw REWT control | rules | 2 | False | 18 | 7.4 | 0.0 |
| Raw REWT control | rules | 2 | True | 6 | 0.0 | 0.0 |
| Raw REWT control | rules | 3 | False | 21 | 1.6 | 0.0 |
| Raw REWT control | rules | 3 | True | 7 | 4.8 | 0.0 |
| Raw REWT control | rules | 4 | False | 17 | 5.9 | 0.0 |
| Raw REWT control | rules | 4 | True | 7 | 14.3 | 0.0 |
| Revised neural energy | arithmetic | 1 | False | 6 | 0.0 | 0.0 |
| Revised neural energy | arithmetic | 1 | True | 3 | 0.0 | 0.0 |
| Revised neural energy | arithmetic | 2 | False | 19 | 3.5 | 0.0 |
| Revised neural energy | arithmetic | 2 | True | 11 | 3.0 | 0.0 |
| Revised neural energy | arithmetic | 3 | False | 22 | 1.5 | 0.0 |
| Revised neural energy | arithmetic | 3 | True | 11 | 0.0 | 0.0 |
| Revised neural energy | arithmetic | 4 | False | 19 | 0.0 | 0.0 |
| Revised neural energy | arithmetic | 4 | True | 9 | 3.7 | 0.0 |
| Revised neural energy | recall | 1 | True | 1 | 0.0 | 0.0 |
| Revised neural energy | recall | 4 | False | 18 | 5.6 | 5.6 |
| Revised neural energy | recall | 4 | True | 14 | 4.8 | 4.8 |
| Revised neural energy | recall | 8 | False | 19 | 7.0 | 7.0 |
| Revised neural energy | recall | 8 | True | 13 | 7.7 | 7.7 |
| Revised neural energy | recall | 16 | False | 21 | 11.1 | 11.1 |
| Revised neural energy | recall | 16 | True | 14 | 2.4 | 2.4 |
| Revised neural energy | rules | 1 | False | 18 | 7.4 | 0.0 |
| Revised neural energy | rules | 1 | True | 6 | 0.0 | 0.0 |
| Revised neural energy | rules | 2 | False | 18 | 5.6 | 0.0 |
| Revised neural energy | rules | 2 | True | 6 | 0.0 | 0.0 |
| Revised neural energy | rules | 3 | False | 21 | 6.3 | 0.0 |
| Revised neural energy | rules | 3 | True | 7 | 0.0 | 0.0 |
| Revised neural energy | rules | 4 | False | 17 | 7.8 | 0.0 |
| Revised neural energy | rules | 4 | True | 7 | 0.0 | 0.0 |
| Revised Fourier REWT | arithmetic | 1 | False | 6 | 5.6 | 0.0 |
| Revised Fourier REWT | arithmetic | 1 | True | 3 | 0.0 | 0.0 |
| Revised Fourier REWT | arithmetic | 2 | False | 19 | 3.5 | 0.0 |
| Revised Fourier REWT | arithmetic | 2 | True | 11 | 9.1 | 0.0 |
| Revised Fourier REWT | arithmetic | 3 | False | 22 | 0.0 | 0.0 |
| Revised Fourier REWT | arithmetic | 3 | True | 11 | 3.0 | 0.0 |
| Revised Fourier REWT | arithmetic | 4 | False | 19 | 3.5 | 0.0 |
| Revised Fourier REWT | arithmetic | 4 | True | 9 | 0.0 | 0.0 |
| Revised Fourier REWT | recall | 1 | True | 1 | 33.3 | 33.3 |
| Revised Fourier REWT | recall | 4 | False | 18 | 5.6 | 5.6 |
| Revised Fourier REWT | recall | 4 | True | 14 | 11.9 | 11.9 |
| Revised Fourier REWT | recall | 8 | False | 19 | 10.5 | 10.5 |
| Revised Fourier REWT | recall | 8 | True | 13 | 5.1 | 5.1 |
| Revised Fourier REWT | recall | 16 | False | 21 | 14.3 | 14.3 |
| Revised Fourier REWT | recall | 16 | True | 14 | 11.9 | 11.9 |
| Revised Fourier REWT | rules | 1 | False | 18 | 3.7 | 0.0 |
| Revised Fourier REWT | rules | 1 | True | 6 | 0.0 | 0.0 |
| Revised Fourier REWT | rules | 2 | False | 18 | 5.6 | 1.9 |
| Revised Fourier REWT | rules | 2 | True | 6 | 0.0 | 0.0 |
| Revised Fourier REWT | rules | 3 | False | 21 | 9.5 | 0.0 |
| Revised Fourier REWT | rules | 3 | True | 7 | 0.0 | 0.0 |
| Revised Fourier REWT | rules | 4 | False | 17 | 9.8 | 0.0 |
| Revised Fourier REWT | rules | 4 | True | 7 | 0.0 | 0.0 |
| Revised full REWT | arithmetic | 1 | False | 6 | 0.0 | 0.0 |
| Revised full REWT | arithmetic | 1 | True | 3 | 0.0 | 0.0 |
| Revised full REWT | arithmetic | 2 | False | 19 | 1.8 | 0.0 |
| Revised full REWT | arithmetic | 2 | True | 11 | 3.0 | 0.0 |
| Revised full REWT | arithmetic | 3 | False | 22 | 4.5 | 0.0 |
| Revised full REWT | arithmetic | 3 | True | 11 | 3.0 | 0.0 |
| Revised full REWT | arithmetic | 4 | False | 19 | 0.0 | 0.0 |
| Revised full REWT | arithmetic | 4 | True | 9 | 3.7 | 0.0 |
| Revised full REWT | recall | 1 | True | 1 | 0.0 | 0.0 |
| Revised full REWT | recall | 4 | False | 18 | 11.1 | 11.1 |
| Revised full REWT | recall | 4 | True | 14 | 16.7 | 16.7 |
| Revised full REWT | recall | 8 | False | 19 | 8.8 | 8.8 |
| Revised full REWT | recall | 8 | True | 13 | 7.7 | 7.7 |
| Revised full REWT | recall | 16 | False | 21 | 6.3 | 6.3 |
| Revised full REWT | recall | 16 | True | 14 | 2.4 | 2.4 |
| Revised full REWT | rules | 1 | False | 18 | 5.6 | 1.9 |
| Revised full REWT | rules | 1 | True | 6 | 0.0 | 0.0 |
| Revised full REWT | rules | 2 | False | 18 | 13.0 | 0.0 |
| Revised full REWT | rules | 2 | True | 6 | 0.0 | 0.0 |
| Revised full REWT | rules | 3 | False | 21 | 7.9 | 0.0 |
| Revised full REWT | rules | 3 | True | 7 | 0.0 | 0.0 |
| Revised full REWT | rules | 4 | False | 17 | 5.9 | 0.0 |
| Revised full REWT | rules | 4 | True | 7 | 4.8 | 0.0 |

Training-length comparisons use validation learning curves at the declared quarter and full byte budgets. The longer budget changes both data seen and optimizer steps. The main raw/revised comparison also bundles several repairs, so neither curve establishes a single-component causal effect.
