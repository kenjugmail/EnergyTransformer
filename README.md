# EnergyTransformer

Energy-function recurrent memory, selective retention, and learned text interfaces. Experimental research by Kenju Tomita with implementation and analysis developed in this conversation.

## Current result: v14 text-to-memory bridge

A **73,741-parameter** sentence parser and bounded byte-answer decoder were trained at three seeds. Text is mapped to a proposed write/delete/hold/query, protected quadratic coefficients retain the association, and a learned decoder reads that memory to produce a location name or `unknown`.

```text
sentence -> learned operation/entity/value -> protected P/S memory
question -> learned entity query -> energy read -> learned answer bytes
```

**This is not a general-purpose chat model.** It supports 16 named entities and 16 locations, with supplied sentence boundaries and strong synthetic slot supervision. The older byte LM is not newly instruction-tuned or jointly trained with this bridge.

| Test | Result |
|---|---:|
| In-grammar held-out sentence parse | 100% across 3 seeds |
| In-grammar recall after 1,024 distractors | 100% |
| Same learned parser with ordinary dictionary | 100% (matched control) |
| Parse accuracy on held-out wording | 57.8% mean |
| Guarded recall after 1,024 new-wording distractors | 1.04% |

The new-wording failure is significant: high confidence does not prevent incorrect edits. **Writes require explicit `trusted=True`; this flag is a caller-supplied trust boundary, not proof that the model understood a sentence.** Do not deploy unattended on arbitrary text.

## Run

```bash
pip install -r requirements.txt
python scripts/train_bridge.py --seed 11
python scripts/demo.py
python -m pytest -q tests
```

```python
from energy_transformer import EnergyTextSession
s = EnergyTextSession('checkpoints/v14/bridge_seed11.pt')
s.handle('Alice is in the garden.', trusted=True)
print(s.handle('Where is Alice?'))  # bounded learned answer: garden
```

Train seeds 22 and 33, then run `python scripts/evaluate_bridge.py` to reproduce the complete v14 experiment. Training needs no external corpus. Tests requiring weights are skipped when no checkpoint exists. Full local testing with weights passed 20 tests.

## Publication contents

The API-published checkout contains the new runnable source, tests, results summary, architecture, and publication tooling. **Large historical experiment ZIPs and trained binary checkpoints are not included in that initial API upload.** The accompanying full research package contains all 13 original experiment archives, historical standalone reports/results, and v14 checkpoints and raw evaluations. Its archive manifest records sizes and SHA-256 checksums.

To complete the bulk upload from a machine with authorized Git access, extract the full package and run:

```bash
python scripts/publish_all.py --push
```

This clones this repository, overlays the package, makes a normal commit, and pushes without force. It does not change repository visibility. Until that completes, do not describe the remote repository as containing every artifact.

## Read first

- `ARCHITECTURE.md`: equations, interfaces, and implementation boundaries.
- `reports/V14_REPORT.md`: training, controls, failures, and reproducibility.
- `reports/HISTORY.md`: v1-v13 lineage; historical measurements are not new runs.
- `reports/ARCHIVE_MANIFEST.json`: exact inventory of the original archives.
- `MODEL_CARD.md`: use limitations and data scope.
- `RIGHTS.md`: no new blanket license is imposed; third-party terms remain applicable.

The repository name does not imply affiliation with other published Energy Transformer projects. No universal superiority to Transformer attention, lossless unlimited context, fractal reasoning, or calibrated factual confidence is established.
