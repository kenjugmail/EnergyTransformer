# EnergyTransformer

Research on recurrent memory represented by energy-function coefficients. The trained byte language models, spectral experiments, controls, and negative results are preserved in the full research bundle. This repository is not a competitive general-purpose LLM.

## Available now

[Download rebuilt bounded checkpoints](https://github.com/kenjugmail/EnergyTransformer/releases/tag/rebuilt-v14-34695580553) (requires repository access). GitHub's runner trained all three seeds, passed 43 tests, and published the ZIP plus its checksum. These are new CPU reproductions, not the original historical files. [Publication receipt](reports/CI_PUBLICATION_v15.json).

## Current continuation: reviewed memory edits (v15)

The v14 sentence parser fails on unfamiliar wording. V15 therefore separates **proposing** a memory edit from **executing** an explicitly reviewed, authorized edit. This is a deployment correction, not a new semantic-generalization result. Original models and benchmark results are unchanged.

```python
from energy_transformer.review import Edit, ReviewedEnergySession
s = ReviewedEnergySession('checkpoints/v14/bridge_seed11.pt')
p = s.propose('Alice is in the garden.')  # does not edit memory
print(p)                                # inspect the proposed meaning
# A real application must obtain independent review/authorization here.
if p['status'] == 'needs_review':
    s.approve(p['proposal']['proposal_id'],
              reviewed_edit=Edit('set', 'alice', 'garden'),
              expected_revision=p['proposal']['revision'], authorized=True)
print(s.answer('alice'))
```

Never implement review by automatically copying a parser prediction into `reviewed_edit`. The hosting application, not this library, is responsible for authenticating and authorizing users. Incorrectly approved edits can still corrupt memory.

## Run

```bash
pip install -r requirements.txt
python scripts/train_bridge.py --seed 11  # only if no checkpoint is present
python scripts/demo_reviewed.py
python -m pytest -q tests
```

The complete downloadable bundle includes all three original v14 checkpoints. Source-only clones may not: model-specific tests explicitly skip if those files are absent. CI checkpoint rebuilds, when available under Releases, are **new reproductions**, not byte-identical uploads of the original artifacts.

## What is implemented

- V14: learned parsing of a small sentence grammar into a 16-name/16-location protected energy store, plus a separately trained bounded byte-answer decoder.
- V15: proposals that never auto-write, explicit structured approval, revision/state binding, atomic edits, pending-proposal limits, and checkpoint-bound snapshots.
- Reproduction scripts and tests; all earlier results retain their original limitations.

The old unrestricted byte decoder is not jointly trained with this text bridge. The new wrapper does not fix unfamiliar-language parsing or establish general factual reasoning.

## Full artifact publication

`reports/ARTIFACT_INDEX_v15.json` lists **14 original experiment archives and 55 standalone artifacts**, with byte counts and SHA-256 hashes. This is an inventory of the supplied research bundle, NOT a claim that all those bytes are already on GitHub.

Direct Git access from the research container failed. The connected API can publish source, while the remaining original binaries need a bulk transfer. From a normally connected machine with an authenticated GitHub CLI:

```bash
python scripts/publish_release.py \
  --archive /path/to/EnergyTransformer_All_Research_v1_v15.zip --publish
```

The uploader checks the archive, refuses a non-private target, never clobbers existing assets, verifies uploaded bytes by downloading and hashing them, and publishes only after verification. It never changes visibility or licensing. GitHub release storage avoids putting the entire historical binary collection into Git history. The older `scripts/publish_all.py --push` remains for users deliberately choosing Git storage instead.

## Documentation

- [Architecture](ARCHITECTURE.md)
- [Original bounded text-bridge results](reports/V14_REPORT.md)
- [Reviewed-edit and publication continuation](reports/V15_REPORT.md)
- [Model limitations](MODEL_CARD.md)
- [Historical experiments](reports/HISTORY.md)
- [Artifact inventory](reports/ARTIFACT_INDEX_v15.json)
- [Rights and third-party boundaries](RIGHTS.md)

All scores from the small news corpora are development evidence. No claim of universal Transformer superiority, unlimited memory, or established fractal reasoning is made.
