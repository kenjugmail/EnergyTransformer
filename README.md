# EnergyTransformer

Experimental energy-function memory, bounded text parsing, and explicitly reviewed
memory edits. **Not a competitive general-purpose LLM or instruction-following assistant.**

## V16: durable memory and portable checkpoints

`EnergyLedger` persists protected P/S coefficients in local SQLite. Commits require
an explicit reviewed edit, expected revision, authorization, and request identifier.
Concurrent stale writes fail; exact retries do not apply an edit twice. Audit replay
checks live coefficients against the journal. This is not distributed consensus,
authentication, cryptographic tamper proofing, or improved language understanding.

The existing three v14 sentence-parser/answer checkpoints can be exported to
safetensors plus JSON without changing tensor values. Their 16-name/16-location
scope and unfamiliar-phrasing failure remain unchanged. No weights were newly
trained in v16. All 67 local tests pass, including inherited tests.

[Architecture](ARCHITECTURE.md) | [V16 report](reports/V16_REPORT.md) |
[Storage and upload options](STORAGE.md) | [Releases](https://github.com/kenjugmail/EnergyTransformer/releases)

The existing `rebuilt-v14-34695580553` release contains three GitHub-runner reproductions.
The new portable-package workflow reuses those bytes and tests conversion; a workflow
file alone is not proof that publication finished. Read its run/release receipt.
**A portable checkpoint release is not the original full v1-v15 research archive.**

## Run

Download a checkpoint-bearing release, extract it, and enter `EnergyTransformer`:

```bash
pip install -r requirements.txt
python -m pytest -q tests
python scripts/demo_reviewed.py
python scripts/demo_durable.py
```

A source-only clone may lack checkpoints; related tests explicitly skip when absent.
Use `python scripts/train_bridge.py --seed 11` for a new bounded training reproduction,
not to claim recovery of historical weights. `scripts/export_portable.py` exports all
three checkpoints to a new directory and refuses to overwrite an existing export.

## Explicit durable edits

```python
from energy_transformer.bridge import EnergyTextSession
from energy_transformer.ledger import EnergyLedger
from energy_transformer.review import Edit
engine = EnergyTextSession('checkpoints/v14/bridge_seed11.pt')
store = EnergyLedger('memory.db', engine.signature)
state, revision = store.read()
# Obtain actual external review and authorization before calling commit.
receipt = store.commit([Edit('set', 'alice', 'garden')],
    expected_revision=revision, request_id='reviewed-request-001', authorized=True)
print(store.audit())
```

Do not automatically copy parser predictions into authorized edits. A Boolean flag
is not an authentication layer. Journal history retains deleted values; deletion
only clears the live association. A full database rollback is not detected without
an external trusted reference. The memory payload remains 1,088 raw coefficient
bytes, but the database, metadata, and journal add real storage that grows.

## Publication boundary

GitHub Releases can host the roughly 300 MB original collection. Hugging Face is
optional. `scripts/publish_release.py` handles a supplied local archive; the optional
`scripts/publish_hf.py` requires an explicitly named existing PRIVATE HF repository.
Both default to validation without writes and verify re-downloaded bytes on upload.
No token belongs in chat or this repository. No visibility or license is changed.

The connected source API cannot transfer the large local archive, and direct network
access from the working container is blocked. The full original collection is preserved
in downloadable packages, not represented as uploaded merely because its manifest is
on GitHub. Prior results and failed controls retain their original qualifications.
See [rights](RIGHTS.md), [model card](MODEL_CARD.md), and [history](reports/HISTORY.md).
