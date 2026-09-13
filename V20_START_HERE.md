# V20: supervised token-derived writes

Read reports/V20_SUMMARY.md first. A supplied32-address symbolic task is solved with training-only write labels; changed layout and general-pretraining readiness are not solved. Existing deployed models and authorization interfaces are unchanged.

```
pip install -r requirements-router.txt
mkdir -p results/v20
python scripts/run_v20.py
python scripts/evaluate_router.py
python scripts/assess_router.py
python scripts/demo_router.py
python -m pytest -q tests/test_write_router.py
```

The conversation download already includes every trained checkpoint, so run_v20.py is unnecessary for its demo. A source-only GitHub checkout must train first. Training is CPU-only by default, bounded, and does not provision remote hardware. All twelve fits are needed before the complete evaluator runs.

The downloaded full worktree additionally includes the full report, prior v19 source, optimizer/scan and fresh-process audits, raw test inputs/predictions, source hashes and timing repetitions. Only source and small reports/results are published in this commit, not checkpoint binaries or the complete historical archive.

The state includes a three-embedding local buffer plus P/S. It is not a new claim that all activation buffers or all latent variables disappear. These tiny symbolic models are not chat assistants.
