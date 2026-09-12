# V18: precision-aware energy-memory editing

This release returns to the numerical memory mechanism. It does not modify the existing text parser, approvals, ledger, or language-model weights.

## Run

```
pip install -r requirements-research.txt
python scripts/demo_recall.py
python -m pytest -q tests/test_geometric_memory.py
```

The demo requires `checkpoints/v18/projected_delayed_seed11.pt` from the accompanying experiment download. The GitHub source commit does not imply that the binary archive has been transferred.

To reproduce all 21 new small fits, evaluations, three algebraic conversions, and numerical audits:

```
python scripts/run_v18.py
```

CPU is the tested environment. `projected` has 756 learned scalars and stores 768 bytes of P/S per episode in float32. It is a continuous-key associative module, not a chat model. The task supplies operation flags and the raw query keys; it is not natural-language fact extraction. RESET means a zero-valued association, not semantic deletion or UNKNOWN.

Read `reports/V18_REPORT.md` for all results, the equivalent classical control, stronger baselines, capacity failure, and full mathematical derivation. `reports/RESEARCH_v18.md` separates newly implemented mechanisms from proposed next experiments. Exact-key table recall wins this supplied-record task with zero error, and a small softmax comparator has a better overloaded result and much faster batch processing.

The previous parser and approved-memory protections remain unchanged. No unattended writes or source-visibility changes are enabled by this experiment.
