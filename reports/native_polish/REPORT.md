# Native-model usability and reliability follow-up

The trained architecture and checkpoint weights are unchanged. This pass fixes
early training validation and makes native sampling available from the installed
package. The completed [competitive study](../competitive/REPORT.md) and its
checkpoints remain preserved; no models were retrained or new quality gains claimed.

## Changes

- Invalid layer counts, widths, head counts and hybrid attention intervals now
  produce a validation error before parameter-matching arithmetic, GPU checks,
  corpus access or output-directory creation.
- `python -m energy_real.sample` works outside the checkout. Greedy decoding is
  the default; temperature, top-k and a local seed enable reproducible sampling.
  Sampling preserves model weights, training mode and the caller's global RNG.
- Prompt processing defaults to 256-byte chunks. This bounds per-call activations
  and logits; a full-attention model still keeps its growing KV cache.
- `--output` saves exact bytes to a new file. Existing files are protected.
  The console uses UTF-8 replacement for invalid sequences. Malformed checkpoints
  and invalid options produce concise CLI errors.
- The old `scripts.sample_native` command delegates to the packaged sampler.
  Generated build and distribution directories are excluded from Git.

```bash
python -m energy_real.sample --checkpoint /path/to/selected_hybrid.pt \
  --prompt 'The ' --tokens 128 --temperature 0.8 --top-k 40 --seed 42 \
  --prefill-chunk 256 --output sample.bin
```

`--tokens` counts new bytes. Omit temperature/top-k/seed for greedy output.
Sampling controls change decoding, not the tiny model's language capabilities.
For external facts, use the existing `HybridSession` interface.

## Verification

**372 passed, 34 skipped**: 289 root tests plus 83 Spark tests. Checks cover
chunked greedy decoding against an independent full-prompt reference, seeded
reproducibility, tiny temperatures, exact binary output, malformed checkpoints,
early validation and exact training resume.

The non-editable wheel was imported under isolated Python from `/tmp`. All
**51 package source files** match the workspace. The revised training-config
builder exactly matches all four frozen study configurations.

Using the actual selected mixed checkpoint, chunked greedy generation matched
the full-prompt reference for a 2,048-byte prompt. A separate 32,768-byte prompt
used at most **256 input bytes per model call**, a **256 KiB returned logits
tensor**, and **37,512 bytes of carried state**. These are tensor measurements,
not total process-memory measurements. Exact-byte CLI output also matched the
library API. The native model source itself is byte-identical to the prior version.

- [Installation and checkpoint verification](verification.json)
- [Root test log](tests_root.log) · [Spark test log](tests_spark.log)
- [Package build](build.log) · [Package installation](install.log)
- [Wheel](../../runs/native_polish_install/wheels/energy_transformer_research-0.16.0-py3-none-any.whl)
- [Preserved prior source/reports](../../runs/native_polish_preservation/before.tar.gz)
- [Current source archive](../../runs/native_polish_preservation/after.tar.gz)
- [Artifact hashes and preservation audit](artifacts.json)

The installed-source fingerprint excludes the separate Spark source tree;
the artifact manifest records the full workspace source fingerprint. Earlier
experiments retain their original source archives and fingerprints.
