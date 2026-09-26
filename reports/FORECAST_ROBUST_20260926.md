# General square-root conditioning, artifact integrity and package integration

Prepared September 26, 2026. Parent research commit:
`7b067026719d55ff7f830a7f106290f4992143f7`.
The original teacher, heads, mean anchors, calibration temperatures and online gate
are unchanged. No neural training, CUDA execution, Spark job, data acquisition,
new forecast-quality evaluation or decoder modification occurred.

## Reproduced defects

The previous prefix-conditioning fix did not cover `marginal()` or arbitrary
observed coordinates. With a valid float32 triangular factor containing rows
[1,0,0] and [1,1e-4,0], forming a covariance rounds 1+1e-8 to 1. Covariance-based
marginalization and generic conditioning then fail Cholesky factorization.
Four diagnostic operations failed on that constructed fixture: leading marginal,
reordered marginal, reordered conditioning and nonprefix conditioning. All four
now complete. Conditioning on coordinate 1 preserves variance approximately 1e-8
in coordinate 0. This is a reproducible numerical fixture, not an assertion that
ordinary trained forecasts were previously all failing.

The research workflow also omitted the entire forecasting test suite, and the
root package-discovery configuration omitted `energy_forecast`. Both are fixed.
Existing local execution from the checkout hid the packaging omission.

## Exact method

For any selected rows B of an invertible covariance factor, compute reduced QR:

    B.T = Q R
    B B.T = R.T R

Normalize the row signs of R so its diagonal is positive and use R.T as the new
lower factor. This avoids forming the covariance and squaring its conditioning.
For arbitrary conditioning, permute factor rows to [observed,remaining], apply
this factorization, and use the established prefix square-root formula. Remaining
coordinates retain ascending original indices. Reordered leading observations are
canonicalized without QR; true prefix marginals simply copy the relevant factor.

This is classical QR/Gaussian algebra, not a new energy family. The conditional
mixture stays normalized and differentiable for the full-rank inputs assumed by
QR. Rank loss and nonfinite factors fail explicitly; there is no added jitter,
precision downgrade, value-dependent algorithm selection or hidden approximation.
QR has finite-precision limitations too and is not a cure for all ill-conditioning.
PyTorch's reduced QR supports gradients; the sign ambiguity and full-rank condition
are documented at https://docs.pytorch.org/docs/stable/generated/torch.linalg.qr.html .
The implementation uses the locally installed PyTorch 2.10.0+cpu, not a claim to
have tested the later version displayed by the current documentation.

Reduced factors now own only their declared storage. Mutating a returned reduced
factor does not change its source factor. This concerns factor ownership, not an
immutable public tensor API for every class field.

## Numerical parity on the existing trained head

The calibrated seed-11 head was evaluated without fitting on all 634 existing test
origins using leading, reversed-leading and nonleading-even coordinate sets.
Arbitrary subsets in this audit are offline mathematics checks, not evidence that
those future coordinates are available at the original forecast time.

Maximum absolute differences against the independent covariance reference:

| Quantity | Difference |
|---|---:|
| Conditional mean | 3.331e-15 |
| Component covariance | 2.665e-15 |
| Log evidence | 1.066e-14 |
| Conditional energy | 2.132e-14 |
| Marginal energy | 1.421e-14 |

These are float64 numerical checks, not bitwise equivalence across algorithms or
hardware. The original model-selection decision remains unchanged; +64 mean
fallback and the disabled far-only conditioning gate are retained.

## Cost: robustness is not universally faster

Nine randomized interleaved repetitions, twelve calls per repetition, one CPU
thread, float64, same frozen head and data. Times exclude teacher execution, head
construction, token generation and public-session validation/other work.

| Operation | Batch | Previous/reference ms | Revised ms |
|---|---:|---:|---:|
| Arbitrary conditioning | 1 | 0.3443 | 0.3877 |
| Remaining-target marginal | 1 | 0.1212 | 0.1772 |
| Exact online policy required work | 1 | 0.3947 | 0.2321 |
| Arbitrary conditioning | 64 | 4.3286 | 3.7732 |
| Remaining-target marginal | 64 | 1.9251 | 2.6599 |
| Exact online policy required work | 64 | 3.0637 | 1.1514 |

The default guarded pool still requires a prior marginal; it does not receive the
exact-only policy's saving. QR marginalization is slower in this test. The faster
exact-only operation removes an unused prior construction but does not make exact
conditioning a better predictor; the prior study found it often hurt quality.
Historical `prefix_benchmark` now calls a preserved covariance implementation,
not a reversed input that the improved dispatch would simply canonicalize. This
avoids accidentally benchmarking the fast implementation against itself.

## Artifact and package fixes

`atomic_save` writes a unique same-directory temporary file, flushes its contents,
publishes it atomically and syncs the directory. Serialization/replace errors
leave the old completed destination intact and clean their own temporary file.
The old predictable `.pt.tmp` name is not touched. Dataset preparation requests
create-only publication through an atomic hard link, so exactly one competing
writer can publish a new path. This is a local POSIX guarantee, not distributed
coordination or a multi-file transaction. A process crash may leave an unreferenced
temporary file; it must not leave a partial completed dataset. A failure after
publication/directory-sync may report an error despite the new completed file
existing; no exactly-once application-level claim is made.

The root package now includes `energy_forecast*` and a `forecast` optional dependency
group. CI invokes forecast tests and checks imports outside the checkout. Existing
pinned action identities and read-only workflow permissions are preserved. No
workflow downloads private checkpoints or starts training.

## Verification and scope

The full standalone focused suite passed 126 tests (89 inherited, 37 new), including
independent densities and gradients, cancellation, original teacher nonmutation,
existing fresh-process inference/training resume, failed-write fault injection and
racing create-only publication. A separate source-only copy passed 108 tests with
18 explicit missing-artifact skips. An isolated wheel import worked outside the
checkout; this test built a forecasting-only source copy, not the entire repository.
All original .pt checkpoints retain their archived hashes. The full repository's
historical suites and remote CI are not counted as locally executed here.

A first optional audit invocation used a nonexistent numeric-data filename; it
failed before evaluation and was rerun with results/state_data.pt. That setup log
is retained. No model-selection or scientific result was changed by this fix.

The attached conceptual note distinguishes speculative nonlinear landscapes from
the older quadratic backbone (Pasted markdown, lines483-514). This patch preserves
that separation: it makes the existing normalized future-energy distribution more
reliable, without presenting it as a new backbone or a demonstrated fractal model.

See results/robust/audit.json for per-chunk parity and raw timing repetitions, and
FORECAST_ROBUST_START_HERE.md for standalone and repository commands.
