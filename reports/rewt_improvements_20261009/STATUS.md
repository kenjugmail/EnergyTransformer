# Stopped follow-up: full-model training instability

This attempt is preserved and is not a completed performance study. All fifteen language fits and the twelve non-full reasoning fits completed. Two full-model reasoning fits stopped advancing; the third completed reasoning and began evaluation before the parent run was terminated. Partial test outputs are not pooled with a later repaired study.

An exact replay of seed 202 reproduced the failure: all gradient components were finite, but the float32 norm overflowed. The true float64 norm was approximately 5.20e20. Wider norm and Fourier-power arithmetic resolved that false failure; replaying further then exposed genuinely non-finite components in the long recurrent backward chain. Both replay records are retained.

The implementation now computes norms and spectral powers in float64 and rejects actual non-finite components. A training-only temporal boundary can detach carried latent-state gradients every sixteen observed tokens while retaining all forward state values, observed embeddings, and the full 128-byte input window. The joint energy, state gradients, inference weights, and accepted inference updates are unchanged.

The repaired full reasoning fits will restart from their completed course-trained language checkpoints. Completed control weights can be reused with their original source and training records explicitly identified. A new held-out evaluation sample must be declared, because seed 303 produced some evaluation outputs before termination. No improved-quality claim is made from this stopped attempt.
