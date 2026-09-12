# Model card: v14 bounded text-memory bridge

**Intended use:** reproducible synthetic research on parsing memory edits and retrieving protected associations. Not a production assistant.

**Inputs:** one sentence at a time; fixed training vocabulary; 16 known names and 16 locations. Sentence boundaries and the permissible relation are supplied. Values and operations are not supplied as IDs at inference.

**Training:** three seeds; each uses 900 parser updates and 500 answer-head updates, batch 64. Synthetic operation/entity/value labels provide strong supervision. The answer head is trained on supplied memory features. The old byte LM is unchanged.

**Results:** perfect tested known-grammar recall is matched by an ordinary dictionary with the same parser. On held-out wording, average parse accuracy is about 57.8%, and repeated falsely interpreted distractors damage memory. Confidence is not calibrated to domain shift.

**Safety default:** `handle()` does not authorize writes unless the caller explicitly sets `trusted=True`. Untrusted data cannot mutate the state through this interface. Trusted does not mean correctly parsed. Do not automatically mark retrieved documents, web pages, tool output, or quoted content as trusted.

**Output:** a trained, bounded byte head emits one location name or `unknown`. It does not generate explanations or perform unrestricted reasoning.

**Privacy:** no personal corpus or external training dataset is used in v14. Names are synthetic labels. State snapshots contain model hashes and numerical coefficients, not prompt transcripts.

**Not established:** general language understanding, instruction following, factual confidence, broad robustness, fractal reasoning, lossless compression, LLM-scale attention superiority, or independent replication.

Historical v1-v13 reports are retained as artifacts, not newly verified by this release.
