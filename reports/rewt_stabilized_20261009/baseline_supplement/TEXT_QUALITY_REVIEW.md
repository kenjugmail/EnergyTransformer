# Exploratory assistant review of generated text

All 90 anonymous 96-byte continuations and their 32-byte prompts were read in the fixed shuffled order before reading the seed identity key. The supplemental baseline family was known. The ratings were locked with hashes before the seed identity join. The reviewer is Codex, which also implemented the models; these are exploratory assistant ratings, not independent human ratings. Human ratings in the separate review interface remain pending.

Fixed rubric: 1 = unusable/gibberish; 2 = occasional intelligible fragments with major failures; 3 = partly readable with clear errors; 4 = mostly clear and appropriate with minor errors; 5 = clear, consistent and appropriate throughout. Fluency measures readability, coherence measures internal consistency, and relevance measures connection to the prompt. Short continuations can score well if clear.

Means below average the per-seed means, with sample standard deviations across three seeds. These subjective ordinal scores are only a coarse quality check, particularly near the bottom of the scale. They do not establish human preference or general generation quality.

| Variant | Samples | Fluency / 5 | Coherence / 5 | Relevance / 5 |
|---|---:|---:|---:|---:|
| Supplemental Transformer | 90 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.09 ± 0.02 |

The [anonymous review](text_review.html) retains unfilled human scores. All manual assistant ratings and notes are saved in assistant_text_ratings.json, the original batches in assistant_rating_batches.jsonl, and the pre-identity lock in assistant_ratings_lock.json. No uninspected sample was assigned a score.
