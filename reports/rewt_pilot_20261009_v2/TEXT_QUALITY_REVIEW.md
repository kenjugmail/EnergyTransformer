# Assistant review of the primary generated text

The assistant read all 360 anonymous prompt/continuation pairs, scored them with
the fixed 1–5 fluency/coherence/relevance rubric, and saved and locked the scores
before joining the separate model/seed identity key. Each model contributed 90
continuations: 30 identical held-out prefixes across three trained seeds.

| Model | Reviewed continuations | Fluency | Coherence | Relevance |
|---|---:|---:|---:|---:|
| Transformer | 90 | 1.00/5 | 1.00/5 | 1.00/5 |
| Non-Fourier neural energy | 90 | 1.00/5 | 1.00/5 | 1.00/5 |
| Basic Fourier REWT | 90 | 1.00/5 | 1.00/5 | 1.00/5 |
| Full REWT | 90 | 1.00/5 | 1.00/5 | 1.00/5 |

The continuations are dominated by invented words, broken Wikipedia markup and
unconnected fragments. A few familiar words or brief grammatical openings do
not make a whole 96-byte continuation intelligible. On this whole-continuation
rubric, every reviewed sample remains at the lowest category, “unusable/gibberish.”
This is a floor effect: these ratings do not distinguish the four architectures
or imply that their predicted-byte distributions are identical.

The reviewer also implemented the models. These are exploratory assistant
judgments, not an independent human study. Human ratings remain pending in the
anonymous review page, and no assistant scores are prefilled there. The raw
scores, case hash and saved rating lock are retained in assistant_text_ratings.json
and assistant_ratings_lock.json.

The small primary budget is 61,440 training target bytes per model/seed.
These samples therefore support a limited conclusion: this pilot has not
produced usable generated English. Lower held-out bits per byte alone should
not be presented as demonstrated fluency or reasoning quality.

An exploratory extension generated text from the already trained equal-time
models: the first ten frozen prefixes, identical sampling settings, four models
and three seeds (120 additional continuations). The assistant read all of them
anonymously and locked scores before joining identities. All remain at 1/5 on
the same rubric; the short-budget full model often emits nontext bytes as well.
Raw outputs and separately labeled scores are saved in time_text/. These extra
samples were not used to choose hyperparameters or checkpoints.
