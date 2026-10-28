"""Lock manually assigned anonymous assistant ratings before joining identities."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
FIELDS=("fluency","coherence","relevance")
LABELS=dict(transformer="Supplemental Transformer")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_ratings():
    cases=json.loads((ROOT/"text_review_cases.json").read_text())
    batches=[json.loads(line) for line in (ROOT/"assistant_rating_batches.jsonl").read_text().splitlines()]
    ratings=[r for batch in batches for r in batch["ratings"]]
    assert len(ratings)==len(cases) and len({r['id'] for r in ratings})==len(ratings)
    assert {r['id'] for r in ratings}=={c['id'] for c in cases}
    assert all(type(r[f]) is int and 1<=r[f]<=5 for r in ratings for f in FIELDS)
    return cases,ratings,batches


def lock():
    cases,ratings,batches=checked_ratings()
    path=ROOT/"assistant_ratings_lock.json"
    assert not path.exists(), "Ratings already locked"
    value=dict(locked_at=datetime.now(timezone.utc).isoformat(),reviewer_type="assistant",
        reviewer="Codex; also the implementation author",rubric_version=1,
        case_count=len(cases),rated_count=len(ratings),case_sha256=sha(ROOT/"text_review_cases.json"),
        batches_sha256=sha(ROOT/"assistant_rating_batches.jsonl"),
        process="All supplemental prompts and continuations read and scored before opening the seed identity key; baseline family known",
        scope="Exploratory assistant assessment; not an independent human evaluation")
    path.write_text(json.dumps(value,indent=2)+"\n")
    (ROOT/"assistant_text_ratings.json").write_text(json.dumps({**value,"ratings":ratings},indent=2)+"\n")
    print(json.dumps(value,indent=2))


def summarize():
    cases,ratings,batches=checked_ratings()
    lock=json.loads((ROOT/"assistant_ratings_lock.json").read_text())
    assert lock["case_sha256"]==sha(ROOT/"text_review_cases.json")
    assert lock["batches_sha256"]==sha(ROOT/"assistant_rating_batches.jsonl")
    key=json.loads((ROOT/"text_review_key.json").read_text())
    identity={r["id"]:r for r in key}
    groups=defaultdict(list)
    per_seed=defaultdict(list)
    for r in ratings:
        identified={**r,**identity[r['id']]}
        groups[identified['arm']].append(identified)
        per_seed[(identified['arm'],identified['seed'])].append(identified)
    aggregate={}
    for arm in LABELS:
        seeds=sorted(seed for a,seed in per_seed if a==arm)
        aggregate[arm]={}
        for field in FIELDS:
            values=[statistics.mean(r[field] for r in per_seed[(arm,seed)]) for seed in seeds]
            aggregate[arm][field]=dict(mean=statistics.mean(values),std=statistics.stdev(values) if len(values)>1 else 0.,
                per_seed=dict(zip(map(str,seeds),values)),histogram={str(n):sum(r[field]==n for r in groups[arm]) for n in range(1,6)})
    (ROOT/"assistant_text_quality_aggregate.json").write_text(json.dumps(aggregate,indent=2)+"\n")
    output=["# Exploratory assistant review of generated text", "",
        f"All {len(cases)} anonymous 96-byte continuations and their 32-byte prompts were read in the fixed shuffled order before reading the seed identity key. The supplemental baseline family was known. The ratings were locked with hashes before the seed identity join. The reviewer is Codex, which also implemented the models; these are exploratory assistant ratings, not independent human ratings. Human ratings in the separate review interface remain pending.", "",
        "Fixed rubric: 1 = unusable/gibberish; 2 = occasional intelligible fragments with major failures; 3 = partly readable with clear errors; 4 = mostly clear and appropriate with minor errors; 5 = clear, consistent and appropriate throughout. Fluency measures readability, coherence measures internal consistency, and relevance measures connection to the prompt. Short continuations can score well if clear.", "",
        "Means below average the per-seed means, with sample standard deviations across three seeds. These subjective ordinal scores are only a coarse quality check, particularly near the bottom of the scale. They do not establish human preference or general generation quality.", "",
        "| Variant | Samples | Fluency / 5 | Coherence / 5 | Relevance / 5 |",
        "|---|---:|---:|---:|---:|"]
    for arm in LABELS:
        fields=[f"{aggregate[arm][f]['mean']:.2f} ± {aggregate[arm][f]['std']:.2f}" for f in FIELDS]
        output.append(f"| {LABELS[arm]} | {len(groups[arm])} | "+" | ".join(fields)+" |")
    output += ["", "The [anonymous review](text_review.html) retains unfilled human scores. All manual assistant ratings and notes are saved in assistant_text_ratings.json, the original batches in assistant_rating_batches.jsonl, and the pre-identity lock in assistant_ratings_lock.json. No uninspected sample was assigned a score.", ""]
    (ROOT/"TEXT_QUALITY_REVIEW.md").write_text("\n".join(output))
    verification=json.loads((ROOT/"verification.json").read_text())
    verification.update(assistant_text_samples_actually_reviewed=len(ratings),
                        assistant_ratings_locked_before_identity_join=True,
                        assistant_rating_cases_and_batches_hashes_verified=True,human_ratings_pending=True)
    (ROOT/"verification.json").write_text(json.dumps(verification,indent=2)+"\n")
    print(json.dumps(aggregate,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("lock","summarize"))
    args=parser.parse_args()
    lock() if args.mode=="lock" else summarize()
