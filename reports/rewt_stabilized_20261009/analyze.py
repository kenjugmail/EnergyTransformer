"""Read-only checks and summaries of the completed repaired-model experiment."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
LABELS = dict(transformer="Transformer", legacy_fourier="Raw REWT control",
              neural_energy="Revised neural energy", fourier_rewt="Revised Fourier REWT",
              full_rewt="Revised full REWT")


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    return dict(mean=statistics.mean(values), std=statistics.stdev(values) if len(values)>1 else 0., values=values)


def main():
    assert (ROOT/"complete.json").exists(), "The main run has not completed"
    plan=read(ROOT/"selected_protocol.json")
    seeds=plan["settings"]["seeds"]
    rows=[read(ROOT/f"seed_{seed}"/f"{arm}.json") for seed in seeds for arm in LABELS]
    tasks={split:lines(ROOT/f"tasks_{split}.jsonl") for split in ("train","valid","test")}
    ids={split:{t["id"] for t in group} for split,group in tasks.items()}
    assert len(rows)==len(seeds)*len(LABELS)
    assert not ids["train"]&ids["valid"] and not ids["train"]&ids["test"] and not ids["valid"]&ids["test"]
    assert not ids["test"]&set(plan["previous_test_exclusions"])
    source={name:sha(REPO/"rewt"/name)==value and sha(ROOT/"source"/name)==value
            for name,value in plan["source_sha256"].items()}
    assert all(source.values()), source
    previous=read(REPO/"reports"/"rewt_pilot_20261009_v2"/"plan.json")
    old_offsets=plan.get("forbidden_evaluation_offsets", previous["test_offsets"]+[p["offset"] for p in previous["text_prompts"]])
    fresh_offsets=plan["test_offsets"]+[p["offset"] for p in plan["text_prompts"]]
    assert all(abs(a-b)>128 for a in fresh_offsets for b in old_offsets)
    reused=[]
    for fit in plan.get("reused_fits",[]):
        original=Path(fit["original_checkpoint"])
        copied=Path(fit["copied_checkpoint"])
        assert sha(original)==sha(copied)==fit["checkpoint_sha256"]
        assert sha(original.parent/"training.json")==sha(copied.parent/"training.json")==fit["training_record_sha256"]
        assert fit["source_sha256"]==plan["parent_source_sha256"]
        reused.append(dict(seed=fit["seed"],variant=fit["variant"],phase=fit["phase"],
                           exact_checkpoint_bytes_preserved=True,training_record_preserved=True))
    assert len(reused)==27
    parity=read(ROOT/"reused_inference_parity.json")
    assert len(parity)==27 and all(p["exact_logits_equal"] for p in parity)
    expected=plan["settings"]["lm_updates"]*plan["settings"]["batch_size"]*128
    assert all(r["training"]["processed_target_bytes"]==expected for r in rows)
    policy_checks=[]
    for row in rows:
        for kind in ("language_policy","reasoning_policy"):
            policy=row[kind]
            if not policy["validation"]:
                continue
            chosen=[v for v in policy["validation"] if v["rounds"]==policy["rounds"]
                    and v["halting_tolerance"]==policy["halting_tolerance"]][0]
            assert chosen["bpb"]<=policy["best_validation_bpb"]+plan["policy_bpb_tolerance"]+1e-10
            policy_checks.append(dict(arm=row["arm"],seed=row["seed"],kind=kind,**chosen))
    grouped=defaultdict(list)
    diagnostics=[]
    for row in rows:
        records=lines(ROOT/f"seed_{row['seed']}"/f"{row['arm']}_reasoning.jsonl")
        assert len(records)==len(tasks["test"])
        assert {r["id"] for r in records}==ids["test"]
        for record in records:
            grouped[(row["arm"],record["family"],record["steps"],record["template"]==2)].append(record)
        d=row["local_diagnostics"]
        if d:
            rounds={v["rounds"]:v for v in d["rounds"]}
            diagnostics.append(dict(arm=row["arm"],seed=row["seed"],
                common_gradient_share=d["mean_common_output_squared_gradient_fraction"],
                common_shift_energy=d["common_shift_mean_energy_change"],
                common_shift_probability=d["common_shift_max_probability_change"],
                output_ce_cosine=d["mean_output_ce_gradient_cosine"],
                fraction_first_order_descent=d["fraction_first_order_ce_descent"],
                two_round_bpb_change=rounds[2]["bpb"]-rounds[0]["bpb"],
                eight_round_bpb_change=rounds[8]["bpb"]-rounds[0]["bpb"],
                two_round_energy_change=rounds[2]["mean_energy"]-rounds[0]["mean_energy"]))
            assert all(rounds[b]["mean_energy"]<=rounds[a]["mean_energy"]+1e-5
                       for a,b in zip((0,1,2,4),(1,2,4,8)))
    breakdown=[]
    for (arm,family,steps,ood),records in grouped.items():
        by_seed=[]
        for seed in seeds:
            # Task ordering is identical in every seed, so use its saved record directly.
            group=[r for r in lines(ROOT/f"seed_{seed}"/f"{arm}_reasoning.jsonl")
                   if r["family"]==family and r["steps"]==steps and (r["template"]==2)==ood]
            by_seed.append(dict(seed=seed,count=len(group),answer=sum(r["answer_correct"] for r in group)/len(group),
                                complete=sum(r["both_correct"] for r in group)/len(group)))
        breakdown.append(dict(arm=arm,family=family,steps=steps,withheld_template=ood,
                              answer=stats([r["answer"] for r in by_seed]),
                              complete=stats([r["complete"] for r in by_seed]),by_seed=by_seed))
    summaries={arm:{field:stats([d[field] for d in diagnostics if d["arm"]==arm])
                      for field in ("common_gradient_share","common_shift_energy","output_ce_cosine",
                                    "fraction_first_order_descent","two_round_bpb_change","eight_round_bpb_change")}
               for arm in LABELS if any(d["arm"]==arm for d in diagnostics)}
    validation_curves=read(ROOT/"validation_learning_curves.json")
    result=dict(source_unchanged=source,source_snapshot_unchanged=all(source.values()),
                fresh_test_offsets=True,semantic_splits_disjoint=True,prior_test_tasks_excluded=True,
                target_bytes_per_model=expected,policy_checks=policy_checks,
                reused_fit_provenance=reused,reused_checkpoint_parity=parity,
                local_diagnostics=diagnostics,local_summaries=summaries,reasoning_breakdown=breakdown,
                validation_learning_curves=validation_curves)
    (ROOT/"diagnostics.json").write_text(json.dumps(result,indent=2)+"\n")
    verification=read(ROOT/"verification.json")
    verification.update(source_current_and_snapshot_unchanged=all(source.values()),
                        fresh_test_windows_and_prefixes_verified=True,semantic_splits_disjoint=True,
                        previous_test_tasks_excluded=True,common_training_bytes_verified=True,
                        selected_policies_within_declared_validation_tolerance=True,
                        reused_weights_and_original_training_records_unchanged=True,
                        reused_inference_bitwise_parity_verified=True)
    (ROOT/"verification.json").write_text(json.dumps(verification,indent=2)+"\n")
    output=["# Follow-up diagnostics", "",
        "These summaries consume saved outputs after the experiment completes. They do not change model weights or select training settings. The current source hashes and snapshot match the frozen restart protocol. All twenty-seven reused checkpoint bytes and original training records also match their separate parent provenance. Language fits use the same declared byte budget; evaluation windows/prefixes and synthetic identities exclude both prior runs.", "",
        "Local refinement freezes each original prefix's conditioning. Negative bpb change means local refinement improves prediction. Common-gradient share is a share of squared norm. Energy monotonicity is checked separately from prediction quality.", "",
        "| Variant | Initial common gradient share % | Two-round bpb change | Eight-round bpb change | Gradient / CE cosine | First-order CE descent % |",
        "|---|---:|---:|---:|---:|---:|"]
    for arm,group in summaries.items():
        output.append(f"| {LABELS[arm]} | {group['common_gradient_share']['mean']*100:.6f} | {group['two_round_bpb_change']['mean']:+.6f} | {group['eight_round_bpb_change']['mean']:+.6f} | {group['output_ce_cosine']['mean']:+.4f} | {group['fraction_first_order_descent']['mean']*100:.1f} |")
    output += ["", "Reasoning breakdowns below average identical task groups across seeds. Short subgroups have limited precision; withheld wording and longer chains test different forms of generalization. Recall is reported separately from arithmetic and rules.", "",
               "| Variant | Family | Steps/delay | Withheld wording | Tasks per seed | Exact answer % | Complete % |",
               "|---|---|---:|---|---:|---:|---:|"]
    for d in sorted(breakdown,key=lambda x:(list(LABELS).index(x['arm']),x['family'],x['steps'],x['withheld_template'])):
        output.append(f"| {LABELS[d['arm']]} | {d['family']} | {d['steps']} | {d['withheld_template']} | {d['by_seed'][0]['count']} | {d['answer']['mean']*100:.1f} | {d['complete']['mean']*100:.1f} |")
    output += ["", "Training-length comparisons use validation learning curves at the declared quarter and full byte budgets. The longer budget changes both data seen and optimizer steps. The main raw/revised comparison also bundles several repairs, so neither curve establishes a single-component causal effect.", ""]
    (ROOT/"DIAGNOSTICS.md").write_text("\n".join(output))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,2,figsize=(11,4))
    for arm in LABELS:
        group=[c for c in validation_curves if c["arm"]==arm]
        budgets=sorted(set(c["target_bytes"] for c in group))
        values=[stats([c["validation_bpb"] for c in group if c["target_bytes"]==b]) for b in budgets]
        axes[0].errorbar([b/1e6 for b in budgets],[v['mean'] for v in values],
                         yerr=[v['std'] for v in values],marker='o',capsize=3,label=LABELS[arm])
    axes[0].set_xlabel("Language target bytes (millions)");axes[0].set_ylabel("Validation bpb ↓")
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    arms=list(summaries)
    axes[1].bar(range(len(arms)),[summaries[a]['two_round_bpb_change']['mean'] for a in arms],
                yerr=[summaries[a]['two_round_bpb_change']['std'] for a in arms],capsize=3)
    axes[1].axhline(0,color='black',linewidth=.8)
    axes[1].set_xticks(range(len(arms)),[LABELS[a] for a in arms],rotation=15,ha='right')
    axes[1].set_ylabel("Two local refinement rounds: bpb change ↓")
    axes[1].grid(axis='y',alpha=.2)
    figure.tight_layout();figure.savefig(ROOT/"diagnostics.png",dpi=160);plt.close(figure)
    print(json.dumps(dict(source_unchanged=True,rows=len(rows),local_summaries=summaries),indent=2))


if __name__=="__main__":
    main()
