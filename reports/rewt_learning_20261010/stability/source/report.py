"""Summarize actual saved measurements; prepare anonymous text review."""
from __future__ import annotations

from collections import defaultdict
import argparse
import html
import json
from pathlib import Path
import random

import numpy as np


LABELS = dict(transformer="Transformer", neural_energy="Neural energy",
              fourier_rewt="Fourier REWT", full_rewt="Full REWT")


def statistics(values):
    return dict(mean=float(np.mean(values)),
                std=float(np.std(values, ddof=1)) if len(values) > 1 else 0.,
                values=values, seeds=len(values))


def review_pack(out, rows):
    records = []
    for row in rows:
        path = out / f"seed_{row['seed']}" / f"{row['arm']}_text.jsonl"
        for line in path.read_text().splitlines():
            case = json.loads(line)
            records.append(dict(arm=row["arm"], seed=row["seed"], **case))
    random.Random(577).shuffle(records)
    anonymous, mapping = [], []
    for index, record in enumerate(records):
        case_id = f"C{index:03d}"
        anonymous.append(dict(id=case_id, prompt=record["prompt"], text=record["generation"],
                              fluency=None, coherence=None, relevance=None, note=""))
        mapping.append(dict(id=case_id, arm=record["arm"], seed=record["seed"], prompt_id=record["prompt_id"]))
    (out / "text_review_cases.json").write_text(json.dumps(anonymous, indent=2, ensure_ascii=True) + "\n")
    (out / "text_review_key.json").write_text(json.dumps(mapping, indent=2) + "\n")
    payload = json.dumps(anonymous, ensure_ascii=True).replace("</", "<\\/")
    document = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>REWT anonymous text review</title>
<style>body{font:17px/1.5 system-ui;max-width:950px;margin:35px auto;padding:0 18px;background:#f6f7fb;color:#202536}
header{position:sticky;top:0;background:#f6f7fb;padding:10px 0;border-bottom:1px solid #ccd}
article{padding:20px;background:white;border:1px solid #dde;margin:20px 0;border-radius:8px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fa;padding:12px}
label{display:inline-block;margin-right:20px}select,input{font:inherit;padding:5px}button{font:inherit;padding:8px 14px}
input{width:95%;margin-top:12px}h1{font-size:26px}p{margin:7px 0}</style>
<header><h1>Anonymous generated-text review</h1><button id="download">Save ratings</button>
<span id="progress"></span></header>
<p>Rate the 96-byte continuation after its prompt. Judge readability (fluency), consistency within the continuation (coherence), and connection to the prompt (relevance).</p>
<p>Fixed rubric: 1 = unusable/gibberish; 2 = occasional intelligible fragments with major failures; 3 = partly readable with clear errors; 4 = mostly clear and appropriate with minor errors; 5 = clear, consistent and appropriate throughout. Short continuations can still earn a high score.</p>
<p>These are anonymous samples in fixed shuffled order. No ratings have been filled in by the training program. Keep the separate identity key closed while scoring.</p>
<main id="cases"></main><script>
const cases = __PAYLOAD__;
const stored = localStorage.getItem("rewt-ratings-v1-" + cases.length);
if(stored){try{const old=JSON.parse(stored);old.forEach((x,i)=>{if(cases[i]&&cases[i].id===x.id){["fluency","coherence","relevance","note"].forEach(k=>cases[i][k]=x[k])}})}catch(e){}}
function save(){localStorage.setItem("rewt-ratings-v1-"+cases.length,JSON.stringify(cases));document.getElementById("progress").textContent=" "+cases.filter(x=>x.fluency&&x.coherence&&x.relevance).length+"/"+cases.length+" rated"}
for(const c of cases){
 const a=document.createElement("article"),h=document.createElement("h2");h.textContent=c.id;a.append(h);
 for(const [label,text] of [["Prompt",c.prompt],["Continuation",c.text]]){const p=document.createElement("p");p.textContent=label;const pre=document.createElement("pre");pre.textContent=text;a.append(p,pre)}
 for(const key of ["fluency","coherence","relevance"]){const label=document.createElement("label");label.textContent=key+" ";const s=document.createElement("select");for(let n=0;n<=5;n++){const o=document.createElement("option");o.value=n;o.textContent=n||"Select";s.append(o)}s.value=c[key]||0;s.onchange=()=>{c[key]=Number(s.value)||null;save()};label.append(s);a.append(label)}
 const note=document.createElement("input");note.placeholder="Optional note";note.value=c.note;note.oninput=()=>{c.note=note.value;save()};a.append(note);document.getElementById("cases").append(a)
}
document.getElementById("download").onclick=()=>{const data={reviewer_type:"human",rubric_version:1,ratings:cases};const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:"application/json"}));a.download="rewt-text-ratings.json";a.click();URL.revokeObjectURL(a.href)};save();
</script></html>"""
    (out / "text_review.html").write_text(document.replace("__PAYLOAD__", payload))
    return len(anonymous)


def create_report(plan):
    out = Path(plan["out"])
    rows = [json.loads((out / f"seed_{seed}" / f"{arm}.json").read_text())
            for seed in plan["settings"]["seeds"] for arm in LABELS]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["arm"]].append(row)
    aggregate = {}
    for arm, group in grouped.items():
        fields = dict(
            test_bpb=[r["language"]["bpb"] for r in group],
            brier=[r["language"]["brier"] for r in group],
            language_training_cpu=[r["training"]["training_cpu_seconds"] for r in group],
            language_pipeline_cpu=[r["training"]["training_plus_shared_teacher_cpu_seconds"] for r in group],
            inference_cpu_ms_per_byte=[r["language"]["cpu_ms_per_byte"] for r in group],
            equal_time_test_bpb=[r["equal_cpu_time"]["language"]["bpb"] for r in group],
            equal_time_pipeline_cpu=[r["equal_cpu_time"]["training"]["training_plus_shared_teacher_cpu_seconds"] for r in group],
            reasoning_answer=[r["adapted_reasoning"]["all"]["answer_correct"] for r in group],
            reasoning_valid_steps=[r["adapted_reasoning"]["all"]["both_correct"] for r in group],
            arithmetic_answer=[r["adapted_reasoning"]["by_family"]["arithmetic"]["answer_correct"] for r in group],
            rules_answer=[r["adapted_reasoning"]["by_family"]["rules"]["answer_correct"] for r in group],
            recall_answer=[r["adapted_reasoning"]["by_family"]["recall"]["answer_correct"] for r in group],
            printable_fraction=[r["text"]["printable_fraction"] for r in group],
            repeated_byte_trigram_fraction=[r["text"]["repeated_byte_trigram_fraction"] for r in group],
            raw_gradient_spike_fraction=[r["training"]["raw_spike_fraction"] for r in group])
        aggregate[arm] = {field: statistics(values) for field, values in fields.items()}
    (out / "aggregate.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    review_count = review_pack(out, rows)
    s = plan["settings"]
    lines = [
        "# REWT implementation and controlled pilot",
        "",
        "This report contains measurements from newly initialized models, not the proposal's expected results. "
        "The pilot implements the joint energy, recurrent state, Fourier modes, forecasting, denoising proposals, "
        "specialists, communication, learned critic/router, and a separate Fourier gradient-history filter. "
        "It is a small feasibility experiment, not a completed twelve-GPU-hour course study.",
        "",
        f"Primary design: four model families × {len(s['seeds'])} seeds ({', '.join(map(str,s['seeds']))}), "
        f"about 250,000 active parameters each, {s['lm_updates']} LM updates of "
        f"{s['batch_size']} × 128 bytes ({s['lm_updates']*s['batch_size']*128:,} target bytes per fit). "
        f"All models start from random weights. AdamW LR {s['learning_rate']}, common batches, context resets "
        "at 128 bytes, validation-only checkpoint selection, and a checksum-verified 90M/5M/5M enwik8 split. "
        "The full model's frozen forecast teacher is its seed's from-scratch baseline.",
        "",
        f"Reasoning: copies of those models receive {s['reason_updates']} answer-only adaptation updates, "
        f"batch {s['reason_batch_size']}, on {s['task_train']:,} semantic-hash-partitioned training examples. "
        f"The disjoint test has {s['task_test']} arithmetic/rule-chain/recall examples, including withheld "
        "wording templates and four-step chains. Reported reasoning answers are actually generated greedily; "
        "teacher-forced answer likelihood is kept separately.",
        "",
        "## Held-out prediction and cost",
        "",
        "Numbers are mean ± sample standard deviation across seeds. Lower bits per byte (bpb) is better. "
        "Training CPU seconds include optimizer, losses, auxiliary teacher forwards, and gradient filtering; "
        "the pipeline column additionally charges shared teacher training once. Validation/evaluation times "
        "and cumulative worker peak RSS are saved separately. Timing is CPU-only on a shared host.",
        "",
        "| Model | Active parameters | Test bpb ↓ | Brier ↓ | Training CPU s | With teacher CPU s | Inference ms/byte |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    def fmt(arm, field, scale=1, digits=3):
        v = aggregate[arm][field]
        return f"{v['mean']*scale:.{digits}f} ± {v['std']*scale:.{digits}f}"
    for arm in LABELS:
        lines.append(f"| {LABELS[arm]} | {plan['counts'][arm]:,} | {fmt(arm,'test_bpb')} | {fmt(arm,'brier')} | "
                     f"{fmt(arm,'language_training_cpu',digits=1)} | {fmt(arm,'language_pipeline_cpu',digits=1)} | "
                     f"{fmt(arm,'inference_cpu_ms_per_byte')} |")
    lines += ["", "## Separate equal training-time control", "",
        f"Every fresh fit has a {s['equal_cpu_seconds']:g}-CPU-second training budget. Full REWT spends 25% "
        "on a fresh forecast teacher and the remainder on itself. The final complete update can overshoot "
        "the budget; actual total CPU time and update counts are shown in the saved JSON. This protocol uses "
        "four initial proposal-only warm-up updates and selects the final validation checkpoint, so it is "
        "separate from the equal-byte primary comparison.",
        "", "| Model | Test bpb ↓ | Actual pipeline CPU s |", "|---|---:|---:|"]
    for arm in LABELS:
        lines.append(f"| {LABELS[arm]} | {fmt(arm,'equal_time_test_bpb')} | {fmt(arm,'equal_time_pipeline_cpu',digits=1)} |")
    lines += ["", "## Generated reasoning and recall", "",
        "Exact answer requires the declared response format. Valid steps requires every intermediate state "
        "and the final answer to agree with a deterministic checker. These bounded tasks do not establish general reasoning.",
        "", "| Model | All exact answers % | Correct answers + steps % | Arithmetic % | Rule chains % | Recall % |",
        "|---|---:|---:|---:|---:|---:|"]
    for arm in LABELS:
        lines.append(f"| {LABELS[arm]} | {fmt(arm,'reasoning_answer',100,1)} | {fmt(arm,'reasoning_valid_steps',100,1)} | "
                     f"{fmt(arm,'arithmetic_answer',100,1)} | {fmt(arm,'rules_answer',100,1)} | {fmt(arm,'recall_answer',100,1)} |")
    lines += ["", "## Generated-text evidence", "",
        f"All {review_count} continuations use the same 30 held-out 32-byte prefixes, 96 generated bytes, "
        "temperature 0.8 and paired sampling uniforms. Raw bytes, readable text, and reference continuations "
        "are preserved in each seed's text JSONL. Printable and repetition rates are diagnostics, not fluency scores.",
        "", "| Model | Printable bytes % ↑ | Repeated byte trigrams % ↓ |",
        "|---|---:|---:|"]
    for arm in LABELS:
        lines.append(f"| {LABELS[arm]} | {fmt(arm,'printable_fraction',100,1)} | {fmt(arm,'repeated_byte_trigram_fraction',100,1)} |")
    lines += ["", "[Open anonymous text review](text_review.html). The fixed 1–5 rubric covers fluency, coherence, "
        "and relevance. Human ratings remain pending; the training program never manufactures them. "
        "Any later assistant review must identify itself separately from human ratings.",
        "", "## Mechanism and component checks", "",
        "Refinement audits save the energy at each round and verify that accepted states never increase the "
        "full energy. Unit tests compare analytic derivatives to independent autograd, test causal prefixes, "
        "chunk replay, fixed inference weights, disjoint task identities and strict answer checking.",
        "",
        "The ablation folder contains short single-seed retraining for learned/fixed frequencies, coupled/"
        "separate modes, coarse-to-fine scheduling, and none/clip/EMA/Fourier gradient filtering. A second "
        "file holds inference-only round sweeps (0/1/2/4/8), state/output updates, and module removals. "
        "Those removals are interventions on trained weights, not equivalent to retraining without a module. "
        "Compare these diagnostics within their own protocol, not against the longer primary fits.",
        "",
        "## Limits and reproducibility", "",
        "The data and training budget are deliberately small. The recurrent state carries up to the common "
        "128-byte reset, with a 16-token attention/history buffer; the baseline attends across the whole "
        "128-byte window. Frequency modes are a finite learned energy family, not an exact reconstruction "
        "of an arbitrary landscape. Lower energy is not guaranteed to improve the next-byte probability.",
        "",
        "Fixed-frequency retraining has fewer trainable parameters; the primary four-arm counts are matched "
        "within 0.5%. Shared-host wall timings can fluctuate, so use recorded process CPU time and sequential "
        "profiles for cost interpretation. Cumulative worker RSS is not a per-model activation-memory peak.",
        "",
        "The plan stores data/source hashes, every dataset offset, task semantic identity, all configurations, "
        "loss weights, preset spike threshold and generation settings. The source snapshot is in source/. "
        "Best and resumable optimizer checkpoints are saved under the plan's runs path.",
        "",
        "Reproduce with .venv/bin/python -m rewt.experiment. Resume the exact plan with --resume. "
        "Do not tune settings from this held-out pilot and then call its test split unseen in a later study; "
        "predeclare a new evaluation sample for any follow-up selected from these results.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))
    # Optional plots consume saved data only and cannot affect model selection.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arms = list(LABELS)
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for axis, field, title, scale in ((axes[0], "test_bpb", "Held-out bits per byte ↓", 1),
        (axes[1], "reasoning_answer", "Generated exact answers (%) ↑", 100),
        (axes[2], "inference_cpu_ms_per_byte", "Inference CPU ms / byte ↓", 1)):
        values = [aggregate[a][field]["mean"] * scale for a in arms]
        errors = [aggregate[a][field]["std"] * scale for a in arms]
        axis.bar(range(len(arms)), values, yerr=errors, capsize=4,
                 color=["#4f6b94", "#8d77af", "#2c937a", "#bd7d38"])
        axis.set_xticks(range(len(arms)), ["Transformer", "Neural\nenergy", "Fourier\nREWT", "Full\nREWT"])
        axis.set_title(title)
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(out / "comparison.png", dpi=170)
    plt.close(figure)
    return aggregate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    args = parser.parse_args()
    create_report(json.loads((Path(args.directory) / "plan.json").read_text()))
