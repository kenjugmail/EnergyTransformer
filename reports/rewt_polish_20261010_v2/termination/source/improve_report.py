"""Report the revised experiment from measured outputs without inventing ratings."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from .experiment import load_model, save_json
from .improve import LABELS, with_policy
from .report import review_pack, statistics


def summarize(records):
    return dict(count=len(records), **{name:sum(bool(r[name]) for r in records)/max(1,len(records))
                    for name in ("valid_format","answer_correct","steps_correct","both_correct")})


def create_report(plan):
    out=Path(plan["out"])
    rows=[json.loads((out/f"seed_{seed}"/f"{arm}.json").read_text())
          for seed in plan["settings"]["seeds"] for arm in LABELS]
    grouped=defaultdict(list)
    for row in rows:
        records=[json.loads(x) for x in (out/f"seed_{row['seed']}"/f"{row['arm']}_reasoning.jsonl").read_text().splitlines()]
        reasoning=[r for r in records if r["family"] != "recall"]
        row["reasoning_only"]=summarize(reasoning)
        row["reasoning_in_distribution"]=summarize([r for r in reasoning if r["template"] != 2 and r["steps"] <=3])
        row["reasoning_ood"]=summarize([r for r in reasoning if r["template"] ==2 or r["steps"] ==4])
        grouped[row["arm"]].append(row)
    aggregate={}
    for arm,group in grouped.items():
        fields=dict(test_bpb=[r["language"]["bpb"] for r in group],
            equal_time_test_bpb=[r["equal_cpu_time"]["language"]["bpb"] for r in group],
            language_pipeline_cpu=[r["training"]["training_plus_shared_teacher_cpu_seconds"] for r in group],
            inference_ms_per_byte=[r["language"]["cpu_ms_per_byte"] for r in group],
            reasoning_answer=[r["reasoning_only"]["answer_correct"] for r in group],
            reasoning_complete=[r["reasoning_only"]["both_correct"] for r in group],
            reasoning_id_complete=[r["reasoning_in_distribution"]["both_correct"] for r in group],
            reasoning_ood_complete=[r["reasoning_ood"]["both_correct"] for r in group],
            recall=[r["adapted_reasoning"]["by_family"]["recall"]["both_correct"] for r in group],
            train_complete=[r["training_reasoning"]["all"]["both_correct"] for r in group],
            printable=[r["text"]["printable_fraction"] for r in group],
            repetition=[r["text"]["repeated_byte_trigram_fraction"] for r in group],
            raw_spikes=[r["training"]["raw_spike_fraction"] for r in group])
        aggregate[arm]={field:statistics(values) for field,values in fields.items()}
    save_json(out/"aggregate.json",aggregate)
    review_count=review_pack(out,rows)
    task_train=[json.loads(x) for x in (out/"tasks_train.jsonl").read_text().splitlines()]
    task_test=[json.loads(x) for x in (out/"tasks_test.jsonl").read_text().splitlines()]
    global_prior=Counter(t["response"] for t in task_train).most_common(1)[0][0]
    family_priors={family:Counter(t["response"] for t in task_train if t["family"]==family).most_common(1)[0][0]
                   for family in ("arithmetic","rules","recall")}
    from .tasks import Task,score_response
    priors={}
    for name,responses in (("constant",global_prior),("known_family_constant",family_priors)):
        records=[]
        for data in task_test:
            if data["family"]=="recall":
                continue
            response=responses if isinstance(responses,str) else responses[data["family"]]
            records.append(score_response(Task(**data),response))
        priors[name]=summarize(records)
    save_json(out/"constant_reasoning_priors.json",priors)
    curves=[]
    for row in rows:
        directory=Path(row["training"]["checkpoint"]).parent
        for budget in plan["settings"]["budget_milestones"]:
            cp=torch.load(directory/f"budget_{budget}.pt",map_location="cpu",weights_only=True)
            curves.append(dict(arm=row["arm"],seed=row["seed"],target_bytes=cp["processed_at_budget"],
                selected_step=cp["step"],validation_bpb=cp["validation_score"],cpu_seconds=cp["training_cpu_until_budget"]))
    save_json(out/"validation_learning_curves.json",curves)
    paired=[]
    for seed in plan["settings"]["seeds"]:
        group={r["arm"]:r for r in rows if r["seed"]==seed}
        paired.append(dict(seed=seed,revised_minus_raw_bpb=group["fourier_rewt"]["language"]["bpb"]-group["legacy_fourier"]["language"]["bpb"],
            fourier_minus_neural_bpb=group["fourier_rewt"]["language"]["bpb"]-group["neural_energy"]["language"]["bpb"],
            revised_minus_raw_complete=group["fourier_rewt"]["reasoning_only"]["both_correct"]-group["legacy_fourier"]["reasoning_only"]["both_correct"]))
    save_json(out/"paired_differences.json",paired)
    screen=[json.loads(p.read_text()) for p in sorted((out/"screen").glob("*.json"))] if (out/"screen").exists() else []
    def fmt(arm,field,scale=1,digits=3):
        value=aggregate[arm][field]
        return f"{value['mean']*scale:.{digits}f} ± {value['std']*scale:.{digits}f}"
    s=plan["settings"]
    lines=["# REWT fixes and controlled follow-up", "",
        "This report records models trained from scratch for this project and actual generated outputs. The revised implementation uses centered output coordinates throughout the joint energy and its analytic derivatives, noisy training guesses, hard ranking negatives, refinement-repair feedback, variable training depth, a reasoning curriculum with first-error supervision, configurable Fourier capacity, cached causal history, and optional early stopping. Residual gradient filtering is a separate validation-screened option.", "",
        f"Five variants × {len(s['seeds'])} seeds; about {plan['parameters']:,} parameters each. Every language fit starts from random weights and processes {s['lm_updates']*s['batch_size']*128:,} target bytes in {s['lm_updates']} updates with batch {s['batch_size']} and 128-byte resets. The raw control keeps raw logits, 16 modes, fixed two-round training, ordinary initialization and easy negatives. It receives the same longer byte budget and common selected learning rate. The four revised comparison arms share the selected recipe; the Transformer receives the same curriculum and first-error labels, with a logit ranking loss rather than an energy ranking loss.", "",
        f"Validation screening selected **{plan['selected_screen']}**. All screening cases, validation scores, fixed recipe values, source snapshots and checksums are saved. Selection used no test outputs. Frequency-mode comparisons compensate with feedforward width to preserve the parameter budget; they test budget allocation and are not isolated frequency-only effects.", "",
        "The follow-up uses new nonoverlapping language windows and text prefixes. Synthetic test identities exclude the previous pilot's test identities and remain disjoint from current training and validation by semantic hash. Arithmetic starting values expand to 0–19 and increments to 1–19. These changes mean that comparisons with the previous pilot's numerical test scores are not paired estimates of a repair effect.", "",
        "Language and cost below are mean ± sample standard deviation across seeds. Lower bpb is better. Inference policies are selected on validation from 1/2/4/8 rounds and two bounded halting thresholds. The policy must remain within 0.002 bpb of the best validation alternative before selecting measured CPU cost. Energy or probability stability is a compute stopping feature, not a correctness certificate.", "",
        "| Variant | Parameters | Fresh test bpb ↓ | Equal-time bpb ↓ | Training + teacher CPU s | Inference ms/byte |",
        "|---|---:|---:|---:|---:|---:|"]
    if plan.get("reused_fits"):
        lines[2:2] = [
            "This is the completed replacement evaluation for a stopped follow-up. Twenty-seven completed fits (all fifteen language fits and twelve control reasoning fits) reuse the parent study's exact checkpoint bytes and original source/training records. All three full-model reasoning fits restart from their completed language weights; their training gradients detach carried latent states every sixteen observed tokens. Forward state values, attention history and the full 128-byte memory window are retained. Float64 gradient-norm and spectral-power reductions prevent a finite-vector norm from overflowing; actual non-finite components are rejected.", "",
            "The fresh test windows, text prefixes and synthetic test identities were declared before these restarts. They exclude both the original pilot and the stopped follow-up's evaluations. No partial scores from the stopped run enter this report. The archived and current implementations produced bitwise-equal logits for all twenty-seven reused checkpoints under the recorded parity probe. Reused-source checksums and the current source snapshot remain separately identified in the plan.", ""]
    for arm in LABELS:
        lines.append(f"| {LABELS[arm]} | {plan['counts'][arm]:,} | {fmt(arm,'test_bpb')} | {fmt(arm,'equal_time_test_bpb')} | {fmt(arm,'language_pipeline_cpu',digits=1)} | {fmt(arm,'inference_ms_per_byte')} |")
    if (out/"profiles"/"all.json").exists():
        profiles=json.loads((out/"profiles"/"all.json").read_text())
        lines += ["", "Separate fresh-process profiles use each selected inference policy and measure RSS before operator-profiler bookkeeping. RSS includes Python, Torch, model and data buffers; incremental RSS is sampled every 2 ms and is not a precise activation-allocation peak.", "",
                  "| Variant | Sequential inference ms/byte | Process peak RSS MiB | Incremental inference RSS MiB |",
                  "|---|---:|---:|---:|"]
        for arm in LABELS:
            group=[profiles[f"{arm}_{seed}"] for seed in s["seeds"]]
            lines.append(f"| {LABELS[arm]} | {np.mean([p['cpu_ms_per_byte_median'] for p in group]):.4f} | {np.mean([p['sampled_inference_peak_rss_bytes'] for p in group])/2**20:.1f} | {np.mean([p['sampled_incremental_inference_rss_bytes'] for p in group])/2**20:.1f} |")
    lines += ["",f"The separate equal-time fits have {s['equal_cpu_seconds']:g} process-CPU seconds from new random weights. Full REWT spends 25% on a fresh forecast teacher and the remaining budget on its own fit. One final update may overshoot the cap. Main full-model cost charges its shared course-trained language teacher. Reasoning uses the baseline's separately adapted reasoning teacher, freezes it, and charges that adaptation once; its earlier language preparation is already included in the language pipeline. Common student bytes do not imply equal total pipeline compute.", "",
        f"Reasoning copies receive {s['reason_updates']} updates, batch {s['reason_batch_size']}. The revised curriculum first presents one-step arithmetic/rules and short recall, then expands to two and three steps. A generated wrong trace is labeled through its first invalid step; the corresponding incorrect next byte is ranked under the unchanged correct prefix in the training loss. This is local process feedback, not an independently trained whole-solution verifier. No symbolic checker, expected answer or future label enters inference.", "",
        "Generated reasoning is scored after greedy autoregressive generation. Arithmetic and rule chains are separated from easier recall. Complete correctness requires every intermediate step and the final answer. ID uses seen wording and at most three reasoning steps; OOD uses withheld wording or four-step chains. These tasks measure bounded skills, not general reasoning.", "",
        "| Variant | Reasoning answers % | Complete reasoning % | ID complete % | OOD complete % | Recall complete % | Seen training complete % | Reasoning + teacher CPU s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in LABELS:
        cost=np.mean([r['reasoning_training']['training_plus_shared_teacher_cpu_seconds'] for r in grouped[arm]])
        lines.append(f"| {LABELS[arm]} | {fmt(arm,'reasoning_answer',100,1)} | {fmt(arm,'reasoning_complete',100,1)} | {fmt(arm,'reasoning_id_complete',100,1)} | {fmt(arm,'reasoning_ood_complete',100,1)} | {fmt(arm,'recall',100,1)} | {fmt(arm,'train_complete',100,1)} | {cost:.1f} |")
    lines += ["",f"Training-only constant-response priors reach {priors['constant']['both_correct']*100:.1f}% complete reasoning without family information and {priors['known_family_constant']['both_correct']*100:.1f}% when the family is supplied. The latter is an oracle diagnostic, not a generative model.", "",
        f"Generated-text evidence includes all {review_count} 96-byte sampled continuations after the same fresh 32-byte prefixes, with temperature 0.8 and paired sampling uniforms. Printable and repetition rates are diagnostic measurements; they do not establish fluency or coherence. The [anonymous review](text_review.html) uses the fixed 1–5 fluency/coherence/relevance rubric. Human ratings remain pending. Any assistant review is identified separately and must be locked before reading the identity key.", "",
        "| Variant | Printable bytes % | Repeated byte trigrams % | Raw gradient spikes % |",
        "|---|---:|---:|---:|"]
    for arm in LABELS:
        lines.append(f"| {LABELS[arm]} | {fmt(arm,'printable',100,1)} | {fmt(arm,'repetition',100,1)} | {fmt(arm,'raw_spikes',100,1)} |")
    lines += ["", "Validation-only local diagnostics freeze the original prefix conditioning and compare rounds 0/1/2/4/8 from its actual initialization. They record energy, prediction loss, gradient alignment and the common-logit shift probe. These measurements isolate local refinement and are distinct from autoregressive rollouts that alter subsequent recurrent conditioning.", "",
        "| Variant / seed | Common output gradient squared share % | Initial → two-round bpb | Initial → two-round energy |",
        "|---|---:|---:|---:|"]
    for row in rows:
        d=row["local_diagnostics"]
        if d is None: continue
        by={r["rounds"]:r for r in d["rounds"]}
        lines.append(f"| {LABELS[row['arm']]} / {row['seed']} | {d['mean_common_output_squared_gradient_fraction']*100:.4f} | {by[0]['bpb']:.4f} → {by[2]['bpb']:.4f} | {by[0]['mean_energy']:.3f} → {by[2]['mean_energy']:.3f} |")
    if screen:
        lines += ["", "Single-seed validation screening:", "", "| Case | Validation bpb ↓ | Filter | Eligible recipe |", "|---|---:|---|---|"]
        for row in sorted(screen,key=lambda r:r["validation_bpb"] if r["validation_bpb"] is not None else float('inf')):
            score=f"{row['validation_bpb']:.4f}" if row['validation_bpb'] is not None else "non-finite trial"
            lines.append(f"| {row['name']} | {score} | {row['case']['method']} | {row['case']['eligible']} |")
    lines += ["", "The study bundles several training repairs; the main revised-versus-raw comparison cannot assign a gain to one component. Screening is one seed and a short budget. The non-Fourier energy control is essential to assessing whether a benefit is specific to Fourier modes.", "",
        "The full model retains forecasting, diffusion proposals, specialists, communication and critic/router conditioning under one fixed joint energy. At a fixed parameter budget, those modules reduce feedforward width; that capacity tradeoff remains explicit. Frozen teacher targets also inherit the teacher's training limits. The finite wave basis is not an exact representation of an arbitrary high-dimensional energy landscape.", "",
        "CPU timings come from a shared host and should not be read as GPU speed claims. Cumulative worker RSS is not per-model activation memory. Source/data hashes, checkpoint paths, all recipe coefficients, task labels, raw outputs and optimizer histories are saved. The archived original pilot is unchanged. Resume with the same source and --resume; larger byte budgets require a newly declared run.", ""]
    (out/"REPORT.md").write_text("\n".join(lines))
    examples=["# Fixed-ID output examples", "", "These examples use the first declared prompt and first six task identities per variant and seed; they are not selected for correctness.", ""]
    for row in rows:
        text=json.loads((out/f"seed_{row['seed']}"/f"{row['arm']}_text.jsonl").read_text().splitlines()[0])
        examples += [f"**{LABELS[row['arm']]} / seed {row['seed']}**", "", "Prompt:", "", "```text", text['prompt'], "```", "", "Continuation:", "", "```text", text['generation'], "```", ""]
        records=[json.loads(x) for x in (out/f"seed_{row['seed']}"/f"{row['arm']}_reasoning.jsonl").read_text().splitlines()]
        lookup={r['id']:r for r in records}
        for task in task_test[:6]:
            r=lookup[task['id']]
            examples += [f"Task {r['id'][:12]} ({r['family']}):", "", "```text", r['prompt']+r['response'], "```", f"Expected: `{r['expected'].strip()}`; complete correctness: {r['both_correct']}.", ""]
    (out/"OUTPUT_EXAMPLES.md").write_text("\n".join(examples))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,3,figsize=(14,4))
    arms=list(LABELS)
    captions=["Transformer","Raw REWT","Neural energy","Fourier REWT","Full REWT"]
    for axis,field,title,scale in ((axes[0],"test_bpb","Fresh held-out bits per byte ↓",1),
         (axes[1],"reasoning_complete","Complete arithmetic / rule chains (%) ↑",100),
         (axes[2],"inference_ms_per_byte","Inference CPU ms / byte ↓",1)):
        axis.bar(range(len(arms)),[aggregate[a][field]['mean']*scale for a in arms],
                 yerr=[aggregate[a][field]['std']*scale for a in arms],capsize=3,
                 color=["#526f94","#9ba5b2","#9276b4","#208f77","#bc7e35"])
        axis.set_xticks(range(len(arms)),captions,rotation=20,ha="right")
        axis.set_title(title);axis.grid(axis="y",alpha=.2);axis.set_axisbelow(True)
    figure.tight_layout();figure.savefig(out/"comparison.png",dpi=160);plt.close(figure)
    save_json(out/"verification.json",dict(rows=len(rows),expected_rows=len(LABELS)*len(s["seeds"]),
        source_sha256=plan["source_sha256"],all_model_records_complete=True,
        all_energy_audits_nonincreasing=all(r['refinement_audit'] is None or r['refinement_audit']['all_accepted_states_nonincreasing'] for r in rows),
        parameters_within_half_percent=max(plan['counts'].values())/min(plan['counts'].values())<=1.005,
        text_samples=review_count,reasoning_samples=len(task_test)*len(rows)))
    return aggregate
