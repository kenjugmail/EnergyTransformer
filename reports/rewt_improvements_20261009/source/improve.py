"""Validation-selected REWT repairs with fresh held-out evaluation samples.

All model weights originate in this experiment. The original pilot remains
archived. Screening touches validation only; no test outputs select a recipe.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import platform
import shutil
import os
import subprocess
import sys

import numpy as np
import torch
from torch.nn import functional as F

from .data import Corpus, file_sha256
from .evaluate import (language_metrics, model_output, reasoning_evaluation, refinement_audit,
                       runtime_profile, task_loss, text_evaluation)
from .experiment import (event, load_model, new_model, save_json, save_lines, train)
from .model import Conditioning, JointEnergyLM, build_model, matched_configurations
from .tasks import corrupt_trace, generate_tasks, task_manifest


LABELS = dict(transformer="Transformer", legacy_fourier="Raw REWT control",
              neural_energy="Revised neural energy", fourier_rewt="Revised Fourier REWT",
              full_rewt="Revised full REWT")
REPAIR_RECIPE = dict(variable_rounds=True, minimum_rounds=1, maximum_rounds=4,
                     corruption_probability=.25, output_noise=.5, state_noise=.05,
                     hard_negatives=True, repair_weight=.1, curriculum=True,
                     process_supervision=True, process_weight=.05,
                     trace_training_energy=False, residual_gain=.25)


def worker_setup(threads):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def nonoverlapping_offsets(corpus, split, count, context, seed, exclusions):
    selected = []
    candidates = corpus.offsets(split, count * 100, context, seed)
    for candidate in candidates:
        if any(abs(candidate - old) <= context for old in (*exclusions, *selected)):
            continue
        selected.append(candidate)
        if len(selected) == count:
            return selected
    raise ValueError("Could not reserve the required disjoint evaluation windows")


def configurations(parameters, *, modes=32, frequency_bound=2., step_size=.25, geometry="centered",
                   normalized=True):
    configs = matched_configurations(target=parameters, modes=modes, frequency_bound=frequency_bound,
        output_geometry=geometry, normalize_frequency_blocks=normalized, cache_history=True,
        early_backtrack=True, step_size=step_size)
    return {arm: replace(c, backtracks=4) if arm != "transformer" else c for arm,c in configs.items()}


def screening_cases(parameters, learning_rate):
    cases = {}
    def add(name, *, modes=32, bound=2., step=.25, geometry="centered", normalized=True,
            repaired=True, method="clip", lr=None):
        config = configurations(parameters, modes=modes, frequency_bound=bound, step_size=step,
                                geometry=geometry, normalized=normalized)["fourier_rewt"]
        cases[name] = dict(config=asdict(config), recipe=dict(REPAIR_RECIPE) if repaired else
                          {"trace_training_energy": False}, method=method,
                          learning_rate=learning_rate if lr is None else lr,
                          eligible=repaired and method not in ("fourier","ema","none"))
    add("raw_16", modes=16, geometry="raw", normalized=False, repaired=False)
    add("centered_16", modes=16, normalized=False, repaired=False)
    add("repair_16", modes=16)
    add("repair_32")
    add("repair_64", modes=64)
    add("repair_32_small_step", step=.125)
    add("repair_32_low_band", bound=1.)
    add("repair_32_high_band", bound=4.)
    add("repair_32_residual_filter", method="fourier_residual")
    add("repair_32_ema_residual", method="ema_residual")
    add("repair_32_replacement_filter", method="fourier")
    add("repair_32_ema_replacement", method="ema")
    add("repair_32_no_clip", method="none")
    add("repair_32_faster_lr", lr=learning_rate * 2)
    return cases


def freeze(args):
    out, runs = Path(args.out).resolve(), Path(args.runs).resolve()
    existing = out / "plan.json"
    if existing.exists():
        if not args.resume:
            raise ValueError("This output exists; use --resume or a fresh directory")
        plan = json.loads(existing.read_text())
        for name,value in plan["source_sha256"].items():
            if file_sha256(Path(__file__).parent / name) != value:
                raise ValueError(f"Frozen source changed: {name}")
        return plan
    if (out.exists() and any(out.iterdir())) or (runs.exists() and any(runs.iterdir())):
        raise ValueError("Fresh output and checkpoint directories must be empty")
    corpus = Corpus(args.data)
    previous_path = Path(args.previous) / "plan.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    old_test_tasks = previous.get("task_manifests", {}).get("test", {}).get("semantic_ids", [])
    settings = dict(seeds=args.seeds, threads=args.threads, context=128, batch_size=args.batch_size,
        reason_batch_size=args.reason_batch_size, lm_updates=args.lm_updates, reason_updates=args.reason_updates,
        screen_updates=args.screen_updates, validate_every=args.validate_every, warm_updates=min(16,args.lm_updates//4),
        task_train=args.task_train, task_valid=args.task_valid, task_test=args.task_test,
        task_number_range=20, task_seed=1729, learning_rate=args.learning_rate,
        equal_cpu_seconds=args.equal_cpu_seconds,
        budget_milestones=sorted(set((max(1,args.lm_updates//4),args.lm_updates))))
    tasks = {split: generate_tasks(split,count,seed=1729,number_range=20,
             exclude_ids=old_test_tasks if split == "test" else ())
             for split,count in (("train",args.task_train),("valid",args.task_valid),("test",args.task_test))}
    ids = {split:{t.id for t in group} for split,group in tasks.items()}
    assert not ids["train"] & ids["valid"] and not ids["train"] & ids["test"] and not ids["valid"] & ids["test"]
    assert not ids["test"] & set(old_test_tasks)
    old_text = [p["offset"] for p in previous.get("text_prompts", [])]
    old_test_windows=previous.get("test_offsets",[])+old_text
    text_seed = 1730
    while True:
        prompts = corpus.text_prompts(count=args.text_prompts, seed=text_seed)
        if all(all(abs(p["offset"]-old)>128 for old in old_test_windows) for p in prompts):
            break
        text_seed += 1
    package = Path(__file__).parent
    source = {p.name:file_sha256(p) for p in sorted(package.glob("*.py"))}
    plan = dict(version=2, created=datetime.now(timezone.utc).isoformat(), out=str(out), runs=str(runs),
        settings=settings, parameters=args.parameters, corpus=corpus.identity(), source_sha256=source,
        validation_offsets=nonoverlapping_offsets(corpus,"valid",args.validation_windows,128,1801,
                                                  previous.get("validation_offsets",[])),
        test_offsets=nonoverlapping_offsets(corpus,"test",args.evaluation_windows,128,1802,
                                            old_test_windows),
        text_prompts=prompts, text_selection_seed=text_seed,
        previous_test_exclusions=old_test_tasks,
        task_manifests={split:task_manifest(group) for split,group in tasks.items()},
        screening_cases=screening_cases(args.parameters,args.learning_rate),
        screen_enabled=not args.skip_screen,
        selection_rule="Lowest validation bpb among repaired candidates; raw and centered-only are diagnostic controls",
        inference_policy_rule="Fastest measured validation policy within 0.002 bpb of the best; thresholds 0, 0.001, 0.005 with max rounds 8; fixed round alternatives 1,2,4,8",
        fixed_rounds=[1,2,4,8], halting_tolerances=[.001,.005], policy_bpb_tolerance=.002,
        device="cpu", torch_version=str(torch.__version__), python_version=platform.python_version(),
        protocol="Fresh evaluation samples, random initialization, validation-only selection, no pretrained weights")
    plan["plan_sha256"] = digest(plan)
    out.mkdir(parents=True,exist_ok=True)
    runs.mkdir(parents=True,exist_ok=True)
    snapshot = out / "source"
    snapshot.mkdir()
    for p in package.glob("*.py"):
        shutil.copy2(p,snapshot/p.name)
    save_json(existing,plan)
    for split,group in tasks.items():
        save_lines(out/f"tasks_{split}.jsonl",[t.to_dict() for t in group])
    save_lines(out/"process_training_labels.jsonl",[dict(id=t.id,**corrupt_trace(t,seed=1810)) for t in tasks["train"]])
    save_json(out/"text_prompts.json",prompts)
    return plan


def fit_screen(plan,name):
    case = plan["screening_cases"][name]
    local = dict(plan)
    local["settings"] = {**plan["settings"], "lm_updates":plan["settings"]["screen_updates"],
                         "learning_rate":case["learning_rate"], "budget_milestones":[],
                         "validate_every":max(1,plan["settings"]["screen_updates"]//2)}
    seed = plan["settings"]["seeds"][0]
    model = new_model("fourier_rewt",case["config"],seed)
    event("screen_case_started",name=name)
    try:
        model,summary = train(model,"fourier_rewt",seed,"validation_screen",local,Corpus(plan["corpus"]["path"]),
            Path(plan["runs"])/"screen"/name,recipe=case["recipe"],method=case["method"],
            updates=local["settings"]["lm_updates"],warm_updates=min(8,local["settings"]["lm_updates"]//4))
    except FloatingPointError as error:
        record=dict(name=name,case=case,validation_bpb=None,failed=str(error),
                    evidence_scope="Non-finite validation-screen trial; ineligible for selection")
        save_json(Path(plan["out"])/"screen"/f"{name}.json",record)
        return record
    record = dict(name=name,case=case,training=summary,
        validation_bpb=summary["best_validation_score"], evidence_scope="Single-seed validation screening; no test evaluation")
    save_json(Path(plan["out"])/"screen"/f"{name}.json",record)
    return record


def select_protocol(plan,screen):
    if screen:
        selected = min((r for r in screen if r["case"]["eligible"] and r["validation_bpb"] is not None),
                       key=lambda r:(r["validation_bpb"],r["name"]))
        name,case = selected["name"],selected["case"]
    else:
        name,case = "repair_32",plan["screening_cases"]["repair_32"]
    cfg = case["config"]
    configs = configurations(plan["parameters"],modes=cfg["modes"],frequency_bound=cfg["frequency_bound"],
                             step_size=cfg["step_size"])
    legacy = matched_configurations(target=plan["parameters"])["fourier_rewt"]
    variants = {arm:dict(arm=arm,config=asdict(c),recipe=case["recipe"],method=case["method"])
                for arm,c in configs.items()}
    variants["legacy_fourier"] = dict(arm="fourier_rewt",config=asdict(replace(legacy,backtracks=2)),
                                      recipe={"trace_training_energy":False},method="clip")
    variants = {name:variants[name] for name in LABELS}
    counts = {name:new_model(v["arm"],v["config"],1).parameters_count() for name,v in variants.items()}
    assert max(counts.values())/min(counts.values()) <= 1.005
    selected_plan = {**plan,"settings":{**plan["settings"],"learning_rate":case["learning_rate"]},
                     "selected_screen":name,"variants":variants,"counts":counts}
    selected_plan["plan_sha256"] = digest({k:v for k,v in selected_plan.items() if k != "plan_sha256"})
    save_json(Path(plan["out"])/"selected_protocol.json",selected_plan)
    return selected_plan


def with_policy(model,arm,rounds,halting=0.):
    config = replace(model.config,rounds=rounds,halting_tolerance=halting)
    copy = build_model(arm,config)
    copy.load_state_dict(model.state_dict(),strict=True)
    return copy.eval()


def policy_selection(model,arm,plan,corpus,*,tasks=None):
    if not isinstance(model,JointEnergyLM):
        return model,dict(rounds=None,halting_tolerance=0.,validation=[])
    values = []
    settings = plan["settings"]
    for rounds,halting in [(r,0.) for r in plan["fixed_rounds"]]+[(8,t) for t in plan["halting_tolerances"]]:
        copy = with_policy(model,arm,rounds,halting)
        if tasks is None:
            metric = language_metrics(copy,corpus,plan["validation_offsets"],batch_size=settings["batch_size"],seed=1820)
            score,cpu = metric["bpb"],metric["cpu_seconds"]
        else:
            import time
            started=time.process_time()
            metric=task_loss(copy,tasks,batch_size=settings["reason_batch_size"],seed=1821)
            score,cpu=metric["answer_bpb"],time.process_time()-started
        values.append(dict(rounds=rounds,halting_tolerance=halting,bpb=score,cpu_seconds=cpu,metrics=metric))
    best=min(r["bpb"] for r in values)
    eligible=[r for r in values if r["bpb"] <= best+plan["policy_bpb_tolerance"]]
    chosen=min(eligible,key=lambda r:(r["cpu_seconds"],r["rounds"],r["halting_tolerance"]))
    return with_policy(model,arm,chosen["rounds"],chosen["halting_tolerance"]),dict(
        rounds=chosen["rounds"],halting_tolerance=chosen["halting_tolerance"],best_validation_bpb=best,validation=values)


def local_diagnostics(model,corpus,offsets,seed):
    """Read-only local trajectories under frozen original-prefix conditioning."""
    if not isinstance(model,JointEnergyLM):
        return None
    batch=corpus.batch("valid",None,len(offsets),128,starts=offsets)
    with torch.inference_mode():
        output=model_output(model,batch.tokens,torch.Generator().manual_seed(seed),details=True)
        initial=output.details["initial_joint"].flatten(0,1)
        condition=output.details["conditioning"]
        cond=Conditioning(condition.anchor.flatten(0,1),condition.amplitudes.flatten(0,1),
                          condition.routing.flatten(0,1) if condition.routing is not None else None,
                          condition.features.flatten(0,1),{},condition.frequency,condition.precision)
        target=batch.targets.flatten()
        sl=model.blocks["output"]
        gradient=model.gradient(initial,cond)[:,sl]
        common=256*gradient.mean(-1).square()/gradient.square().sum(-1).clamp_min(1e-12)
        ce_gradient=initial[:,sl].softmax(-1)-F.one_hot(target,256)
        cosine=F.cosine_similarity(gradient,ce_gradient,dim=-1)
        shifted=initial.clone();shifted[:,sl]+=1
        probability_delta=float((initial[:,sl].softmax(-1)-shifted[:,sl].softmax(-1)).abs().max())
        energy_delta=float((model.energy(shifted,cond)-model.energy(initial,cond)).mean())
        values=[]
        for rounds in (0,1,2,4,8):
            state,_,_=model.refine(initial,cond,rounds=rounds)
            nll=F.cross_entropy(state[:,sl],target)/np.log(2)
            values.append(dict(rounds=rounds,bpb=float(nll),mean_energy=float(model.energy(state,cond).mean())))
    return dict(positions=batch.tokens.numel(),scope="Fixed previous states and conditioning; not full rollout",
        mean_common_output_squared_gradient_fraction=float(common.mean()),
        mean_output_ce_gradient_cosine=float(cosine.mean()),
        fraction_first_order_ce_descent=float((cosine>0).float().mean()),
        common_shift_max_probability_change=probability_delta,
        common_shift_mean_energy_change=energy_delta,
        rounds=values)


def get_tasks(plan,split):
    s=plan["settings"]
    return generate_tasks(split,s[f"task_{split}"],seed=s["task_seed"],number_range=s["task_number_range"],
                          exclude_ids=plan["previous_test_exclusions"] if split=="test" else ())


def fit_seed(plan,seed):
    corpus=Corpus(plan["corpus"]["path"])
    base=Path(plan["runs"])/f"seed_{seed}"
    out=Path(plan["out"])/f"seed_{seed}"
    tasks=get_tasks(plan,"train")
    valid_tasks=get_tasks(plan,"valid")
    test_tasks=get_tasks(plan,"test")
    models,reason_models,metadata,reason_metadata={}, {}, {}, {}
    for name,variant in plan["variants"].items():
        model=new_model(variant["arm"],variant["config"],seed)
        teacher=models.get("transformer") if name=="full_rewt" else None
        model,summary=train(model,variant["arm"],seed,"language",plan,corpus,base/name/"language",
                            recipe=variant["recipe"],method=variant["method"],teacher=teacher)
        models[name],metadata[name]=model,summary
    for name,variant in plan["variants"].items():
        copy=new_model(variant["arm"],variant["config"],seed)
        copy,summary=train(copy,variant["arm"],seed,"reasoning",plan,corpus,base/name/"reasoning",
            recipe=variant["recipe"],method=variant["method"],tasks=tasks,
            initial_weights=models[name].state_dict(),updates=plan["settings"]["reason_updates"],warm_updates=0,
            teacher=reason_models.get("transformer") if name=="full_rewt" else None,
            teacher_charge=reason_metadata["transformer"]["training_cpu_seconds"] if name=="full_rewt" else 0.)
        reason_models[name],reason_metadata[name]=copy,summary
    for name,variant in plan["variants"].items():
        saved=out/f"{name}.json"
        if saved.exists():
            continue
        model,language_policy=policy_selection(models[name],variant["arm"],plan,corpus)
        reason,reason_policy=policy_selection(reason_models[name],variant["arm"],plan,corpus,tasks=valid_tasks)
        metrics=language_metrics(model,corpus,plan["test_offsets"],batch_size=plan["settings"]["batch_size"],seed=1830)
        text,records=text_evaluation(model,plan["text_prompts"],seed=seed+1831)
        save_lines(out/f"{name}_text.jsonl",records)
        reasoning,records=reasoning_evaluation(reason,test_tasks,seed=seed+1832)
        save_lines(out/f"{name}_reasoning.jsonl",records)
        training_reasoning,records=reasoning_evaluation(reason,tasks[:min(96,len(tasks))],seed=seed+1833)
        save_lines(out/f"{name}_training_reasoning.jsonl",records)
        equal= new_model(variant["arm"],variant["config"],seed)
        teacher,teacher_charge,teacher_fit=None,0.,None
        budget=plan["settings"]["equal_cpu_seconds"]
        if name=="full_rewt":
            teacher=new_model("transformer",plan["variants"]["transformer"]["config"],seed)
            teacher,teacher_fit=train(teacher,"transformer",seed,"equal_time_teacher",plan,corpus,
                base/name/"equal_time_teacher",updates=20_000,cpu_budget=budget*.25,warm_updates=4,
                recipe=variant["recipe"],method=variant["method"])
            teacher_charge=teacher_fit["training_cpu_seconds"]
        equal,equal_fit=train(equal,variant["arm"],seed,"equal_time",plan,corpus,base/name/"equal_time",
            updates=20_000,cpu_budget=max(.1,budget-teacher_charge),warm_updates=4,
            recipe=variant["recipe"],method=variant["method"],teacher=teacher,teacher_charge=teacher_charge)
        audit_batch=corpus.batch("valid",None,2,16,starts=plan["validation_offsets"][:2])
        row=dict(arm=name,physical_arm=variant["arm"],seed=seed,training=metadata[name],
            reasoning_training=reason_metadata[name],language=metrics,language_policy=language_policy,
            text=text,adapted_reasoning=reasoning,training_reasoning=training_reasoning,
            reasoning_policy=reason_policy,reasoning_teacher_forced=task_loss(reason,test_tasks),
            equal_cpu_time=dict(training=equal_fit,teacher_training=teacher_fit,
                language=language_metrics(equal,corpus,plan["test_offsets"],batch_size=plan["settings"]["batch_size"],seed=1834)),
            refinement_audit=refinement_audit(model,audit_batch.tokens),
            local_diagnostics=local_diagnostics(models[name],corpus,plan["validation_offsets"][:4],seed=1835))
        save_json(saved,row)
        event("revised_evaluation",seed=seed,variant=name,test_bpb=metrics["bpb"],reasoning=reasoning["all"])
    save_json(out/"complete.json",dict(seed=seed,completed=True))
    return dict(seed=seed,completed=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",default="data/enwik8")
    parser.add_argument("--previous",default="reports/rewt_pilot_20261009_v2")
    parser.add_argument("--out",default="reports/rewt_improvements_20261009")
    parser.add_argument("--runs",default="runs/rewt_improvements_20261009")
    parser.add_argument("--seeds",type=int,nargs="+",default=[101,202,303])
    parser.add_argument("--workers",type=int,default=3)
    parser.add_argument("--threads",type=int,default=1)
    parser.add_argument("--parameters",type=int,default=250_000)
    parser.add_argument("--batch-size",type=int,default=32)
    parser.add_argument("--reason-batch-size",type=int,default=16)
    parser.add_argument("--lm-updates",type=int,default=256)
    parser.add_argument("--screen-updates",type=int,default=64)
    parser.add_argument("--reason-updates",type=int,default=800)
    parser.add_argument("--learning-rate",type=float,default=.002)
    parser.add_argument("--validate-every",type=int,default=64)
    parser.add_argument("--task-train",type=int,default=6144)
    parser.add_argument("--task-valid",type=int,default=192)
    parser.add_argument("--task-test",type=int,default=300)
    parser.add_argument("--validation-windows",type=int,default=16)
    parser.add_argument("--evaluation-windows",type=int,default=32)
    parser.add_argument("--text-prompts",type=int,default=30)
    parser.add_argument("--equal-cpu-seconds",type=float,default=60.)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--skip-screen",action="store_true")
    args=parser.parse_args(argv)
    if any(v<1 for v in (args.workers,args.threads,args.parameters,args.batch_size,args.reason_batch_size,
                         args.lm_updates,args.screen_updates,args.reason_updates,args.validate_every,
                         args.task_train,args.task_valid,args.task_test,args.validation_windows,
                         args.evaluation_windows,args.text_prompts)):
        parser.error("Counts must be positive")
    if args.equal_cpu_seconds<=0 or args.learning_rate<=0 or len(set(args.seeds))!=len(args.seeds):
        parser.error("Positive time/LR and distinct seeds required")
    torch.set_num_threads(args.threads)
    plan=freeze(args)
    context=multiprocessing.get_context("spawn")
    screen=[]
    if plan["screen_enabled"]:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=context,
                                 initializer=worker_setup,initargs=(plan["settings"]["threads"],)) as pool:
            futures=[pool.submit(fit_screen,plan,name) for name in plan["screening_cases"]]
            for future in as_completed(futures):
                row=future.result();screen.append(row)
                event("screen_complete",name=row["name"],validation_bpb=row["validation_bpb"])
    selected=select_protocol(plan,screen)
    event("revised_protocol_selected",selected=selected["selected_screen"],counts=selected["counts"],settings=selected["settings"])
    with ProcessPoolExecutor(max_workers=min(args.workers,len(selected["settings"]["seeds"])),mp_context=context,
                             initializer=worker_setup,initargs=(selected["settings"]["threads"],)) as pool:
        futures=[pool.submit(fit_seed,selected,seed) for seed in selected["settings"]["seeds"]]
        for future in as_completed(futures):
            event("revised_seed_complete",**future.result())
    from .improve_report import create_report
    subprocess.run([sys.executable,"-m","rewt.profile",plan["out"]],check=True,
                   env={**os.environ,"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1"})
    create_report(selected)
    save_json(Path(plan["out"])/"complete.json",dict(completed=datetime.now(timezone.utc).isoformat(),
              seeds=selected["settings"]["seeds"],all_primary_fits_complete=True))
    event("revised_experiment_complete",report=str(Path(plan["out"])/"REPORT.md"))


if __name__=="__main__":
    main()
