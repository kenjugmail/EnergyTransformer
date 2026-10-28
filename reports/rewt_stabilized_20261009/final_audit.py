"""Read-only model audit; save final report checksums without changing weights."""
import ast
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

P = Path(__file__).resolve().parent
ROOT = P.parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    plan = read(P / "plan.json")
    verification = read(P / "verification.json")
    diag = read(P / "diagnostics.json")
    seeds, arms = plan["settings"]["seeds"], list(plan["variants"])
    assert len(seeds) == 3 and len(arms) == 5
    for name, expected in plan["source_sha256"].items():
        assert sha(ROOT / "rewt" / name) == expected == sha(P / "source" / name)
        ast.parse((ROOT / "rewt" / name).read_text())
    parent = Path(plan["parent_stopped_report"])
    ledger = read(P / "reused_fits.json")
    assert len(ledger) == 27
    for item in ledger:
        old, new = Path(item["original_checkpoint"]), Path(item["copied_checkpoint"])
        assert sha(old) == sha(new) == item["checkpoint_sha256"]
        old_record = Path(item["original_directory"]) / "training.json"
        assert sha(old_record) == sha(new.parent / "training.json") == item["training_record_sha256"]
        for name, expected in item["source_sha256"].items():
            assert sha(parent / "source" / name) == expected
    parity = read(P / "reused_inference_parity.json")
    assert len(parity) == 27 and all(x["exact_logits_equal"] for x in parity)
    rows, checkpoints = {}, set()

    def collect_checkpoints(value):
        if isinstance(value, dict):
            for v in value.values():
                collect_checkpoints(v)
        elif isinstance(value, list):
            for v in value:
                collect_checkpoints(v)
        elif isinstance(value, str) and value.startswith("/") and value.endswith(".pt"):
            path = Path(value)
            assert path.is_file(), str(path)
            checkpoints.add(path)

    for seed in seeds:
        assert read(P / f"seed_{seed}/complete.json")
        for arm in arms:
            r = read(P / f"seed_{seed}/{arm}.json")
            assert r["seed"] == seed and r["arm"] == arm
            assert r["training"]["processed_target_bytes"] == 1048576
            assert r["training"]["updates"] == 256 and r["reasoning_training"]["updates"] == 800
            if arm != "transformer":
                assert r["refinement_audit"]["all_accepted_states_nonincreasing"]
            for phase in ["training", "reasoning_training"]:
                assert math.isfinite(r[phase]["maximum_raw_gradient_norm"])
            for suffix, count in [("_text.jsonl", 30), ("_reasoning.jsonl", 300)]:
                assert len((P / f"seed_{seed}/{arm}{suffix}").read_text().splitlines()) == count
            rows[(arm, seed)] = r
            collect_checkpoints(r)
    profiles = read(P / "profiles/all.json")
    assert len(profiles) == 15
    assert len({r["isolated_process_id"] for r in profiles.values()}) == 15
    for key, r in profiles.items():
        assert r == read(P / "profiles" / f"{key}.json")
    gen = read(P / "generation_runtime.json")
    assert len(gen["records"]) == 15
    assert {(r["arm"], r["seed"]) for r in gen["records"]} == set(rows)
    for r in gen["records"]:
        assert sha(Path(rows[(r["arm"], r["seed"])]["training"]["checkpoint"])) == r["checkpoint_sha256"]
        assert {m["batch_size"] for m in r["measurements"]} == {1, 10}
        assert all(m["includes_prefix_prefill"] and len(m["cpu_seconds"]) == 3 for m in r["measurements"])
    signal = read(P / "energy_signal.json")["records"]
    assert len(signal) == 12
    for r in signal:
        assert sha(Path(rows[(r["arm"], r["seed"])]["training"]["checkpoint"])) == r["checkpoint_sha256"]
    for folder, count in [(P, 450), (P / "baseline_supplement", 90)]:
        lock = read(folder / "assistant_ratings_lock.json")
        cases = read(folder / "text_review_cases.json")
        ratings = read(folder / "assistant_text_ratings.json")["ratings"]
        assert lock["case_count"] == lock["rated_count"] == len(cases) == len(ratings) == count
        assert sha(folder / "text_review_cases.json") == lock["case_sha256"]
        assert sha(folder / "assistant_rating_batches.jsonl") == lock["batches_sha256"]
        assert {r["id"] for r in ratings} == {r["id"] for r in cases}
        assert all(c[field] is None for c in cases for field in ["fluency", "coherence", "relevance"])
        assert all(r["coherence"] == 1 for r in ratings)
        assert all(1 <= r[field] <= 5 for r in ratings for field in ["fluency", "coherence", "relevance"])
    sup = P / "baseline_supplement"
    sup_plan = read(sup / "plan.json")
    assert sha(P / "baseline_check.py") == sup_plan["supplement_script_sha256"]
    sup_gen = read(sup / "generation_runtime.json")
    assert len(sup_gen["records"]) == 3
    for seed in seeds:
        r = read(sup / f"seed_{seed}/transformer.json")
        collect_checkpoints(r)
        assert r["training"]["processed_target_bytes"] == 1048576 and r["training"]["updates"] == 256
        assert r["reasoning_training"]["updates"] == 800
        assert len((sup / f"seed_{seed}/transformer_text.jsonl").read_text().splitlines()) == 30
        assert len((sup / f"seed_{seed}/transformer_reasoning.jsonl").read_text().splitlines()) == 300
        gr = next(g for g in sup_gen["records"] if g["seed"] == seed)
        assert sha(Path(r["training"]["checkpoint"])) == gr["checkpoint_sha256"]
    assert verification["all_energy_audits_nonincreasing"] and verification["source_current_and_snapshot_unchanged"]
    for key in ["fresh_test_windows_and_prefixes_verified", "semantic_splits_disjoint", "previous_test_tasks_excluded", "common_training_bytes_verified", "selected_policies_within_declared_validation_tolerance", "reused_weights_and_original_training_records_unchanged", "reused_inference_bitwise_parity_verified", "assistant_rating_cases_and_batches_hashes_verified"]:
        assert verification[key], key
    assert diag["target_bytes_per_model"] == 1048576
    assert diag["fresh_test_offsets"] and diag["semantic_splits_disjoint"] and diag["prior_test_tasks_excluded"]
    stability = read(P / "stability_summary.json")
    assert len(stability) == 12 and all(x["steps"] == 800 and x["all_objectives_and_norms_finite"] for x in stability)
    findings, agg = (P / "FINDINGS.md").read_text(), read(P / "aggregate.json")
    labels = {"transformer": "Transformer", "legacy_fourier": "Raw REWT control", "neural_energy": "Revised neural energy", "fourier_rewt": "Revised Fourier REWT", "full_rewt": "Revised full REWT"}
    for arm, label in labels.items():
        stats = agg[arm]
        formatted = [f"{stats[k]['mean'] * scale:.{dec}f} ± {stats[k]['std'] * scale:.{dec}f}" for k, scale, dec in [("test_bpb", 1, 3), ("equal_time_test_bpb", 1, 3), ("reasoning_answer", 100, 1), ("reasoning_complete", 100, 1)]]
        expected = "| " + label + " | " + " | ".join(formatted) + " |"
        assert expected in findings, expected
    links_checked = 0
    documents = [P / name for name in ["FINDINGS.md", "IMPLEMENTATION.md", "REPORT.md", "DIAGNOSTICS.md", "STABILITY.md", "TEXT_QUALITY_REVIEW.md", "ENERGY_SIGNAL.md", "GENERATION_RUNTIME.md"]]
    documents += [sup / name for name in ["REPORT.md", "TEXT_QUALITY_REVIEW.md", "GENERATION_RUNTIME.md"]]
    for document in documents:
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", document.read_text()):
            if target.startswith(("http:", "https:", "codex:")):
                continue
            target = re.sub(r":\d+$", "", target.strip("<>").split("#", 1)[0])
            if not target:
                continue
            local = Path(target) if target.startswith("/") else document.parent / target
            assert local.exists(), str(local)
            links_checked += 1
    for helper in P.glob("*.py"):
        ast.parse(helper.read_text())
    manifest = {str(f.relative_to(P)): sha(f) for f in sorted(P.rglob("*")) if f.is_file() and f.name not in {"FINAL_VERIFICATION.json", "artifact_manifest.json"} and "__pycache__" not in f.parts}
    (P / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    result = dict(audited_at=datetime.now(timezone.utc).isoformat(), all_assertions_passed=True,
                  frozen_source_files=len(plan["source_sha256"]), reused_fits_sha_verified=len(ledger),
                  reused_archived_inference_parity_records=len(parity), primary_model_records=len(rows),
                  primary_reasoning_generations=4500, primary_text_samples=450,
                  supplemental_reasoning_generations=900, supplemental_text_samples=90,
                  assistant_rating_hashes_verified_for_all=540, human_ratings_pending=True,
                  isolated_teacher_forced_profiles=15, actual_generation_profiles=18,
                  finite_800_update_reasoning_traces=12, full_optimization_still_has_severe_finite_spikes=True,
                  quality_table_matches_saved_aggregate=True, report_file_links_checked=links_checked,
                  model_checkpoints_final_sha256={str(f): sha(f) for f in sorted(checkpoints)},
                  report_artifact_files_hashed=len(manifest), artifact_manifest_sha256=sha(P / "artifact_manifest.json"),
                  audit_script_sha256=sha(Path(__file__)),
                  pre_run_implementation_tests_passed=read(P / "PRE_RUN_VERIFICATION.json")["tests"]["passed"],
                  test_scope="Implementation tests previously passed with the same unchanged frozen source; the final audit verifies artifacts and does not claim an empirical quality gain.")
    (P / "FINAL_VERIFICATION.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "model_checkpoints_final_sha256"}, indent=2))
    print("checkpoint_files_final_sha_verified", len(checkpoints))


if __name__ == "__main__":
    main()
