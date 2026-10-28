import json
from pathlib import Path

import numpy as np
import torch

from rewt.data import Batch
from rewt.evaluate import generate_batch, language_metrics, repetition_metrics
from rewt.experiment import train, load_model, new_model
from rewt.model import TransformerConfig


class TinyCorpus:
    def batch(self, split, rng, batch, context, starts=None):
        if rng is None:
            rng = np.random.default_rng(sum(starts))
        raw = rng.integers(97, 104, (batch, context + 1))
        tokens = torch.tensor(raw, dtype=torch.long)
        return Batch(tokens[:, :-1], tokens[:, 1:])


def plan_for(tmp_path):
    return dict(runs=str(tmp_path), source_sha256={"test": "test"}, plan_sha256="test",
        validation_offsets=[90_000_001, 90_000_030],
        settings=dict(context=8, batch_size=2, reason_batch_size=2, warm_updates=2,
                      lm_updates=6, learning_rate=.002, task_valid=12, validate_every=1))


def test_optimizer_resume_replays_the_same_fit(tmp_path):
    torch.set_num_threads(1)
    cfg = TransformerConfig(width=8, layers=1, heads=2, feedforward=16, context=8)
    plan = plan_for(tmp_path)
    full = new_model("transformer", cfg.__dict__, 701)
    full, full_summary = train(full, "transformer", 701, "language", plan,
                              TinyCorpus(), tmp_path / "continuous", updates=6)
    part = new_model("transformer", cfg.__dict__, 701)
    train(part, "transformer", 701, "language", plan, TinyCorpus(),
          tmp_path / "resumed", updates=3)
    (tmp_path / "resumed" / "training.json").unlink()
    resumed = new_model("transformer", cfg.__dict__, 999)
    resumed, resumed_summary = train(resumed, "transformer", 701, "language", plan,
                                    TinyCorpus(), tmp_path / "resumed", updates=6)
    for key, value in full.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
    assert full_summary["best_step"] == resumed_summary["best_step"]
    assert resumed_summary["updates"] == 6
    assert resumed_summary["processed_target_bytes"] == 6 * 2 * 8
    rows = [json.loads(row) for row in (tmp_path / "resumed" / "trace.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == list(range(1, 7))
    loaded, checkpoint = load_model(tmp_path / "resumed" / "best.pt")
    assert checkpoint["plan_sha256"] == "test"


def test_time_cap_stops_at_a_complete_update_and_saves_a_model(tmp_path):
    torch.set_num_threads(1)
    cfg = TransformerConfig(width=8, layers=1, heads=2, feedforward=16, context=8)
    model = new_model("transformer", cfg.__dict__, 701)
    model, summary = train(model, "transformer", 701, "budget", plan_for(tmp_path),
                          TinyCorpus(), tmp_path / "budget", updates=100, cpu_budget=1e-9)
    assert summary["updates"] == 1
    assert Path(summary["checkpoint"]).exists()
    measured = language_metrics(model, TinyCorpus(), [95_000_001, 95_000_030],
                                context=8, batch_size=2)
    assert np.isfinite(measured["bpb"])
    assert measured["evaluated_bytes"] == 16


def test_generation_enforces_the_common_context_and_saved_diagnostics():
    torch.set_num_threads(1)
    model = new_model("transformer", TransformerConfig().__dict__, 702)
    one = generate_batch(model, [b"a b c", b"d e f"], max_new=4, sampling=True, seed=703)
    two = generate_batch(model, [b"a b c", b"d e f"], max_new=4, sampling=True, seed=703)
    assert one == two
    assert len(one[0]) == 4
    repeated = repetition_metrics(b"aaaaaaaa")
    assert repeated["repeated_byte_trigram_fraction"] == 5 / 6
    assert repeated["printable_fraction"] == 1
