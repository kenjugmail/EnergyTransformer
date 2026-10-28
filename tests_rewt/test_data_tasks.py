import torch

from rewt.data import task_batch
from rewt.tasks import generate_tasks, solve, score_response, task_manifest
from rewt.optim import GradientFilter


def test_task_splits_are_semantically_disjoint_and_true_answers_pass_checker():
    splits = {name: generate_tasks(name, 150) for name in ("train", "valid", "test")}
    identities = {name: set(task.id for task in tasks) for name, tasks in splits.items()}
    assert not identities["train"] & identities["valid"]
    assert not identities["train"] & identities["test"]
    assert not identities["valid"] & identities["test"]
    assert all(task.template != 2 and (task.steps < 4 or task.family == "recall") for task in splits["train"])
    assert any(task.template == 2 for task in splits["test"])
    assert any(task.steps == 4 and task.family != "recall" for task in splits["test"])
    for tasks in splits.values():
        for task in tasks:
            assert len(task.prompt + task.response) <= 128
            assert score_response(task, task.response)["both_correct"]
            assert not score_response(task, "I think the answer is correct")["answer_correct"]
    assert task_manifest(splits["train"])["sha256"] == task_manifest(generate_tasks("train", 150))["sha256"]


def test_answer_only_mask_does_not_train_on_prompt_or_padding():
    tasks = generate_tasks("train", 15)
    batch = task_batch(tasks, [0, 1, 2])
    for row, task in enumerate(tasks[:3]):
        first = len(task.prompt) - 1
        assert (batch.targets[row, :first] == -100).all()
        actual = bytes(batch.targets[row][batch.valid[row]].tolist()).decode("ascii")
        assert actual == task.response
        assert batch.tokens[row, :len(task.prompt)].tolist() == list(task.prompt.encode())


def test_checker_rejects_truncated_or_wrong_intermediate_steps():
    task = next(t for t in generate_tasks("test", 15) if t.family == "arithmetic")
    trace, answer = solve(task.payload)
    score = score_response(task, "steps=999;ans=" + answer + "\n")
    assert score["answer_correct"] and not score["steps_correct"]
    assert not score_response(task, task.response[:-4])["both_correct"]


def test_fourier_filter_damps_high_frequency_without_seeing_future_gradients():
    parameter = torch.nn.Parameter(torch.zeros(4))
    filter = GradientFilter([parameter], method="fourier", history=8, max_norm=100)
    for t in range(8):
        parameter.grad = torch.full_like(parameter, 2 + (-1) ** t)
        metrics = filter.apply()
    torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 2.0))
    assert metrics["high_frequency_fraction"] > 0
    # Replaying the same history must give exactly the same causal result.
    other = torch.nn.Parameter(torch.zeros(4))
    replay = GradientFilter([other], method="fourier", history=8, max_norm=100)
    for t in range(8):
        other.grad = torch.full_like(other, 2 + (-1) ** t)
        replay.apply()
    torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0)
