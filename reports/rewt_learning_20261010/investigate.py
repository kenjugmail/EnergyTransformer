"""Validation/training-only failure localization for the next learning repair."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F
from rewt.evaluate import generate_batch
from rewt.experiment import load_model, save_json
from rewt.model import JointEnergyLM
from rewt.tasks import Task, score_response

OUT = Path(__file__).resolve().parent
PARENT = ROOT / "reports/rewt_stabilized_20261009"


def tasks(split):
    all_tasks = [Task(**json.loads(line)) for line in (PARENT / f"tasks_{split}.jsonl").read_text().splitlines()]
    return [task for family in ("arithmetic", "rules", "recall") for task in
            [t for t in all_tasks if t.family == family][:6]]


def semantic_mask(response):
    mask = [False] * len(response)
    split = response.index(";ans=")
    for i in list(range(len("steps="), split)) + list(range(split + len(";ans="), len(response))):
        mask[i] = response[i].isalnum() or response[i] == "-"
    return torch.tensor(mask)


@torch.inference_mode()
def probe(arm):
    record = json.loads((PARENT / "seed_101" / f"{arm}.json").read_text())
    checkpoint = Path(record["reasoning_training"]["checkpoint"])
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model, _ = load_model(checkpoint)
    model.eval()
    rows, parity = [], []
    for split in ("train", "valid"):
        for index, task in enumerate(tasks(split)):
            raw = (task.prompt + task.response).encode("ascii")
            x = torch.tensor([list(raw[:-1])], dtype=torch.long)
            p = len(task.prompt)
            noise = None
            if isinstance(model, JointEnergyLM) and model.config.support:
                noise = torch.randn(1, len(raw)-1, model.config.diffusion_samples,
                                    model.config.width, generator=torch.Generator().manual_seed(970 + index))
            kw = {"noise": noise} if noise is not None else {}
            full = model(x, **kw)
            first = model(x[:, :p], **({"noise": noise[:, :p]} if noise is not None else {}))
            chunks = [first.logits]
            state = first.state
            for t in range(p, x.shape[1]):
                piece = model(x[:, t:t+1], state=state,
                              **({"noise": noise[:, t:t+1]} if noise is not None else {}))
                chunks.append(piece.logits)
                state = piece.state
            parity.append(float((full.logits - torch.cat(chunks, 1)).abs().max()))
            pred = full.logits[0, p-1:].argmax(-1)
            target = torch.tensor(list(task.response.encode("ascii")))
            mask = semantic_mask(task.response)
            greedy = generate_batch(model, [task.prompt], max_new=min(40,128-p),
                                    stop_newline=True, seed=9080 + index)[0].decode("ascii", errors="replace")
            first_error = next((i for i,(a,b) in enumerate(zip(greedy,task.response)) if a != b),
                               min(len(greedy),len(task.response)) if greedy != task.response else None)
            row = dict(split=split, family=task.family, steps=task.steps, id=task.id,
                       prompt=task.prompt, expected=task.response, generated=greedy,
                       semantic_byte_accuracy=float((pred[mask] == target[mask]).float().mean()),
                       formatting_byte_accuracy=float((pred[~mask] == target[~mask]).float().mean()),
                       teacher_forced_response_bpb=float(F.cross_entropy(full.logits[0,p-1:],target))/0.6931471805599453,
                       first_generated_error=first_error,
                       first_generated_error_is_semantic=bool(mask[first_error]) if first_error is not None and first_error<len(mask) else None,
                       **{key:value for key,value in score_response(task,greedy).items() if isinstance(value,bool)})
            rows.append(row)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before
    summary = {}
    for split in ("train","valid"):
        selected=[r for r in rows if r['split']==split]
        summary[split] = {key:sum(float(r[key]) for r in selected)/len(selected)
                          for key in ('semantic_byte_accuracy','formatting_byte_accuracy','teacher_forced_response_bpb','valid_format','answer_correct','both_correct')}
    return dict(arm=arm, checkpoint=str(checkpoint), checkpoint_sha256=before,
                max_full_vs_incremental_logit_difference=max(parity), summary=summary, rows=rows)


def main():
    torch.set_num_threads(1)
    result = {arm:probe(arm) for arm in ("fourier_rewt","full_rewt","transformer")}
    save_json(OUT / "failure_localization.json", result)
    for arm,r in result.items():
        print(json.dumps({key:value for key,value in r.items() if key!='rows'}))
        print(json.dumps(r['rows'][:3]))


if __name__ == "__main__":
    main()
