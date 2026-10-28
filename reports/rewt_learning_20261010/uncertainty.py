"""Conditional test-set uncertainty for frozen, already selected checkpoints."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from rewt.bpe import load_story_tokenizer
from rewt.data import file_sha256
from rewt.experiment import save_json, save_lines
from rewt.position_memory import load_learning_checkpoint
from rewt.story_data import StoryCorpus

OUT = Path(__file__).resolve().parent


def interval(values):
    return [float(x) for x in np.quantile(values, [.025, .975])]


def arithmetic():
    policies = {}
    for rounds in (0, 2):
        rows = [json.loads(line) for line in (OUT / 'onehop_test' / f'rounds_{rounds}_outputs.jsonl').read_text().splitlines()]
        groups = {}
        for row in rows:
            if row['family'] == 'arithmetic' and row['template'] < 2:
                groups.setdefault(row['id'], []).append(float(row['both_correct']))
        assert len(groups) == 66 and all(len(values) == 2 for values in groups.values())
        policies[rounds] = np.array([np.mean(groups[key]) for key in sorted(groups)])
    rng = np.random.default_rng(2029)
    indices = rng.integers(0, 66, (10000, 66))
    return dict(semantic_pairs=66, prompts_per_pair=2,
        two_round_complete_accuracy=float(policies[2].mean()),
        two_round_95_percent_interval=interval(policies[2][indices].mean(1)),
        zero_round_complete_accuracy=float(policies[0].mean()),
        zero_round_95_percent_interval=interval(policies[0][indices].mean(1)),
        same_weight_refinement_difference=float((policies[2] - policies[0]).mean()),
        difference_95_percent_interval=interval((policies[2] - policies[0])[indices].mean(1)))


@torch.inference_mode()
def language():
    plan = json.loads((OUT / 'language_test/plan.json').read_text())
    training = json.loads((OUT / 'language_bpe_extended/plan.json').read_text())
    corpus = StoryCorpus(training['data_directory'])
    tokenizer = load_story_tokenizer(plan['tokenizer']['path'])
    lengths = torch.tensor(tokenizer.byte_lengths, dtype=torch.long)
    all_rows, losses, sizes = [], {}, None
    for record in plan['checkpoints']:
        model, saved = load_learning_checkpoint(record['checkpoint'])
        model.eval()
        checkpoint_sha = file_sha256(Path(record['checkpoint']))
        assert checkpoint_sha == record['checkpoint_sha256']
        rows = []
        for offset in plan['test_offsets']:
            batch = corpus.batch('test', None, 1, plan['context_tokens'], starts=[offset])
            output = model(batch.tokens)
            bits = float(F.cross_entropy(output.logits.flatten(0, 1), batch.targets.flatten(), reduction='sum')) / math.log(2)
            raw_bytes = int(lengths[batch.targets].sum())
            rows.append(dict(offset=offset, nll_bits=bits, raw_bytes=raw_bytes,
                             bpb=bits / raw_bytes, case=record['case']))
        losses[record['case']] = np.array([r['nll_bits'] for r in rows])
        current_sizes = np.array([r['raw_bytes'] for r in rows])
        if sizes is None:
            sizes = current_sizes
        else:
            assert np.array_equal(sizes, current_sizes)
        all_rows.extend(rows)
        assert file_sha256(Path(record['checkpoint'])) == checkpoint_sha
    save_lines(OUT / 'language_test/per_document_loss.jsonl', all_rows)
    rng = np.random.default_rng(2031)
    indices = rng.integers(0, len(sizes), (10000, len(sizes)))
    denominator = sizes[indices].sum(1)
    estimates = {case: values[indices].sum(1) / denominator for case, values in losses.items()}
    primary = plan['primary']
    return dict(documents=len(sizes), evaluated_raw_bytes=int(sizes.sum()),
        models={case: dict(bpb=float(values.sum() / sizes.sum()),
                          interval_95_percent=interval(estimates[case])) for case, values in losses.items()},
        primary_minus_control={case: dict(difference_bpb=float((losses[primary] - losses[case]).sum() / sizes.sum()),
            interval_95_percent=interval(estimates[primary] - estimates[case]))
            for case in losses if case != primary})


def main():
    torch.set_num_threads(1)
    result = dict(created=datetime.now(timezone.utc).isoformat(), bootstrap_replicates=10000,
        method='Paired percentile bootstrap over 66 semantic arithmetic pairs and 64 held-out story documents. Language bpb uses each resampled sum of loss bits divided by its resampled sum of raw bytes.',
        limitations='Conditional test-sample uncertainty only; this does not represent variability across training seeds, validation exploration, datasets, or model families. Zero rounds is an intervention on the same trained weights. Per-document reevaluation may have tiny floating-point differences from the batched evaluation.',
        arithmetic=arithmetic(), language=language(),
        script_sha256=file_sha256(Path(__file__)))
    save_json(OUT / 'test_uncertainty.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
