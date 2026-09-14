"""One ranking and confidence contract for study and practical retrieval."""
import math
import torch


def rank_candidates(scores, records):
    finite = bool(torch.isfinite(scores).all())
    # Event index is the stable secondary key, including exactly tied scores.
    indices = sorted(records, key=lambda i: (-float(scores[i]), i))[:2] if finite else []
    score = float(scores[indices[0]]) if indices else None
    runner_up = float(scores[indices[1]]) if len(indices) > 1 else 0.
    return {'score':score, 'margin':score-runner_up if score is not None else None,
            'candidates':[records[i] for i in indices], 'finite':finite}


def accepts(row, score_threshold, margin_threshold):
    if not all(math.isfinite(x) for x in (score_threshold, margin_threshold)) or margin_threshold < 0:
        raise ValueError('Invalid confidence thresholds')
    return row['score'] is not None and row['score'] > score_threshold and row['margin'] >= margin_threshold
