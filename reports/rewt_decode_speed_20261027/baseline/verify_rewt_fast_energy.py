"""Compare real greedy continuations with fixed trained weights, without tuning."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from rewt.bpe import StoryBPE
from rewt.fast_energy import optimize_energy
from rewt.streaming import load_streaming
from rewt.streaming_decode import StreamingSession


@torch.inference_mode()
def generate(model, prompt, steps):
    device = model.embedding.weight.device
    session = StreamingSession(model, capacity=len(prompt)+steps)
    for start in range(0, len(prompt), 128):
        scores = session.consume(torch.tensor([list(prompt[start:start+128])], device=device))
    generated = []
    for _ in range(steps):
        token = scores[:, -1].argmax(-1)
        generated.append(int(token))
        scores = session.consume(token[:, None])
    return generated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = []; sources = []
    tasks = [json.loads(line) for line in (ROOT/'reports/rewt_streaming_20261026/tasks_valid.jsonl').read_text().splitlines()]
    selected = {}; comparisons = 0
    for row in tasks:
        selected.setdefault((row['family'], row['steps'], row['difficulty']), row['prompt'])
    for seed in (26301, 26302, 26303):
        for kind in ('language', 'tasks'):
            path = ROOT/f'runs/rewt_streaming_20261026/{kind}_hybrid_energy_{seed}/best.pt'
            original, saved = load_streaming(path); original.to(args.device).eval()
            fast = optimize_energy(copy.deepcopy(original), fused=args.device == 'cuda')
            if kind == 'language':
                tokenizer = StoryBPE.load(saved['tokenizer'])
                prompts = ['Once upon a time, ', 'The little girl opened the door and ', 'A dog and a cat were ']
                encodings = [tokenizer.encode(prompt) for prompt in prompts]; steps = 96
            else:
                prompts = list(selected.values()); encodings = [prompt.encode() for prompt in prompts]; steps = 40
            for prompt, encoded in zip(prompts, encodings):
                expected = generate(original, encoded, steps); actual = generate(fast, encoded, steps)
                assert actual == expected, (args.device, seed, kind, prompt)
                comparisons += 1
                rows.append(dict(seed=seed, kind=kind, prompt=prompt, steps=steps, equal=True, token_ids=actual))
            sources.append(dict(checkpoint=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            print('matched', args.device, seed, kind, len(prompts), flush=True)
    result = dict(status='passed', device=args.device, comparisons=comparisons,
                  task_selection='First validation prompt per family, depth and wording category; answers not supplied to inference.',
                  sources=sources, rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
