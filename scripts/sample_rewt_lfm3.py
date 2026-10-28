"""Generate with the standalone converted model and the inherited chat tokenizer."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from transformers import AutoTokenizer
from rewt.lfm_conversion import load_converted


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, default=ROOT/'runs/lfm3_conversion_20261028/initial')
    p.add_argument('--prompt', required=True)
    p.add_argument('--tokens', type=int, default=128)
    p.add_argument('--temperature', type=float, default=0.)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--rounds', type=int, default=None)
    p.add_argument('--seed', type=int, default=281704)
    p.add_argument('--json', action='store_true')
    a = p.parse_args()
    if a.tokens < 1 or a.temperature < 0:
        p.error('invalid generation settings')
    torch.set_num_threads(4); torch.manual_seed(a.seed)
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    model, metadata = load_converted(a.checkpoint, device=a.device)
    model.eval()
    tokens = tokenizer.apply_chat_template([dict(role='user', content=a.prompt)], tokenize=True,
                add_generation_prompt=True, enable_thinking=False, return_dict=False, return_tensors='pt').to(a.device)
    output = []; start = time.perf_counter()
    with torch.inference_mode(), torch.autocast(a.device, dtype=torch.bfloat16):
        state = None
        # Bounded prefill chunks avoid a quadratic mask for a very long prompt.
        for offset in range(0, tokens.shape[1], 256):
            logits, state = model(tokens[:, offset:offset+256], state, return_state=True,
                                  last_only=True, rounds=a.rounds)
        for _ in range(a.tokens):
            next_id = logits[:, -1].argmax(-1, keepdim=True) if a.temperature == 0 else torch.multinomial(
                (logits[:, -1].float()/a.temperature).softmax(-1), 1)
            value = int(next_id.item()); output.append(value)
            if value == model.config.eos_token_id:
                break
            logits, state = model(next_id, state, return_state=True, last_only=True, rounds=a.rounds)
    result = dict(prompt=a.prompt, response=tokenizer.decode(output, skip_special_tokens=True),
                  token_ids=output, seconds=time.perf_counter()-start, state_bytes=state.allocated_bytes(),
                  checkpoint=str(a.checkpoint.resolve()), metadata=metadata)
    print(json.dumps(result, indent=2) if a.json else result['response'])


if __name__ == '__main__':
    main()
