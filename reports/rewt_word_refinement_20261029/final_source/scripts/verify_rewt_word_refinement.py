"""Verify trained word refinement against complete cached generation paths."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from rewt.data import file_sha256
from rewt.experiment import save_json
from rewt.fast_decode import FastDecodeSession
from rewt.word_lm import load_word_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--backend',choices=['eager','compile','graph'],default='graph')
    p.add_argument('--rounds',type=int,default=32)
    a = p.parse_args()
    if a.output.exists():raise ValueError('Use a new audit output file')
    torch.set_num_threads(1)
    model,tokenizer,_ = load_word_model(a.checkpoint); model.to(a.device).eval()
    prompts = ['Once upon a', 'The little girl went to the park. '*7,
               'One day the dog saw a little boy in the garden. '*8]
    rows=[]; started=time.perf_counter()
    with torch.inference_mode():
        for prompt in prompts:
            tokens=torch.tensor([tokenizer.encode(prompt,bos=True)],device=a.device)
            expected,state,energies=model(tokens,return_state=True,rounds=a.rounds,details=True)
            rise=float(energies.diff(dim=-1).max()) if a.rounds else 0.
            assert rise <= 1e-5, rise
            cached=FastDecodeSession(model,capacity=256,rounds=a.rounds,backend=a.backend)
            actual=cached.consume(tokens)
            torch.testing.assert_close(actual,expected,atol=2e-4,rtol=2e-4)
            cached.prepare(); differences=[]; continuation=[]
            for _ in range(24):
                token=expected[:,-1].argmax(-1,keepdim=True)
                assert torch.equal(token,actual[:,-1].argmax(-1,keepdim=True))
                continuation.append(int(token))
                expected,state,_=model(token,state,return_state=True,rounds=a.rounds)
                actual=cached.consume(token)
                differences.append(float((expected-actual).abs().max()))
                torch.testing.assert_close(actual,expected,atol=2e-4,rtol=2e-4)
            rows.append(dict(prefix_tokens=tokens.numel(),checked_generation_steps=24,
                             exact_greedy_token_match=True,max_logit_error=max(differences),
                             maximum_energy_increase=rise,generated_ids=continuation,
                             crosses_attention_window=tokens.numel()+24>model.config.window))
    result=dict(status='passed',checkpoint=str(a.checkpoint.resolve()),checkpoint_sha256=file_sha256(a.checkpoint),
                device=a.device,backend=a.backend,rounds=a.rounds,rows=rows,
                checked_generation_steps=sum(r['checked_generation_steps'] for r in rows),
                wall_seconds=time.perf_counter()-started,
                note='Correctness audit with fixed prompts; not a quality or isolated latency benchmark.')
    save_json(a.output,result);print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
