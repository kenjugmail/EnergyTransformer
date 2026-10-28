"""Generate text with a frozen streaming checkpoint and owned inference cache."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from rewt.bpe import StoryBPE
from rewt.streaming import load_streaming
from rewt.streaming_decode import StreamingSession


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--prompt',default='Once upon a time, ');p.add_argument('--tokens',type=int,default=128)
    p.add_argument('--temperature',type=float,default=0.);p.add_argument('--top-k',type=int,default=40)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--rounds',type=int);p.add_argument('--output',type=Path)
    p.add_argument('--fast-energy',action='store_true',help='exact algebraic energy optimization')
    p.add_argument('--compile-energy',action='store_true',help='also compile energy kernels; pays a first-call cost')
    p.add_argument('--fused-energy',action='store_true',help='fuse two-round float32 CUDA inference')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--decoder',choices=['reference','eager','compile','graph'],default='reference',help='fixed-shape decoder; graph requires CUDA and has a first-use capture cost')
    a=p.parse_args()
    if not a.prompt or a.tokens<1 or a.temperature<0 or not 1<=a.top_k<=256:p.error('invalid sampling options')
    if a.output and a.output.exists():p.error('output already exists')
    torch.set_num_threads(1);model,saved=load_streaming(a.checkpoint);model.eval()
    model.to(a.device)
    if a.fused_energy and (a.device!='cuda' or (a.rounds if a.rounds is not None else model.config.rounds)!=2):p.error('fused energy requires CUDA and two rounds')
    if a.fast_energy or a.compile_energy or a.fused_energy:
        from rewt.fast_energy import optimize_energy
        optimize_energy(model,compiled=a.compile_energy,fused=a.fused_energy)
    tokenizer=StoryBPE.load(saved['tokenizer']) if saved.get('tokenizer') else None
    prompt=tokenizer.encode(a.prompt) if tokenizer else a.prompt.encode('utf-8')
    if a.decoder=='reference':
        session=StreamingSession(model,capacity=len(prompt)+a.tokens,rounds=a.rounds)
    else:
        from rewt.fast_decode import FastDecodeSession
        session=FastDecodeSession(model,capacity=len(prompt)+a.tokens,rounds=a.rounds,backend=a.decoder)
    generator=torch.Generator(device=a.device).manual_seed(a.seed);generated=[]
    with torch.inference_mode():
        for start in range(0,len(prompt),128):logits=session.consume(torch.tensor([list(prompt[start:start+128])],device=a.device))
        if a.decoder!='reference':
            generated=session.generate(a.tokens,temperature=a.temperature,top_k=a.top_k,generator=generator)[0].cpu().tolist()
        else:
            for _ in range(a.tokens):
                scores=logits[:,-1]
                if a.temperature:
                    values,ids=(scores/a.temperature).topk(a.top_k,-1)
                    token=ids.gather(-1,torch.multinomial(values.softmax(-1),1,generator=generator)).squeeze(-1)
                else:token=scores.argmax(-1)
                generated.append(int(token));logits=session.consume(token[:,None])
    raw=tokenizer.decode(generated) if tokenizer else bytes(generated)
    if a.output:a.output.write_bytes(raw)
    sys.stdout.buffer.write(a.prompt.encode()+raw+b'\n')


if __name__=='__main__':main()
