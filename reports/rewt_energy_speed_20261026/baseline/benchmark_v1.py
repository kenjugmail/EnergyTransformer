"""Same-checkpoint energy optimization parity and alternating warmed timings."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F
from rewt.fast_energy import optimize_energy
from rewt.streaming import load_streaming
from rewt.streaming_decode import StreamingSession


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utilization():
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader'], capture_output=True, text=True)
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'runs/rewt_streaming_20261026/language_hybrid_energy_26301/best.pt')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--repetitions', type=int, default=11)
    args = parser.parse_args()
    output = args.output or ROOT/f'reports/rewt_energy_speed_20261026/{args.device}.json'
    if output.exists():
        parser.error('output exists; choose a new path to preserve measurements')
    torch.set_num_threads(1)
    affinity = os.sched_getaffinity(0)
    core = 5 if 5 in affinity else min(affinity)
    os.sched_setaffinity(0, {core})
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, _ = load_streaming(args.checkpoint)
    model.to(args.device).eval()
    names = ['original', 'fast_eager', 'fast_compiled', 'energy_off']
    models = {name: copy.deepcopy(model) for name in names}
    optimize_energy(models['fast_eager'])
    optimize_energy(models['fast_compiled'], compiled=True)
    rounds = {name: 0 if name == 'energy_off' else 2 for name in names}
    torch.manual_seed(28901)
    x = torch.randint(256, (24, 128), device=args.device)
    y = torch.randint(256, x.shape, device=args.device)
    anchor = torch.randn(24, 128, 64, device=args.device)
    token_anchor = anchor[:1, :1].contiguous()
    tokens = torch.randint(256, (1, 8192 + 32), device=args.device)
    result = dict(device=args.device, gpu=torch.cuda.get_device_name() if args.device=='cuda' else None,
                  torch=str(torch.__version__), checkpoint=str(args.checkpoint), checkpoint_sha256=sha(args.checkpoint),
                  cpu_core=core, threads=1, tf32=False, dtype='float32', rounds=2, repetitions=args.repetitions,
                  gpu_utilization_before=utilization(), warmup_seconds={}, parity=[], samples=[],
                  scope='Same fixed weights, two exact energy updates, eager backbone. Warm timings exclude compilation and preparation; process CPU and wall latency; CUDA events also recorded. Shared machine/GPU; no retraining or quality comparison. Energy-off is only a timing lower bound.')

    def sync():
        if args.device == 'cuda':
            torch.cuda.synchronize()

    def timed(function):
        sync()
        start_cpu = time.process_time(); start_wall = time.perf_counter()
        if args.device == 'cuda':
            start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
            start.record()
        value = function()
        if args.device == 'cuda':
            end.record(); end.synchronize()
        row = dict(cpu_ms=1000*(time.process_time()-start_cpu), wall_ms=1000*(time.perf_counter()-start_wall))
        if args.device == 'cuda':
            row['gpu_ms'] = start.elapsed_time(end)
        return value, row

    def train(m, r):
        m.zero_grad(set_to_none=True)
        logits = m(x, rounds=r)[0]
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        loss.backward()
        return logits.detach()

    # Gradient parity through the full model, including the energy coefficients.
    reference = None; gradients = None
    for name in names:
        m = models[name]; m.train()
        started = time.perf_counter(); actual = train(m, rounds[name]); sync()
        result['warmup_seconds'][f'train_{name}'] = time.perf_counter()-started
        if name == 'original':
            reference = actual; gradients = {key: p.grad.clone() for key, p in m.named_parameters()}
        elif name != 'energy_off':
            error = float((actual-reference).abs().max())
            torch.testing.assert_close(actual, reference, atol=3e-5, rtol=3e-4)
            maximum = 0.
            for key, parameter in m.named_parameters():
                torch.testing.assert_close(parameter.grad, gradients[key], atol=3e-5, rtol=3e-4)
                maximum = max(maximum, float((parameter.grad-gradients[key]).abs().max()))
            result['parity'].append(dict(variant=name, logit_max_error=error, gradient_max_error=maximum,
                                        greedy_equal=bool((actual.argmax(-1)==reference.argmax(-1)).all())))
        print('warmed training', name, flush=True)

    prepared = {}
    for name in names:
        m = models[name]; m.eval()
        if name.startswith('fast_'):
            prepared[name] = m.energy_head.prepare_inference(2)
        elif name == 'original':
            # Match the original session's precomputed fixed constants exactly.
            session = StreamingSession(m, capacity=1)
            w, q, norm2, maximum = session.energy_constants
            h = m.energy_head
            def original_prepared(a, h=h, w=w, q=q, norm2=norm2, maximum=maximum):
                amplitude=h.amplitude(a).tanh()/h.config.modes**.5
                eta=.95/(maximum+amplitude.abs()@norm2)[...,None]
                delta=torch.zeros_like(a)
                for _ in range(2):
                    delta=delta-eta*(q*delta-(amplitude*F.linear(delta,w,h.phase).sin())@w)
                return a+delta
            prepared[name] = original_prepared
        else:
            prepared[name] = lambda a:a
        with torch.inference_mode():
            started=time.perf_counter()
            for a in [token_anchor, anchor[:1], anchor]:
                actual=prepared[name](a)
                if name != 'energy_off':
                    expected=models['original'].energy_head(a,2)[0]
                    torch.testing.assert_close(actual,expected,atol=3e-6,rtol=3e-5)
            sync();result['warmup_seconds'][f'inference_{name}']=time.perf_counter()-started
        print('warmed inference', name, flush=True)

    # Capture operator counts separately from timing, without profiler overhead.
    with torch.inference_mode():
        profiles={}
        for name in names[:-1]:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                prepared[name](token_anchor)
            profiles[name]=[dict(operator=row.key,count=row.count,self_cpu_us=row.self_cpu_time_total) for row in prof.key_averages()]
        result['operator_profiles']=profiles

    for repetition in range(args.repetitions):
        order=names if repetition%2==0 else names[::-1]
        for name in order:
            m=models[name];m.train()
            _, row=timed(lambda:train(m,rounds[name]))
            result['samples'].append(dict(kind='full_training',variant=name,repetition=repetition,**row))
            m.eval()
            with torch.inference_mode():
                # Batched loop amortizes event/timer overhead for tiny energy heads.
                def run_head():
                    for _ in range(64):
                        prepared[name](token_anchor)
                _, row=timed(run_head)
                result['samples'].append(dict(kind='token_energy',variant=name,repetition=repetition,**{k:v/64 for k,v in row.items()}))
                for prefix in (128,8192):
                    session=StreamingSession(m,capacity=prefix+32,rounds=rounds[name])
                    for i in range(0,prefix,128):
                        session.consume(tokens[:,i:i+128])
                    def decode():
                        for i in range(prefix,prefix+32):
                            session.consume(tokens[:,i:i+1])
                    _,row=timed(decode)
                    result['samples'].append(dict(kind=f'decode32_prefix{prefix}',variant=name,repetition=repetition,**row))
        print('repetition', repetition+1, flush=True)

    # Same prompts, greedy continuation; this is equivalence checking, not tuning.
    with torch.inference_mode():
        trajectories={}
        for name in names[:-1]:
            m=models[name].eval();session=StreamingSession(m,capacity=160)
            logits=session.consume(tokens[:,:128]);sequence=[]
            for _ in range(32):
                token=logits[:,-1].argmax(-1);sequence.append(int(token))
                logits=session.consume(token[:,None])
            trajectories[name]=sequence
        assert all(v==trajectories['original'] for v in trajectories.values())
        result['greedy_trajectories']=trajectories
    result['medians']={kind:{name:{key:statistics.median(r[key] for r in result['samples'] if r['kind']==kind and r['variant']==name)
            for key in (['cpu_ms','wall_ms','gpu_ms'] if args.device=='cuda' else ['cpu_ms','wall_ms'])} for name in names}
            for kind in sorted({r['kind'] for r in result['samples']})}
    result['gpu_utilization_after']=utilization()
    result['source_sha256']={str(path.relative_to(ROOT)):sha(path) for path in [ROOT/'rewt/streaming.py',ROOT/'rewt/streaming_decode.py',ROOT/'rewt/fast_energy.py',Path(__file__)]}
    output.parent.mkdir(exist_ok=True,parents=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['parity','warmup_seconds','medians']},indent=2))


if __name__=='__main__':
    main()
