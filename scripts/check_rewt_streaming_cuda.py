"""Small float32 numerical and optional shared-device timing checks on CUDA."""
import argparse,json,statistics,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from torch.nn import functional as F
from rewt.data import file_sha256
from rewt.experiment import save_json
from rewt.streaming import load_streaming
from rewt.streaming_decode import StreamingSession


def utilization():
    r=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader'],capture_output=True,text=True)
    return r.stdout.strip() if r.returncode==0 else 'unavailable'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,default=ROOT/'reports/rewt_streaming_20261026');p.add_argument('--timing',action='store_true');a=p.parse_args()
    torch.set_num_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    names=['transformer','window_transformer','hybrid','hybrid_energy'];models={};rows=[]
    for name in names:
        r=json.loads((a.study/f'language_{name}_26301.json').read_text());model,_=load_streaming(r['checkpoint']);model.eval()
        torch.manual_seed(27001);x=torch.randint(256,(2,37))
        with torch.inference_mode():cpu,_,_=model(x)
        model=model.cuda();session=StreamingSession(model,batch=2,capacity=40)
        with torch.inference_mode():
            gpu,_,energy=model(x.cuda(),details=True)
            cached=torch.cat([session.consume(x[:,:13].cuda()),session.consume(x[:,13:36].cuda()),session.consume(x[:,36:].cuda())],1)
        error=float((gpu.cpu()-cpu).abs().max());cached_error=float((cached-gpu).abs().max())
        assert error<.002 and cached_error<.002
        rows.append(dict(variant=name,max_cpu_gpu_error=error,max_cached_error=cached_error,greedy_equal=bool((gpu.cpu().argmax(-1)==cpu.argmax(-1)).all()),energy_nonincreasing=bool((energy.diff(dim=-1)<=1e-5).all()) if energy is not None else None))
        models[name]=model
    result=dict(status='passed',device=torch.cuda.get_device_name(),dtype='float32',tf32=False,rows=rows,
        source_sha256={str(p.relative_to(ROOT)):file_sha256(p) for p in [ROOT/'rewt/streaming.py',ROOT/'rewt/streaming_decode.py',Path(__file__)]})
    if a.timing:
        samples=[];before=utilization();torch.manual_seed(27003);x=torch.randint(256,(24,128),device='cuda');y=torch.randint(256,(24,128),device='cuda')
        for m in models.values():
            m.train()
            for _ in range(3):
                m.zero_grad(set_to_none=True);logits,_,_=m(x);F.cross_entropy(logits.flatten(0,1),y.flatten()).backward()
        torch.cuda.synchronize()
        for repetition in range(15):
            for name in (names if repetition%2==0 else names[::-1]):
                m=models[name];m.zero_grad(set_to_none=True)
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize();wall=time.perf_counter();start.record()
                logits,_,_=m(x);F.cross_entropy(logits.flatten(0,1),y.flatten()).backward();end.record();end.synchronize()
                samples.append(dict(variant=name,repetition=repetition,gpu_ms=start.elapsed_time(end),wall_ms=1000*(time.perf_counter()-wall)))
        result.update(scope='Shared-device float32 eager forward+cross-entropy+backward at 24x128; no optimizer, compilation, mixed precision, or exclusive GPU. Alternating order, 15 measured repetitions. Events include gaps between host launches.',
            utilization_before=before,utilization_after=utilization(),samples=samples,
            medians={name:{key:statistics.median(r[key] for r in samples if r['variant']==name) for key in ('gpu_ms','wall_ms')} for name in names})
    save_json(a.study/('cuda_benchmark.json' if a.timing else 'cuda_verification.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k!='samples'},indent=2))


if __name__=='__main__':main()
