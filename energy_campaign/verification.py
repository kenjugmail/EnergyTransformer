"""Final held-out scores and cache-aware teacher-forced serving measurements."""
from pathlib import Path
import json
import math
import statistics
import time
import torch
from energy_real.model import LM,Config
from energy_real.static_cache import with_static_cache,allocated_state_bytes
from energy_real.runtime import atomic_json,sha256,sync
from energy_real.train import evaluate
from .data import Documents
from .spec import source_identity,FAMILIES


def state_bytes(state):
    return sum(t.numel()*t.element_size() for layer in state for t in layer)


def run(job):
    root=Path(job['root']);device=job.get('device','cuda');corpus=Documents(job['data'],'test');rows=[]
    checkpoints=sorted((root/'matched_data').glob('*/best.pt'))+sorted((root/'matched_time').glob('*/best.pt'))
    if not checkpoints:raise ValueError('No matched-data checkpoints')
    # Freeze hashes before opening test targets for any selection.
    selection={str(p.relative_to(root)):sha256(p) for p in checkpoints};atomic_json(selection,root/'final_selection.json')
    for path in checkpoints:
        saved=torch.load(path,map_location='cpu',weights_only=True);model=LM(Config(**saved['config'])).to(device).eval();model.load_state_dict(saved['weights'])
        score=evaluate(model,corpus,512,65536,torch.device(device),amp=device=='cuda')
        fit=json.loads((path.parent/'result.json').read_text())
        rows.append({'checkpoint':str(path.relative_to(root)),'comparison':path.parent.parent.name,
                     'fit_status':fit['status'],'kind':'trained_context','context':512,**score})
        atomic_json({'source':source_identity(),'status':'partial','rows':rows},job['out'])
        del model
    # Representative seed is fixed, not selected on the test results.
    for family in FAMILIES:
        path=root/f'matched_data/{family}_2_404/best.pt'
        if not path.exists():continue
        saved=torch.load(path,map_location='cpu',weights_only=True);cfg=Config(**saved['config']);base=LM(cfg).to(device).eval();base.load_state_dict(saved['weights'])
        for length in (512,2048,8192,32768):
            score=evaluate(base,corpus,length,65536,torch.device(device),amp=device=='cuda')
            rows.append({'family':family,'kind':'context_loss','context':length,**score})
            for batch in (1,8):
                times=[];prefills=[];memory=[]
                try:
                    model=with_static_cache(base,length+32) if cfg.kind=='softmax' and not cfg.attention_window else base
                    generator=torch.Generator().manual_seed(110);tokens=torch.randint(256,(batch,length+32),generator=generator).to(device)
                    for repeat in range(4):
                        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=device=='cuda'):
                            sync(device);start=time.monotonic();state=None
                            for i in range(0,length,512):_,state=model(tokens[:,i:min(i+512,length)],state)
                            sync(device);prefill=time.monotonic()-start;start=time.monotonic()
                            for i in range(length,length+32):_,state=model(tokens[:,i:i+1],state)
                            sync(device);decode=(time.monotonic()-start)/32
                            # First trial is warmup; count all reserved cache storage.
                            if repeat:
                                times.append(decode);prefills.append(prefill)
                                memory.append(allocated_state_bytes(model,state))
                    rows.append({'family':family,'kind':'serving','context':length,'batch':batch,'decode_seconds_per_step':statistics.median(times),
                                 'prefill_seconds':statistics.median(prefills),'allocated_state_bytes':max(memory),'teacher_forced':True})
                except torch.OutOfMemoryError as e:
                    rows.append({'family':family,'kind':'serving','context':length,'batch':batch,'status':'oom','error':str(e)})
                atomic_json({'source':source_identity(),'status':'partial','rows':rows},job['out'])
        del base
    from .recall import run as recall_run
    recall_run(root,device)
    result={'source':source_identity(),'status':'completed','rows':rows,'promotion':False,'recall':'separate learning experiment; not inferred from these context losses'}
    atomic_json(result,job['out']);return result
