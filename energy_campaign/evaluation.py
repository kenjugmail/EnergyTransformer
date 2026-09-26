import json
import math
from pathlib import Path
import torch
from energy_real.model import LM,Config
from energy_real.runtime import atomic_json,sha256
from energy_real.train import evaluate
from .data import Documents
from .spec import source_identity


def evaluate_checkpoint(checkpoint,data,out,device='cpu',max_targets=1048576):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True);model=LM(Config(**saved['config'])).to(device)
    model.load_state_dict(saved.get('weights',saved.get('model')),strict=True)
    corpus=Documents(data,'test');results={}
    for length in (512,2048,8192,32768):
        results[str(length)]=evaluate(model,corpus,length,max_targets,torch.device(device),batch=1,amp=device=='cuda')
    result={'checkpoint_sha256':sha256(checkpoint),'data_sha256':corpus.file_identity(),'source_sha256':source_identity(),
            'trained_context':512,'results':results,'scope':'loss; context extrapolation is not a recall test'}
    atomic_json(result,out);return result


def summarize(root,out):
    root=Path(root);runs=[]
    for p in sorted(root.rglob('result.json')):
        runs.append({'path':str(p.relative_to(root)),**json.loads(p.read_text())})
    report={'format':'energy_campaign_summary_v1','runs':runs,'completed':sum(r.get('status')=='completed' for r in runs),
            'failed':sum(r.get('status')=='failed' for r in runs),'scaling_exponents':'unresolved: independent crossed data-budget grid required',
            'promotion':False}
    atomic_json(report,out);return report
