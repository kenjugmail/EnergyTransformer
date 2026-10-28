import sys,json,re,time
from pathlib import Path
sys.path.insert(0,'/home/kt/Documents/EnergyTransformer')
import torch
from rewt.position_memory import load_learning_checkpoint
from rewt.tasks import Task,solve
from rewt.evaluate import reasoning_evaluation
from rewt.data import file_sha256
from rewt.experiment import save_json,save_lines
root=Path('/home/kt/Documents/EnergyTransformer');out=root/'reports/rewt_learning_20261010'
torch.set_num_threads(1)
tasks=[Task(**json.loads(x)) for x in (out/'multistep_position/tasks_valid.jsonl').read_text().splitlines()]
tasks=[t for t in tasks if t.template<2 and t.family in ('arithmetic','rules') and t.steps==2]
for folder,case in [('multistep_position','progress'),('multistep_huber','progress'),('multistep_huber','workspace')]:
 path=root/'runs/rewt_learning_20261010'/folder/case/'last.pt'
 before=file_sha256(path);model,saved=load_learning_checkpoint(path);assert file_sha256(path)==before
 start=time.process_time();summary,rows=reasoning_evaluation(model,tasks,seed=1851)
 originals={t.id:t for t in tasks};groups={}
 for family in ('arithmetic','rules'):
  chosen=[r for r in rows if r['family']==family];first=correct_length=0
  for r in chosen:
   m=re.search(r'steps=([^;\n]+)',r['response']);trace=m[1].split(',') if m else []
   expected,_=solve(originals[r['id']].payload)
   first+=bool(trace and trace[0]==expected[0]);correct_length+=len(trace)==2
  groups[family]=dict(count=len(chosen),first_step_correct=first,correct_trace_length=correct_length,complete_correct=sum(r['both_correct'] for r in chosen))
 result=dict(validation_only=True,checkpoint_sha256=before,step=saved['step'],groups=groups,cpu_seconds=time.process_time()-start)
 save_json(out/folder/f'{case}_twohop_diagnostic.json',result);save_lines(out/folder/f'{case}_twohop_diagnostic_outputs.jsonl',rows)
 print(json.dumps(dict(folder=folder,case=case,**result)),flush=True)
 print(json.dumps(rows[:3]),flush=True)
