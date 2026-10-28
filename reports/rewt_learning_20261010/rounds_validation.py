import json,sys,time
from pathlib import Path
sys.path.insert(0,'/home/kt/Documents/EnergyTransformer')
import torch
from rewt.memory import load_rewt_checkpoint
from rewt.tasks import Task
from rewt.evaluate import reasoning_evaluation
from rewt.data import file_sha256
from rewt.experiment import save_json
root=Path('/home/kt/Documents/EnergyTransformer')
out=root/'reports/rewt_learning_20261010'
torch.set_num_threads(1)
record=json.loads((out/'precision/one_head.json').read_text())
model,_=load_rewt_checkpoint(record['checkpoint'])
tasks=[Task(**json.loads(s)) for s in (out/'foundation/tasks_valid.jsonl').read_text().splitlines()]
tasks=[t for t in tasks if t.template<2]
results=[]
for rounds in [0,1,2,4,8]:
 start=time.process_time();summary,rows=reasoning_evaluation(model,tasks,seed=1661,rounds=rounds)
 row=dict(rounds=rounds,cpu_seconds=time.process_time()-start,summary=summary)
 results.append(row);print(json.dumps(row),flush=True)
save_json(out/'precision/refinement_validation.json',dict(checkpoint=record['checkpoint'],checkpoint_sha256=file_sha256(Path(record['checkpoint'])),validation_only=True,results=results))
