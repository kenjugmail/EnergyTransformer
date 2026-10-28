import sys,json
from pathlib import Path
sys.path.insert(0,'/home/kt/Documents/EnergyTransformer')
import numpy as np
import torch
from rewt.evaluate import generate_batch,repetition_metrics
from rewt.memory import load_rewt_checkpoint
from rewt.story_data import StoryCorpus
from rewt.bpe import load_story_tokenizer
from rewt.experiment import save_json,save_lines
from rewt.data import file_sha256
root=Path('/home/kt/Documents/EnergyTransformer')
out=root/'reports/rewt_learning_20261010/language_bpe'
torch.set_num_threads(1)
corpus=StoryCorpus(root/'data/tinystories_bpe_20261010')
tokenizer=load_story_tokenizer(corpus.manifest['tokenizer']['path'])
eligible=np.flatnonzero(corpus.documents['valid'][:,1]-corpus.documents['valid'][:,0]>=128)
docs=np.random.default_rng(1771).choice(eligible,8,replace=False)
prompts=[dict(index=int(i),tokens_hex=bytes(corpus.data['valid'][int(corpus.documents['valid'][i,0]):int(corpus.documents['valid'][i,0])+32]).hex()) for i in docs]
plan=dict(validation_only=True,prompts=prompts,temperatures=[None,.3,.5,.7],seed=1773,max_new_tokens=96,
checkpoint=str(root/'runs/rewt_learning_20261010/language_bpe/fourier80/best.pt'))
plan['checkpoint_sha256']=file_sha256(Path(plan['checkpoint']))
save_json(out/'bpe_decoding_protocol.json',plan)
model,_=load_rewt_checkpoint(plan['checkpoint']);rows=[]
for temp in plan['temperatures']:
 generated=generate_batch(model,[bytes.fromhex(p['tokens_hex']) for p in prompts],max_new=96,sampling=temp is not None,temperature=temp or 1.,seed=plan['seed'])
 for prompt,tokens in zip(prompts,generated):
  raw=tokenizer.decode(tokens)
  rows.append(dict(document_index=prompt['index'],temperature=temp,prompt=tokenizer.decode(bytes.fromhex(prompt['tokens_hex'])).decode('ascii'),generation=raw.decode('ascii'),generation_hex=raw.hex(),metrics=repetition_metrics(raw)))
save_lines(out/'bpe_decoding_validation.jsonl',rows)
assert file_sha256(Path(plan['checkpoint']))==plan['checkpoint_sha256']
print(json.dumps(dict(samples=len(rows),output=str(out/'bpe_decoding_validation.jsonl'))))
