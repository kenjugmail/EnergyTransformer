"""Frozen before/after evaluation: ARC label ranking and exact-answer probes.

This is a small, custom evaluation, not the official ARC/lm-eval protocol.
No probe is used in the recovery corpus or for selecting a checkpoint.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from rewt.lfm_conversion import load_converted
from scripts.prepare_rewt_lfm3 import TEACHER_REVISION,sha,save_json


def make_suite():
    rows=[]
    def exact(category,prompt,answer):
        rows.append(dict(id=f'{category}_{sum(x["category"]==category for x in rows)}',category=category,
                         messages=[dict(role='user',content=prompt)],expected=answer,scorer='exact'))
    for prompt,answer in [
        ('Reply with exactly the word lavender.','lavender'),
        ('Write the word telescope in uppercase. Output only that word.','TELESCOPE'),
        ('Reverse the characters in drawer. Output only the reversed word.','reward'),
        ('Sort these words alphabetically: plum, apple, pear. Return only the words separated by commas and spaces.','apple, pear, plum'),
        ('Return only the last word in this sentence: The old train crossed the bridge','bridge'),
        ('Is 29 even or odd? Answer with exactly one lowercase word.','odd'),
        ('Write exactly three lines: red on the first, green on the second, blue on the third.','red\ngreen\nblue'),
        ('Replace each space in "silver fox runs" with an underscore. Return only the result.','silver_fox_runs')]:exact('format',prompt,answer)
    for expression,answer in [('38 + 27','65'),('91 - 36','55'),('12 * 7','84'),('144 / 12','12'),
                              ('3 * (8 + 5)','39'),('2 ** 6','64'),('250 * 0.12','30'),('7 * 8 - 9','47')]:
        exact('arithmetic',f'Calculate {expression}. Answer with just the number.',answer)
    for question,answer in [
        ('What is the capital of Japan?','Tokyo'),('What is the largest planet in our solar system?','Jupiter'),
        ('What gas do plants absorb from the atmosphere during photosynthesis?','carbon dioxide'),
        ('What is the chemical symbol for gold?','Au'),('How many sides does a hexagon have?','6'),
        ('Which organ pumps blood through the human body?','heart'),
        ('What is the freezing point of water in degrees Celsius at standard atmospheric pressure?','0'),
        ('What process changes a liquid into a gas at its surface?','evaporation')]:
        exact('knowledge',question+' Answer with only the name, symbol, or number; no explanation.',answer)
    for code,answer in [
        ('print(sum([2, 4, 6]))','12'),('print("stressed"[::-1])','desserts'),
        ('print(len(set([1, 1, 2, 3, 3])))','3'),('print(17 // 5, 17 % 5)','3 2'),
        ('print([x*x for x in range(4)])','[0, 1, 4, 9]'),
        ('d = {"a": 2, "b": 5}\nprint(d.get("c", 7) + d["a"])','9')]:
        exact('code_output','What does this Python code print? Return only the printed output, without code fences.\n'+code,answer)
    for text,expected in [
        ('Name: Imani. City: Nairobi.',{'name':'Imani','city':'Nairobi'}),
        ('Name: Hugo. City: Porto.',{'name':'Hugo','city':'Porto'}),
        ('Name: Yuki. City: Kyoto.',{'name':'Yuki','city':'Kyoto'}),
        ('Name: Amara. City: Accra.',{'name':'Amara','city':'Accra'})]:
        rows.append(dict(id=f'json_{len(rows)}',category='json',messages=[dict(role='user',content=
            f'Extract the name and city as JSON with exactly the keys name and city. Return only JSON. Text: {text}')],
            expected=expected,scorer='json'))
    for city,unit in [('Oslo','celsius'),('Lima','celsius'),('Miami','fahrenheit'),('Rome','celsius')]:
        rows.append(dict(id=f'tool_json_{len(rows)}',category='tool_json',messages=[dict(role='system',content=
            'You translate weather requests into JSON. Return exactly one object with keys tool and arguments. '
            'The only tool is get_weather. Its arguments are city (string) and unit (celsius or fahrenheit). '
            'Do not call a real service and do not add commentary.'),dict(role='user',content=
            f'Get the weather in {city}, in {unit}.')],
            expected={'tool':'get_weather','arguments':{'city':city,'unit':unit}},scorer='json'))
    # Random secrets and distractors are generated independently of training.
    rng=np.random.default_rng(281711)
    for lines in (4,48,112):
        for j in range(4):
            secret=str(int(rng.integers(10000,99999)))
            irrelevant='\n'.join(f'Record {i}: item {int(rng.integers(100,999))} is archived.' for i in range(lines))
            prompt=f'Remember the access code: {secret}.\n{irrelevant}\nWhat was the access code? Answer with only the five digits.'
            exact(f'recall_{lines}_lines',prompt,secret)
    arc=[]
    for subset in ('ARC-Easy','ARC-Challenge'):
        all_rows=pq.read_table(ROOT/'data/lfm3_capability_eval'/subset/'test-00000-of-00001.parquet').to_pylist()
        for row in sorted(all_rows,key=lambda r:hashlib.sha256(('281711/'+r['id']).encode()).hexdigest())[:64]:
            labels=[chr(65+i) for i in range(len(row['choices']['text']))]
            correct=row['choices']['label'].index(row['answerKey'])
            prompt=row['question']+'\n'+'\n'.join(f'{label}. {answer}' for label,answer in zip(labels,row['choices']['text']))
            prompt+='\nAnswer with only the letter of the best choice.'
            arc.append(dict(id=row['id'],subset=subset,messages=[dict(role='user',content=prompt)],
                            labels=labels,expected=labels[correct]))
    return dict(version=1,seed=281711,probes=rows,arc=arc,
                protocol='Greedy generation, original non-thinking chat template, 64 new tokens. Exact probes strip surrounding whitespace; knowledge matching is case-insensitive. JSON uses parsed object equality. ARC uses first-position logits restricted to single-token A/B/C/D(/E) labels, a custom zero-shot protocol.',
                limits='Small exploratory suite; code-output prediction does not establish coding ability; JSON routing does not establish real tool use. ARC sample is not the full benchmark. Training data may overlap semantic facts or donor pretraining.')


def score_probe(row,response):
    response=response.strip()
    if row['scorer']=='json':
        try:return json.loads(response)==row['expected']
        except (ValueError,TypeError):return False
    if row['category']=='knowledge':return response.casefold()==row['expected'].casefold()
    return response==row['expected']


def interval(correct,total):
    # Wilson 95% interval, reflecting sample size, not task-distribution coverage.
    z=1.96;p=correct/total;d=1+z*z/total
    mid=(p+z*z/(2*total))/d
    radius=z*((p*(1-p)/total+z*z/(4*total*total))**.5)/d
    return [mid-radius,mid+radius]


def summarize(rows,category_key):
    groups=defaultdict(list)
    for row in rows:groups[row[category_key]].append(row['correct'])
    return {name:dict(correct=sum(values),total=len(values),accuracy=sum(values)/len(values),
                     wilson95=interval(sum(values),len(values))) for name,values in groups.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=['teacher','student'],default='student')
    p.add_argument('--checkpoint',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--suite',type=Path,default=ROOT/'reports/lfm3_capability_recovery_20261028/evaluation_suite.json')
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--device',default='cuda')
    a=p.parse_args()
    if not a.suite.exists():save_json(a.suite,make_suite())
    if a.prepare_only:
        print(a.suite);return
    if a.output is None or a.output.exists():p.error('a new output file is required')
    if a.kind=='student' and a.checkpoint is None:p.error('--checkpoint is required for student')
    torch.set_num_threads(4);torch.manual_seed(281711)
    teacher_path=ROOT/'data/lfm3_1_7b_teacher'/TEACHER_REVISION
    tokenizer=AutoTokenizer.from_pretrained(teacher_path,local_files_only=True,trust_remote_code=True)
    if a.kind=='teacher':
        model=AutoModelForCausalLM.from_pretrained(teacher_path,trust_remote_code=True,local_files_only=True,
            dtype=torch.bfloat16,device_map=a.device,attn_implementation='sdpa').eval().requires_grad_(False)
        identity=dict(teacher_revision=TEACHER_REVISION,sha256=sha(teacher_path/'model.safetensors'))
    else:
        model,metadata=load_converted(a.checkpoint,device=a.device);model.eval().requires_grad_(False)
        identity=dict(checkpoint=str(a.checkpoint.resolve()),sha256=sha(a.checkpoint),metadata=metadata)
    suite=json.loads(a.suite.read_text());report=dict(model=identity,suite_sha256=sha(a.suite),protocol=suite['protocol'],arc=[],probes=[])
    start=time.perf_counter()
    def encode(messages):
        return tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False,
            return_dict=False,return_tensors='pt').to(a.device)
    with torch.no_grad(),torch.autocast(a.device,dtype=torch.bfloat16):
        for row in suite['arc']:
            tokens=encode(row['messages'])
            if a.kind=='teacher':
                hidden=model.model(tokens,use_cache=False).last_hidden_state[:,-1]
                logits=torch.nn.functional.linear(hidden,model.model.embed_tokens.weight)[0]
            else:logits=model(tokens,last_only=True)[0][0,-1]
            ids=[tokenizer.encode(label,add_special_tokens=False) for label in row['labels']]
            if any(len(x)!=1 for x in ids):raise ValueError('label must be single token')
            scores=[float(logits[x[0]]) for x in ids]
            prediction=row['labels'][int(np.argmax(scores))]
            report['arc'].append(dict(id=row['id'],subset=row['subset'],expected=row['expected'],prediction=prediction,
                label_logits=scores,correct=prediction==row['expected'],prompt_tokens=tokens.numel()))
            if len(report['arc'])%32==0:
                save_json(a.output,report);print(json.dumps(dict(stage='arc',completed=len(report['arc']))),flush=True)
        for row in suite['probes']:
            tokens=encode(row['messages']);generated=[]
            if a.kind=='teacher':
                # Override the donor's saved repetition penalty (1.1), since the
                # standalone student's argmax decoder applies no such penalty.
                generated=model.generate(tokens,max_new_tokens=64,do_sample=False,repetition_penalty=1.,
                    pad_token_id=model.config.pad_token_id)[0,tokens.shape[1]:].tolist()
            else:
                state=None
                for offset in range(0,tokens.shape[1],256):
                    logits,state=model(tokens[:,offset:offset+256],state,return_state=True,last_only=True)
                for _ in range(64):
                    value=logits[:,-1].argmax(-1,keepdim=True);generated.append(int(value.item()))
                    if generated[-1]==model.config.eos_token_id:break
                    logits,state=model(value,state,return_state=True,last_only=True)
            response=tokenizer.decode(generated,skip_special_tokens=True)
            report['probes'].append(dict(id=row['id'],category=row['category'],response=response,token_ids=generated,
                correct=score_probe(row,response),expected=row['expected'],prompt_tokens=tokens.numel(),
                terminated=bool(generated and generated[-1]==model.config.eos_token_id)))
            save_json(a.output,report)
            print(json.dumps(dict(stage='probe',**report['probes'][-1])),flush=True)
    report['summary']=dict(arc=summarize(report['arc'],'subset'),probes=summarize(report['probes'],'category'),
        overall_probes=dict(correct=sum(r['correct'] for r in report['probes']),total=len(report['probes'])),
        elapsed_seconds=time.perf_counter()-start)
    save_json(a.output,report);print(json.dumps(report['summary'],indent=2),flush=True)

if __name__=='__main__':main()
