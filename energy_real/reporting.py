"""Generate a portable summary from explicit study, ledger, and test evidence."""
import argparse
import json
from pathlib import Path
import re
import statistics
from .runtime import atomic_json, sha256


def read(path):
    return json.loads(Path(path).read_text())


def test_result(path):
    text = Path(path).read_text()
    summaries = [line for line in text.splitlines() if re.search(r'\d+ (passed|failed|error)',line)]
    if not summaries:
        raise ValueError(f'No pytest result summary in {path}')
    line = summaries[-1]
    counts = {key:int((re.search(rf'(\d+) {key}',line) or [None,0])[1])
              for key in ('passed','failed','skipped','error')}
    return {**counts,'status':'passed' if counts['passed'] and not counts['failed'] and not counts['error'] else 'failed',
            'sha256':sha256(path)}


def build(study, ledger, out, test_logs):
    study, ledger, out = Path(study), Path(ledger), Path(out)
    run = read(study/'summary.json')
    stages = read(ledger)
    groups = {}
    for row in run.get('test_results',[]):
        key = f"{row['kind']}/{row.get('training_mode','all')}"
        groups.setdefault(key,[]).append(row)
    table = []
    for key, rows in groups.items():
        table.append({'mechanism':key,'seeds':[r['seed'] for r in rows],
                      'initial_accuracy':statistics.mean(r['initial']['summary']['accuracy'] for r in rows),
                      'selected_accuracy':statistics.mean(r['trained']['summary']['accuracy'] for r in rows),
                      'unsupported_answers':sum(r['trained']['summary']['unsupported_answers'] for r in rows),
                      'per_seed':[{'seed':r['seed'],'initial':r['initial']['summary'],'selected':r['trained']['summary'],
                                   'learning_interval':r['learning_interval']} for r in rows]})
    suites = {name:test_result(path) for name,path in test_logs.items()}
    result = {'status':run['status'],'source_sha256':run['source_sha256'],
              'study_summary_sha256':sha256(study/'summary.json'),'ledger_sha256':sha256(ledger),
              'fits':len(run.get('records',[])),'updates':sum(r['steps'] for r in run.get('records',[])),
              'elapsed_s':run.get('elapsed_s'),'gpu_seconds':run.get('gpu_seconds'),
              'tests':suites,'verification_status':('passed' if suites and all(s['status']=='passed' for s in suites.values())
                                                   else 'failed' if suites else 'not_supplied'),
              'memory':table,'stage_usage':stages['stages']}
    for name in ('decision','costs','exact_oracle'):
        if (study/f'{name}.json').exists():
            value=read(study/f'{name}.json')
            result[name]=value.get('summary',value) if isinstance(value,dict) else value
    out.mkdir(parents=True,exist_ok=False)
    for name,path in test_logs.items():
        if not re.fullmatch(r'[A-Za-z0-9_-]+',name):raise ValueError('Invalid test suite name')
        (out/f'tests_{name}.log').write_bytes(Path(path).read_bytes())
    atomic_json(result,out/'results.json')
    lines=['# EnergyTransformer follow-up results','',
           f"Study: **{result['status']}**. Verification: **{result['verification_status']}**.",
           f"Fits: {result['fits']}; optimizer updates: {result['updates']}; measured study time: {result['elapsed_s']:.2f} seconds.",'',
           '| Mechanism / training | Initial accuracy | Selected accuracy | Unsupported answers |',
           '|---|---:|---:|---:|']
    for row in table:
        lines.append(f"| {row['mechanism']} | {100*row['initial_accuracy']:.2f}% | {100*row['selected_accuracy']:.2f}% | {row['unsupported_answers']} |")
    lines+=['','These are structured-memory results. Live reasoning and device measurements have separate stage statuses.','',
            '## Stage status','']
    for name,stage in stages['stages'].items():
        lines.append(f"- {name}: {stage['status']}; charged time {stage['used_s']:.2f} seconds.")
    lines+=['','## Test evidence','']
    for name,row in suites.items():
        lines.append(f"- {name}: {row['passed']} passed, {row['skipped']} skipped, {row['failed']} failed, {row['error']} errors.")
    lines+=['','[Machine-readable results, paired intervals, costs, and decisions](results.json).','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--ledger',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--test-log',action='append',default=[],metavar='NAME=PATH')
    a=p.parse_args()
    logs=dict(item.split('=',1) for item in a.test_log)
    result=build(a.study,a.ledger,a.out,logs)
    print(json.dumps({'status':result['status'],'verification_status':result['verification_status']}))


if __name__=='__main__':main()
