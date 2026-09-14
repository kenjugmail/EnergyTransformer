import json
import time
import pytest
import torch
from energy_real import diagnostics,recall_study,recall_followup


def test_controller_only_keeps_geometry_and_gain_unchanged(tmp_path):
    train=diagnostics.generate('train',1,271101)
    valid=diagnostics.generate('valid',1,271102)
    record=recall_study.fit('precision',16,.001,11,3,tmp_path/'fit',train,valid,time.perf_counter()+60,
                           training_mode='controller_only')
    assert record['training_mode']=='controller_only' and record['frozen_parameters_unchanged']
    initial=torch.load(record['initial_controller'],weights_only=True)
    final=torch.load(record['controller'],weights_only=True)
    assert torch.equal(initial['weights']['key.weight'],final['weights']['key.weight'])
    assert torch.equal(initial['weights']['log_gain'],final['weights']['log_gain'])
    assert record['steps']==3 and len(record['history'])==1
    with pytest.raises(ValueError,match='training mode'):
        recall_study.fit('precision',16,.001,11,1,tmp_path/'invalid',train,valid,time.perf_counter()+1,training_mode='unknown')


def test_followup_selects_before_fresh_test_and_runs_both_modes(tmp_path,monkeypatch):
    monkeypatch.setattr(recall_study,'measure_peaks',lambda *args,**kwargs:[])
    args=recall_followup.parser().parse_args(['--out',str(tmp_path/'study'),'--steps','1',
           '--train-per-family','1','--valid-per-family','1','--test-per-family','1','--max-seconds','120'])
    result=recall_followup.run(args)
    assert result['status']=='completed' and len(result['records'])==12 and len(result['test_results'])==12
    assert len(result['selected'])==6
    assert (args.out/'selection.json').stat().st_mtime_ns<=(args.out/'test_episodes.json').stat().st_mtime_ns
    assert all(r['frozen_parameters_unchanged'] for r in result['records'] if r['training_mode']=='controller_only')
    fresh={e.id for e in diagnostics.generate('test',1,271103)}
    assert not fresh & {e.id for e in diagnostics.generate('test',1,271003)}
    assert {r['episode_id'] for r in result['test_results'][0]['trained']['rows']}==fresh
    assert json.loads((args.out/'exact_oracle.json').read_text())['summary']['accuracy']==1
    decision=json.loads((args.out/'decision.json').read_text())
    assert set(decision['learning_claims'])=={f'{k}/{m}' for k in recall_study.KINDS for m in recall_followup.MODES}
    for kind in recall_study.KINDS:
        all_initial=next(t for t in result['test_results'] if (t['kind'],t['training_mode'],t['seed'])==(kind,'all',11))['initial']['rows']
        frozen_initial=next(t for t in result['test_results'] if (t['kind'],t['training_mode'],t['seed'])==(kind,'controller_only',11))['initial']['rows']
        assert all_initial==frozen_initial
    with pytest.raises(FileExistsError):recall_followup.run(args)


def test_mode_selection_uses_seed11_validation_only():
    def row(seed,mode,accuracy):
        return {'kind':'precision','seed':seed,'training_mode':mode,
                'validation':{'selected':{'accuracy':accuracy,'unsupported_answers':0,'support_accuracy':accuracy}}}
    records=[row(11,'all',.9),row(11,'controller_only',.8),row(22,'controller_only',1.)]
    assert recall_followup.choose(records,'precision')=='all'
    records[1]['validation']['selected']['accuracy']=.9
    records[1]['validation']['selected']['support_accuracy']=.9
    assert recall_followup.choose(records,'precision')=='controller_only'
