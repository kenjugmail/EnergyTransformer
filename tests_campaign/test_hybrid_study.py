import json
import torch
import pytest
from energy_real.train import parser,run
from energy_campaign.hybrid_study import run as study,compare_restart
from energy_campaign.hybrid_validation import check_model
from energy_real.model import LM,Config


def test_study_waits_without_charging_or_starting(tmp_path,monkeypatch):
    import energy_campaign.hybrid_study as module
    monkeypatch.setattr(module,'available_host_bytes',lambda:19*2**30)
    result=study(tmp_path/'campaign',tmp_path/'study')
    assert result['status']=='WAITING_FOR_HOST_HEADROOM'
    assert not (tmp_path/'campaign/budget.json').exists()


@pytest.mark.parametrize('extra',[['--kind','energy'],['--steps','129'],['--max-cuda-gib','5'],['--compile']])
def test_hybrid_study_does_not_authorize_excluded_or_unbounded_work(tmp_path,extra):
    args=parser().parse_args(['--out',str(tmp_path/'fit'),'--shared-hybrid-study','--kind','hybrid',
          '--width','160','--layers','3','--heads','4','--slots','16','--ff-width','640','--attention-window','128',
          '--batch','1','--seq','512','--accum','16','--steps','128','--max-cuda-gib','4','--max-seconds','500',*extra])
    with pytest.raises(ValueError):run(args)
    assert not (tmp_path/'fit').exists()


def test_hybrid_audit_has_bounded_cosine_and_streaming():
    torch.manual_seed(404);model=LM(Config(kind='hybrid',width=16,layers=3,heads=2,slots=4,ff_width=32,attention_window=8,chunk=8))
    tokens=torch.randint(256,(1,34));row=check_model(model,tokens[:,:-1],tokens[:,1:],True)
    assert row['passed'] and -1<=row['gradient_cosine']<=1
    assert row['future_max_abs']==0 and row['checkpoint_gradient_relative']==0


def test_restart_audit_checks_rng_optimizer_and_committed_steps(tmp_path):
    state={k:torch.tensor([1.]) for k in ('weights','optimizer','torch_rng','cuda_rng','numpy_rng','coverage')}
    state.update(step=4,tokens=32768)
    for name in ('restart_whole','restart_split'):
        p=tmp_path/name;p.mkdir();torch.save(state,p/'last.pt')
    log=tmp_path/'restart_split/steps.jsonl'
    log.write_text(''.join(json.dumps({'event':'step','step':i})+'\n' for i in (1,2,3,4)))
    compare_restart(tmp_path)
    log.write_text(log.read_text()+json.dumps({'event':'step','step':4})+'\n')
    with pytest.raises(AssertionError,match='committed'):compare_restart(tmp_path)
