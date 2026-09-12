from pathlib import Path
import json
import subprocess
import sys
import pytest
import torch
from energy_transformer.review import Edit, ReviewedEnergySession, state_digest
from energy_transformer.bridge import ProtectedState

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / 'checkpoints/v14/bridge_seed11.pt'
torch.set_num_threads(1)

@pytest.fixture
def s():
    if not CHECKPOINT.exists():
        pytest.skip('Model-specific: requires the packaged or locally trained v14 checkpoint')
    return ReviewedEnergySession(CHECKPOINT)

def pending(s):
    response = s.propose('Alice is in the garden.')
    assert response['status'] == 'needs_review'
    return response['proposal']

def test_proposal_never_writes(s):
    before = s.state_payload()
    pending(s)
    assert all(torch.equal(before[k], s.state_payload()[k]) for k in before)
    assert s.revision == 0 and s.answer('alice') == 'unknown'

def test_approval_default_denies(s):
    p = pending(s)
    with pytest.raises(PermissionError):
        s.approve(p['proposal_id'], reviewed_edit=Edit('set','alice','garden'), expected_revision=0)
    assert s.answer('alice') == 'unknown'

def test_exact_review_required(s):
    p = pending(s)
    with pytest.raises(ValueError, match='meaning'):
        s.approve(p['proposal_id'], reviewed_edit=Edit('set','alice','kitchen'), expected_revision=0, authorized=True)
    assert s.revision == 0

def test_approval_and_replay(s):
    p = pending(s)
    args = dict(reviewed_edit=Edit('set','alice','garden'),expected_revision=0,authorized=True)
    assert s.approve(p['proposal_id'], **args) == 1
    assert s.answer('alice') == 'garden'
    with pytest.raises(ValueError):
        s.approve(p['proposal_id'], **args)

def test_stale_proposals_are_invalidated(s):
    p = pending(s)
    s.apply_edits([Edit('set','bob','office')],expected_revision=0,authorized=True)
    with pytest.raises(ValueError):
        s.approve(p['proposal_id'],reviewed_edit=Edit('set','alice','garden'),expected_revision=1,authorized=True)

def test_atomic_batch_rollback(s):
    with pytest.raises(ValueError):
        s.apply_edits([Edit('set','alice','garden'),Edit('set','zara','garden')],expected_revision=0,authorized=True)
    assert s.revision == 0 and s.answer('alice') == 'unknown'

def test_revision_type_and_value(s):
    for revision in [True, -1, 1, '0']:
        with pytest.raises(ValueError):
            s.apply_edits([Edit('set','alice','garden')],expected_revision=revision,authorized=True)

def test_direct_write_requires_authorization(s):
    with pytest.raises(PermissionError):
        s.apply_edits([Edit('set','alice','garden')],expected_revision=0)

def test_public_payload_cannot_mutate_state(s):
    p = s.state_payload()
    p['information'].fill_(99)
    assert s.state_payload()['information'].sum() == 0

def test_bounded_pending(s):
    s._max_pending = 1
    p = pending(s)
    assert s.propose('Alice is in the garden.')['reason'] == 'pending_limit'
    assert s.discard(p['proposal_id'])
    pending(s)

def test_save_restore_invalidates_pending(s, tmp_path):
    pending(s)
    s.save(tmp_path/'state.pt')
    s.restore(tmp_path/'state.pt')
    assert not s._pending and s.revision == 0

def test_model_and_integrity_checks(s,tmp_path):
    p = tmp_path/'s.pt'
    s.save(p)
    data = torch.load(p,weights_only=True)
    data['state']['information'][0,0] = 1
    torch.save(data,p)
    with pytest.raises(ValueError,match='integrity'):
        s.restore(p)

def test_fresh_process_review_restart(s,tmp_path):
    s.apply_edits([Edit('set','alice','garden')],expected_revision=0,authorized=True)
    file = tmp_path/'s.pt'; s.save(file)
    code = '''import json,sys,torch
from energy_transformer.review import ReviewedEnergySession
torch.set_num_threads(1)
s=ReviewedEnergySession(sys.argv[1]);s.restore(sys.argv[2]);print(json.dumps([s.revision,s.answer('alice')]))'''
    out = subprocess.run([sys.executable,'-c',code,str(CHECKPOINT),str(file)],cwd=ROOT,capture_output=True,text=True,check=True)
    assert json.loads(out.stdout) == [1,'garden']

def test_edit_validation_without_weights():
    for edit in [Edit('hold','alice'), Edit('set','zara','garden'),Edit('delete','alice','garden'), Edit('set','alice')]:
        with pytest.raises(ValueError):
            edit.validate()
    before = ProtectedState.initial()
    after = Edit('set','alice','garden').apply(before)
    assert state_digest(before) != state_digest(after)
