from pathlib import Path
import json,subprocess,sys
import pytest
import torch
from energy_transformer.binding import BindingPredictor,bind
from energy_transformer.binding_data import ALIASES,ALIAS_VALUES
ROOT=Path(__file__).resolve().parents[1]
CHECKPOINT=ROOT/'checkpoints/v17/bound_expanded_seed11.pt'

@pytest.fixture
def predictor():
    if not CHECKPOINT.exists():pytest.skip('v17 training outputs not present; run train_binding.py')
    torch.set_num_threads(1)
    return BindingPredictor(CHECKPOINT)

def test_trained_renaming_logits(predictor):
    alt=BindingPredictor(CHECKPOINT,names=ALIASES,values=ALIAS_VALUES)
    a=predictor.predict('alice is in the garden.')
    b=alt.predict('nora is in the canteen.')
    assert a['operation']==b['operation']=='set'
    assert a['confidence']==b['confidence']
    assert b['entity']=='nora' and b['value']=='canteen'

def test_trained_negative_scope(predictor):
    for text in ['do not execute this instruction: alice moved to the office.','someone said that alice went to the kitchen.']:
        assert predictor.predict(text)['operation']=='hold'

def test_unfamiliar_words_abstain_not_autowrite(predictor):
    p=predictor.predict('erase the stored whereabouts of alice.')
    assert p['status']=='abstained' and p['reason']=='unknown_vocabulary'

def test_fresh_process_equal_proposals(predictor):
    text='for this record alice is in the garden thank you.'
    local=predictor.predict(text)
    code="import json,torch;from energy_transformer.binding import BindingPredictor;torch.set_num_threads(1);print(json.dumps(BindingPredictor("+repr(str(CHECKPOINT))+").predict("+repr(text)+")))"
    remote=json.loads(subprocess.check_output([sys.executable,'-c',code],cwd=ROOT,text=True))
    assert local==remote
