from pathlib import Path
import json
import pytest
import torch
from energy_transformer.binding import bind,OperationParser,BindingPredictor
from energy_transformer.binding_data import ALIASES,ALIAS_VALUES,catalog,pair_heldout
from energy_transformer.bridge import NAMES,VALUES,TextCodec

def test_binding_semantics_not_predicted_identity():
    a=bind('Alice is in the garden.')
    assert a.key==0 and a.value==2 and a.canonical=='entitytoken is in the valuetoken .'

def test_registered_renaming_equivariance():
    for k in range(16):
        for v in range(16):
            a=bind(f'{NAMES[k]} moved to the {VALUES[v]}.')
            b=bind(f'{ALIASES[k]} moved to the {ALIAS_VALUES[v]}.',ALIASES,ALIAS_VALUES)
            assert a==b

@pytest.mark.parametrize('text',['alice and bob are in the garden','alice says alice is in the garden','alice moved from the garden to the office'])
def test_multiple_mentions_rejected(text):assert bind(text).reason is not None

def test_empty_and_long_rejected():
    assert bind('').reason
    assert bind('x '*41).reason

def test_bad_registries_rejected():
    with pytest.raises(ValueError):bind('text',NAMES[:-1],VALUES)
    with pytest.raises(ValueError):bind('text',NAMES,NAMES)

def test_operation_gradients_and_padding():
    torch.manual_seed(11);m=OperationParser(12);x=torch.randint(1,12,(3,9));a=m(x)
    b=m(torch.cat([x,torch.zeros(3,6,dtype=torch.long)],1))
    torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
    a.sum().backward();assert torch.isfinite(m.embedding.weight.grad).all()

def test_split_invariants():
    root=Path(__file__).resolve().parents[1]/'results/v17/data'
    if not root.exists():pytest.skip('Generate v17 benchmark first')
    datasets={p.stem:[json.loads(s) for s in p.read_text().splitlines()] for p in root.glob('*.jsonl')}
    train={r['text'] for key in ('train_basic','train_expanded') for r in datasets[key]}
    for key in ('validation','iid','composition','pairs','legacy_wording','aliases'):
        assert not train.intersection(r['text'] for r in datasets[key])
    templates={r['template_id'] for r in datasets['train_expanded']}
    assert not templates.intersection(r['template_id'] for r in datasets['composition'])
    for key in ('train_basic','train_expanded'):
        assert not any(r['op']==1 and pair_heldout(r['key'],r['value']) for r in datasets[key])


def test_predictor_is_proposal_only(tmp_path):
    codec=TextCodec.fit(['entitytoken is in the valuetoken .'])
    m=OperationParser(len(codec.vocabulary))
    with torch.no_grad():
        for p in m.parameters():p.zero_()
        m.op.bias[1]=10
    path=tmp_path/'bound.pt'
    torch.save({'format':'binding_proposer_v17','kind':'bound','vocabulary':codec.vocabulary,'weights':m.state_dict()},path)
    parser=BindingPredictor(path)
    result=parser.predict('alice is in the garden.')
    assert result['requires_review'] and result['status']=='proposal_only'
    assert not hasattr(parser,'state') and not hasattr(parser,'approve')
    assert parser.predict('alice is allegedly in the garden.')['status']=='abstained'
