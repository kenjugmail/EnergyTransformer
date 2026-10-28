import pytest
import torch
from rewt.streaming import StreamingConfig,StreamingLM
from rewt.streaming_decode import StreamingSession


@pytest.mark.parametrize('hybrid,window',[(False,0),(False,8),(True,8),(True,1)])
def test_preallocated_session_matches_full_forward_and_owns_output(hybrid,window):
    torch.manual_seed(74)
    model=StreamingLM(StreamingConfig(width=16,heads=2,hidden=24,modes=4,hybrid=hybrid,window=window)).eval()
    x=torch.randint(256,(2,51));session=StreamingSession(model,batch=2,capacity=60)
    with torch.inference_mode():expected,_,_=model(x)
    outputs=[]
    for a,b in [(0,1),(1,18),(18,19),(19,48),(48,51)]:outputs.append(session.consume(x[:,a:b]))
    torch.testing.assert_close(torch.cat(outputs,1),expected,atol=2e-6,rtol=2e-5)
    preserved=outputs[-1].clone();size=session.allocated_state_bytes()
    session.consume(torch.randint(256,(2,1)))
    torch.testing.assert_close(outputs[-1],preserved,atol=0,rtol=0)
    assert size==session.allocated_state_bytes()


def test_session_rejects_capacity_batch_and_mutation():
    model=StreamingLM(StreamingConfig(width=16,heads=2,hidden=24,modes=4)).eval()
    session=StreamingSession(model,capacity=4)
    with pytest.raises(ValueError,match='capacity'):session.consume(torch.zeros(1,5,dtype=torch.long))
    with pytest.raises(ValueError,match='session_batch'):session.consume(torch.zeros(2,1,dtype=torch.long))
    with torch.no_grad():model.embedding.weight.add_(1.)
    with pytest.raises(ValueError,match='weights'):session.consume(torch.zeros(1,1,dtype=torch.long))
