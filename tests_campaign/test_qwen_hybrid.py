import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
from energy_campaign.qwen_hybrid import QwenHybrid

def tiny():
    torch.manual_seed(17)
    return Qwen2ForCausalLM(Qwen2Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2)).eval()

def test_unmodified_attention_parity():
    base=tiny();x=torch.randint(64,(1,19))
    with torch.no_grad():expected=base(x).logits;actual,_=QwenHybrid(base,window=32,energy_layers=())(x)
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)

def test_full_hybrid_streaming_causality_and_save(tmp_path):
    m=QwenHybrid(tiny(),window=8,slots=4).eval();x=torch.randint(64,(1,23))
    with torch.no_grad():
        full,_=m(x);s=None;parts=[]
        for chunk in x.split(5,dim=1):
            y,s=m(chunk,s);parts.append(y)
        torch.testing.assert_close(torch.cat(parts,1),full,atol=2e-5,rtol=2e-4)
        assert s['offset']==23 and s['layers'][2][0].shape[2]==8
        changed=x.clone();changed[:,12:]=0
        torch.testing.assert_close(m(changed)[0][:,:12],full[:,:12],atol=1e-6,rtol=0)
        p=tmp_path/'m.pt';m.save(p);loaded=QwenHybrid.load(p).eval()
        torch.testing.assert_close(loaded(x)[0],full,atol=0,rtol=0)
    assert m.energy_layers==(0,1)
