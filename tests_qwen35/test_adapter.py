import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
from energy_campaign.qwen35_hybrid import Qwen35Hybrid

def tiny(types):
    torch.manual_seed(17)
    c=Qwen3_5TextConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=len(types),num_attention_heads=2,num_key_value_heads=1,head_dim=16,linear_num_key_heads=2,linear_num_value_heads=2,linear_key_head_dim=16,linear_value_head_dim=16,layer_types=types,rope_parameters={'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':1.,'mrope_section':[3,3,2]})
    return Qwen3_5ForCausalLM(c).eval()

def test_attention_preserves_gates_and_rotary():
    base=tiny(['full_attention']);x=torch.randint(64,(1,19))
    with torch.no_grad():expected=base(x).logits;actual,_=Qwen35Hybrid(base,window=32,energy_layers=())(x)
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)

def test_hybrid_causal_stream_save(tmp_path):
    m=Qwen35Hybrid(tiny(['linear_attention']*3+['full_attention']),window=8,slots=4);x=torch.randint(64,(1,23))
    with torch.no_grad():
        z,_=m(x);state=None;parts=[]
        for chunk in x.split(5,dim=1):y,state=m(chunk,state);parts.append(y)
        torch.testing.assert_close(torch.cat(parts,1),z,atol=2e-5,rtol=2e-4)
        assert state['layers'][2][0].shape[2]==8
        changed=x.clone();changed[:,12:]=0;torch.testing.assert_close(m(changed)[0][:,:12],z[:,:12],atol=1e-6,rtol=0)
        p=tmp_path/'model.pt';m.save(p);torch.testing.assert_close(Qwen35Hybrid.load(p)(x)[0],z,atol=0,rtol=0)


def test_full_pattern_trainable_adapters_and_backward():
    m=Qwen35Hybrid(tiny(['linear_attention']*3+['full_attention']),window=8,slots=4,lora=True)
    assert m.energy_layers==(0,1,3)
    assert all(not hasattr(l,'linear_attn') and not hasattr(l,'self_attn') for l in m.base.model.layers)
    x=torch.randint(64,(1,9))
    with torch.autocast('cpu',dtype=torch.bfloat16):z,_=m(x);loss=z.float().square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
    assert not m.base.model.embed_tokens.weight.requires_grad
