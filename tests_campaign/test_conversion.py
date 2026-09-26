import torch
from transformers import LlamaConfig,LlamaForCausalLM
from energy_campaign.conversion import ConvertedLM,cache_bytes


def test_conversion_stream_cache_and_portable_checkpoint(tmp_path):
    torch.manual_seed(3)
    cfg=LlamaConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128)
    model=ConvertedLM(LlamaForCausalLM(cfg),layers=(2,),slots=4,lora=True).eval()
    tokens=torch.randint(64,(2,11))
    with torch.no_grad():
        whole=model(tokens).logits;cache=None;parts=[]
        for i in range(11):
            r=model(tokens[:,i:i+1],past_key_values=cache,use_cache=True);cache=r.past_key_values;parts.append(r.logits)
        torch.testing.assert_close(whole,torch.cat(parts,1),atol=2e-5,rtol=2e-5)
        assert cache.energy[2][0].shape==(2,4,4) and cache_bytes(cache)>0
        path=tmp_path/'converted.pt';model.save(path);restored=ConvertedLM.load(path).eval()
        torch.testing.assert_close(whole,restored(tokens).logits,atol=0,rtol=0)


def test_static_mixed_cache_reset_and_teacher_forced_parity():
    from energy_campaign.conversion import EnergyStaticCache
    cfg=LlamaConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128)
    model=ConvertedLM(LlamaForCausalLM(cfg),layers=(2,),slots=4).eval();x=torch.randint(64,(1,12))
    with torch.no_grad():
        full=model(x).logits;cache=EnergyStaticCache(cfg,12);a=model(x[:,:8],past_key_values=cache,use_cache=True).logits
        b=model(x[:,8:],past_key_values=cache,use_cache=True).logits
        torch.testing.assert_close(full,torch.cat((a,b),1),atol=2e-5,rtol=2e-5)
        size=cache_bytes(cache);cache.reset();assert not cache.energy
        again=model(x,past_key_values=cache,use_cache=True).logits
        torch.testing.assert_close(full,again,atol=2e-5,rtol=2e-5);assert cache_bytes(cache)==size
