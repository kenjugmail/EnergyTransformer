import torch

from rewt.model import TransformerConfig, TransformerLM


def test_training_attention_details_preserve_predictions_and_are_causal():
    torch.manual_seed(4120)
    model = TransformerLM(TransformerConfig(width=8, layers=2, heads=2, feedforward=15, context=32))
    tokens = torch.randint(0, 128, (2, 18))
    plain = model(tokens)
    detailed = model(tokens, details=True)
    torch.testing.assert_close(plain.logits, detailed.logits, rtol=0, atol=0)
    changed = tokens.clone()
    changed[:, 11:] = (changed[:, 11:] + 17) % 128
    other = model(changed, details=True)
    for key in ('hidden', 'attention_query', 'attention_key', 'attention_value', 'attention_inputs'):
        torch.testing.assert_close(detailed.details[key][:, :11], other.details[key][:, :11], rtol=0, atol=0)
    q = detailed.details['attention_query']
    assert q.shape == (2, 18, 2, 4)
    # The exposed tensors retain gradients for training-only fact labels.
    (q.square().mean() + detailed.details['attention_key'].square().mean()).backward()
    assert model.layers[-1].qkv.weight.grad is not None


def test_training_details_use_incremental_output_positions():
    model = TransformerLM(TransformerConfig(width=8, layers=2, heads=2, feedforward=15, context=32))
    tokens = torch.tensor([[65, 66, 67, 68, 69]])
    first = model(tokens[:, :3])
    continued = model(tokens[:, 3:], state=first.state, details=True)
    whole = model(tokens, details=True)
    torch.testing.assert_close(continued.logits, whole.logits[:, 3:], rtol=0, atol=0)
    for key in ('hidden', 'attention_query', 'attention_key', 'attention_value', 'attention_inputs'):
        torch.testing.assert_close(continued.details[key], whole.details[key][:, 3:], rtol=0, atol=0)
