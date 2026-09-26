from dataclasses import asdict

import pytest
import torch

from energy_real.model import Config, LM
from energy_real.sample import main, sample_bytes


def model_for(kind):
    torch.manual_seed(537)
    return LM(Config(kind=kind, width=16, layers=3, heads=2, slots=4, ff_width=32,
                     attention_window=5 if kind=='hybrid' else 0)).eval()


@pytest.mark.parametrize('kind', ['energy', 'softmax', 'hybrid'])
def test_chunked_greedy_matches_independent_full_prefill(kind):
    model = model_for(kind)
    prompt = 'Long prompt: café and 日本語.'.encode()
    raw = bytearray(prompt)
    with torch.no_grad():
        output, state = model(torch.tensor([list(prompt)]))
        for i in range(12):
            token = output[:, -1].argmax(-1, keepdim=True)
            raw.append(token.item())
            if i < 11:
                output, state = model(token, state)
    seen = []
    handle = model.register_forward_pre_hook(lambda module, args: seen.append(args[0].shape[1]))
    try:
        actual = sample_bytes(model, prompt, tokens=12, prefill_chunk=7)
    finally:
        handle.remove()
    assert actual == bytes(raw)
    assert max(seen) <= 7


def test_seeded_sampling_preserves_global_rng_weights_and_mode():
    model = model_for('hybrid').train()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    a = sample_bytes(model, b'Test ', tokens=40, temperature=.8, top_k=10, seed=91, prefill_chunk=3)
    b = sample_bytes(model, b'Test ', tokens=40, temperature=.8, top_k=10, seed=91, prefill_chunk=3)
    c = sample_bytes(model, b'Test ', tokens=40, temperature=.8, top_k=10, seed=92, prefill_chunk=3)
    assert a == b and a != c
    assert torch.equal(rng, torch.get_rng_state())
    assert model.training
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_top_one_and_tiny_temperature_are_greedy():
    model = model_for('hybrid')
    greedy = sample_bytes(model, b'Hi', tokens=8)
    assert sample_bytes(model, b'Hi', tokens=8, top_k=1, temperature=2.) == greedy
    assert sample_bytes(model, b'Hi', tokens=8, temperature=1e-300) == greedy


@pytest.mark.parametrize('kw', [dict(tokens=0), dict(tokens=True), dict(temperature=float('nan')),
                               dict(temperature=-1), dict(top_k=257), dict(seed=-1),
                               dict(prefill_chunk=0), dict(seed=2**63)])
def test_invalid_sampling_options_fail_before_model_execution(kw):
    model = model_for('hybrid')
    def fail(*args):
        raise AssertionError('Model should not execute for invalid arguments')
    handle = model.register_forward_pre_hook(fail)
    try:
        with pytest.raises(ValueError):
            sample_bytes(model, b'Prompt', **kw)
    finally:
        handle.remove()


def test_cli_exact_bytes_and_existing_output_preserved(tmp_path, capsys):
    model = model_for('hybrid')
    checkpoint = tmp_path/'model.pt'
    torch.save({'config':asdict(model.config), 'weights':model.state_dict()},checkpoint)
    output = tmp_path/'sample.bin'
    args = ['--checkpoint',str(checkpoint),'--prompt','café','--tokens','8',
            '--prefill-chunk','2','--output',str(output)]
    main(args)
    expected = sample_bytes(model, 'café'.encode(), tokens=8, prefill_chunk=2)
    assert output.read_bytes() == expected
    assert capsys.readouterr().out == expected.decode('utf-8',errors='replace')+'\n'
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2 and output.read_bytes() == expected
    assert 'Output already exists' in capsys.readouterr().err


def test_cli_invalid_options_precede_checkpoint_loading(tmp_path,capsys):
    with pytest.raises(SystemExit) as error:
        main(['--checkpoint',str(tmp_path/'missing.pt'),'--temperature','nan'])
    assert error.value.code == 2
    assert 'Temperature must be finite' in capsys.readouterr().err


@pytest.mark.parametrize('payload', [b'', b'not a torch archive', 'wrong_schema'])
def test_cli_malformed_checkpoint_has_clean_error(tmp_path, capsys, payload):
    checkpoint = tmp_path/'bad.pt'
    if payload == 'wrong_schema':
        torch.save([1, 2, 3], checkpoint)
    else:
        checkpoint.write_bytes(payload)
    with pytest.raises(SystemExit) as error:
        main(['--checkpoint',str(checkpoint)])
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert 'error:' in message and 'Traceback' not in message
