import json
from pathlib import Path
import pytest
import torch
from energy_transformer.bridge import EnergyTextSession, ProtectedState
from energy_transformer.portable import export_portable, PortableEnergyTextSession

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('seed',[11,22,33])
def test_all_tensor_and_prediction_parity(tmp_path,seed):
    path=ROOT/f'checkpoints/v14/bridge_seed{seed}.pt'
    if not path.exists():pytest.skip('Checkpoint required')
    source=EnergyTextSession(path)
    export_portable(path,tmp_path/'p')
    restored=PortableEnergyTextSession(tmp_path/'p')
    for a,b in [(source.parser,restored.parser),(source.decoder,restored.decoder)]:
        for k,v in a.state_dict().items():assert torch.equal(v,b.state_dict()[k])
    texts=['Alice is in the garden.','Where is Alice?','Forget the location of Alice.']
    with torch.inference_mode():
        x=source.codec.encode(texts)
        for a,b in zip(source.parser(x),restored.parser(x)):assert torch.equal(a,b)
        for value in range(17):
            state=ProtectedState.initial()
            if value<16:state=state.edit(1,0,value)
            assert torch.equal(source.decoder(state.feature(0)[None]),restored.decoder(state.feature(0)[None]))


def test_portable_tamper_and_overwrite(tmp_path):
    path=ROOT/'checkpoints/v14/bridge_seed11.pt'
    if not path.exists():pytest.skip('Checkpoint required')
    export_portable(path,tmp_path/'p')
    with pytest.raises(FileExistsError):export_portable(path,tmp_path/'p')
    f=tmp_path/'p'/'model.safetensors'; data=bytearray(f.read_bytes());data[-1]^=1;f.write_bytes(data)
    with pytest.raises(ValueError,match='checksum'):PortableEnergyTextSession(tmp_path/'p')


def test_bad_portable_schema(tmp_path):
    (tmp_path/'config.json').write_text('{}')
    with pytest.raises(ValueError):PortableEnergyTextSession(tmp_path)
