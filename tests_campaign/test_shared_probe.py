import signal,subprocess
import pytest
from energy_campaign import shared_probe
from energy_real.train import parser,run


def test_shared_probe_stops_own_child_on_memory_pressure(monkeypatch):
    class Child:
        pid=123;calls=0
        def wait(self,timeout=None):
            self.calls+=1
            if self.calls==1:raise subprocess.TimeoutExpired('fixture',timeout)
            return 0
        def poll(self):return None
    signals=[]
    monkeypatch.setattr(shared_probe,'available_host_bytes',lambda:15*2**30)
    monkeypatch.setattr(shared_probe.os,'killpg',lambda pid,s:signals.append((pid,s)))
    status,reason=shared_probe.supervise_shared(Child(),120,shared_probe.time.monotonic())
    assert status=='interrupted' and reason=='host_memory_reserve_exhausted'
    assert signals==[(123,signal.SIGTERM)]


def test_shared_flag_cannot_authorize_full_training(tmp_path):
    args=parser().parse_args(['--out',str(tmp_path/'run'),'--shared-gpu-diagnostic'])
    with pytest.raises(ValueError,match='Shared diagnostic'):run(args)
    assert not (tmp_path/'run').exists()


@pytest.mark.parametrize('device,index',[('cuda',2),('cuda:1',1)])
def test_memory_cap_resolves_explicit_device_index(monkeypatch,device,index):
    import torch
    from types import SimpleNamespace
    from energy_real.device import set_memory_limit
    calls=[]
    monkeypatch.setattr(torch.cuda,'current_device',lambda:2)
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda i:SimpleNamespace(total_memory=64*2**30))
    monkeypatch.setattr(torch.cuda,'set_per_process_memory_fraction',lambda fraction,i:calls.append((fraction,i)))
    set_memory_limit(device,4)
    assert calls==[(4/64,index)]
