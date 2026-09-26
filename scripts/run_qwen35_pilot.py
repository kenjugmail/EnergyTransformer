import os,sys,json,time,signal,subprocess
from pathlib import Path
from energy_campaign.budget import Ledger
from energy_campaign.shared_probe import supervise_shared
from energy_campaign.spec import source_identity
from energy_real.device import available_host_bytes
from energy_real.runtime import atomic_json,sha256
root=Path('runs/campaign/qwen08');receipt=json.loads((root/'cpu_receipt.json').read_text())
if receipt['source']!=source_identity() or receipt['status']!='passed':raise RuntimeError('Qwen CPU receipt does not match')
if available_host_bytes()<24*2**30:print('WAITING_FOR_HOST_HEADROOM');sys.exit(0)
if (root/'pilot002').exists() or (root/'gpu_receipt002.json').exists():raise RuntimeError('Existing pilot requires inspection; no automatic retry')
ledger=Ledger('runs/campaign/budget.json');key=ledger.reserve('conversion',1800,{'source':source_identity(),'job':{'action':'qwen35_full_hybrid_feasibility'},'protocol':sha256(root/'protocol.json'),'runtime':'transformers5.17.0_existing_torch2.14','cap_gib':4});start=time.monotonic();p=None;status='failed';reason=None
try:
 with (root/'gpu_output002.log').open('w') as log:
  p=subprocess.Popen([sys.executable,'-m','energy_campaign.qwen35_pilot','run',str(root),'pilot002'],start_new_session=True,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
  status,reason=supervise_shared(p,1800,start)
finally:
 if p is not None and p.poll() is None:os.killpg(p.pid,signal.SIGKILL);p.wait()
 elapsed=time.monotonic()-start;ledger.settle(key,elapsed,status);atomic_json({'status':status,'reason':reason,'seconds':elapsed,'reservation':key},root/'gpu_receipt002.json')
print(json.dumps({'status':status,'reason':reason,'seconds':elapsed}))
