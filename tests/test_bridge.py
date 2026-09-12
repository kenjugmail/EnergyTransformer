from pathlib import Path
import subprocess,sys,json
import pytest
import torch
from energy_transformer.bridge import *
from energy_transformer.data import *
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)

def session():
 p=ROOT/'checkpoints/v14/bridge_seed11.pt'
 if not p.exists():pytest.skip('Train checkpoint before model-specific tests')
 return EnergyTextSession(p)

def test_memory_edits():
 s=ProtectedState.initial().edit(SET,3,7)
 assert s.feature(3)[:16].argmax()==7
 s=s.edit(SET,3,2);assert s.feature(3)[:16].argmax()==2
 s=s.edit(DELETE,3);assert torch.equal(s.precision,ProtectedState.initial().precision)
 assert torch.equal(s.information,ProtectedState.initial().information)

def test_hold_identity():
 s=ProtectedState.initial().edit(SET,2,9)
 for _ in range(8192):assert s.edit(HOLD,2,3) is s

def test_other_keys_unchanged():
 s=ProtectedState.initial().edit(SET,3,2);r=s.edit(SET,5,8)
 assert torch.equal(s.information[:5],r.information[:5]);assert torch.equal(s.precision[:5],r.precision[:5])

def test_memory_does_not_mutate_input():
 s=ProtectedState.initial();r=s.edit(SET,1,2)
 assert s.information.sum()==0 and r.information.sum()==1

def test_storage_payload():
 s=ProtectedState.initial();assert set(s.payload())=={'precision','information'}
 assert sum(x.numel()*x.element_size() for x in s.payload().values())==1088

def test_bad_state():
 with pytest.raises(ValueError):ProtectedState.restore({'precision':torch.zeros(16),'information':torch.zeros(16,16)})
 with pytest.raises(ValueError):ProtectedState.initial().edit(SET,16,0)

def test_parser_training_gradient():
 c=TextCodec.fit(vocabulary_texts());m=TextParser(len(c.vocabulary));out=m(c.encode(['Alice is in the garden.']))
 sum(x.square().mean() for x in out).backward()
 assert m.embedding.weight.grad.abs().sum()>0

def test_decoder_gradient():
 m=AnswerDecoder();f=torch.randn(2,17,requires_grad=True);m(f).square().mean().backward();assert f.grad.abs().sum()>0

def test_text_length_validation():
 c=TextCodec.fit(vocabulary_texts())
 with pytest.raises(ValueError):c.encode([''])
 with pytest.raises(ValueError):c.encode(['word '*41])

def test_split_disjoint():
 tr={r['text'] for r in dataset(1500,1,'train')};va={r['text'] for r in dataset(500,2,'validation')};te={r['text'] for r in dataset(500,3,'iid')}
 assert not tr.intersection(va|te) and not va.intersection(te)
 assert not set(sum(TRAIN_TEMPLATES.values(),[])).intersection(sum(HELDOUT_TEMPLATES.values(),[]))

def test_namespace_guard():
 assert not lexical_consistent('Zara is in the kitchen.',SET,0,1)
 assert not lexical_consistent('Alice and Bob are in the kitchen.',SET,0,1)
 assert lexical_consistent('Alice is in the kitchen.',SET,0,1)

def test_decoder_all_values():
 s=session()
 for i in range(17):
  state=ProtectedState.initial()
  if i<16:state=state.edit(SET,0,i)
  assert s.decoder.text(state.feature(0)[None])[0]==(VALUES[i] if i<16 else 'unknown')

def test_language_roundtrip():
 s=session();s.handle('Alice is in the garden.',trusted=True)
 assert s.handle('Where is Alice?')['answer']=='garden'
 s.handle('Correction Alice is now in the office.',trusted=True)
 assert s.handle('Where is Alice?')['answer']=='office'
 s.handle('Forget the location of Alice.',trusted=True)
 assert s.handle('Where is Alice?')['answer']=='unknown'

def test_untrusted_cannot_write():
 s=session();p=s.state.payload();result=s.handle('Alice is in the garden.',trusted=False)
 assert result['status']=='blocked_untrusted_write'
 assert all(torch.equal(v,s.state.payload()[k]) for k,v in p.items())

def test_default_is_non_writing():
 s=session();assert s.handle('Alice is in the garden.')['status']=='blocked_untrusted_write'

def test_questions_do_not_write_in_supported_grammar():
 s=session();old=s.state;s.handle('Where is Alice?');assert s.state is old

def test_batch_padding_parity():
 s=session();text='Where is Alice?';a=s.parser(s.codec.encode([text]));b=s.parser(s.codec.encode([text,'if bob were in the library it would be quiet.']))
 for x,y in zip(a,b):assert torch.allclose(x[0],y[0],atol=1e-5,rtol=1e-5)

def test_fresh_process_restart(tmp_path):
 s=session();s.handle('Alice is in the garden.',trusted=True);s.save(tmp_path/'state.pt')
 code='''from energy_transformer.bridge import EnergyTextSession
import sys,json,torch
torch.set_num_threads(1)
s=EnergyTextSession(sys.argv[1]);s.restore(sys.argv[2]);print(json.dumps(s.handle("Where is Alice?")))'''
 run=subprocess.run([sys.executable,'-c',code,str(ROOT/'checkpoints/v14/bridge_seed11.pt'),str(tmp_path/'state.pt')],cwd=ROOT,capture_output=True,text=True,check=True)
 assert json.loads(run.stdout)['answer']==s.handle('Where is Alice?')['answer']

def test_model_mismatch_rejected(tmp_path):
 s=session();s.save(tmp_path/'s.pt');d=torch.load(tmp_path/'s.pt',weights_only=True);d['model_sha256']='bad';torch.save(d,tmp_path/'s.pt')
 with pytest.raises(ValueError):s.restore(tmp_path/'s.pt')

def test_state_not_answer_cache():
 s=session();s.state=s.state.edit(SET,0,1);a=s.handle('Where is Alice?')['answer']
 s.state=s.state.edit(SET,0,2);b=s.handle('Where is Alice?')['answer']
 assert a=='kitchen' and b=='garden' and set(s.state.payload())=={'precision','information'}
