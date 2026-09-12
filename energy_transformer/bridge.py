"""v14: supervised text-to-protected-energy bridge.

Finite-namespace, sentence-level research prototype. Sentence boundaries are
supplied; no entity IDs or operation labels are passed to the deployed parser.
The neural parser is stateless across records. P/S alone are protected memory.
The answer-byte head is trained separately; this is NOT general instruction tuning.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Sequence
import torch
from torch import Tensor, nn
from torch.nn import functional as F

NAMES = ('alice bob carol david emma frank grace henry iris jack kate liam maya noah olivia peter').split()
VALUES = ('attic kitchen garden office library garage basement balcony bedroom courtyard cellar lobby studio workshop pantry terrace').split()
HOLD, SET, DELETE, ASK = range(4)
OPERATIONS = ['hold', 'set', 'delete', 'ask']
MAX_WORDS = 40
MAX_ANSWER = 12

def words(text: str) -> list[str]:
    return re.findall(r"[a-z]+|[?.!]", text.lower())

def lexical_consistent(text: str, op: int, key: int, value: int) -> bool:
    """Reject out-of-namespace/multi-entity edits; validate, never infer, predicted IDs.

    This is a supplied lexical guard, not learned open-world entity resolution.
    """
    if op == HOLD: return True
    tokens=set(words(text))
    if NAMES[key] not in tokens or len(tokens.intersection(NAMES)) != 1: return False
    if op == SET and (VALUES[value] not in tokens or len(tokens.intersection(VALUES)) != 1): return False
    return True

class TextCodec:
    def __init__(self, vocabulary: Sequence[str]):
        self.vocabulary = list(vocabulary)
        if self.vocabulary[:2] != ['<pad>', '<unk>'] or len(set(self.vocabulary)) != len(self.vocabulary):
            raise ValueError('Invalid vocabulary')
        self.index = {x:i for i,x in enumerate(self.vocabulary)}

    @classmethod
    def fit(cls, texts: Sequence[str]):
        return cls(['<pad>','<unk>'] + sorted({w for t in texts for w in words(t)}))

    def encode(self, texts: Sequence[str]) -> Tensor:
        if not texts: raise ValueError('Nonempty batch required')
        encoded = []
        for t in texts:
            ws = words(t)
            if not ws or len(ws) > MAX_WORDS: raise ValueError('Empty or over-length text')
            encoded.append([self.index.get(w,1) for w in ws])
        out = torch.zeros(len(texts),max(7,max(map(len,encoded))),dtype=torch.long)
        for i,row in enumerate(encoded): out[i,:len(row)] = torch.tensor(row)
        return out

class TextParser(nn.Module):
    """Stateless, position-sensitive convolutional record encoder."""
    def __init__(self, vocab: int, width: int=40, channels: int=40):
        super().__init__()
        self.embedding = nn.Embedding(vocab,width,padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(width,channels,k,padding=k//2) for k in (3,5,7)])
        self.hidden = nn.Linear(6*channels,96)
        self.op = nn.Linear(96,4)
        self.key = nn.Linear(96,len(NAMES))
        self.value = nn.Linear(96,len(VALUES))

    def forward(self, tokens: Tensor) -> tuple[Tensor,Tensor,Tensor]:
        mask = tokens.ne(0)[:,None,:]
        e = self.embedding(tokens).transpose(1,2)
        feats = []
        for conv in self.convs:
            a = F.silu(conv(e))
            feats.extend([(a*mask).sum(-1)/mask.sum(-1).clamp_min(1),
                          a.masked_fill(~mask,-1e4).amax(-1)])
        h = F.silu(self.hidden(torch.cat(feats,-1)))
        return self.op(h), self.key(h), self.value(h)

class AnswerDecoder(nn.Module):
    """Bounded byte answer head, conditioned directly on the energy read.

All positions are predicted in parallel from transient memory features. No old
answer, autoregressive hidden state, or answer lookup table is used at inference.
It learns only the supplied location names and the word 'unknown'.
"""
    def __init__(self):
        super().__init__()
        self.position = nn.Embedding(MAX_ANSWER,16)
        self.net = nn.Sequential(nn.Linear(len(VALUES)+1+16,64),nn.SiLU(),nn.Linear(64,257))
    def forward(self, feature: Tensor) -> Tensor:
        b = feature.shape[0]
        pos = self.position(torch.arange(MAX_ANSWER,device=feature.device))
        x = torch.cat([feature[:,None,:].expand(b,MAX_ANSWER,-1),pos[None].expand(b,-1,-1)],-1)
        return self.net(x)
    @torch.inference_mode()
    def text(self, feature: Tensor) -> list[str]:
        answer = self(feature).argmax(-1).cpu().tolist()
        texts=[]
        for row in answer:
            if 256 in row: row=row[:row.index(256)]
            texts.append(bytes(row).decode('utf-8',errors='replace'))
        return texts

@dataclass(frozen=True)
class ProtectedState:
    precision: Tensor
    information: Tensor
    @classmethod
    def initial(cls):
        return cls(torch.full((len(NAMES),),.25),torch.zeros(len(NAMES),len(VALUES)))
    def edit(self, op: int, key: int, value: int=0):
        if op not in range(4) or not 0<=key<len(NAMES) or not 0<=value<len(VALUES):
            raise ValueError('Invalid edit')
        if op in (HOLD,ASK): return self
        p,s=self.precision.clone(),self.information.clone()
        p[key]=.25; s[key].zero_()
        if op==SET: p[key]+=1.;s[key,value]=1.
        return ProtectedState(p,s)
    def feature(self,key: int) -> Tensor:
        if not 0<=key<len(NAMES): raise ValueError('Unknown key')
        return torch.cat([self.information[key]/self.precision[key],
                          (self.precision[key]-.25).reshape(1)])
    def payload(self):
        return {'precision':self.precision.clone(),'information':self.information.clone()}
    @classmethod
    def restore(cls,d):
        if set(d)!={'precision','information'}: raise ValueError('Unexpected state fields')
        p,s=d['precision'],d['information']
        if p.dtype!=torch.float32 or s.dtype!=torch.float32 or p.shape!=(16,) or s.shape!=(16,16):
            raise ValueError('Bad state shape or dtype')
        if not torch.isfinite(p).all() or not torch.isfinite(s).all() or (p<.25).any():
            raise ValueError('Invalid energy coefficients')
        return cls(p.clone(),s.clone())

class EnergyTextSession:
    def __init__(self, checkpoint: str|Path, threshold: float|None=None):
        checkpoint=Path(checkpoint)
        d=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if d.get('format')!='energy_text_bridge_v14': raise ValueError('Unsupported checkpoint')
        self.codec=TextCodec(d['vocabulary'])
        self.parser=TextParser(len(self.codec.vocabulary));self.parser.load_state_dict(d['parser']);self.parser.eval()
        self.decoder=AnswerDecoder();self.decoder.load_state_dict(d['decoder']);self.decoder.eval()
        self.threshold=float(d['threshold'] if threshold is None else threshold)
        if not 0<=self.threshold<=1: raise ValueError('Threshold outside [0,1]')
        self.signature=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        self.state=ProtectedState.initial()

    @torch.inference_mode()
    def handle(self, text: str, *, trusted: bool=False) -> dict:
        try: tokens=self.codec.encode([text])
        except ValueError as e: return {'status':'abstained','reason':str(e)}
        outputs=self.parser(tokens)
        confidence=[float(x.softmax(-1).amax()) for x in outputs]
        op,key,val=[int(x.argmax(-1)) for x in outputs]
        relevant=[confidence[0]]
        if op!=HOLD: relevant.append(confidence[1])
        if op==SET: relevant.append(confidence[2])
        result={'operation':OPERATIONS[op],'confidence':min(relevant)}
        if not lexical_consistent(text,op,key,val):
            return {**result,'status':'abstained','reason':'namespace_or_binding_check'}
        if min(relevant)<self.threshold:
            return {**result,'status':'abstained'}
        if op in (SET,DELETE) and not trusted:
            return {**result,'status':'blocked_untrusted_write'}
        self.state=self.state.edit(op,key,val)
        if op==ASK:
            return {**result,'status':'answered','entity':NAMES[key],
                    'answer':self.decoder.text(self.state.feature(key)[None])[0]}
        return {**result,'status':'updated' if op in (SET,DELETE) else 'unchanged'}

    def save(self, path: str|Path):
        path=Path(path);temp=path.with_name(path.name+'.tmp')
        torch.save({'format':'energy_text_state_v14','model_sha256':self.signature,
                    'state':self.state.payload()},temp);temp.replace(path)
    def restore(self,path: str|Path):
        d=torch.load(path,map_location='cpu',weights_only=True)
        if set(d)!={'format','model_sha256','state'} or d['format']!='energy_text_state_v14' or d['model_sha256']!=self.signature:
            raise ValueError('Incompatible snapshot')
        self.state=ProtectedState.restore(d['state'])
