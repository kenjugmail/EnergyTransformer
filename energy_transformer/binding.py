"""Stateless factorized text proposals. NEVER writes or authorizes memory.

Registered names/values and one-entity grammar are supplied, not learned NER.
The operation is learned; the key/value are explicit lexical bindings. An optional
unknown-word guard is an abstention policy, not a correctness certificate.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .bridge import NAMES, VALUES, HOLD, SET, DELETE, ASK, OPERATIONS, TextCodec, TextParser, words, MAX_WORDS

@dataclass(frozen=True)
class Binding:
    canonical: str
    key: int | None
    value: int | None
    reason: str | None


def bind(text: str, names: Sequence[str]=NAMES, values: Sequence[str]=VALUES) -> Binding:
    if not isinstance(text,str):raise TypeError('text must be a string')
    if len(names)!=16 or len(values)!=16 or len(set(names)|set(values))!=32:
        raise ValueError('Expected disjoint registries of 16 names and 16 values')
    if any(words(s)!=[s] for s in [*names,*values]):
        raise ValueError('Registry entries must be unique lowercase word tokens')
    ws=words(text)
    if not ws or len(ws)>MAX_WORDS:return Binding('',None,None,'empty_or_overlength')
    # Count mentions, not just distinct identities: repeated or multi-clause inputs
    # are outside this deliberately bounded single-entity interface.
    nk=[names.index(w) for w in ws if w in names]
    nv=[values.index(w) for w in ws if w in values]
    if len(nk)>1 or len(nv)>1:return Binding('',None,None,'ambiguous_multiple_mentions')
    canonical=' '.join('entitytoken' if w in names else 'valuetoken' if w in values else w for w in ws)
    return Binding(canonical,nk[0] if nk else None,nv[0] if nv else None,None)

class OperationParser(nn.Module):
    """Same convolutional encoder as the v14 parser; only semantic operation is learned."""
    def __init__(self,vocab:int):
        super().__init__()
        self.embedding=nn.Embedding(vocab,40,padding_idx=0)
        self.convs=nn.ModuleList([nn.Conv1d(40,40,k,padding=k//2) for k in (3,5,7)])
        self.hidden=nn.Linear(240,96)
        self.op=nn.Linear(96,4)
    def forward(self,tokens:Tensor):
        mask=tokens.ne(0)[:,None,:]; e=self.embedding(tokens).transpose(1,2)
        feats=[]
        for conv in self.convs:
            a=F.silu(conv(e))
            feats.extend([(a*mask).sum(-1)/mask.sum(-1).clamp_min(1),a.masked_fill(~mask,-1e4).amax(-1)])
        return self.op(F.silu(self.hidden(torch.cat(feats,-1))))

class BindingPredictor:
    def __init__(self,checkpoint:str|Path,*,names:Sequence[str]=NAMES,values:Sequence[str]=VALUES):
        d=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if d.get('format')!='binding_proposer_v17' or d.get('kind')!='bound':raise ValueError('Expected a v17 bound proposer')
        self.codec=TextCodec(d['vocabulary']); self.model=OperationParser(len(self.codec.vocabulary))
        self.model.load_state_dict(d['weights']);self.model.eval()
        self.names=tuple(names);self.values=tuple(values);bind('test',self.names,self.values)
        self.threshold=float(d.get('threshold',.90))
        if not 0<=self.threshold<=1:raise ValueError('Invalid threshold')

    @torch.inference_mode()
    def predict(self,text:str,*,threshold:float|None=None,reject_unknown:bool=True)->dict:
        b=bind(text,self.names,self.values)
        if b.reason:return {'status':'abstained','reason':b.reason}
        unknown=sorted({w for w in words(b.canonical) if w not in self.codec.index})
        if reject_unknown and unknown:
            return {'status':'abstained','reason':'unknown_vocabulary','unknown':unknown}
        logits=self.model(self.codec.encode([b.canonical])); prob=logits.softmax(-1)
        op=int(prob.argmax(-1));confidence=float(prob.max())
        cutoff=self.threshold if threshold is None else float(threshold)
        if not 0<=cutoff<=1:raise ValueError('Invalid threshold')
        if confidence<cutoff:return {'status':'abstained','reason':'low_score','confidence':confidence}
        if op!=HOLD and b.key is None:return {'status':'abstained','reason':'missing_registered_entity'}
        if op==SET and b.value is None:return {'status':'abstained','reason':'missing_registered_value'}
        if op in (ASK,DELETE) and b.value is not None:
            return {'status':'abstained','reason':'unexpected_value_for_operation'}
        return {'status':'proposal_only' if op in (SET,DELETE) else 'parsed',
                'operation':OPERATIONS[op], 'entity':self.names[b.key] if b.key is not None else None,
                'value':self.values[b.value] if op==SET else None, 'confidence':confidence,
                'requires_review':op in (SET,DELETE)}
