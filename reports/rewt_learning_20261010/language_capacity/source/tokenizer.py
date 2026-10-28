"""A reversible 256-piece ASCII vocabulary learned from training text only."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re


class StoryTokenizer:
    def __init__(self, pieces):
        self.pieces=tuple(pieces)
        if (len(self.pieces)!=256 or self.pieces[:128]!=tuple(bytes([i]) for i in range(128))
                or len(set(self.pieces))!=256
                or any(not p or any(b>=128 for b in p) for p in self.pieces)):
            raise ValueError('Expected 128 ASCII bytes and 128 distinct ASCII chunks')
        self.by_initial={i:[] for i in range(128)}
        for token,piece in enumerate(self.pieces[128:],128):
            self.by_initial[piece[0]].append((piece,token))
        for values in self.by_initial.values():values.sort(key=lambda item:(-len(item[0]),item[0]))

    @classmethod
    def learn(cls, raw):
        if any(b>=128 for b in raw):raise ValueError('Training text must be ASCII')
        counts=Counter(re.findall(rb' ?[A-Za-z]+',raw))
        ranked=sorted(((n,p) for p,n in counts.items() if len(p)>1),key=lambda item:(-item[0],item[1]))
        if len(ranked)<128:raise ValueError('Training text needs 128 distinct multi-byte chunks')
        return cls([bytes([i]) for i in range(128)]+[p for _,p in ranked[:128]])

    @classmethod
    def load(cls,path):
        saved=json.loads(Path(path).read_text())
        if saved['kind']!='ascii_word_chunks_v1':raise ValueError('Unknown tokenizer')
        return cls([bytes.fromhex(p) for p in saved['pieces_hex']])

    def save(self,path,*,source):
        Path(path).write_text(json.dumps(dict(kind='ascii_word_chunks_v1',vocabulary_size=256,
            pieces_hex=[p.hex() for p in self.pieces],training_source=source,
            training_only_vocabulary=True),indent=2,sort_keys=True)+'\n')

    def encode(self,raw):
        raw=raw.encode('ascii') if isinstance(raw,str) else bytes(raw)
        if any(b>=128 for b in raw):raise ValueError('This vocabulary accepts ASCII text')
        encoded=[];position=0
        while position<len(raw):
            match=next(((p,t) for p,t in self.by_initial[raw[position]] if raw.startswith(p,position)),None)
            if match is None:
                encoded.append(raw[position]);position+=1
            else:
                piece,token=match;encoded.append(token);position+=len(piece)
        return bytes(encoded)

    def decode(self,tokens):
        return b''.join(self.pieces[int(token)] for token in tokens)

    @property
    def byte_lengths(self):
        return tuple(len(p) for p in self.pieces)

    def identity(self):
        return hashlib.sha256(json.dumps([p.hex() for p in self.pieces]).encode()).hexdigest()
