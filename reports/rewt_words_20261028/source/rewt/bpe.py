"""Small ASCII byte-pair vocabularies learned entirely from training text."""
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

CHUNKS=re.compile(rb' ?[A-Za-z]+| ?[0-9]+|[^\sA-Za-z0-9]+|\s+')


class StoryBPE:
    def __init__(self,pieces,merges):
        self.pieces=tuple(pieces)
        self.merges=tuple(tuple(map(int,m)) for m in merges)
        if (len(self.pieces)!=256 or len(set(self.pieces))!=256
                or self.pieces[:128]!=tuple(bytes([i]) for i in range(128))
                or any(not p or any(b>=128 for b in p) for p in self.pieces)):
            raise ValueError('Expected 128 ASCII bytes and 128 distinct merged pieces')
        for left,right,target in self.merges:
            if not all(0<=i<256 for i in (left,right,target)) or self.pieces[left]+self.pieces[right]!=self.pieces[target]:
                raise ValueError('Invalid byte-pair merge')

    @staticmethod
    def merge(sequence,pair,target):
        result=[];index=0
        while index<len(sequence):
            if index+1<len(sequence) and (sequence[index],sequence[index+1])==pair:
                result.append(target);index+=2
            else:result.append(sequence[index]);index+=1
        return tuple(result)

    @classmethod
    def learn(cls,raw):
        raw=bytes(raw)
        if any(b>=128 for b in raw):raise ValueError('Training text must be ASCII')
        counts=Counter(CHUNKS.findall(raw));sequences={tuple(word):count for word,count in counts.items()}
        pieces=[bytes([i]) for i in range(128)];ids={p:i for i,p in enumerate(pieces)};merges=[]
        while len(pieces)<256:
            pairs=Counter()
            for sequence,count in sequences.items():
                for pair in zip(sequence,sequence[1:]):pairs[pair]+=count
            if not pairs:raise ValueError('Training text cannot supply 128 distinct merged pieces')
            pair=min(pairs,key=lambda p:(-pairs[p],p))
            piece=pieces[pair[0]]+pieces[pair[1]]
            target=ids.get(piece)
            if target is None:
                target=len(pieces);ids[piece]=target;pieces.append(piece)
            merges.append((*pair,target));merged=Counter()
            for sequence,count in sequences.items():merged[cls.merge(sequence,pair,target)]+=count
            sequences=dict(merged)
        return cls(pieces,merges)

    @lru_cache(maxsize=65536)
    def encode_chunk(self,chunk):
        sequence=tuple(chunk)
        for left,right,target in self.merges:sequence=self.merge(sequence,(left,right),target)
        return bytes(sequence)

    def encode(self,raw):
        raw=raw.encode('ascii') if isinstance(raw,str) else bytes(raw)
        if any(b>=128 for b in raw):raise ValueError('This vocabulary accepts ASCII text')
        chunks=CHUNKS.findall(raw)
        assert b''.join(chunks)==raw
        return b''.join(self.encode_chunk(chunk) for chunk in chunks)

    def decode(self,tokens):return b''.join(self.pieces[int(token)] for token in tokens)

    @property
    def byte_lengths(self):return tuple(len(p) for p in self.pieces)

    def identity(self):
        return hashlib.sha256(json.dumps(dict(pieces=[p.hex() for p in self.pieces],merges=self.merges),sort_keys=True).encode()).hexdigest()

    def save(self,path,*,source):
        Path(path).write_text(json.dumps(dict(kind='ascii_bpe_v1',vocabulary_size=256,
            pieces_hex=[p.hex() for p in self.pieces],merges=self.merges,training_source=source,
            training_only_vocabulary=True),indent=2,sort_keys=True)+'\n')

    @classmethod
    def load(cls,path):
        saved=json.loads(Path(path).read_text())
        if saved['kind']!='ascii_bpe_v1':raise ValueError('Unknown byte-pair vocabulary')
        return cls([bytes.fromhex(p) for p in saved['pieces_hex']],saved['merges'])


def load_story_tokenizer(path):
    kind=json.loads(Path(path).read_text())['kind']
    if kind=='ascii_bpe_v1':return StoryBPE.load(path)
    if kind=='ascii_word_chunks_v1':
        from .tokenizer import StoryTokenizer
        return StoryTokenizer.load(path)
    if kind=='whole_words_v1':
        from .words import WordTokenizer
        return WordTokenizer.load(path)
    raise ValueError('Unknown story vocabulary')


def checkpoint_tokenizer(saved):
    """New word checkpoints embed their vocabulary; retain legacy path formats."""
    if saved.get('tokenizer_spec') is not None:
        from .words import WordTokenizer
        return WordTokenizer.from_dict(saved['tokenizer_spec'])
    info = saved.get('tokenizer')
    if not info:
        if saved.get('config', {}).get('vocab_size', 256) != 256:
            raise ValueError('A non-byte checkpoint must include its tokenizer')
        return None
    if isinstance(info, str):
        return load_story_tokenizer(info)
    from .data import file_sha256
    if file_sha256(info['path']) != info['sha256']:
        raise ValueError('The checkpoint vocabulary has changed')
    return load_story_tokenizer(info['path'])
