"""Train-only, whole-word tokenization with no byte or subword fallback.

Case is retained. Words, numbers and contractions are indivisible; punctuation
has separate tokens. Whitespace is normalized on decoding. Unseen words map to
one <unk>, so this is a closed vocabulary rather than a lossless byte codec.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


TOKEN = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*|[^\w\s]|_", re.UNICODE)
SPECIAL = ('<unk>', '<bos>', '<eos>')


class WordTokenizer:
    kind = 'whole_words_v1'
    unk_id, bos_id, eos_id = range(3)

    def __init__(self, vocabulary):
        self.vocabulary = tuple(vocabulary)
        if (len(self.vocabulary) < 4 or self.vocabulary[:3] != SPECIAL
                or len(set(self.vocabulary)) != len(self.vocabulary)
                or any(not isinstance(w, str) or not TOKEN.fullmatch(w)
                       for w in self.vocabulary[3:])):
            raise ValueError('Invalid whole-word vocabulary')
        self.ids = {word: i for i, word in enumerate(self.vocabulary)}
        self.pieces = tuple(word.encode('utf-8') for word in self.vocabulary)
        self.word_ids = tuple(i for i, word in enumerate(self.vocabulary)
                              if i >= 3 and self.is_word(word))

    @staticmethod
    def split(text):
        if isinstance(text, bytes):
            text = text.decode('utf-8')
        return TOKEN.findall(text)

    @staticmethod
    def is_word(piece):
        return bool(piece) and piece[0].isalnum()

    @classmethod
    def learn(cls, documents, *, vocab_size=8192, min_count=2):
        if type(vocab_size) is not int or vocab_size < 4 or type(min_count) is not int or min_count < 1:
            raise ValueError('vocab_size >= 4 and min_count >= 1 required')
        if isinstance(documents, (str, bytes)):
            documents = [documents]
        counts = Counter()
        for document in documents:
            counts.update(cls.split(document))
        # Keep all observed punctuation before applying the word-frequency cap.
        ranked = sorted((w for w, n in counts.items() if n >= min_count or not cls.is_word(w)),
                        key=lambda w: (cls.is_word(w), -counts[w], w))
        return cls([*SPECIAL, *ranked[:vocab_size - 3]])

    def encode(self, text, *, bos=False, eos=False):
        tokens = [self.ids.get(word, self.unk_id) for word in self.split(text)]
        return ([self.bos_id] if bos else []) + tokens + ([self.eos_id] if eos else [])

    def decode_continuation(self, tokens, prefix=''):
        """Return a readable suffix, including its separator from a given prefix."""
        if isinstance(prefix, bytes):
            prefix = prefix.decode('utf-8')
        text = prefix
        for token in tokens:
            token = int(token)
            if not 0 <= token < len(self.vocabulary):
                raise ValueError('Token ID outside the word vocabulary')
            if token == self.eos_id:
                break
            if token == self.bos_id:
                continue
            word = self.vocabulary[token]
            attached = word in '.,!?;:%)]}' or (text and text[-1] in '([{')
            separator = '' if not text or text[-1].isspace() or attached else ' '
            text += separator + word
        return text[len(prefix):].encode('utf-8')

    def decode(self, tokens):
        return self.decode_continuation(tokens)

    def to_dict(self):
        return dict(kind=self.kind, vocabulary_size=len(self.vocabulary),
                    vocabulary=list(self.vocabulary), training_only_vocabulary=True,
                    unknown_policy='one <unk> per unseen word; no byte fallback',
                    whitespace='normalized', punctuation='separate tokens')

    @classmethod
    def from_dict(cls, saved):
        if saved.get('kind') != cls.kind:
            raise ValueError('Unknown word tokenizer')
        result = cls(saved['vocabulary'])
        if saved.get('vocabulary_size') != len(result.vocabulary):
            raise ValueError('Word vocabulary size mismatch')
        return result

    def save(self, path, *, source=None):
        saved = self.to_dict()
        saved['training_source'] = source
        Path(path).write_text(json.dumps(saved, ensure_ascii=False, indent=2) + '\n')

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def identity(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True,
                                        ensure_ascii=False).encode()).hexdigest()


def next_words(logits, tokenizer, count=10):
    """Rank complete words at the next position; retain original probabilities.

    This excludes punctuation/special tokens from the displayed candidates; it
    does not condition away their probability mass or skip a predicted token.
    """
    import torch
    if not isinstance(tokenizer, WordTokenizer) or logits.ndim != 1:
        raise ValueError('A word tokenizer and one next-token logit vector required')
    if len(logits) != len(tokenizer.vocabulary) or type(count) is not int or count < 1:
        raise ValueError('Invalid vocabulary size or candidate count')
    ids = torch.tensor(tokenizer.word_ids, dtype=torch.long, device=logits.device)
    probabilities = logits.float().softmax(-1)
    values, positions = probabilities[ids].topk(min(count, len(ids)))
    return [dict(word=tokenizer.vocabulary[int(ids[p])], token_id=int(ids[p]), probability=float(v))
            for v, p in zip(values, positions)]
