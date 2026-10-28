"""Prepare whole-word stories without learning vocabulary from held-out text."""
import json
from pathlib import Path

import numpy as np

from .data import file_sha256
from .experiment import save_json
from .story_data import StoryCorpus
from .words import WordTokenizer


class WordCorpus(StoryCorpus):
    def __init__(self, directory):
        super().__init__(directory)
        if self.manifest.get('tokenization') != WordTokenizer.kind or self.dtype != np.dtype('<u4'):
            raise ValueError('Expected a whole-word corpus')
        info = self.manifest['tokenizer']
        path = self.directory / info['filename']
        if file_sha256(path) != info['sha256']:
            raise ValueError('Word vocabulary changed')
        self.tokenizer = WordTokenizer.load(path)
        self.word_masks = {}
        for split, info in self.manifest['splits'].items():
            path = Path(info['word_mask_path'])
            if file_sha256(path) != info['word_mask_sha256'] or path.stat().st_size != len(self.data[split]):
                raise ValueError('Word-position mask changed')
            self.word_masks[split] = np.memmap(path, dtype=np.uint8, mode='r')


def prepare_word_corpus(source, output, *, vocab_size=8192, min_count=2):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a new word-corpus directory')
    original = StoryCorpus(source)
    if original.dtype != np.dtype('uint8') or original.manifest.get('vocabulary'):
        raise ValueError('Source must contain raw UTF-8 stories, not encoded token IDs')

    def documents(split):
        for start, end in original.documents[split]:
            yield bytes(original.data[split][start:end]).decode('utf-8')

    vocabulary = WordTokenizer.learn(documents('train'), vocab_size=vocab_size, min_count=min_count)
    output.mkdir(parents=True)
    vocabulary.save(output / 'vocabulary.json', source=original.identity())
    splits = {}
    for split in ('train', 'valid', 'test'):
        source_records = json.loads(Path(original.manifest['splits'][split]['index_path']).read_text())
        path, index, mask_path = (output / f'{split}{suffix}' for suffix in ('.bin', '_documents.json', '_words.bin'))
        records = []
        position = total_words = unknown_words = unknown_tokens = 0
        with path.open('wb') as stream, mask_path.open('wb') as masks:
            for text, source_record in zip(documents(split), source_records):
                pieces = vocabulary.split(text)
                tokens = [vocabulary.bos_id, *(vocabulary.ids.get(w, vocabulary.unk_id) for w in pieces), vocabulary.eos_id]
                word_mask = [0, *(int(vocabulary.is_word(w)) for w in pieces), 0]
                np.asarray(tokens, dtype='<u4').tofile(stream)
                np.asarray(word_mask, dtype=np.uint8).tofile(masks)
                records.append(dict(id=source_record['id'], start=position, end=position + len(tokens)))
                position += len(tokens)
                total_words += sum(word_mask)
                unknown_words += sum(word and token == vocabulary.unk_id for word, token in zip(word_mask, tokens))
                unknown_tokens += tokens.count(vocabulary.unk_id)
        save_json(index, records)
        splits[split] = dict(path=str(path), index_path=str(index), word_mask_path=str(mask_path),
                             sha256=file_sha256(path), index_sha256=file_sha256(index),
                             word_mask_sha256=file_sha256(mask_path), documents=len(records),
                             tokens=position, words=total_words, unknown_words=unknown_words,
                             unknown_tokens=unknown_tokens, unknown_word_rate=unknown_words / max(1, total_words))
    manifest = dict(dataset=original.manifest['dataset'], revision=original.manifest['revision'],
                    license=original.manifest.get('license'), dtype='<u4',
                    tokenization=WordTokenizer.kind, splits=splits,
                    tokenizer=dict(filename='vocabulary.json', sha256=file_sha256(output / 'vocabulary.json'),
                                   identity=vocabulary.identity(), vocabulary_size=len(vocabulary.vocabulary)),
                    source=original.identity(), min_count=min_count, requested_vocab_size=vocab_size,
                    protocol='Preserve source document splits. Vocabulary fitted only on train. '
                             'BOS/EOS per document. No byte/subword fallback. No cross-document crops.')
    save_json(output / 'manifest.json', manifest)
    return WordCorpus(output)
