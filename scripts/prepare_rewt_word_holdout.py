"""Prepare an unused official TinyStories range with the existing word vocabulary."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from rewt.data import file_sha256
from rewt.experiment import save_json
from rewt.word_data import WordCorpus

URL = 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/TinyStories-valid.txt'
START, END = 16777216, 19447281


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, default=ROOT/'data/rewt_words_20261028')
    p.add_argument('--output', type=Path, default=ROOT/'data/rewt_word_holdout_20261029')
    a = p.parse_args(); out = a.output.resolve()
    if out.exists(): raise FileExistsError('Use a new holdout directory')
    base = WordCorpus(a.base); tokenizer = base.tokenizer
    excluded = set(); indexes = {}
    for path in sorted((ROOT/'data').rglob('*documents.json')):
        records = json.loads(path.read_text())
        if not isinstance(records, list): continue
        excluded.update(row['id'] for row in records if 'id' in row)
        indexes[str(path.relative_to(ROOT))] = file_sha256(path)
    request = urllib.request.Request(URL+'?download=true&word_holdout=20261029',
                                     headers={'Range':f'bytes={START}-{END}'})
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read(END-START+2)
        content_range = response.headers.get('Content-Range')
        if response.status != 206 or content_range != f'bytes {START}-{END}/{END+1}' or len(raw) != END-START+1:
            raise ValueError(f'Unexpected source range: {response.status}, {content_range}, {len(raw)}')
    out.mkdir(parents=True); (out/'source.txt').write_bytes(raw)
    save_json(out/'source.json', dict(url=URL, offset=START, bytes=len(raw), content_range=content_range,
                                    sha256=hashlib.sha256(raw).hexdigest(), excluded_indexes=indexes))
    shutil.copy2(a.base/'vocabulary.json', out/'vocabulary.json')
    records = []; position = words = unknown_words = unknown_tokens = duplicates = 0
    seen = set(excluded)
    data, mask, index = out/'test.bin', out/'test_words.bin', out/'test_documents.json'
    with data.open('wb') as stream, mask.open('wb') as masks:
        for text in raw.decode('utf-8', errors='ignore').split('<|endoftext|>')[1:-1]:
            normalized = text.strip().replace('\r','').translate(str.maketrans(
                {'“':'"','”':'"','‘':"'",'’':"'",'—':'-','–':'-','\u00a0':' '}))
            encoded = normalized.encode('ascii', errors='replace')
            identity = hashlib.sha256(encoded).hexdigest()
            if identity in seen:
                duplicates += 1; continue
            if len(encoded) < 200: continue
            seen.add(identity)
            pieces = tokenizer.split(encoded.decode('ascii'))
            ids = [tokenizer.bos_id, *(tokenizer.ids.get(w, tokenizer.unk_id) for w in pieces), tokenizer.eos_id]
            word_mask = [0, *(int(tokenizer.is_word(w)) for w in pieces), 0]
            np.asarray(ids, dtype='<u4').tofile(stream); np.asarray(word_mask, dtype=np.uint8).tofile(masks)
            records.append(dict(id=identity, start=position, end=position+len(ids))); position += len(ids)
            words += sum(word_mask); unknown_tokens += ids.count(tokenizer.unk_id)
            unknown_words += sum(w and t == tokenizer.unk_id for w,t in zip(word_mask,ids))
    save_json(index, records)
    manifest = copy.deepcopy(base.manifest)
    manifest['base_corpus_manifest_sha256'] = base.manifest_sha256
    manifest['splits']['test'] = dict(path=str(data), index_path=str(index), word_mask_path=str(mask),
        sha256=file_sha256(data), index_sha256=file_sha256(index), word_mask_sha256=file_sha256(mask),
        documents=len(records), tokens=position, words=words, unknown_words=unknown_words,
        unknown_tokens=unknown_tokens, unknown_word_rate=unknown_words/max(1,words))
    manifest['holdout_source'] = dict(url=URL, start=START, end=END, sha256=hashlib.sha256(raw).hexdigest(),
                                     excluded_unique_ids=len(excluded), duplicates_skipped=duplicates)
    manifest['protocol'] = ('Training and validation reused unchanged. Test consists of complete documents in the unused '
                            'official validation byte range 16MiB..end; all previously indexed IDs excluded. '
                            'Vocabulary reused exactly, without refitting. Test labels are for final evaluation only.')
    save_json(out/'manifest.json', manifest)
    corpus = WordCorpus(out)
    offsets = corpus.offsets('test', 1024, 64, 291902)
    save_json(out/'evaluation_offsets.json', offsets)
    print(json.dumps(dict(documents=len(records), tokens=position, unknown_word_rate=unknown_words/words,
                          duplicates_skipped=duplicates, evaluation_documents=len(offsets),
                          manifest_sha256=corpus.manifest_sha256)))


if __name__ == '__main__': main()
