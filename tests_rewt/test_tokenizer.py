import pytest

from rewt.tokenizer import StoryTokenizer


def tokenizer():
    raw=b' '.join(('word'+chr(65+i//26)+chr(65+i%26)).encode() for i in range(130))
    return StoryTokenizer.learn(raw)


def test_vocabulary_is_reversible_for_every_ascii_byte_and_unknown_words(tmp_path):
    model=tokenizer()
    raw=bytes(range(128))+b' unknown vocabulary wordAB and A=29;B=A+17;B=?\n'
    assert model.decode(model.encode(raw))==raw
    path=tmp_path/'vocabulary.json';model.save(path,source={'split':'train'})
    restored=StoryTokenizer.load(path)
    assert restored.identity()==model.identity()
    assert restored.encode(raw)==model.encode(raw)


def test_vocabulary_does_not_change_when_encoding_validation_or_test_text():
    model=tokenizer();before=model.identity()
    model.encode(b'This rareword never appeared in training.')
    assert model.identity()==before
    assert len(model.pieces)==256


def test_non_ascii_input_is_rejected():
    with pytest.raises(ValueError):tokenizer().encode(bytes([255]))
