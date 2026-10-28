import json
import pytest
from rewt.bpe import StoryBPE,load_story_tokenizer


@pytest.fixture(scope='module')
def vocabulary():
    training=b' '.join(bytes([a,b]) for a in range(97,123) for b in range(97,123))
    return StoryBPE.learn(training)


def test_bpe_exact_inverse_unknown_words_and_ascii(vocabulary):
    text=bytes(range(128))+b"\nA new word, unfamiliar_789!  He's excited.\n"
    assert vocabulary.decode(vocabulary.encode(text))==text
    assert len(vocabulary.pieces)==256


def test_encoding_never_learns_from_new_text(vocabulary):
    before=vocabulary.identity()
    vocabulary.encode(b'A completely new word: qzx_914 and happy puppies.')
    assert vocabulary.identity()==before


def test_bpe_checkpoint_roundtrip_and_validation(vocabulary,tmp_path):
    path=tmp_path/'vocab.json';vocabulary.save(path,source={'split':'train'})
    loaded=load_story_tokenizer(path)
    text=b'ab cd qq unknown!'
    assert loaded.identity()==vocabulary.identity()
    assert loaded.encode(text)==vocabulary.encode(text)
    saved=json.loads(path.read_text());saved['merges'][0][2]=0
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError,match='Invalid byte-pair merge'):StoryBPE.load(path)


def test_bpe_rejects_non_ascii_and_insufficient_training(vocabulary):
    with pytest.raises(ValueError):vocabulary.encode(b'\xff')
    with pytest.raises(ValueError):StoryBPE.learn(b'aaa')
