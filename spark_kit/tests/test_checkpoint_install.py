from pathlib import Path
import zipfile
import pytest
from energy_spark.install_checkpoint import install, NAME, ROOT
import hashlib

@pytest.fixture
def sample_checkpoint(tmp_path, monkeypatch):
    import energy_spark.install_checkpoint as module
    source = tmp_path / NAME
    source.write_bytes(b"generated installer fixture; not original trained weights")
    monkeypatch.setattr(module, "EXPECTED", hashlib.sha256(source.read_bytes()).hexdigest())
    monkeypatch.setattr(module, "EXPECTED_BYTES", source.stat().st_size)
    return source



def test_generated_local_install_and_no_overwrite(tmp_path, sample_checkpoint):
    source = sample_checkpoint
    root = tmp_path / 'kit'
    assert install(source, root)['status'] == 'INSTALLED_VERIFIED_ORIGINAL'
    assert install(source, root)['status'] == 'ALREADY_VERIFIED'
    assert (root/'checkpoints'/NAME).read_bytes() == source.read_bytes()
    assert not list((root/'checkpoints').glob('*.tmp'))


def test_known_kit_extracts_only_expected_member(tmp_path, sample_checkpoint):
    source = sample_checkpoint
    zpath = tmp_path/'kit.zip'
    with zipfile.ZipFile(zpath, 'w') as archive:
        archive.write(source, 'kit/checkpoints/'+NAME)
        archive.writestr('../../never-extract-this.txt', 'not extracted')
    root = tmp_path/'output'
    assert install(zpath, root)['sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert list(root.rglob('*')) == [root/'checkpoints', root/'checkpoints'/NAME]


def test_wrong_original_is_rejected_without_install(tmp_path):
    source=tmp_path/'wrong.pt';source.write_bytes(b'not the trained model')
    root=tmp_path/'output'
    with pytest.raises(ValueError):install(source,root)
    assert not (root/'checkpoints'/NAME).exists()
    assert not list((root/'checkpoints').glob('*.tmp'))


def test_conflicting_destination_and_symlink_rejected(tmp_path, sample_checkpoint):
    source = sample_checkpoint
    target=tmp_path/'output/checkpoints'/NAME;target.parent.mkdir(parents=True)
    target.write_bytes(b'different')
    with pytest.raises(FileExistsError):install(source,tmp_path/'output')
    assert target.read_bytes()==b'different'
    target.unlink();target.symlink_to(source)
    with pytest.raises(ValueError):install(source,tmp_path/'output')
