from pathlib import Path
import zipfile
import pytest
from energy_spark.install_checkpoint import install, NAME, EXPECTED, ROOT


def test_original_local_install_and_no_overwrite(tmp_path):
    source = ROOT / 'checkpoints' / NAME
    root = tmp_path / 'kit'
    assert install(source, root)['status'] == 'INSTALLED_VERIFIED_ORIGINAL'
    assert install(source, root)['status'] == 'ALREADY_VERIFIED'
    assert (root/'checkpoints'/NAME).read_bytes() == source.read_bytes()
    assert not list((root/'checkpoints').glob('*.tmp'))


def test_known_kit_extracts_only_expected_member(tmp_path):
    source = ROOT/'checkpoints'/NAME
    zpath = tmp_path/'kit.zip'
    with zipfile.ZipFile(zpath, 'w') as archive:
        archive.write(source, 'kit/checkpoints/'+NAME)
        archive.writestr('../../never-extract-this.txt', 'not extracted')
    root = tmp_path/'output'
    assert install(zpath, root)['sha256'] == EXPECTED
    assert list(root.rglob('*')) == [root/'checkpoints', root/'checkpoints'/NAME]


def test_wrong_original_is_rejected_without_install(tmp_path):
    source=tmp_path/'wrong.pt';source.write_bytes(b'not the trained model')
    root=tmp_path/'output'
    with pytest.raises(ValueError):install(source,root)
    assert not (root/'checkpoints'/NAME).exists()
    assert not list((root/'checkpoints').glob('*.tmp'))


def test_conflicting_destination_and_symlink_rejected(tmp_path):
    source=ROOT/'checkpoints'/NAME
    target=tmp_path/'output/checkpoints'/NAME;target.parent.mkdir(parents=True)
    target.write_bytes(b'different')
    with pytest.raises(FileExistsError):install(source,tmp_path/'output')
    assert target.read_bytes()==b'different'
    target.unlink();target.symlink_to(source)
    with pytest.raises(ValueError):install(source,tmp_path/'output')
