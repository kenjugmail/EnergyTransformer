from pathlib import Path
from types import SimpleNamespace
import shutil
import sys
import zipfile
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from publish_hf import publish_archive

class Backend:
    def __init__(self,private=True): self.private=private;self.files={};self.commits=0;self.sha='old'
    def repo_info(self,**kw):return SimpleNamespace(private=self.private,id='test/model',sha=self.sha)
    def list_repo_files(self,**kw):return list(self.files)
    def create_commit(self,**kw):
        assert kw['parent_commit']==self.sha
        for op in kw['operations']:self.files[op.path_in_repo]=Path(op.path_or_fileobj).read_bytes()
        self.commits+=1;self.sha='new';return SimpleNamespace(oid=self.sha)
    def download(self,**kw):
        path=Path(kw['local_dir'])/kw['filename'];path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(self.files[kw['filename']]);return str(path)

def archive(tmp_path):
    p=tmp_path/'example.zip'
    with zipfile.ZipFile(p,'w') as z:z.writestr('README.md','research example')
    return p

def test_hf_dry_run_no_connection(tmp_path):
    assert not publish_archive(archive(tmp_path),'test/model')['verified']

def test_hf_private_roundtrip_and_retry(tmp_path):
    p=archive(tmp_path);api=Backend()
    kw=dict(publish=True,api=api,download=api.download,add_operation=lambda **k:SimpleNamespace(**k))
    assert publish_archive(p,'test/model',**kw)['verified']
    assert publish_archive(p,'test/model',**kw)['verified']
    assert api.commits==1

def test_hf_refuses_public(tmp_path):
    api=Backend(private=False)
    with pytest.raises(RuntimeError,match='private'):
        publish_archive(archive(tmp_path),'test/model',publish=True,api=api)
    assert api.commits==0

def test_hf_refuses_existing_mismatch(tmp_path):
    p=archive(tmp_path);api=Backend()
    kw=dict(publish=True,api=api,download=api.download,add_operation=lambda **k:SimpleNamespace(**k))
    receipt=publish_archive(p,'test/model',**kw);api.files[receipt['path']]=b'wrong'
    with pytest.raises(RuntimeError,match='overwrite'):publish_archive(p,'test/model',**kw)

def test_hf_download_corruption_fails(tmp_path):
    p=archive(tmp_path);api=Backend()
    def bad_download(**kw):
        target=Path(api.download(**kw));target.write_bytes(b'bad');return str(target)
    with pytest.raises(RuntimeError,match='verification'):
        publish_archive(p,'test/model',publish=True,api=api,download=bad_download,add_operation=lambda **k:SimpleNamespace(**k))
