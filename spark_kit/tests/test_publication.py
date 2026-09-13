import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import zipfile
import pytest
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('publish_release', ROOT/'publish_release.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

@pytest.fixture
def archive(tmp_path):
    p = tmp_path/'research.zip'
    with zipfile.ZipFile(p,'w') as z:
        z.writestr('EnergyTransformer/README.md','research')
    return p

def test_dry_run_has_no_network(archive):
    def forbidden(*args, **kwargs):
        raise AssertionError('Unexpected external call')
    assert not m.upload_archive(archive, runner=forbidden)['published']

@pytest.mark.parametrize('tag',['--help','x;rm -rf','../bad','a b'])
def test_bad_release_tag(archive,tag):
    with pytest.raises(ValueError):
        m.upload_archive(archive,tag=tag)

def test_zip_path_escape_rejected(tmp_path):
    p = tmp_path/'bad.zip'
    with zipfile.ZipFile(p,'w') as z: z.writestr('../escape','x')
    with pytest.raises(ValueError): m.check_archive(p)

def test_non_404_is_not_missing_release():
    def denied(*args,**kwargs):
        return subprocess.CompletedProcess(args,1,'','HTTP 403: denied')
    with pytest.raises(RuntimeError): m.release_metadata('test', denied)

def test_publish_simulated_and_resumable(archive,tmp_path):
    state = {'metadata':None, 'assets':{}, 'calls':[]}
    def fake(*args,check=True):
        state['calls'].append(args)
        out=''
        if args[:3] == ('gh','auth','status'):
            pass
        elif args[:2] == ('gh','api'):
            if args[2] == f'repos/{m.REPO}':
                out=json.dumps({'full_name':m.REPO,'private':True})
            elif state['metadata'] is None:
                return subprocess.CompletedProcess(args,1,'','HTTP 404: Not Found')
            else:
                state['metadata']['assets']=[{'name':n} for n in state['assets']]
                out=json.dumps(state['metadata'])
        elif args[:3] == ('gh','release','create'):
            state['metadata']={'body':m.MARKER,'draft':True,'html_url':'https://github.com/kenjugmail/EnergyTransformer/releases/tag/test'}
        elif args[:3] == ('gh','release','upload'):
            p=Path(args[4]);state['assets'][p.name]=p.read_bytes()
        elif args[:3] == ('gh','release','download'):
            name=args[args.index('--pattern')+1]; dest=Path(args[args.index('--output')+1])
            dest.write_bytes(state['assets'][name])
        elif args[:3] == ('gh','release','edit'):
            state['metadata']['draft']=False
        else: raise AssertionError(args)
        return subprocess.CompletedProcess(args,0,out,'')
    assert m.upload_archive(archive,publish=True,tag='test',runner=fake)['published']
    before=sum(c[:3]==('gh','release','upload') for c in state['calls'])
    assert before==2
    assert m.upload_archive(archive,publish=True,tag='test',runner=fake)['published']
    assert sum(c[:3]==('gh','release','upload') for c in state['calls'])==2
    assert not any('--clobber' in c or '--force' in c for c in state['calls'])
    state['assets'][archive.name]=b'corrupt'
    with pytest.raises(RuntimeError): m.upload_archive(archive,publish=True,tag='test',runner=fake)

def test_public_repository_refused(archive):
    def fake(*args,**kwargs):
        out=json.dumps({'full_name':m.REPO,'private':False}) if args[:2]==('gh','api') else ''
        return subprocess.CompletedProcess(args,0,out,'')
    with pytest.raises(RuntimeError,match='PRIVATE'):
        m.upload_archive(archive,publish=True,runner=fake)
