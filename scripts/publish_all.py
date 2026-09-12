#!/usr/bin/env python3
"""Publish the complete local payload with the user's existing Git authentication.

No token is read or embedded. No force push, visibility change, or license change.
Run with --push to authorize the final push; without it this prints the inventory.
"""
from pathlib import Path
import argparse,shutil,subprocess,tempfile
REPO='https://github.com/kenjugmail/EnergyTransformer.git'
ROOT=Path(__file__).resolve().parents[1]
EXCLUDED={'.git','__pycache__','.pytest_cache','.venv'}

def eligible(p,root):return not any(x in EXCLUDED for x in p.relative_to(root).parts)
def run(*args,cwd=None):return subprocess.run(args,cwd=cwd,check=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=ROOT);p.add_argument('--push',action='store_true');a=p.parse_args()
 source=a.source.resolve();files=[x for x in source.rglob('*') if x.is_file() and eligible(x,source)]
 if not files:raise SystemExit('Empty source')
 if any(x.name=='.env' for x in files):raise SystemExit('Refusing a payload containing .env')
 oversized=[str(x) for x in files if x.stat().st_size>=100*1024*1024]
 if oversized:raise SystemExit('Files require separate large-file handling: '+str(oversized))
 print(len(files),'files;',sum(x.stat().st_size for x in files),'bytes; target',REPO)
 if not a.push:print('Dry run. Add --push to copy, commit, and push.');return
 with tempfile.TemporaryDirectory(prefix='energy-publish-') as tmp:
  checkout=Path(tmp)/'repository';run('git','clone','--',REPO,str(checkout))
  run('git','checkout','main',cwd=checkout)
  for f in files:
   dest=checkout/f.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
  # Preserve remote-only files. The source is the explicit replacement payload.
  run('git','add','--all',cwd=checkout)
  changed=subprocess.run(['git','diff','--cached','--quiet'],cwd=checkout).returncode
  if changed==0:print('No changes to publish.');return
  if changed!=1:raise SystemExit('Unable to inspect staged changes')
  run('git','commit','-m','Publish complete v1-v14 energy-memory research artifacts',cwd=checkout)
  run('git','push','origin','HEAD:main',cwd=checkout)
  print('Complete payload pushed without force. Repository visibility unchanged.')
if __name__=='__main__':main()
