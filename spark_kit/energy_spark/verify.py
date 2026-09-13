"""Verify the distributed kit before execution. Hashes are not digital signatures."""
import argparse,json
from pathlib import Path
from .common import ROOT,sha256

def verify(root=ROOT):
    root=Path(root);manifest=root/'BUNDLE_MANIFEST.json'
    if not manifest.exists():manifest=root/'SOURCE_MANIFEST.json'
    m=json.loads(manifest.read_text());bad=[]
    for rel,digest in m['files'].items():
        path=Path(rel)
        if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe manifest path')
        p=root/path
        if not p.is_file() or sha256(p)!=digest:bad.append(rel)
    if bad:raise ValueError('Changed or absent files: '+', '.join(bad))
    return {'checked':len(m['files']),'manifest':manifest.name,'status':'PASS','scope':'Integrity against included manifest; no external signature or hardware-validation claim'}

if __name__=='__main__':print(json.dumps(verify(),indent=2))
