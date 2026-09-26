"""Read-only launch provenance audit; never creates a CUDA context or grants GPU acceptance."""
import json
from pathlib import Path
from energy_real.runtime import sha256
from .spec import DATA,MODEL,source_identity


def audit(root):
    root=Path(root);checks=[]
    def check(name,fn):
        try:
            detail=fn();checks.append({'name':name,'status':'passed','detail':detail})
        except (OSError,ValueError,KeyError,TypeError,AssertionError) as error:
            checks.append({'name':name,'status':'failed','error':str(error)})
    def source():
        spec=json.loads((root/'spec.json').read_text());receipt=json.loads((root/'cpu_verification.json').read_text())
        if spec['source_sha256']!=source_identity() or receipt['source_sha256']!=source_identity() or receipt['status']!='passed':
            raise ValueError('Frozen specification or CPU verification does not match current source')
        return source_identity()
    def corpus():
        path=root/'fineweb';m=json.loads((path/'manifest.json').read_text());source=m['source']
        if any(source[k]!=DATA[k] for k in ('id','revision','config')):raise ValueError('Wrong public corpus identity')
        groups=[];total=0
        for split in ('train','valid','test'):
            if sha256(path/(split+'.bin'))!=m['shards'][split]:raise ValueError(f'{split} shard changed')
            docs=m['documents'][split];offset=0
            if not docs:raise ValueError('Empty split')
            with (path/(split+'.bin')).open('rb') as data:
                for d in docs:
                    if d['offset']!=offset or d['bytes']<=0:raise ValueError('Invalid document offsets')
                    if __import__('hashlib').sha256(data.read(d['bytes'])).hexdigest()!=d['sha256']:raise ValueError('Document content changed')
                    offset+=d['bytes']
                if data.read(1):raise ValueError('Unindexed bytes')
            total+=offset;groups.append({d['group'] for d in docs})
        if any(a&b for i,a in enumerate(groups) for b in groups[i+1:]):raise ValueError('Duplicate group crosses splits')
        if total!=m['bytes'] or not 0<total<=DATA['bytes']:raise ValueError('Corpus byte budget mismatch')
        return {'manifest_sha256':sha256(path/'manifest.json'),'bytes':total}
    def donor():
        m=json.loads((root/'donor_manifest.json').read_text())
        if m['model']!=MODEL:raise ValueError('Donor revision/architecture manifest mismatch')
        required={'config.json','model.safetensors','tokenizer.json','tokenizer_config.json'}
        if not required.issubset(m['files']):raise ValueError('Incomplete donor manifest')
        for name,record in m['files'].items():
            if Path(name).name!=name:raise ValueError('Invalid donor file name')
            p=root/'donor'/name
            if p.stat().st_size!=record['bytes'] or sha256(p)!=record['sha256']:raise ValueError(f'Donor changed: {name}')
        c=json.loads((root/'donor/config.json').read_text())
        expected={'num_hidden_layers':30,'hidden_size':576,'num_attention_heads':9,'num_key_value_heads':3,'max_position_embeddings':8192}
        if any(c.get(k)!=v for k,v in expected.items()):raise ValueError('Donor configuration differs')
        return {'revision':MODEL['revision'],'files':len(m['files'])}
    check('source_and_cpu_receipt',source);check('document_integrity_and_split_groups',corpus);check('donor_integrity',donor)
    return {'status':'passed' if all(c['status']=='passed' for c in checks) else 'failed','checks':checks,
            'gpu_acceptance':False,'scope':'CPU preparation integrity only; exclusive GPU and numerical acceptance still required'}
