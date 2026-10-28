"""Add verified tokenizer metadata without replacing frozen training checkpoints."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from rewt.data import file_sha256
from rewt.experiment import save_json,save_checkpoint


def main():
    out=ROOT/'reports/rewt_architecture_20261022/language'
    plan=json.loads((out/'plan.json').read_text())
    vocabulary=ROOT/'data/tinystories_bpe_20261010/vocabulary.json'
    assert file_sha256(vocabulary)==plan['tokenizer_sha256']
    records=[]
    for case in plan['cases']:
        result=json.loads((out/f'{case["name"]}.json').read_text())
        source=Path(result['checkpoint']);before=file_sha256(source)
        assert before==result['checkpoint_sha256']
        saved=torch.load(source,map_location='cpu',weights_only=False)
        saved['tokenizer']=dict(path=str(vocabulary),sha256=plan['tokenizer_sha256'])
        saved['source_checkpoint']=dict(path=str(source),sha256=before)
        target=source.with_name('inference.pt');save_checkpoint(target,saved)
        check=torch.load(target,map_location='cpu',weights_only=False)
        assert all(torch.equal(v,check['weights'][k]) for k,v in saved['weights'].items())
        assert file_sha256(source)==before
        records.append(dict(case=case['name'],source=str(source),source_sha256=before,
                            checkpoint=str(target),sha256=file_sha256(target),weights_unchanged=True))
    save_json(out/'inference_checkpoints.json',records)
    print(json.dumps(dict(packaged=len(records),source_checkpoints_unchanged=True)))

if __name__=='__main__':main()
