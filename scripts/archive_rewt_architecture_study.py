"""Freeze the delivered source and inventory completed study artifacts."""
import hashlib,json,platform,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from rewt.data import file_sha256
from rewt.experiment import save_json
OUT=ROOT/'reports/rewt_architecture_20261022'


def main():
    required=['test_results.json','language/test_results.json','long_memory/test_results.json',
        'binding_memory/test_results.json','memory_ablation/test_results.json','summary.json',
        'trained_geometry.json','benchmark_final.json','training_replay.json','verification.json',
        'triton_tiles.json','equal_compute/test_results.json','FINDINGS.md','RESULTS.md','REPRODUCING.md']
    for name in required:assert (OUT/name).exists(),name
    plan=json.loads((OUT/'plan.json').read_text())
    scripts=['study_rewt_architecture','study_rewt_architecture_language','study_rewt_equal_compute','study_rewt_long_memory',
        'study_rewt_binding_memory','study_rewt_memory_ablation','sample_rewt',
        'benchmark_rewt_architecture','benchmark_rewt_directional','audit_rewt_architecture_models',
        'audit_rewt_language_documents','audit_rewt_refinement_contribution','check_rewt_architecture_cuda','check_rewt_triton_tiles',
        'check_rewt_triton_interpreter','inspect_rewt_triton_resources','verify_rewt_architecture',
        'verify_rewt_training_replay','summarize_rewt_architecture','plot_rewt_memory_study',
        'plot_rewt_architecture_results','write_rewt_architecture_findings','package_rewt_language_checkpoints','archive_rewt_architecture_study',
        'train_rewt_curriculum']
    source=sorted(set([*(ROOT/'rewt').glob('*.py'),ROOT/'rewt/README.md',
                       *(ROOT/'tests_rewt').glob('*.py'),*[ROOT/'scripts'/f'{n}.py' for n in scripts]]))
    archive=OUT/'final_source.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for path in source:tar.add(path,arcname=path.relative_to(ROOT))
    checkpoints=[]
    for name in ['training_results.json','language/training_results.json','long_memory/training_results.json',
                 'binding_memory/training_results.json','memory_ablation/training_results.json','equal_compute/training_results.json']:
        for result in json.loads((OUT/name).read_text()):
            path=Path(result['checkpoint']);digest=file_sha256(path)
            assert digest==result['checkpoint_sha256']
            checkpoints.append(dict(path=str(path.relative_to(ROOT)),sha256=digest,bytes=path.stat().st_size))
    assert len(checkpoints)==96
    artifacts=[p for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'
               and '__pycache__' not in p.parts and p.suffix not in ('.pyc',)]
    data=[p for p in (ROOT/'data/rewt_architecture_20261022').rglob('*') if p.is_file()]
    records=lambda paths:[dict(path=str(p.relative_to(ROOT)),sha256=file_sha256(p),bytes=p.stat().st_size) for p in sorted(paths)]
    save_json(OUT/'manifest.json',dict(source=records(source),artifacts=records(artifacts),data=records(data),
        checkpoints=checkpoints,python=sys.version,pytorch=torch.__version__,platform=platform.platform(),
        total_new_fits=96,note='Original best.pt hashes checked against frozen fit records. Language inference.pt copies add tokenizer metadata. Full training traces and last.pt optimizer checkpoints remain in the run directories. Archive contains delivered source, not model weights.'))
    print(json.dumps(dict(source_files=len(source),artifacts=len(artifacts),data_files=len(data),checkpoints=len(checkpoints))))

if __name__=='__main__':main()
