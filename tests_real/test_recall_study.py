import json
from pathlib import Path
import time
import pytest
import torch
from energy_real import diagnostics,recall_study
from energy_real.episodes import oracle
from energy_real.episodic import EnergyRetriever,FactStore,parse_query
from energy_real.runtime import atomic_checkpoint


def test_shared_benchmark_disjoint_entities_wording_and_mixed_paths(tmp_path):
    splits={split:diagnostics.generate(split,1) for split in ('train','valid','test')}
    entities={split:{e.query.subject for e in episodes} for split,episodes in splits.items()}
    wording={split:set(diagnostics.TEMPLATES[split].values()) for split in splits}
    for a,b in [('train','valid'),('valid','test'),('train','test')]:
        assert not entities[a]&entities[b] and not wording[a]&wording[b]
    for episode in splits['test']:
        assert not episode.natural_question.startswith('Follow')
        store=FactStore(tmp_path/f'{episode.id}.sqlite')
        store.apply(episode.edits,expected_revision=0,request_id='episode',authorized=True)
        assert store.snapshot()[2]==diagnostics.events_for(episode)
        expected,indices=oracle(episode)
        answer=store.query(episode.query)
        assert answer['answer']==expected
        assert [f['fact_id'] for f in answer['facts']]==[store.snapshot()[2][i]['fact_id'] for i in indices]
        if episode.family=='missing_three_hop':
            assert answer['abstained'] and len(answer['trace'])==3 and len(answer['facts'])==2
        if episode.family=='three_hop':assert len(set(episode.query.relations))==3


def test_episode_identity_and_content_are_stable_across_sample_sizes():
    small={e.id:e.payload() for e in diagnostics.generate('test',1)}
    large={e.id:e.payload() for e in diagnostics.generate('test',3)}
    assert set(small)<=set(large)
    assert all(payload==large[identifier] for identifier,payload in small.items())


@pytest.mark.parametrize('kind',recall_study.KINDS)
def test_delayed_loss_has_finite_gradients_for_edits_and_confusable_keys(kind):
    model=recall_study.new_model(kind,16,.001,11)
    episode=next(e for e in diagnostics.generate('train',1) if e.family=='similar')
    loss,parts=recall_study.recall_loss(model,episode)
    assert set(parts)=={'delayed_ce','locality_mse'}
    assert float(loss.detach())==pytest.approx(parts['delayed_ce']+.1*parts['locality_mse'])
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    deleted=next(e for e in diagnostics.generate('train',1) if e.family=='delete')
    *_,targets,ids=recall_study.tensors(deleted,16)
    assert targets[ids[0]]==0  # Deleted address trains the explicit null class.
    assert targets[ids[-1]]>0  # Unrelated fact remains a positive retrieval target.


def test_thresholds_trace_and_checkpoint_compatibility(tmp_path):
    ep=diagnostics.generate('valid',1)[0]
    events=diagnostics.events_for(ep)
    model=recall_study.new_model('precision',16,.001,11)
    path=tmp_path/'controller.pt'
    recall_study.save_controller(model,path,11,0)
    retriever=EnergyRetriever(events,prior=.001,controller=path,score_threshold=100.)
    assert retriever.read(ep.query.subject,ep.query.relations[0]) is None
    trace=retriever.last_diagnostic
    assert trace['score'] is not None and trace['margin'] is not None
    assert trace['candidate_fact_ids']==[events[0]['fact_id']] and trace['selected_fact_id'] is None
    assert not trace['accepted']
    row=recall_study.measure(ep,{'answer':None,'abstained':True,'facts':[],'trace':[trace]},events)
    assert row['confidence_rejections']==1 and row['wrong_top_candidates']==0
    for config in ({'prior':.01},{'prior':.001,'dimension':32},{'prior':.001,'capacity':16}):
        with pytest.raises(ValueError,match='configuration mismatch'):
            EnergyRetriever(events,controller=path,**config)


def test_calibration_uses_absence_and_missing_links_and_never_test_labels():
    model=recall_study.new_model('precision',16,.01,11)
    valid=diagnostics.generate('valid',1)
    result=recall_study.calibrate(model,valid,time.perf_counter()+30)
    assert result['selected']['episodes']==len(diagnostics.FAMILIES)
    assert 'missing_three_hop' in result['selected']['families']
    default=next(r for r in result['grid'] if r['score_threshold']==.5 and r['margin_threshold']==0.)
    assert result['selected']['accuracy']>=default['accuracy']
    with pytest.raises(ValueError,match='validation'):
        recall_study.calibrate(model,diagnostics.generate('test',1),time.perf_counter()+30)


def test_small_study_freezes_selection_and_preserves_initial_controls(tmp_path,monkeypatch):
    monkeypatch.setattr(recall_study,'DIMENSIONS',(16,))
    monkeypatch.setattr(recall_study,'PRIORS',(.01,))
    # Native allocation measurement is covered separately; keep this study test focused.
    monkeypatch.setattr(recall_study,'measure_peaks',lambda *args:[])
    args=recall_study.parser().parse_args(['--out',str(tmp_path/'study'),'--steps','2',
         '--train-per-family','1','--valid-per-family','1','--test-per-family','1','--max-seconds','60'])
    result=recall_study.run(args)
    assert result['status']=='completed' and len(result['records'])==6
    assert {(r['kind'],r['seed']) for r in result['selected']}=={(k,s) for k in recall_study.KINDS for s in (11,22,33)}
    assert (args.out/'selection.json').stat().st_mtime_ns <= (args.out/'test_episodes.json').stat().st_mtime_ns
    assert json.loads((args.out/'exact_oracle.json').read_text())['summary']['accuracy']==1
    assert all('initial' in t and 'learning_interval' in t for t in result['test_results'])
    config=recall_study.selected_memory(args.out/'selection.json')
    assert config['dimension']==16 and config['capacity']==256
    config['controller'].write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):
        recall_study.selected_memory(args.out/'selection.json')
    with pytest.raises(FileExistsError):recall_study.run(args)


def test_cpu_peak_accounting_includes_process_runtime():
    result=recall_study.peak_worker(None,diagnostics.generate('test',1))
    assert result['process_peak_rss_bytes']>=result['import_peak_rss_bytes']>0


@pytest.mark.parametrize('query',[{'subject':'x','relations':'abc'}, {'subject':'x'}, {'subject':'x','relations':[]}])
def test_malformed_interpretations_are_rejected(query):
    with pytest.raises(ValueError):parse_query(query)
