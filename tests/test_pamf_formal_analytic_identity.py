import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.pamf_formal_analytic_identity import design as d,runtime as r,runner,analysis
from experiments.pamf_compact_identity_probe.analysis import analyze as legacy


def args():
    root=Path('/data/runs')
    return SimpleNamespace(runs_root=root,output=root/d.RUN,source_cache=Path('/source_cache'),
        formal_reference=root/d.FORMAL/'reference',target_truth=root/d.TRUTH_RUN/'data/truth.json',expected_commit='pinned')


@pytest.mark.parametrize('field,bad',[('output',Path('/other')/d.RUN),('formal_reference',Path('/other/reference')),
    ('target_truth',Path('/other/truth.json'))])
def test_wrong_formal_paths_reject_before_IO(monkeypatch,field,bad):
    value=args();setattr(value,field,bad)
    monkeypatch.setattr(r,'read_json',lambda *a:pytest.fail('No IO before path validation'))
    with pytest.raises(ValueError):r.validate_parent(value)


def test_parent_not_full_formal_complete_keeps_arrays_and_labels_closed(monkeypatch):
    value=args();seen=[]
    def read(path):
        seen.append(path.name)
        return dict(commit=d.PARENT_COMMIT) if path.name=='launch.json' else dict(status='SOURCE_TRAINED')
    monkeypatch.setattr(r,'read_json',read)
    monkeypatch.setattr(r.np,'load',lambda *a,**k:pytest.fail('No target arrays before full completion'))
    with pytest.raises(ValueError):r.inputs(value,with_arrays=True)
    assert seen==['launch.json','complete.json']


@pytest.mark.parametrize('field,bad',[('epochs',19),('updates',4939),('F_outputs_read',1),('TX_labels_read',1),
    ('target_reads',1),('V_reads',1),('parameters',794529)])
def test_wrong_compression_source_training_rejects(field,bad,monkeypatch):
    arm=d.NEW_ROWS[0]
    trained=dict(status='SOURCE_TRAINED',arm=arm,epochs=20,updates=4940,L_exposures=126000,U_exposures=1134000,
        TX_labels_read=0,F_weights_read=0,F_outputs_read=0,training_F_access=False,V_reads=0,target_reads=0,parameters=d.parent.PARAMETERS[arm])
    trained[field]=bad
    def read(path):
        if path.name=='complete.json':return dict(status='ANALYZED',arm=arm,role='target',all_predictions_before_truth=True,target_feedback=False)
        assert path.name=='training_complete.json'
        return trained
    monkeypatch.setattr(r,'read_json',read)
    with pytest.raises(ValueError):r.validate_scored_row(Path('/runs')/d.PARENT,arm,new=True)


@pytest.mark.parametrize('bad',['missing_features','bad_marker','wrong_CUDA'])
def test_target_truth_stays_closed_until_all_features_validate(monkeypatch,bad):
    value=args();ids=np.arange(2)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0' if bad=='wrong_CUDA' else '')
    monkeypatch.setattr(runner,'inputs',lambda *a,**k:(ids,{},{}))
    if bad=='missing_features':
        def absent(*a):raise FileNotFoundError()
        monkeypatch.setattr(runner,'load_features',absent)
    else:monkeypatch.setattr(runner,'read_json',lambda *a:dict(status='INCOMPLETE'))
    with pytest.raises((ValueError,FileNotFoundError)):
        runner.score(value,truth_loader=lambda *a:pytest.fail('Truth must stay closed'))


def test_feature_marker_all112_precedes_every_array_and_truth(monkeypatch):
    monkeypatch.setattr(runner,'read_json',lambda *a:dict(status='FEATURES_COMPLETE',count=d.COUNT,cells=111))
    monkeypatch.setattr(runner.np,'load',lambda *a,**k:pytest.fail('No arrays with incomplete marker'))
    with pytest.raises(ValueError):runner.load_features(args(),np.arange(2))


def test_reused_maths_has_exact_alias_equivalence_and_retains_all_groups():
    rng=np.random.default_rng(45);labels=np.tile(np.repeat(np.arange(6),2),9)
    rx=np.repeat(np.arange(3),36);day=np.tile(np.repeat(np.arange(3),12),3);ids=np.asarray(['id'+str(i) for i in range(len(labels))])
    features={}
    for group,size in d.GROUPS.items():
        centers=rng.normal(size=(6,size));clean=centers[labels]+.05*rng.normal(size=(len(labels),size))
        for view in d.VIEWS:
            for name in d.READOUTS:
                features[(view,name,group)]=clean.copy() if (view,name)==('clean','raw') else clean+.2*rng.normal(size=clean.shape)
    result=analysis.analyze(features,ids,labels,rx,day)
    expected=legacy({(v,d.ALIASES[name],g):a for (v,name,g),a in features.items()},ids,labels,rx,day)
    reverse={old:new for new,old in d.ALIASES.items()}
    for key in ('records','query_label_negative_controls','comparisons'):
        for record in expected[key]:
            for field in ('candidate','control'):
                if field in record:record[field]=reverse[record[field]]
        assert result[key]==expected[key]
    assert len(result['records'])==1092 and len(result['query_label_negative_controls'])==252 and len(result['comparisons'])==84
    assert {record['group'] for record in result['records']}==set(d.GROUPS)
    assert result['selected_candidate'] is None and not result['target_decisions_changed']


def test_no_model_or_classifier_parameters_and_fixed_formal_configuration():
    assert list(runner.extractor().parameters())==[]
    config=d.configuration('pinned')
    assert d.COUNT==168000 and len(d.READOUTS)*len(d.VIEWS)*len(d.GROUPS)==112
    assert d.COUNT*len(d.READOUTS)*len(d.VIEWS)==4704000
    assert config['GPU_workers']==config['head_fit_updates']==config['F_weights_read']==config['neural_training_updates']==0
    assert config['controls']['permutations']==199 and not config['controls']['is_p_value']
    assert config['parent_training_or_decisions_changed'] is False
