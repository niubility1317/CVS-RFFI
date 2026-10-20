from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.pamf_group_formal_identity import design as d,runtime as r,runner,analysis
from experiments.pamf_compression_formal_identity import design as old_design,analysis as old_analysis

def args():
    root=Path('/data/runs')
    return SimpleNamespace(runs_root=root,output=root/d.RUN,source_cache=Path('/source_cache'),formal_reference=root/d.FORMAL/'reference',
        target_truth=root/d.TRUTH_RUN/'data/truth.json',expected_commit='pinned')

@pytest.mark.parametrize('field,bad',[('output',Path('/other')/d.RUN),('formal_reference',Path('/other/reference')),('target_truth',Path('/other/truth.json'))])
def test_input_paths_reject_before_IO(monkeypatch,field,bad):
    value=args();setattr(value,field,bad);monkeypatch.setattr(r,'read_json',lambda *a:pytest.fail('No IO before path validation'))
    with pytest.raises(ValueError):r.validate_parent(value)

def test_both_parents_must_finish_before_target_arrays(monkeypatch):
    seen=[]
    def read(path):
        seen.append(path.name)
        return dict(commit=(d.CONTROL_COMMIT if path.parent.name==d.CONTROL_RUN else d.PARENTS[path.parent.name]['commit'])) if path.name=='launch.json' else dict(status='SOURCE_TRAINED')
    monkeypatch.setattr(r,'read_json',read);monkeypatch.setattr(r.np,'load',lambda *a,**k:pytest.fail('Target arrays stay closed'))
    with pytest.raises(ValueError):r.inputs(args(),with_arrays=True)
    assert seen==['launch.json']*2+['complete.json']

@pytest.mark.parametrize('field,bad',[('epochs',19),('updates',4939),('F_outputs_read',1),('TX_labels_read',1),('target_reads',1),('V_reads',1),('parameters',794529)])
def test_parent_training_permissions_and_fixed_final_required(monkeypatch,field,bad):
    value=d.ROWS[d.NEW_ROWS[0]];arm=value['row'];trained=dict(status='SOURCE_TRAINED',arm=arm,epochs=20,updates=4940,L_exposures=126000,U_exposures=1134000,
        TX_labels_read=0,F_weights_read=0,F_outputs_read=0,training_F_access=False,V_reads=0,target_reads=0,parameters=value['parameters'])
    trained[field]=bad
    def read(path):
        if path.name=='complete.json':return dict(status='ANALYZED',arm=arm,role='target',all_predictions_before_truth=True,target_feedback=False)
        assert path.name=='training_complete.json';return trained
    monkeypatch.setattr(r,'read_json',read)
    with pytest.raises(ValueError):r.validate_scored_row(Path('/runs')/value['run'],arm,parameters=value['parameters'])

@pytest.mark.parametrize('bad',['missing','marker','CUDA'])
def test_labels_stay_closed_until_every_new_and_cached_feature_cell_validates(monkeypatch,bad):
    ids=np.arange(2);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0' if bad=='CUDA' else '')
    monkeypatch.setattr(runner,'inputs',lambda *a,**k:(ids,{},{}))
    if bad=='missing':
        def absent(*a):raise FileNotFoundError()
        monkeypatch.setattr(runner,'load_features',absent)
    else:monkeypatch.setattr(runner,'read_json',lambda *a:dict(status='INCOMPLETE'))
    with pytest.raises((ValueError,FileNotFoundError)):runner.score(args(),truth_loader=lambda *a:pytest.fail('Truth must stay closed'))

def test_incomplete_feature_marker_prevents_array_open(monkeypatch):
    monkeypatch.setattr(runner,'read_json',lambda *a:dict(status='FEATURES_COMPLETE',count=d.COUNT,cells=d.TOTAL_CELLS-1))
    monkeypatch.setattr(runner.np,'load',lambda *a,**k:pytest.fail('Arrays stay closed'))
    with pytest.raises(ValueError):runner.load_features(args(),np.arange(2))

@pytest.mark.parametrize('bad',['commit','feature_labels','calibration','readouts','ids'])
def test_R56_cached_coordinates_require_same_source_moments_and_IDs(monkeypatch,bad):
    ids=np.arange(2)
    def read(path):
        if path.name=='launch.json':return dict(commit='wrong' if bad=='commit' else d.CACHED_COMMIT)
        if path.name=='complete.json':return dict(status='ANALYZED')
        if path.name=='feature_complete.json':return dict(status='FEATURES_COMPLETE',count=d.COUNT,views=list(d.VIEWS),groups=d.GROUPS,
            readouts=d.CACHED_FEATURE_READOUTS[:-1] if bad=='readouts' else d.CACHED_FEATURE_READOUTS,cells=280,TX_labels_read=int(bad=='feature_labels'),F_access=0,
            model_checkpoint_loads=0,extra_model_predictions=0,new_calibration_samples=0)
        if path.name=='calibration_reference.json':return dict(run=d.REFERENCE,commit=d.REFERENCE_COMMIT,count=8192,L_count=1000,U_count=7192,
            new_calibration_samples=int(bad=='calibration'),new_parameters_fitted=0)
        pytest.fail('Unexpected IO')
    monkeypatch.setattr(r,'read_json',read)
    def load(*a,**k):
        assert bad=='ids','Bad cached metadata must fail before array open';return ids[::-1]
    monkeypatch.setattr(r.np,'load',load)
    with pytest.raises(ValueError):r.cached_reference(args(),ids)

def test_unchanged_math_matches_R50_under_exact_synthetic_readout_aliases():
    rng=np.random.default_rng(56);labels=np.tile(np.repeat(np.arange(6),2),9);rx=np.repeat(np.arange(3),36);day=np.tile(np.repeat(np.arange(3),12),3)
    ids=np.asarray(['id'+str(i) for i in range(len(labels))]);features={}
    for group,size in d.GROUPS.items():
        centers=rng.normal(size=(6,size));clean=centers[labels]+.05*rng.normal(size=(len(labels),size))
        for view in d.VIEWS:
            for name in d.READOUTS:features[(view,name,group)]=clean.copy() if (view,name)==('clean','raw') else clean+.2*rng.normal(size=clean.shape)
    from unittest.mock import patch
    actual=analysis.analyze(features,ids,labels,rx,day)
    with patch.object(old_analysis,'d',d):expected=old_analysis.analyze(features,ids,labels,rx,day)
    assert actual==expected
    assert len(actual['records'])==1456 and len(actual['query_label_negative_controls'])==336 and len(actual['comparisons'])==140
    assert {(v['candidate'],v['control']) for v in actual['comparisons']}==set(d.COMPARISONS)
    assert actual['selected_candidate'] is None and not actual['target_decisions_changed'] and not actual['new_source_thresholds']


def test_no_new_model_F_head_or_source_calibration_and_both_group_rows_and_fixed_control_are_included():
    cfg=d.configuration('pinned');assert list(runner.extractor().parameters())==[]
    assert len(d.PARENTS)==1 and len(d.NEW_ROWS)==3 and len(d.CACHED_READOUTS)==2
    assert len(d.READOUTS)==5 and len(d.CANDIDATES)==4 and len(d.COMPARISONS)==5
    assert (d.CACHED_CELLS,d.NEW_CELLS,d.TOTAL_CELLS)==(56,84,140)
    assert (cfg['new_analytic_transform_samples'],d.EXPECTED_RECORDS,d.EXPECTED_CONTROLS,d.EXPECTED_COMPARISONS)==(3528000,2016,336,140)
    assert cfg['GPU_workers']==cfg['model_checkpoint_loads']==cfg['new_calibration_samples']==cfg['head_fit_updates']==cfg['F_weights_read']==cfg['neural_training_updates']==0
    assert cfg['controls']['permutations']==199 and not cfg['controls']['is_p_value']
    assert set(d.CACHED_READOUTS).isdisjoint(d.NEW_ROWS) and cfg['parent_training_or_decisions_changed'] is False


@pytest.mark.parametrize('field,bad',[('context_GN_groups',4),('prior_GN_groups',1),('prior_parameters_changed',False),('new_calibration_IQ',1),('new_trainable_parameters',1)])
def test_R62_grouping_and_no_expansion_required(monkeypatch,field,bad):
    arm='group_preserve';trained=training_marker(arm);trained[field]=bad
    def read(path):
        if path.name=='complete.json':return dict(status='ANALYZED',arm=arm,role='target',all_predictions_before_truth=True,target_feedback=False)
        if path.name=='training_complete.json':return trained
        pytest.fail('Reject before summary and target arrays')
    monkeypatch.setattr(r,'read_json',read)
    with pytest.raises(ValueError):r.validate_scored_row(Path('/runs')/d.PARENT,arm,parameters=248828)

def training_marker(arm):
    return dict(status='SOURCE_TRAINED',arm=arm,epochs=20,updates=4940,L_exposures=126000,U_exposures=1134000,
        TX_labels_read=0,F_weights_read=0,F_outputs_read=0,training_F_access=False,V_reads=0,target_reads=0,parameters=248828,
        context_GN_groups={'group_preserve':3,'regroup_control':1}[arm],prior_GN_groups=4,prior_parameters_changed=True,new_calibration_IQ=0,new_trainable_parameters=0)

@pytest.mark.parametrize('arm',['group_preserve','regroup_control'])
def test_both_groupings_allow_only_original_formal_prediction_path(monkeypatch,arm):
    def read(path):
        if path.name=='complete.json':return dict(status='ANALYZED',arm=arm,role='target',all_predictions_before_truth=True,target_feedback=False)
        if path.name=='training_complete.json':return training_marker(arm)
        assert path.name=='summary.json'
        return dict(status='ANALYZED',role='target',count=d.COUNT,original_default_head=True,all_predictions_before_truth=True,
            training_F_access=False,training_TX_labels_read=0,head_fit_updates=0,F_run=d.parent.F_RUN,F_seed=d.parent.F_SEED)
    monkeypatch.setattr(r,'read_json',read);root=Path('/runs')/d.PARENT
    assert r.validate_scored_row(root,arm,parameters=248828)==root/arm/'target_confirmation/native_map_prior/prediction'

def test_child_commands_and_control_bindings():
    command=runner.command(args(),'score');assert command[command.index('-m')+1]=='experiments.pamf_group_formal_identity.runner'
    assert d.CONTROL_ROW=='retain_refine' and d.ROWS['r49_retain_control']['parameters']==350521
    assert set(d.CACHED_READOUTS)=={'raw','r46_350521'}
