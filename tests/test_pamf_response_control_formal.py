import copy
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.pamf_response_control_formal import design as d
from experiments.pamf_response_control_formal.runtime import qualify, source_counts, validate_training, load_control, audit_target, parent_selection
from experiments.pamf_response_control_formal.runner import command, PATH_ARGS
from experiments.pamf_uncertainty_restore.evaluation import VIEWS
from experiments.pamf_fcrossrx_readout.frozen import REFERENCE_RUN
from experiments.pamf_restoration.data import write_json


def source(readout='restored', correct=1600):
    cm=np.zeros((6,6),dtype=int);cm[0,0]=correct;cm[0,1]=1800-correct
    return dict(status='ANALYZED',kind='reference_controls' if readout=='identity' else d.KIND,
        role='source',count=1800,F_run=REFERENCE_RUN,F_seed=2026092701,original_default_head=True,
        all_predictions_before_truth=True,training_TX_labels_read=0,training_F_access=False,
        head_fit_updates=0,target_access=False,records=[dict(view=v,readout=readout,dimension='overall',
        value='all',count=1800,accuracy=100*correct/1800,confusion=cm.tolist()) for v in VIEWS])


def test_ties_and_negative_source_gains_never_open_target():
    for n in (1599,1600):
        result=qualify(source(correct=n),source(),source('identity',1600))
        assert not result['qualified'] and result['target_candidates']==[]


def test_single_net_correct_gain_is_positive_without_float_tie():
    candidate=source();r=candidate['records'][1]
    r['confusion'][0][0]+=1;r['confusion'][0][1]-=1;r['accuracy']=100*1601/1800
    result=qualify(candidate,source(),source('identity',1600))
    assert result['qualified'] and result['six_scene_net_correct_gain']==1
    assert result['mean6_gain_pp']==pytest.approx(100/10800)
    assert result['target_candidates']==[d.KIND]


def test_clean_harm_is_measured_against_raw_F50():
    c=source(correct=1601);raw=source('identity',1603)
    result=qualify(c,source(),raw)
    assert not result['qualified'] and result['clean_drop_vs_raw_pp']>.1


@pytest.mark.parametrize('key,value',[('role','target'),('target_access',True),('F_seed',1),
    ('head_fit_updates',1),('training_F_access',True),('all_predictions_before_truth',False)])
def test_wrong_source_contract_rejected(key,value):
    s=source();s[key]=value
    with pytest.raises(ValueError):source_counts(s,'restored')


@pytest.mark.parametrize('mutation',['missing','duplicate','fractional','nan','inconsistent'])
def test_incomplete_or_invalid_confusion_cannot_qualify(mutation):
    s=source()
    if mutation=='missing':s['records'].pop()
    elif mutation=='duplicate':s['records'][-1]=copy.deepcopy(s['records'][0])
    elif mutation=='fractional':s['records'][0]['confusion'][0][0]=1600.5
    elif mutation=='nan':s['records'][0]['accuracy']=float('nan')
    else:s['records'][0]['accuracy']=1.
    with pytest.raises(ValueError):source_counts(s,'restored')


def trained():
    return dict(status='SOURCE_TRAINED',candidate=d.CANDIDATE,epochs=20,updates=4940,
        L_exposures=126000,U_exposures=1134000,L_waveform_forwards=378000,U_waveform_forwards=3402000,
        TX_labels_read=0,F_outputs_read=0,F_weights_read=0,target_reads=0,frozen_context_updates=0,
        prior_initialization='scratch',inherited_prior_epochs=0,response_constraint_active=False)


@pytest.mark.parametrize('key,value',[('epochs',19),('F_weights_read',1),('candidate','observable_response'),
    ('U_exposures',0),('response_constraint_active',True)])
def test_training_or_row_mismatch_rejected(key,value):
    t=trained();t[key]=value
    with pytest.raises(ValueError):validate_training(t)


def test_strict_loader_rejects_primary_or_partial_checkpoint(monkeypatch,tmp_path):
    from experiments.pamf_response_control_formal import runtime
    for payload in [dict(candidate='observable_response',epoch=20,updates=4940),
                    dict(candidate=d.CANDIDATE,epoch=19,updates=4693)]:
        monkeypatch.setattr(runtime,'original_load',lambda *args:(None,payload))
        with pytest.raises(ValueError):load_control(tmp_path/'final.pt','cpu')


def test_negative_or_wrong_decision_stays_before_target(tmp_path):
    args=SimpleNamespace(output=tmp_path,kind=d.KIND,expected_commit='a')
    write_json(tmp_path/'resolved_config.json',d.configuration('a'))
    write_json(tmp_path/'source_decision.json',qualify(source(),source(),source('identity')))
    with pytest.raises(ValueError):audit_target(args)


def test_only_new_eval_module_and_zero_training_are_registered(tmp_path):
    args=SimpleNamespace(**{k:tmp_path/k for k in PATH_ARGS},expected_commit='a')
    assert command(args,'worker')[3]=='experiments.pamf_response_control_formal.runner'
    config=d.configuration('a')
    assert config['new_training_updates']==0 and config['max_own_GPU_workers']==1
    assert config['parent_run'].endswith('-r32') and config['candidate']=='pair_aug_control'


@pytest.mark.parametrize('complete',[{},dict(status='RUNNING'),dict(status='ANALYZED',source_rows=[])])
def test_unfinished_parent_stops_before_any_source_or_target_read(tmp_path,complete):
    parent=tmp_path/d.PARENT;parent.mkdir()
    write_json(parent/'launch.json',dict(commit=d.PARENT_COMMIT))
    write_json(parent/'complete.json',complete)
    with pytest.raises(ValueError):parent_selection(SimpleNamespace(parent_run=parent))


def test_wrong_parent_code_rejected(tmp_path):
    parent=tmp_path/d.PARENT;parent.mkdir()
    write_json(parent/'launch.json',dict(commit='wrong'))
    with pytest.raises(ValueError):parent_selection(SimpleNamespace(parent_run=parent))


def test_negative_control_finishes_without_smoke_checkpoint_or_gpu(monkeypatch,tmp_path):
    from experiments.pamf_response_control_formal import runner
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(runner,'parent_selection',lambda args:qualify(source(),source(),source('identity')))
    def forbidden(*args):raise AssertionError('Source-negative candidate attempted model or GPU access')
    monkeypatch.setattr(runner,'smoke',forbidden);monkeypatch.setattr(runner,'queue',forbidden)
    output=tmp_path/'new';runner.run(SimpleNamespace(output=output,expected_commit='a'))
    assert (output/'complete.json').is_file() and not (output/d.KIND).exists()
    assert not (output/'target_complete.json').exists()
