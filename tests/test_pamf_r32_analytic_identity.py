import copy
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.pamf_r32_analytic_identity import design as d
from experiments.pamf_r32_analytic_identity.analysis import analyze
from experiments.pamf_r32_analytic_identity.runtime import validate_training,validate_source_row


def toy():
    labels=np.tile(np.arange(6),8);rx=np.repeat(np.arange(4),12);day=np.tile(np.repeat(np.arange(2),6),4)
    ids=np.asarray(['p%03d'%i for i in range(len(labels))]);clean=np.eye(6)[labels]
    arrays={(v,r,g):clean.copy() for v in d.VIEWS for r in d.READOUTS for g in d.GROUPS}
    return ids,labels,rx,day,arrays


def test_matched_source_comparison_keeps_every_row_group_and_scene():
    ids,y,rx,day,features=toy()
    result=analyze(features,ids,y,rx,day)
    assert len(result['records'])==3*4*7*(1+6+4+2)
    assert len(result['comparisons'])==3*4*7
    assert all(abs(r['margin_change'])<1e-12 for r in result['comparisons'])
    assert result['selected_candidate'] is None and not result['source_selection_changed']


def test_wrong_TX_detail_is_negative_even_when_other_candidates_are_good():
    ids,y,rx,day,features=toy()
    for v in d.VIEWS:
        for g in d.GROUPS:features[(v,'observable_response',g)]=np.roll(features[(v,'observable_response',g)],1,axis=1)
    result=analyze(features,ids,y,rx,day)
    row=next(r for r in result['records'] if (r['candidate'],r['group'],r['view'],r['dimension'])==('observable_response','combined','practical_low_urban','overall'))
    assert row['restored']['mean_margin']==pytest.approx(-.2)
    assert len(result['records'])==3*4*7*(1+6+4+2)


def training(row):
    return dict(status='SOURCE_TRAINED',candidate=row,epochs=20,updates=4940,L_exposures=126000,U_exposures=1134000,
        L_waveform_forwards=378000,U_waveform_forwards=3402000,TX_labels_read=0,F_outputs_read=0,F_weights_read=0,
        target_reads=0,frozen_context_updates=0,prior_initialization='scratch',inherited_prior_epochs=0,
        response_constraint_active=row=='observable_response')


@pytest.mark.parametrize('key,value',[('epochs',19),('candidate','other'),('F_weights_read',1),('U_exposures',0),('response_constraint_active',True)])
def test_incomplete_or_wrong_training_never_becomes_new_source_input(key,value):
    value_dict=training('pair_aug_control');value_dict[key]=value
    with pytest.raises(ValueError):validate_training(value_dict,'pair_aug_control')


def test_both_actual_row_roles_are_checked():
    for row in d.NEW_ROWS:
        validate_training(training(row),row)
    wrong=training('observable_response');wrong['response_constraint_active']=False
    with pytest.raises(ValueError):validate_training(wrong,'observable_response')


@pytest.mark.parametrize('key,value',[('role','target'),('candidate','observable_response'),('all_predictions_before_truth',False),('training_TX_F_access',True)])
def test_source_only_completeness_and_candidate_match(key,value):
    complete=dict(status='ANALYZED',candidate='pair_aug_control',role='source',all_predictions_before_truth=True,training_TX_F_access=False)
    complete[key]=value
    with pytest.raises(ValueError):validate_source_row(complete,'pair_aug_control')


def test_missing_candidate_cell_blocks_whole_analysis():
    ids,y,rx,day,features=toy();features.pop(next(iter(features)))
    with pytest.raises(ValueError):analyze(features,ids,y,rx,day)


@pytest.mark.parametrize('bad',[('cells',56),('TX_label_reads',1),('F_embeddings_logits_read',1),('views',['clean']),('new_calibration_samples',1)])
def test_feature_header_mismatch_prevents_source_label_open(tmp_path,monkeypatch,bad):
    from experiments.pamf_r32_analytic_identity import runner
    from experiments.pamf_restoration.data import write_json
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(runner,'sources',lambda *args,**kwargs:(np.arange(1800),None,None,None,None,None))
    header=dict(status='FEATURES_COMPLETE',count=1800,views=list(d.VIEWS),groups=d.GROUPS,readouts=list(d.READOUTS),cells=112,
        TX_label_reads=0,F_weights_read=0,F_embeddings_logits_read=0,extra_restorer_predictions=0,new_calibration_samples=0)
    header[bad[0]]=bad[1];write_json(tmp_path/'feature_complete.json',header)
    def forbidden(*args,**kwargs):raise AssertionError('Premature source truth access')
    with pytest.raises(ValueError):runner.score(SimpleNamespace(output=tmp_path),label_loader=forbidden)


def test_absent_112_cell_blocks_label_open(tmp_path,monkeypatch):
    from experiments.pamf_r32_analytic_identity import runner
    from experiments.pamf_restoration.data import write_json
    ids=np.arange(1800);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(runner,'sources',lambda *args,**kwargs:(ids,None,None,None,None,None))
    write_json(tmp_path/'feature_complete.json',dict(status='FEATURES_COMPLETE',count=1800,views=list(d.VIEWS),groups=d.GROUPS,
        readouts=list(d.READOUTS),cells=112,TX_label_reads=0,F_weights_read=0,F_embeddings_logits_read=0,extra_restorer_predictions=0,new_calibration_samples=0))
    np.save(tmp_path/'physical_ids.npy',ids);np.savez(tmp_path/'analytic_features.npz',missing=np.zeros((1,1)))
    def forbidden(*args,**kwargs):raise AssertionError('Premature source truth access')
    with pytest.raises(ValueError):runner.score(SimpleNamespace(output=tmp_path),label_loader=forbidden)


def test_R34_metric_matches_exactly_for_unchanged_reference_on_toy_data():
    from tools.analyze_pamf_source_cross_record import reference_pool,contrasts,summarize
    ids,y,rx,day,features=toy();result=analyze(features,ids,y,rx,day)
    clean=features[('clean','raw','mirror')];ref=reference_pool(clean,ids,y,rx,day)
    expected=contrasts(clean,ref,y,rx,day)
    row=next(r for r in result['records'] if (r['candidate'],r['group'],r['view'],r['dimension'])==('r17_reference','mirror','clean','overall'))
    assert row['baseline']==summarize(expected,expected,expected,np.ones(len(y),bool))['baseline']
