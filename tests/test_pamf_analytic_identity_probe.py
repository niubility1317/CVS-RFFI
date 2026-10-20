from types import SimpleNamespace
import numpy as np
import pytest
from experiments.pamf_analytic_identity_probe.analysis import class_similarities,contrasts,permuted_queries,null_control,representations,fit_moments
from tools.analyze_pamf_source_cross_record import reference_pool,contrasts as original_contrasts


def panel():
    labels=np.tile(np.arange(6),8);rx=np.repeat(np.arange(4),12);day=np.tile(np.repeat(np.arange(2),6),4)
    ids=np.asarray(['p%03d'%i for i in range(len(labels))]);features=np.eye(6)[labels]
    return ids,labels,rx,day,features


def test_actual_TX_matches_cross_RX_and_day_and_agrees_with_existing_metric():
    ids,y,rx,day,x=panel();ref=reference_pool(x,ids,y,rx,day)
    values=class_similarities(x,ref,rx,day);new=contrasts(values,y);old=original_contrasts(x,ref,y,rx,day)
    for key in old:np.testing.assert_allclose(new[key],old[key])
    assert np.all(new['valid']) and np.allclose(new['margin'],1.)


def test_wrong_TX_vectors_reverse_real_identity_margin():
    ids,y,rx,day,x=panel();ref=reference_pool(x,ids,y,rx,day)
    result=contrasts(class_similarities(np.roll(x,1,axis=1),ref,rx,day),y)
    assert np.allclose(result['margin'],-.2)


def test_zero_or_incomplete_reference_is_reported_without_fabricated_validity():
    ids,y,rx,day,x=panel();x[:]=0
    ref=reference_pool(x,ids,y,rx,day);result=contrasts(class_similarities(x,ref,rx,day),y)
    assert not result['valid'].any() and result['zero_query'].all() and (result['missing_cells']>0).all()


def test_query_label_negative_control_preserves_each_RX_day_histogram():
    _,y,rx,day,_=panel();p=permuted_queries(y,rx,day)
    assert p.shape==(199,len(y))
    for r in np.unique(rx):
        for d in np.unique(day):
            mask=(rx==r)&(day==d)
            assert all(np.array_equal(np.sort(row[mask]),np.sort(y[mask])) for row in p)
    np.testing.assert_array_equal(p,permuted_queries(y,rx,day))


def test_clean_signal_exceeds_query_permutation_but_constant_features_do_not():
    ids,y,rx,day,x=panel();p=permuted_queries(y,rx,day)
    for value,expected in [(x,True),(np.ones_like(x),False)]:
        scores=class_similarities(value,reference_pool(value,ids,y,rx,day),rx,day)
        result=null_control(scores,y,p,scores['valid'])
        assert result['above_null_p95']==expected and result['is_p_value'] is False


def test_fixed_coordinate_selection_and_equal_group_dimension_weights():
    sizes=dict(mirror=128,spectral=4,reference=28);raw={k:np.ones((3,n)) for k,n in sizes.items()}
    moments={k:dict(mean=[0.]*n,std=[1.]*n) for k,n in sizes.items()}
    result=representations(raw,moments)
    assert {k:v.shape[1] for k,v in result.items()}==dict(mirror=64,spectral=4,reference=24,combined=92)
    assert np.allclose(np.sum(result['combined']**2,axis=1),1.)


def test_calibration_is_streamed_from_unlabelled_training_vectors_only():
    values=[{k:np.arange(4*n,dtype=float).reshape(4,n) for k,n in dict(mirror=128,spectral=4,reference=28).items()}]
    result=fit_moments(values)
    assert result['count']==4 and result['label_reads']==0 and result['F_access'] is False
    for key,value in values[0].items():
        np.testing.assert_allclose(result['moments'][key]['mean'],value.mean(0))
        np.testing.assert_allclose(result['moments'][key]['std'],np.maximum(value.std(0),.05))


@pytest.mark.parametrize('bad',[np.zeros((2,0)),np.full((2,6),np.nan)])
def test_nonfinite_or_empty_features_rejected(bad):
    ids,y,rx,day,x=panel();ref=reference_pool(x,ids,y,rx,day)
    with pytest.raises(ValueError):class_similarities(bad,ref,rx[:2],day[:2])


def feature_header():
    from experiments.pamf_analytic_identity_probe import design as d
    return dict(status='FEATURES_COMPLETE',count=1800,views=list(d.VIEWS),groups=d.GROUPS,readouts=['raw','restored'],
        TX_label_reads=0,F_weights_read=0,F_embeddings_logits_read=0,extra_restorer_predictions=0)


@pytest.mark.parametrize('key,value',[('count',168000),('TX_label_reads',1),('F_weights_read',1),('F_embeddings_logits_read',1),
    ('extra_restorer_predictions',1),('views',['clean'])])
def test_incomplete_or_wrong_permission_feature_header_keeps_labels_closed(tmp_path,monkeypatch,key,value):
    from experiments.pamf_analytic_identity_probe import runner
    from experiments.pamf_restoration.data import write_json
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');ids=np.arange(1800)
    monkeypatch.setattr(runner,'inputs',lambda args:(ids,None,None,{}))
    header=feature_header();header[key]=value;write_json(tmp_path/'feature_complete.json',header)
    def forbidden(*args,**kwargs):raise AssertionError('TX labels opened before all feature cells validated')
    with pytest.raises(ValueError):runner.score(SimpleNamespace(output=tmp_path),label_loader=forbidden)


def test_missing_feature_cell_keeps_labels_closed(tmp_path,monkeypatch):
    from experiments.pamf_analytic_identity_probe import runner
    from experiments.pamf_restoration.data import write_json
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');ids=np.arange(1800)
    monkeypatch.setattr(runner,'inputs',lambda args:(ids,None,None,{}))
    write_json(tmp_path/'feature_complete.json',feature_header());np.save(tmp_path/'physical_ids.npy',ids)
    np.savez(tmp_path/'analytic_features.npz',missing=np.zeros((1,1)))
    def forbidden(*args,**kwargs):raise AssertionError('TX labels opened before all feature cells validated')
    with pytest.raises(ValueError):runner.score(SimpleNamespace(output=tmp_path),label_loader=forbidden)


@pytest.mark.parametrize('commit,role',[('wrong','source'),('cd97744884b5e460926e48c8b7efb7d78c6da0f4','target')])
def test_wrong_parent_or_target_row_rejected_before_IQ_or_labels(tmp_path,commit,role):
    from experiments.pamf_analytic_identity_probe import design as d,runner
    from experiments.pamf_restoration.data import write_json
    parent=tmp_path/d.PARENT;row=parent/d.CANDIDATE;row.mkdir(parents=True)
    write_json(parent/'launch.json',dict(commit=commit))
    write_json(row/'complete.json',dict(status='ANALYZED',candidate=d.CANDIDATE,role=role,
        all_predictions_before_truth=True,training_TX_F_access=False))
    with pytest.raises(ValueError):runner.inputs(SimpleNamespace(runs_root=tmp_path))


def test_no_target_or_F_or_checkpoint_argument_is_exposed():
    from experiments.pamf_analytic_identity_probe.runner import PATH_ARGS
    assert set(PATH_ARGS)=={'runs_root','parent_action','source_cache','output'}
