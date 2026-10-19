from types import SimpleNamespace
import numpy as np
import pytest
from tools.analyze_pamf_paired_identity import paired_readouts,correction_components,robust_views,analyze


def test_readout_repairs_and_clean_reference_errors_are_separate():
    truth=np.array([0,1,2,3,4,5]);clean=np.array([0,1,2,0,0,0])
    before=np.array([1,1,3,3,1,0]);after=np.array([0,2,2,0,4,5])
    r=paired_readouts(truth,clean,before,after)
    assert r['count']==6 and r['repaired']==4 and r['damaged']==2
    assert r['clean_correct_count']==3 and r['clean_correct_restored_correct']==2
    assert r['clean_wrong_count']==3 and r['clean_wrong_inherited_error']==1 and r['clean_wrong_restored_correct']==2
    assert r['repaired_clean_correct']==2 and r['repaired_clean_wrong']==2
    assert r['damaged_to_clean_wrong_decision']==1
    assert r['baseline_clean_agreement_percent']==pytest.approx(100/3)
    assert r['restored_clean_agreement_percent']==50


def test_empty_or_unavailable_conditional_denominators_are_null():
    r=paired_readouts(np.array([0]),np.array([0]),np.array([0]),np.array([0]))
    assert r['clean_wrong_count']==0 and r['clean_wrong_restored_accuracy_percent'] is None
    assert paired_readouts(np.array([],int),np.array([],int),np.array([],int),np.array([],int))['accuracy_percent'] is None


@pytest.mark.parametrize('bad',[np.array([1.2]),np.array([6]),np.array([-1]),np.array([[0]])])
def test_invalid_decision_arrays_rejected(bad):
    with pytest.raises(ValueError):paired_readouts(np.array([0]),np.array([0]),np.array([0]),bad)


def test_projection_preserves_signed_overshoot_and_orthogonal_energy():
    x=np.array([[[1.],[0.]],[[1.],[0.]],[[1.],[0.]]]);r=np.zeros_like(x)
    y=np.array([[[.5],[0.]],[[1.5],[0.]],[[1.],[2.]]])
    values=correction_components(x,r,y)
    np.testing.assert_allclose(values['projection_fraction'],[.5,1.5,1.])
    np.testing.assert_allclose(values['orthogonal_correction_nmse'],[0.,0.,4.])
    np.testing.assert_allclose(values['input_nmse'],[1.,1.,1.])
    np.testing.assert_allclose(values['restored_nmse'],[.25,.25,4.])


def test_no_input_error_has_no_projection_and_reports_introduced_error():
    x=np.ones((2,2,3));y=x+.1;values=correction_components(x,x,y)
    assert not values['projection_defined'].any()
    np.testing.assert_allclose(values['introduced_error_nmse'],.01)
    np.testing.assert_allclose(values['restored_nmse'],.01)


def test_correction_shape_and_nonfinite_rejected():
    x=np.ones((2,2,3))
    with pytest.raises(ValueError):correction_components(x,x[:1],x)
    with pytest.raises(ValueError):correction_components(x,x,np.full_like(x,np.nan))


def test_view_robustness_is_a_diagnostic_count_not_reassignment():
    truth=np.array([0,1,2]);raw=np.tile(np.array([0,0,2]),(6,1));restored=np.tile(truth,(6,1));restored[0,0]=1
    original=restored.copy();out=robust_views(truth,raw,restored)
    assert out['baseline_all_six_correct']==2 and out['restored_all_six_correct']==2
    assert out['baseline_number_correct_histogram']==[1,0,0,0,0,0,2]
    assert out['restored_number_correct_histogram']==[0,0,0,0,0,1,2]
    np.testing.assert_array_equal(original,restored)


def test_missing_completed_score_keeps_truth_closed(tmp_path):
    opened=[];args=SimpleNamespace(row=tmp_path,baseline=tmp_path,reference=tmp_path,truth=tmp_path/'truth.json')
    def truth_loader(path):opened.append(path);raise AssertionError('truth must stay closed')
    with pytest.raises((FileNotFoundError,ValueError)):analyze(args,truth_loader=truth_loader)
    assert not opened


def test_complete_cpu_analysis_joins_same_physical_samples_after_all_logits(tmp_path,monkeypatch):
    import json
    from tools.analyze_pamf_paired_identity import VIEWS,F_RUN
    row=tmp_path/'row';(row/'prediction').mkdir(parents=True)
    baseline=tmp_path/'baseline';baseline.mkdir();reference=tmp_path/'reference';reference.mkdir()
    n=168000;ids=np.arange(n);y=ids%6;logits=np.eye(6,dtype=np.float32)[y]
    score=dict(status='ANALYZED',role='target',count=n,all_predictions_before_truth=True,original_default_head=True,F_run=F_RUN,F_seed=2026092701,
        records=[dict(view=v,dimension='overall',readout='restored',accuracy=100.,repaired=0,damaged=0) for v in VIEWS])
    (row/'summary.json').write_text(json.dumps(score),encoding='utf-8')
    (row/'prediction/complete.json').write_text(json.dumps(dict(status='PREDICTIONS_COMPLETE',views=list(VIEWS))),encoding='utf-8')
    (baseline/'complete.json').write_text(json.dumps(dict(status='PREDICTIONS_COMPLETE',views=list(VIEWS),count=n,F_run=F_RUN,F_seed=2026092701,original_default_head=True)),encoding='utf-8')
    (reference/'manifest.json').write_text(json.dumps(dict(status='FORMAL_REFERENCE_READY',views=list(VIEWS),count=n)),encoding='utf-8')
    logit_reads=[];clean=np.ones((n,2,1));received=clean*.5
    def load(path,**kwargs):
        path=str(path)
        if path.endswith('physical_ids.npy'):return ids
        if path.endswith('.logits.npy'):logit_reads.append(path);return logits
        if path.endswith('clean.npy') or path.endswith('.restored.iq.npy'):return clean
        return received
    monkeypatch.setattr(np,'load',load)
    class Truth:
        def __getitem__(self,key):
            i=int(key);return dict(label=i%6,receiver=i%7,day=i%4)
    def truth_loader(path):assert len(logit_reads)==14;return Truth()
    args=SimpleNamespace(row=row,baseline=baseline,reference=reference,truth=tmp_path/'truth.json')
    result=analyze(args,truth_loader=truth_loader)
    assert len(result['records'])==126 and result['extra_model_predictions']==result['parameters_fitted']==0
    assert result['all_six_robustness']['restored_all_six_correct']==n
    low=next(r for r in result['records'] if r['view']=='practical_low_urban' and r['dimension']=='overall')
    assert low['correction']['mean_projection_fraction']==1 and low['correction']['orthogonal_correction_nmse']==0
    assert low['correction']['restored_nmse']==0 and low['accuracy_percent']==100
