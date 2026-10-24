import math
import pytest
from experiments.pamf_ssl_research.source_gradient_trajectory import (
    GROUPS, analyze_segment, distribution, group_observation, metric)


def values(**kw):
    return dict(dict(L_norm=1.,U_norm=2.,applied_U_norm=1.,dot_before=-1.,dot_after=0.,
                     conflict=1,capped=1,tiny_L=0,conflicts=1,caps=1),**kw)


def test_logged_weighted_gradient_retention_and_orthogonality():
    v=group_observation(values())
    assert v['U_to_L']==2 and v['applied_U_to_L']==1 and v['retained_U_norm']==.5
    assert v['cosine_before']==-.5 and v['cosine_after']==0


def test_zero_norm_ratio_is_undefined_not_zero():
    v=group_observation(values(L_norm=0.,U_norm=0.,applied_U_norm=0.,dot_before=0.,conflict=0,capped=0,tiny_L=1))
    assert v['retained_U_norm'] is None and v['cosine_before'] is None and v['U_to_L'] is None


@pytest.mark.parametrize('kw',[{'applied_U_norm':1.1},{'dot_before':3.},{'dot_after':-.1},{'tiny_L':1},{'L_norm':math.nan}])
def test_impossible_gradient_records_rejected(kw):
    with pytest.raises(ValueError):group_observation(values(**kw))


def test_missing_or_ambiguous_metric_is_not_silently_selected():
    with pytest.raises(ValueError):metric({},'HPL_weight')
    with pytest.raises(ValueError):metric({'train/HPL_weight':6,'other/HPL_weight':6},'HPL_weight')


def test_quantiles_and_undefined_values_have_explicit_denominators():
    v=distribution([None,0.,1.,2.,3.])
    assert v['count']==4 and v['median']==1.5 and v['p10']==pytest.approx(.3) and v['p90']==pytest.approx(2.7)
    assert distribution([None])['mean'] is None


def native_records():
    for step in range(1,2221):
        epoch,rem=divmod(step,222);seen=epoch*56700+rem*256
        m=dict(HPL_weight=6,HPL_GP_calls=step,HPL_calls=step,HPL_seen=seen,HPL_accepted=seen,
               HPL_student_views=6*seen,HPL_EMA_updates=0,T1_L_packets=768)
        for group in GROUPS:
            for k,v in values(conflicts=step,caps=step).items():m['train/HPL_GP_'+group+'_'+k]=v
        yield dict(step=step,metrics=m)


def test_complete_segment_has_exact_epoch_and_group_denominators():
    a=analyze_segment(native_records(),2220,20)
    assert a['seen']==567000 and a['accepted']==567000 and a['records']==2220 and a['final_epoch']==30
    assert len(a['epochs'])==10 and all(e['records']==222 for e in a['epochs'])
    assert a['groups']['t2']['counts']==dict(conflict=2220,capped=2220,tiny_L=0)
    assert a['whole_parameter_gradient']['retained_weighted_U_norm']['mean']==.5


def test_partial_segment_cannot_be_presented_as_complete():
    with pytest.raises(ValueError):analyze_segment(iter(list(native_records())[:-1]),2220,20)


def test_U_truth_is_not_read_as_a_source_metric():
    record=next(native_records());record['metrics']['train/pseudo_truth_available']=1
    with pytest.raises(ValueError):analyze_segment([record],2220,20)


def cap2_records():
    for record in native_records():
        metrics=record['metrics'];metrics['train/HPL_GP_U_to_L_norm_cap']=2.
        for group in GROUPS:
            metrics['train/HPL_GP_'+group+'_U_norm']=3.
            metrics['train/HPL_GP_'+group+'_applied_U_norm']=2.
        yield record


def test_cap2_requires_declared_ratio_and_counts_actual_steps_above_old_cap():
    result=analyze_segment(cap2_records(),2220,20,norm_cap=2.)
    assert result['U_to_L_norm_cap']==2.
    assert result['applied_U_above_L_counts']=={g:2220 for g in GROUPS}
    assert result['groups']['t2']['distributions']['applied_U_to_L']['mean']==2.
    assert result['whole_parameter_gradient']['retained_weighted_U_norm']['mean']==pytest.approx(2/3)


def test_default_cap1_still_rejects_cap2_records():
    with pytest.raises(ValueError,match='norm cap'):analyze_segment(cap2_records(),2220,20)
    with pytest.raises(ValueError):group_observation(values(applied_U_norm=2.))


def test_cap2_missing_scalar_metric_cannot_be_assumed_from_group_magnitude():
    with pytest.raises(ValueError,match='HPL_GP_U_to_L_norm_cap'):
        analyze_segment(native_records(),2220,20,norm_cap=2.)


def test_cap2_bound_and_negative_projection_remain_checked():
    with pytest.raises(ValueError):group_observation(values(U_norm=3.,applied_U_norm=2.1),norm_cap=2.)
    with pytest.raises(ValueError):group_observation(values(U_norm=3.,applied_U_norm=2.,dot_after=-.1),norm_cap=2.)


@pytest.mark.parametrize('ratio',[0.,1.5,math.inf,True])
def test_source_analysis_does_not_add_unregistered_caps(ratio):
    with pytest.raises(ValueError,match='fixed source cap1/cap2'):
        group_observation(values(),norm_cap=ratio)
