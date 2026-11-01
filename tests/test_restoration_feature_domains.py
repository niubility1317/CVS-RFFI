import numpy as np
import pytest
from experiments.pamf_phase3_restoration.feature_domains import domain_geometry,check_handles


def test_exact_balanced_RX_day_and_interaction_decomposition_with_constant_dimension():
    rx=np.repeat([-1,1],4);day=np.tile(np.repeat([-1,1],2),2);noise=np.tile([-1,1],4)
    z=np.column_stack((2*rx,3*day,rx*day,noise,np.ones(8)))
    result=domain_geometry(z,rx,day,np.ones(5),np.arange(5).reshape(1,5))
    np.testing.assert_allclose(result['source_total_variance'],[4,9,1,1,0],atol=0,rtol=0)
    np.testing.assert_allclose(result['source_between_variance']['RX'],[4,0,0,0,0],atol=0,rtol=0)
    np.testing.assert_allclose(result['source_between_variance']['day'],[0,9,0,0,0],atol=0,rtol=0)
    np.testing.assert_allclose(result['source_between_variance']['RX_and_interaction_added_given_day'],[4,0,1,0,0],atol=0,rtol=0)
    assert result['per_dimension_fraction']['RX'][4] is None and result['zero_variance_dimensions']==[4]
    assert result['summaries']['ALL']['RX']['raw_energy_fraction']==4/15


def test_unbalanced_conditional_increment_is_exact_but_not_a_pure_main_effect():
    rx=np.asarray([0,0,0,1,1]);day=np.asarray([0,0,1,0,1]);z=np.asarray([[0.,0.],[1,0],[4,0],[8,0],[16,0]])
    result=domain_geometry(z,rx,day,np.asarray([2.,1.]),np.arange(2).reshape(1,2))
    total=np.var(z[:,0]);joint=sum((z[(rx==r)&(day==d),0].mean()-z[:,0].mean())**2*int(((rx==r)&(day==d)).sum()) for r,d in set(zip(rx,day)))/5
    byday=sum((z[day==d,0].mean()-z[:,0].mean())**2*int((day==d).sum()) for d in set(day))/5
    np.testing.assert_allclose(result['source_between_variance']['RX_and_interaction_added_given_day'][0],joint-byday)
    assert result['conditional_increments_include_interactions'] and result['marginal_RX_day_fractions_must_not_be_added']
    assert result['summaries']['ALL']['RX_day_joint']['raw_energy_fraction']==pytest.approx(joint/total)


def test_invalid_or_wrong_role_features_are_rejected_before_analysis():
    with pytest.raises(ValueError):check_handles({'status':'VALIDATED_ONCE','count':63000},{},{})
    with pytest.raises(ValueError):domain_geometry(np.ones((8,2)),np.zeros(8),np.zeros(8),np.ones(2),np.arange(2).reshape(1,2))
    with pytest.raises(ValueError):domain_geometry(np.asarray([[np.nan,0],[1,2]]),np.zeros(2),np.zeros(2),np.ones(2),np.arange(2).reshape(1,2))
    with pytest.raises(ValueError):domain_geometry(np.ones((8,2)),np.zeros(7),np.zeros(8),np.ones(2),np.arange(2).reshape(1,2))
