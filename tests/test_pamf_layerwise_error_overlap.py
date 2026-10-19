import numpy as np
import pytest
from experiments.pamf_layerwise_refine.error_overlap import overlap


def test_source_error_overlap_and_conditional_counts():
    labels=np.array([0,1,2,3,4,5]);base=np.array([0,0,2,0,4,0]);new=np.array([0,1,0,0,4,5])
    result=overlap(labels,base,new)
    assert result['baseline_wrong']==3 and result['candidate_wrong']==2
    assert result['both_wrong']==1 and result['both_correct']==2
    assert result['corrected_vs_baseline']==2 and result['damaged_vs_baseline']==1
    assert result['net_corrected']==1 and result['decision_disagreements']==3
    assert result['baseline_error_overlap_fraction']==pytest.approx(1/3)
    assert result['truth_oracle_coverage_upper_bound_percent']==pytest.approx(100*5/6)
    subset=overlap(labels,base,new,np.array([True,False,True,True,True,False]))
    assert subset['count']==4 and subset['corrected_vs_baseline']==0 and subset['damaged_vs_baseline']==1


def test_invalid_source_vectors_and_mask_rejected():
    labels=np.array([0,1,2])
    with pytest.raises(ValueError):overlap(labels,np.array([0,1]),labels)
    with pytest.raises(ValueError):overlap(labels,labels,np.array([0,1,6]))
    with pytest.raises(ValueError):overlap(labels,labels,labels,np.ones(3,int))
    empty=overlap(labels,labels,labels,np.zeros(3,bool))
    assert empty['truth_oracle_coverage_upper_bound_percent'] is None
    assert empty['baseline_error_overlap_fraction'] is None
