import numpy as np
import pytest
import torch
from experiments.pamf_phase3_restoration.prior_physics import source_feature_variance,feature_errors,summarize,METHODS,SCENES,METRICS


def test_feature_geometry_requires_original_full_source_and_floors_constant_dimensions():
    total=torch.ones(160,dtype=torch.float64)*63000;squared=total.clone();squared[80:]*=2
    var,scale,floor=source_feature_variance(total,squared,63000)
    assert torch.equal(var[:80],torch.zeros(80,dtype=torch.float64))
    assert float(floor)==.005 and (scale>0).all()
    for n in [512,56700,27000]:
        with pytest.raises(ValueError,match='All original63000'):source_feature_variance(total,squared,n)


def test_feature_metric_is_per_packet_and_feature_permutation_invariant():
    torch.manual_seed(7);a=torch.randn(3,160);b=torch.randn_like(a);scale=torch.rand(160)+.01
    errors=feature_errors(a,b,scale)
    torch.testing.assert_close(errors[:1],feature_errors(a[:1],b[:1],scale),atol=0,rtol=0)
    p=torch.randperm(160)
    torch.testing.assert_close(errors[:,p],feature_errors(a[:,p],b[:,p],scale[p]),atol=0,rtol=0)
    with pytest.raises(ValueError):feature_errors(a,b[:1],scale)


def test_partial_source_panel_cannot_be_reported_complete():
    with pytest.raises(ValueError,match='All fixed'):summarize({'scene':np.asarray(SCENES)})
    records={'scene':np.resize(np.asarray(SCENES),10240)}
    for m in METHODS:
        for k in (*METRICS,*['variance_band'+str(i) for i in range(4)]):records[m+'__'+k]=np.ones(10240)
    result=summarize(records)
    assert sum(result[s]['count'] for s in SCENES)==10240
    assert result['ALL']['methods']['MSAT_oracle64']['feature_standardized_mse']==1.
