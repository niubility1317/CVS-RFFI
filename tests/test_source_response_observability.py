import numpy as np
import pytest
import torch
from experiments.pamf_phase3_restoration.source_response_observability import aggregate,proxy


def test_source_fisher_proxy_and_variance_scale():
    action=torch.eye(512,dtype=torch.float64)[None]
    delta=torch.ones((1,4,2,256),dtype=torch.float64)
    torch.testing.assert_close(proxy(action,delta,torch.tensor([2.])),torch.full((1,4),256.,dtype=torch.float64))
    with pytest.raises(ValueError):proxy(action,delta,torch.tensor([0.]))


def test_bins_cover_boundaries_and_use_energy_weighted_error():
    values=np.array([0.,1.,8.99,9.,25.]);stats=np.array([[1.,1.,1.,1.],[1.,1.,2.,3.],[3.,1.,3.,9.],[2.,1.,1.,2.],[4.,1.,8.,4.]])
    result=aggregate(values,stats,stats)
    assert [row['count'] for row in result]==[1,2,1,1]
    assert result[1]['iq_projection_gain']==1.25
    assert result[1]['feature_relative_error']==3.
    empty=aggregate(np.array([]),np.zeros((0,4)),np.zeros((0,4)))
    assert all(row['iq_projection_gain'] is None for row in empty)
