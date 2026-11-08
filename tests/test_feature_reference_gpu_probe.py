import pytest
import torch

from experiments.pamf_phase3_restoration.feature_reference_gpu_probe import compare_named, compare_tensors


def test_elementwise_comparison_records_tolerance_and_count():
    a=torch.tensor([0.,1.,10.]);b=torch.tensor([0.,1.000001,10.01])
    s=compare_tensors(a,b)
    assert s['notclose']==1 and s['elements']==3 and s['max_abs']==pytest.approx(.01,abs=1e-6)
    with pytest.raises(ValueError):compare_tensors(a,b[:2])
    with pytest.raises(ValueError):compare_tensors(a,torch.tensor([0.,float('nan'),0.]))


def test_named_comparison_preserves_gradient_presence():
    a={'one':torch.tensor([1.,2.]),'unused':None}
    b={'one':torch.tensor([1.,2.]),'unused':None}
    assert compare_named(a,b)=={'max_abs':0.,'notclose':0,'elements':2,'tensors':1}
    with pytest.raises(ValueError):compare_named(a,{'one':torch.tensor([1.,2.]),'unused':torch.tensor([0.])})
    with pytest.raises(ValueError):compare_named(a,{'other':torch.tensor([1.,2.])})
