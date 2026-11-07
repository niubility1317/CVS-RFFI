import torch
from torch import nn

from experiments.pamf_phase3_restoration.margin_uncertainty_source import (
    margin_values, input_margin_directions, projected_variances,
)


def test_local_margin_direction_is_exact_for_linear_frozen_reader():
    class Reader(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(8,6,bias=True)

        def features(self,x):
            return x.flatten(1)

        def classify_features(self,z):
            return self.head(z)

    reader=Reader().eval().requires_grad_(False)
    x=torch.randn(3,2,4,generator=torch.Generator().manual_seed(23))
    score=reader.classify_features(reader.features(x))
    order=score.argsort(dim=1,descending=True)
    logits,directions=input_margin_directions(reader,x,order)
    torch.testing.assert_close(logits,score)
    expected=(reader.head.weight[order[:,:1]]-reader.head.weight[order[:,1:]]).transpose(1,2)
    torch.testing.assert_close(directions,expected)
    delta=torch.randn(x.shape,generator=torch.Generator().manual_seed(15))
    actual=margin_values(reader.classify_features(reader.features(x+delta)),order)-margin_values(score,order)
    predicted=(directions*delta.flatten(1)[:,:,None]).sum(1)
    torch.testing.assert_close(actual,predicted)
    assert all(p.grad is None for p in reader.parameters())


def test_projected_variance_matches_precision_form():
    rng=torch.Generator().manual_seed(18)
    a=torch.randn(2,6,6,generator=rng,dtype=torch.float64)
    h=torch.randn(6,6,generator=rng,dtype=torch.float64)
    cov=h@h.T+torch.eye(6,dtype=torch.float64)
    var=torch.tensor([.01,.3],dtype=torch.float64)
    g=torch.randn(2,6,5,generator=rng,dtype=torch.float64)
    expected_post=torch.linalg.inv(torch.linalg.inv(cov)+a.transpose(1,2)@a/var[:,None,None])
    prior,post=projected_variances(a,cov,var,g)
    torch.testing.assert_close(prior,(g*(cov@g)).sum(1))
    torch.testing.assert_close(post,(g*(expected_post@g)).sum(1),atol=1e-10,rtol=1e-9)
    assert (post<prior).all()
