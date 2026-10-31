import pytest
import torch

from experiments.pamf_identity_flow.trajectory_probe import path_metrics


class Reader(torch.nn.Module):
    def features(self,x):return torch.cat((x.mean(2),x.square().mean(2)),dim=1)
    def classify_features(self,x):return x[:,:2]


class ConstantFlow(torch.nn.Module):
    def integrate(self,received,noise,*,noise_scale,steps):
        return received+noise_scale*noise+.25


def test_antithetic_mean_and_metrics_do_not_use_labels():
    x=torch.randn(4,2,256);teacher=Reader().eval().requires_grad_(False)
    value=path_metrics(ConstantFlow(),teacher,x,x-.25,torch.ones(1,2,256),steps=4)
    assert value['IQ_NMSE_mean']==pytest.approx(0.,abs=1e-12)
    assert value['feature_cosine_mean']==pytest.approx(1.,abs=1e-6)
    assert value['F0_argmax_agreement']==1.
    assert value['count']==4
