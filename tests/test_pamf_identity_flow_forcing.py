import pytest
import torch

from experiments.pamf_identity_flow.forcing_probe import teacher_forced_metrics, noise_metrics


class ConstantFlow(torch.nn.Module):
    def __init__(self,value):super().__init__();self.value=value
    def velocity(self,state,received,time):return self.value.expand_as(state)
    def integrate(self,received,noise,*,noise_scale,steps):return received+noise_scale*noise+self.value


def test_teacher_forced_path_recovers_known_constant_velocity():
    received=torch.zeros(3,2,256);clean=torch.ones_like(received);noise=torch.zeros_like(received)
    result=teacher_forced_metrics(ConstantFlow(torch.tensor(1.)),clean,received,noise,.5)
    assert result['velocity_NMSE_mean']==pytest.approx(0.)
    assert result['projected_endpoint_NMSE_mean']==pytest.approx(0.)


def test_antithetic_noise_cancels_for_constant_velocity():
    received=torch.zeros(3,2,256);clean=torch.ones_like(received);noise=torch.ones(1,2,256)
    result=noise_metrics(ConstantFlow(torch.tensor(1.)),clean,received,noise,.02)
    assert result['mean_path_NMSE']==pytest.approx(0.)
    assert result['opposite_path_difference_MSE']==pytest.approx(.0016,abs=1e-8)
