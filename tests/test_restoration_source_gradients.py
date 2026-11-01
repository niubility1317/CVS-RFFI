import numpy as np
import pytest
import torch
from experiments.pamf_phase3_restoration.source_gradients import geometry,parameter_group


def test_geometry_exposes_objective_conflict_and_global_clip_coupling():
    vectors=np.asarray([[1.,0.],[0.,0.],[-2.,1.]])
    gram=vectors@vectors.T;weak=geometry(gram,.1);strong=geometry(gram,1.)
    assert weak['main_dot_total']>0 and strong['main_dot_total']<0
    assert weak['main_latent_cosine']==pytest.approx(-2/np.sqrt(5))
    assert strong['clip1_factor_if_clipped_alone']==pytest.approx(1/(np.sqrt(2)+1e-6))
    assert strong['physical_dot_total']==pytest.approx(-1.)
    assert parameter_group('feature_regressor.1.weight')=='latent_head'
    assert parameter_group('position')=='encoder'
    with pytest.raises(ValueError):parameter_group('unrecognized.weight')


def test_zero_auxiliary_gradient_is_explicitly_undefined_cosine():
    q=geometry(np.diag([4.,1.,0.]),1.)
    assert q['main_latent_cosine'] is None and q['latent_unit_gradient_norm']==0.
    assert q['total_gradient_norm']==pytest.approx(np.sqrt(5))
    with pytest.raises(ValueError):geometry(np.ones((2,2)),1.)
    with pytest.raises(ValueError):geometry(np.diag([np.nan,1.,1.]),1.)


def test_autograd_derivatives_do_not_accumulate_or_update_parameters():
    parameter=torch.nn.Parameter(torch.tensor([2.,3.]))
    before=parameter.detach().clone();first=parameter.square().mean();second=(parameter-1).square().mean()
    gradients=[torch.autograd.grad(first,parameter,retain_graph=True)[0],torch.autograd.grad(second,parameter)[0]]
    assert parameter.grad is None and torch.equal(parameter,before)
    matrix=torch.stack(gradients);assert torch.allclose(matrix@matrix.T,torch.tensor([[13.,8.],[8.,5.]]))
