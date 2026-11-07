import pytest
import torch

from experiments.pamf_phase3_restoration import response_joint_prior_dense_trial as dense
from experiments.pamf_phase3_restoration import response_reliability_trial as candidate


def test_weight_matches_action_noise_proxy_and_is_detached():
    delta=torch.zeros(3,2,256,requires_grad=True)
    with torch.no_grad():delta[:,0,0]=torch.tensor([0.,3.,6.])
    action=torch.eye(512,dtype=torch.float64).repeat(3,1,1).requires_grad_()
    variance=torch.tensor([1.,1.,1.],requires_grad=True)
    weight,kappa=candidate.response_weight(delta,action,variance)
    torch.testing.assert_close(kappa,torch.tensor([0.,9.,36.],dtype=torch.float64))
    torch.testing.assert_close(weight,torch.tensor([0.,.5,.8],dtype=torch.float64))
    assert not weight.requires_grad and not kappa.requires_grad
    with pytest.raises(ValueError,match='positive'):candidate.response_weight(delta,action,torch.zeros(3))


def test_candidate_preserves_dense_control_initialization_and_budget():
    before=dense.factory(dense.ARM);after=candidate.factory(candidate.ARM)
    assert before.state_dict().keys()==after.state_dict().keys()
    assert all(torch.equal(value,after.state_dict()[key]) for key,value in before.state_dict().items())
    old=dense.configuration(dense.ARM);new=candidate.configuration(candidate.ARM)
    for key in ('epochs','updates','batch','source_draws','source_response_epsilon','source_response_directions','training_solver'):
        assert old[key]==new[key]
    assert new['source_action_deployment_input'] is False
    assert new['source_noise_deployment_input'] is False


def test_resource_admission_requires_twelve_gib(monkeypatch):
    monkeypatch.setattr(candidate,'recovery_gpu_choice',lambda **kwargs:[
        {'GPU':0,'free_memory_MB':12287},{'GPU':2,'free_memory_MB':12288}])
    assert candidate.gpu_choice()==[{'GPU':2,'free_memory_MB':12288}]
