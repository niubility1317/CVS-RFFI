import pytest
import torch

from experiments.pamf_phase3_restoration.joint_prior import scratch_joint_prior
from experiments.pamf_phase3_restoration.objective import PairedObjective
from experiments.pamf_phase3_restoration.response_joint_prior import response_loss,scratch_response_joint_prior,selected_perturbation
from experiments.pamf_phase3_restoration import response_joint_prior_trial as trial
from experiments.pamf_phase3_restoration.source_fingerprint_response import perturbations


def bound_model(model):
    model.physical.bind_prior(torch.zeros(512),torch.eye(512)*.3)
    return model.eval()


def test_identical_initial_parameters_and_dense_deployment():
    torch.set_num_threads(2)
    control=bound_model(scratch_joint_prior(123))
    model=bound_model(scratch_response_joint_prior(123))
    assert set(control.state_dict())==set(model.state_dict())
    assert all(torch.equal(value,model.state_dict()[key]) for key,value in control.state_dict().items())
    torch.manual_seed(4);x=torch.randn(2,2,256)
    with torch.no_grad():
        torch.testing.assert_close(model(x),control(x),rtol=0,atol=0)
        efficient,aux=model.source_readout(x)
        torch.testing.assert_close(efficient,model(x),rtol=2e-5,atol=2e-6)
        assert aux['relative_solve_residual'].max()<1e-8


def test_fixed_directions_cover_each_record_and_keep_one_percent_energy():
    torch.manual_seed(8);x=torch.randn(8,2,256);index=torch.arange(8)
    all_directions=perturbations(x);choices=[]
    for epoch in range(1,5):
        delta,choice=selected_perturbation(x,index,epoch);choices.append(choice)
        torch.testing.assert_close(delta,all_directions[torch.arange(8),choice],rtol=0,atol=0)
        torch.testing.assert_close(delta.square().sum((1,2))/x.square().sum((1,2)),torch.full((8,),.0001),rtol=2e-6,atol=1e-10)
        assert torch.bincount(choice,minlength=4).tolist()==[2,2,2,2]
    assert all(set(v)==set(range(4)) for v in torch.stack(choices).T.tolist())


def test_response_loss_backpropagates_both_paired_outputs():
    base=torch.zeros(2,2,256,requires_grad=True);changed=torch.zeros(2,2,256,requires_grad=True)
    delta=torch.ones_like(base)*.01
    loss=response_loss(base,changed,delta)
    torch.testing.assert_close(loss,torch.ones(2));loss.mean().backward()
    assert base.grad.abs().sum()>0 and changed.grad.abs().sum()>0
    torch.testing.assert_close(base.grad,-changed.grad,rtol=0,atol=0)
    with pytest.raises(ValueError):response_loss(base,changed,torch.zeros_like(delta))


def test_response_only_gradient_reaches_both_prior_heads_and_not_physical():
    torch.set_num_threads(2);torch.manual_seed(18)
    model=bound_model(scratch_response_joint_prior(123))
    clean=torch.randn(2,2,256);received=clean+.1*torch.randn_like(clean)
    class FeaturesOnly:
        def features(self,x):return x.flatten(1)[:,:160]
    reader=FeaturesOnly();features=reader.features(clean).detach()
    kernel=torch.zeros(2,2,64);kernel[:,0,0]=1
    coefficients=torch.zeros(2,4,256);coefficients[:,0]=1
    trace=dict(received=received,kernel=kernel,coefficients=coefficients,variance=torch.full((2,),.01))
    priors=tuple(value[None].expand(5,*value.shape).clone() for value in (model.physical.prior_mean,model.physical.prior_covariance,model.physical.prior_modes))
    _,aux,loss,metrics=trial.response_readout(model,reader,received,clean,features,torch.ones(160),PairedObjective(),
        source_indices=torch.arange(2),epoch=0,priors=priors,assignment=torch.zeros(2,dtype=torch.long),source_trace=trace)
    assert torch.isfinite(loss).all() and model.response_smoke_report['actual_source_response_gradients']
    assert model.response_smoke_report['actual_training_deployment_posterior_agreement']
    loss.mean().backward()
    assert all(p.grad is None for p in model.physical.parameters())
    assert model.detail.mean_output.weight.grad.abs().sum()>0 and model.detail.output.weight.grad.abs().sum()>0
    assert aux['relative_solve_residual'].max()<1e-8
    torch.testing.assert_close(metrics['source_response_weighted_loss'],.01*metrics['source_response_relative_error'])


def test_wrong_source_pair_rejected_before_network_use():
    x=torch.randn(2,2,256)
    with pytest.raises(ValueError):
        trial.response_readout(None,None,x,x,None,None,None,source_indices=torch.arange(2),epoch=1,priors=None,assignment=None,source_trace=dict(received=x+1))
    with pytest.raises(ValueError):selected_perturbation(x,torch.arange(2),31)


def test_registered_single30_scope_and_original_dense_deployment():
    cfg=trial.configuration(trial.ARM)
    assert len(cfg['source_draws'])==30 and cfg['source_draws']==list(range(1,21))+list(range(1,11))
    assert cfg['training_posterior_calls_per_record']==2
    assert cfg['source_response_epsilon']==cfg['source_response_weight']==.01
    assert cfg['matched_control']=='20261107-phase1-Restore-JointPrior-E30-s2026092701-r01/TF_JointPrior'
    assert 'Original dense' in cfg['deployment_solver']
