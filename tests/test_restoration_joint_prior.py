import torch

from experiments.pamf_phase3_restoration.joint_prior import bounded_prior_mean,scratch_joint_prior
from experiments.pamf_phase3_restoration.coupled_covariance import scratch_coupled_covariance,normalize_modes
from experiments.pamf_phase3_restoration.signed_covariance import source_modes


def test_mean_update_is_source_scaled_bounded_and_has_nonzero_zero_point_gradient():
    base=torch.zeros(2,512,dtype=torch.float64)
    diagonal=torch.stack((torch.ones(512),4*torch.ones(512))).double()
    raw=torch.zeros(2,2,256,dtype=torch.float64,requires_grad=True)
    mean,delta,standardized=bounded_prior_mean(base,diagonal,raw)
    assert torch.equal(mean,base) and delta.count_nonzero()==standardized.count_nonzero()==0
    mean.sum().backward()
    torch.testing.assert_close(raw.grad,diagonal.sqrt().reshape_as(raw),rtol=0,atol=0)
    extreme=torch.full_like(raw,1e5);extreme[1]*=-1
    mean,delta,standardized=bounded_prior_mean(base,diagonal,extreme)
    assert (delta.abs()<diagonal.sqrt().reshape_as(raw)).all()
    assert (delta[0]>0).all() and (delta[1]<0).all() and (standardized.abs()<1).all()


def test_joint_scratch_preserves_every_existing_coupled_parameter_and_zero_mean_start():
    old=scratch_coupled_covariance(2026092701);new=scratch_joint_prior(2026092701)
    original=old.state_dict();current=new.state_dict()
    assert all(torch.equal(value,current[key]) for key,value in original.items())
    assert set(current)-set(original)=={'detail.mean_output.weight','detail.mean_output.bias'}
    assert not new.detail.mean_output.weight.any() and not new.detail.mean_output.bias.any()
    mean=torch.linspace(-.2,.2,512);covariance=.04*torch.eye(512)
    old.physical.bind_prior(mean,covariance);new.physical.bind_prior(mean,covariance)
    observed=torch.randn(2,2,256,generator=torch.Generator().manual_seed(49))
    with torch.no_grad():
        expected=old(observed);actual,aux=new(observed,return_aux=True)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert not aux['mean_update'].any()


def test_actual_noisy_objective_updates_mean_and_covariance_without_physical_gradients():
    from experiments.pamf_phase3_restoration.joint_prior_trial import joint_readout,mean_smoke_gradient_check
    from experiments.pamf_phase3_restoration.covariance_detail import detail_objective
    from experiments.pamf_phase3_restoration.objective import PairedObjective
    model=scratch_joint_prior(2026092701)
    model.physical.bind_prior(torch.zeros(512),.04*torch.eye(512))
    observed=torch.randn(1,2,256,generator=torch.Generator().manual_seed(91));clean=.7*observed
    class Reader:
        def features(self,x):return x.flatten(1)[:,:160]
    reader=Reader();target=reader.features(clean);scale=torch.ones(160);objective=PairedObjective()
    covariance=.08*torch.eye(512);modes=normalize_modes(covariance,source_modes(covariance))
    priors=(.1*torch.ones(5,512),covariance[None].expand(5,-1,-1),modes[None].expand(5,-1,-1))
    before={key:value.clone() for key,value in model.state_dict().items()}
    restored,aux,loss,metrics=joint_readout(model,reader,observed,clean,target,scale,objective,
        source_indices=torch.tensor([0]),epoch=2,priors=priors,assignment=torch.tensor([0]))
    expected,_=detail_objective(reader,restored,clean,target,scale,objective)
    torch.testing.assert_close(loss,expected,rtol=0,atol=0)
    assert metrics['receiver_mismatch_prior_fraction'].item()==1 and not aux['mean_update'].any()
    loss.mean().backward()
    assert all(p.grad is None for p in model.physical.parameters())
    assert model.detail.output.weight.grad.abs().sum()>0
    assert mean_smoke_gradient_check(model)['actual_mean_feature_gradient']
    assert all(torch.equal(value,model.state_dict()[key]) for key,value in before.items())


def test_mean_changes_deployed_posterior_and_keeps_fixed_prior_buffers():
    model=scratch_joint_prior(2026092701)
    model.physical.bind_prior(torch.zeros(512),.01*torch.eye(512))
    observed=torch.randn(1,2,256,generator=torch.Generator().manual_seed(53))
    before={key:value.clone() for key,value in model.physical.state_dict().items() if key.startswith('prior_')}
    with torch.no_grad():
        original=model(observed)
        model.detail.mean_output.bias.fill_(.5)
        changed,aux=model(observed,return_aux=True)
    assert not torch.equal(original,changed) and torch.isfinite(changed).all()
    assert (aux['mean_standardized_update'].abs()<1).all() and aux['relative_solve_residual'].max()<1e-8
    assert all(torch.equal(value,model.physical.state_dict()[key]) for key,value in before.items())


def test_fixed_budget_and_dependency_never_consult_score(tmp_path):
    import json
    from experiments.pamf_phase3_restoration.joint_prior_trial import ARM,configuration,previous_gpu_finished
    from experiments.pamf_phase3_restoration import coupled_covariance_trial as control
    cfg=configuration(ARM)
    assert (cfg['epochs'],cfg['updates'],cfg['training_stages'])==(30,7410,1)
    assert cfg['matched_control']==control.RUN+'/'+control.ARM and cfg['source_F0_guidance_readout']=='Actual restored IQ with realized source noise present.'
    folder=tmp_path/'runs'/control.RUN;folder.mkdir(parents=True)
    (folder/'results.json').write_text('unreadable score deliberately ignored')
    assert not previous_gpu_finished(tmp_path)
    (folder/f'terminal-{control.ARM}-train.json').write_text(json.dumps(dict(exit_code=0)))
    assert not previous_gpu_finished(tmp_path)
    (folder/f'terminal-{control.ARM}-cross_both.json').write_text(json.dumps(dict(exit_code=0)))
    assert previous_gpu_finished(tmp_path)
