import torch


def test_scratch_physical_initialization_and_zero_detail_match_original():
    from experiments.pamf_phase3_restoration.covariance_detail import scratch_covariance_detail
    from experiments.pamf_phase3_restoration.covariance_model import scratch_covariance
    original=scratch_covariance(71);model=scratch_covariance_detail(71)
    for key,value in original.state_dict().items():torch.testing.assert_close(value,model.physical.state_dict()[key],rtol=0,atol=0)
    x=torch.randn(2,2,256)
    torch.testing.assert_close(model(x),original(x),rtol=0,atol=0)


def test_IQ_and_feature_gradients_cannot_modify_physical_parameters():
    from experiments.pamf_phase3_restoration.covariance_detail import scratch_covariance_detail,detail_objective
    from experiments.pamf_phase3_restoration.covariance_model import source_action_loss
    from experiments.pamf_phase3_restoration.objective import PairedObjective
    class Reader(torch.nn.Module):
        def __init__(self):super().__init__();self.layer=torch.nn.Linear(512,160)
        def features(self,x):return self.layer(x.flatten(1))
    torch.manual_seed(41);model=scratch_covariance_detail(8);F=Reader().eval().requires_grad_(False)
    x=torch.randn(2,2,256);clean=torch.randn_like(x);target=F.features(clean).detach().requires_grad_(True)
    output,aux=model(x,return_aux=True)
    loss,metrics=detail_objective(F,output,clean,target,torch.ones(160),PairedObjective())
    loss.mean().backward()
    assert all(p.grad is None for p in model.physical.parameters())
    assert model.detail.output.weight.grad is not None and model.detail.output.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in F.parameters()) and target.grad is None
    assert metrics['iq_nmse'].shape==(2,) and metrics['feature_standardized_error'].shape==(2,)
    model.zero_grad(set_to_none=True)
    physical,_=source_action_loss(aux,torch.randn(2,2,64),torch.randn(2,4,256),torch.tensor([.01,.2]))
    physical.mean().backward()
    assert all(p.grad is None for p in model.detail.parameters())
    for head in [model.physical.kernel_head,model.physical.coefficient_head,model.physical.noise_head]:
        assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all() and head.weight.grad.abs().sum()>0


def test_single_stage_state_reload_and_per_record_inference():
    from experiments.pamf_phase3_restoration.covariance_detail import scratch_covariance_detail
    model=scratch_covariance_detail(9).eval();copy=scratch_covariance_detail(8).eval()
    with torch.no_grad():model.detail.output.weight.normal_(std=.001)
    copy.load_state_dict(model.state_dict(),strict=True);x=torch.randn(2,2,256)
    with torch.inference_mode():
        torch.testing.assert_close(copy(x),model(x),rtol=0,atol=0)
        torch.testing.assert_close(copy(x),torch.cat([copy(row[None]) for row in x]),rtol=2e-5,atol=2e-6)


def test_detail_gradient_clipping_cannot_rescale_physical_update():
    from experiments.pamf_phase3_restoration.covariance_detail import scratch_covariance_detail,clip_branch_gradients
    model=scratch_covariance_detail(3);physical=model.physical.noise_head.bias;detail=model.detail.output.bias
    physical.grad=torch.full_like(physical,.2);detail.grad=torch.full_like(detail,100.)
    before=physical.grad.clone();norms=clip_branch_gradients(model)
    torch.testing.assert_close(physical.grad,before,rtol=0,atol=0)
    assert norms['detail']>100 and detail.grad.norm()<=1


def test_trial_full_exposure_budget_and_frozen_training_roles():
    from experiments.pamf_phase3_restoration.covariance_detail_trial import configuration,ARM
    c=configuration(ARM)
    assert c['training_stages']==1 and c['training_F0_forward_rows']==63000*30
    assert c['training_posterior_calls']==30*sum((min(256,63000-i)+31)//32 for i in range(0,63000,256))
    assert c['source_draws']==list(range(1,21))+list(range(1,11))
    assert c['F0_guidance_for_G_updates'] and c['detail_full_source_exposure']
    assert c['new_G_initialization_checkpoints']==0 and not c['unit_power_projection']
