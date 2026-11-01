import torch
import pytest
from experiments.pamf_phase3_restoration.physical import scratch_physical,action_supervision
from experiments.pamf_phase3_restoration.physical_operators import Action,soft_map,real_dot
from experiments.pamf_phase3_restoration.multiscale import scratch_multiscale


def test_real_linear_reflect_adjoint_and_differentiable_solver():
    torch.set_num_threads(2);torch.manual_seed(10)
    h=torch.randn(2,5,dtype=torch.complex64,requires_grad=True)
    main=torch.randn(2,16,dtype=torch.complex64,requires_grad=True)
    image=torch.randn(2,16,dtype=torch.complex64,requires_grad=True)
    action=Action(h,main,image);x=torch.randn(2,16,dtype=torch.complex64);y=torch.randn_like(x)
    torch.testing.assert_close(real_dot(action(x),y),real_dot(x,action.adjoint(y)),rtol=2e-5,atol=2e-5)
    z=soft_map(action,y,x,torch.ones(2)*.05,.01,steps=4);z.abs().square().mean().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum()>0 for t in (h,main,image))


def test_scratch_near_identity_is_per_record_and_preserves_rng_and_old_default():
    torch.set_num_threads(2);torch.manual_seed(8);before=torch.get_rng_state().clone()
    g=scratch_physical(2026092701).eval();assert torch.equal(before,torch.get_rng_state())
    x=torch.randn(2,2,256)
    with torch.no_grad():a=g(x);b=g(x[:1])
    torch.testing.assert_close(a,x,rtol=2e-5,atol=2e-5);torch.testing.assert_close(a[:1],b,rtol=2e-5,atol=2e-5)
    assert g.priors[0].output.weight.data_ptr()!=g.priors[1].output.weight.data_ptr()
    old=scratch_multiscale(2026092701);assert sum(p.numel() for p in old.parameters())==4249922
    with torch.no_grad():assert torch.equal(old(x),x)


def test_joint_source_action_and_IQ_losses_reach_all_output_heads():
    torch.set_num_threads(2);g=scratch_physical(2026092701);clean=torch.randn(2,2,256);received=.7*clean.roll(3,-1)+.02*torch.randn_like(clean)
    value,aux=g(received,return_aux=True)
    kernel=torch.zeros(2,2,64);kernel[:,0,3]=1
    coefficients=torch.zeros(2,4,256);coefficients[:,0]=.7
    loss,terms=action_supervision(aux,clean,kernel,coefficients,torch.ones(2)*.002)
    ((value-clean).square().mean()+loss.mean()).backward()
    heads=[g.estimator.kernel_head,g.estimator.coefficient_head,g.estimator.noise_head,*[p.output for p in g.priors]]
    assert all(h.weight.grad is not None and torch.isfinite(h.weight.grad).all() and h.weight.grad.abs().sum()>0 for h in heads)
    assert all(torch.isfinite(v).all() for v in terms.values())
    with pytest.raises(ValueError):action_supervision(aux,clean,kernel,coefficients,-torch.ones(2))
    with pytest.raises(ValueError):g(torch.zeros(2,2,255))


def test_trial_binding_strict_E0_checkpoint_and_admitted_memory_floor(tmp_path,monkeypatch):
    from experiments.pamf_phase3_restoration import physical_trial as phy
    from experiments.pamf_phase3_restoration import multiview_trial as trial
    names=('RUN','MODULE','ARMS','configuration','factory','train','feature_guide','gpu_choice')
    with monkeypatch.context() as patch:
        for name in names:patch.setattr(trial,name,getattr(trial,name))
        phy.install(torch.ones(160));g=trial.factory(phy.ARM)
        path=tmp_path/'E0.pt';trial.save(path,g,phy.ARM,'a'*40,0)
        restored=trial.load(path,phy.ARM,'a'*40,'cpu',epoch=0)
        assert all(torch.equal(value,restored.state_dict()[key]) for key,value in g.state_dict().items())
        assert trial.configuration(phy.ARM)['training_stages']==1 and trial.configuration(phy.ARM)['ancestor_checkpoints']==[]
        patch.setattr(phy,'recovery_gpu_choice',lambda:[dict(GPU=4,free_memory_MB=15500),dict(GPU=6,free_memory_MB=20000)])
        assert phy.choices()==[dict(GPU=6,free_memory_MB=20000)]
