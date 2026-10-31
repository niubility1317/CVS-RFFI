import torch
from torch import nn

from experiments.pamf_identity_flow.model import IdentityConditionalFlow
from experiments.pamf_identity_flow.rollout_objective import rollout_objective
from experiments.pamf_identity_flow import rollout_train


class Reader(nn.Module):
    def features(self,x):return torch.cat((x.mean(2),x.square().mean(2),x[:,:,:64].mean(2)),1)


def test_rollout_loss_reaches_field_and_never_updates_teacher():
    torch.manual_seed(14)
    model=IdentityConditionalFlow(width=16,layers=1,heads=4)
    teacher=Reader().eval().requires_grad_(False)
    clean=torch.randn(3,2,256);received=clean+.3*torch.randn_like(clean);noise=torch.randn_like(clean)
    loss,parts=rollout_objective(model,teacher,clean,received,noise)
    assert loss.shape==(3,) and torch.isfinite(loss).all()
    assert parts['path_a'].shape==clean.shape and parts['path_b'].shape==clean.shape
    loss.mean().backward()
    assert model.output.weight.grad is not None and model.output.weight.grad.abs().sum()>0
    assert all(v.grad is None for v in teacher.parameters())


def test_training_rollout_starts_from_received_not_clean():
    class Probe(nn.Module):
        def __init__(self):super().__init__();self.inputs=[]
        def integrate(self,received,noise,*,noise_scale,steps):
            self.inputs.append((received.detach().clone(),noise.detach().clone(),noise_scale,steps))
            return received+noise_scale*noise
    model=Probe();teacher=Reader().eval().requires_grad_(False)
    received=torch.randn(2,2,256);clean=torch.randn_like(received);noise=torch.randn_like(received)
    rollout_objective(model,teacher,clean,received,noise)
    assert len(model.inputs)==2
    assert torch.equal(model.inputs[0][0],received) and torch.equal(model.inputs[1][0],received)
    assert torch.equal(model.inputs[0][1],noise) and torch.equal(model.inputs[1][1],-noise)
    assert all(item[2:]==(.02,4) for item in model.inputs)


def test_fixed_rollout_checkpoint_roundtrip_and_budget(tmp_path):
    assert rollout_train.EPOCHS==30 and rollout_train.STEPS==14790
    assert [rollout_train.source_view(i) for i in (1,20,21,30)]==[1,20,1,10]
    model=rollout_train.make_model();path=tmp_path/'flow.pt';commit='a'*40
    rollout_train.save(path,model,commit,epoch=0,updates=0)
    loaded=rollout_train.load(path,'cpu',commit,epoch=0)
    with torch.no_grad():
        x=torch.randn(2,2,256)
        torch.testing.assert_close(loaded(x),x,rtol=1e-6,atol=1e-6)
