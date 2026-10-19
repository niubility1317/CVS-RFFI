import copy
import importlib
import importlib.util

import pytest
import torch
from torch import nn

torch.set_num_threads(2)


def implementation():
    assert importlib.util.find_spec('experiments.pamf_attention_restore') is not None,'Attention prototype missing'
    return importlib.import_module('experiments.pamf_attention_restore.models')


def inputs(count=2):
    return .2*torch.randn((count,9,256),generator=torch.Generator().manual_seed(24001))


def opened_prior(checkpoint_blocks=True):
    torch.manual_seed(24);model=implementation().TemporalAttentionPrior(checkpoint_blocks=checkpoint_blocks)
    with torch.no_grad():model.out.weight.normal_(0,.003)
    return model


def test_full_resolution_zero_output_and_fixed_capacity():
    model=implementation().TemporalAttentionPrior();x=inputs()
    assert torch.equal(model(x),torch.zeros(2,2,256))
    assert len(model.blocks)==6 and model.width==128 and model.heads==4
    assert model.position.shape==(1,256,128) and not model.position.requires_grad
    assert sum(p.numel() for p in model.parameters())==901890
    assert not any(isinstance(m,(nn.Dropout,nn.BatchNorm1d)) for m in model.modules())


def test_every_attention_local_and_MLP_block_has_finite_gradient():
    model=opened_prior().train();x=inputs().requires_grad_();model(x).square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.norm()>0
    for block in model.blocks:
        for part in (block.qkv,block.project,block.local,block.mix,block.mlp):
            gradients=[p.grad for p in part.parameters()]
            assert all(g is not None and torch.isfinite(g).all() and g.norm()>0 for g in gradients)


def test_checkpointed_and_plain_outputs_and_parameter_gradients_match():
    a=opened_prior(True).train();b=copy.deepcopy(a);b.checkpoint_blocks=False;x=inputs()
    aa=a(x);bb=b(x);assert torch.equal(aa,bb)
    aa.square().mean().backward();bb.square().mean().backward()
    for pa,pb in zip(a.parameters(),b.parameters()):assert torch.allclose(pa.grad,pb.grad,atol=1e-7,rtol=1e-5)


def test_packet_independence_order_and_state_preservation():
    model=opened_prior().eval();x=inputs(3);before=copy.deepcopy(model.state_dict())
    with torch.no_grad():
        whole=model(x);single=torch.cat([model(v[None]) for v in x]);permuted=model(x[[2,0,1]])
    assert torch.allclose(whole,single,atol=1e-6,rtol=1e-5)
    assert torch.allclose(whole[[2,0,1]],permuted,atol=1e-6,rtol=1e-5)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())


@pytest.mark.parametrize('bad',[torch.ones(2,9,128),torch.ones(9,256),torch.ones(2,9,256,dtype=torch.int64),torch.full((2,9,256),float('nan'))])
def test_invalid_input_rejected(bad):
    with pytest.raises(ValueError):implementation().TemporalAttentionPrior()(bad)


@pytest.mark.parametrize('amplitude',[0.,1e-10,1e3])
def test_silent_small_and_large_inputs_have_finite_gradients(amplitude):
    model=opened_prior().train();x=(inputs()*amplitude).requires_grad_();out=model(x)
    out.square().mean().backward();assert torch.isfinite(out).all() and torch.isfinite(x.grad).all()


class Context(nn.Module):
    def __init__(self):super().__init__();self.scale=nn.Parameter(torch.tensor(.9))
    def estimate(self,x):
        from experiments.pamf_action_restore.operators import Action
        main=torch.full((len(x),256),.9+.1j,dtype=torch.complex64,device=x.device)
        kernel=torch.ones((len(x),1),dtype=main.dtype,device=x.device)
        return self.scale*x,Action(kernel,main,.02*main),torch.full((len(x),),.01,device=x.device)


def test_composed_three_stage_models_start_with_same_output_and_frozen_context():
    from experiments.pamf_unrolled_restore.models import UnrolledRestorer
    m=implementation();x=inputs()[:,:2];initial=None
    for candidate in ('unet_control','attention_prior'):
        torch.manual_seed(24);model=m.AttentionRestorer(candidate,Context()).train()
        stages,*_=model.stage_outputs(x)
        if initial is None:initial=stages[-1].detach()
        else:assert torch.equal(initial,stages[-1].detach())
        (stages[-1]-.95*x-.03).square().mean().backward()
        assert len(stages)==3 and all(core.out.weight.grad is not None and torch.isfinite(core.out.weight.grad).all() and core.out.weight.grad.norm()>0 for core in model.prior)
        assert not model.frozen.training and not any(p.requires_grad or p.grad is not None for p in model.frozen.parameters())
        assert len({p.data_ptr() for core in model.prior for p in core.parameters()})==sum(len(list(core.parameters())) for core in model.prior)


def test_unknown_restorer_candidate_rejected():
    with pytest.raises(ValueError):implementation().AttentionRestorer('unknown',Context())
