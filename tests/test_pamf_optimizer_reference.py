from copy import deepcopy
import pytest
import torch
from torch import nn
from experiments.cric_dura_comparison.f_optimizer import build, clip_physical
from experiments.pamf_ssl_research.optimizer_reference import labeled_step


class CoordinateSinc(nn.Module):
    def __init__(self):
        super().__init__()
        self.sample_rate=25000000.; self.scale_hz=25000.; self.mode='CENTER_25K'
        self.low_coordinate=nn.Parameter(torch.zeros(24,1))
        self.band_coordinate=nn.Parameter(torch.zeros(24,1))
        self.register_buffer('low_origin_hz',torch.full((24,1),50000.))
        self.register_buffer('band_origin_hz',torch.full((24,1),75000.))


def make():
    model=nn.Module(); node=model
    for name in ('id_backbone','encoder','core','id_backbone'):
        child=nn.Module(); node.add_module(name,child); node=child
    node.sinc=CoordinateSinc(); node.cls_head=nn.Linear(160,6,bias=False)
    node.body=nn.Parameter(torch.tensor([.25,-.5]))
    return model


def grad(model,number):
    return tuple(torch.full_like(p,number*(i+1)) for i,p in enumerate(model.parameters()))


@pytest.mark.parametrize('history',[0,1,3])
@pytest.mark.parametrize('unused',[False,True])
def test_reference_matches_native_with_current_moments_and_preserves_live(history,unused):
    torch.manual_seed(7); model=make(); named=list(model.named_parameters())
    opt=build(model,model.parameters(),'CENTER_UNITS',lr=2e-4,weight_decay=.01)
    for step in range(history):
        for p,g in zip(model.parameters(),grad(model,3.+step)):p.grad=g.clone()
        clip_physical(opt,model.parameters(),5.); opt.step()
    opt.param_groups[0]['lr']=7e-5
    gradients=list(grad(model,-2.)); gradients[0]=None if unused else gradients[0]
    original=deepcopy(model.state_dict()); state=deepcopy(opt.state_dict())
    live_grads=[p.grad.clone() if p.grad is not None else None for p in model.parameters()]
    expected=deepcopy(model)
    direct=build(expected,expected.parameters(),'CENTER_UNITS',lr=2e-4,weight_decay=.01)
    direct.load_state_dict(deepcopy(state))
    for p,g in zip(expected.parameters(),gradients):p.grad=None if g is None else g.clone()
    clip_physical(direct,expected.parameters(),5.); direct.step()
    result,meta=labeled_step(model,named,opt,gradients,clip_impl=torch.nn.utils.clip_grad_norm_)
    for actual,p in zip(result,expected.parameters()):torch.testing.assert_close(actual,p,rtol=0,atol=0)
    assert all(torch.equal(model.state_dict()[k],v) for k,v in original.items())
    for p,g in zip(model.parameters(),live_grads):
        assert (p.grad is None) if g is None else torch.equal(p.grad,g)
    for index,values in state['state'].items():
        for key,value in values.items():
            if torch.is_tensor(value):assert torch.equal(opt.state_dict()['state'][index][key],value)
    assert meta['reference_optimizer_steps']==1 and meta['reference_physical_clip_calls']==1
    assert meta['copied_existing_moments'] is True
    assert meta['live_optimizer_updates']==0 and meta['model_forward_calls']==0
    assert opt._sinc_unit_steps==history and opt._sinc_physical_clip_calls==history


@pytest.mark.parametrize('bad',['order','missing','dtype','nonfinite','wrong_optimizer'])
def test_reference_rejects_misaligned_or_illegal_scope(bad):
    model=make(); named=list(model.named_parameters()); gradients=list(grad(model,1.))
    opt=build(model,model.parameters(),'CENTER_UNITS',lr=2e-4,weight_decay=.01)
    if bad=='order':named.reverse()
    elif bad=='missing':gradients.pop()
    elif bad=='dtype':gradients[0]=gradients[0].double()
    elif bad=='nonfinite':gradients[0].fill_(float('nan'))
    else:
        other=make();opt=build(other,other.parameters(),'CENTER_UNITS',lr=2e-4,weight_decay=.01)
    with pytest.raises(ValueError):labeled_step(model,named,opt,gradients,clip_impl=torch.nn.utils.clip_grad_norm_)
