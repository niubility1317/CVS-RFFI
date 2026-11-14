import pytest
import torch
from torch import nn

from experiments.pamf_phase2_practical.vendor_native.comparison_suite.registration import (
    RegistrationConfig, _clip_finite_gradients, register_new_classes,
)
from experiments.pamf_phase2_practical.group2_cil_bridge import from_protonet, make_anchor


def test_clipping_uses_float64_norm_without_losing_finite_large_gradients():
    p=nn.Parameter(torch.zeros(2));p.grad=torch.tensor([3e30,4e30])
    raw,clipped=_clip_finite_gradients([p],1.0)
    assert raw==pytest.approx(5e30,rel=1e-6) and clipped
    assert torch.allclose(p.grad,torch.tensor([.6,.8]),rtol=1e-6,atol=1e-6)


def test_nonfinite_gradient_never_silently_clipped():
    p=nn.Parameter(torch.zeros(2));p.grad=torch.tensor([float('nan'),1.])
    with pytest.raises(FloatingPointError):_clip_finite_gradients([p],1.0)


def test_clipping_is_opt_in_and_requires_positive_finite_limit():
    assert RegistrationConfig().mopc_max_grad_norm is None
    for value in (0.,-1.,float('inf'),float('nan')):
        with pytest.raises(ValueError):RegistrationConfig(mopc_max_grad_norm=value)


def test_bridge_supports_historical_paper_semantics_for_diagnostics(monkeypatch):
    from experiments.pamf_phase2_practical import group2_cil_bridge as bridge
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__();self.layer=nn.Linear(2,4)
        def forward(self,x):return {'features':self.layer(x)}
    model=bridge.from_protonet(Encoder().eval(),torch.randn(2,4))
    anchor=bridge.make_anchor(model,torch.randn(4,2),torch.tensor([0,0,1,1]),method='mopc_hr')
    seen=[]
    def capture(state,x,y,ids,config):
        seen.append((config.mopc_semantics,config.mopc_max_grad_norm))
        return state
    monkeypatch.setattr(bridge,'register_new_classes',capture)
    bridge.register(anchor,torch.randn(2,2),torch.tensor([2,2]),seed=7,mopc_semantics='paper',mopc_max_grad_norm=1.)
    bridge.register(anchor,torch.randn(2,2),torch.tensor([2,2]),seed=7)
    assert seen==[('paper',1.),('public_code',None)]


@pytest.mark.parametrize('seed',[1731,1732,1733])
def test_diagnostic_variant_handles_large_synthetic_support(seed):
    class Encoder(nn.Module):
        def __init__(self):super().__init__();self.layer=nn.Linear(2,4)
        def forward(self,x):return {'features':self.layer(x)}
    torch.manual_seed(seed)
    model=from_protonet(Encoder().eval(),torch.randn(2,4)*.01)
    old_x=torch.randn(8,2)*100;old_y=torch.tensor([0]*4+[1]*4)
    anchor=make_anchor(model,old_x,old_y,method='mopc_hr')
    new_x=torch.randn(8,2)*100;new_y=torch.tensor([2]*4+[3]*4)
    cfg=RegistrationConfig(seed=seed,mopc_semantics='public_code',mopc_epochs=20,
                           mopc_batch_size=4,mopc_max_grad_norm=1.)
    state=register_new_classes(anchor,new_x,new_y,[2,3],config=cfg)
    assert state.protocol_metadata['gradient_clipped_updates']>0
    assert state.protocol_metadata['author_optimizer_unchanged'] is False
    assert all(bool(torch.isfinite(p).all()) for p in state.model.parameters())
