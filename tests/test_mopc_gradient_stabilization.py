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


def test_public_code_does_not_evaluate_discarded_paper_objective(monkeypatch):
    from experiments.pamf_phase2_practical.vendor_native.comparison_suite import registration

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__();self.layer=nn.Linear(2,4)
        def forward(self,x):return {'features':self.layer(x)}

    torch.manual_seed(7)
    model=from_protonet(Encoder().eval(),torch.randn(2,4)*.01)
    anchor=make_anchor(model,torch.randn(4,2),torch.tensor([0,0,1,1]),method='mopc_hr')
    def reject_paper(*args,**kwargs):
        raise AssertionError('public code evaluated unused paper objective')
    monkeypatch.setattr(registration,'mopc_hr_incremental_objective',reject_paper)
    cfg=RegistrationConfig(seed=7,mopc_semantics='public_code',mopc_epochs=1,mopc_batch_size=4)
    result=register_new_classes(anchor,torch.randn(4,2),torch.tensor([2,2,3,3]),[2,3],config=cfg)
    assert result.loss_trace and all(torch.isfinite(torch.tensor(row['loss'])) for row in result.loss_trace)


def test_public_code_objective_and_gradients_match_previous_implementation():
    from experiments.pamf_phase2_practical.vendor_native.comparison_suite import registration
    import torch.nn.functional as F

    torch.manual_seed(11)
    logits=torch.randn(4,3,requires_grad=True)
    proto_logits=torch.randn(4,3,requires_grad=True)
    parameters={'a':torch.randn(3,2,requires_grad=True),'b':torch.randn(3,requires_grad=True)}
    previous={name:torch.randn_like(value) for name,value in parameters.items()}
    labels=torch.tensor([0,1,2,0]); proto_labels=torch.tensor([1,2,0,1])
    paper=registration.mopc_hr_incremental_objective(logits,labels,proto_logits,proto_labels,
                                                       parameters,previous,lambda_max=.7)
    old_hr=sum(.7*(1.0-j/len(parameters))*(p-previous[name]).norm(2)
               for j,(name,p) in enumerate(parameters.items()))
    old_loss=paper.cross_entropy+F.cross_entropy(proto_logits/2.,proto_labels)+old_hr
    current=registration._mopc_public_objective(logits,labels,proto_logits,proto_labels,
                                                  parameters,previous,lambda_max=.7)
    variables=[logits,proto_logits,*parameters.values()]
    old_grad=torch.autograd.grad(old_loss,variables,retain_graph=True)
    new_grad=torch.autograd.grad(current.total,variables)
    assert torch.equal(old_loss,current.total)
    assert all(torch.equal(a,b) for a,b in zip(old_grad,new_grad))


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
