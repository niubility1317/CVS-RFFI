import math
import json
from pathlib import Path
import pytest
import torch
from torch.nn import functional as F
from experiments.cvs_channel_response_identity.model import VARIANTS,build,relative_cross_tokens
from experiments.cvs_neural_residual_identity.model import build as base_build

torch.set_num_threads(2)


@pytest.mark.parametrize('variant',VARIANTS)
def test_exact_initial_function_and_first_step_compensation_gradient(variant):
    torch.manual_seed(45);model=build(variant)
    torch.manual_seed(45);base=base_build('neural_residual_shallow')
    x=torch.randn(6,2,256);y=torch.arange(6)
    model.eval();base.eval()
    torch.testing.assert_close(model(x),base(x),rtol=0,atol=0)
    model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=.0002)
    all_positive=[]
    for step in range(3):
        optimizer.zero_grad();loss=F.cross_entropy(model(x),y);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        exit_grad=model.response_branch.compensation.context[-1].weight.grad
        assert exit_grad.norm()>0
        all_positive.append(all(p.grad.norm()>0 for p in model.response_parameters()))
        optimizer.step()
    assert all_positive[-1], 'All response tensors should receive nonzero CE gradient by step3'


@pytest.mark.parametrize('variant',VARIANTS)
def test_identity_null_after_nonzero_learning_and_global_phase(variant):
    torch.manual_seed(2026);model=build(variant).eval();b=model.response_branch
    with torch.no_grad():
        b.compensation.context[-1].weight.normal_(std=.1);b.compensation.context[-1].bias.normal_(std=.1)
    x=torch.randn(4,2,256);angle=.91
    y=torch.stack((x[:,0]*math.cos(angle)-x[:,1]*math.sin(angle),x[:,0]*math.sin(angle)+x[:,1]*math.cos(angle)),dim=1)
    with torch.no_grad():
        a=b(x);other=b(y)
        assert a['residual'].norm()>0
        torch.testing.assert_close(a['residual'],other['residual'],rtol=2e-4,atol=2e-6)
        torch.testing.assert_close(model(x),model(y),rtol=2e-4,atol=2e-5)
        batched=model(x);single=torch.cat([model(row[None]) for row in x])
        torch.testing.assert_close(batched,single,rtol=2e-4,atol=2e-5)
    old=b.compensation.coefficients
    try:
        b.compensation.coefficients=lambda z:torch.zeros(len(z),2,4,device=z.device)
        assert torch.count_nonzero(b(x)['residual'])==0
    finally:b.compensation.coefficients=old


def test_signed_cross_has_gradient_at_zero_change():
    u=torch.randn(2,2,8,21);delta=torch.zeros_like(u,requires_grad=True)
    tokens=relative_cross_tokens(u,delta)
    assert torch.count_nonzero(tokens)==0
    tokens.sum().backward();assert delta.grad.norm()>0


@pytest.mark.parametrize('variant',VARIANTS)
def test_driver_contract_and_diagnostic_state(variant):
    from experiments.cvs_channel_response_identity.model import response_contract
    from experiments.cvs_channel_response_identity.source import validate_config
    root=Path(__file__).resolve().parents[1]
    cfg=json.loads((root/'experiments/cvs_channel_order_identity/configs/channel_dual-s2026092701.json').read_text(encoding='utf-8'))
    cfg.pop('channel');cfg.update(method='cvs_channel_response_identity',variant=variant,response=response_contract(variant))
    validate_config(cfg)
    bad=dict(cfg,extra_losses=['contrastive'])
    with pytest.raises(ValueError):validate_config(bad)
    model=build(variant).train();state={k:v.clone() for k,v in model.state_dict().items()}
    diagnostics=model.diagnostics(torch.randn(2,2,256))
    assert diagnostics['channel_response']['active']
    assert diagnostics['channel_response']['response_projection_relative']==0
    assert model.training
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())
