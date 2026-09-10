from copy import deepcopy

import pytest
import torch
from torch import nn

from cvsrffi.cross_response.training import SourceResponseGate, parameter_roles, response_backward
from cvsrffi.cross_response.integration import forward_labeled
from model import MixStyle1D


class Branch(nn.Module):
    def __init__(self, shared):
        super().__init__()
        self.stem = shared
        self.front = nn.Linear(3, 3, bias=False)
        self.cls_head = nn.Linear(3, 2, bias=False)

    def forward(self, x):
        return self.cls_head(self.front(self.stem(x)))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        shared = nn.Linear(3, 3, bias=False)
        self.id_backbone = Branch(shared)
        self.dom_backbone = Branch(shared)
        self.dom_head = nn.Linear(2, 2, bias=False)

    def forward(self, x):
        return self.id_backbone(x), self.dom_backbone(x)


@pytest.mark.parametrize("joint,head_only", [(False,False),(True,False),(False,True)])
def test_response_gradient_isolation_and_baseline_preservation(joint, head_only):
    torch.manual_seed(11)
    model = Model()
    aux = nn.Linear(4, 2)
    roles = parameter_roles(model, aux, ["id_backbone.cls_head"])
    assert len(roles["shared"]) == 1
    flat = [id(p) for rows in roles.values() for _, p in rows]
    assert len(flat) == len(set(flat))
    x = torch.randn(7, 3)
    zi, zd = model(x)
    base = zi.square().mean() + zd.square().mean()
    response = aux(torch.cat([zi, zd], 1)).square().mean()
    base_grads = torch.autograd.grad(base, list(model.parameters()), retain_graph=True, allow_unused=True)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    logs = response_backward(baseline_loss=base, identity_loss=zi.square().mean(), response_loss=response,
        decision_loss=zi.sum()*0, cross_loss=zi.sum()*0, roles=roles, scaler=scaler,
        lambda_resp=.5, lambda_dec=0, lambda_cross=0, joint_open=joint, gradient_cap=.05, head_only=head_only)
    expected = {id(p): g for p, g in zip(model.parameters(), base_grads)}
    for role in ("shared", "identity_front", "other"):
        for _, p in roles[role]:
            if expected[id(p)] is None:
                assert p.grad is None
            else:
                torch.testing.assert_close(p.grad, expected[id(p)])
    assert logs["response_grad_shared"] == logs["response_grad_identity_front"] == 0
    assert logs["response_grad_auxiliary"] > 0
    assert (logs["response_grad_domain"] > 0) != head_only
    assert (logs["response_grad_identity_tail"] > 0) == (joint and not head_only)
    assert logs["response_grad_identity_tail"] <= .05*logs["response_identity_reference_norm"] + 1e-7


def test_gate_rejects_target_and_requires_consecutive_real_source_evidence():
    gate = SourceResponseGate(min_blocks=2, error_ratio=.9, stable_checks=2)
    with pytest.raises(ValueError):
        gate.observe(response_error=.1, constant_error=1, blocks=3, source_role="target")
    def observe(error, blocks=2):
        return gate.observe(response_error=error, constant_error=1, blocks=blocks, source_role="source_validation")
    assert not observe(.1, blocks=1)["gate_open"]
    assert not observe(.1)["gate_open"]
    assert not observe(float("nan"))["gate_open"]
    assert not observe(.1)["gate_open"]
    state = gate.state_dict()
    restored = SourceResponseGate(min_blocks=2, error_ratio=.9, stable_checks=2)
    restored.load_state_dict(state)
    assert restored.observe(response_error=.1,constant_error=1,blocks=2,source_role="source_validation")["gate_open"]
    with pytest.raises(ValueError):
        SourceResponseGate().load_state_dict(state)


def test_zero_reference_identity_gradient_cannot_amplify_response():
    model, aux = Model(), nn.Linear(4, 2)
    zi, zd = model(torch.randn(3,3))
    logs = response_backward(baseline_loss=zi.square().mean()+zd.square().mean(), identity_loss=zi.sum()*0,
        response_loss=aux(torch.cat([zi,zd],1)).square().mean(), decision_loss=zi.sum()*0,cross_loss=zi.sum()*0,
        roles=parameter_roles(model,aux,["id_backbone.cls_head"]), scaler=torch.amp.GradScaler("cuda",enabled=False),
        lambda_resp=1,lambda_dec=0,lambda_cross=0,joint_open=True,gradient_cap=.1)
    assert logs["response_grad_identity_tail"] == 0
    assert logs["response_grad_domain"] > 0


def test_actual_mixstyle_never_uses_disallowed_query_even_across_two_layers():
    module = MixStyle1D(p=1., mix="same_tx_crossdomain", fallback="random")
    module.train()
    mask = torch.zeros(6,6,dtype=torch.bool)
    mask[:3,:3] = True
    mask[3:,3:] = True
    mask.fill_diagonal_(False)
    module._cross_response_allowed = mask
    x = torch.randn(6,4,12)
    changed = x.clone()
    changed[3:] = changed[3:]*10+50
    y, d = torch.zeros(6,dtype=torch.long), torch.arange(6)
    torch.manual_seed(7)
    first = module(module(x,d,y),d,y)
    torch.manual_seed(7)
    second = module(module(changed,d,y),d,y)
    torch.testing.assert_close(first[:3],second[:3],atol=0,rtol=0)
    module._cross_response_allowed = torch.zeros(6,6,dtype=torch.bool)
    torch.testing.assert_close(module(x,d,y),x)


def test_disabled_forward_is_exact_original_rng_and_gradient():
    model = Model()
    copy = deepcopy(model)
    x = torch.randn(3,3)
    state = torch.get_rng_state()
    a = forward_labeled(model,None,None,x)
    after = torch.get_rng_state()
    torch.set_rng_state(state)
    b = copy(x)
    assert torch.equal(after,torch.get_rng_state())
    for aa,bb in zip(a,b):
        torch.testing.assert_close(aa,bb,atol=0,rtol=0)
    sum(z.sum() for z in a).backward()
    sum(z.sum() for z in b).backward()
    for pa,pb in zip(model.parameters(),copy.parameters()):
        if pa.grad is None:
            assert pb.grad is None
        else:
            torch.testing.assert_close(pa.grad,pb.grad,atol=0,rtol=0)
