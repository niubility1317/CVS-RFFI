from copy import deepcopy
import torch
from cvsrffi.xuc_fusion.ir_telemetry import isolated_diagnostics, recovery_summary, source_identity_summary, diagnostics_due


def test_diagnostics_do_not_change_update():
    torch.manual_seed(78)
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(3), torch.nn.Linear(3, 2))
    opt = torch.optim.AdamW(model.parameters())
    before = deepcopy(model.state_dict()); rng = torch.get_rng_state().clone()
    flags = [p.requires_grad for p in model.parameters()]
    with isolated_diagnostics(model, opt):
        model.train(); opt.zero_grad(); model(torch.randn(5, 3)).sum().backward(); opt.step()
        model.eval(); next(model.parameters()).requires_grad_(False)
    assert torch.equal(rng, torch.get_rng_state()) and model.training
    assert [p.requires_grad for p in model.parameters()] == flags
    assert not opt.state
    for key, value in before.items():
        torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)


def test_negative_recovery_gain_not_clamped():
    result = recovery_summary(1., 2., .8, .3)
    assert result['ce_reduction'] == -1 and result['accuracy_gain'] < 0


def test_source_identity_slices():
    z = torch.tensor([[1.,0.],[.8,.2],[0.,1.],[.2,.8]])
    y = torch.tensor([0,0,1,1]); rx = torch.tensor([1,3,1,3])
    result = source_identity_summary(z, z, y, rx)
    assert result['worst_tx_accuracy'] == 1 and len(result['rx_tx']) == 4
    assert result['statistical_interaction_norm'] >= 0
    assert result['physical_parameter_recovery_claim'] is False


def test_boundary_and_thousand_accepted_diagnostics():
    assert diagnostics_due(1000, False)
    assert diagnostics_due(23, True)
    assert not diagnostics_due(999, False)


def test_clipping_attribution_uses_real_adamw_without_commit():
    from cvsrffi.xuc_fusion.ir_telemetry import clip_attribution
    model=torch.nn.Linear(2,1,bias=False);opt=torch.optim.AdamW(model.parameters(),lr=.02,weight_decay=.1)
    params=list(model.parameters());before=deepcopy(model.state_dict())
    result=clip_attribution(model,opt,params,[torch.tensor([[4.,1.]])],[torch.tensor([[-3.,1.]])],max_grad_norm=1.,common_coefficient=.2)
    assert result['native_clip_comparison']['difference_norm']>0
    assert result['controls_training'] is False and not opt.state
    torch.testing.assert_close(model.weight,before['weight'],rtol=0,atol=0)
