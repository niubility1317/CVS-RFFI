import pytest
import torch
from torch import nn
from torch.nn import functional as F

from experiments.cvs_equivariant_identity.model import rotate_pair
from experiments.cvs_neural_residual_identity.model import build as control
from experiments.cvs_channel_order_identity.model import (
    BASE_VARIANT, CROP, RHO, VARIANTS, BoundedPacketFilter, ChannelBranch,
    build, channel_contract, cross_order_readout, phase_invariant_context,
)


torch.set_num_threads(2)


@pytest.mark.parametrize('variant,total', zip(VARIANTS, [247451, 234587, 247387, 249947]))
def test_contract_scratch_baseline_rng_and_strict_state(variant, total):
    torch.manual_seed(23)
    base = control(BASE_VARIANT).eval()
    rng = torch.get_rng_state().clone()
    state = base.state_dict()
    torch.manual_seed(23)
    model = build(variant).eval()
    assert torch.equal(torch.get_rng_state(), rng)
    assert model.contract() == channel_contract(variant)
    assert model.contract()['total_parameters'] == total
    assert sum(p.numel() for p in model.channel_parameters()) == total - 220987
    assert not ({id(p) for p in model.channel_parameters()} & {id(p) for p in model.residual_parameters()})
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in state.items())
    x = torch.randn(4, 2, 256)
    with torch.no_grad():
        torch.testing.assert_close(model.features(x), base.features(x), atol=0, rtol=0)
        torch.testing.assert_close(model(x), base(x), atol=0, rtol=0)
    replica = build(variant).eval()
    replica.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(replica(x), model(x), atol=0, rtol=0)
    with pytest.raises(RuntimeError):
        replica.load_state_dict(state, strict=True)


def test_g_bound_context_phase_and_packet_independence():
    torch.manual_seed(29)
    g = BoundedPacketFilter().double()
    with torch.no_grad():
        g.context[-1].weight.normal_(std=10.)
        g.context[-1].bias.normal_(std=10.)
        g.basis.mul_(30.)
    x = torch.randn(4, 2, 256, dtype=torch.float64)
    z = torch.randn(4, 2, 8, 256, dtype=torch.float64)
    phases = torch.tensor([.4, -.1, 1.3, -2.4], dtype=torch.float64)
    coefficients = g.coefficients(x)
    basis = g.effective_basis()
    assert torch.linalg.vector_norm(coefficients, dim=1).sum(-1).max() <= 1 + 1e-14
    assert torch.linalg.vector_norm(basis, dim=1).sum(-1).max() <= 1 + 1e-14
    output = g.apply_filter(z, coefficients)
    norms = z.flatten(1).norm(dim=1)
    assert ((output - z).flatten(1).norm(dim=1) <= RHO * norms + 1e-12).all()
    assert (output.flatten(1).norm(dim=1) <= (1 + RHO) * norms + 1e-12).all()
    assert (output.flatten(1).norm(dim=1) >= (1 - RHO) * norms - 1e-12).all()
    torch.testing.assert_close(phase_invariant_context(rotate_pair(x, phases)), phase_invariant_context(x), atol=1e-14, rtol=1e-12)
    torch.testing.assert_close(g.coefficients(rotate_pair(x, phases)), coefficients, atol=1e-14, rtol=1e-12)
    torch.testing.assert_close(g.apply_filter(rotate_pair(z, phases), coefficients), rotate_pair(output, phases), atol=1e-13, rtol=1e-12)
    torch.testing.assert_close(torch.cat([g(packet[None])[0] for packet in x]), g(x)[0], atol=1e-13, rtol=1e-12)


def test_common_valid_crop_removes_linear_boundary_interaction_but_preserves_nonlinear_d():
    torch.manual_seed(33)
    branch = ChannelBranch('channel_order').double()
    with torch.no_grad():
        branch.compensation.context[-1].bias.copy_(torch.tensor([.3, -.2, .1, .1, -.1, .2, .1, .1], dtype=torch.float64))
    x = torch.randn(3, 2, 256, dtype=torch.float64)
    nonlinear = branch(x)
    assert nonlinear['difference'].shape == (3, 2, 8, 256 - 2 * CROP)
    assert nonlinear['difference'].norm() / nonlinear['u'].norm() > 1e-4
    branch.feature[1] = nn.Identity()
    branch.feature[3] = nn.Identity()
    linear = branch(x)
    assert linear['difference'].abs().max() < 2e-14
    # Without the registered interior crop there really is a padding artifact.
    gx, coefficients = branch.compensation(x)
    full_u = branch.feature(x[:, :, None])
    full_v = branch.feature(gx[:, :, None])
    full_w = branch.compensation.apply_filter(full_u, coefficients)
    assert (full_v - full_w).abs().max() > 1e-5


def test_order_readout_has_nonzero_d_derivative_at_zero():
    torch.manual_seed(35)
    u = torch.randn(2, 2, 8, 246)
    difference = torch.zeros_like(u, requires_grad=True)
    readout = cross_order_readout(u, difference)
    assert torch.count_nonzero(readout) == 0
    readout.sum().backward()
    assert torch.isfinite(difference.grad).all() and difference.grad.norm() > 0


@pytest.mark.parametrize('variant', VARIANTS)
def test_ce_reaches_g_exit_within_two_steps_and_all_parameters_by_third(variant):
    torch.manual_seed(37)
    model = build(variant).train()
    x, labels = torch.randn(4, 2, 256), torch.tensor([0, 1, 2, 3])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0002, weight_decay=.0001)
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(x), labels).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        branch = model.channel_branch
        if step == 0:
            if variant == 'channel_order':
                assert branch.compensation.context[-1].weight.grad.norm() > 0
                assert branch.compensation.context[-1].bias.grad.norm() > 0
                assert branch.last_d_gradient_norm > 0
                assert branch.d_project.weight.grad.norm() == 0
            if branch.compensation is not None:
                assert branch.compensation.basis.grad.norm() == 0
        elif step == 1 and variant in ('channel_compensated', 'channel_dual'):
            assert branch.compensation.context[-1].weight.grad.norm() > 0
            assert branch.compensation.basis.grad.norm() == 0
            assert all(p.grad.norm() > 0 for p in branch.feature.parameters())
        else:
            assert all(p.grad.norm() > 0 for p in model.channel_parameters())
        optimizer.step()
    diagnostic = model.diagnostics(x)
    assert 'neural_residual' in diagnostic and 'adaptive_input' in diagnostic
    actual = diagnostic['channel_order']
    assert actual['active'] is True and actual['valid_tokens'] == 246
    assert actual['residual_relative_output_mean'] > 0
    if variant == 'channel_capacity':
        assert all(actual[key] is None for key in actual if key.startswith(('g_', 'd_')))
    else:
        assert actual['g_magnitude_mean'] > 0
        assert 0 < actual['g_operator_delta_bound_max'] <= .2500001
    if variant == 'channel_order':
        assert actual['d_relative_output_mean'] > 0
        assert actual['d_gradient_norm'] > 0 and actual['d_readout_gradient_norm'] > 0
    else:
        assert actual['d_relative_output_mean'] is None


@pytest.mark.parametrize('variant', VARIANTS)
def test_nonzero_auxiliary_features_phase_packet_independence_and_no_inference_updates(variant):
    torch.manual_seed(43)
    model = build(variant).eval()
    with torch.no_grad():
        for parameter in model.channel_parameters():
            parameter.add_(torch.randn_like(parameter) * .003)
    x = torch.randn(4, 2, 256)
    permutation = torch.tensor([2, 0, 3, 1])
    phases = torch.tensor([.17, -.7, 1.3, 2.4])
    state = {key: value.clone() for key, value in model.state_dict().items()}
    last_d_gradient = model.channel_branch.last_d_gradient_norm
    with torch.no_grad():
        output = model(x)
        torch.testing.assert_close(torch.cat([model(packet[None]) for packet in x]), output, atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(model(x[permutation]), output[permutation], atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(model(rotate_pair(x, phases)), output, atol=2e-3, rtol=2e-3)
        modified = x.clone()
        modified[1:] *= 2.5
        torch.testing.assert_close(model(modified)[:1], output[:1], atol=0, rtol=0)
    model.train()
    model.core.time[0].eval()
    flags = [module.training for module in model.modules()]
    rng = torch.get_rng_state().clone()
    model.diagnostics(x)
    assert torch.equal(rng, torch.get_rng_state())
    assert [module.training for module in model.modules()] == flags
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in state.items())
    assert model.channel_branch.last_d_gradient_norm == last_d_gradient
    assert not model.channel_branch._forward_hooks


def test_contract_inspects_actual_rho_and_unknown_variant_is_rng_neutral():
    model = build('channel_order')
    model.channel_branch.compensation.rho = .7
    assert model.contract()['channel_active'] is False
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match='Unregistered'):
        build('unknown')
    assert torch.equal(rng, torch.get_rng_state())
