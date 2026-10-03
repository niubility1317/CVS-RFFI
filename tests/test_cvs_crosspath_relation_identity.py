import pytest
import torch
from torch.nn import functional as F

from experiments.cvs_equivariant_identity.model import InvariantReadout, rotate_pair
from experiments.cvs_neural_residual_identity.model import build as control
from experiments.cvs_crosspath_relation_identity.model import (
    BASE_VARIANT, VARIANTS, POWER_FLOOR, build, readout_contract, relation_statistics,
)

torch.set_num_threads(2)


@pytest.mark.parametrize('variant', VARIANTS)
def test_scratch_function_state_rng_parameters_and_active_contract(variant):
    torch.manual_seed(13)
    base = control(BASE_VARIANT).eval()
    state, rng = base.state_dict(), torch.get_rng_state().clone()
    torch.manual_seed(13)
    model = build(variant).eval()
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())
    assert sum(p.numel() for p in model.parameters()) == 233275
    assert sum(p.numel() for p in model.readout_parameters()) == 12288
    assert model.contract() == readout_contract(variant)
    x = torch.randn(3, 2, 256)
    with torch.no_grad():
        assert torch.equal(model(x), base(x))
        assert torch.equal(model.features(x), model.core.features(x))
    model.core.relation.channel_normalized = not model.core.relation.channel_normalized
    assert model.contract()['learned_readout_active'] is False


def test_variant_projection_initialization_is_paired():
    torch.manual_seed(77)
    a = build(VARIANTS[0])
    torch.manual_seed(77)
    b = build(VARIANTS[1])
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())


@pytest.mark.parametrize('normalized', [False, True])
def test_statistics_match_independent_complex_loop(normalized):
    torch.manual_seed(3)
    a, b = (torch.randn(2, 2, 8, 64, dtype=torch.float64) for _ in range(2))
    actual = relation_statistics(a, b, normalized)
    ac, bc = torch.complex(a[:, 0], a[:, 1]), torch.complex(b[:, 0], b[:, 1])
    matrices = []
    for lag in (0, 4, 8):
        x, y = ac[..., 8-lag:64-lag], bc[..., 8:64]
        ex, ey = x.abs().square().mean(-1), y.abs().square().mean(-1)
        if not normalized:
            ex, ey = ex.mean(-1, keepdim=True), ey.mean(-1, keepdim=True)
        matrix = torch.stack([sum(x[..., j:j+1] * y[..., j:j+1].conj().transpose(1, 2)
                                 for j in range(56))[i] for i in range(2)]) / 56
        matrix = matrix / (ex.clamp_min(POWER_FLOOR).sqrt().unsqueeze(-1)
                           * ey.clamp_min(POWER_FLOOR).sqrt().unsqueeze(-2)) / 8
        matrices.append(torch.stack([matrix.real, matrix.imag], 1))
    torch.testing.assert_close(actual, torch.stack(matrices, 1), atol=1e-14, rtol=1e-13)


@pytest.mark.parametrize('normalized', [False, True])
def test_common_phase_gain_boundary_and_per_lag_norm_bound(normalized):
    torch.manual_seed(4)
    a, b = (torch.randn(3, 2, 8, 64, dtype=torch.float64) for _ in range(2))
    expected = relation_statistics(a, b, normalized)
    phase = torch.tensor([.3, -.9, 1.7], dtype=a.dtype)
    actual = relation_statistics(rotate_pair(a, phase), rotate_pair(b, phase), normalized)
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-13)
    assert expected.flatten(2).norm(dim=-1).max() <= 1 + 1e-12
    torch.testing.assert_close(relation_statistics(2*a, .7*b, normalized), expected)
    gains = torch.linspace(.4, 3., 8, dtype=a.dtype)[None, None, :, None]
    per_channel = relation_statistics(a*gains, b/gains, normalized)
    if normalized:
        torch.testing.assert_close(per_channel, expected, atol=1e-14, rtol=1e-13)
    else:
        assert (per_channel - expected).abs().max() > .01
    # Independent feature phases remain observable; no complex-gain invariant claim.
    changed = relation_statistics(a, rotate_pair(b, phase), normalized)
    assert (changed - expected).abs().max() > .001


@pytest.mark.parametrize('normalized', [False, True])
def test_lag_direction_common_window_and_zero_gradient_finiteness(normalized):
    a, b = torch.zeros(1, 2, 8, 64), torch.zeros(1, 2, 8, 64)
    a[0, 0, 0, 13], b[0, 0, 0, 21] = 1, 1
    out = relation_statistics(a, b, normalized)
    assert torch.count_nonzero(out[:, :2]) == 0
    assert out[0, 2, 0, 0, 0] > 0
    # Behavior positions outside the common window are excluded in every lag.
    b.zero_(); b[..., :8] = 1
    assert torch.count_nonzero(relation_statistics(a, b, normalized)) == 0
    for scale in (0., 1e-12):
        x = (torch.randn(2, 2, 8, 64) * scale).requires_grad_()
        y = (torch.randn(2, 2, 8, 64) * scale).requires_grad_()
        z = relation_statistics(x, y, normalized)
        z.square().sum().backward()
        assert torch.isfinite(z).all() and torch.isfinite(x.grad).all() and torch.isfinite(y.grad).all()


def test_local_relation_information_is_absent_from_separate_readouts():
    torch.manual_seed(15)
    a, b = torch.randn(2, 2, 32, 64), torch.randn(2, 2, 32, 64)
    rotated = rotate_pair(b, torch.tensor([.9, -.6]))
    torch.testing.assert_close(InvariantReadout()(b), InvariantReadout()(rotated), atol=2e-6, rtol=2e-6)
    old = relation_statistics(a[:, :, :8], b[:, :, :8], True)
    new = relation_statistics(a[:, :, :8], rotated[:, :, :8], True)
    assert (old - new).abs().max() > .005


@pytest.mark.parametrize('variant', VARIANTS)
def test_ce_connectivity_batch_independence_phase_and_diagnostics_state(variant):
    torch.manual_seed(21)
    model = build(variant)
    x = torch.randn(4, 2, 256)
    x[0].zero_()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0002, weight_decay=.0001)
    r = model.core.relation
    for step in range(3):
        model.train(); optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(x), torch.arange(4)).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert r.project.weight.grad.norm() > 0
        for name, p in r.named_parameters():
            if name == 'project.weight':
                continue
            assert (p.grad.norm() == 0) if step == 0 else (p.grad.norm() > 0)
        optimizer.step()
    model.eval()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    flags = [m.training for m in model.modules()]
    diagnostics = model.diagnostics(x)
    assert diagnostics['learned_readout']['records'][0]['relative_output_change_mean'] > 0
    assert flags == [m.training for m in model.modules()]
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())
    assert not r._forward_hooks
    with torch.no_grad():
        logits = model(x)
        torch.testing.assert_close(logits, torch.cat([model(row[None]) for row in x]), atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(logits, model(rotate_pair(x, torch.tensor([.3, -.7, .8, 1.1]))), atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize('shape', [(2, 8, 64), (1, 2, 8, 63), (1, 2, 7, 64)])
def test_reject_invalid_relation_shape(shape):
    with pytest.raises(ValueError, match='paired complex'):
        relation_statistics(torch.zeros(shape), torch.zeros(shape), True)
