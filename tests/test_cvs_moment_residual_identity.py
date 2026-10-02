import math
import pytest
import torch

from experiments.cvs_adaptive_volterra_identity.model import build as build_control, adaptive_basis_from_clipped
from experiments.cvs_moment_residual_identity.model import build, moment_basis, moment_contract, VARIANTS


@pytest.mark.parametrize('variant', VARIANTS)
def test_own_scratch_initial_function_and_shared_parameters_exactly_retain_control(variant):
    torch.set_num_threads(2)
    torch.manual_seed(18)
    control = build_control('adaptive_volterra_lag4')
    torch.manual_seed(18)
    model = build(variant)
    assert sum(p.numel() for p in model.parameters()) == 202557
    assert model.contract() == moment_contract(variant)
    for key, value in control.state_dict().items():
        assert torch.equal(value, model.state_dict()[key]), key
    x = torch.randn(4, 2, 256)
    # The retained PA/frequency projections contain train-mode dropout.
    # Compare the complete initial stochastic function under the same RNG.
    torch.manual_seed(80)
    reference = control(x)
    torch.manual_seed(80)
    assert torch.equal(reference, model(x))
    control.eval()
    model.eval()
    assert torch.equal(control(x), model(x))
    assert model.core.behavior[0].projection_raw.tolist() == [0., 0.]


@pytest.mark.parametrize('lag', [0, 4])
def test_zero_new_gates_retain_nonzero_phase_memory_function_and_packet_independence(lag):
    torch.manual_seed(21)
    z = torch.randn(4, 2, 256, dtype=torch.float64)
    phase = torch.tensor([.2, -.3], dtype=z.dtype)
    projection = torch.zeros(2, dtype=z.dtype)
    result = moment_basis(z, phase, projection, lag)
    assert torch.equal(result, adaptive_basis_from_clipped(z, 4, phase))
    projection = torch.tensor([-.1, .2], dtype=z.dtype)
    result = moment_basis(z, phase, projection, lag)
    assert torch.equal(result[:1], moment_basis(z[:1], phase, projection, lag))
    assert torch.equal(result[:, :, 0], z)


@pytest.mark.parametrize('lag', [0, 4])
@pytest.mark.parametrize('omega', [0., .08])
def test_learned_residual_input_has_delay_correct_affine_phase_covariance(lag, omega):
    torch.manual_seed(36)
    z = torch.randn(3, 2, 256, dtype=torch.float64)
    phase = z.new_tensor([.15, -.1])
    projection = z.new_tensor([-.08, .12])
    angle = .37 + omega * torch.arange(256, dtype=z.dtype)
    def rotate(x, a):
        return torch.stack((x[:, 0]*a.cos()-x[:, 1]*a.sin(), x[:, 0]*a.sin()+x[:, 1]*a.cos()), 1)
    result = moment_basis(z, phase, projection, lag)
    other = moment_basis(rotate(z, angle), phase, projection, lag)
    for m in range(4):
        for order in range(3):
            assert (other[:, :, 3*m+order] - rotate(result[:, :, 3*m+order], angle-omega*m)).abs().max() < 2e-12


@pytest.mark.parametrize('variant', VARIANTS)
def test_actual_ce_updates_all_four_gates_and_all_identity_parameters(variant):
    torch.set_num_threads(2)
    torch.manual_seed(95)
    model = build(variant)
    x, y = torch.randn(4, 2, 256), torch.tensor([0, 1, 2, 3])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0002)
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    assert sum(p.numel() for p in model.parameters() if p.grad is not None) == 202557
    for gate in (model.core.behavior[0].mix_raw, model.core.behavior[0].projection_raw):
        assert gate.grad is not None and torch.isfinite(gate.grad).all()
        assert (gate.grad.abs() > 1e-10).all()
    optimizer.step()
    assert (model.core.behavior[0].projection_raw.abs() > 0).all()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    diagnostics = model.diagnostics(x)
    record = diagnostics['moment_input']['records'][0]
    assert record['input_formula_max_abs_error'] == 0
    assert record['raw_order1_max_abs_error'] == 0
    assert record['degree3_moment_residual_relative_change_mean'] > 0
    assert record['degree5_moment_residual_relative_change_mean'] > 0
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())


@pytest.mark.parametrize('lag', [0, 4])
@pytest.mark.parametrize('amplitude', [0., 1e-15, 1.])
def test_degenerate_packets_and_moment_gradients_are_finite(lag, amplitude):
    z = torch.zeros(2, 2, 256, dtype=torch.float64)
    z[:, 0] = amplitude
    z.requires_grad_()
    phase = z.new_tensor([.1, -.1], requires_grad=True)
    projection = z.new_tensor([-.2, .2], requires_grad=True)
    result = moment_basis(z, phase, projection, lag)
    result.square().mean().backward()
    assert torch.isfinite(result).all() and torch.isfinite(z.grad).all()
    assert torch.isfinite(phase.grad).all() and torch.isfinite(projection.grad).all()


def test_unregistered_variant_gate_shape_and_lag_rejected():
    with pytest.raises(ValueError):
        build('unregistered')
    with pytest.raises(ValueError):
        moment_basis(torch.zeros(2, 2, 256), torch.zeros(3), torch.zeros(2), 4)
    with pytest.raises(ValueError):
        moment_basis(torch.zeros(2, 2, 256), torch.zeros(2), torch.zeros(2), 1)
