import copy
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from experiments.cvs_equivariant_identity.model import rotate_pair, behavior_basis
from experiments.cvs_neural_residual_identity.model import build as control
from experiments.cvs_frontfilter_identity.model import (
    BASE_VARIANT, VARIANTS, RHO, build, filter_contract,
)
from experiments.cvs_frontfilter_identity.physics import frozen_synthetic_diagnostics

torch.set_num_threads(2)
SEEDS = (2026092701, 2026092702, 2026092703, 2026092704)


def activate(model, scale=.15):
    with torch.no_grad():
        if model.frontfilter_variant == VARIANTS[0]:
            model.frontfilter.coeff_raw.normal_(std=scale)
        else:
            model.frontfilter.context[-1].weight.normal_(std=scale)
            model.frontfilter.context[-1].bias.normal_(std=scale)


@pytest.mark.parametrize('seed', SEEDS)
def test_all_seed_exact_scratch_function_states_rng_and_shared_basis(seed):
    torch.manual_seed(seed)
    base = control(BASE_VARIANT).eval()
    state = base.state_dict()
    rng = torch.get_rng_state().clone()
    x = torch.randn(3, 2, 256)
    basis = []
    for variant in VARIANTS:
        torch.manual_seed(seed)
        model = build(variant).eval()
        assert torch.equal(rng, torch.get_rng_state())
        assert all(torch.equal(value, model.state_dict()[key]) for key, value in state.items())
        with torch.no_grad():
            assert torch.equal(model.features(x), base.features(x))
            assert torch.equal(model(x), base(x))
        basis.append(model.frontfilter.basis.detach().clone())
    assert torch.equal(*basis)


@pytest.mark.parametrize('variant,total', zip(VARIANTS, (221035, 221307)))
def test_contract_parameter_count_strict_load_and_actual_bound(variant, total):
    model = build(variant).eval()
    assert model.contract() == filter_contract(variant)
    assert model.contract()['total_parameters'] == total
    assert sum(p.numel() for p in model.frontfilter_parameters()) == total - 220987
    activate(model)
    replica = build(variant).eval()
    replica.load_state_dict(model.state_dict(), strict=True)
    x = torch.randn(3, 2, 256)
    with torch.no_grad():
        assert torch.equal(model(x), replica(x))
    with pytest.raises(RuntimeError):
        replica.load_state_dict(control(BASE_VARIANT).state_dict(), strict=True)
    model.frontfilter.rho = .7
    assert not model.contract()['frontfilter_active']


@pytest.mark.parametrize('variant', VARIANTS)
def test_nonzero_fir_fixed_coefficient_singular_bound_phase_and_packet_independence(variant):
    torch.manual_seed(91)
    model = build(variant).double().eval()
    activate(model, 10.)
    g = model.frontfilter
    with torch.no_grad():
        g.basis.mul_(30.)
    x = torch.randn(4, 2, 256, dtype=torch.float64)
    y, a = g(x)
    assert torch.linalg.vector_norm(a, dim=1).sum(-1).max() <= 1 + 1e-14
    assert torch.linalg.vector_norm(g.effective_basis(), dim=1).sum(-1).max() <= 1 + 1e-14
    phase = x.new_tensor([.3, -.7, 1.8, 2.4])
    torch.testing.assert_close(g(rotate_pair(x, phase))[0], rotate_pair(y, phase), atol=1e-13, rtol=1e-12)
    torch.testing.assert_close(torch.cat([g(v[None])[0] for v in x]), y, atol=1e-13, rtol=1e-12)
    changed = x.clone()
    changed[1:] *= 4
    assert torch.equal(g(changed)[0][:1], y[:1])
    # Build the real finite-dimensional matrix for one frozen packet kernel.
    # Each example is one coordinate basis vector; same fixed coefficients.
    length = 17
    eye = torch.eye(2 * length, dtype=torch.float64).reshape(2 * length, 2, 1, length)
    matrix = g.apply_filter(eye, a[:1].expand(2 * length, -1, -1)).flatten(1).T
    singular = torch.linalg.svdvals(matrix)
    assert singular.min() >= 1 - RHO - 1e-12
    assert singular.max() <= 1 + RHO + 1e-12
    norms = x.flatten(1).norm(dim=1)
    assert ((y - x).flatten(1).norm(dim=1) <= RHO * norms + 1e-12).all()


@pytest.mark.parametrize('variant', VARIANTS)
def test_all_backbone_input_paths_receive_filtered_iq_without_renormalization(variant):
    torch.manual_seed(72)
    model = build(variant).eval()
    activate(model)
    x = torch.randn(4, 2, 256)
    gx = model.frontfilter(x)[0]
    assert not torch.equal(gx, x)
    b = model.core.id_backbone
    captures = []
    original_sinc, original_spectral = b._sinc_on_iq, b._mirror_compressed_features

    def sinc(q):
        captures.append(('time', q.detach().clone()))
        return original_sinc(q)

    def spectral(q, **kwargs):
        captures.append(('frequency', q.detach().clone()))
        return original_spectral(q, **kwargs)

    def behavior(q):
        captures.append(('behavior', q.detach().clone()))
        return behavior_basis(q)

    with patch.object(b, '_sinc_on_iq', sinc), patch.object(b, '_mirror_compressed_features', spectral), patch(
            'experiments.cvs_equivariant_identity.model.behavior_basis', behavior):
        model.features(x)
    assert [name for name, _ in captures] == ['time', 'behavior', 'frequency']
    assert all(torch.equal(actual, gx) for _, actual in captures)


@pytest.mark.parametrize('variant', VARIANTS)
def test_fp32_single_ce_three_steps_reaches_every_filter_parameter(variant):
    torch.manual_seed(37)
    model = build(variant).train()
    x, labels = torch.randn(4, 2, 256), torch.tensor([0, 1, 2, 3])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0002, weight_decay=.0001)
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        assert logits.dtype == torch.float32
        F.cross_entropy(logits, labels).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        if step == 0:
            assert model.frontfilter.basis.grad.norm() == 0
            exit_params = ([model.frontfilter.coeff_raw] if variant == VARIANTS[0]
                           else model.frontfilter.context[-1].parameters())
            assert all(p.grad.norm() > 0 for p in exit_params)
        if step == 2:
            assert all(p.grad.norm() > 0 for p in model.frontfilter_parameters())
        optimizer.step()
    record = model.diagnostics(x)['frontfilter']['records'][0]
    assert record['relative_input_change_mean'] > 0
    assert record['basis_gradient_norm'] > 0


@pytest.mark.parametrize('variant', VARIANTS)
def test_quiet_inputs_finite_and_diagnostics_preserve_rng_modes_state(variant):
    torch.manual_seed(81)
    model = build(variant)
    activate(model)
    x = torch.cat([torch.zeros(1, 2, 256), torch.randn(1, 2, 256) * 1e-15,
                   torch.randn(2, 2, 256)], dim=0)
    model.eval()
    with torch.no_grad():
        assert torch.isfinite(model(x)).all()
        assert torch.count_nonzero(model.frontfilter(x)[0][0]) == 0
    model.train()
    model.core.time[0].eval()
    flags = [m.training for m in model.modules()]
    state = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    d = model.diagnostics(x)
    assert torch.equal(rng, torch.get_rng_state())
    assert flags == [m.training for m in model.modules()]
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in state.items())
    assert 'original_input_x' == d['diagnostic_input_scopes']['coordinate_cfo_and_coherence']
    assert all(not m._forward_hooks for m in model.modules())
    assert set(('adaptive_input', 'neural_residual', 'normalization', 'frontfilter')) <= set(d)
    r = d['frontfilter']['records'][0]
    assert r['input_norm_ratio_eligible_packets'] == 3
    assert .75 - 1e-6 <= r['input_norm_ratio_min'] <= r['input_norm_ratio_max'] <= 1.25 + 1e-6
    quiet = model.frontfilter_diagnostics(torch.zeros(2, 2, 256))['records'][0]
    assert quiet['input_norm_ratio_min'] is None and quiet['input_norm_ratio_max'] is None


def test_static_coefficients_global_dynamic_coefficients_packet_specific():
    x = torch.randn(4, 2, 256)
    x[1] *= 5
    for variant in VARIANTS:
        model = build(variant)
        activate(model)
        coeff = model.frontfilter.coefficients(x)
        value = model.frontfilter_diagnostics(x)['records'][0]['coefficient_packet_variance']
        if variant == VARIANTS[0]:
            assert torch.equal(coeff[0], coeff[1]) and value == 0
        else:
            assert not torch.equal(coeff[0], coeff[1]) and value > 0


@pytest.mark.parametrize('variant', VARIANTS)
def test_public_physics_uses_actual_frontfiltered_forward_thirty_packets(variant):
    model = build(variant).train()
    activate(model)
    rng = torch.get_rng_state().clone()
    diagnostic = frozen_synthetic_diagnostics(model)
    assert model.training and torch.equal(rng, torch.get_rng_state())
    assert len(diagnostic['records']) == 30
    assert diagnostic['frontfilter']['records'][0]['packets'] == 30
    assert diagnostic['frontfilter']['records'][0]['relative_input_change_mean'] > 0
    assert diagnostic['phase_tolerance_pass']
    assert not diagnostic['target_access'] and not diagnostic['formal_data_access']
    assert diagnostic['adaptive_input']['records']


def test_unknown_variant_does_not_consume_rng():
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match='Unregistered'):
        build('unknown')
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize('variant', VARIANTS)
def test_trained_phase_property_separates_dc_spectral_roundoff_from_frontfilter(variant):
    # A near-identity zero-padded FIR gives a pure DC packet tiny edge spectra.
    # The retained FP32 logR/asym frequency features amplify roundoff there.
    # Do not assert a uniform full-network FP32 tolerance on that degenerate
    # input; test the FIR itself and the same-state double-precision network.
    torch.manual_seed(SEEDS[0])
    model = build(variant).train()
    x = torch.randn(4, 2, 256, generator=torch.Generator().manual_seed(2026100300))
    x[0].zero_()
    x[1].fill_(1.)
    t = torch.arange(256, dtype=x.dtype)
    x[2] = torch.stack([torch.cos(.13 * t) + .15 * torch.cos(.39 * t),
                        torch.sin(.13 * t) + .15 * torch.sin(.39 * t)])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0002, weight_decay=.0001)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(x), torch.arange(4)).backward()
        optimizer.step()
    model.eval()
    phase = torch.tensor([.43, -.6, 1.2, -2.])
    with torch.no_grad():
        rotated = rotate_pair(x, phase)
        logits, other = model(x), model(rotated)
        assert torch.isfinite(logits).all() and torch.isfinite(other).all()
        torch.testing.assert_close(logits[[0, 2, 3]], other[[0, 2, 3]], atol=2e-3, rtol=2e-3)
        gx, coeff = model.frontfilter(x)
        gr, coeffr = model.frontfilter(rotated)
        torch.testing.assert_close(gr, rotate_pair(gx, phase), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(coeffr, coeff, atol=1e-7, rtol=1e-6)
        high = copy.deepcopy(model).double()
        high_x = x.double()
        torch.testing.assert_close(high(high_x), high(rotate_pair(high_x, phase.double())),
                                   atol=1e-5, rtol=1e-5)
