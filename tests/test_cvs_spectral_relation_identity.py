"""Independent operator formulas and scratch wiring on synthetic tensors only."""
import io
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from experiments.cvs_neural_residual_identity.model import build as control
from experiments.cvs_spectral_relation_identity.model import (
    BASE_VARIANT, BASE_PARAMETERS, NEW_PARAMETERS, VARIANTS, POWER_FLOOR, RELATIVE_POWER_FLOOR,
    SpectralTemporalRelation, build, relation_contract,
)

torch.set_num_threads(2)


@pytest.mark.parametrize('variant', VARIANTS)
def test_scratch_base_function_state_rng_counts_and_contract(variant):
    torch.manual_seed(83)
    base = control(BASE_VARIANT)
    base_state, rng = base.state_dict(), torch.get_rng_state().clone()
    torch.manual_seed(83)
    model = build(variant)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in base_state.items())
    assert model.contract() == relation_contract(variant)
    assert sum(p.numel() for p in model.parameters()) == BASE_PARAMETERS + NEW_PARAMETERS == 247731
    assert sum(p.numel() for p in model.spectral_relation_parameters()) == 26744
    x = torch.randn(3, 2, 256)
    for training in (False, True):
        base.train(training); model.train(training)
        torch.manual_seed(401)
        with torch.no_grad(): expected = base(x)
        base_rng = torch.get_rng_state().clone()
        torch.manual_seed(401)
        with torch.no_grad(): actual = model(x)
        assert torch.equal(actual, expected)
        assert torch.equal(torch.get_rng_state(), base_rng)
    model.eval()
    with torch.no_grad(): torch.testing.assert_close(model.features(x), model.core.features(x), atol=0, rtol=0)


def test_variants_have_identical_initialized_parameters_and_buffers():
    torch.manual_seed(91); a = build(VARIANTS[0])
    torch.manual_seed(91); b = build(VARIANTS[1])
    assert a.state_dict().keys() == b.state_dict().keys()
    assert all(torch.equal(value, b.state_dict()[key]) for key, value in a.state_dict().items())
    assert a.core.spectral_relation.per_frequency is False
    assert b.core.spectral_relation.per_frequency is True


def test_STFT_matches_independent_direct_DFT_and_signed_frequency_shift():
    module = SpectralTemporalRelation(True).double()
    x = torch.randn(2, 2, 256, dtype=torch.float64)
    complex_iq = torch.complex(x[:, 0], x[:, 1])
    sample = torch.arange(64, dtype=torch.float64)
    frequency = torch.arange(-32, 32, dtype=torch.float64)
    dft = torch.exp(-2j*math.pi*frequency[:, None]*sample[None]/64)
    # Stored FP32 window is the deployed deterministic Hann, promoted by double().
    direct = torch.stack([torch.einsum('fn,bn->bf', dft,
                         complex_iq[:, start:start+64]*module.window)/module.window.norm()
                          for start in range(0, 193, 32)], -1)
    actual = module.spectral(x)
    assert actual.shape == (2, 2, 64, 7)
    torch.testing.assert_close(torch.complex(actual[:, 0], actual[:, 1]), direct, atol=2e-14, rtol=2e-13)
    # Positive and negative complex tones locate opposite signed bins, no real FFT.
    n = torch.arange(256, dtype=torch.float64)
    for tone in (7, -11):
        signal = torch.exp(2j*math.pi*tone*n/64)
        paired = torch.stack((signal.real, signal.imag), 0)[None]
        power = module.spectral(paired).square().sum(1)
        assert power.argmax(1).tolist() == [[32+tone]*7]


@pytest.mark.parametrize('per_frequency', [False, True])
def test_complex_mixing_and_outer_product_match_independent_formula(per_frequency):
    torch.manual_seed(90)
    module = SpectralTemporalRelation(per_frequency).double()
    s = torch.randn(2, 2, 64, 7, dtype=torch.float64)
    sc = torch.complex(s[:, 0], s[:, 1])
    a = torch.complex(module.mix_real, module.mix_imag)
    expected_v = torch.einsum('kt,bft->bfk', a, sc)
    v = module.mix_spectra(s)
    torch.testing.assert_close(torch.complex(v[:, 0], v[:, 1]), expected_v, atol=2e-15, rtol=2e-14)
    per_bin_power = expected_v.abs().square().sum(-1)
    power = per_bin_power if per_frequency else per_bin_power.mean(-1, keepdim=True)
    floor = (per_bin_power.mean(-1, keepdim=True)*RELATIVE_POWER_FLOOR).clamp_min(POWER_FLOOR)
    expected_q = expected_v[..., :, None]*expected_v[..., None, :].conj()/torch.maximum(power, floor)[..., None, None]
    q = module.relations(v)
    actual = torch.complex(q[:, 0], q[:, 1])
    torch.testing.assert_close(actual, expected_q, atol=2e-15, rtol=2e-14)
    torch.testing.assert_close(actual, actual.conj().transpose(-1, -2), atol=0, rtol=0)
    eigenvalues = torch.linalg.eigvalsh(actual)
    assert eigenvalues[..., 0].min() > -1e-12
    assert eigenvalues[..., :-1].abs().max() < 1e-12
    # Non-real product fixes v_i conj(v_j), not the conjugate convention.
    example = torch.zeros(1, 2, 64, 4, dtype=torch.float64)
    example[:, 0, :, 0] = 1; example[:, 1, :, 1] = 1
    assert (module.relations(example)[:, 1, :, 0, 1] < 0).all()


@pytest.mark.parametrize('per_frequency', [False, True])
def test_ideal_gain_properties_bounds_and_floor_counterexample(per_frequency):
    torch.manual_seed(45)
    module = SpectralTemporalRelation(per_frequency).double()
    s = torch.randn(2, 2, 64, 7, dtype=torch.float64)
    sc = torch.complex(s[:, 0], s[:, 1])
    phase = torch.linspace(-1, 2, 64, dtype=torch.float64)
    amplitude = torch.linspace(.75, 2, 64, dtype=torch.float64)
    gain = amplitude*torch.exp(1j*phase)
    v = module.mix_spectra(s)
    q = module.relations(v)
    shifted = sc*gain[None, :, None]
    sv = module.mix_spectra(torch.stack((shifted.real, shifted.imag), 1))
    assert not module.floor_active(v).any() and not module.floor_active(sv).any()
    if per_frequency:
        torch.testing.assert_close(module.relations(sv), q, atol=2e-14, rtol=2e-13)
    else:
        assert (module.relations(sv)-q).abs().max() > .1
        phase_only = sc*torch.exp(1j*phase)[None, :, None]*2
        phase_v = module.mix_spectra(torch.stack((phase_only.real, phase_only.imag), 1))
        torch.testing.assert_close(module.relations(phase_v), q, atol=2e-14, rtol=2e-13)
    trace = q[:, 0].diagonal(dim1=-2, dim2=-1).sum(-1)
    norm = q.square().sum((1, 3, 4)).sqrt()
    torch.testing.assert_close(trace, norm, atol=2e-14, rtol=2e-13)
    assert float(norm.max().detach()) <= (1 if per_frequency else 64)+1e-12
    concentrated = torch.zeros(1, 2, 64, 4, dtype=torch.float64)
    concentrated[:, 0, 3, 0] = 1
    limit = module.relations(concentrated)[0, 0, 3, 0, 0]
    assert float(limit) == (1 if per_frequency else 64)
    tiny = torch.ones_like(v)*1e-5
    torch.testing.assert_close(module.relations(tiny*2), module.relations(tiny)*4)
    assert not torch.allclose(module.relations(tiny*2), module.relations(tiny))


@pytest.mark.parametrize('per_frequency', [False, True])
@pytest.mark.parametrize('scale', [0., 1e-12])
def test_zero_and_floor_inputs_have_finite_forward_and_gradients(per_frequency, scale):
    module = SpectralTemporalRelation(per_frequency)
    x = (torch.randn(2, 2, 256)*scale).requires_grad_()
    q = module.statistics(x)
    q.sum().backward()
    assert torch.isfinite(q).all() and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in (module.mix_real, module.mix_imag))
    if scale == 0: assert torch.count_nonzero(q) == 0
    module.zero_grad(set_to_none=True)
    module(x.detach()).square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())


def test_relation_channel_order_is_real_then_imaginary_row_major():
    module = SpectralTemporalRelation(True)
    q = torch.arange(2*64*4*4, dtype=torch.float32).reshape(1, 2, 64, 4, 4)
    module.statistics = lambda x: q
    seen = []
    hook = module.encoder[0].register_forward_pre_hook(lambda m, args: seen.append(args[0].clone()))
    try: module(torch.zeros(1, 2, 256))
    finally: hook.remove()
    for component in range(2):
        for i in range(4):
            for j in range(4):
                assert torch.equal(seen[0][0, component*16+i*4+j], q[0, component, :, i, j])


@pytest.mark.parametrize('per_frequency', [False, True])
def test_relative_floor_formula_weak_bin_stability_and_finite_gradients(per_frequency):
    module = SpectralTemporalRelation(per_frequency).double()
    v = torch.zeros(1, 2, 64, 4, dtype=torch.float64)
    v[:, 0, :, 0] = 1
    v[:, 0, -1, 0] = .001
    v.requires_grad_()
    expected_floor = max((63+1e-6)/64/64, POWER_FLOOR)
    torch.testing.assert_close(module.energy_floor(v), torch.tensor([[expected_floor]], dtype=torch.float64))
    q = module.relations(v)
    if per_frequency:
        assert module.floor_active(v).sum() == 1
        torch.testing.assert_close(q[0, 0, -1, 0, 0], torch.tensor(1e-6/expected_floor, dtype=torch.float64))
        changed = v.detach().clone(); changed[:, 0, -1, 0] = 0; changed[:, 0, -1, 1] = .001
        # The previously unit-normalized weak-bin direction now carries only its
        # explicit energy/floor weight. This is a numerical operator check.
        distance = (module.relations(changed)-q).square().sum().sqrt()
        torch.testing.assert_close(distance, torch.tensor(math.sqrt(2)*1e-6/expected_floor, dtype=torch.float64))
        gain_changed = v.detach().clone(); gain_changed[..., -1, :] *= 2
        assert not torch.allclose(module.relations(gain_changed), q)
        torch.testing.assert_close(module.relations(v*2), q, atol=1e-15, rtol=1e-14)
    else:
        assert not module.floor_active(v).any()
        # Relative floor does not change the packet-energy control's arithmetic.
        ar, ai = v.unbind(1)
        numerator = torch.stack((ar[..., :, None]*ar[..., None, :]+ai[..., :, None]*ai[..., None, :],
                                 ai[..., :, None]*ar[..., None, :]-ar[..., :, None]*ai[..., None, :]), 1)
        old = numerator/module.energy(v).clamp_min(POWER_FLOOR)[:, None, :, None, None]
        assert torch.equal(q, old)
    q.square().sum().backward()
    assert torch.isfinite(v.grad).all()


@pytest.mark.parametrize('variant', VARIANTS)
def test_actual_frequency_injection_preserves_other_fusion_inputs(variant):
    torch.manual_seed(57); base = control(BASE_VARIANT).eval()
    torch.manual_seed(57); model = build(variant).eval()
    with torch.no_grad(): model.core.spectral_relation.project.weight.normal_(std=.01)
    x = torch.randn(3, 2, 256); seen = [[], [], []]
    hooks = [base.core.id_backbone.fuse.register_forward_pre_hook(lambda m, a: seen[0].append(a[0].clone())),
             model.core.id_backbone.fuse.register_forward_pre_hook(lambda m, a: seen[1].append(a[0].clone())),
             model.core.spectral_relation.register_forward_pre_hook(lambda m, a: seen[2].append(a[0].clone()))]
    try:
        with torch.no_grad(): base(x); model(x)
    finally:
        for hook in hooks: hook.remove()
    assert len(seen[2]) == 1 and torch.equal(seen[2][0], x)
    before, after = seen[0][0], seen[1][0]
    assert torch.equal(before[:, :160], after[:, :160])
    assert torch.equal(before[:, 320:], after[:, 320:])
    with torch.no_grad(): delta = model.core.spectral_relation(x)
    torch.testing.assert_close(after[:, 160:320]-before[:, 160:320], delta, atol=2e-7, rtol=1e-5)
    assert delta.norm() > 0


@pytest.mark.parametrize('variant', VARIANTS)
def test_three_CE_updates_use_all_new_parameters_and_readonly_diagnostics(variant):
    torch.manual_seed(29); model = build(variant)
    x = torch.randn(4, 2, 256); x[0].zero_()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0002, weight_decay=.0001)
    r = model.core.spectral_relation
    for step in range(3):
        model.train(); optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(x), torch.arange(4)).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        for name, p in r.named_parameters():
            if step == 0 and name != 'project.weight': assert p.grad.norm() == 0
            else: assert p.grad.norm() > 0
        optimizer.step()
    model.train(); model.core.id_backbone.f_proj.eval()
    flags = [m.training for m in model.modules()]
    state = {k: v.clone() for k, v in model.state_dict().items()}
    grads = [p.grad.clone() for p in model.parameters()]
    rng = torch.get_rng_state().clone()
    diagnostics = model.diagnostics(x)['spectral_relation']; record = diagnostics['records'][0]
    assert diagnostics['active'] and record['packets'] == 4
    assert record['relative_output_change_mean'] > 0 and record['projection_norm'] > 0
    assert record['mix_gradient_norm'] > 0 and record['encoder_gradient_norm'] > 0
    assert .25 <= record['floor_fraction'] <= 1
    assert len(record['relation_norm_by_frequency']) == 64
    assert flags == [m.training for m in model.modules()] and torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())
    assert all(torch.equal(g, p.grad) for g, p in zip(grads, model.parameters()))
    assert not r._forward_hooks and not model.core.id_backbone.f_proj._forward_hooks
    model.eval()
    with torch.no_grad():
        logits = model(x)
        torch.testing.assert_close(logits, torch.cat([model(row[None]) for row in x]), atol=2e-4, rtol=2e-4)
        alone = r(x[:1]); together = r(torch.cat((x[:1], torch.randn(3, 2, 256))))[:1]
        torch.testing.assert_close(alone, together, atol=2e-7, rtol=2e-6)
    serialized = io.BytesIO(); torch.save(model.state_dict(), serialized); serialized.seek(0)
    restored = build(variant).eval(); restored.load_state_dict(torch.load(serialized, weights_only=True), strict=True)
    with torch.no_grad(): torch.testing.assert_close(restored(x), logits, rtol=0, atol=0)
    assert restored.contract() == relation_contract(variant)


@pytest.mark.parametrize('change', ['normalization', 'window', 'mix_shape', 'kernel', 'activation',
                                   'pool', 'project_bias', 'hop', 'floor', 'relative_floor', 'bypassed_forward', 'bypassed_core'])
def test_contract_detects_changed_actual_structure(change):
    model = build(VARIANTS[0]); r = model.core.spectral_relation
    if change == 'normalization': r.per_frequency = True
    elif change == 'window': r.window[3] += .01
    elif change == 'mix_shape': r.mix_real = nn.Parameter(torch.zeros(4, 8))
    elif change == 'kernel': r.encoder[0] = nn.Conv1d(32, 32, 5, padding=2)
    elif change == 'activation': r.encoder[1] = nn.ReLU()
    elif change == 'pool': r.encoder[4] = nn.AdaptiveAvgPool1d(2)
    elif change == 'project_bias': r.project = nn.Linear(128, 160)
    elif change == 'hop': r.frame_hop = 16
    elif change == 'floor': r.power_floor = .1
    elif change == 'relative_floor': r.relative_power_floor = .1
    elif change == 'bypassed_forward': r.forward = lambda x: x.new_zeros(len(x), 160)
    elif change == 'bypassed_core': model.core.features = lambda x: x.new_zeros(len(x), 160)
    assert model.contract()['spectral_relation_active'] is False


@pytest.mark.parametrize('method,shape', [('spectral', (2, 2, 255)), ('spectral', (2, 256)),
                                        ('mix_spectra', (2, 2, 7, 64)), ('relations', (2, 2, 4, 64))])
def test_shape_contracts(method, shape):
    with pytest.raises(ValueError): getattr(SpectralTemporalRelation(True), method)(torch.zeros(shape))


def test_diagnostic_hooks_removed_on_failure():
    model = build(VARIANTS[0]); before = [m.training for m in model.modules()]
    def fail(x): raise RuntimeError('synthetic failure')
    model.features = fail
    with pytest.raises(RuntimeError): model.spectral_relation_diagnostics(torch.zeros(1, 2, 256))
    assert before == [m.training for m in model.modules()]
    assert not model.core.spectral_relation._forward_hooks and not model.core.id_backbone.f_proj._forward_hooks
