import pytest
import torch
from torch import nn
from torch.nn import functional as F

from experiments.cvs_equivariant_identity.model import ComplexConv, InvariantReadout, rotate_pair
from experiments.cvs_neural_residual_identity.model import build as control, neural_contract
from experiments.cvs_neural_readout_identity.model import (
    BASE_VARIANT, VARIANTS, LearnedInvariantReadout, build, readout_contract,
)


torch.set_num_threads(2)


@pytest.mark.parametrize('variant,new_parameters,total', zip(VARIANTS, [85160, 89256], [306147, 310243]))
def test_contract_parameters_old_state_and_scratch_rng_are_exact(variant, new_parameters, total):
    torch.manual_seed(17)
    base = control(BASE_VARIANT).eval()
    old_state = base.state_dict()
    expected_rng = torch.get_rng_state().clone()
    expected_following = torch.rand(16)
    torch.manual_seed(17)
    model = build(variant).eval()
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert torch.equal(torch.rand(16), expected_following)
    assert model.contract() == readout_contract(variant)
    assert sum(p.numel() for p in model.parameters()) == total
    assert sum(p.numel() for p in model.readout_parameters()) == new_parameters
    # The inherited residual parameter group remains exactly the old group.
    assert sum(p.numel() for p in model.residual_parameters()) == 18432
    assert not ({id(p) for p in model.readout_parameters()} & {id(p) for p in model.residual_parameters()})
    state = model.state_dict()
    for name, value in old_state.items():
        assert torch.equal(state[name], value), name
    expected = neural_contract(BASE_VARIANT)
    contract = model.contract()
    for name in ('phase_lag', 'actual_phase_lag', 'envelope_lag', 'actual_envelope_lag',
                 'neural_paths', 'neural_depth_per_path', 'whole_affine_phase_invariant'):
        assert contract[name] == expected[name]
    assert contract['base_trainable_parameters'] == 220987
    assert contract['arbitrary_channel_rx_invariant'] is False
    assert isinstance(model.core.readout, InvariantReadout)
    time, behavior = [block for _, block in model.readout_blocks()]
    assert time is not behavior
    assert not ({id(p) for p in time.parameters()} & {id(p) for p in behavior.parameters()})
    x = torch.randn(4, 2, 256)
    with torch.no_grad():
        torch.testing.assert_close(model(x), base(x), atol=0, rtol=0)


def test_paired_initialization_and_complex_identity():
    torch.manual_seed(21)
    plain = build(VARIANTS[0])
    torch.manual_seed(21)
    mixed = build(VARIANTS[1])
    z = torch.randn(3, 2, 32, 64)
    for (_, a), (_, b) in zip(plain.readout_blocks(), mixed.readout_blocks()):
        for name, value in a.scores.state_dict().items():
            assert torch.equal(value, b.scores.state_dict()[name]), name
        assert torch.equal(a.project.weight, b.project.weight)
        torch.testing.assert_close(b.mixer.weight_real, torch.eye(32).unsqueeze(-1), atol=0, rtol=0)
        assert torch.count_nonzero(b.mixer.weight_imag) == 0
        torch.testing.assert_close(b.mixer(z), z, atol=0, rtol=0)


@pytest.mark.parametrize('variant', VARIANTS)
def test_two_ce_steps_activate_attention_and_output_affects_logits(variant):
    torch.manual_seed(31)
    model = build(variant).train()
    x = torch.randn(4, 2, 256)
    labels = torch.tensor([0, 1, 2, 3])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0002, weight_decay=.0001)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(x), labels).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        for _, block in model.readout_blocks():
            assert block.project.weight.grad.norm() > 0
            hidden = list(block.scores.parameters()) + list(block.mixer.parameters())
            if step == 0:
                assert all(torch.count_nonzero(p.grad) == 0 for p in hidden)
            else:
                assert all(p.grad.norm() > 0 for p in hidden)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        actual = model(x)
        hooks = [block.register_forward_hook(lambda module, inputs, output: module.skip(inputs[0]))
                 for _, block in model.readout_blocks()]
        try:
            without_readouts = model(x)
        finally:
            for hook in hooks:
                hook.remove()
    assert (actual - without_readouts).abs().max() > 1e-5
    diagnostic = model.diagnostics(x)
    assert 'neural_residual' in diagnostic
    assert len(diagnostic['neural_residual']['records']) == 2
    records = diagnostic['learned_readout']['records']
    assert [record['block'] for record in records] == ['time.readout', 'behavior.readout']
    assert all(record['relative_output_change_mean'] > 0 for record in records)
    assert all(record['attention_sum_max_abs_error'] <= 4 * torch.finfo(x.dtype).eps for record in records)
    assert all(record['attention_heads'] == 4 and record['attention_tokens'] == 64 for record in records)
    assert all(1 <= record['attention_effective_tokens_mean'] <= 64.0001 for record in records)


@pytest.mark.parametrize('variant', VARIANTS)
def test_trained_readout_is_packet_local_and_global_phase_invariant_without_state_updates(variant):
    torch.manual_seed(41)
    model = build(variant).eval()
    # Exercise nonzero learned exits and nonidentity mixing, not only the zero skip.
    with torch.no_grad():
        for _, block in model.readout_blocks():
            block.project.weight.normal_(std=.01)
            if isinstance(block.mixer, ComplexConv):
                block.mixer.weight_real.add_(torch.randn_like(block.mixer.weight_real) * .03)
                block.mixer.weight_imag.normal_(std=.03)
    x = torch.randn(4, 2, 256)
    phase = torch.tensor([.43, -1.13, 2.2, .09])
    permutation = torch.tensor([2, 0, 3, 1])
    state = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        actual = model(x)
        torch.testing.assert_close(model(x[permutation]), actual[permutation], atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(torch.cat([model(packet[None]) for packet in x]), actual, atol=2e-4, rtol=2e-4)
        changed = x.clone()
        changed[1:] *= 3
        torch.testing.assert_close(model(changed)[:1], actual[:1], atol=0, rtol=0)
        torch.testing.assert_close(model(rotate_pair(x, phase)), actual, atol=2e-3, rtol=2e-3)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in state.items())
    model.train()
    # A mixed set of flags must also be restored exactly after diagnostics.
    model.core.time[0].eval()
    flags = [module.training for module in model.modules()]
    rng = torch.get_rng_state().clone()
    diagnostic = model.diagnostics(x)
    assert diagnostic['learned_readout']['active'] is True
    assert [module.training for module in model.modules()] == flags
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in state.items())
    assert all(not module._forward_hooks for _, block in model.readout_blocks() for module in block.modules())


@pytest.mark.parametrize('mixing', [False, True])
def test_readout_softmax_axis_formula_and_phase_invariance(mixing):
    torch.manual_seed(51)
    block = LearnedInvariantReadout(mixing)
    with torch.no_grad():
        block.project.weight.normal_(std=.03)
        if mixing:
            block.mixer.weight_imag.normal_(std=.1)
    z = torch.randn(3, 2, 32, 64)
    values = torch.log1p(block.mixer(z).square().sum(1))
    attention = block.attention(block.scores(values))
    assert attention.shape == (3, 4, 64)
    torch.testing.assert_close(attention.sum(-1), torch.ones(3, 4), atol=2e-7, rtol=0)
    pooled = (attention[:, :, None, :] * values[:, None, :, :]).sum(-1).flatten(1)
    expected = block.skip(z) + F.linear(pooled, block.project.weight)
    torch.testing.assert_close(block(z), expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(block(rotate_pair(z, torch.tensor([.1, -1., 2.]))), block(z), atol=2e-6, rtol=2e-5)
    with pytest.raises(ValueError, match='readout input'):
        block(z[..., :-1])


def test_contract_detects_actual_structure_and_parameter_changes():
    model = build(VARIANTS[0])
    model.core.time_readout.scores[0] = nn.Conv1d(32, 16, 1)
    actual = model.contract()
    assert actual['learned_readout_active'] is False
    assert actual['new_trainable_parameters'] == 85160 - 1024
    model = build(VARIANTS[0])
    model.core.behavior_readout = model.core.time_readout
    assert model.contract()['learned_readout_active'] is False


def test_invalid_variant_rejected_before_backbone_initialization():
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match='Unregistered'):
        build('unknown')
    assert torch.equal(torch.get_rng_state(), rng)
