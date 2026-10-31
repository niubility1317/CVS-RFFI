import copy
import pytest
import torch
from experiments.pamf_dura_efficiency.signature_context_prior import SignatureContextPrior
from experiments.pamf_dura_efficiency.signature_full_rate import FullRateSignaturePrior, BOUND
from test_dura_signature_context_prior import Core


@pytest.mark.parametrize('mode,added', [('plain', 1034), ('conditioned', 12626)])
def test_initial_IQ_SIG_state_and_rng_are_preserved(mode, added):
    torch.set_num_threads(2)
    torch.manual_seed(11)
    original = SignatureContextPrior(Core(), 31)
    reference = copy.deepcopy(original)
    rng = torch.get_rng_state().clone()
    model = FullRateSignaturePrior(original, 71, mode)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in reference.state_dict().items())
    assert sum(p.numel() for p in model.full_rate.parameters()) == added
    value = torch.randn(2, 9, 256)
    iq, sig = model.forward_with_signature(value)
    expected_iq, expected_sig = reference.forward_with_signature(value)
    torch.testing.assert_close(iq, expected_iq, atol=0, rtol=0)
    torch.testing.assert_close(sig, expected_sig, atol=0, rtol=0)
    iq.square().mean().backward()
    assert model.full_rate.output.weight.grad.abs().sum() > 0
    assert torch.equal(model.full_rate.input[0].weight.grad, torch.zeros_like(model.full_rate.input[0].weight))
    if mode == 'conditioned':
        assert torch.equal(model.full_rate.conditioner.weight.grad, torch.zeros_like(model.full_rate.conditioner.weight))


@pytest.mark.parametrize('mode', ['plain', 'conditioned'])
def test_full_rate_path_runs_at_256_and_reaches_output_without_batch_statistics(mode):
    torch.set_num_threads(2)
    torch.manual_seed(12)
    model = FullRateSignaturePrior(SignatureContextPrior(Core(), 32), 72, mode)
    with torch.no_grad():
        model.full_rate.output.weight.normal_(std=.1)
        if mode == 'conditioned':
            model.full_rate.conditioner.weight.normal_(std=.1)
    shapes = []
    hooks = [b.register_forward_hook(lambda _, args, out: shapes.append(tuple(out.shape))) for b in model.full_rate.blocks]
    value = torch.randn(3, 9, 256)
    try:
        output = model(value)
    finally:
        for hook in hooks:
            hook.remove()
    assert shapes == [(3, 12, 256)] * 3
    output.square().mean().backward()
    for parameter in [model.full_rate.input[0].weight, *[b[0].weight for b in model.full_rate.blocks]]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    modified = value.clone()
    modified[1] *= 10
    torch.testing.assert_close(model(modified)[0], output[0], atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(model(value[[2, 0, 1]]), output[[2, 0, 1]], atol=1e-7, rtol=1e-6)


def test_conditioned_path_uses_predicted_SIG_and_common_branch_initialization_matches():
    torch.set_num_threads(2)
    torch.manual_seed(13)
    original = SignatureContextPrior(Core(), 33)
    plain = FullRateSignaturePrior(copy.deepcopy(original), 73, 'plain')
    conditioned = FullRateSignaturePrior(copy.deepcopy(original), 73, 'conditioned')
    assert all(torch.equal(v, conditioned.full_rate.state_dict()[k]) for k, v in plain.full_rate.state_dict().items())
    with torch.no_grad():
        conditioned.full_rate.output.weight.normal_(std=.1)
        conditioned.full_rate.conditioner.weight.normal_(std=.1)
    value = torch.randn(2, 9, 256)
    first = conditioned(value)
    with torch.no_grad():
        conditioned.signature_projection.bias.add_(.3)
    second, signature = conditioned.forward_with_signature(value)
    assert not torch.allclose(first, second)
    second.square().mean().backward()
    for parameter in [conditioned.full_rate.conditioner.weight, conditioned.signature_projection.weight]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    assert (BOUND * conditioned.full_rate.conditioner(signature).tanh()).abs().max() <= BOUND
    with pytest.raises(ValueError):
        conditioned(torch.zeros(2, 8, 256))
