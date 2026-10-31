import io
import pytest
import torch
from experiments.pamf_phase3_restoration.model import (
    SpectralFilter, scratch_model, architecture, PARAMETERS, CONV_LINEAR_MAC)


def test_independent_factory_preserves_rng_identity_output_and_strict_state_reload():
    torch.set_num_threads(2)
    rng = torch.get_rng_state().clone()
    model = scratch_model(71)
    assert torch.equal(rng, torch.get_rng_state())
    assert sum(p.numel() for p in model.parameters()) == PARAMETERS
    assert model.initialization_report['ancestors'] == model.initialization_report['checkpoint_sources'] == []
    x = torch.randn(2, 2, 256)
    torch.testing.assert_close(model(x), x, atol=0, rtol=0)
    stream = io.BytesIO(); torch.save(model.state_dict(), stream); stream.seek(0)
    other = scratch_model(72); other.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    assert all(torch.equal(v, other.state_dict()[k]) for k, v in model.state_dict().items())
    torch.testing.assert_close(other(x), x, atol=0, rtol=0)


def test_actual_output_gradient_and_time_frequency_gate_paths_after_output_moves():
    torch.set_num_threads(2)
    model = scratch_model(73); x = torch.randn(2, 2, 256); target = torch.randn_like(x)
    (model(x)-target).square().mean().backward()
    assert model.output.weight.grad.abs().sum() > 0
    assert torch.equal(model.stem[0].weight.grad, torch.zeros_like(model.stem[0].weight))
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.output.weight.normal_(std=.01)
        for block in model.blocks:
            block.gate[-1].weight.normal_(std=.01)
    (model(x)-target).square().mean().backward()
    selected = [model.stem[0].weight]
    for b in model.blocks:
        selected += [b.time_depthwise.weight, b.time_projection.weight, b.spectral.real,
                     b.spectral.imaginary, b.frequency_projection.weight, b.gate[0].weight, b.gate[-1].weight]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in selected)


def test_spectral_endpoints_forward_and_parameter_gradients_match_full_hermitian_DFT():
    torch.set_num_threads(2)
    torch.manual_seed(74); layer = SpectralFilter(3).double(); x = torch.randn(2, 3, 256, dtype=torch.float64)
    weight = layer.weight(); assert torch.equal(weight[:, [0, -1]].imag, torch.zeros(3, 2, dtype=torch.float64))
    hermitian = torch.cat([weight, weight[:, 1:-1].flip(-1).conj()], dim=-1)
    reference = torch.fft.ifft(torch.fft.fft(x, norm='ortho') * hermitian, norm='ortho').real
    actual = layer(x); torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)
    parameters = [layer.real, layer.imaginary]
    a = torch.autograd.grad(actual.square().mean(), parameters, retain_graph=True)
    b = torch.autograd.grad(reference.square().mean(), parameters)
    for left, right in zip(a, b): torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-12)


def test_output_uses_only_current_record_and_permutation_is_equivariant():
    torch.set_num_threads(2)
    model = scratch_model(75)
    with torch.no_grad(): model.output.weight.normal_(std=.01)
    x = torch.randn(3, 2, 256); y = model(x); changed = x.clone(); changed[1] *= 11
    torch.testing.assert_close(model(changed)[0], y[0], atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(model(x[[2, 0, 1]]), y[[2, 0, 1]], atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(model(x[:1])[0], y[0], atol=2e-7, rtol=2e-6)


def test_complete_forward_MAC_and_spectral_transform_counts_are_explicit():
    torch.set_num_threads(2)
    model = scratch_model(76); macs = []; spectral = []
    def count(module, args, output):
        if isinstance(module, torch.nn.Conv1d): macs.append(output.numel()*module.weight[0].numel())
        else: macs.append(output.numel()*module.in_features)
    handles = [m.register_forward_hook(count) for m in model.modules() if isinstance(m, (torch.nn.Conv1d, torch.nn.Linear))]
    handles += [b.spectral.register_forward_hook(lambda m, a, o: spectral.append(o.shape)) for b in model.blocks]
    try:
        model(torch.randn(1, 2, 256))
    finally:
        for handle in handles: handle.remove()
    assert sum(macs) == CONV_LINEAR_MAC and spectral == [(1, 64, 256)]*6
    assert architecture()['rFFT_calls'] == architecture()['irFFT_calls'] == 6
    assert architecture()['transformed_real_channels_per_packet'] == 768
    assert not architecture()['performance_verified']


@pytest.mark.parametrize('shape,dtype', [((0,2,256), torch.float32), ((2,3,256), torch.float32),
                                       ((2,2,255), torch.float32), ((2,2,256), torch.float64)])
def test_received_IQ_boundary_rejects_wrong_shape_or_precision(shape, dtype):
    with pytest.raises(ValueError, match='FP32'):
        scratch_model(77)(torch.zeros(shape, dtype=dtype))
