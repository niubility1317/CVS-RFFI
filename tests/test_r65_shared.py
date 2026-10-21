import copy

import pytest
import torch
from torch import nn

from experiments.pamf_r46_delayed_prune.models import CompactSuccessor
from experiments.pamf_r65_four_route.shared import SharedContextRestorer


@pytest.fixture
def parent():
    torch.set_num_threads(2)
    torch.manual_seed(65)
    model = CompactSuccessor('prune_after10').eval().requires_grad_(False)
    # Exercise all trained paths: freshly constructed Core output heads are
    # zero, which would otherwise hide encoder gradients in this synthetic test.
    with torch.no_grad():
        for module in (model.frozen.coarse.spectral, model.frozen.coarse.temporal,
                       model.frozen.estimator.core, *model.prior):
            module.out.weight.normal_(0, .005)
        model.frozen.estimator.global_head.weight.normal_(0, .005)
        model.frozen.estimator.global_head.bias.zero_()
    return model


def test_shared_shapes_finite_and_one_context_execution(parent):
    model = SharedContextRestorer(parent)
    calls = []
    handles = [model.shared.encoder.register_forward_hook(lambda *args: calls.append('encoder'))]
    handles += [block.register_forward_hook(lambda *args: calls.append('decode'))
                for block in model.shared.decode]
    try:
        outputs, variance, action, base = model.stage_outputs(torch.randn(2, 2, 256)*.1)
    finally:
        for handle in handles:
            handle.remove()
    assert len(outputs) == 3 and all(v.shape == (2, 2, 256) for v in outputs)
    assert base.shape == (2, 2, 256) and variance.shape == (2, 256)
    assert calls == ['encoder', 'decode', 'decode', 'decode']
    assert action.kernel.shape == (2, 64)
    assert action.main.shape == action.image.shape == (2, 256)
    assert all(torch.isfinite(v).all() for v in (*outputs, variance, base,
                                                action.kernel, action.main, action.image))
    torch.testing.assert_close(action.kernel.abs().square().sum(-1), torch.ones(2))
    assert ((variance >= 1e-8) & (variance <= 1.)).all()


def test_all_retained_paths_train_and_parent_is_unchanged(parent):
    before = {key: value.clone() for key, value in parent.state_dict().items()}
    model = SharedContextRestorer(parent).train()
    assert all(parameter.requires_grad for parameter in model.parameters())
    received = torch.randn(2, 2, 256)*.1
    outputs, variance, _, _ = model.stage_outputs(received)
    loss = sum((value-received*.8).square().mean() for value in outputs)+variance.mean()
    loss.backward()
    paths = (model.spectral.encoder, model.shared.encoder, model.shared.decode,
             model.coarse_head, model.action_head, model.global_head, *model.prior)
    for module in paths:
        gradients = [parameter.grad for parameter in module.parameters()]
        assert gradients and all(grad is not None and torch.isfinite(grad).all() for grad in gradients)
        assert any(torch.count_nonzero(grad) for grad in gradients)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(parameter.grad, alpha=-1e-3)
    assert not parent.training and all(not parameter.requires_grad for parameter in parent.parameters())
    assert all(torch.equal(value, before[key]) for key, value in parent.state_dict().items())
    assert not ({parameter.data_ptr() for parameter in model.parameters()}
                & {parameter.data_ptr() for parameter in parent.parameters()})


def test_weight_origins_aliases_and_removed_context_graph(parent):
    model = SharedContextRestorer(parent)
    for name in ('encoder', 'decode'):
        source = getattr(parent.frozen.coarse.temporal, name).state_dict()
        assert all(torch.equal(value, source[key])
                   for key, value in getattr(model.shared, name).state_dict().items())
    for child, source in ((model.spectral, parent.frozen.coarse.spectral),
                          (model.coarse_head, parent.frozen.coarse.temporal.out),
                          (model.action_head, parent.frozen.estimator.core.out),
                          (model.global_head, parent.frozen.estimator.global_head)):
        assert all(torch.equal(value, source.state_dict()[key]) for key, value in child.state_dict().items())
    modules = [dict(prior.named_modules()) for prior in model.prior]
    for name, module in modules[0].items():
        if isinstance(module, nn.Conv1d) and module.kernel_size[0] > 1:
            assert module.weight is modules[1][name].weight is modules[2][name].weight
    assert model.prior[0].out.weight is not model.prior[1].out.weight
    assert not hasattr(model, 'frozen') and not hasattr(model.shared, 'out')
    assert not any(key.startswith(('frozen.', 'temporal.', 'estimator.')) for key in model.state_dict())
    report = model.initialization_report
    assert report['new_parameters'] == 0 and not report['exact_initial_function_equivalence']
    assert report['parameters'] == 199629 and report['discarded_parameters'] == 74136
    assert report['copied_parameters'] == sum(parameter.numel() for parameter in model.parameters())
    assert all(report[key] == 0 for key in ('IQ_examples', 'TX_labels_read', 'F_reads', 'V_reads',
                                           'target_reads', 'optimizer_updates', 'new_calibration_IQ'))


def test_original_coarse_spectral_and_native_variance_formula_are_retained(parent):
    model = SharedContextRestorer(parent).eval()
    received = torch.randn(2, 2, 256)*.1
    with torch.no_grad():
        base, action, native = model.estimate_context(received)
        torch.testing.assert_close(model.spectral_observation(received),
                                   parent.frozen.coarse.spectral_observation(received), rtol=0, atol=0)
        torch.testing.assert_close(base, parent.frozen.coarse(received), rtol=0, atol=0)
        _, deepest = model.shared(torch.cat((received, model.spectral_observation(received)), 1))
        global_value = model.global_head(deepest.mean(-1))
        expected = (torch.log(torch.tensor(1e-8))
                    +(torch.log(torch.tensor(1.))-torch.log(torch.tensor(1e-8)))
                    *torch.sigmoid(global_value[:, -1]+.351)).exp()
        torch.testing.assert_close(native, expected)
    assert torch.count_nonzero(model.initial_kernel) == 1 and model.initial_kernel[24] == 1


def test_independent_batches_and_strict_standalone_state_roundtrip(parent):
    model = SharedContextRestorer(parent).eval()
    restored = SharedContextRestorer().eval()
    assert restored.initialization_report['copied_parameters'] == 0
    assert restored.initialization_report['new_parameters'] == 199629
    restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    received = torch.randn(2, 2, 256)*.1
    with torch.no_grad():
        expected = model(received)
        torch.testing.assert_close(restored(received), expected, rtol=0, atol=0)
        torch.testing.assert_close(model(received[:1]), expected[:1], rtol=2e-4, atol=3e-6)
        changed = torch.cat((received[:1], torch.randn_like(received[1:])*3), 0)
        torch.testing.assert_close(model(changed)[:1], expected[:1], rtol=2e-4, atol=3e-6)
        final, variance, action, base = model.training_outputs(received)
        assert final.shape == base.shape == expected.shape and variance.shape == (2, 256)
        torch.testing.assert_close(final, expected, rtol=0, atol=0)
    aliases = [dict(prior.named_modules()) for prior in restored.prior]
    for name, module in aliases[0].items():
        if isinstance(module, nn.Conv1d) and module.kernel_size[0] > 1:
            assert module.weight is aliases[1][name].weight is aliases[2][name].weight


def test_only_fixed_parent_and_current_iq_shape(parent):
    with pytest.raises(ValueError, match='designated compact R65'):
        SharedContextRestorer(CompactSuccessor('prune_after5'))
    with pytest.raises(ValueError, match='IQ2x256'):
        SharedContextRestorer(parent)(torch.zeros(1, 2, 255))
