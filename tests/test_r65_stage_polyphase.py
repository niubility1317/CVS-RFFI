import io

import pytest
import torch
from torch import nn

from experiments.pamf_r17_compact.models import stages as original_stages
from experiments.pamf_r46_delayed_prune.models import CompactSuccessor
from experiments.pamf_r65_four_route import stage_polyphase as models


@pytest.fixture
def parent():
    torch.set_num_threads(2)
    torch.manual_seed(6501)
    value = CompactSuccessor('prune_after10').eval().requires_grad_(False)
    # Mimic nonzero learned source weights without loading any IQ or checkpoint.
    with torch.no_grad():
        for parameter in value.parameters():
            parameter.add_(.003*torch.randn_like(parameter))
    value.initialization_report = {'mode': 'synthetic_parent', 'IQ_examples': 0}
    return value


def _assert_aliases(priors):
    maps = [dict(prior.named_modules()) for prior in priors]
    for name, module in maps[0].items():
        if isinstance(module, nn.Conv1d) and module.kernel_size[0] > 1:
            assert all(module.weight is other[name].weight for other in maps[1:])


def test_time_shuffle_exact_order_roundtrip_and_sample_isolation():
    value = torch.arange(3*9*256).reshape(3, 9, 256)
    shuffled = models.time_unshuffle(value)
    assert shuffled.shape == (3, 18, 128)
    assert torch.equal(shuffled[:, 0::2], value[..., ::2])
    assert torch.equal(shuffled[:, 1::2], value[..., 1::2])
    assert torch.equal(models.time_shuffle(shuffled), value)
    assert torch.equal(shuffled[1:2], models.time_unshuffle(value[1:2]))
    assert torch.equal(models.time_shuffle(shuffled[2:3]), value[2:3])


@pytest.mark.parametrize('function,value', [
    (models.time_unshuffle, torch.empty(1, 9, 255)),
    (models.time_unshuffle, torch.empty(9, 256)),
    (models.time_shuffle, torch.empty(1, 3, 128)),
    (models.time_shuffle, torch.empty(4, 128)),
])
def test_shuffle_rejects_invalid_shape(function, value):
    with pytest.raises(ValueError):
        function(value)


def test_stage2_retains_last_two_priors_and_parent_weights(parent):
    before = {name: value.clone() for name, value in parent.state_dict().items()}
    model = models.StageDistilled(parent)
    assert model.arm == 'stage2' and model.STAGE_TARGETS == models.STAGE_TARGETS == (1, 2)
    assert len(model.prior) == 2
    _assert_aliases(model.prior)
    for index, prior in enumerate(model.prior):
        assert all(torch.equal(value, parent.prior[index+1].state_dict()[name])
                   for name, value in prior.state_dict().items())
    assert all(torch.equal(value, before[name]) for name, value in parent.state_dict().items())
    assert not any(parameter.requires_grad for parameter in parent.parameters())
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert model.initialization_report['copied_parameters'] == sum(p.numel() for p in model.parameters())
    assert model.initialization_report['transformed_parameters'] == 0
    assert model.initialization_report['CG_steps'] == 4


def test_stage2_exact_original_algorithm_with_two_iterations(parent):
    model = models.StageDistilled(parent).eval()
    value = torch.randn(1, 2, 256)
    with torch.no_grad():
        context = model.estimate_context(value)
        actual = model.stage_outputs(value, context=context)[0]
        # The legacy routine can be used as a reference with a harmless third
        # prior: only its first two outputs are the new two-stage computation.
        expected = original_stages([model.prior[0], model.prior[1], model.prior[1]], value, context)[0][:2]
    assert len(actual) == 2
    assert all(torch.equal(left, right) for left, right in zip(actual, expected))


def test_polyphase_weight_copy_transform_and_parent_unchanged(parent):
    before = {name: value.clone() for name, value in parent.state_dict().items()}
    model = models.PolyphaseRestorer(parent)
    assert model.arm == 'polyphase' and model.STAGE_TARGETS == (0, 1, 2)
    assert len(model.prior) == 3
    _assert_aliases(model.prior)
    transformed = {'encoder.blocks.0.0.net.0.weight', 'encoder.blocks.0.0.skip.weight', 'out.weight', 'out.bias'}
    for source, target in zip(parent.prior, model.prior):
        old = source.state_dict()
        for name, value in target.core.state_dict().items():
            if name in {'encoder.blocks.0.0.net.0.weight', 'encoder.blocks.0.0.skip.weight'}:
                assert torch.equal(value, old[name].repeat_interleave(2, dim=1)/2)
            elif name in {'out.weight', 'out.bias'}:
                assert torch.equal(value, old[name].repeat_interleave(2, dim=0))
            else:
                assert name not in transformed and torch.equal(value, old[name])
    assert all(torch.equal(value, before[name]) for name, value in parent.state_dict().items())
    report = model.initialization_report
    assert report['copied_parameters'] + report['transformed_parameters'] == report['parameters']
    assert report['new_parameters'] == report['parameters'] - models.PARENT_PARAMETERS == 942
    assert report['transformed_parameters'] == 1884
    assert report['random_parameters'] == 0 and not report['exact_function_equivalence']
    assert report['CG_steps'] == 6 and report['whole_IQ_window']
    assert not any(parameter.requires_grad for parameter in parent.parameters())


@pytest.mark.parametrize('factory', [models.StageDistilled, models.PolyphaseRestorer])
@pytest.mark.parametrize('batch', [1, 3])
def test_finite_full_window_outputs_and_component_gradients(parent, factory, batch):
    model = factory(parent).train()
    value = torch.randn(batch, 2, 256)
    stages, variance, action, base = model.stage_outputs(value)
    assert len(stages) == len(model.STAGE_TARGETS)
    assert all(stage.shape == value.shape and torch.isfinite(stage).all() for stage in stages)
    assert base.shape == value.shape and variance.shape == (batch, 256)
    assert torch.isfinite(variance).all() and (variance > 0).all()
    assert torch.isfinite(action(torch.complex(value[:, 0], value[:, 1]))).all()
    loss = sum(stage.square().mean() for stage in stages) + .0001*variance.log().square().mean()
    loss.backward()
    assert all(parameter.requires_grad and parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    components = [model.frozen.coarse.spectral, model.frozen.coarse.temporal, model.frozen.estimator, *model.prior]
    assert all(any(parameter.grad.abs().sum() > 0 for parameter in component.parameters()) for component in components)
    assert all(parameter.grad is None for parameter in parent.parameters())


@pytest.mark.parametrize('factory', [models.StageDistilled, models.PolyphaseRestorer])
def test_strict_checkpoint_roundtrip_preserves_aliases_and_public_interfaces(parent, factory):
    model = factory(parent).eval()
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = factory().eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    _assert_aliases(restored.prior)
    assert restored.initialization_report['construction'] == 'checkpoint_loading_skeleton'
    assert all(parameter.requires_grad for parameter in restored.parameters())
    value = torch.randn(3, 2, 256)
    with torch.no_grad():
        expected = model(value)
        actual = restored(value)
        assert torch.equal(expected, actual)
        assert torch.equal(restored.training_outputs(value)[0], actual)
        # Both a singleton last batch and an odd-sized batch retain samples.
        assert torch.allclose(actual, torch.cat([restored(item[None]) for item in value]), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize('factory', [models.StageDistilled, models.PolyphaseRestorer])
def test_rejects_other_parent_before_initialization(factory):
    with pytest.raises(ValueError):
        factory(CompactSuccessor('prune_after5'))


@pytest.mark.parametrize('factory', [models.StageDistilled, models.PolyphaseRestorer])
def test_initialization_has_zero_new_data_or_optimizer_access(parent, factory):
    report = factory(parent).initialization_report
    assert report['inherited_initialization'] == parent.initialization_report
    assert all(report[key] == 0 for key in (
        'IQ_examples', 'TX_labels_read', 'F_reads', 'V_reads', 'target_reads', 'optimizer_updates'))
