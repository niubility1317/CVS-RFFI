import torch
from torch import nn
import pytest

from experiments.cric_dura_comparison.f_optimizer import build, clip_physical
from experiments.pamf_ssl_research.source_optimizer_shadow import shadow_step


class CoordinateSinc(nn.Module):
    def __init__(self):
        super().__init__()
        self.sample_rate = 25000000.0
        self.scale_hz = 25000.0
        self.mode = 'CENTER_25K'
        self.low_coordinate = nn.Parameter(torch.zeros(24, 1))
        self.band_coordinate = nn.Parameter(torch.zeros(24, 1))
        self.register_buffer('low_origin_hz', torch.full((24, 1), 50000.0))
        self.register_buffer('band_origin_hz', torch.full((24, 1), 75000.0))


def model():
    value = nn.Module()
    core = nn.Module()
    core.sinc = CoordinateSinc()
    core.cls_head = nn.Linear(160, 6, bias=False)
    core.body = nn.Parameter(torch.tensor([0.25, -0.5]))
    value.id_backbone = nn.Module()
    value.id_backbone.encoder = nn.Module()
    value.id_backbone.encoder.core = nn.Module()
    value.id_backbone.encoder.core.id_backbone = core
    return value


def vectors(value, body):
    return tuple(torch.tensor(body) if name.endswith('.body') else torch.zeros_like(p)
                 for name, p in value.named_parameters())


def test_shadow_matches_direct_native_physical_step_and_preserves_source():
    torch.manual_seed(11)
    value = model()
    named = list(value.named_parameters())
    original = {name: p.detach().clone() for name, p in named}
    gradients = tuple(torch.full_like(p, 8.0 if 'coordinate' in name else 2.0) for name, p in named)
    observed, meta = shadow_step(value, named, gradients, base_lr=2e-4, weight_decay=1e-4)
    optimizer = build(value, [p for _, p in named], 'CENTER_UNITS', lr=2e-4, weight_decay=1e-4)
    assert not optimizer.state
    for (_, p), gradient in zip(named, gradients):
        assert p.grad is None
        p.grad = gradient.clone()
    clip_physical(optimizer, [p for _, p in named], 5.0)
    optimizer.step()
    for (name, p), descent in zip(named, observed):
        torch.testing.assert_close(descent, original[name] - p.detach(), rtol=0, atol=0)
    assert meta['optimizer_steps'] == 1
    assert meta['physical_clip_calls'] == 1
    assert meta['fresh_moments'] is True
    assert meta['parameter_groups'][1]['lr'] == pytest.approx(0.2)
    assert meta['parameter_groups'][1]['eps'] == pytest.approx(1e-11)


def test_source_parameters_gradients_and_buffers_are_untouched():
    value = model()
    before = {name: tensor.detach().clone() for name, tensor in value.state_dict().items()}
    named = list(value.named_parameters())
    shadow_step(value, named, vectors(value, [1., 100.]), base_lr=2e-4, weight_decay=1e-4)
    assert all(p.grad is None for _, p in named)
    assert all(torch.equal(tensor, before[name]) for name, tensor in value.state_dict().items())


def test_positive_gradient_dot_can_have_negative_extra_adam_descent_dot():
    value = model()
    named = list(value.named_parameters())
    labeled = vectors(value, [1., 100.])
    unlabeled = vectors(value, [-2., 1.])
    assert sum(float((a.double() * b.double()).sum()) for a, b in zip(labeled, unlabeled)) > 0
    combined = tuple(a + b for a, b in zip(labeled, unlabeled))
    baseline, _ = shadow_step(value, named, labeled, base_lr=2e-4, weight_decay=0.)
    alternative, _ = shadow_step(value, named, combined, base_lr=2e-4, weight_decay=0.)
    extra = tuple(a - b for a, b in zip(alternative, baseline))
    assert sum(float((g.double() * d.double()).sum()) for g, d in zip(labeled, extra)) < -1e-5


@pytest.mark.parametrize('change', ['lr', 'decay', 'dtype', 'order'])
def test_rejects_wrong_shadow_scope(change):
    value = model()
    named = list(value.named_parameters())
    gradients = vectors(value, [1., 2.])
    options = dict(base_lr=2e-4, weight_decay=1e-4)
    if change == 'lr':
        options['base_lr'] = 0.
    elif change == 'decay':
        options['weight_decay'] = -1.
    elif change == 'dtype':
        gradients = tuple(g.double() for g in gradients)
    elif change == 'order':
        named = list(reversed(named))
    with pytest.raises(ValueError):
        shadow_step(value, named, gradients, **options)
