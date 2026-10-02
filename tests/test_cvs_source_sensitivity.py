import math
import pytest
import torch
from torch import nn
from experiments.cvs_source_sensitivity.audit import transform, Groups, capture, STAGES


def test_registered_phase_and_signed_cfo_match_analytic_received_rotation():
    torch.manual_seed(3)
    x = torch.randn(4, 2, 256, dtype=torch.float64)
    x /= x.square().sum(1).mean(-1).sqrt()[:, None, None]
    original = x.clone()
    z = torch.complex(x[:, 0], x[:, 1])
    for name, angle in [('constant_phase', .37), ('cfo_plus', 2*math.pi*80000/25000000*torch.arange(256, dtype=x.dtype)),
                         ('cfo_minus', -2*math.pi*80000/25000000*torch.arange(256, dtype=x.dtype))]:
        y = transform(x, name)
        expected = z*torch.exp(1j*torch.as_tensor(angle, dtype=x.dtype))
        torch.testing.assert_close(torch.complex(y[:, 0], y[:, 1]), expected, atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(y.square().sum(1).mean(-1), torch.ones(4, dtype=x.dtype))
    assert torch.equal(x, original)
    with pytest.raises(ValueError): transform(x, 'unregistered')


def test_received_lti_uses_zero_history_and_image_cubic_equations():
    x = torch.zeros(1, 2, 256, dtype=torch.float64); x[0, 0, -1] = 2.
    assert transform(x, 'received_lti')[0, :, 0].abs().sum() == 0
    x = torch.randn(3, 2, 256, dtype=torch.float64)
    z = torch.complex(x[:, 0], x[:, 1])
    for name, expected in [('received_image', z+(.02+.02j)*z.conj()), ('received_cubic', z-.03*z*z.abs().square())]:
        expected /= expected.abs().square().mean(-1).sqrt()[:, None]
        y = transform(x, name)
        torch.testing.assert_close(torch.complex(y[:, 0], y[:, 1]), expected)


def test_source_group_means_use_packet_counts_and_preserve_maximum():
    g = Groups()
    g.add(dict(label=torch.tensor([0, 0, 1]), receiver=torch.tensor([1, 1, 3]), day=torch.tensor([2, 2, 3])),
          dict(correct=torch.tensor([1., 0., 1.]), distance=torch.tensor([.1, .3, .8])))
    g.add(dict(label=torch.tensor([1]), receiver=torch.tensor([3]), day=torch.tensor([3])),
          dict(correct=torch.tensor([1.]), distance=torch.tensor([1.])))
    r = g.result(4)
    assert r['mean']['correct'] == .75
    assert r['mean']['distance'] == pytest.approx(.55)
    assert r['maximum']['distance'] == 1.
    assert [c['count'] for c in r['groups']] == [2, 2]
    with pytest.raises(ValueError): g.result(5)


class ObserverModel(nn.Module):
    def __init__(self):
        super().__init__(); self.id_backbone = nn.Module()
        b = self.id_backbone
        for name in ('t_proj', 'f_proj', 'pa_proj'): setattr(b, name, nn.Linear(2, 2))
        b.cls_head = nn.Module(); b.cls_head.base_norm = nn.LayerNorm(2); b.cls_head.pa_norm = nn.LayerNorm(2)
        b.cls_head.classify = lambda x: x
        self.fail = False
    def features(self, x):
        b = self.id_backbone
        v = x.mean(-1)
        t, f, p = b.t_proj(v), b.f_proj(v), b.pa_proj(v)
        if self.fail: raise RuntimeError('test forward failure')
        return b.cls_head.base_norm(t+f)+b.cls_head.pa_norm(p)


def test_observation_hooks_cannot_change_forward_or_state_and_always_remove():
    model = ObserverModel().eval().requires_grad_(False)
    x = torch.randn(3, 2, 256)
    original = model.features(x)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    observed = capture(model, x)
    assert set(observed) == set(STAGES) | {'features', 'logits'}
    torch.testing.assert_close(observed['features'], original, rtol=0, atol=0)
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    assert not any(m._forward_hooks for m in model.modules())
    model.fail = True
    with pytest.raises(RuntimeError): capture(model, x)
    assert not any(m._forward_hooks for m in model.modules())


@pytest.mark.parametrize('variant', ['residual_fusion', 'equivariant_memory'])
def test_real_source_architectures_execute_each_registered_stage_once(variant):
    from experiments.cvs_residual_identity.model import build as residual
    from experiments.cvs_equivariant_identity.model import build as equivariant
    model = (residual if variant == 'residual_fusion' else equivariant)(variant).eval().requires_grad_(False)
    x = torch.zeros(2, 2, 256)
    result = capture(model, x)
    assert result['features'].shape == (2, 160)
    assert result['logits'].shape == (2, 6)
    torch.testing.assert_close(result['logits'], model(x), rtol=0, atol=0)
