import unittest

import torch
from torch import nn

from experiments.cric_dura_comparison.f_optimizer import build, clip_physical
from experiments.pamf_repl_weight.source_adamw_diagnostic import capture_guard_derivatives, difference
from experiments.pamf_ssl_research.source_optimizer_shadow import shadow_step


class CoordinateSinc(nn.Module):
    def __init__(self):
        super().__init__()
        self.sample_rate, self.scale_hz, self.mode = 25000000., 25000., 'CENTER_25K'
        self.low_coordinate = nn.Parameter(torch.zeros(24, 1))
        self.band_coordinate = nn.Parameter(torch.zeros(24, 1))
        self.register_buffer('low_origin_hz', torch.full((24, 1), 50000.))
        self.register_buffer('band_origin_hz', torch.full((24, 1), 75000.))


def model():
    value, core = nn.Module(), nn.Module()
    core.sinc = CoordinateSinc()
    core.cls_head = nn.Linear(160, 6, bias=False)
    core.body = nn.Parameter(torch.tensor([.25, -.5]))
    value.id_backbone = nn.Module()
    value.id_backbone.encoder = nn.Module()
    value.id_backbone.encoder.core = nn.Module()
    value.id_backbone.encoder.core.id_backbone = core
    return value


class SourceAdamWChecks(unittest.TestCase):
    def test_capture_and_restore_on_success_and_failure(self):
        original = torch.autograd.grad
        x = torch.tensor([2.], requires_grad=True)
        with capture_guard_derivatives() as captured:
            torch.autograd.grad(x.square().sum(), [x], retain_graph=True)
            torch.autograd.grad((3*x).sum(), [x])
        self.assertTrue(torch.equal(captured[0][0], torch.tensor([4.])))
        self.assertTrue(torch.equal(captured[1][0], torch.tensor([3.])))
        self.assertIs(torch.autograd.grad, original)
        with self.assertRaises(RuntimeError):
            with capture_guard_derivatives():
                raise RuntimeError('expected')
        self.assertIs(torch.autograd.grad, original)

    def test_incremental_displacement_handles_disconnected_parameters(self):
        p = nn.Parameter(torch.tensor([0., 0.]))
        named = [('a', p), ('b', p), ('c', p)]
        observed = difference((torch.tensor([2., 3.]), None, None),
                              (torch.tensor([1., 1.]), torch.tensor([4., 5.]), None), named)
        self.assertTrue(torch.equal(observed[0], torch.tensor([1., 2.])))
        self.assertTrue(torch.equal(observed[1], torch.tensor([-4., -5.])))
        self.assertIsNone(observed[2])

    def test_shadow_matches_direct_native_step_without_mutating_source(self):
        value = model()
        named = list(value.named_parameters())
        before = {name:p.detach().clone() for name,p in named}
        gradients = tuple(torch.full_like(p, 8. if 'coordinate' in n else 2.) for n,p in named)
        observed, meta = shadow_step(value, named, gradients, base_lr=2e-4, weight_decay=1e-4)
        self.assertTrue(all(p.grad is None and torch.equal(before[n],p) for n,p in named))
        optimizer = build(value,[p for _,p in named],'CENTER_UNITS',lr=2e-4,weight_decay=1e-4)
        for (_,p),gradient in zip(named,gradients):
            p.grad = gradient.clone()
        clip_physical(optimizer,[p for _,p in named],5.)
        optimizer.step()
        for (name,p),descent in zip(named,observed):
            torch.testing.assert_close(descent,before[name]-p.detach(),rtol=0,atol=0)
        self.assertEqual(meta['optimizer_steps'],1)
        self.assertTrue(meta['fresh_moments'])

    def test_rejects_wrong_shadow_scope(self):
        value = model()
        named = list(value.named_parameters())
        gradients = tuple(torch.zeros_like(p) for _,p in named)
        for options in (dict(base_lr=0.,weight_decay=0.),dict(base_lr=1e-4,weight_decay=-1.)):
            with self.assertRaises(ValueError):
                shadow_step(value,named,gradients,**options)


if __name__ == '__main__':
    unittest.main()
