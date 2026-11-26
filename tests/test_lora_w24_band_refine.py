from types import SimpleNamespace

import torch

from experiments.pamf_lora_main.w24_band_refine import BandRefiner, band_edit_risk, sidebands
from experiments.pamf_lora_main.w24_formal import FULL, cases
from experiments.pamf_lora_main.w24_third_prior import fit_inputs, trial


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.priors = torch.nn.ModuleList(torch.nn.Linear(1, 1) for _ in range(2))

    def propose(self, y, quality, rms, edges):
        samples = torch.arange(y.shape[-1], dtype=y.real.dtype, device=y.device)
        phase = torch.polar(torch.ones_like(samples), 2 * torch.pi * samples / 24)
        return (y, y, y + .2 * phase), None, None, None, None


def test_refiner_preserves_other_bins_and_the_physical_band_energy():
    training = SimpleNamespace(DuraW24=Backbone)
    assert sum(p.numel() for p in BandRefiner(training, polar=False).parameters()) == sum(
        p.numel() for p in BandRefiner(training, polar=True).parameters())
    y = torch.randn(2, 8, 384, dtype=torch.complex64)
    rms = torch.rand(2, 8, 1) + .5
    q = torch.rand(2, 8, 1)
    for polar in (False, True):
        model = BandRefiner(training, polar=polar).train()
        assert not model.backbone.training
        assert not any(p.requires_grad for p in model.backbone.parameters())
        (c, first, base), *args = model.backbone.propose(y, q, rms, None)
        initial, *_ = model.propose(y, q, rms, None)
        torch.testing.assert_close(initial[-1], base)
        with torch.no_grad():
            model.head[-1].bias.fill_(.8)
        outputs, *_ = model.propose(y, q, rms, None)
        assert outputs[0] is c and outputs[1] is first
        before = torch.fft.fft((base-y)*rms, norm='ortho')
        after = torch.fft.fft((outputs[-1]-y)*rms, norm='ortho')
        torch.testing.assert_close(after[..., 33:-32], before[..., 33:-32], rtol=0, atol=2e-6)
        torch.testing.assert_close(sidebands(after).abs().square().sum((1, 2, 3)),
                                   sidebands(before).abs().square().sum((1, 2, 3)))
        band_edit_risk((outputs[-1]-y)*rms, y*rms).backward()
        assert model.head[-1].bias.grad.norm() > 0
        assert all(p.grad is None for p in model.backbone.parameters())


def test_band_loss_is_zero_without_a_usable_source_pair():
    wave = torch.randn(2, 8, 384, dtype=torch.complex64).requires_grad_()
    loss = band_edit_risk(wave, torch.zeros_like(wave))
    assert loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(wave.grad) == 0


def test_band_loss_keeps_the_two_perturbed_views_per_packet():
    edits = torch.randn(14, 2, 8, 384, dtype=torch.complex64).requires_grad_()
    targets = torch.randn_like(edits)
    targets[0, 1] = 0
    loss = band_edit_risk(edits, targets)
    torch.testing.assert_close(loss, band_edit_risk(edits.flatten(0, 1), targets.flatten(0, 1)))
    loss.backward()
    assert torch.isfinite(edits.grad).all()
    assert torch.count_nonzero(edits.grad[0, 1]) == 0


def test_refiner_only_receives_source_paths_and_uses_full_phase1():
    spec = trial('band-refine')
    assert spec.steps == 1200 and spec.source_modes[-1] == 'outer_parent_half'
    assert set(cases('band-refine')) == {f'{FULL}-{i}' for i in range(3)}
    runtime = dict(source_root='/U', source_parent='/G', run_root='/run', classifier_root='/F', target_root='/T')
    for arm in spec.arms:
        assert fit_inputs(runtime, arm) == dict(source_root='/U', source_parent='/G', run_root=f'/run/{arm}')
