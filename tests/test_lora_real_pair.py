from dataclasses import replace

import torch

from experiments.pamf_lora_main import real_pair
from experiments.pamf_lora_main.dura_repair import PacketRestorer, real_iq
from experiments.pamf_lora_main.real_pair_source import RECIPES, PAIR_WEIGHT


def test_pair_reuses_exact_same_additive_draw_and_echo(monkeypatch):
    gen = torch.Generator().manual_seed(314)
    clean = torch.randn(5, 6, 256, dtype=torch.complex64, generator=gen)
    taps = tuple(torch.tensor(x, dtype=torch.complex64).expand(3, 1, 1)
                 for x in (.1 + .25j, -.05j))
    monkeypatch.setattr(real_pair, 'echo_taps', lambda _, _rng: taps)
    out = real_pair.paired_view(clean, gen)
    assert out.shape == clean.shape
    expected = real_pair.apply_echo(real_pair.signed_difference(clean[:4]), tuple(t[:2] for t in taps))
    torch.testing.assert_close(real_pair.signed_difference(out[:4]), expected)


def test_pair_geometry_is_signed_and_penalizes_wrong_phase():
    gen = torch.Generator().manual_seed(97)
    delta = torch.randn(2, 1, 6, 256, dtype=torch.complex64, generator=gen)
    severity = torch.ones(2)
    perfect = real_pair.band_risks(delta.expand(-1, 2, -1, -1), delta, severity)
    different = real_pair.band_risks(1j * delta.expand(-1, 2, -1, -1), delta, severity)
    torch.testing.assert_close(perfect, torch.zeros(4))
    assert (different > 0).all()


def test_matched_views_and_odd_holdout_batches_are_finite():
    tracker = PacketRestorer('consensus').eval()
    recipe = replace(RECIPES['point'], restore=lambda w, y, _q, _rot: (w * y, w * y))
    data_rng = torch.Generator().manual_seed(222)
    clean = real_iq(torch.randn(3, 6, 256, dtype=torch.complex64, generator=data_rng))
    weight = torch.nn.Parameter(torch.tensor(.8))
    options = dict(with_identity=False, recipe=recipe, step=1, tail_weight=.45)
    control, values = real_pair.objective(weight, tracker, clean, torch.Generator().manual_seed(123),
                                          pair_weight=0., **options)
    candidate, changed = real_pair.objective(weight, tracker, clean, torch.Generator().manual_seed(123),
                                             pair_weight=PAIR_WEIGHT, **options)
    torch.testing.assert_close(values, changed, atol=0, rtol=0)
    assert torch.isfinite(candidate) and candidate > control
    candidate.backward()
    assert weight.grad is not None and weight.grad.isfinite()
