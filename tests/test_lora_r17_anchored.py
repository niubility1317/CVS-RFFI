import torch

from experiments.pamf_lora_main.r17_anchored import AnchoredDirection, direction_loss
from experiments.pamf_lora_main.residual_dura_bandfit import band_masks, power


def test_anchored_direction_retains_each_band_dose_and_all_out_of_band_bins():
    torch.set_num_threads(2)
    rng = torch.Generator().manual_seed(51)
    values = torch.complex(*torch.randn((2, 5, 4, 256), generator=rng)) * .1
    model = AnchoredDirection()
    base = values[:, -1]
    torch.testing.assert_close(model(values), base)
    with torch.no_grad():
        model.head.out.weight.normal_(generator=rng)
    out = model(values)
    bands = band_masks('cpu')
    for band in bands:
        torch.testing.assert_close(power(out[:, band]), power(base[:, band]))
    assert torch.equal(out[:, ~(bands[0] | bands[1])], base[:, ~(bands[0] | bands[1])])
    assert not torch.allclose(out, base)
    assert (out.conj() * base).real.sum(-1).gt(0).all()


def test_clean_like_rows_with_no_correction_remain_exact_and_have_finite_gradients():
    torch.manual_seed(41)
    model = AnchoredDirection()
    values = torch.complex(*torch.randn(2, 4, 4, 256))
    values[:, -1].zero_()
    base = values[:, -1]
    out = model(values)
    assert torch.equal(out, base)
    loss = direction_loss(out, base, values[:, 0])
    loss.mean().backward()
    assert torch.isfinite(loss).all() and all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_angle_training_reaches_the_zero_initialized_head():
    torch.manual_seed(48)
    model = AnchoredDirection()
    values = torch.complex(*torch.randn(2, 4, 4, 256)) * .2
    base, target = values[:, -1], values[:, 0]
    value = direction_loss(model(values), base, target).mean()
    value.backward()
    assert value > 0 and model.head.out.weight.grad.norm() > 0
