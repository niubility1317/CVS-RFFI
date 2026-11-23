import torch

from experiments.pamf_lora_main.direct_detail import CoefficientHead, band_loss, physical_prediction
from experiments.pamf_lora_main.repeatable_detail import basis_spec


def test_direct_head_learns_inside_frozen_basis_and_keeps_doses():
    torch.manual_seed(12)
    torch.set_num_threads(2)
    masks, mask = basis_spec('cpu')
    n = int(mask.sum()) * 12
    basis = torch.linalg.qr(torch.randn(n, 6)).Q
    head = CoefficientHead(n * 3, 6)
    features = torch.randn(4, 3 * n)
    init = torch.randn(4, 6, 256, dtype=torch.complex64) * .03
    desired = torch.randn_like(init) * .03
    output = physical_prediction(head(features), init, basis)
    loss = band_loss(output, init, desired).mean()
    loss.backward()
    assert torch.isfinite(loss) and head.out.weight.grad.norm() > 0
    assert torch.equal(output[..., ~mask], init[..., ~mask])
    for band in masks:
        torch.testing.assert_close(output[..., band].abs().square().sum(-1),
                                   init[..., band].abs().square().sum(-1))


def test_silence_keeps_identity_and_finite_gradients():
    _, mask = basis_spec('cpu')
    raw = torch.randn(2, int(mask.sum()) * 12, requires_grad=True)
    zeros = torch.zeros(2, 6, 256, dtype=torch.complex64)
    value = physical_prediction(raw, zeros, None)
    loss = band_loss(value, zeros, zeros).mean()
    loss.backward()
    assert torch.equal(value, zeros) and loss == 0 and torch.isfinite(raw.grad).all()
