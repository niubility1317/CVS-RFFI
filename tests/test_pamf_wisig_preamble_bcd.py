import pytest
import torch
from torch import nn

from experiments.cvs_reference_identity.model import ExcitationResponse
from experiments.pamf_wisig_preamble_bcd.features import (
    WIDTH, cycle_descriptor, fit_drift, CycleResponse, CycleProjection,
)


def periodic_packets():
    generator = torch.Generator().manual_seed(9)
    period = torch.randn(2, 2, 20, generator=generator)
    return period.repeat(1, 1, 13)[..., :256].contiguous()


def test_descriptor_preserves_period_order_and_removes_shared_rotation():
    x = periodic_packets()
    ph = .7 + torch.arange(256) * .012
    rotated = torch.view_as_real(torch.complex(x[:, 0], x[:, 1]) * torch.exp(1j * ph)).transpose(1, 2)
    torch.testing.assert_close(cycle_descriptor(x), cycle_descriptor(rotated), atol=2e-5, rtol=0)
    changed = x.clone()
    changed[..., 20:40] *= .4
    h = cycle_descriptor(changed) - cycle_descriptor(x)
    assert h[:, :4].abs().sum() > .1
    assert torch.count_nonzero(h[:, 4:]) == 0
    assert cycle_descriptor(torch.zeros_like(x)).isfinite().all()


def test_fit_uses_within_identity_same_day_receiver_differences():
    rows = []
    for tx in range(2):
        for day in range(2):
            for rx in range(3):
                for rep in range(2):
                    h = torch.zeros(WIDTH)
                    h[:2] = rx + rep * .01 + tx * 12 + day * 30
                    rows.append((h, tx, rx, day))
    values = torch.stack([r[0] for r in rows])
    args = [list(x) for x in zip(*(r[1:] for r in rows))]
    fit = fit_drift(values, *args, expected=(2, 3, 2))
    diagonal, low_rank = fit['diagonal'], fit['low_rank']
    assert diagonal.shape == (WIDTH, WIDTH)
    assert low_rank.shape == (2, WIDTH)
    torch.testing.assert_close(diagonal.square().sum(), low_rank.square().sum())
    assert (low_rank.T @ low_rank)[0, 1] > 0
    assert fit['rx_pairs'] == 12
    with pytest.raises(ValueError):
        fit_drift(values[:-2], *(a[:-2] for a in args), expected=(2, 3, 2))


def test_legacy_projection_migrates_only_on_allowed_prefix():
    old = nn.Linear(29, 64)
    detector = ExcitationResponse('reference_response')
    x = periodic_packets()
    response = CycleResponse(detector, torch.eye(WIDTH), seed=120)
    module = CycleProjection(old, allow_prefix=True)
    module.load_state_dict(old.state_dict(), strict=True)
    response.eval()
    torch.testing.assert_close(module(response(x)), old(detector(x)), rtol=0, atol=0)
    torch.testing.assert_close(response(x)[0], response(x[:1])[0])
    response.train()
    # The perturbation leaves the deployed features and global RNG stream intact.
    rng = torch.random.get_rng_state()
    perturbed = response(x)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert not torch.equal(perturbed[:, 29:], cycle_descriptor(x))
    torch.testing.assert_close(perturbed[:, :29], detector(x))
    response.eval()
    torch.testing.assert_close(response(x)[:, 29:], cycle_descriptor(x))
    frozen = CycleProjection(old, allow_prefix=False)
    with pytest.raises(RuntimeError, match='cycle_weight'):
        frozen.load_state_dict(old.state_dict(), strict=True)
    frozen.load_state_dict(module.state_dict(), strict=True)
