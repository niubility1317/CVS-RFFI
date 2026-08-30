from __future__ import annotations

import pytest
import torch

from cvsrffi.phase1_bicad_xr.losses import (
    DetachedEMA,
    apply_margin_tail,
    group_margin_cvar,
    three_layer_group_margin_cvar,
)


def test_group_margin_cvar_keeps_only_the_worst_sample_in_each_half_group() -> None:
    margins = torch.tensor([2.0, -1.0, 0.0, -3.0], requires_grad=True)
    groups = torch.tensor([0, 0, 1, 1])

    risk = group_margin_cvar(margins, groups, tail_fraction=0.5)

    expected = torch.stack(
        [torch.nn.functional.softplus(torch.tensor(1.0)), torch.nn.functional.softplus(torch.tensor(3.0))]
    ).mean()
    assert risk.item() == pytest.approx(expected.item())
    risk.backward()
    assert margins.grad is not None
    assert torch.isfinite(margins.grad).all()


def test_three_layer_group_margin_cvar_combines_tx_xdc_and_tangent_risks() -> None:
    tx = torch.tensor([-2.0, 2.0], requires_grad=True)
    xdc = torch.tensor([-1.0, 1.0], requires_grad=True)
    tangent = torch.tensor([0.0, 0.0], requires_grad=True)
    groups = torch.tensor([0, 0])

    risk = three_layer_group_margin_cvar(
        tx,
        xdc,
        tangent,
        tx_groups=groups,
        xdc_groups=groups,
        tangent_groups=groups,
        tail_fraction=1.0,
        weights=(0.6, 0.3, 0.1),
    )

    expected = (
        0.6
        * torch.stack(
            [
                torch.nn.functional.softplus(torch.tensor(2.0)),
                torch.nn.functional.softplus(torch.tensor(-2.0)),
            ]
        ).mean()
        + 0.3
        * torch.stack(
            [
                torch.nn.functional.softplus(torch.tensor(1.0)),
                torch.nn.functional.softplus(torch.tensor(-1.0)),
            ]
        ).mean()
        + 0.1 * torch.nn.functional.softplus(torch.tensor(0.0))
    )
    assert risk.item() == pytest.approx(expected.item())
    risk.backward()
    for margin in (tx, xdc, tangent):
        assert margin.grad is not None
        assert torch.isfinite(margin.grad).all()


def test_detached_ema_updates_without_attaching_the_running_state_to_margin_graph() -> None:
    ema = DetachedEMA(decay=0.5)
    first = torch.tensor(2.0, requires_grad=True)
    second = torch.tensor(6.0, requires_grad=True)

    first_state = ema.update(first)
    second_state = ema.update(second)

    assert first_state.item() == pytest.approx(2.0)
    assert second_state.item() == pytest.approx(4.0)
    assert not first_state.requires_grad
    assert not second_state.requires_grad
    assert ema.value is not None and not ema.value.requires_grad
    assert first.grad is None
    assert second.grad is None


def test_margin_tail_only_adds_the_three_allowed_risks() -> None:
    base = torch.tensor(10.0, requires_grad=True)
    tx = torch.tensor(1.0, requires_grad=True)
    xdc = torch.tensor(2.0, requires_grad=True)
    tangent = torch.tensor(3.0, requires_grad=True)

    total = apply_margin_tail(
        base,
        tx_risk=tx,
        xdc_risk=xdc,
        tangent_risk=tangent,
        weights=(0.6, 0.3, 0.1),
    )

    assert total.item() == pytest.approx(10.0 + 0.6 + 0.6 + 0.3)
    total.backward()
    assert base.grad is not None and base.grad.item() == pytest.approx(1.0)
    assert tx.grad is not None and tx.grad.item() == pytest.approx(0.6)
    assert xdc.grad is not None and xdc.grad.item() == pytest.approx(0.3)
    assert tangent.grad is not None and tangent.grad.item() == pytest.approx(0.1)
