from __future__ import annotations

import pytest
import torch

from cvsrffi.phase1_bicad_xr.heads import (
    DomainFactors,
    FactorizedAdversarialHeads,
    FactorizedDomainProjector,
    conditional_outer,
)
from cvsrffi.phase1_bicad_xr.losses import conditional_cross_covariance


def test_conditional_outer_uses_true_one_hot() -> None:
    z = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = torch.tensor([0, 1])

    out = conditional_outer(z, y, num_classes=2)

    assert out.tolist() == [[1.0, 0.0, 2.0, 0.0], [0.0, 3.0, 0.0, 4.0]]
    out.sum().backward()
    assert torch.equal(z.grad, torch.ones_like(z))


@pytest.mark.parametrize(
    ("z", "tx", "num_classes", "message"),
    [
        (torch.randn(4, 160), None, 6, "TX labels"),
        (torch.randn(4, 160), torch.tensor([0, 1]), 6, "batch"),
        (torch.randn(2, 160), torch.tensor([0.0, 1.0]), 6, "integer"),
        (torch.randn(2, 160), torch.tensor([-1, 1]), 6, "range"),
        (torch.randn(2, 160), torch.tensor([0, 6]), 6, "range"),
    ],
)
def test_conditional_outer_validates_true_tx_labels(
    z: torch.Tensor,
    tx: torch.Tensor | None,
    num_classes: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        conditional_outer(z, tx, num_classes)


def test_factorized_heads_return_three_identity_adversaries_and_three_environment_heads() -> None:
    heads = FactorizedAdversarialHeads(160, 6, 4, 3, 4)
    z_id = torch.randn(8, 160, requires_grad=True)
    z_dom = torch.randn(8, 160, requires_grad=True)
    tx = torch.arange(8) % 6

    out = heads(z_id, z_dom, tx, grl_identity=0.2, grl_tx=0.08)

    assert set(out) == {
        "id_receiver",
        "id_day",
        "id_channel",
        "dom_receiver",
        "dom_day",
        "dom_channel",
        "dom_tx",
    }
    assert out["id_receiver"].shape == (8, 4)
    assert out["id_day"].shape == (8, 3)
    assert out["id_channel"].shape == (8, 4)
    assert out["dom_receiver"].shape == (8, 4)
    assert out["dom_day"].shape == (8, 3)
    assert out["dom_channel"].shape == (8, 4)
    assert out["dom_tx"].shape == (8, 6)


def test_factorized_heads_keep_all_head_gradients_finite_and_reachable() -> None:
    heads = FactorizedAdversarialHeads(8, 3, 2, 2, 4)
    z_id = torch.randn(6, 8, requires_grad=True)
    z_dom = torch.randn(6, 8, requires_grad=True)
    tx = torch.tensor([0, 1, 2, 0, 1, 2])

    out = heads(z_id, z_dom, tx, grl_identity=0.2, grl_tx=0.08)
    loss = sum(value.square().mean() for value in out.values())
    loss.backward()

    assert z_id.grad is not None and torch.isfinite(z_id.grad).all()
    assert z_dom.grad is not None and torch.isfinite(z_dom.grad).all()
    for parameter in heads.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_factorized_heads_require_tx_labels_for_conditional_identity_heads() -> None:
    heads = FactorizedAdversarialHeads(8, 3, 2, 2, 4)

    with pytest.raises(ValueError, match="TX labels"):
        heads(torch.randn(4, 8), torch.randn(4, 8), None)


def test_factorized_domain_projector_returns_independent_factor_projections() -> None:
    projector = FactorizedDomainProjector(feature_dim=6, factor_dim=3, interaction_dim=2)
    z_dom = torch.randn(5, 6, requires_grad=True)

    factors = projector(z_dom)

    assert isinstance(factors, DomainFactors)
    assert factors.z_r.shape == (5, 3)
    assert factors.z_d.shape == (5, 3)
    assert factors.z_c.shape == (5, 3)
    assert factors.z_int.shape == (5, 2)
    assert factors.z_int is not factors.z_r
    assert factors.z_int.data_ptr() != factors.z_r.data_ptr()
    assert all(torch.isfinite(value).all() for value in vars(factors).values())

    sum(factor.square().mean() for factor in vars(factors).values()).backward()
    assert z_dom.grad is not None
    assert torch.isfinite(z_dom.grad).all()
    assert all(parameter.grad is not None for parameter in projector.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in projector.parameters())


@pytest.mark.parametrize(
    ("z_dom", "message"),
    [
        (torch.randn(6), "shape"),
        (torch.randn(2, 5), "feature_dim"),
        (torch.tensor([[float("nan")] * 6]), "finite"),
    ],
)
def test_factorized_domain_projector_rejects_malformed_or_nonfinite_input(
    z_dom: torch.Tensor, message: str
) -> None:
    projector = FactorizedDomainProjector(feature_dim=6, factor_dim=3, interaction_dim=2)

    with pytest.raises(ValueError, match=message):
        projector(z_dom)


def test_conditional_cross_covariance_is_zero_without_valid_tx_group() -> None:
    z_id = torch.randn(3, 4, requires_grad=True)
    z_dom = torch.randn(3, 5, requires_grad=True)
    tx = torch.tensor([0, 1, 2])

    loss = conditional_cross_covariance(z_id, z_dom, tx)

    assert loss.item() == 0.0
    loss.backward()
    assert z_id.grad is not None
    assert z_dom.grad is not None
    assert torch.isfinite(z_id.grad).all()
    assert torch.isfinite(z_dom.grad).all()


def test_conditional_cross_covariance_averages_valid_tx_groups_only() -> None:
    z_id = torch.tensor(
        [[0.0, 0.0], [2.0, 0.0], [0.0, 0.0], [0.0, 0.0], [10.0, 10.0]],
        requires_grad=True,
    )
    z_dom = torch.tensor(
        [[0.0, 0.0], [0.0, 3.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0]],
        requires_grad=True,
    )
    tx = torch.tensor([0, 0, 1, 1, 2])

    loss = conditional_cross_covariance(z_id, z_dom, tx)

    # TX=0 contributes 9/(2*2)=2.25; TX=1 contributes zero; TX=2 is a singleton.
    assert loss.item() == pytest.approx(1.125)
    loss.backward()
    assert torch.isfinite(z_id.grad).all()
    assert torch.isfinite(z_dom.grad).all()
