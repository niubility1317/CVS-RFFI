from __future__ import annotations

from dataclasses import replace
import inspect
import math

import pytest
import torch

from cvsrffi import stage2_d127_da_candidates as d127


def _fsrg_asset(candidate_id: str) -> d127.FSRGAsset:
    torch.manual_seed(41 if candidate_id == d127.CANDIDATE_A else 43)
    dimension = 7
    return d127.FSRGAsset(
        candidate_id=candidate_id,
        tap_name=d127.TAP_A if candidate_id == d127.CANDIDATE_A else d127.TAP_B,
        U=torch.randn(
            dimension, d127.RANK, dtype=torch.float64, requires_grad=True
        ),
        V=torch.randn(
            d127.RANK, dimension, dtype=torch.float64, requires_grad=True
        ),
        d_f_diag=torch.tensor([0.25, 1.75], dtype=torch.float64),
        rho=0.23,
    )


def _support_loss(rows: torch.Tensor) -> torch.Tensor:
    direction = torch.tensor(
        [0.8, -0.5, 0.3, -0.7, 0.4, 0.1, -0.2],
        dtype=rows.dtype,
        device=rows.device,
    )
    target = torch.tensor(
        [-0.8, -0.3, 0.2, 0.7], dtype=rows.dtype, device=rows.device
    )
    return (rows @ direction - target).square()


@pytest.mark.parametrize("candidate_id", [d127.CANDIDATE_A, d127.CANDIDATE_B])
def test_fsrg_contract_projection_outer_gradient_and_query_immutability(
    candidate_id: str,
) -> None:
    asset = _fsrg_asset(candidate_id)
    support = torch.tensor(
        [
            [0.4, -0.2, 0.5, 0.1, -0.3, 0.6, 0.2],
            [0.3, 0.7, -0.1, 0.2, 0.5, -0.4, 0.8],
            [-0.6, 0.2, 0.4, 0.9, -0.3, 0.1, -0.5],
            [0.8, -0.4, 0.2, -0.7, 0.6, 0.3, -0.1],
        ],
        dtype=torch.float64,
    )
    labels = torch.tensor([11, 11, 29, 29], dtype=torch.int64)

    state = d127.fit_fsrg_support_state(support, labels, asset, _support_loss)
    assert torch.linalg.vector_norm(state.support_gradient).item() > 0.0
    assert torch.linalg.vector_norm(state.a).item() <= state.rho + 1.0e-12
    assert torch.all(torch.abs(state.a) <= state.a_max + 1.0e-12)
    assert state.rho == asset.rho
    assert state.a_max == asset.a_max
    assert state.a.requires_grad is False
    assert state.receipt.protocol_closed
    assert state.receipt.phase2_backward_calls == 1
    assert state.receipt.phase2_optimizer_steps == 0
    assert state.receipt.adapter_macs_per_sample == 4 * asset.dimension
    assert asset.U.grad is None and asset.V.grad is None

    outer = d127.apply_fsrg_outer(support[:2], asset, state).square().mean()
    outer.backward()
    assert asset.U.grad is not None and asset.V.grad is not None
    assert torch.linalg.vector_norm(asset.U.grad).item() > 0.0
    assert torch.linalg.vector_norm(asset.V.grad).item() > 0.0

    query = support[:2].detach().clone().requires_grad_(True)
    before = state.a.clone()
    adapted = d127.adapt_fsrg_query(query, asset, state)
    assert adapted.requires_grad is False
    assert query.grad is None
    assert torch.equal(before, state.a)
    assert not torch.equal(adapted, query.detach())


def test_rank2_projection_enforces_l2_and_coordinate_budget() -> None:
    rho = 0.4
    value = d127.project_rank2_coefficients(
        torch.tensor([20.0, -3.0], dtype=torch.float64), rho
    )
    assert torch.linalg.vector_norm(value).item() <= rho + 1.0e-12
    assert torch.all(torch.abs(value) <= rho / math.sqrt(2.0) + 1.0e-12)


def test_fsrg_rejects_nonfrozen_candidate_tap_pair() -> None:
    with pytest.raises(d127.D127DACandidateError, match="binding"):
        d127.FSRGAsset(
            candidate_id=d127.CANDIDATE_A,
            tap_name=d127.TAP_B,
            U=torch.ones(4, 2, dtype=torch.float64),
            V=torch.ones(2, 4, dtype=torch.float64),
            d_f_diag=torch.ones(2, dtype=torch.float64),
            rho=0.1,
        )


def test_fsrg_channel_time_tap_has_same_shape_gradient_and_sealed_budget() -> None:
    torch.manual_seed(57)
    support = torch.randn(4, 4, 3, dtype=torch.float64)
    labels = torch.tensor([1, 1, 8, 8], dtype=torch.int64)
    rho = d127.derive_phase1_fsrg_rho(support, dimension=4)
    expected = 0.05 * torch.quantile(
        torch.linalg.vector_norm(support.reshape(4, -1), dim=1), 0.5
    ).item()
    assert rho == expected
    asset = d127.FSRGAsset(
        candidate_id=d127.CANDIDATE_A,
        tap_name=d127.TAP_A,
        U=torch.randn(4, 2, dtype=torch.float64, requires_grad=True),
        V=torch.randn(2, 4, dtype=torch.float64, requires_grad=True),
        d_f_diag=torch.tensor([0.5, 1.25], dtype=torch.float64),
        rho=rho,
    )
    observed_shapes: list[tuple[int, ...]] = []

    def loss(rows: torch.Tensor) -> torch.Tensor:
        observed_shapes.append(tuple(rows.shape))
        return rows.square().mean(dim=(1, 2))

    state = d127.fit_fsrg_support_state(support, labels, asset, loss)
    amplified = d127.fit_fsrg_support_state(support * 7.0, labels, asset, loss)
    assert observed_shapes == [(4, 4, 3), (4, 4, 3)]
    assert state.rho == asset.rho == amplified.rho
    assert state.a_max == asset.rho / math.sqrt(2.0)
    assert torch.linalg.vector_norm(state.a).item() <= asset.rho + 1.0e-12
    assert torch.all(torch.abs(state.a) <= asset.a_max + 1.0e-12)
    assert state.receipt.adapter_macs_per_sample == 4 * 4 * 3

    outer = d127.apply_fsrg_outer(support, asset, state).square().mean()
    outer.backward()
    assert asset.U.grad is not None and asset.V.grad is not None
    assert torch.linalg.vector_norm(asset.U.grad).item() > 0.0
    assert torch.linalg.vector_norm(asset.V.grad).item() > 0.0
    query = support[:1].detach().clone().requires_grad_(True)
    adapted = d127.adapt_fsrg_query(query, asset, state)
    assert tuple(adapted.shape) == (1, 4, 3)
    assert adapted.requires_grad is False
    assert query.grad is None


def _rdha_asset(*, trainable: bool = False) -> d127.RDHAAsset:
    U = torch.zeros(320, 2, dtype=torch.float64)
    U[0, 0] = 1.0
    U[1, 1] = 1.0
    V = torch.zeros(2, 320, dtype=torch.float64)
    V[0, 0] = 1.0
    V[1, 1] = 1.0
    Q = torch.tensor(
        [[0.7, -0.4, 0.9, 0.2, -0.3], [-0.5, 0.6, -0.2, 0.8, 0.4]],
        dtype=torch.float64,
    )
    b = torch.tensor([0.03, -0.04], dtype=torch.float64)
    if trainable:
        for value in (U, V, Q, b):
            value.requires_grad_(True)
    return d127.RDHAAsset(
        U=U,
        V=V,
        Q=Q,
        b=b,
        mean_p1=torch.tensor([0.12, -0.18, 0.03, -0.04, 0.05], dtype=torch.float64),
        std_p1=torch.tensor([0.8, 1.1, 0.7, 1.3, 0.9], dtype=torch.float64),
        a_max=0.12,
    )


def _rdha_support() -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.zeros(6, 320, dtype=torch.float64)
    rows[:, 2] = torch.linspace(-0.2, 0.3, 6, dtype=torch.float64)
    rows[0, :2] = torch.tensor([1.5, -0.3])
    rows[1, :2] = torch.tensor([1.2, -0.1])
    rows[2, :2] = torch.tensor([-0.8, 1.0])
    rows[3, :2] = torch.tensor([-0.7, 0.8])
    rows[4, :2] = torch.tensor([0.2, -1.4])
    rows[5, :2] = torch.tensor([0.4, -1.1])
    return rows, torch.tensor([4, 4, 9, 9, 17, 17], dtype=torch.int64)


def test_rdah_summary_is_permutation_invariant_and_has_no_phase2_backward() -> None:
    asset = _rdha_asset()
    support, labels = _rdha_support()
    state = d127.fit_rdah_support_state(support, labels, asset)
    permuted = d127.fit_rdah_support_state(
        support, torch.tensor([21, 21, 5, 5, 13, 13], dtype=torch.int64), asset
    )

    assert torch.equal(state.summary, permuted.summary)
    assert torch.equal(state.a, permuted.a)
    assert not torch.allclose(state.a, asset.a_max * torch.tanh(asset.b))
    assert (
        torch.linalg.vector_norm(state.a).item()
        <= math.sqrt(2.0) * asset.a_max + 1.0e-12
    )
    assert torch.all(torch.abs(state.a) <= asset.a_max + 1.0e-12)
    assert state.receipt.protocol_closed
    assert state.receipt.phase2_backward_calls == 0
    assert state.receipt.phase2_optimizer_steps == 0
    assert state.receipt.support_gradient_calls == 0

    query = support[:2].detach().clone().requires_grad_(True)
    before = state.a.clone()
    adapted = d127.adapt_rdah_query(query, asset, state)
    assert adapted.requires_grad is False
    assert query.grad is None
    assert torch.equal(before, state.a)
    assert not torch.equal(adapted, query.detach())


def test_rdah_rejects_unequal_k_shot_support() -> None:
    asset = _rdha_asset()
    support, _labels = _rdha_support()
    with pytest.raises(d127.D127DACandidateError, match="same K-shot"):
        d127.fit_rdah_support_state(
            support[:5], torch.tensor([0, 0, 1, 1, 2], dtype=torch.int64), asset
        )


def test_rdah_uses_sealed_mean_shift_before_std_scaling() -> None:
    asset = _rdha_asset()
    support, labels = _rdha_support()
    shifted = d127.fit_rdah_support_state(support, labels, asset)
    zero_mean = d127.fit_rdah_support_state(
        support, labels, replace(asset, mean_p1=torch.zeros_like(asset.mean_p1))
    )
    expected = (shifted.summary - asset.mean_p1) / asset.std_p1
    assert torch.equal(shifted.standardized_summary, expected)
    assert torch.allclose(
        shifted.standardized_summary - zero_mean.standardized_summary,
        -asset.mean_p1 / asset.std_p1,
        atol=1.0e-14,
        rtol=0.0,
    )
    assert not torch.equal(shifted.a, zero_mean.a)


def test_rdah_phase1_outer_has_nonzero_uvqb_gradients_and_phase2_stays_no_grad() -> None:
    asset = _rdha_asset(trainable=True)
    support, labels = _rdha_support()
    with pytest.raises(d127.D127DACandidateError, match="separate tensors"):
        d127.apply_rdah_outer(support, labels, support, asset)
    query = support[:2].detach().clone()
    query[:, :3] += torch.tensor([0.15, -0.07, 0.11], dtype=query.dtype)
    assert support.data_ptr() != query.data_ptr()

    outer = d127.apply_rdah_outer(support, labels, query, asset)
    alternate = d127.apply_rdah_outer(support, labels, query * 1.7, asset)
    assert tuple(outer.adapted_support.shape) == tuple(support.shape)
    assert tuple(outer.adapted_query.shape) == tuple(query.shape)
    assert outer.a.requires_grad
    assert torch.equal(outer.summary, alternate.summary)
    assert torch.equal(outer.a, alternate.a)
    outer_loss = (
        outer.adapted_support.square().mean()
        + outer.adapted_query.square().mean()
    )
    outer_loss.backward()
    for value in (asset.U, asset.V, asset.Q, asset.b):
        assert value.grad is not None
        assert torch.linalg.vector_norm(value.grad).item() > 0.0

    for value in (asset.U, asset.V, asset.Q, asset.b):
        value.grad = None
    state = d127.fit_rdah_support_state(support, labels, asset)
    phase2_query = d127.adapt_rdah_query(
        query.detach().clone().requires_grad_(True), asset, state
    )
    assert phase2_query.requires_grad is False
    assert all(value.grad is None for value in (asset.U, asset.V, asset.Q, asset.b))


def test_public_phase2_surfaces_have_no_truth_role_quota_or_query_labels() -> None:
    assert set(inspect.signature(d127.fit_fsrg_support_state).parameters) == {
        "support_taps",
        "support_labels",
        "asset",
        "per_sample_loss",
    }
    assert set(inspect.signature(d127.fit_rdah_support_state).parameters) == {
        "support_hidden",
        "support_labels",
        "asset",
    }
    forbidden = {
        "truth",
        "role",
        "quota",
        "selection",
        "query_labels",
        "query_truth",
        "global_assignment",
        "source_rows",
        "clean_rows",
    }
    for function in (
        d127.adapt_fsrg_query,
        d127.adapt_rdah_query,
        d127.adapt_fsrg_support,
        d127.adapt_rdah_support,
    ):
        assert not set(inspect.signature(function).parameters) & forbidden
