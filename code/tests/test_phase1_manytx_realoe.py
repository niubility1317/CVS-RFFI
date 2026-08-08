from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from SSDG.train_ssdg import (
    _MANYTX_REAL_OE_LOCKED_TARGET_NEW_TX,
    _MANYTX_REAL_OE_PARTITION_ROOT_SHA256,
    _MANYTX_REAL_OE_PROXY_TX,
    _MANYTX_REAL_OE_RESERVE_TX,
    _MANYTX_REAL_OE_SOURCE_DAY_LABELS,
    _MANYTX_REAL_OE_SOURCE_RX_LABELS,
    _MANYTX_REAL_OE_TARGET_RX_LABELS,
    _MANYTX_REAL_OE_TRAIN_TX,
    _ManyTxRealOeBalancedBatchSampler,
    _manytx_real_oe_coverage_meets_contract,
    _phase1_tx_partition_view,
    _validate_manytx_real_oe_config,
    build_arg_parser,
)
from cvsrffi.losses import real_oe_energy_ranking_loss


def _csv(items) -> str:
    return ",".join(str(item) for item in items)


def _frozen_args(*, enabled: bool = True):
    argv = [
        "--output_dir",
        "unused",
        "--phase1_source_train_tx_ids",
        "14-10,14-7,20-15,20-19,6-15",
        "--phase1_source_known_validation_tx_ids",
        "8-20",
        "--wisig_train_days",
        _csv(_MANYTX_REAL_OE_SOURCE_DAY_LABELS),
        "--wisig_train_rxs",
        _csv(_MANYTX_REAL_OE_SOURCE_RX_LABELS),
        "--wisig_test_rxs",
        _csv(_MANYTX_REAL_OE_TARGET_RX_LABELS),
        "--phase1_allow_empty_proxy_unknown",
        "true",
        "--manytx_real_oe_protocol_enabled",
        "true",
        "--manytx_real_oe_enabled",
        str(enabled).lower(),
        "--manytx_real_oe_train_tx_ids",
        _csv(_MANYTX_REAL_OE_TRAIN_TX),
        "--manytx_real_oe_proxy_tx_ids",
        _csv(_MANYTX_REAL_OE_PROXY_TX),
        "--manytx_real_oe_reserve_tx_ids",
        _csv(_MANYTX_REAL_OE_RESERVE_TX),
        "--manytx_locked_target_new_tx_ids",
        _csv(_MANYTX_REAL_OE_LOCKED_TARGET_NEW_TX),
        "--manytx_real_oe_partition_root_sha256",
        _MANYTX_REAL_OE_PARTITION_ROOT_SHA256,
        "--manytx_real_oe_days",
        _csv(_MANYTX_REAL_OE_SOURCE_DAY_LABELS),
        "--manytx_real_oe_rxs",
        _csv(_MANYTX_REAL_OE_SOURCE_RX_LABELS),
        "--lambda_manytx_real_oe",
        "0.02" if enabled else "0",
    ]
    if enabled:
        argv.extend(["--manytx_real_oe_pkl", "manytx-source-only.pkl"])
    return build_arg_parser().parse_args(argv)


def test_manytx_realoe_parser_and_frozen_partition_receipt():
    args = _frozen_args(enabled=True)
    receipt = _validate_manytx_real_oe_config(args)

    assert receipt["schema"] == "cvs.phase1.manytx_real_oe_receipt.v2"
    assert receipt["enabled"] is True
    assert receipt["partition_root_sha256"] == _MANYTX_REAL_OE_PARTITION_ROOT_SHA256
    assert (len(receipt["oe_train_tx"]), len(receipt["proxy_tx"]), len(receipt["reserve_tx"])) == (80, 20, 16)
    assert receipt["eligible_extra_count"] == 116
    assert receipt["known_source_day_labels"] == list(_MANYTX_REAL_OE_SOURCE_DAY_LABELS)
    assert receipt["known_source_receiver_labels"] == list(_MANYTX_REAL_OE_SOURCE_RX_LABELS)
    assert receipt["known_target_receiver_labels"] == list(_MANYTX_REAL_OE_TARGET_RX_LABELS)
    assert receipt["oe_source_receiver_labels"] == list(_MANYTX_REAL_OE_SOURCE_RX_LABELS)
    assert receipt["proxy_loaded_by_training"] is False
    assert receipt["locked_target_new_loaded_by_training"] is False
    assert receipt["reserve_loaded_by_training"] is False


def test_manytx_realoe_rejects_any_locked_target_new_redefinition():
    args = _frozen_args(enabled=False)
    args.manytx_locked_target_new_tx_ids = _csv(
        ("10-4",) + _MANYTX_REAL_OE_LOCKED_TARGET_NEW_TX[1:]
    )

    with pytest.raises(ValueError, match="does not match the frozen ManyTx authority list"):
        _validate_manytx_real_oe_config(args)


def test_manytx_realoe_rejects_stacked_proxy_or_virtual_loss():
    args = _frozen_args(enabled=True)
    args.lambda_proxy_unknown = 0.01

    with pytest.raises(ValueError, match="forbids stacked proxy/virtual/geometry losses"):
        _validate_manytx_real_oe_config(args)


@pytest.mark.parametrize(
    ("field", "raw_indices"),
    [
        ("wisig_train_days", "0,1"),
        ("wisig_train_rxs", "0,1,2,3,4,5"),
        ("wisig_test_rxs", "7,8,9,10,11"),
        ("manytx_real_oe_rxs", "0,1,2,3,4,5"),
    ],
)
def test_manytx_realoe_rejects_raw_receiver_or_day_indices(field, raw_indices):
    args = _frozen_args(enabled=False)
    setattr(args, field, raw_indices)

    with pytest.raises(ValueError, match="rejects raw index strings"):
        _validate_manytx_real_oe_config(args)


def test_manytx_realoe_coverage_allows_two_of_the_six_common_physical_receivers():
    assert _manytx_real_oe_coverage_meets_contract(
        400,
        {3, 7},
        {1, 4},
        expected_days=[3, 7],
    )
    assert not _manytx_real_oe_coverage_meets_contract(
        400,
        {3, 7},
        {1},
        expected_days=[3, 7],
    )
    assert not _manytx_real_oe_coverage_meets_contract(
        399,
        {3, 7},
        {1, 4},
        expected_days=[3, 7],
    )


def test_empty_main_proxy_is_only_allowed_for_the_frozen_external_protocol():
    ds = {
        "tx_list": ["14-10", "14-7", "20-15", "20-19", "6-15", "8-20"],
        "data": [[name] for name in ("14-10", "14-7", "20-15", "20-19", "6-15", "8-20")],
        "rx_list": ["rx0"],
        "capture_date_list": ["day0"],
    }
    with pytest.raises(ValueError, match="requires non-empty"):
        _phase1_tx_partition_view(
            ds,
            train_spec="14-10,14-7,20-15,20-19,6-15",
            known_validation_spec="8-20",
            proxy_unknown_spec="",
        )

    _, receipt = _phase1_tx_partition_view(
        ds,
        train_spec="14-10,14-7,20-15,20-19,6-15",
        known_validation_spec="8-20",
        proxy_unknown_spec="",
        allow_empty_proxy_unknown=True,
    )
    assert receipt["allow_empty_proxy_unknown"] is True
    assert receipt["source_proxy_unknown_tx"] == []


def test_real_oe_energy_loss_only_backpropagates_through_observed_oe_logits():
    known = torch.randn(4, 5, requires_grad=True)
    oe = torch.randn(4, 5, requires_grad=True)

    loss, metrics = real_oe_energy_ranking_loss(known, oe, margin=1.0, temperature=1.0, tau=1.0)
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["active"] == 1.0
    assert known.grad is None
    assert oe.grad is not None
    assert float(oe.grad.abs().sum()) > 0.0


@pytest.mark.parametrize(
    ("known", "oe", "match"),
    [
        (torch.randn(2, 3), torch.randn(2, 4), "class dimension mismatch"),
        (torch.full((2, 3), float("nan")), torch.randn(2, 3), "non-finite known"),
        (torch.randn(2, 3), torch.full((2, 3), float("inf")), "non-finite OE"),
    ],
)
def test_real_oe_energy_loss_rejects_invalid_logit_contract(known, oe, match):
    with pytest.raises(ValueError, match=match):
        real_oe_energy_ranking_loss(known, oe)


def test_realoe_sampler_balances_hidden_base_tx_metadata_without_returning_labels():
    base = SimpleNamespace(
        index=[
            SimpleNamespace(tx_i=tx_i)
            for tx_i in range(3)
            for _ in range(4)
        ]
    )
    hidden_subset = SimpleNamespace(base=base, selected=list(range(12)))
    sampler = _ManyTxRealOeBalancedBatchSampler(
        hidden_subset,
        tx_per_batch=2,
        samples_per_tx=3,
        batches_per_epoch=1,
        seed=7,
    )

    batch = next(iter(sampler))
    tx_counts = {}
    for sample_index in batch:
        tx_i = base.index[hidden_subset.selected[sample_index]].tx_i
        tx_counts[tx_i] = tx_counts.get(tx_i, 0) + 1

    assert len(batch) == 6
    assert sorted(tx_counts.values()) == [3, 3]


def test_lite_d_realoe_forward_backward_smoke_has_no_query_input():
    from model_dual_cvsincnet import build_dual_model

    torch.manual_seed(31)
    model = build_dual_model(
        num_classes=5,
        num_domains=4,
        dataset="wisig",
        input_len=128,
        model_variant="lite_d",
    ).train()
    x_known = torch.randn(4, 2, 128)
    x_oe = torch.randn(4, 2, 128)
    y_known = torch.tensor([0, 1, 2, 3])
    d_known = torch.tensor([0, 1, 2, 3])

    out_known = model(x_known, y_tx=y_known, domain_labels=d_known, return_aux=True)
    out_oe = model(x_oe, y_tx=None, domain_labels=None, return_aux=True)
    loss, _ = real_oe_energy_ranking_loss(out_known["tx_logits"], out_oe["tx_logits"])
    loss.backward()

    assert tuple(out_oe["z_id"].shape) == (4, 160)
    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
        for parameter in model.id_backbone.parameters()
    )
