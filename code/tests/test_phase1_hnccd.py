from __future__ import annotations

"""Mechanical contract tests for frozen P1-HNCCD.

The scientific core belongs to ``cvsrffi.phase1_hnccd``.  This file checks
only its already-frozen public contract; it never selects a fold, receiver,
day, or proxy result.
"""

import ast
import inspect
import json
import re
import subprocess
import sys
import textwrap
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi import phase1_hnccd as HNCCD  # noqa: E402
import SSDG.train_ssdg as train_ssdg  # noqa: E402


CLASS_IDS = HNCCD.FROZEN_HNCCD_CLASS_IDS
RX_IDS = tuple(range(HNCCD.FROZEN_HNCCD_SOURCE_RECEIVER_COUNT))
SCENARIOS = HNCCD.FROZEN_HNCCD_SCENARIOS
LAUNCHER = CODE_ROOT / "scripts" / "launch_phase1_hnccd12_20260811.sh"


def _frozen_args(*, enabled: bool = True, epochs: int = 40, batch_size: int = 128) -> SimpleNamespace:
    args = {
        "phase1_hnccd_frozen_mode": True,
        "phase1_hnccd_enabled": enabled,
        "lambda_hnccd": HNCCD.FROZEN_HNCCD_LAMBDA if enabled else 0.0,
        "batch_size": batch_size,
        "from_scratch": False,
        "baseline_ckpt": "geosat_c_final.pth",
        "freeze_backbone": False,
        "amp": True,
        "id_feature_key": "feat_joint",
        "epochs": epochs,
        "label_epochs": epochs,
        "pseudo_epochs": 0,
        "checkpoint_selection": "final_only",
        "phase1_source_val_selection_only": True,
        "use_sat_consistency": True,
        "lambda_sat_cons": 0.10,
        "lambda_sat_cls": 0.0,
        "sat_cons_start_epoch": 1,
        "sat_view_prob": 1.0,
        "sat_train_scenarios": ",".join(SCENARIOS),
        "sat_view_schedule": "",
        "use_concat_sat_channel_aug": False,
        "use_unlabeled": False,
        "use_tx_rx_balanced_sampler": False,
        "use_aug": False,
        "use_mixstyle": False,
        "reject_head": False,
        "manytx_real_oe_enabled": False,
        "manytx_real_oe_protocol_enabled": False,
        "use_ema_teacher": False,
        "teacher_ckpt": "",
        "lambda_teacher_clean_kl": 0.0,
        "lambda_teacher_sat_kl": 0.0,
        "lambda_teacher_zid_mse": 0.0,
    }
    for peer in (
        "ccpc_leo",
        "pamr",
        "cb_sfce",
        "gd_proto_nll",
        "icmt",
        "cagm",
        "rcrmd",
        "rcat",
        "rcmmc",
        "hscf",
        "recte",
        "cp_sfce",
    ):
        args[f"phase1_{peer}_frozen_mode"] = False
        args[f"phase1_{peer}_enabled"] = False
        args[f"lambda_{peer}"] = 0.0
    for name in (
        "lambda_domain",
        "lambda_adv",
        "lambda_orth",
        "lambda_cons",
        "lambda_group_ce",
        "lambda_fishr",
        "lambda_u",
        "lambda_ent",
        "lambda_u_domain",
        "lambda_u_adv",
        "lambda_u_sat_cons",
        "lambda_u_direct_metric_accept",
        "lambda_u_quarantine_accept",
        "lambda_zid_receiver_invariance",
        "lambda_zid_day_invariance",
        "lambda_zid_channel_invariance",
        "lambda_u_zid_receiver_invariance",
        "lambda_u_zid_day_invariance",
        "lambda_u_zid_channel_invariance",
        "lambda_tx_proto",
        "lambda_rx_proto",
        "lambda_mask_aux",
        "lambda_tx_supcon_masked",
        "lambda_rx_supcon_masked",
        "lambda_txrx_rect",
        "lambda_proto",
        "lambda_open_world_feat",
        "lambda_zid_compact",
        "lambda_proxy_unknown",
        "lambda_manytx_real_oe",
        "lambda_soft_unknown_mixup",
        "lambda_source_episode",
        "lambda_direct_metric_accept",
    ):
        args[name] = 0.0
    for name in (
        "use_phase2_ground_prototypes",
        "use_feature_masks",
        "use_txrx_geometry_losses",
        "use_proto_memory",
        "os_gradient_surgery",
        "os_budget_controller",
        "os_objective_budget_controller",
        "phase1_v2_hard_gates",
    ):
        args[name] = False
    return SimpleNamespace(**args)


class _BindingModel(torch.nn.Module):
    def __init__(self, dim: int = HNCCD.FROZEN_HNCCD_FEATURE_DIM) -> None:
        super().__init__()
        self.id_backbone = torch.nn.Module()
        self.id_backbone.encoder = torch.nn.Linear(dim, dim, bias=False)
        self.id_backbone.cls_head = torch.nn.Module()
        self.id_backbone.cls_head.head = torch.nn.Linear(dim, len(CLASS_IDS), bias=True)
        with torch.no_grad():
            self.id_backbone.encoder.weight.copy_(torch.eye(dim))
            self.id_backbone.cls_head.head.weight.zero_()
            self.id_backbone.cls_head.head.weight[:, : len(CLASS_IDS)] = torch.eye(len(CLASS_IDS))
            self.id_backbone.cls_head.head.bias.zero_()

    def paired_output(self, x: torch.Tensor) -> dict[str, torch.Tensor | str]:
        z_id = self.id_backbone.encoder(x)
        return {
            "z_id": z_id,
            "z_id_key": "feat_joint",
            "tx_logits": self.id_backbone.cls_head.head(z_id),
        }


class _RecoveringScaler:
    """CPU GradScaler double with the one skip/backoff behavior HNCCD seals."""

    def __init__(self, scale: float = 1.0e5) -> None:
        self._scale = float(scale)
        self._found_nonfinite = False
        self.scale_calls = 0
        self.unscale_calls = 0
        self.step_calls = 0
        self.update_calls = 0

    def get_scale(self) -> float:
        return self._scale

    def scale(self, value: torch.Tensor) -> torch.Tensor:
        self.scale_calls += 1
        return value * self._scale

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        self.unscale_calls += 1
        self._found_nonfinite = False
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                parameter.grad.div_(self._scale)
                if not bool(torch.isfinite(parameter.grad.detach()).all().item()):
                    self._found_nonfinite = True

    def step(self, optimizer: torch.optim.Optimizer):
        self.step_calls += 1
        if self._found_nonfinite:
            return None
        return optimizer.step()

    def update(self) -> None:
        self.update_calls += 1
        if self._found_nonfinite:
            self._scale *= 0.5


class _SavedTensorToken:
    __slots__ = ("tensor", "__weakref__")

    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor.detach()


class _FiniteForwardNonfiniteBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor) -> torch.Tensor:
        return value.clone()

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> torch.Tensor:
        return torch.full_like(gradient, float("nan"))


def _all_cells(repeats: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.tensor(list(CLASS_IDS) * len(RX_IDS) * repeats, dtype=torch.long)
    receivers = torch.tensor([rx for rx in RX_IDS for _ in CLASS_IDS] * repeats, dtype=torch.long)
    return labels, receivers


def _batch_cells() -> tuple[torch.Tensor, torch.Tensor]:
    labels, receivers = _all_cells(repeats=4)
    extra_labels, extra_receivers = _all_cells(repeats=1)
    labels = torch.cat((labels, extra_labels[:16]))
    receivers = torch.cat((receivers, extra_receivers[:16]))
    assert labels.numel() == HNCCD.FROZEN_HNCCD_BATCH_SIZE
    return labels, receivers


def _loss(
    leo: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    receivers: torch.Tensor,
    *,
    tokens: tuple[int, ...] = RX_IDS,
    frozen: bool = False,
):
    return HNCCD.hnccd_loss(
        leo,
        weight,
        labels,
        receivers,
        tokens,
        require_frozen_shape=frozen,
    )


def _source_split_receipt() -> dict[str, object]:
    return {
        "schema": "cvs.phase1.source_split_receipt.v1",
        "labeled_indices_sha256": "1" * 64,
        "split_manifest_sha256": "2" * 64,
        "source_receivers": list(RX_IDS),
    }


def _class_binding() -> dict[str, object]:
    return HNCCD.resolve_hnccd_local_head_class_binding(
        local_class_order=["20-15", "20-19", "6-15", "8-20"],
        source_train_tx=["20-15", "20-19", "6-15", "8-20"],
        checkpoint_train_tx=["20-15", "20-19", "6-15", "8-20"],
        dataset_class_order=["14-10", "14-7", "20-15", "20-19", "6-15", "8-20"],
        local_data_class_count=4,
        checkpoint_head_class_count=4,
        live_head_class_count=4,
    )


def _sealed_receipt(*, enabled: bool) -> dict[str, object]:
    config = HNCCD.HNCCDConfig(True, enabled, 0.02 if enabled else 0.0)
    receipt = HNCCD.hnccd_config_receipt(config)
    receipt = HNCCD.bind_hnccd_source_data_order(receipt, _source_split_receipt())
    model = _BindingModel()
    receipt.update(
        HNCCD.strict_hnccd_warm_start(
            model,
            _BindingModel().state_dict(),
            baseline_path="geosat_c_final.pth",
            baseline_sha256="a" * 64,
            checkpoint_epoch=40,
            checkpoint_role="training_final_only",
        )
    )
    binding = _class_binding()
    receipt.update(binding)
    receipt.update(
        {
            "source_train_tx_count": 4,
            "source_known_validation_tx_count": 1,
            "source_proxy_unknown_tx_count": 1,
            "source_partition_sha256": "b" * 64,
            "common_l_base_head_input_path_verified": True,
        }
    )
    return HNCCD.bind_hnccd_optimizer_initial_state(
        receipt,
        torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4),
    )


def _metadata(rows: int, offset: int) -> dict[str, torch.Tensor]:
    return {
        "base_index": torch.arange(offset, offset + rows),
        "sig_i": torch.arange(offset, offset + rows) + 10000,
    }


def _common_bind(
    receipt: dict[str, object], *, epoch: int, batch_index: int, scenario: str
) -> dict[str, object]:
    labels, receivers = _batch_cells()
    bound = HNCCD.update_hnccd_common_batch_sequence_receipt(
        receipt,
        epoch=epoch,
        batch_index=batch_index,
        scenario=scenario,
        source_tx_labels=labels,
        source_rx_labels=receivers,
        source_receiver_tokens=RX_IDS,
        metadata=_metadata(int(labels.numel()), batch_index * 1000),
    )
    return HNCCD.update_hnccd_resource_receipt(
        bound,
        peak_memory_bytes=1_000_000 + batch_index,
        step_time_seconds=0.01 * batch_index,
    )


def _build_c_terminal_receipt() -> dict[str, object]:
    receipt = _sealed_receipt(enabled=False)
    for index, scenario in enumerate(SCENARIOS, start=1):
        receipt = _common_bind(receipt, epoch=1, batch_index=index, scenario=scenario)
    return receipt


def _build_g_terminal_receipt() -> dict[str, object]:
    receipt = _sealed_receipt(enabled=True)
    labels, receivers = _batch_cells()
    for index, scenario in enumerate(SCENARIOS, start=1):
        torch.manual_seed(900 + index)
        model = _BindingModel().train()
        clean = model.paired_output(torch.randn(128, 160))
        leo = model.paired_output(torch.randn(128, 160))
        weight = HNCCD.validate_hnccd_binding(
            model=model,
            out_clean=clean,
            out_leo=leo,
            tx_labels=labels,
            source_rx_labels=receivers,
            expected_class_ids=CLASS_IDS,
            source_receiver_tokens=RX_IDS,
            enforce_frozen_shape=True,
        )
        loss, info = _loss(leo["z_id"], weight, labels, receivers, frozen=True)
        audit = HNCCD.hnccd_aux_gradient_audit(
            loss,
            clean["z_id"],
            leo["z_id"],
            HNCCD.hnccd_shared_encoder_and_head_parameters(model),
        )
        receipt = _common_bind(receipt, epoch=1, batch_index=index, scenario=scenario)
        receipt = HNCCD.update_hnccd_receipt(
            receipt, info, scenario=scenario, epoch=1, batch_index=index
        )
        receipt = HNCCD.update_hnccd_gradient_audit_receipt(
            receipt, audit, scenario=scenario
        )
        receipt = HNCCD.update_hnccd_optimizer_step_receipt(receipt)
    return receipt


def _pending_g_amp_receipt() -> dict[str, object]:
    receipt = HNCCD.hnccd_config_receipt(HNCCD.HNCCDConfig(True, True, 0.02))
    receipt["hnccd_g_batch_aux"] = [
        {"epoch": 1, "batch_index": 1, "scenario": SCENARIOS[0]}
    ]
    return receipt


def test_frozen_constants_shape_ledger_and_disabled_old_routes() -> None:
    assert HNCCD.FROZEN_HNCCD_LAMBDA == pytest.approx(0.02)
    assert HNCCD.FROZEN_HNCCD_BATCH_SIZE == 128
    assert HNCCD.FROZEN_HNCCD_FEATURE_DIM == 160
    assert HNCCD.FROZEN_HNCCD_SOURCE_RECEIVER_COUNT == 7
    assert HNCCD.FROZEN_HNCCD_CELL_COUNT == 28
    assert HNCCD.FROZEN_HNCCD_TERM_DIVISOR == 28
    assert tuple(SCENARIOS) == ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
    ledger = HNCCD.hnccd_shape_ledger()
    assert ledger["batch_size"] == 128 and ledger["feature_dim"] == 160
    assert ledger["cell_count"] == 28 and ledger["cross_batch_cache"] is False
    assert ledger["forbids_batch_d2_materialization"] is True
    assert ledger["forbids_batch_cell_d2_materialization"] is True
    assert int(ledger["conservative_live_tensor_upper_bound_bytes"]) > 0

    assert HNCCD.validate_hnccd_args(_frozen_args(enabled=True)) == HNCCD.HNCCDConfig(True, True, 0.02)
    assert HNCCD.validate_hnccd_args(_frozen_args(enabled=False)) == HNCCD.HNCCDConfig(True, False, 0.0)
    for name, value in (
        ("lambda_hnccd", 0.01),
        ("batch_size", 64),
        ("epochs", 39),
        ("phase1_rcmmc_enabled", True),
        ("phase1_hscf_frozen_mode", True),
        ("use_unlabeled", True),
    ):
        bad = _frozen_args(enabled=True)
        setattr(bad, name, value)
        with pytest.raises(HNCCD.HNCCDConfigurationError):
            HNCCD.validate_hnccd_args(bad)


def test_totalized_zero_is_safe_and_nonfinite_is_fatal() -> None:
    zeros = torch.zeros((2, 4), requires_grad=True)
    totalized = HNCCD.totalized_l2(zeros)
    assert torch.equal(totalized, torch.zeros_like(totalized))
    totalized.sum().backward()
    assert zeros.grad is not None and int(torch.count_nonzero(zeros.grad).item()) == 0

    labels = torch.tensor([0, 1], dtype=torch.long)
    receivers = torch.tensor([0, 1], dtype=torch.long)
    weight = torch.eye(4, requires_grad=True)
    loss, info = _loss(zeros.detach().requires_grad_(True), weight, labels, receivers)
    assert float(loss.detach()) == pytest.approx(0.0)
    assert info["leo_zero_rows"] == 2 and info["n_lt_2_differentiable_zero"] is True
    with pytest.raises(HNCCD.HNCCDRuntimeError, match="non-finite"):
        HNCCD.totalized_l2(torch.tensor([[float("nan"), 0.0]]))
    bad = torch.zeros((2, 4))
    bad[0, 0] = float("inf")
    with pytest.raises(HNCCD.HNCCDRuntimeError, match="non-finite"):
        _loss(bad, torch.eye(4), labels, receivers)


def test_fixed_28_n_lt_2_zero_and_no_active_renormalization() -> None:
    labels = torch.tensor([0, 1], dtype=torch.long)
    receivers = torch.tensor([0, 1], dtype=torch.long)
    leo = torch.tensor([[1.0, 2.0, 0.0, 0.0], [0.0, 1.0, 2.0, 0.0]], requires_grad=True)
    weight = torch.eye(4, requires_grad=True)
    loss, info = _loss(leo, weight, labels, receivers)
    assert torch.isfinite(loss.detach()) and float(loss.detach()) == pytest.approx(0.0)
    assert info["global_denominator"] == 28 and info["no_active_renormalization"] is True
    cells = info["cells"]
    assert len(cells) == 28 and info["insufficient_cells"] == 28
    assert all(cell["n_rc"] < 2 and cell["loss_contribution"] == 0.0 for cell in cells.values())
    loss.backward()
    assert leo.grad is not None and int(torch.count_nonzero(leo.grad).item()) == 0


def test_cholesky_rank_failure_and_no_fallback() -> None:
    torch.manual_seed(41)
    weight = torch.randn(4, 11, requires_grad=True)
    q = HNCCD.hnccd_head_null_basis(weight)
    assert q.shape == (11, 4)
    assert torch.allclose(q.transpose(0, 1).matmul(q), torch.eye(4), atol=1e-5, rtol=1e-5)
    singular = torch.zeros((4, 11))
    singular[0, 0] = singular[1, 0] = singular[2, 1] = singular[3, 2] = 1.0
    with pytest.raises(HNCCD.HNCCDRuntimeError, match="Cholesky|full-row-rank"):
        HNCCD.hnccd_head_null_basis(singular)
    nonfinite = weight.detach().clone()
    nonfinite[0, 0] = float("nan")
    with pytest.raises(HNCCD.HNCCDRuntimeError, match="non-finite"):
        HNCCD.hnccd_head_null_basis(nonfinite)
    source = Path(HNCCD.__file__).read_text(encoding="utf-8")
    assert "torch.linalg.cholesky" in source and "torch.linalg.solve_triangular" in source
    assert "torch.linalg.pinv" not in source and "torch.linalg.lstsq" not in source


def test_gauge_class_and_receiver_permutation_invariance() -> None:
    torch.manual_seed(42)
    labels, receivers = _batch_cells()
    leo = torch.randn(128, 160)
    weight = torch.randn(4, 160)
    direct, _ = _loss(leo, weight, labels, receivers, frozen=True)
    gauge = torch.tensor(
        [[2.0, 0.0, 0.0, 0.0], [0.2, 0.5, 0.0, 0.0], [0.0, 0.1, 1.5, 0.0], [0.0, 0.0, 0.3, 0.7]]
    )
    gauged, _ = _loss(leo, gauge.matmul(weight), labels, receivers, frozen=True)
    row_permuted, _ = _loss(leo, weight.index_select(0, torch.tensor([2, 0, 3, 1])), labels, receivers, frozen=True)
    class_permuted, _ = _loss(leo, weight, torch.tensor([2, 0, 3, 1])[labels], receivers, frozen=True)
    receiver_permuted, _ = _loss(leo, weight, labels, receivers, tokens=tuple(reversed(RX_IDS)), frozen=True)
    for candidate in (gauged, row_permuted, class_permuted, receiver_permuted):
        assert torch.allclose(direct, candidate, atol=2e-5, rtol=2e-5)


def test_loss_bound_lambda_and_control_identity() -> None:
    torch.manual_seed(43)
    labels, receivers = _batch_cells()
    leo = torch.randn(128, 160, requires_grad=True)
    weight = torch.randn(4, 160, requires_grad=True)
    loss, info = _loss(leo, weight, labels, receivers, frozen=True)
    assert torch.isfinite(loss.detach()) and 0.0 <= float(loss.detach()) <= 1.0 + 1e-5
    assert 0 <= info["positive_c_cells"] <= 28 and info["finite_c_cells"] == 28
    base = torch.tensor(0.0, requires_grad=True)
    control = HNCCD.add_hnccd_to_loss(base, None, HNCCD.HNCCDConfig(True, False, 0.0))
    assert control is base
    combined = HNCCD.add_hnccd_to_loss(base, loss, HNCCD.HNCCDConfig(True, True, 0.02))
    assert torch.allclose(combined - base, loss * HNCCD.FROZEN_HNCCD_LAMBDA)
    assert float((combined - base).detach()) <= 0.02 + 1e-6
    with pytest.raises(HNCCD.HNCCDRuntimeError, match="auxiliary"):
        HNCCD.add_hnccd_to_loss(base, None, HNCCD.HNCCDConfig(True, True, 0.02))


def test_source_only_permissions_and_class_receiver_binding() -> None:
    names = tuple(inspect.signature(HNCCD.hnccd_loss).parameters)
    assert names[:5] == (
        "leo_feat_joint",
        "head_weight",
        "source_tx_labels",
        "source_rx_labels",
        "source_receiver_tokens",
    )
    assert not {"clean", "query", "proxy", "target", "day"}.intersection(names)
    assert HNCCD.resolve_hnccd_source_receiver_tokens(
        {"schema": "cvs.phase1.source_split_receipt.v1", "source_receivers": list(RX_IDS)}
    ) == RX_IDS
    with pytest.raises(HNCCD.HNCCDConfigurationError, match="seven|unique"):
        HNCCD.resolve_hnccd_source_receiver_tokens({"source_receivers": [0, 1]})
    binding = HNCCD.resolve_hnccd_local_head_class_binding(
        local_class_order=["20-15", "20-19", "6-15", "8-20"],
        source_train_tx=["20-15", "20-19", "6-15", "8-20"],
        checkpoint_train_tx=["20-15", "20-19", "6-15", "8-20"],
        dataset_class_order=["14-10", "14-7", "20-15", "20-19", "6-15", "8-20"],
        local_data_class_count=4,
        checkpoint_head_class_count=4,
        live_head_class_count=4,
    )
    assert binding["local_to_dataset_class_ids"] == [2, 3, 4, 5]
    assert torch.equal(HNCCD.remap_hnccd_local_labels_to_head_rows(torch.tensor([3, 0, 2]), binding["local_to_head_class_ids"]), torch.tensor([3, 0, 2]))


def test_raw_vjp_scope_leo_encoder_weight_clean_and_bias() -> None:
    torch.manual_seed(44)
    model = _BindingModel().train()
    labels, receivers = _batch_cells()
    clean = model.paired_output(torch.randn(128, 160))
    leo = model.paired_output(torch.randn(128, 160))
    clean["z_id"].retain_grad()
    loss, _ = _loss(
        leo["z_id"],
        HNCCD.resolve_hnccd_classifier_weight(model),
        labels,
        receivers,
        frozen=True,
    )
    audit = HNCCD.hnccd_aux_gradient_audit(
        loss,
        clean["z_id"],
        leo["z_id"],
        HNCCD.hnccd_shared_encoder_and_head_parameters(model),
    )
    for key in ("feat_joint_leo", "shared_encoder", "head_weight"):
        assert audit[key]["norm"] > 0.0 and torch.isfinite(torch.tensor(audit[key]["norm"]))
    for key in ("clean_feat_joint", "head_bias"):
        assert audit[key]["nonzero_parameters"] == 0.0
        assert audit[key]["none_parameters"] + audit[key]["zero_parameters"] == audit[key]["parameter_count"]
    assert audit["raw_unscaled"] is True and audit["touches_amp_optimizer_rng"] is False
    loss.backward()
    assert clean["z_id"].grad is None


def test_current_batch_resource_contract_has_no_bxd2_or_cache() -> None:
    source = Path(HNCCD.__file__).read_text(encoding="utf-8")
    assert "B,28,d,d" not in source and "28,d,d" not in source
    assert "einsum('bi,bj->bij'" not in source and 'einsum("bi,bj->bij"' not in source
    ledger = HNCCD.hnccd_shape_ledger()
    assert ledger["current_batch_layout"] == "Q[d,4],h[B,4],b[B,d],per_cell_H_B_C[4,d]"
    assert ledger["uses_current_batch_only"] is True and ledger["cross_batch_cache"] is False


def test_source_order_warm_start_new_adamw_and_binding_receipts() -> None:
    receipt = HNCCD.hnccd_config_receipt(HNCCD.HNCCDConfig(True, True, 0.02))
    bound = HNCCD.bind_hnccd_source_data_order(receipt, _source_split_receipt())
    assert bound["source_receiver_count"] == 7
    assert len(bound["source_receiver_order_sha256"]) == 64
    assert "source_receivers" not in bound
    model = _BindingModel()
    warm = HNCCD.strict_hnccd_warm_start(
        model,
        _BindingModel().state_dict(),
        baseline_path="geosat_c_final.pth",
        baseline_sha256="a" * 64,
        checkpoint_epoch=40,
        checkpoint_role="training_final_only",
    )
    assert warm["strict_model_keys"] is True
    assert warm["optimizer_state_restored"] is False and warm["rng_state_restored"] is False
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    initial = HNCCD.bind_hnccd_optimizer_initial_state(bound, optimizer)
    assert initial["optimizer_type"] == "AdamW" and initial["optimizer_initial_state_empty"] is True
    with pytest.raises(HNCCD.HNCCDConfigurationError, match="AdamW"):
        HNCCD.bind_hnccd_optimizer_initial_state(bound, torch.optim.SGD(model.parameters(), lr=1e-3))
    with pytest.raises(HNCCD.HNCCDConfigurationError, match="training_final_only"):
        HNCCD.strict_hnccd_warm_start(
            model,
            _BindingModel().state_dict(),
            baseline_path="geosat_c_final.pth",
            baseline_sha256="a" * 64,
            checkpoint_epoch=40,
            checkpoint_role="best_metric",
        )


def test_common_c_and_g_three_scene_terminal_closure() -> None:
    control = _build_c_terminal_receipt()
    c_terminal = HNCCD.validate_hnccd_terminal_receipt(control)
    assert c_terminal["hnccd_terminal_contract_passed"] is True
    assert c_terminal["hnccd_terminal_contract"].startswith("CONTROL_ARM")
    assert c_terminal["hnccd_batches"] == 0 and c_terminal["hnccd_loss_sum"] == 0.0

    g_terminal = HNCCD.validate_hnccd_terminal_receipt(_build_g_terminal_receipt())
    assert g_terminal["hnccd_terminal_contract_passed"] is True
    assert g_terminal["hnccd_total_rows"] == 3 * 128
    assert set(g_terminal["hnccd_scenes"]) == set(SCENARIOS)
    assert set(g_terminal["hnccd_gradient_audit_scenes"]) == set(SCENARIOS)
    assert g_terminal["hnccd_optimizer_step_attempts"] == 3
    assert g_terminal["hnccd_effective_optimizer_steps"] == 3
    assert all(
        len(g_terminal["hnccd_scenes"][scenario]) == 28 for scenario in SCENARIOS
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "empty",
        "missing_one",
        "bool_peak",
        "negative_peak",
        "nan_step",
        "negative_step",
        "selection_feedback",
    ),
)
def test_terminal_rejects_incomplete_or_tampered_resource_receipts(mutation: str) -> None:
    receipt = _build_c_terminal_receipt()
    observations = [
        dict(observation) for observation in receipt["hnccd_resource_observations"]
    ]
    if mutation == "empty":
        observations = []
    elif mutation == "missing_one":
        observations = observations[:-1]
    elif mutation == "bool_peak":
        observations[0]["peak_memory_bytes"] = True
    elif mutation == "negative_peak":
        observations[0]["peak_memory_bytes"] = -1
    elif mutation == "nan_step":
        observations[0]["step_time_seconds"] = float("nan")
    elif mutation == "negative_step":
        observations[0]["step_time_seconds"] = -0.01
    elif mutation == "selection_feedback":
        observations[0]["selection_feedback"] = True
    receipt["hnccd_resource_observations"] = observations
    with pytest.raises(HNCCD.HNCCDRuntimeError, match="resource|peak|step|select"):
        HNCCD.validate_hnccd_terminal_receipt(receipt)


def test_common_physical_order_and_terminal_drift_fail_closed() -> None:
    labels, receivers = _batch_cells()
    receipt = _sealed_receipt(enabled=False)
    with pytest.raises(HNCCD.HNCCDRuntimeError, match="metadata|physical"):
        HNCCD.update_hnccd_common_batch_sequence_receipt(
            receipt,
            epoch=1,
            batch_index=1,
            scenario=SCENARIOS[0],
            source_tx_labels=labels,
            source_rx_labels=receivers,
            source_receiver_tokens=RX_IDS,
            metadata={},
        )
    bound = _common_bind(receipt, epoch=1, batch_index=1, scenario=SCENARIOS[0])
    drifted = dict(bound)
    drifted["common_batch_sequence_rows"] = int(drifted["common_batch_sequence_rows"]) + 1
    with pytest.raises(HNCCD.HNCCDRuntimeError, match="row|sequence|coverage|receipt"):
        HNCCD.validate_hnccd_terminal_receipt(drifted)


def test_gradient_audit_receipt_rejects_missing_or_tampered_clean_proof() -> None:
    torch.manual_seed(78)
    model = _BindingModel().train()
    labels, receivers = _batch_cells()
    clean = model.paired_output(torch.randn(128, 160))
    leo = model.paired_output(torch.randn(128, 160))
    loss, _ = _loss(leo["z_id"], HNCCD.resolve_hnccd_classifier_weight(model), labels, receivers, frozen=True)
    audit = HNCCD.hnccd_aux_gradient_audit(
        loss,
        clean["z_id"],
        leo["z_id"],
        HNCCD.hnccd_shared_encoder_and_head_parameters(model),
    )
    tampered = dict(audit)
    tampered.pop("clean_feat_joint")
    with pytest.raises(HNCCD.HNCCDRuntimeError, match="clean|VJP|audit"):
        HNCCD.update_hnccd_gradient_audit_receipt(
            HNCCD.hnccd_config_receipt(HNCCD.HNCCDConfig(True, True, 0.02)),
            tampered,
            scenario=SCENARIOS[0],
        )


def test_amp_scaled_overflow_skips_once_and_raw_nonfinite_fails_closed() -> None:
    torch.manual_seed(79)
    model = _BindingModel().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    scaler = _RecoveringScaler()
    logits = model.paired_output(torch.ones((128, 160)))["tx_logits"]
    loss = logits.sum() * 1.0e33
    before = [parameter.detach().clone() for parameter in model.parameters()]
    overflow = HNCCD.hnccd_scaled_backward_and_classify(
        model=model, optimizer=optimizer, scaler=scaler, loss=loss
    )
    assert overflow["amp_overflow_detected"] is True
    assert overflow["amp_overflow_kind"] == "COMBINED_SCALED_OVERFLOW_RAW_FINITE"
    assert overflow["scaled_backward_count"] == 1 and overflow["optimizer_unscale_count"] == 1
    finalized = HNCCD.finalize_hnccd_amp_overflow_skip(
        optimizer=optimizer, scaler=scaler, overflow=overflow
    )
    assert finalized["optimizer_state_unchanged"] is True
    assert finalized["optimizer_step_applied"] is False
    assert finalized["post_scale"] < finalized["pre_scale"]
    assert all(torch.equal(before_value, parameter.detach()) for before_value, parameter in zip(before, model.parameters()))
    amp_receipt = HNCCD.update_hnccd_amp_overflow_receipt(
        _pending_g_amp_receipt(), overflow=overflow, finalized_skip=finalized
    )
    assert amp_receipt["hnccd_amp_overflow_raw_finite_batches"] == 1
    assert amp_receipt["hnccd_effective_optimizer_steps"] == 0
    assert amp_receipt["hnccd_optimizer_events"][0]["action"] == "AMP_OVERFLOW_SKIP"

    raw_model = _BindingModel().train()
    raw_optimizer = torch.optim.AdamW(raw_model.parameters(), lr=1e-3, weight_decay=0.0)
    raw_logits = raw_model.paired_output(torch.ones((128, 160)))["tx_logits"]
    raw_failure = HNCCD.hnccd_scaled_backward_and_classify(
        model=raw_model,
        optimizer=raw_optimizer,
        scaler=_RecoveringScaler(),
        loss=_FiniteForwardNonfiniteBackward.apply(raw_logits).sum(),
    )
    assert raw_failure["amp_overflow_kind"] == "COMBINED_RAW_NONFINITE_OR_DISCONNECTED"
    with pytest.raises(HNCCD.HNCCDRuntimeError, match="raw-finite|skip"):
        HNCCD.finalize_hnccd_amp_overflow_skip(
            optimizer=raw_optimizer, scaler=_RecoveringScaler(), overflow=raw_failure
        )
    raw_receipt = HNCCD.update_hnccd_amp_overflow_receipt(
        _pending_g_amp_receipt(), overflow=raw_failure
    )
    assert raw_receipt["hnccd_amp_overflow_raw_nonfinite_batches"] == 1
    material = HNCCD.record_hnccd_material_nonfinite_receipt(
        _pending_g_amp_receipt(), reason="total_loss_nonfinite"
    )
    assert material["hnccd_amp_overflow_material_nonfinite_batches"] == 1
    with pytest.raises(HNCCD.HNCCDRuntimeError, match="material loss is non-finite"):
        HNCCD.hnccd_scaled_backward_and_classify(
            model=_BindingModel(),
            optimizer=torch.optim.AdamW(_BindingModel().parameters(), lr=1e-3),
            scaler=_RecoveringScaler(),
            loss=torch.tensor(float("nan"), requires_grad=True),
        )


def test_normal_amp_batch_has_one_backward_and_no_raw_material_vjp(monkeypatch) -> None:
    model = _BindingModel().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    loss = model.paired_output(torch.ones((128, 160)))["tx_logits"].square().mean()

    def _unexpected_raw_vjp(*args, **kwargs):
        raise AssertionError("finite HNCCD batch must not run a raw material VJP")

    monkeypatch.setattr(torch.autograd, "grad", _unexpected_raw_vjp)
    info = HNCCD.hnccd_scaled_backward_and_classify(
        model=model, optimizer=optimizer, scaler=_RecoveringScaler(scale=1.0), loss=loss
    )
    assert info["amp_overflow_detected"] is False
    assert info["scaled_backward_count"] == 1 and info["optimizer_unscale_count"] == 1
    assert inspect.getsource(HNCCD.hnccd_scaled_backward_and_classify).count(".backward(") == 1


@pytest.mark.parametrize("scaled_overflow", [False, True])
def test_retained_graph_roots_release_without_gc_or_second_forward(scaled_overflow: bool) -> None:
    saved_tokens: list[weakref.ReferenceType[_SavedTensorToken]] = []

    def pack(tensor: torch.Tensor) -> _SavedTensorToken:
        token = _SavedTensorToken(tensor)
        saved_tokens.append(weakref.ref(token))
        return token

    def unpack(token: _SavedTensorToken) -> torch.Tensor:
        return token.tensor

    def run_one_batch() -> tuple[dict[str, object], dict[str, object]]:
        model = _BindingModel().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
        scaler = _RecoveringScaler(scale=1e5 if scaled_overflow else 1.0)
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            output = model.paired_output(torch.ones((128, 160)))
            loss = output["tx_logits"].sum() * 1e33 if scaled_overflow else output["tx_logits"].square().mean()
            backward = HNCCD.hnccd_scaled_backward_and_classify(
                model=model, optimizer=optimizer, scaler=scaler, loss=loss
            )
            if scaled_overflow:
                HNCCD.finalize_hnccd_amp_overflow_skip(
                    optimizer=optimizer, scaler=scaler, overflow=backward
                )
            else:
                scaler.step(optimizer)
                scaler.update()
        roots = {"out_l": output, "loss": loss}
        del output, loss
        return roots, backward

    roots, backward = run_one_batch()
    assert saved_tokens and backward["amp_overflow_detected"] is scaled_overflow
    if not scaled_overflow:
        assert any(reference() is not None for reference in saved_tokens)
    HNCCD.release_hnccd_retained_graph_roots(roots)
    assert roots == {} and all(reference() is None for reference in saved_tokens)
    assert "gc.collect" not in inspect.getsource(HNCCD.release_hnccd_retained_graph_roots)


def test_terminal_rejects_persistent_overflow_or_zero_effective_steps() -> None:
    persistent = _build_g_terminal_receipt()
    persistent["hnccd_consecutive_amp_overflow_skips"] = 1
    persistent["hnccd_max_consecutive_amp_overflow_skips"] = 1
    persistent["hnccd_persistent_amp_overflow"] = True
    with pytest.raises(HNCCD.HNCCDRuntimeError, match="AMP|effective"):
        HNCCD.validate_hnccd_terminal_receipt(persistent)
    zero_step = _build_g_terminal_receipt()
    zero_step["hnccd_effective_optimizer_steps"] = 0
    with pytest.raises(HNCCD.HNCCDRuntimeError, match="AMP|effective"):
        HNCCD.validate_hnccd_terminal_receipt(zero_step)


def test_failure_receipt_is_atomic_scalar_and_fail_closed(tmp_path: Path) -> None:
    target = HNCCD.write_hnccd_failure_receipt(
        tmp_path,
        candidate_id="F1G_HNCCD12",
        run_id="phase1_hnccd12_20260811_v1",
        receipt=HNCCD.hnccd_config_receipt(HNCCD.HNCCDConfig(True, True, 0.02)),
        error=HNCCD.HNCCDRuntimeError("exact head Cholesky failure"),
        failure_stage="trainer_initialization",
    )
    assert target == tmp_path / "phase1_hnccd_failure_receipt.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["schema"] == "cvs.phase1.hnccd_failure_receipt.v1"
    assert payload["exception_fingerprint"] == "HNCCD_EXACT_HEAD_CHOLESKY_FAILURE"
    assert payload["candidate_id"] == "F1G_HNCCD12"
    assert "source_receiver_tokens" not in json.dumps(payload["receipt"], ensure_ascii=False)


def test_train_declares_the_real_hnccd_cli_and_failure_path() -> None:
    train = CODE_ROOT / "SSDG" / "train_ssdg.py"
    source = train.read_text(encoding="utf-8")
    for flag in (
        "--phase1_hnccd_frozen_mode",
        "--phase1_hnccd_enabled",
        "--lambda_hnccd",
    ):
        assert flag in source
    assert "_persist_hnccd_failure_receipt" in source
    assert "validate_hnccd_args" in source and "hnccd_config_receipt" in source
    help_result = subprocess.run(
        [sys.executable, str(train), "--help"],
        cwd=str(CODE_ROOT),
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert help_result.returncode == 0, help_result.stderr
    for flag in (
        "--phase1_hnccd_frozen_mode",
        "--phase1_hnccd_enabled",
        "--lambda_hnccd",
    ):
        assert flag in help_result.stdout


def test_launcher_has_exact_hnccd_cg_matrix_and_only_real_cli() -> None:
    launcher_text = LAUNCHER.read_text(encoding="utf-8")
    assert launcher_text.startswith("#!/usr/bin/env bash\n")
    assert 'RUN_ID="${RUN_ID:-phase1_hnccd12_20260811_v1}"' in launcher_text
    assert "--phase1_hnccd_frozen_mode true" in launcher_text
    assert "--phase1_hnccd_enabled false --lambda_hnccd 0" in launcher_text
    assert "--phase1_hnccd_enabled true --lambda_hnccd 0.02" in launcher_text
    assert "--epochs 40 --label_epochs 40 --pseudo_epochs 0" in launcher_text
    calls = re.findall(r"^launch_arm (\d) ([CG]) (\d)$", launcher_text, flags=re.MULTILINE)
    assert len(calls) == 12 and {arm for _, arm, _ in calls} == {"C", "G"}
    assert [gpu for _, _, gpu in calls] == ["0", "0", "1", "1", "2", "2", "3", "3", "4", "5", "6", "7"]
    assert sum(arm == "C" for _, arm, _ in calls) == 6
    assert sum(arm == "G" for _, arm, _ in calls) == 6
    for flag in (
        "ccpc_leo",
        "pamr",
        "cb_sfce",
        "gd_proto_nll",
        "icmt",
        "cagm",
        "rcrmd",
        "rcat",
        "rcmmc",
        "hscf",
        "recte",
        "cp_sfce",
    ):
        assert f"--phase1_{flag}_enabled false" in launcher_text
        assert f"--lambda_{flag} 0" in launcher_text
    relative = f"scripts/{LAUNCHER.name}"
    syntax = subprocess.run(["bash", "-n", relative], cwd=str(CODE_ROOT), text=True, capture_output=True)
    assert syntax.returncode == 0, syntax.stderr
    dry = subprocess.run(["bash", relative, "--dry-run"], cwd=str(CODE_ROOT), text=True, capture_output=True)
    assert dry.returncode == 0, dry.stderr
    lines = [line for line in dry.stdout.splitlines() if "[DRY-RUN]" in line]
    assert len(lines) == 12 and all("phase1_hnccd12_20260811_v1" in line for line in lines)
    assert sum("--phase1_hnccd_enabled false" in line for line in lines) == 6
    assert sum("--phase1_hnccd_enabled true" in line for line in lines) == 6
    assert all("--phase1_rcmmc_enabled false" in line for line in lines)
    assert all("--phase1_hscf_enabled false" in line for line in lines)


def test_train_hnccd_path_is_reachable_g_only_and_terminal_is_post_loop() -> None:
    source = inspect.getsource(train_ssdg.train)
    tree = ast.parse(textwrap.dedent(source))

    def calls(name: str) -> list[ast.Call]:
        return [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
        ]

    required = (
        "validate_hnccd_binding",
        "update_hnccd_common_batch_sequence_receipt",
        "hnccd_loss",
        "add_hnccd_to_loss",
        "update_hnccd_receipt",
        "hnccd_aux_gradient_audit",
        "update_hnccd_gradient_audit_receipt",
        "hnccd_scaled_backward_and_classify",
        "finalize_hnccd_amp_overflow_skip",
        "update_hnccd_amp_overflow_receipt",
        "update_hnccd_optimizer_step_receipt",
        "record_hnccd_material_nonfinite_receipt",
        "update_hnccd_resource_receipt",
        "release_hnccd_retained_graph_roots",
        "validate_hnccd_terminal_receipt",
    )
    line = {}
    for name in required:
        found = calls(name)
        assert found, f"HNCCD train path does not call {name}"
        line[name] = min(node.lineno for node in found)
    assert len(calls("hnccd_loss")) == 1
    assert len(calls("add_hnccd_to_loss")) == 1
    assert len(calls("hnccd_scaled_backward_and_classify")) == 1
    assert len(calls("validate_hnccd_terminal_receipt")) == 1
    assert line["validate_hnccd_binding"] < line["hnccd_loss"] < line["hnccd_aux_gradient_audit"]
    assert line["hnccd_loss"] < line["add_hnccd_to_loss"] < line["hnccd_scaled_backward_and_classify"]
    assert line["update_hnccd_receipt"] < line["update_hnccd_optimizer_step_receipt"]
    assert line["release_hnccd_retained_graph_roots"] < line["validate_hnccd_terminal_receipt"]
    assert "phase1_hnccd_terminal_receipt.json" in source

    root_assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "hnccd_retained_graph_roots" for target in node.targets)
    )
    release_call = next(node for node in calls("release_hnccd_retained_graph_roots"))
    assert root_assignment.lineno < release_call.lineno
    mapped_roots = {
        key.value
        for key in root_assignment.value.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    deleted_roots = {
        name.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Delete) and root_assignment.lineno < node.lineno < release_call.lineno
        for target in node.targets
        for name in ast.walk(target)
        if isinstance(name, ast.Name)
    }
    assert mapped_roots == deleted_roots
    enclosing_loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and node.lineno <= root_assignment.lineno
        and getattr(node, "end_lineno", node.lineno) >= release_call.lineno
    ]
    batch_loop = min(enclosing_loops, key=lambda node: int(node.end_lineno) - int(node.lineno))
    post_release_loads = {
        node.id
        for node in ast.walk(batch_loop)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and release_call.lineno < node.lineno <= int(batch_loop.end_lineno)
    }
    assert not mapped_roots.intersection(post_release_loads)
    assert line["validate_hnccd_terminal_receipt"] > int(batch_loop.end_lineno)

    loss_source = inspect.getsource(HNCCD.hnccd_loss)
    amp_source = inspect.getsource(HNCCD.hnccd_scaled_backward_and_classify)
    release_source = inspect.getsource(HNCCD.release_hnccd_retained_graph_roots)
    assert "query" not in inspect.signature(HNCCD.hnccd_loss).parameters
    assert amp_source.count(".backward(") == 1 and amp_source.count(".unscale_(") == 1
    assert "torch.linalg.cholesky" in inspect.getsource(HNCCD.hnccd_head_null_basis)
    assert "torch.linalg.pinv" not in loss_source
    for forbidden in ("gc.collect", "empty_cache", ".backward(", ".unscale_(", "forward("):
        assert forbidden not in release_source
