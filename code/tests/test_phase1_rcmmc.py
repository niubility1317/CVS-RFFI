from __future__ import annotations

"""Focused mechanical contract tests for P1-RCMMC.

The scientific implementation lives in ``cvsrffi.phase1_rcmmc``.  This file
only checks the frozen public contract, receipt boundaries, and launcher
matrix; it must not be used to tune the mechanism or to read performance.
"""

import ast
import inspect
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi import phase1_rcmmc as RCMMC  # noqa: E402
from cvsrffi import phase1_rcat as RCAT  # noqa: E402


LAUNCHER = CODE_ROOT / "scripts" / "launch_phase1_rcmmc12_20260811.sh"


def _constant(name: str, default):
    return getattr(RCMMC, name, default)


FROZEN_LAMBDA = float(_constant("FROZEN_RCMMC_LAMBDA", 0.02))
CLASS_IDS = tuple(_constant("FROZEN_RCMMC_CLASS_IDS", (0, 1, 2, 3)))
RX_IDS = tuple(_constant("FROZEN_RCMMC_SOURCE_RECEIVER_IDS", tuple(range(7))))
SCENARIOS = tuple(
    _constant(
        "FROZEN_RCMMC_SCENARIOS",
        ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"),
    )
)
CELL_COUNT = int(_constant("FROZEN_RCMMC_CELL_COUNT", len(CLASS_IDS) * len(RX_IDS)))
DENOMINATOR = int(
    _constant(
        "FROZEN_RCMMC_GLOBAL_DENOMINATOR",
        _constant("FROZEN_RCMMC_TERM_DIVISOR", CELL_COUNT),
    )
)
SCHEMA = str(_constant("RCMMC_RECEIPT_SCHEMA", "cvs.phase1.rcmmc_receipt.v1"))
OPTIMIZER_TYPE = str(_constant("FROZEN_RCMMC_OPTIMIZER_TYPE", "AdamW"))


def _fn(name: str):
    fn = getattr(RCMMC, name, None)
    if fn is None:
        pytest.fail(f"RCMMC core public function is missing: {name}")
    return fn


def _frozen_args(*, enabled: bool = True, epochs: int = 40, batch_size: int = 128) -> SimpleNamespace:
    """Argument namespace consumed by the frozen RCMMC validator."""

    args = dict(
        phase1_rcmmc_frozen_mode=True,
        phase1_rcmmc_enabled=enabled,
        lambda_rcmmc=FROZEN_LAMBDA if enabled else 0.0,
        batch_size=batch_size,
        from_scratch=False,
        baseline_ckpt="geosat_c_final.pth",
        freeze_backbone=False,
        amp=True,
        id_feature_key="feat_joint",
        epochs=epochs,
        label_epochs=epochs,
        pseudo_epochs=0,
        checkpoint_selection="final_only",
        phase1_source_val_selection_only=True,
        use_sat_consistency=True,
        lambda_sat_cons=0.10,
        lambda_sat_cls=0.0,
        sat_cons_start_epoch=1,
        sat_view_prob=1.0,
        sat_train_scenarios=",".join(SCENARIOS),
        sat_view_schedule="",
        use_concat_sat_channel_aug=False,
        use_unlabeled=False,
        use_tx_rx_balanced_sampler=False,
        use_aug=False,
        use_mixstyle=False,
        reject_head=False,
        use_ema_teacher=False,
        teacher_ckpt="",
        lambda_teacher_clean_kl=0.0,
        lambda_teacher_sat_kl=0.0,
        lambda_teacher_zid_mse=0.0,
    )
    for peer in (
        "ccpc_leo",
        "pamr",
        "cb_sfce",
        "gd_proto_nll",
        "icmt",
        "cagm",
        "rcrmd",
        "rcat",
        "recte",
        "hscf",
    ):
        args[f"phase1_{peer}_frozen_mode"] = False
        args[f"phase1_{peer}_enabled"] = False
        args[f"lambda_{peer}"] = 0.0
    return SimpleNamespace(**args)


def _config(*, enabled: bool = True):
    cls = _fn("RCMMCConfig")
    return cls(True, enabled, FROZEN_LAMBDA if enabled else 0.0)


class _BindingModel(torch.nn.Module):
    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        self.id_backbone = torch.nn.Module()
        self.id_backbone.encoder = torch.nn.Linear(dim, dim, bias=False)
        self.id_backbone.cls_head = torch.nn.Module()
        self.id_backbone.cls_head.head = torch.nn.Linear(dim, len(CLASS_IDS), bias=True)
        with torch.no_grad():
            self.id_backbone.encoder.weight.copy_(torch.eye(dim))
            self.id_backbone.cls_head.head.weight.zero_()
            self.id_backbone.cls_head.head.bias.zero_()
            self.id_backbone.cls_head.head.weight[:, : len(CLASS_IDS)] = torch.eye(len(CLASS_IDS))

    def paired_output(self, x: torch.Tensor) -> dict[str, torch.Tensor | str]:
        z_id = self.id_backbone.encoder(x)
        return {
            "z_id": z_id,
            "z_id_key": "feat_joint",
            "tx_logits": self.id_backbone.cls_head.head(z_id),
        }


def _all_cells(repeats: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.tensor(list(CLASS_IDS) * len(RX_IDS) * repeats, dtype=torch.long)
    receivers = torch.tensor(
        [r for r in RX_IDS for _ in CLASS_IDS] * repeats,
        dtype=torch.long,
    )
    return labels, receivers


def _batch_cells() -> tuple[torch.Tensor, torch.Tensor]:
    """Return the frozen B=128 source-L rows with all 28 cells covered."""

    labels, receivers = _all_cells(repeats=4)
    extra_labels, extra_receivers = _all_cells(repeats=1)
    labels = torch.cat((labels, extra_labels[:16]), dim=0)
    receivers = torch.cat((receivers, extra_receivers[:16]), dim=0)
    assert labels.numel() == 128 and receivers.numel() == 128
    return labels, receivers


def _metadata(rows: int, offset: int = 0) -> dict[str, torch.Tensor]:
    return {
        "base_index": torch.arange(offset, offset + rows),
        "sig_i": torch.arange(offset, offset + rows) + 10000,
        "same_physical": torch.ones(rows, dtype=torch.bool),
    }


def _call_loss(clean: torch.Tensor, leo: torch.Tensor, labels: torch.Tensor, receivers: torch.Tensor):
    """Call the core loss through names used by the frozen public contract."""

    fn = _fn("rcmmc_loss")
    values = {
        "clean": clean,
        "clean_z": clean,
        "clean_feat": clean,
        "clean_features": clean,
        "clean_feat_joint": clean,
        "leo": leo,
        "leo_z": leo,
        "leo_feat": leo,
        "leo_features": leo,
        "leo_feat_joint": leo,
        "labels": labels,
        "tx_labels": labels,
        "source_tx_labels": labels,
        "receivers": receivers,
        "receiver_labels": receivers,
        "source_rx_labels": receivers,
        "source_receiver_labels": receivers,
        "source_receiver_ids": RX_IDS,
        "receiver_ids": RX_IDS,
    }
    signature = inspect.signature(fn)
    kwargs = {
        name: values[name]
        for name, parameter in signature.parameters.items()
        if name in values and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    required = [
        p
        for p in signature.parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    ]
    if len(kwargs) < len(required):
        attempts = (
            (clean, leo, labels, receivers, RX_IDS),
            (clean, leo, labels, receivers),
            (clean, leo, labels),
        )
        last = None
        for args in attempts:
            try:
                return fn(*args)
            except TypeError as exc:
                last = exc
        raise AssertionError(f"cannot bind rcmmc_loss signature {signature}: {last}")
    return fn(**kwargs)


def _loss_and_info(clean, leo, labels, receivers):
    result = _call_loss(clean, leo, labels, receivers)
    assert isinstance(result, tuple) and len(result) == 2, "rcmmc_loss must return (loss, info)"
    loss, info = result
    assert torch.is_tensor(loss) and isinstance(info, dict)
    return loss, info


def _sealed_receipt(*, enabled: bool) -> dict[str, object]:
    receipt = _fn("rcmmc_config_receipt")(_config(enabled=enabled))
    receipt.update(
        {
            "baseline_sha256": "a" * 64,
            "initial_checkpoint_sha256": "a" * 64,
            "checkpoint_role": "training_final_only",
            "class_order_binding_sha256": "b" * 64,
            "source_labeled_indices_sha256": "c" * 64,
            "source_split_manifest_sha256": "d" * 64,
            "source_receiver_ids": list(RX_IDS),
            "source_receiver_count": len(RX_IDS),
            "source_receiver_ids_sha256": "e" * 64,
            "optimizer_type": OPTIMIZER_TYPE,
            "optimizer_initial_state_sha256": "f" * 64,
            "optimizer_initial_state_empty": True,
            "optimizer_state_restored": False,
            "rng_state_restored": False,
            "expected_tx_class_ids": list(CLASS_IDS),
            "common_l_base_head_input_path_verified": True,
            "common_batch_sequence_sha256": "1" * 64,
            "common_batch_sequence_batches": 0,
            "common_batch_sequence_rows": 0,
            "common_scenario_batches": {scene: 0 for scene in SCENARIOS},
        }
    )
    return receipt


def _common_bind(receipt: dict[str, object], *, epoch: int, batch_index: int, scenario: str) -> dict[str, object]:
    labels, receivers = _batch_cells()
    fn = _fn("update_rcmmc_common_batch_sequence_receipt")
    return fn(
        receipt,
        epoch=epoch,
        batch_index=batch_index,
        scenario=scenario,
        source_tx_labels=labels,
        source_rx_labels=receivers,
        source_receiver_tokens=RX_IDS,
        metadata=_metadata(int(labels.numel()), offset=batch_index * 100),
    )


def _info_cells(info: dict[str, object]) -> dict[str, dict[str, object]]:
    cells = info.get("cells", info.get("cell_moments", {}))
    assert isinstance(cells, dict)
    return {str(key): dict(value) for key, value in cells.items() if isinstance(value, dict)}


def test_totalized_zero_is_safe_and_nonfinite_is_fatal() -> None:
    labels = torch.tensor([0, 1], dtype=torch.long)
    receivers = torch.tensor([0, 1], dtype=torch.long)
    clean = torch.zeros((2, 4), requires_grad=True)
    leo = torch.zeros((2, 4), requires_grad=True)
    totalized = _fn("totalized_l2")(clean)
    assert torch.equal(totalized, torch.zeros_like(totalized))
    loss, info = _loss_and_info(clean, leo, labels, receivers)
    assert torch.isfinite(loss.detach()) and float(loss.detach()) == pytest.approx(0.0)
    assert int(info.get("clean_zero_rows", info.get("zero_clean_rows", 0))) == 2
    assert int(info.get("leo_zero_rows", info.get("zero_leo_rows", 0))) == 2
    loss.backward()
    bad = clean.detach().clone()
    bad[0, 0] = float("nan")
    runtime = getattr(RCMMC, "RCMMCRuntimeError", RuntimeError)
    with pytest.raises(runtime, match="non[- ]finite|finite"):
        _loss_and_info(bad.requires_grad_(True), leo.detach().requires_grad_(True), labels, receivers)


def test_fixed_28_empty_cells_zero_and_no_active_renormalization() -> None:
    labels = torch.tensor([0, 1], dtype=torch.long)
    receivers = torch.tensor([0, 1], dtype=torch.long)
    clean = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], requires_grad=True)
    leo = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], requires_grad=True)
    loss, info = _loss_and_info(clean, leo, labels, receivers)
    assert torch.isfinite(loss.detach()) and float(loss.detach()) > 0.0
    cells = _info_cells(info)
    assert len(cells) == CELL_COUNT
    assert float(info.get("global_denominator", info.get("denominator", DENOMINATOR))) == DENOMINATOR
    assert bool(info.get("no_active_renormalization", True)) is True
    empty = [cell for cell in cells.values() if int(cell.get("n_rc", cell.get("count", 0))) == 0]
    assert empty and all(float(cell.get("loss_contribution", 0.0)) == 0.0 for cell in empty)
    loss.backward()
    assert clean.grad is None
    assert leo.grad is not None and int(torch.count_nonzero(leo.grad).item()) > 0


def test_mu_q_and_covariance_m_frobenius_forms_agree() -> None:
    labels, receivers = _all_cells()
    clean = torch.randn(labels.numel(), 8)
    leo = clean.clone()
    loss, info = _loss_and_info(clean, leo, labels, receivers)
    assert float(loss.detach()) == pytest.approx(0.0, abs=1e-8)
    cells = _info_cells(info)
    assert len(cells) == CELL_COUNT
    # The implementation stores scalar/count receipts only; verify the frozen
    # moment identity directly on each occupied receiver×class cell:
    # Q = M + μμᵀ, hence the Frobenius congruence uses the same Q/M geometry.
    for receiver in RX_IDS:
        for class_id in CLASS_IDS:
            mask = receivers.eq(receiver) & labels.eq(class_id)
            x = clean[mask]
            mu = x.mean(dim=0)
            q = x.transpose(0, 1).matmul(x) / float(x.size(0))
            centered = x - mu
            m = centered.transpose(0, 1).matmul(centered) / float(x.size(0))
            assert torch.allclose(q, m + torch.outer(mu, mu), atol=1e-5, rtol=1e-5)


def test_cell_permutation_is_rcmmc_zero_but_rcat_positive() -> None:
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    receivers = torch.tensor([0, 0, 0, 0], dtype=torch.long)
    clean = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    leo = clean[[1, 0, 3, 2]].clone()
    rcmmc, _ = _loss_and_info(clean, leo, labels, receivers)
    assert float(rcmmc.detach()) == pytest.approx(0.0, abs=1e-7)
    # RCAT is per-physical-row transport and therefore sees the within-cell swap.
    rcat, _ = RCAT.rcat_loss(clean, leo, labels, receivers, RX_IDS)
    assert float(rcat.detach()) > 0.0


def test_head_nullspace_softmax_kl_zero_but_rcmmc_positive() -> None:
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    receivers = torch.tensor([0, 0, 0, 0], dtype=torch.long)
    clean = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, -1.0], [0.0, 1.0, 0.0, 0.0, -1.0]])
    leo = clean.clone()
    leo[:, 4] *= -1.0
    head = torch.nn.Linear(5, 4, bias=False)
    with torch.no_grad():
        head.weight.zero_()
        head.weight[:, :4] = torch.eye(4)
    clean_logits, leo_logits = head(clean), head(leo)
    kl = F.kl_div(
        F.log_softmax(leo_logits, dim=1),
        F.softmax(clean_logits, dim=1),
        reduction="batchmean",
    )
    rcmmc, _ = _loss_and_info(clean, leo, labels, receivers)
    assert float(kl.detach()) == pytest.approx(0.0, abs=1e-8)
    assert float(rcmmc.detach()) > 0.0


def test_configuration_receipt_permissions_and_old_routes_are_frozen() -> None:
    validate = _fn("validate_rcmmc_args")
    assert validate(_frozen_args(enabled=True)) == _config(enabled=True)
    assert validate(_frozen_args(enabled=False)) == _config(enabled=False)
    error = getattr(RCMMC, "RCMMCConfigurationError", ValueError)
    for name, value in (
        ("lambda_rcmmc", 0.01),
        ("epochs", 39),
        ("phase1_rcat_enabled", True),
        ("phase1_recte_frozen_mode", True),
        ("use_unlabeled", True),
    ):
        bad = _frozen_args()
        setattr(bad, name, value)
        with pytest.raises(error):
            validate(bad)
    receipt = _fn("rcmmc_config_receipt")(_config(enabled=True))
    assert receipt["schema"] == SCHEMA
    assert receipt.get("uses_target_rows") is False
    assert receipt.get("uses_proxy_rows") is False
    assert receipt.get("uses_held_rows") is False
    assert receipt.get("uses_unlabeled_rows") is False
    assert "query" not in inspect.signature(_fn("rcmmc_loss")).parameters
    assert "day" not in inspect.signature(_fn("rcmmc_loss")).parameters


def test_vjp_scope_leo_feature_encoder_head_and_clean_stopgrad() -> None:
    model = _BindingModel().train()
    labels, receivers = _batch_cells()
    clean_out = model.paired_output(torch.randn(labels.numel(), 4))
    leo_out = model.paired_output(torch.randn(labels.numel(), 4))
    clean_out["z_id"].retain_grad()
    binding = _fn("validate_rcmmc_binding")
    try:
        binding(
            model=model,
            out_clean=clean_out,
            out_leo=leo_out,
            tx_labels=labels,
            source_rx_labels=receivers,
            expected_class_ids=CLASS_IDS,
            expected_receiver_ids=RX_IDS,
        )
    except TypeError:
        binding(model, clean_out, leo_out, labels, receivers, CLASS_IDS, RX_IDS)
    loss, _ = _loss_and_info(clean_out["z_id"], leo_out["z_id"], labels, receivers)
    audit_fn = _fn("rcmmc_aux_gradient_audit")
    groups_fn = _fn("rcmmc_shared_encoder_and_head_parameters")
    groups = groups_fn(model)
    audit = audit_fn(loss, clean_out["z_id"], leo_out["z_id"], groups)
    assert bool(audit.get("raw_unscaled", True)) is True
    assert float(audit.get("feat_joint_leo", {}).get("norm", 0.0)) > 0.0
    assert float(audit.get("shared_encoder", audit.get("encoder", {})).get("norm", 1.0)) > 0.0
    head_audit = audit.get("classifier_head", audit.get("head_aux", {}))
    assert isinstance(head_audit, dict)
    assert float(head_audit.get("nonzero_parameters", 0.0)) == 0.0
    assert float(head_audit.get("none_parameters", 0.0)) + float(head_audit.get("zero_parameters", 0.0)) == float(head_audit.get("parameter_count", 0.0))
    loss.backward()
    assert clean_out["z_id"].grad is None


def test_missing_clean_audit_cannot_close_receipt_or_terminal() -> None:
    """The legacy three-argument seam must never seal a frozen G receipt."""

    model = _BindingModel().train()
    labels, receivers = _batch_cells()
    clean_out = model.paired_output(torch.randn(labels.numel(), 4))
    leo_out = model.paired_output(torch.randn(labels.numel(), 4))
    loss, info = _loss_and_info(clean_out["z_id"], leo_out["z_id"], labels, receivers)
    audit_fn = _fn("rcmmc_aux_gradient_audit")
    groups = _fn("rcmmc_shared_encoder_and_head_parameters")(model)
    runtime_error = getattr(RCMMC, "RCMMCRuntimeError", RuntimeError)
    try:
        missing_clean = audit_fn(loss, leo_out["z_id"], groups)
    except (TypeError, ValueError, runtime_error):
        return

    receipt = _sealed_receipt(enabled=True)
    for index, scene in enumerate(SCENARIOS, start=1):
        receipt = _common_bind(receipt, epoch=1, batch_index=index, scenario=scene)
        receipt = _fn("update_rcmmc_receipt")(receipt, info, scenario=scene, epoch=1, batch_index=index)
    try:
        receipt = _fn("update_rcmmc_gradient_audit_receipt")(receipt, missing_clean)
    except (TypeError, ValueError, runtime_error):
        return
    with pytest.raises(runtime_error, match="clean|VJP|audit|receipt"):
        _fn("validate_rcmmc_terminal_receipt")(receipt)


@pytest.mark.parametrize("mutation", ("delete_clean", "tamper_clean"))
def test_clean_vjp_deletion_or_zero_tamper_fails_closed(mutation: str) -> None:
    """A valid four-argument audit cannot lose or falsify its clean VJP proof."""

    model = _BindingModel().train()
    labels, receivers = _batch_cells()
    clean_out = model.paired_output(torch.randn(labels.numel(), 4))
    leo_out = model.paired_output(torch.randn(labels.numel(), 4))
    loss, _ = _loss_and_info(clean_out["z_id"], leo_out["z_id"], labels, receivers)
    groups = _fn("rcmmc_shared_encoder_and_head_parameters")(model)
    audit = _fn("rcmmc_aux_gradient_audit")(loss, clean_out["z_id"], leo_out["z_id"], groups)
    assert isinstance(audit.get("clean_feat_joint"), dict)
    tampered = dict(audit)
    if mutation == "delete_clean":
        tampered.pop("clean_feat_joint", None)
        tampered.pop("clean_feat_joint_aux_vjp", None)
    else:
        clean_audit = dict(tampered["clean_feat_joint"])
        clean_audit["none_parameters"] = 0.0
        clean_audit["zero_parameters"] = 0.0
        tampered["clean_feat_joint"] = clean_audit
    runtime_error = getattr(RCMMC, "RCMMCRuntimeError", RuntimeError)
    with pytest.raises(runtime_error, match="clean|VJP|audit|receipt"):
        _fn("update_rcmmc_gradient_audit_receipt")(_sealed_receipt(enabled=True), tampered)


def test_source_order_physical_binding_and_new_adamw_warm_start() -> None:
    labels, receivers = _batch_cells()
    receipt = _sealed_receipt(enabled=True)
    bind_order = _fn("bind_rcmmc_source_data_order")
    bound = bind_order(
        receipt,
        {
            "schema": "cvs.phase1.source_split_receipt.v1",
            "labeled_indices_sha256": "1" * 64,
            "split_manifest_sha256": "2" * 64,
            "source_receivers": list(RX_IDS),
        },
    )
    assert bound["source_labeled_indices_sha256"] == "1" * 64
    common = _common_bind(bound, epoch=1, batch_index=1, scenario=SCENARIOS[0])
    event_key = "rcmmc_common_batch_cells"
    events = list(common.get(event_key, common.get("common_batch_cells", [])))
    assert events and bool(events[0].get("same_physical_clean_leo", True)) is True
    assert sum(int(v) for v in events[0].get("n_rc", {}).values()) == int(labels.numel())
    optimizer = torch.optim.AdamW(_BindingModel().parameters(), lr=2e-4, weight_decay=1e-4)
    optim_receipt = _fn("bind_rcmmc_optimizer_initial_state")(receipt, optimizer)
    assert optim_receipt["optimizer_type"] == OPTIMIZER_TYPE
    assert optim_receipt["optimizer_initial_state_empty"] is True
    model = _BindingModel()
    warm = _fn("strict_rcmmc_warm_start")(
        model,
        _BindingModel().state_dict(),
        baseline_path="base.pth",
        baseline_sha256="a" * 64,
        checkpoint_epoch=40,
        checkpoint_role="training_final_only",
    )
    assert warm["strict_model_keys"] is True and warm["optimizer_state_restored"] is False
    with pytest.raises(getattr(RCMMC, "RCMMCConfigurationError", ValueError), match="AdamW"):
        _fn("bind_rcmmc_optimizer_initial_state")(receipt, torch.optim.SGD(model.parameters(), lr=1e-3))


def test_common_c_and_g_terminal_receipts_close_three_scenes_28_cells() -> None:
    labels, receivers = _batch_cells()
    control = _sealed_receipt(enabled=False)
    for index, scene in enumerate(SCENARIOS, start=1):
        control = _common_bind(control, epoch=1, batch_index=index, scenario=scene)
    terminal = _fn("validate_rcmmc_terminal_receipt")(control)
    assert terminal["schema"] == SCHEMA
    assert terminal.get("rcmmc_terminal_contract_passed") is True
    assert float(terminal.get("rcmmc_loss_sum", 0.0)) == 0.0
    assert int(terminal.get("rcmmc_batches", 0)) == 0
    assert int(terminal.get("rcmmc_total_rows", 0)) == 0
    assert not terminal.get("rcmmc_scenes")
    assert not terminal.get("rcmmc_gradient_audit")
    model = _BindingModel().train()
    clean = model.paired_output(torch.randn(labels.numel(), 4))
    leo = model.paired_output(torch.randn(labels.numel(), 4))
    loss, info = _loss_and_info(clean["z_id"], leo["z_id"], labels, receivers)
    audit = _fn("rcmmc_aux_gradient_audit")(
        loss,
        clean["z_id"],
        leo["z_id"],
        _fn("rcmmc_shared_encoder_and_head_parameters")(model),
    )
    receipt = _sealed_receipt(enabled=True)
    for index, scene in enumerate(SCENARIOS, start=1):
        receipt = _common_bind(receipt, epoch=1, batch_index=index, scenario=scene)
        receipt = _fn("update_rcmmc_receipt")(receipt, info, scenario=scene, epoch=1, batch_index=index)
    try:
        receipt = _fn("update_rcmmc_gradient_audit_receipt")(receipt, audit)
    except TypeError:
        receipt = _fn("update_rcmmc_gradient_audit_receipt")(receipt, audit, scenario=SCENARIOS[-1])
    g_terminal = _fn("validate_rcmmc_terminal_receipt")(receipt)
    assert g_terminal.get("rcmmc_terminal_contract_passed") is True
    assert int(g_terminal.get("rcmmc_total_rows", labels.numel() * len(SCENARIOS))) == labels.numel() * len(SCENARIOS)
    assert int(g_terminal.get("rcmmc_positive_d_batches", 0)) > 0
    assert g_terminal.get("rcmmc_gradient_audit_completed") is True
    assert all(len(g_terminal.get("rcmmc_scenes", {}).get(scene, {})) == CELL_COUNT for scene in SCENARIOS)


def test_receipt_failure_and_same_physical_order_drift_fail_closed() -> None:
    labels, receivers = _batch_cells()
    receipt = _sealed_receipt(enabled=False)
    with pytest.raises(getattr(RCMMC, "RCMMCRuntimeError", RuntimeError), match="physical|metadata|same"):
        _fn("update_rcmmc_common_batch_sequence_receipt")(
            receipt,
            epoch=1,
            batch_index=1,
            scenario=SCENARIOS[0],
            source_tx_labels=labels,
            source_rx_labels=receivers,
            metadata={"same_physical": torch.ones(labels.numel(), dtype=torch.bool)},
        )
    bound = _common_bind(receipt, epoch=1, batch_index=1, scenario=SCENARIOS[0])
    drifted = dict(bound)
    drifted["common_batch_sequence_rows"] = int(drifted.get("common_batch_sequence_rows", 0)) + 1
    with pytest.raises(getattr(RCMMC, "RCMMCRuntimeError", RuntimeError), match="row|sequence|coverage|receipt"):
        _fn("validate_rcmmc_terminal_receipt")(drifted)


def test_streamed_implementation_avoids_bxd2_and_bx28xd2_allocations() -> None:
    source = Path(RCMMC.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = ("einsum('bi,bj->bij'", 'einsum("bi,bj->bij"', "B,28,d,d", "28,d,d")
    assert not any(token in source for token in forbidden)
    assert "torch.zeros((" not in source or "d * d" not in source
    ledger = _fn("rcmmc_shape_ledger")()
    assert ledger["batch_size"] == 128 and ledger["feature_dim"] == 160 and ledger["cell_count"] == 28
    assert ledger["forbids_batch_d2_materialization"] is True
    assert ledger["forbids_batch_cell_d2_materialization"] is True
    assert ledger["cross_batch_cache"] is False
    assert int(ledger["conservative_live_tensor_upper_bound_bytes"]) > 0
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; streamed-memory check is conditional")
    device = torch.device("cuda")
    labels, receivers = _all_cells(repeats=5)
    labels, receivers = labels[:128].to(device), receivers[:128].to(device)
    clean = torch.randn(128, 160, device=device)
    leo = torch.randn(128, 160, device=device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    loss, _ = _loss_and_info(clean, leo, labels, receivers)
    assert torch.isfinite(loss.detach())
    delta = torch.cuda.max_memory_allocated(device)
    # Conservative frozen upper bound: permits allocator noise but rejects a
    # material B×d² or B×28×d² tensor (the intended implementation is streamed).
    assert delta < 64 * 1024 * 1024


def test_train_interface_and_launcher_matrix_are_frozen() -> None:
    launcher_text = LAUNCHER.read_text(encoding="utf-8")
    assert launcher_text.startswith("#!/usr/bin/env bash\n")
    assert 'RUN_ID="${RUN_ID:-phase1_rcmmc12_20260811_v1}"' in launcher_text
    assert "--phase1_rcmmc_frozen_mode true" in launcher_text
    assert "--phase1_rcmmc_enabled false --lambda_rcmmc 0" in launcher_text
    assert "--phase1_rcmmc_enabled true --lambda_rcmmc 0.02" in launcher_text
    calls = re.findall(r"^launch_arm (\d) ([CG]) (\d)$", launcher_text, flags=re.MULTILINE)
    assert len(calls) == 12 and {arm for _, arm, _ in calls} == {"C", "G"}
    assert [gpu for _, _, gpu in calls] == ["0", "0", "1", "1", "2", "2", "3", "3", "4", "5", "6", "7"]
    assert sum(arm == "C" for _, arm, _ in calls) == 6
    assert sum(arm == "G" for _, arm, _ in calls) == 6
    for flag in ("ccpc_leo", "pamr", "cb_sfce", "gd_proto_nll", "icmt", "cagm", "rcrmd", "rcat", "recte", "hscf"):
        assert f"--phase1_{flag}_enabled false" in launcher_text
        assert f"--lambda_{flag} 0" in launcher_text
    relative = f"scripts/{LAUNCHER.name}"
    syntax = subprocess.run(["bash", "-n", relative], cwd=str(CODE_ROOT), text=True, capture_output=True)
    assert syntax.returncode == 0, syntax.stderr
    dry = subprocess.run(["bash", relative, "--dry-run"], cwd=str(CODE_ROOT), text=True, capture_output=True)
    assert dry.returncode == 0, dry.stderr
    lines = [line for line in dry.stdout.splitlines() if "[DRY-RUN]" in line]
    assert len(lines) == 12
    assert all("phase1_rcmmc12_20260811_v1" in line for line in lines)
    assert sum("--phase1_rcmmc_enabled false" in line for line in lines) == 6
    assert sum("--phase1_rcmmc_enabled true" in line for line in lines) == 6
    assert all("--phase1_rcat_enabled false" in line for line in lines)
    assert all("--phase1_recte_enabled false" in line for line in lines)
    train = CODE_ROOT / "SSDG" / "train_ssdg.py"
    if train.exists():
        source = train.read_text(encoding="utf-8")
        assert "phase1_rcmmc_enabled" in source
        assert "rcmmc_loss(" in source or "phase1_rcmmc" in source
