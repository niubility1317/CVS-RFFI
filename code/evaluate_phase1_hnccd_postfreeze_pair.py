#!/usr/bin/env python
"""Final-only, source-paired P1-HNCCD postfreeze closure.

This facade owns HNCCD identity and reopens current raw C/G HNCCD receipts.
The signed generic evaluator is used only for unchanged float64
totalized-L2/diagonal-Gaussian scoring, fixed-proxy raw-logit recomputation,
strict non-compensating floors, and F6 raw-artifact recomputation.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

from cvsrffi import phase1_hnccd as _hnccd
import export_phase1_hnccd_features as _hnccd_export
import export_phase1_hnccd_leo_features as _hnccd_leo
import evaluate_phase1_icmt_postfreeze_pair as _icmt


EXPECTED_TRAINING_RUN_LEAF = "phase1_hnccd12_20260811_v1"
EXPECTED_POSTFREEZE_MATRIX_ID = "phase1_hnccd_postfreeze_20260811_v1"
EXPECTED_LV_EXPORT_SCHEMA = "cvs.phase1.hnccd_lv_export.v1"
EXPECTED_LEO_BINDING_SCHEMA = "cvs.phase1.hnccd_leo_binding.v1"
EXPECTED_PAIR_SCHEMA = "cvs.phase1.hnccd_postfreeze_pair.v1"
EXPECTED_HNCCD_RECEIPT_SCHEMA = "cvs.phase1.hnccd_receipt.v1"
EXPECTED_HEAD_CONTRACT = "dual_cvsincnet_tx_logits_v1"
EXPECTED_CLEAN_ARTIFACT = "icmt_clean_l_v_proxy_final_only.npz"
HNCCD_FLOOR_DELTA_LIMIT_PP = -2.0
LEGACY_IDENTITY_PREFIXES = ("icmt_", "hscf_", "rcmmc_", "rcat_", "recte_", "rcrmd_", "cagm_")
RAW_RECEIVER_TOKEN_FIELDS = frozenset({"source_receiver_ids", "frozen_source_receiver_ids"})

COMMON_TRAINING_BINDING_FIELDS = (
    "loss_rule", "loss_formula", "loss_global_denominator", "local_class_count",
    "fixed_batch_size", "fixed_local_class_count", "frozen_batch_size", "frozen_feature_dim",
    "frozen_source_receiver_count", "exact_head_weight_path", "exact_head_weight_shape",
    "head_null_basis_rule", "head_full_row_rank_required", "local_class_ids", "z_id_key",
    "training_accumulation_dtype", "clean_feature_detached", "same_physical_pairing",
    "receipt_payload", "common_lambda_sat_cons", "common_sat_kl", "head_input_path",
    "common_l_base_head_input_path_verified", "aux_gradient_scope", "uses_new_forward",
    "uses_resampling", "uses_rx_labels", "uses_day_labels", "uses_domain_labels",
    "uses_target_rows", "uses_proxy_rows", "uses_held_rows", "uses_unlabeled_rows",
    "uses_ema_or_state", "uses_threshold", "uses_cross_sample_pairing",
    "uses_cross_receiver_pairing", "resource_selection_feedback", "warm_start_mode",
    "baseline_sha256", "initial_checkpoint_sha256", "checkpoint_epoch", "checkpoint_role",
    "strict_model_keys", "missing_model_keys", "unexpected_model_keys", "optimizer_state_restored",
    "rng_state_restored", "optimizer_type", "optimizer_initial_state_sha256",
    "optimizer_initial_state_empty", "amp_contract", "source_partition_sha256",
    "source_labeled_indices_sha256", "source_split_manifest_sha256", "source_receiver_count",
    "source_receiver_order_sha256", "source_receiver_ids_sha256", "source_receiver_provenance",
    "dataset_tx_class_order", "local_tx_class_order", "checkpoint_train_tx_class_order",
    "local_to_dataset_class_ids", "local_to_head_class_ids", "expected_tx_class_ids",
    "dataset_class_count", "local_data_class_count", "checkpoint_head_class_count",
    "live_head_class_count", "class_order_binding_sha256", "common_batch_sequence_sha256",
    "common_batch_sequence_batches", "common_batch_sequence_rows", "common_scenario_batches",
    "hnccd_common_cells", "hnccd_common_batch_cells",
)
COMMON_TRAINING_SHA_FIELDS = (
    "baseline_sha256", "initial_checkpoint_sha256", "source_partition_sha256",
    "source_labeled_indices_sha256", "source_split_manifest_sha256",
    "source_receiver_order_sha256", "source_receiver_ids_sha256",
    "class_order_binding_sha256", "optimizer_initial_state_sha256", "common_batch_sequence_sha256",
)
FROZEN_POSTFREEZE_CONTRACT = {
    "HNCCD-PF-01": "final-only L-only feat_joint diagonal Gaussian; V/proxy/U zero fit",
    "HNCCD-PF-02": "float64 totalized-L2 masks before divide, preserves exact zeros and rejects nonfinite values",
    "HNCCD-PF-03": "ddof1 class-equal pooled 0.9/0.1 shrink, 1e-6 floor, full NLL and stable u",
    "HNCCD-PF-04": "current raw HNCCD receipts prove B128/d160/local4/source-Rs SHA-count/fixed28/C-G fairness",
    "HNCCD-PF-05": "source-only single-LEO physical TX/RX/day/class/order plus fixed400 proxy JSON/CSV/NPZ current-SHA closure",
    "HNCCD-PF-06": "clean6/LEO18 four floors, fold/global overall, AUROC and u-gap are strict non-compensating gates",
    "HNCCD-PF-07": "F6 reopens F1--F5 sealed raw artifacts and current receipts rather than trusting pair summaries",
}


class HNCCDPostfreezePairError(RuntimeError):
    """Raised when HNCCD postfreeze evidence cannot close fail-closed."""


def _translate(error: BaseException) -> HNCCDPostfreezePairError:
    return HNCCDPostfreezePairError(str(error))


def _require_no_legacy_identity_fields(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, raw_value in value.items():
            key = str(raw_key).lower()
            if key in RAW_RECEIVER_TOKEN_FIELDS:
                raise HNCCDPostfreezePairError(f"raw source receiver token leaked into {label}: {raw_key}")
            if key.startswith(LEGACY_IDENTITY_PREFIXES):
                raise HNCCDPostfreezePairError(f"historical method identity leaked into {label}: {raw_key}")
            _require_no_legacy_identity_fields(raw_value, label=label)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _require_no_legacy_identity_fields(item, label=label)


def _strip_legacy_identity_fields(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key).lower()
            if key in RAW_RECEIVER_TOKEN_FIELDS:
                raise HNCCDPostfreezePairError(f"raw source receiver token leaked into HNCCD pair output: {raw_key}")
            if key.startswith(LEGACY_IDENTITY_PREFIXES):
                continue
            result[str(raw_key)] = _strip_legacy_identity_fields(item)
        return result
    if isinstance(value, list):
        return [_strip_legacy_identity_fields(item) for item in value]
    if isinstance(value, tuple):
        return [_strip_legacy_identity_fields(item) for item in value]
    return value


_LOGITS_REJECT_MODULE: Any | None = None


def _logits_reject_module() -> Any:
    global _LOGITS_REJECT_MODULE
    if _LOGITS_REJECT_MODULE is not None:
        return _LOGITS_REJECT_MODULE
    source = Path(__file__).resolve().parent / "scripts" / "eval_phase1_logits_open_set_reject.py"
    spec = importlib.util.spec_from_file_location("_hnccd_frozen_logits_proxy", source)
    if spec is None or spec.loader is None:
        raise HNCCDPostfreezePairError("cannot load frozen logits proxy scorer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _LOGITS_REJECT_MODULE = module
    return module


def _close_proxy_value(actual: Any, expected: Any, *, field: str, label: str) -> None:
    if actual is None or expected is None:
        if actual is not expected:
            raise HNCCDPostfreezePairError(f"{label} proxy {field} does not match raw logits")
        return
    try:
        left, right = float(actual), float(expected)
    except (TypeError, ValueError) as exc:
        raise HNCCDPostfreezePairError(f"{label} proxy {field} is not numeric") from exc
    if not math.isfinite(left) or not math.isfinite(right) or not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12):
        raise HNCCDPostfreezePairError(f"{label} proxy {field} does not match raw logits")


def _validate_proxy_logits_recompute(
    *,
    clean_npz: str | Path,
    proxy_metrics_json: str | Path,
    proxy_scores_csv: str | Path,
    source_tx_ids: Sequence[str],
    label: str,
) -> dict[str, Any]:
    """Recompute fixed proxy JSON/CSV from current clean NPZ bytes only."""

    try:
        observed = json.loads(Path(proxy_metrics_json).read_text(encoding="utf-8"))
    except Exception as exc:
        raise HNCCDPostfreezePairError(f"{label} proxy diagnostic JSON is invalid") from exc
    if not isinstance(observed, Mapping):
        raise HNCCDPostfreezePairError(f"{label} proxy diagnostic JSON must encode an object")
    with tempfile.TemporaryDirectory(prefix="hnccd_proxy_recompute_") as temporary:
        expected_csv = Path(temporary) / "scores.csv"
        recomputed = _logits_reject_module().evaluate(
            argparse.Namespace(
                feature_npz=str(Path(clean_npz).resolve()),
                source_tx_ids=",".join(str(item) for item in source_tx_ids),
                unknown_tx_ids="",
                known_query_roles="source_validation_known",
                unknown_query_roles="proxy_unknown",
                calibration_roles="source_validation_known",
                conf_quantile=0.05,
                margin_quantile=0.05,
                energy_quantile=0.95,
                disable_conf_gate=False,
                disable_margin_gate=False,
                disable_energy_gate=False,
                unknown_far_target=0.05,
                output_json="",
                score_table_csv=str(expected_csv),
            )
        )
        for field in (
            "AUROC_unknown", "unknown_FAR", "unknown_reject_rate", "known_closed_accuracy_no_reject",
            "known_coverage", "known_full_accuracy_after_reject", "known_accepted_accuracy", "old_retention_vs_closed",
        ):
            _close_proxy_value(observed.get(field), recomputed.get(field), field=field, label=label)
        for field in ("known_query_count", "unknown_query_count"):
            if type(observed.get(field)) is not int or observed.get(field) != recomputed.get(field):
                raise HNCCDPostfreezePairError(f"{label} proxy {field} does not match raw logits")
        if observed.get("gate_policy") != recomputed.get("gate_policy"):
            raise HNCCDPostfreezePairError(f"{label} proxy gate policy does not match raw logits")
        left_calibration, right_calibration = observed.get("calibration"), recomputed.get("calibration")
        if not isinstance(left_calibration, Mapping) or not isinstance(right_calibration, Mapping) or set(left_calibration) != set(right_calibration):
            raise HNCCDPostfreezePairError(f"{label} proxy calibration is malformed")
        for field in sorted(right_calibration):
            _close_proxy_value(left_calibration.get(field), right_calibration.get(field), field=f"calibration.{field}", label=label)
        try:
            with Path(proxy_scores_csv).open("r", encoding="utf-8", newline="") as handle:
                observed_rows = list(csv.DictReader(handle))
            with expected_csv.open("r", encoding="utf-8", newline="") as handle:
                expected_rows = list(csv.DictReader(handle))
        except Exception as exc:
            raise HNCCDPostfreezePairError(f"{label} proxy score CSV is invalid") from exc
        if observed_rows != expected_rows:
            raise HNCCDPostfreezePairError(f"{label} proxy score CSV does not match raw logits")
    return {
        "passed": True,
        "clean_npz_sha256": _icmt._cb._sha256_file(clean_npz),
        "proxy_metrics_json_sha256": _icmt._cb._sha256_file(proxy_metrics_json),
        "proxy_scores_csv_sha256": _icmt._cb._sha256_file(proxy_scores_csv),
    }


def _canonical_training_root(value: str | Path) -> Path:
    root = Path(value).resolve()
    if root.name != EXPECTED_TRAINING_RUN_LEAF or not root.is_dir():
        raise HNCCDPostfreezePairError(f"training run root must be existing {EXPECTED_TRAINING_RUN_LEAF}: {root}")
    return root


def _expected_final_checkpoint(training_root: Path, fold_index: int, arm: str) -> tuple[str, Path]:
    if fold_index not in range(1, 7) or arm not in {"C", "G"}:
        raise HNCCDPostfreezePairError("unsupported frozen HNCCD fold/arm")
    candidate = f"F{fold_index}{arm}_HNCCD12"
    return candidate, (training_root / candidate / "final_ssdg.pth").resolve()


def _strict_current_checkpoint(
    path: str | Path,
    *,
    training_root: Path,
    fold_index: int,
    arm: str,
    source_tx_ids: Sequence[str],
) -> dict[str, Any]:
    candidate, expected_path = _expected_final_checkpoint(training_root, fold_index, arm)
    observed = Path(path).resolve()
    if observed != expected_path or not observed.is_file():
        raise HNCCDPostfreezePairError(f"{arm} final checkpoint path does not bind frozen {candidate}")
    checkpoint = torch.load(observed, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise HNCCDPostfreezePairError(f"{arm} checkpoint payload must be a mapping")
    try:
        _, receipt, observed_arm = _hnccd_export.validate_hnccd_training_checkpoint(
            checkpoint,
            checkpoint_path=observed,
            source_tx_ids=source_tx_ids,
            known_validation_tx_ids=(_icmt.FROZEN_FOLD_KNOWN_HELDOUT_TX[fold_index],),
            proxy_unknown_tx_ids=(_icmt.FROZEN_FOLD_PROXY_TX[fold_index],),
        )
    except _hnccd_export.HNCCDSplitExportError as exc:
        raise _translate(exc) from exc
    if observed_arm != arm:
        raise HNCCDPostfreezePairError(f"{arm} checkpoint receipt arm drifted")
    raw_receipt = checkpoint.get("hnccd_receipt")
    if not isinstance(raw_receipt, Mapping):
        raise HNCCDPostfreezePairError(f"{arm} checkpoint lacks raw hnccd_receipt")
    return {
        "candidate": candidate,
        "path": observed,
        "sha256": _icmt._cb._sha256_file(observed),
        "receipt": dict(receipt),
        "raw_receipt_sha256": _hnccd_export._canonical_json_sha256(dict(raw_receipt)),
    }


def _strict_common_training_projection(receipt: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    """Extract C/G-common HNCCD scalar/count/SHA evidence without raw RX tokens."""

    if not isinstance(receipt, Mapping):
        raise HNCCDPostfreezePairError(f"{label} common training receipt is not a mapping")
    missing = [field for field in COMMON_TRAINING_BINDING_FIELDS if field not in receipt]
    if missing:
        raise HNCCDPostfreezePairError(f"{label} common training binding lacks fields: {','.join(missing)}")
    projection = {field: receipt[field] for field in COMMON_TRAINING_BINDING_FIELDS}
    for field in COMMON_TRAINING_SHA_FIELDS:
        try:
            _hnccd_export._require_sha256(projection[field], field=f"{label} common {field}")
        except _hnccd_export.HNCCDSplitExportError as exc:
            raise _translate(exc) from exc
    expected_literals = {
        "loss_rule": "SOURCE_L_ORDERED_RECEIVER_SLOT_BY_LOCAL4_LEO_HEAD_NULL_CROSS_COVARIANCE_DECORRELATION_TOTALIZED_L2_feat_joint",
        "loss_formula": "Q=W^T chol(WW^T)^(-T);h=Q^T u;b=u-Qh;C_rc=(H-Hbar)^T(B-Bbar)/n_rc;if_n_lt_2_C=0;L=sum_rc||C_rc||F^2/28",
        "z_id_key": "feat_joint",
        "training_accumulation_dtype": "float32_OUTSIDE_AMP",
        "clean_feature_detached": "NOT_READ_BY_HNCCD_AUXILIARY",
        "same_physical_pairing": "SAME_SOURCE_L_PHYSICAL_ROW_COMMON_CLEAN_AND_SINGLE_LEO_FORWARD",
        "receipt_payload": "SCALARS_COUNTS_AND_SHA_ONLY_NO_IQ_FEATURE_COVARIANCE_OR_RECEIVER_TOKEN",
        "common_sat_kl": "sg(clean_tx_logits)_TO_leo_tx_logits",
        "head_input_path": "model_output.tx_logits_from_id_backbone.cls_head.head(feat_joint)",
        "aux_gradient_scope": "LEO_feat_joint_SHARED_ENCODER_EXACT_HEAD_WEIGHT_FINITE_NONZERO;CLEAN_AND_HEAD_BIAS_NONE_OR_ZERO",
        "source_receiver_provenance": _hnccd_export.SOURCE_RECEIVER_PROVENANCE,
        "warm_start_mode": "MODEL_WEIGHTS_ONLY_NEW_ADAMW_AMP",
        "checkpoint_role": "training_final_only",
        "optimizer_type": "AdamW",
        "amp_contract": "COMMON_TRAINER_AMP_ENABLED",
        "exact_head_weight_path": "model.id_backbone.cls_head.head.weight",
        "head_null_basis_rule": "FP32_DIFFERENTIABLE_CHOLESKY_WWT_AND_TRIANGULAR_SOLVE_Q_EQ_WT_LINVTRANSPOSE_NO_PINV_EPSILON_OR_FALLBACK",
    }
    for field, expected in expected_literals.items():
        if projection[field] != expected or type(projection[field]) is not type(expected):
            raise HNCCDPostfreezePairError(f"{label} common training binding {field} drifted")
    if (
        projection["common_l_base_head_input_path_verified"] is not True
        or projection["strict_model_keys"] is not True
        or projection["missing_model_keys"] != []
        or projection["unexpected_model_keys"] != []
        or projection["optimizer_state_restored"] is not False
        or projection["rng_state_restored"] is not False
        or projection["optimizer_initial_state_empty"] is not True
        or projection["head_full_row_rank_required"] is not True
        or projection["resource_selection_feedback"] is not False
    ):
        raise HNCCDPostfreezePairError(f"{label} common strict warm-start/new-AdamW/head-null receipt drifted")
    for field, expected in (
        ("loss_global_denominator", _hnccd.FROZEN_HNCCD_TERM_DIVISOR),
        ("local_class_count", len(_hnccd.FROZEN_HNCCD_CLASS_IDS)),
        ("fixed_batch_size", _hnccd.FROZEN_HNCCD_BATCH_SIZE),
        ("fixed_local_class_count", len(_hnccd.FROZEN_HNCCD_CLASS_IDS)),
        ("frozen_batch_size", _hnccd.FROZEN_HNCCD_BATCH_SIZE),
        ("frozen_feature_dim", _hnccd.FROZEN_HNCCD_FEATURE_DIM),
        ("frozen_source_receiver_count", _hnccd.FROZEN_HNCCD_SOURCE_RECEIVER_COUNT),
        ("source_receiver_count", _hnccd.FROZEN_HNCCD_SOURCE_RECEIVER_COUNT),
        ("local_data_class_count", len(_hnccd.FROZEN_HNCCD_CLASS_IDS)),
        ("checkpoint_head_class_count", len(_hnccd.FROZEN_HNCCD_CLASS_IDS)),
        ("live_head_class_count", len(_hnccd.FROZEN_HNCCD_CLASS_IDS)),
    ):
        if type(projection[field]) is not int or projection[field] != expected:
            raise HNCCDPostfreezePairError(f"{label} common B128/d160/local4/fixed28 receipt drifted")
    if projection["exact_head_weight_shape"] != [len(_hnccd.FROZEN_HNCCD_CLASS_IDS), _hnccd.FROZEN_HNCCD_FEATURE_DIM]:
        raise HNCCDPostfreezePairError(f"{label} common exact head shape drifted")
    if projection["local_class_ids"] != list(_hnccd.FROZEN_HNCCD_CLASS_IDS):
        raise HNCCDPostfreezePairError(f"{label} common local class IDs drifted")
    try:
        sat_lambda = float(projection["common_lambda_sat_cons"])
    except (TypeError, ValueError) as exc:
        raise HNCCDPostfreezePairError(f"{label} common satellite lambda is malformed") from exc
    if not math.isfinite(sat_lambda) or not math.isclose(sat_lambda, 0.10, rel_tol=0.0, abs_tol=1e-12):
        raise HNCCDPostfreezePairError(f"{label} common lambda_sat_cons drifted")
    for field in (
        "uses_new_forward", "uses_resampling", "uses_day_labels", "uses_domain_labels",
        "uses_target_rows", "uses_proxy_rows", "uses_held_rows", "uses_unlabeled_rows",
        "uses_ema_or_state", "uses_threshold", "uses_cross_sample_pairing", "uses_cross_receiver_pairing",
    ):
        if projection[field] is not False:
            raise HNCCDPostfreezePairError(f"{label} common permission {field} drifted")
    if projection["uses_rx_labels"] is not True:
        raise HNCCDPostfreezePairError(f"{label} common source-Rs receiver binding drifted")
    scenarios = projection["common_scenario_batches"]
    cells = projection["hnccd_common_cells"]
    events = projection["hnccd_common_batch_cells"]
    expected_scenarios = tuple(_hnccd.FROZEN_HNCCD_SCENARIOS)
    if (
        type(scenarios) is not dict
        or set(scenarios) != set(expected_scenarios)
        or type(cells) is not dict
        or set(cells) != set(expected_scenarios)
    ):
        raise HNCCDPostfreezePairError(f"{label} common three-scene coverage receipt drifted")
    batches = int(projection["common_batch_sequence_batches"])
    rows = int(projection["common_batch_sequence_rows"])
    if batches <= 0 or rows != batches * _hnccd.FROZEN_HNCCD_BATCH_SIZE or type(events) is not list or len(events) != batches:
        raise HNCCDPostfreezePairError(f"{label} common B128 sequence receipt does not close")
    for scenario in expected_scenarios:
        if (
            type(scenarios[scenario]) is not int
            or scenarios[scenario] <= 0
            or not isinstance(cells[scenario], Mapping)
            or len(cells[scenario]) != _hnccd.FROZEN_HNCCD_TERM_DIVISOR
        ):
            raise HNCCDPostfreezePairError(f"{label} common scene fixed28 coverage drifted")
    for event in events:
        if (
            not isinstance(event, Mapping)
            or event.get("same_physical_clean_leo") is not True
            or len(str(event.get("row_order_sha256", ""))) != 64
            or int(event.get("fixed_denominator", -1)) != _hnccd.FROZEN_HNCCD_TERM_DIVISOR
        ):
            raise HNCCDPostfreezePairError(f"{label} common same-physical/order receipt drifted")
        n_rc = event.get("n_rc")
        if (
            not isinstance(n_rc, Mapping)
            or len(n_rc) != _hnccd.FROZEN_HNCCD_TERM_DIVISOR
            or sum(int(value) for value in n_rc.values()) != _hnccd.FROZEN_HNCCD_BATCH_SIZE
        ):
            raise HNCCDPostfreezePairError(f"{label} common receiver-class event does not close")
    _require_no_legacy_identity_fields(projection, label=f"{label} HNCCD common projection")
    return projection


def validate_hnccd_common_training_binding(c_receipt: Mapping[str, Any], g_receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Fail unless C/G share the complete HNCCD common source/order contract."""

    c_projection = _strict_common_training_projection(c_receipt, label="C")
    g_projection = _strict_common_training_projection(g_receipt, label="G")
    for field in COMMON_TRAINING_BINDING_FIELDS:
        if type(c_projection[field]) is not type(g_projection[field]) or c_projection[field] != g_projection[field]:
            raise HNCCDPostfreezePairError(f"C/G common training binding {field} differs")
    return {"passed": True, "fields": dict(c_projection), "sha256": _hnccd_export._canonical_json_sha256(c_projection)}


def _validate_persisted_common_training_binding(observed: Any, expected: Mapping[str, Any], *, label: str) -> None:
    if type(observed) is not dict or set(observed) != {"passed", "fields", "sha256"} or observed["passed"] is not True:
        raise HNCCDPostfreezePairError(f"{label} persisted HNCCD common binding is malformed")
    fields = observed["fields"]
    if type(fields) is not dict or set(fields) != set(COMMON_TRAINING_BINDING_FIELDS):
        raise HNCCDPostfreezePairError(f"{label} persisted HNCCD common binding fields drifted")
    normalized = _strict_common_training_projection(fields, label=label)
    expected_fields = expected.get("fields")
    if type(expected_fields) is not dict:
        raise HNCCDPostfreezePairError("internal expected HNCCD common binding is malformed")
    for field in COMMON_TRAINING_BINDING_FIELDS:
        if type(normalized[field]) is not type(expected_fields.get(field)) or normalized[field] != expected_fields.get(field):
            raise HNCCDPostfreezePairError(f"{label} persisted HNCCD common binding {field} does not match raw receipts")
    digest = _hnccd_export._canonical_json_sha256(normalized)
    if type(observed["sha256"]) is not str or observed["sha256"] != digest or digest != expected.get("sha256"):
        raise HNCCDPostfreezePairError(f"{label} persisted HNCCD common binding SHA256 does not match raw receipts")


def _require_exact_manifest_fields(manifest: Mapping[str, Any], *, checkpoint_info: Mapping[str, Any], label: str) -> None:
    _require_no_legacy_identity_fields(manifest, label=f"{label} HNCCD manifest")
    receipt = checkpoint_info["receipt"]
    arm = "G" if str(checkpoint_info["candidate"]).endswith("G_HNCCD12") else "C"
    expected = {
        "schema": EXPECTED_LV_EXPORT_SCHEMA,
        "method": "P1_HNCCD",
        "training_run_contract": EXPECTED_TRAINING_RUN_LEAF,
        "hnccd_receipt_schema": EXPECTED_HNCCD_RECEIPT_SCHEMA,
        "hnccd_receipt_sha256": checkpoint_info["raw_receipt_sha256"],
        "hnccd_terminal_contract": str(receipt["hnccd_terminal_contract"]),
        "hnccd_terminal_contract_passed": True,
        "hnccd_enabled": arm == "G",
        "hnccd_lambda": _hnccd_export.FROZEN_HNCCD_LAMBDA if arm == "G" else 0.0,
        "hnccd_frozen_batch_size": _hnccd.FROZEN_HNCCD_BATCH_SIZE,
        "hnccd_feature_dim": _hnccd.FROZEN_HNCCD_FEATURE_DIM,
        "hnccd_local_class_count": len(_hnccd.FROZEN_HNCCD_CLASS_IDS),
        "hnccd_loss_global_denominator": _hnccd.FROZEN_HNCCD_TERM_DIVISOR,
        "hnccd_fixed_batch_size": _hnccd.FROZEN_HNCCD_BATCH_SIZE,
        "hnccd_fixed_feature_dim": _hnccd.FROZEN_HNCCD_FEATURE_DIM,
        "hnccd_fixed_local_class_count": len(_hnccd.FROZEN_HNCCD_CLASS_IDS),
        "hnccd_fixed_cells_per_scene": _hnccd.FROZEN_HNCCD_TERM_DIVISOR,
        "hnccd_source_receiver_count": _hnccd.FROZEN_HNCCD_SOURCE_RECEIVER_COUNT,
        "hnccd_source_receiver_order_sha256": str(receipt["source_receiver_order_sha256"]),
        "hnccd_source_receiver_ids_sha256": str(receipt["source_receiver_ids_sha256"]),
        "hnccd_source_labeled_indices_sha256": str(receipt["source_labeled_indices_sha256"]),
        "hnccd_source_split_manifest_sha256": str(receipt["source_split_manifest_sha256"]),
        "hnccd_source_partition_sha256": str(receipt["source_partition_sha256"]),
        "hnccd_class_order_binding_sha256": str(receipt["class_order_binding_sha256"]),
        "hnccd_common_batch_sequence_sha256": str(receipt["common_batch_sequence_sha256"]),
        "hnccd_common_scenario_batches": {str(key): int(value) for key, value in dict(receipt["common_scenario_batches"]).items()},
        "hnccd_common_cells_sha256": _hnccd_export._canonical_json_sha256(receipt.get("hnccd_common_cells", {})),
        "hnccd_g_scenes_sha256": _hnccd_export._canonical_json_sha256(receipt.get("hnccd_scenes", {})) if arm == "G" else "",
        "hnccd_clean_head_bias_aux_vjp": "N_A_NONE_OR_ZERO_EXPECTED" if arm == "G" else "N_A",
        "hnccd_leo_encoder_head_weight_aux_vjp": "FINITE_NONZERO_REQUIRED" if arm == "G" else "N_A",
        "hnccd_common_physical_order_bound": True,
        "hnccd_common_scene_cycle_bound": True,
        "hnccd_raw_vjp_per_scene_required": True,
        "hnccd_leo_encoder_head_weight_vjp_finite_nonzero": True,
        "hnccd_clean_head_bias_vjp_na_none_or_zero": True,
        "proxy_selection_frozen_not_cli_tunable": True,
    }
    for field, wanted in expected.items():
        if manifest.get(field) != wanted or type(manifest.get(field)) is not type(wanted):
            raise HNCCDPostfreezePairError(f"{label} manifest {field} drifted")
    if str(manifest.get("candidate_id", "")) != str(checkpoint_info["candidate"]):
        raise HNCCDPostfreezePairError(f"{label} manifest HNCCD candidate binding drifted")
    if Path(str(manifest.get("checkpoint", ""))).resolve() != checkpoint_info["path"]:
        raise HNCCDPostfreezePairError(f"{label} manifest final checkpoint path drifted")
    if str(manifest.get("source_checkpoint_sha256", "")) != checkpoint_info["sha256"]:
        raise HNCCDPostfreezePairError(f"{label} manifest final checkpoint SHA256 drifted")


_ORIGINAL_VALIDATE_LV_PAYLOAD = _icmt._validate_lv_payload
_ORIGINAL_LOAD_LEO_BINDING = _icmt._load_icmt_leo_binding
_ORIGINAL_RECOMPUTE_PRIOR = _icmt._recompute_prior_pair_artifacts
_ORIGINAL_MATRIX_AGGREGATE = _icmt._matrix_aggregate
_ORIGINAL_FOLD_GATES = _icmt._fold_gates


def _validate_hnccd_lv_payload(
    payload: Mapping[str, Any],
    source_tx_ids: Sequence[str],
    fold_index: int,
    expected_proxy_count: int,
    *,
    label: str,
) -> dict[str, Any]:
    manifest = payload.get("manifest")
    if not isinstance(manifest, Mapping):
        raise HNCCDPostfreezePairError(f"{label} lacks HNCCD L/V manifest")
    arm = "C" if label.startswith("C") else "G" if label.startswith("G") else ""
    if not arm:
        raise HNCCDPostfreezePairError(f"{label} does not identify a C/G arm")
    checkpoint_value = Path(str(manifest.get("checkpoint", ""))).resolve()
    if len(checkpoint_value.parents) < 2:
        raise HNCCDPostfreezePairError(f"{label} checkpoint layout is invalid")
    training_root = checkpoint_value.parents[1]
    candidate, checkpoint_path = _expected_final_checkpoint(training_root, fold_index, arm)
    if checkpoint_path != checkpoint_value:
        raise HNCCDPostfreezePairError(f"{label} checkpoint layout does not bind candidate")
    checkpoint_info = _strict_current_checkpoint(
        checkpoint_path,
        training_root=training_root,
        fold_index=fold_index,
        arm=arm,
        source_tx_ids=source_tx_ids,
    )
    _require_exact_manifest_fields(manifest, checkpoint_info=checkpoint_info, label=label)
    compatibility_manifest = dict(manifest)
    compatibility_manifest.update(
        {
            "candidate_id": f"F{fold_index}{arm}_ICMT12",
            "icmt_receipt_schema": EXPECTED_HNCCD_RECEIPT_SCHEMA,
            "icmt_enabled": arm == "G",
            "icmt_source_labeled_indices_sha256": manifest["hnccd_source_labeled_indices_sha256"],
            "icmt_source_split_manifest_sha256": manifest["hnccd_source_split_manifest_sha256"],
        }
    )
    compatibility_payload = dict(payload)
    compatibility_payload["manifest"] = compatibility_manifest
    try:
        return _ORIGINAL_VALIDATE_LV_PAYLOAD(compatibility_payload, source_tx_ids, fold_index, expected_proxy_count, label=label)
    except _icmt.ICMTPostfreezePairError as exc:
        raise _translate(exc) from exc


def _load_hnccd_leo_binding(
    path: str | Path,
    leo_payload: Mapping[str, Any],
    clean_payload: Mapping[str, Any],
    *,
    expected_npz: Path,
    expected_checkpoint: Path,
    expected_candidate: str,
    fold_index: int,
    arm: str,
    source_tx_ids: Sequence[str],
    training_root: Path,
    output_root: Path,
    label: str,
) -> dict[str, Any]:
    try:
        result = _ORIGINAL_LOAD_LEO_BINDING(
            path,
            leo_payload,
            clean_payload,
            expected_npz=expected_npz,
            expected_checkpoint=expected_checkpoint,
            expected_candidate=expected_candidate,
            fold_index=fold_index,
            arm=arm,
            source_tx_ids=source_tx_ids,
            training_root=training_root,
            output_root=output_root,
            label=label,
        )
    except _icmt.ICMTPostfreezePairError as exc:
        raise _translate(exc) from exc
    try:
        binding = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise HNCCDPostfreezePairError(f"{label} HNCCD LEO binding JSON is invalid") from exc
    if not isinstance(binding, Mapping):
        raise HNCCDPostfreezePairError(f"{label} HNCCD LEO binding must encode an object")
    _require_no_legacy_identity_fields(binding, label=f"{label} HNCCD LEO binding")
    checkpoint_info = _strict_current_checkpoint(
        expected_checkpoint,
        training_root=training_root,
        fold_index=fold_index,
        arm=arm,
        source_tx_ids=source_tx_ids,
    )
    receipt = checkpoint_info["receipt"]
    expected = {
        "schema": EXPECTED_LEO_BINDING_SCHEMA,
        "method": "P1_HNCCD",
        "hnccd_receipt_schema": EXPECTED_HNCCD_RECEIPT_SCHEMA,
        "hnccd_receipt_sha256": checkpoint_info["raw_receipt_sha256"],
        "hnccd_terminal_contract": str(receipt["hnccd_terminal_contract"]),
        "hnccd_terminal_contract_passed": True,
        "hnccd_enabled": arm == "G",
        "hnccd_lambda": _hnccd_export.FROZEN_HNCCD_LAMBDA if arm == "G" else 0.0,
        "hnccd_frozen_batch_size": _hnccd.FROZEN_HNCCD_BATCH_SIZE,
        "hnccd_feature_dim": _hnccd.FROZEN_HNCCD_FEATURE_DIM,
        "hnccd_local_class_count": len(_hnccd.FROZEN_HNCCD_CLASS_IDS),
        "hnccd_loss_global_denominator": _hnccd.FROZEN_HNCCD_TERM_DIVISOR,
        "hnccd_fixed_batch_size": _hnccd.FROZEN_HNCCD_BATCH_SIZE,
        "hnccd_fixed_feature_dim": _hnccd.FROZEN_HNCCD_FEATURE_DIM,
        "hnccd_fixed_local_class_count": len(_hnccd.FROZEN_HNCCD_CLASS_IDS),
        "hnccd_fixed_cells_per_scene": _hnccd.FROZEN_HNCCD_TERM_DIVISOR,
        "hnccd_source_receiver_count": _hnccd.FROZEN_HNCCD_SOURCE_RECEIVER_COUNT,
        "hnccd_source_receiver_order_sha256": str(receipt["source_receiver_order_sha256"]),
        "hnccd_source_receiver_ids_sha256": str(receipt["source_receiver_ids_sha256"]),
        "hnccd_source_labeled_indices_sha256": str(receipt["source_labeled_indices_sha256"]),
        "hnccd_source_split_manifest_sha256": str(receipt["source_split_manifest_sha256"]),
        "hnccd_source_partition_sha256": str(receipt["source_partition_sha256"]),
        "hnccd_class_order_binding_sha256": str(receipt["class_order_binding_sha256"]),
        "hnccd_common_batch_sequence_sha256": str(receipt["common_batch_sequence_sha256"]),
        "hnccd_common_scenario_batches": {str(key): int(value) for key, value in dict(receipt["common_scenario_batches"]).items()},
        "hnccd_common_cells_sha256": _hnccd_export._canonical_json_sha256(receipt.get("hnccd_common_cells", {})),
        "hnccd_g_scenes_sha256": _hnccd_export._canonical_json_sha256(receipt.get("hnccd_scenes", {})) if arm == "G" else "",
        "hnccd_clean_head_bias_aux_vjp": "N_A_NONE_OR_ZERO_EXPECTED" if arm == "G" else "N_A",
        "hnccd_leo_encoder_head_weight_aux_vjp": "FINITE_NONZERO_REQUIRED" if arm == "G" else "N_A",
        "hnccd_single_leo_forward_bound": True,
        "hnccd_physical_tx_rx_day_binding_required": True,
        "hnccd_common_physical_order_bound": True,
        "hnccd_common_scene_cycle_bound": True,
        "hnccd_raw_vjp_per_scene_required": True,
        "hnccd_leo_encoder_head_weight_vjp_finite_nonzero": True,
        "hnccd_clean_head_bias_vjp_na_none_or_zero": True,
    }
    for field, wanted in expected.items():
        if binding.get(field) != wanted or type(binding.get(field)) is not type(wanted):
            raise HNCCDPostfreezePairError(f"{label} HNCCD LEO binding {field} drifted")
    return dict(result)


def _recompute_hnccd_prior_pair_artifacts(
    record: Mapping[str, Any],
    *,
    output_root: Path,
    matrix_id: str,
    training_root: Path,
    expected_scenarios: Sequence[str],
) -> dict[str, Any]:
    """F6 reopens F1--F5 raw artifacts/current receipt; never a self-report."""

    fold_index = int(record.get("fold_index", -1))
    if fold_index not in range(1, 7):
        raise HNCCDPostfreezePairError("prior HNCCD pair fold is invalid")
    source_tx_ids = _icmt.FROZEN_FOLD_SOURCE_TX[fold_index]
    infos: dict[str, dict[str, Any]] = {}
    for arm in ("C", "G"):
        _, checkpoint = _expected_final_checkpoint(training_root, fold_index, arm)
        infos[arm] = _strict_current_checkpoint(
            checkpoint,
            training_root=training_root,
            fold_index=fold_index,
            arm=arm,
            source_tx_ids=source_tx_ids,
        )
    common_binding = validate_hnccd_common_training_binding(infos["C"]["receipt"], infos["G"]["receipt"])
    _validate_persisted_common_training_binding(record.get("hnccd_common_training_binding"), common_binding, label=f"prior F{fold_index}")
    bindings = record.get("bindings")
    if not isinstance(bindings, Mapping):
        raise HNCCDPostfreezePairError("prior HNCCD pair lacks raw artifact bindings")
    for arm in ("c", "g"):
        _validate_proxy_logits_recompute(
            clean_npz=str(bindings.get(f"{arm}_clean_npz_path", "")),
            proxy_metrics_json=str(bindings.get(f"{arm}_proxy_metrics_json_path", "")),
            proxy_scores_csv=str(bindings.get(f"{arm}_proxy_scores_csv_path", "")),
            source_tx_ids=source_tx_ids,
            label=f"prior F{fold_index} {arm.upper()}",
        )
    try:
        result = dict(
            _ORIGINAL_RECOMPUTE_PRIOR(
                record,
                output_root=output_root,
                matrix_id=matrix_id,
                training_root=training_root,
                expected_scenarios=expected_scenarios,
            )
        )
    except _icmt.ICMTPostfreezePairError as exc:
        raise _translate(exc) from exc
    result["hnccd_common_training_binding"] = common_binding
    result["hnccd_raw_checkpoint_sha256"] = {arm: infos[arm]["sha256"] for arm in ("C", "G")}
    return _strip_legacy_identity_fields(result)


def _hnccd_fold_gates(
    clean_delta: Mapping[str, Any],
    leo_scenarios: Mapping[str, Mapping[str, Any]],
    proxy_guardrail: Mapping[str, Any],
    expected_scenarios: Sequence[str],
) -> dict[str, Any]:
    result = dict(_ORIGINAL_FOLD_GATES(clean_delta, leo_scenarios, proxy_guardrail, expected_scenarios))
    fold_overall = dict(result["fold_three_scenario_equal_weight_overall_delta_pp"])
    fold_value = float(fold_overall["value"])
    fold_overall["threshold_pp"] = HNCCD_FLOOR_DELTA_LIMIT_PP
    fold_overall["passed"] = bool(
        math.isfinite(fold_value) and fold_value >= HNCCD_FLOOR_DELTA_LIMIT_PP
    )
    result["fold_three_scenario_equal_weight_overall_delta_pp"] = fold_overall
    passed = bool(
        result["technical_binding"]["passed"] is True
        and result["clean_four_floors_ge_minus2pp"]["passed"] is True
        and result["leo_scenario_four_floors_ge_minus2pp"]["passed"] is True
        and fold_overall["passed"] is True
        and result["proxy_continuous_two_strict_improvements"]["passed"] is True
    )
    result["fold_verdict"] = (
        "PENDING_MAIN_REVIEW_FULL_6_FOLD" if passed else "REJECT_P1_HNCCD_PERMANENT"
    )
    return result


def _apply_hnccd_matrix_gate_contract(result: Mapping[str, Any]) -> dict[str, Any]:
    """Apply HNCCD's preregistered -2pp floor to fold/global overall gates."""

    corrected = dict(result)
    raw_gates = corrected.get("gates")
    if not isinstance(raw_gates, Mapping):
        raise HNCCDPostfreezePairError("HNCCD matrix aggregate lacks gates")
    gates = {str(name): dict(gate) for name, gate in raw_gates.items()}

    fold_gate = gates.get("fold_three_scenario_equal_weight_overall_delta_pp")
    if not isinstance(fold_gate, dict) or not isinstance(fold_gate.get("values"), Mapping):
        raise HNCCDPostfreezePairError("HNCCD matrix aggregate lacks fold overall values")
    fold_values = {str(name): float(value) for name, value in fold_gate["values"].items()}
    fold_gate["threshold_pp"] = HNCCD_FLOOR_DELTA_LIMIT_PP
    fold_gate["passed"] = bool(
        set(fold_values) == {f"F{fold}" for fold in range(1, 7)}
        and all(math.isfinite(value) and value >= HNCCD_FLOOR_DELTA_LIMIT_PP for value in fold_values.values())
    )

    global_gate = gates.get("global_18_cell_equal_weight_overall_delta_pp")
    if not isinstance(global_gate, dict):
        raise HNCCDPostfreezePairError("HNCCD matrix aggregate lacks global overall gate")
    global_value = float(global_gate.get("value", float("nan")))
    global_gate["threshold_pp"] = HNCCD_FLOOR_DELTA_LIMIT_PP
    global_gate["passed"] = bool(
        math.isfinite(global_value) and global_value >= HNCCD_FLOOR_DELTA_LIMIT_PP
    )

    gates["fold_three_scenario_equal_weight_overall_delta_pp"] = fold_gate
    gates["global_18_cell_equal_weight_overall_delta_pp"] = global_gate
    corrected["gates"] = gates
    required = (
        "technical_binding",
        "clean_6of6_four_floors_ge_minus2pp",
        "leo_18of18_four_floors_ge_minus2pp",
        "fold_three_scenario_equal_weight_overall_delta_pp",
        "global_18_cell_equal_weight_overall_delta_pp",
        "proxy_continuous_6of6_two_strict_improvements",
    )
    passed = all(gates[name].get("passed") is True for name in required)
    corrected["verdict"] = (
        "PHASE1_ADVANCEMENT_CANDIDATE_PENDING_MAIN_REVIEW"
        if passed
        else "REJECT_P1_HNCCD_PERMANENT"
    )
    return corrected


def _hnccd_matrix_aggregate(
    current: Mapping[str, Any],
    prior_paths: Sequence[str],
    *,
    expected_scenarios: Sequence[str],
    output_root: Path,
    matrix_id: str,
    training_root: Path,
) -> dict[str, Any]:
    fold_index = int(current.get("fold_index", -1))
    source_tx_ids = tuple(str(item) for item in current.get("source_tx_ids", []))
    infos: dict[str, dict[str, Any]] = {}
    for arm in ("C", "G"):
        _, checkpoint = _expected_final_checkpoint(training_root, fold_index, arm)
        infos[arm] = _strict_current_checkpoint(
            checkpoint,
            training_root=training_root,
            fold_index=fold_index,
            arm=arm,
            source_tx_ids=source_tx_ids,
        )
    common_binding = validate_hnccd_common_training_binding(infos["C"]["receipt"], infos["G"]["receipt"])
    if not isinstance(current, dict):
        raise HNCCDPostfreezePairError("current pair must be mutable for HNCCD binding receipt")
    current["hnccd_common_training_binding"] = common_binding
    try:
        result = dict(
            _ORIGINAL_MATRIX_AGGREGATE(
                current,
                prior_paths,
                expected_scenarios=expected_scenarios,
                output_root=output_root,
                matrix_id=matrix_id,
                training_root=training_root,
            )
        )
    except _icmt.ICMTPostfreezePairError as exc:
        raise _translate(exc) from exc
    return _strip_legacy_identity_fields(_apply_hnccd_matrix_gate_contract(result))


@contextmanager
def _patched_signed_fairness_kernel() -> Iterator[None]:
    """Use frozen numerical/floor mechanics while persisting only HNCCD identity."""

    saved = {
        "EXPECTED_TRAINING_RUN_LEAF": _icmt.EXPECTED_TRAINING_RUN_LEAF,
        "EXPECTED_LV_EXPORT_SCHEMA": _icmt.EXPECTED_LV_EXPORT_SCHEMA,
        "EXPECTED_PAIR_SCHEMA": _icmt.EXPECTED_PAIR_SCHEMA,
        "EXPECTED_ICMT_RECEIPT_SCHEMA": _icmt.EXPECTED_ICMT_RECEIPT_SCHEMA,
        "FROZEN_POSTFREEZE_CONTRACT": _icmt.FROZEN_POSTFREEZE_CONTRACT,
        "_expected_final_checkpoint": _icmt._expected_final_checkpoint,
        "_validate_lv_payload": _icmt._validate_lv_payload,
        "_load_icmt_leo_binding": _icmt._load_icmt_leo_binding,
        "_recompute_prior_pair_artifacts": _icmt._recompute_prior_pair_artifacts,
        "_fold_gates": _icmt._fold_gates,
        "_matrix_aggregate": _icmt._matrix_aggregate,
    }
    leo_saved = {
        "EXPECTED_TRAINING_RUN_LEAF": _icmt._icmt_leo.EXPECTED_TRAINING_RUN_LEAF,
        "EXPECTED_BINDING_SCHEMA": _icmt._icmt_leo.EXPECTED_BINDING_SCHEMA,
    }
    _icmt.EXPECTED_TRAINING_RUN_LEAF = EXPECTED_TRAINING_RUN_LEAF
    _icmt.EXPECTED_LV_EXPORT_SCHEMA = EXPECTED_LV_EXPORT_SCHEMA
    _icmt.EXPECTED_PAIR_SCHEMA = EXPECTED_PAIR_SCHEMA
    _icmt.EXPECTED_ICMT_RECEIPT_SCHEMA = EXPECTED_HNCCD_RECEIPT_SCHEMA
    _icmt.FROZEN_POSTFREEZE_CONTRACT = FROZEN_POSTFREEZE_CONTRACT
    _icmt._expected_final_checkpoint = _expected_final_checkpoint
    _icmt._validate_lv_payload = _validate_hnccd_lv_payload
    _icmt._load_icmt_leo_binding = _load_hnccd_leo_binding
    _icmt._recompute_prior_pair_artifacts = _recompute_hnccd_prior_pair_artifacts
    _icmt._fold_gates = _hnccd_fold_gates
    _icmt._matrix_aggregate = _hnccd_matrix_aggregate
    _icmt._icmt_leo.EXPECTED_TRAINING_RUN_LEAF = EXPECTED_TRAINING_RUN_LEAF
    _icmt._icmt_leo.EXPECTED_BINDING_SCHEMA = EXPECTED_LEO_BINDING_SCHEMA
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(_icmt, name, value)
        for name, value in leo_saved.items():
            setattr(_icmt._icmt_leo, name, value)


def _prevalidate_current_args(args: argparse.Namespace) -> tuple[Path, tuple[str, ...], int, dict[str, Any]]:
    if str(args.postfreeze_matrix_id).strip() != EXPECTED_POSTFREEZE_MATRIX_ID:
        raise HNCCDPostfreezePairError(f"postfreeze_matrix_id must be {EXPECTED_POSTFREEZE_MATRIX_ID}")
    output_root = Path(args.postfreeze_output_root).resolve()
    if output_root.name != EXPECTED_POSTFREEZE_MATRIX_ID or not output_root.is_dir():
        raise HNCCDPostfreezePairError("postfreeze output root does not bind frozen HNCCD matrix")
    training_root = _canonical_training_root(args.training_run_root)
    source_tx_ids = _icmt._cb._parse_items(args.source_tx_ids, field="source_tx_ids")
    fold_index = int(args.fold_index)
    if fold_index not in range(1, 7) or source_tx_ids != _icmt.FROZEN_FOLD_SOURCE_TX[fold_index]:
        raise HNCCDPostfreezePairError("HNCCD source TX/fold binding drifted")
    if str(args.candidate_pair) != f"F{fold_index}_C_vs_G":
        raise HNCCDPostfreezePairError("HNCCD candidate pair binding drifted")
    infos: dict[str, dict[str, Any]] = {}
    for arm, field in (("C", "c_final_checkpoint"), ("G", "g_final_checkpoint")):
        infos[arm] = _strict_current_checkpoint(
            getattr(args, field),
            training_root=training_root,
            fold_index=fold_index,
            arm=arm,
            source_tx_ids=source_tx_ids,
        )
    common_binding = validate_hnccd_common_training_binding(infos["C"]["receipt"], infos["G"]["receipt"])
    return training_root, source_tx_ids, fold_index, common_binding


def _atomic_rewrite_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".hnccd.tmp")
    if temporary.exists():
        raise HNCCDPostfreezePairError(f"refusing to overwrite temporary pair JSON: {temporary}")
    temporary.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _pair_verdict(metrics: Mapping[str, Any]) -> str:
    aggregate = metrics.get("matrix_aggregate")
    if isinstance(aggregate, Mapping):
        return str(aggregate.get("verdict", "REJECT_P1_HNCCD_PERMANENT"))
    gates = metrics.get("postfreeze_gates")
    if not isinstance(gates, Mapping) or str(gates.get("fold_verdict", "")).startswith("REJECT"):
        return "REJECT_P1_HNCCD_PERMANENT"
    return "PENDING_MAIN_REVIEW_FULL_6_FOLD"


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate one same-fold HNCCD pair; F6 reopens raw sealed evidence."""

    training_root, source_tx_ids, fold_index, common_binding = _prevalidate_current_args(args)
    proxy_recomputation = {
        "C": _validate_proxy_logits_recompute(
            clean_npz=args.c_clean_npz,
            proxy_metrics_json=args.c_proxy_metrics_json,
            proxy_scores_csv=args.c_proxy_scores_csv,
            source_tx_ids=source_tx_ids,
            label="C",
        ),
        "G": _validate_proxy_logits_recompute(
            clean_npz=args.g_clean_npz,
            proxy_metrics_json=args.g_proxy_metrics_json,
            proxy_scores_csv=args.g_proxy_scores_csv,
            source_tx_ids=source_tx_ids,
            label="G",
        ),
    }
    try:
        with _patched_signed_fairness_kernel():
            metrics = dict(_icmt.evaluate(args))
    except (_icmt.ICMTPostfreezePairError, _icmt._cb.CBSFCEPostfreezePairError) as exc:
        raise _translate(exc) from exc
    receipt_binding: dict[str, Any] = {}
    for arm in ("C", "G"):
        info = _strict_current_checkpoint(
            getattr(args, f"{arm.lower()}_final_checkpoint"),
            training_root=training_root,
            fold_index=fold_index,
            arm=arm,
            source_tx_ids=source_tx_ids,
        )
        receipt = info["receipt"]
        receipt_binding[arm] = {
            "candidate": info["candidate"],
            "final_checkpoint_sha256": info["sha256"],
            "raw_hnccd_receipt_sha256": info["raw_receipt_sha256"],
            "terminal_contract": receipt["hnccd_terminal_contract"],
            "terminal_contract_passed": True,
            "enabled": receipt["enabled"],
            "lambda": receipt["lambda"],
            "source_receiver_count": receipt["source_receiver_count"],
            "source_receiver_order_sha256": receipt["source_receiver_order_sha256"],
            "source_labeled_indices_sha256": receipt["source_labeled_indices_sha256"],
            "source_split_manifest_sha256": receipt["source_split_manifest_sha256"],
            "source_partition_sha256": receipt["source_partition_sha256"],
            "fixed_batch_size": receipt["fixed_batch_size"],
            "feature_dim": receipt["frozen_feature_dim"],
            "local_class_count": receipt["local_class_count"],
            "loss_global_denominator": receipt["loss_global_denominator"],
            "raw_vjp_per_scene_required": arm == "G",
            "clean_and_head_bias_aux_vjp": "N_A_NONE_OR_ZERO_EXPECTED" if arm == "G" else "N_A",
            "leo_encoder_and_head_weight_aux_vjp": "FINITE_NONZERO_REQUIRED" if arm == "G" else "N_A",
            "resource_observations_per_common_batch": len(receipt["hnccd_resource_observations"]),
        }
    metrics["hnccd_training_receipt_revalidation"] = receipt_binding
    metrics["hnccd_common_training_binding"] = common_binding
    metrics["hnccd_proxy_logits_recomputation"] = proxy_recomputation
    metrics["hnccd_f6_raw_reopen_required"] = True
    _validate_persisted_common_training_binding(metrics["hnccd_common_training_binding"], common_binding, label=f"current F{fold_index}")
    metrics = dict(_strip_legacy_identity_fields(metrics))
    metrics["verdict"] = _pair_verdict(metrics)
    _require_no_legacy_identity_fields(metrics, label="HNCCD pair output")
    _atomic_rewrite_json(Path(args.output_metrics_json).resolve(), metrics)
    return metrics


def safe_totalized_l2_float64(features: Any, *, label: str = "HNCCD features") -> np.ndarray:
    """Pure evaluation map: mask before divide; exact zero rows remain zero."""

    try:
        return _icmt._normalize_float64(features, label=label)
    except _icmt.ICMTPostfreezePairError as exc:
        raise _translate(exc) from exc


normalize_hnccd_float64 = safe_totalized_l2_float64


def fit_frozen_hnccd_diagonal_gaussian(features: Any, labels: Any, source_tx_ids: Sequence[str]) -> dict[str, Any]:
    """Fit frozen float64 geometry from source-L rows only."""

    try:
        return _icmt.fit_frozen_diagonal_gaussian(features, labels, source_tx_ids)
    except _icmt.ICMTPostfreezePairError as exc:
        raise _translate(exc) from exc


def score_frozen_hnccd_nll(features: Any, geometry: Mapping[str, Any]) -> np.ndarray:
    """Score sealed clean/LEO/proxy rows without updating the geometry."""

    try:
        return _icmt.score_frozen_icmt_nll(features, geometry)
    except _icmt.ICMTPostfreezePairError as exc:
        raise _translate(exc) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c-clean-npz", required=True)
    parser.add_argument("--g-clean-npz", required=True)
    parser.add_argument("--c-leo-npz", required=True)
    parser.add_argument("--g-leo-npz", required=True)
    parser.add_argument("--c-leo-binding-json", required=True)
    parser.add_argument("--g-leo-binding-json", required=True)
    parser.add_argument("--c-final-checkpoint", required=True)
    parser.add_argument("--g-final-checkpoint", required=True)
    parser.add_argument("--c-proxy-metrics-json", required=True)
    parser.add_argument("--g-proxy-metrics-json", required=True)
    parser.add_argument("--c-proxy-scores-csv", required=True)
    parser.add_argument("--g-proxy-scores-csv", required=True)
    parser.add_argument("--source-tx-ids", required=True)
    parser.add_argument("--candidate-pair", required=True)
    parser.add_argument("--fold-index", type=int, required=True)
    parser.add_argument("--postfreeze-matrix-id", default=EXPECTED_POSTFREEZE_MATRIX_ID)
    parser.add_argument("--postfreeze-output-root", required=True)
    parser.add_argument("--training-run-root", required=True)
    parser.add_argument("--expected-scenarios", default=",".join(_icmt.EXPECTED_SCENARIOS))
    parser.add_argument("--expected-source-days", default=",".join(_icmt.EXPECTED_SOURCE_DAYS))
    parser.add_argument("--expected-source-rxs", default=",".join(_icmt.EXPECTED_SOURCE_RXS))
    parser.add_argument("--source-sat-seed", type=int, default=7281718)
    parser.add_argument("--expected-source-count", type=int, default=1600)
    parser.add_argument("--expected-proxy-count", type=int, default=400)
    parser.add_argument("--aggregate-prior-pair-metrics-json", default="")
    parser.add_argument("--output-metrics-json", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    metrics = evaluate(build_parser().parse_args(argv))
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
