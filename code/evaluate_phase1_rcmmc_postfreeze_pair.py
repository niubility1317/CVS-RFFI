#!/usr/bin/env python
"""Final-only, source-paired P1-RCMMC postfreeze closure.

This facade owns RCMMC identity and reopens current raw RCMMC receipts for
both arms.  The signed generic kernel is used only for unchanged float64
totalized-L2 diagonal-Gaussian scoring, fixed proxy binding, strict floors,
and F6 raw-artifact recomputation.
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

from cvsrffi import phase1_rcmmc as _rcmmc
import export_phase1_rcmmc_features as _rcmmc_export
import export_phase1_rcmmc_leo_features as _rcmmc_leo
import evaluate_phase1_icmt_postfreeze_pair as _icmt


EXPECTED_TRAINING_RUN_LEAF = "phase1_rcmmc12_20260811_v1"
EXPECTED_POSTFREEZE_MATRIX_ID = "phase1_rcmmc_postfreeze_20260811_v1"
EXPECTED_LV_EXPORT_SCHEMA = "cvs.phase1.rcmmc_lv_export.v1"
EXPECTED_LEO_BINDING_SCHEMA = "cvs.phase1.rcmmc_leo_binding.v1"
EXPECTED_PAIR_SCHEMA = "cvs.phase1.rcmmc_postfreeze_pair.v1"
EXPECTED_RCMMC_RECEIPT_SCHEMA = "cvs.phase1.rcmmc_receipt.v1"
EXPECTED_HEAD_CONTRACT = "dual_cvsincnet_tx_logits_v1"
EXPECTED_CLEAN_ARTIFACT = "icmt_clean_l_v_proxy_final_only.npz"
LEGACY_IDENTITY_PREFIXES = ("icmt_", "hscf_", "rcat_", "recte_", "rcrmd_", "cagm_")
RAW_RECEIVER_TOKEN_FIELDS = frozenset({"source_receiver_ids", "frozen_source_receiver_ids"})

COMMON_TRAINING_BINDING_FIELDS = (
    "loss_rule", "loss_formula", "loss_global_denominator", "local_class_count",
    "frozen_batch_size", "frozen_feature_dim", "frozen_source_receiver_count",
    "z_id_key", "training_accumulation_dtype", "clean_feature_detached",
    "same_physical_pairing", "receipt_payload", "common_lambda_sat_cons", "common_sat_kl",
    "head_input_path", "common_l_base_head_input_path_verified", "aux_gradient_scope",
    "uses_new_forward", "uses_resampling", "uses_rx_labels", "uses_day_labels",
    "uses_domain_labels", "uses_target_rows", "uses_proxy_rows", "uses_held_rows",
    "uses_unlabeled_rows", "uses_ema_or_state", "uses_threshold", "uses_cross_sample_pairing",
    "uses_cross_receiver_pairing", "warm_start_mode", "baseline_sha256", "initial_checkpoint_sha256",
    "checkpoint_epoch", "checkpoint_role", "strict_model_keys", "missing_model_keys",
    "unexpected_model_keys", "optimizer_state_restored", "rng_state_restored", "optimizer_type",
    "optimizer_initial_state_sha256", "optimizer_initial_state_empty", "amp_contract",
    "source_partition_sha256", "source_labeled_indices_sha256", "source_split_manifest_sha256",
    "source_receiver_count", "source_receiver_order_sha256", "source_receiver_ids_sha256",
    "source_receiver_provenance", "class_order_binding_sha256", "local_data_class_count",
    "checkpoint_head_class_count", "live_head_class_count", "common_batch_sequence_sha256",
    "common_batch_sequence_batches", "common_batch_sequence_rows", "common_scenario_batches",
    "rcmmc_common_cells", "rcmmc_common_batch_cells",
)
COMMON_TRAINING_SHA_FIELDS = (
    "baseline_sha256", "initial_checkpoint_sha256", "source_partition_sha256",
    "source_labeled_indices_sha256", "source_split_manifest_sha256",
    "source_receiver_order_sha256", "source_receiver_ids_sha256",
    "class_order_binding_sha256", "optimizer_initial_state_sha256", "common_batch_sequence_sha256",
)
FROZEN_POSTFREEZE_CONTRACT = {
    "RCMMC-PF-01": "final-only L-only feat_joint diagonal Gaussian; V/proxy/U zero fit",
    "RCMMC-PF-02": "float64 safe totalized-L2 masks before divide, preserves exact zeros and rejects nonfinite values",
    "RCMMC-PF-03": "ddof1 class-equal pooled 0.9/0.1 shrink, 1e-6 floor, full NLL and stable u",
    "RCMMC-PF-04": "current raw RCMMC receipts prove B128/d160/local4/source-Rs SHA-count/fixed28/C-G fairness",
    "RCMMC-PF-05": "source-only single-LEO physical TX/RX/day plus fixed400 proxy JSON/CSV/NPZ current-SHA closure",
    "RCMMC-PF-06": "clean6/LEO18 four floors, fold/global overall, AUROC and u-gap are strict non-compensating gates",
    "RCMMC-PF-07": "F6 reopens F1--F5 sealed raw artifacts and receipts rather than trusting pair summaries",
}


class RCMMCPostfreezePairError(RuntimeError):
    """Raised when RCMMC postfreeze evidence cannot close fail-closed."""


def _translate(error: BaseException) -> RCMMCPostfreezePairError:
    return RCMMCPostfreezePairError(str(error))


def _require_no_legacy_identity_fields(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, raw_value in value.items():
            key = str(raw_key).lower()
            if key in RAW_RECEIVER_TOKEN_FIELDS:
                raise RCMMCPostfreezePairError(f"raw source receiver token leaked into {label}: {raw_key}")
            if key.startswith(LEGACY_IDENTITY_PREFIXES):
                raise RCMMCPostfreezePairError(f"historical method identity leaked into {label}: {raw_key}")
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
                raise RCMMCPostfreezePairError(f"raw source receiver token leaked into RCMMC pair output: {raw_key}")
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
    spec = importlib.util.spec_from_file_location("_rcmmc_frozen_logits_proxy", source)
    if spec is None or spec.loader is None:
        raise RCMMCPostfreezePairError("cannot load frozen logits proxy scorer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _LOGITS_REJECT_MODULE = module
    return module


def _close_proxy_value(actual: Any, expected: Any, *, field: str, label: str) -> None:
    if actual is None or expected is None:
        if actual is not expected:
            raise RCMMCPostfreezePairError(f"{label} proxy {field} does not match raw logits")
        return
    try:
        left, right = float(actual), float(expected)
    except (TypeError, ValueError) as exc:
        raise RCMMCPostfreezePairError(f"{label} proxy {field} is not numeric") from exc
    if not math.isfinite(left) or not math.isfinite(right) or not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12):
        raise RCMMCPostfreezePairError(f"{label} proxy {field} does not match raw logits")


def _validate_proxy_logits_recompute(
    *, clean_npz: str | Path, proxy_metrics_json: str | Path,
    proxy_scores_csv: str | Path, source_tx_ids: Sequence[str], label: str,
) -> dict[str, Any]:
    """Recompute fixed JSON/CSV proxy diagnostics from current clean NPZ bytes."""

    try:
        observed = json.loads(Path(proxy_metrics_json).read_text(encoding="utf-8"))
    except Exception as exc:
        raise RCMMCPostfreezePairError(f"{label} proxy diagnostic JSON is invalid") from exc
    if not isinstance(observed, Mapping):
        raise RCMMCPostfreezePairError(f"{label} proxy diagnostic JSON must encode an object")
    with tempfile.TemporaryDirectory(prefix="rcmmc_proxy_recompute_") as temporary:
        expected_csv = Path(temporary) / "scores.csv"
        recomputed = _logits_reject_module().evaluate(
            argparse.Namespace(
                feature_npz=str(Path(clean_npz).resolve()), source_tx_ids=",".join(str(item) for item in source_tx_ids),
                unknown_tx_ids="", known_query_roles="source_validation_known", unknown_query_roles="proxy_unknown",
                calibration_roles="source_validation_known", conf_quantile=0.05, margin_quantile=0.05,
                energy_quantile=0.95, disable_conf_gate=False, disable_margin_gate=False,
                disable_energy_gate=False, unknown_far_target=0.05, output_json="", score_table_csv=str(expected_csv),
            )
        )
        for field in (
            "AUROC_unknown", "unknown_FAR", "unknown_reject_rate", "known_closed_accuracy_no_reject",
            "known_coverage", "known_full_accuracy_after_reject", "known_accepted_accuracy", "old_retention_vs_closed",
        ):
            _close_proxy_value(observed.get(field), recomputed.get(field), field=field, label=label)
        for field in ("known_query_count", "unknown_query_count"):
            if type(observed.get(field)) is not int or observed.get(field) != recomputed.get(field):
                raise RCMMCPostfreezePairError(f"{label} proxy {field} does not match raw logits")
        if observed.get("gate_policy") != recomputed.get("gate_policy"):
            raise RCMMCPostfreezePairError(f"{label} proxy gate policy does not match raw logits")
        left_calibration, right_calibration = observed.get("calibration"), recomputed.get("calibration")
        if not isinstance(left_calibration, Mapping) or not isinstance(right_calibration, Mapping) or set(left_calibration) != set(right_calibration):
            raise RCMMCPostfreezePairError(f"{label} proxy calibration is malformed")
        for field in sorted(right_calibration):
            _close_proxy_value(left_calibration.get(field), right_calibration.get(field), field=f"calibration.{field}", label=label)
        try:
            with Path(proxy_scores_csv).open("r", encoding="utf-8", newline="") as handle:
                observed_rows = list(csv.DictReader(handle))
            with expected_csv.open("r", encoding="utf-8", newline="") as handle:
                expected_rows = list(csv.DictReader(handle))
        except Exception as exc:
            raise RCMMCPostfreezePairError(f"{label} proxy score CSV is invalid") from exc
        if observed_rows != expected_rows:
            raise RCMMCPostfreezePairError(f"{label} proxy score CSV does not match raw logits")
    return {
        "passed": True,
        "clean_npz_sha256": _icmt._cb._sha256_file(clean_npz),
        "proxy_metrics_json_sha256": _icmt._cb._sha256_file(proxy_metrics_json),
        "proxy_scores_csv_sha256": _icmt._cb._sha256_file(proxy_scores_csv),
    }


def _canonical_training_root(value: str | Path) -> Path:
    root = Path(value).resolve()
    if root.name != EXPECTED_TRAINING_RUN_LEAF or not root.is_dir():
        raise RCMMCPostfreezePairError(f"training run root must be existing {EXPECTED_TRAINING_RUN_LEAF}: {root}")
    return root


def _expected_final_checkpoint(training_root: Path, fold_index: int, arm: str) -> tuple[str, Path]:
    if fold_index not in range(1, 7) or arm not in {"C", "G"}:
        raise RCMMCPostfreezePairError("unsupported frozen RCMMC fold/arm")
    candidate = f"F{fold_index}{arm}_RCMMC12"
    return candidate, (training_root / candidate / "final_ssdg.pth").resolve()


def _strict_current_checkpoint(
    path: str | Path, *, training_root: Path, fold_index: int, arm: str,
    source_tx_ids: Sequence[str],
) -> dict[str, Any]:
    candidate, expected_path = _expected_final_checkpoint(training_root, fold_index, arm)
    observed = Path(path).resolve()
    if observed != expected_path or not observed.is_file():
        raise RCMMCPostfreezePairError(f"{arm} final checkpoint path does not bind frozen {candidate}")
    checkpoint = torch.load(observed, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise RCMMCPostfreezePairError(f"{arm} checkpoint payload must be a mapping")
    try:
        _, receipt, observed_arm = _rcmmc_export.validate_rcmmc_training_checkpoint(
            checkpoint, checkpoint_path=observed, source_tx_ids=source_tx_ids,
            known_validation_tx_ids=(_icmt.FROZEN_FOLD_KNOWN_HELDOUT_TX[fold_index],),
            proxy_unknown_tx_ids=(_icmt.FROZEN_FOLD_PROXY_TX[fold_index],),
        )
    except _rcmmc_export.RCMMCSplitExportError as exc:
        raise _translate(exc) from exc
    if observed_arm != arm:
        raise RCMMCPostfreezePairError(f"{arm} checkpoint receipt arm drifted")
    raw_receipt = checkpoint.get("rcmmc_receipt")
    if not isinstance(raw_receipt, Mapping):
        raise RCMMCPostfreezePairError(f"{arm} checkpoint lacks raw rcmmc_receipt")
    return {
        "candidate": candidate, "path": observed, "sha256": _icmt._cb._sha256_file(observed),
        "receipt": dict(receipt), "raw_receipt_sha256": _rcmmc_export._canonical_json_sha256(dict(raw_receipt)),
    }


def _strict_common_training_projection(receipt: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    """Extract C/G-common scalar/count/SHA RCMMC evidence without RX tokens."""

    if not isinstance(receipt, Mapping):
        raise RCMMCPostfreezePairError(f"{label} common training receipt is not a mapping")
    missing = [field for field in COMMON_TRAINING_BINDING_FIELDS if field not in receipt]
    if missing:
        raise RCMMCPostfreezePairError(f"{label} common training binding lacks fields: {','.join(missing)}")
    projection = {field: receipt[field] for field in COMMON_TRAINING_BINDING_FIELDS}
    for field in COMMON_TRAINING_SHA_FIELDS:
        try:
            _rcmmc_export._require_sha256(projection[field], field=f"{label} common {field}")
        except _rcmmc_export.RCMMCSplitExportError as exc:
            raise _translate(exc) from exc
    expected_literals = {
        "loss_rule": "SOURCE_L_ORDERED_RECEIVER_SLOT_BY_LOCAL4_MOMENT_MATRIX_CONGRUENCE_STOPGRAD_CLEAN_TO_LEO_TOTALIZED_L2_feat_joint",
        "loss_formula": "D_rc=2||mu_L-sg(mu_C)||2^2+||Q_L-sg(Q_C)||F^2;L=sum_rc(A_rc*D_rc)/28",
        "z_id_key": "feat_joint", "training_accumulation_dtype": "float32_OUTSIDE_AMP",
        "clean_feature_detached": True,
        "same_physical_pairing": "SAME_SOURCE_L_PHYSICAL_ROW_COMMON_CLEAN_AND_SINGLE_LEO_FORWARD",
        "receipt_payload": "SCALARS_COUNTS_AND_SHA_ONLY_NO_IQ_FEATURE_MOMENT_MATRIX_OR_RECEIVER_TOKEN",
        "common_sat_kl": "sg(clean_tx_logits)_TO_leo_tx_logits",
        "head_input_path": "model_output.tx_logits_from_id_backbone.cls_head.head(feat_joint)",
        "aux_gradient_scope": "LEO_feat_joint_AND_SHARED_ENCODER_FINITE_NONZERO;EXACT_HEAD_AUX_VJP_NA_NONE_OR_ZERO",
        "source_receiver_provenance": _rcmmc_export.SOURCE_RECEIVER_PROVENANCE,
        "warm_start_mode": "MODEL_WEIGHTS_ONLY_NEW_ADAMW_AMP", "checkpoint_role": "training_final_only",
        "optimizer_type": "AdamW", "amp_contract": "COMMON_TRAINER_AMP_ENABLED",
    }
    for field, expected in expected_literals.items():
        if projection[field] != expected or type(projection[field]) is not type(expected):
            raise RCMMCPostfreezePairError(f"{label} common training binding {field} drifted")
    if (
        projection["common_l_base_head_input_path_verified"] is not True
        or projection["strict_model_keys"] is not True
        or projection["missing_model_keys"] != [] or projection["unexpected_model_keys"] != []
        or projection["optimizer_state_restored"] is not False or projection["rng_state_restored"] is not False
        or projection["optimizer_initial_state_empty"] is not True
    ):
        raise RCMMCPostfreezePairError(f"{label} common strict warm-start/new-AdamW receipt drifted")
    for field, expected in (
        ("loss_global_denominator", _rcmmc.FROZEN_RCMMC_TERM_DIVISOR),
        ("local_class_count", len(_rcmmc.FROZEN_RCMMC_CLASS_IDS)),
        ("frozen_batch_size", _rcmmc.FROZEN_RCMMC_BATCH_SIZE),
        ("frozen_feature_dim", _rcmmc.FROZEN_RCMMC_FEATURE_DIM),
        ("frozen_source_receiver_count", _rcmmc.FROZEN_RCMMC_SOURCE_RECEIVER_COUNT),
        ("source_receiver_count", _rcmmc.FROZEN_RCMMC_SOURCE_RECEIVER_COUNT),
        ("local_data_class_count", len(_rcmmc.FROZEN_RCMMC_CLASS_IDS)),
        ("checkpoint_head_class_count", len(_rcmmc.FROZEN_RCMMC_CLASS_IDS)),
        ("live_head_class_count", len(_rcmmc.FROZEN_RCMMC_CLASS_IDS)),
    ):
        if type(projection[field]) is not int or projection[field] != expected:
            raise RCMMCPostfreezePairError(f"{label} common B128/d160/local4/fixed28 receipt drifted")
    try:
        sat_lambda = float(projection["common_lambda_sat_cons"])
    except (TypeError, ValueError) as exc:
        raise RCMMCPostfreezePairError(f"{label} common satellite lambda is malformed") from exc
    if not math.isfinite(sat_lambda) or not math.isclose(sat_lambda, 0.10, rel_tol=0.0, abs_tol=1e-12):
        raise RCMMCPostfreezePairError(f"{label} common lambda_sat_cons drifted")
    for field in (
        "uses_new_forward", "uses_resampling", "uses_day_labels", "uses_domain_labels",
        "uses_target_rows", "uses_proxy_rows", "uses_held_rows", "uses_unlabeled_rows",
        "uses_ema_or_state", "uses_threshold", "uses_cross_sample_pairing", "uses_cross_receiver_pairing",
    ):
        if projection[field] is not False:
            raise RCMMCPostfreezePairError(f"{label} common permission {field} drifted")
    if projection["uses_rx_labels"] is not True:
        raise RCMMCPostfreezePairError(f"{label} common source-Rs receiver binding drifted")
    scenarios = projection["common_scenario_batches"]
    cells = projection["rcmmc_common_cells"]
    events = projection["rcmmc_common_batch_cells"]
    expected_scenarios = tuple(_rcmmc.FROZEN_RCMMC_SCENARIOS)
    if type(scenarios) is not dict or set(scenarios) != set(expected_scenarios) or type(cells) is not dict or set(cells) != set(expected_scenarios):
        raise RCMMCPostfreezePairError(f"{label} common three-scene coverage receipt drifted")
    batches = int(projection["common_batch_sequence_batches"])
    rows = int(projection["common_batch_sequence_rows"])
    if batches <= 0 or rows != batches * _rcmmc.FROZEN_RCMMC_BATCH_SIZE or type(events) is not list or len(events) != batches:
        raise RCMMCPostfreezePairError(f"{label} common B128 sequence receipt does not close")
    for scene in expected_scenarios:
        if type(scenarios[scene]) is not int or scenarios[scene] <= 0 or not isinstance(cells[scene], Mapping) or len(cells[scene]) != _rcmmc.FROZEN_RCMMC_TERM_DIVISOR:
            raise RCMMCPostfreezePairError(f"{label} common scene fixed28 coverage drifted")
    for event in events:
        if not isinstance(event, Mapping) or event.get("same_physical_clean_leo") is not True or len(str(event.get("row_order_sha256", ""))) != 64:
            raise RCMMCPostfreezePairError(f"{label} common same-physical/order receipt drifted")
        n_rc = event.get("n_rc")
        if not isinstance(n_rc, Mapping) or len(n_rc) != _rcmmc.FROZEN_RCMMC_TERM_DIVISOR or sum(int(value) for value in n_rc.values()) != _rcmmc.FROZEN_RCMMC_BATCH_SIZE:
            raise RCMMCPostfreezePairError(f"{label} common receiver-class event does not close")
    _require_no_legacy_identity_fields(projection, label=f"{label} RCMMC common projection")
    return projection


def validate_rcmmc_common_training_binding(c_receipt: Mapping[str, Any], g_receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Fail unless C/G share the complete RCMMC common source/order contract."""

    c_projection = _strict_common_training_projection(c_receipt, label="C")
    g_projection = _strict_common_training_projection(g_receipt, label="G")
    for field in COMMON_TRAINING_BINDING_FIELDS:
        if type(c_projection[field]) is not type(g_projection[field]) or c_projection[field] != g_projection[field]:
            raise RCMMCPostfreezePairError(f"C/G common training binding {field} differs")
    return {"passed": True, "fields": dict(c_projection), "sha256": _rcmmc_export._canonical_json_sha256(c_projection)}


def _validate_persisted_common_training_binding(observed: Any, expected: Mapping[str, Any], *, label: str) -> None:
    if type(observed) is not dict or set(observed) != {"passed", "fields", "sha256"} or observed["passed"] is not True:
        raise RCMMCPostfreezePairError(f"{label} persisted RCMMC common binding is malformed")
    fields = observed["fields"]
    if type(fields) is not dict or set(fields) != set(COMMON_TRAINING_BINDING_FIELDS):
        raise RCMMCPostfreezePairError(f"{label} persisted RCMMC common binding fields drifted")
    normalized = _strict_common_training_projection(fields, label=label)
    expected_fields = expected.get("fields")
    if type(expected_fields) is not dict:
        raise RCMMCPostfreezePairError("internal expected RCMMC common binding is malformed")
    for field in COMMON_TRAINING_BINDING_FIELDS:
        if type(normalized[field]) is not type(expected_fields.get(field)) or normalized[field] != expected_fields.get(field):
            raise RCMMCPostfreezePairError(f"{label} persisted RCMMC common binding {field} does not match raw receipts")
    digest = _rcmmc_export._canonical_json_sha256(normalized)
    if type(observed["sha256"]) is not str or observed["sha256"] != digest or digest != expected.get("sha256"):
        raise RCMMCPostfreezePairError(f"{label} persisted RCMMC common binding SHA256 does not match raw receipts")


def _require_exact_manifest_fields(manifest: Mapping[str, Any], *, checkpoint_info: Mapping[str, Any], label: str) -> None:
    _require_no_legacy_identity_fields(manifest, label=f"{label} RCMMC manifest")
    receipt = checkpoint_info["receipt"]
    arm = "G" if str(checkpoint_info["candidate"]).endswith("G_RCMMC12") else "C"
    expected = {
        "schema": EXPECTED_LV_EXPORT_SCHEMA, "method": "P1_RCMMC",
        "training_run_contract": EXPECTED_TRAINING_RUN_LEAF,
        "rcmmc_receipt_schema": EXPECTED_RCMMC_RECEIPT_SCHEMA,
        "rcmmc_receipt_sha256": checkpoint_info["raw_receipt_sha256"],
        "rcmmc_terminal_contract": str(receipt["rcmmc_terminal_contract"]),
        "rcmmc_terminal_contract_passed": True, "rcmmc_enabled": arm == "G",
        "rcmmc_lambda": _rcmmc_export.FROZEN_RCMMC_LAMBDA if arm == "G" else 0.0,
        "rcmmc_frozen_batch_size": _rcmmc.FROZEN_RCMMC_BATCH_SIZE,
        "rcmmc_feature_dim": _rcmmc.FROZEN_RCMMC_FEATURE_DIM,
        "rcmmc_local_class_count": len(_rcmmc.FROZEN_RCMMC_CLASS_IDS),
        "rcmmc_loss_global_denominator": _rcmmc.FROZEN_RCMMC_TERM_DIVISOR,
        "rcmmc_fixed_batch_size": _rcmmc.FROZEN_RCMMC_BATCH_SIZE,
        "rcmmc_fixed_feature_dim": _rcmmc.FROZEN_RCMMC_FEATURE_DIM,
        "rcmmc_fixed_local_class_count": len(_rcmmc.FROZEN_RCMMC_CLASS_IDS),
        "rcmmc_fixed_cells_per_scene": _rcmmc.FROZEN_RCMMC_TERM_DIVISOR,
        "rcmmc_source_receiver_count": _rcmmc.FROZEN_RCMMC_SOURCE_RECEIVER_COUNT,
        "rcmmc_source_receiver_order_sha256": str(receipt["source_receiver_order_sha256"]),
        "rcmmc_source_receiver_ids_sha256": str(receipt["source_receiver_ids_sha256"]),
        "rcmmc_source_labeled_indices_sha256": str(receipt["source_labeled_indices_sha256"]),
        "rcmmc_source_split_manifest_sha256": str(receipt["source_split_manifest_sha256"]),
        "rcmmc_source_partition_sha256": str(receipt["source_partition_sha256"]),
        "rcmmc_class_order_binding_sha256": str(receipt["class_order_binding_sha256"]),
        "rcmmc_common_batch_sequence_sha256": str(receipt["common_batch_sequence_sha256"]),
        "rcmmc_common_scenario_batches": {str(key): int(value) for key, value in dict(receipt["common_scenario_batches"]).items()},
        "rcmmc_common_cells_sha256": _rcmmc_export._canonical_json_sha256(receipt.get("rcmmc_common_cells", {})),
        "rcmmc_g_scenes_sha256": _rcmmc_export._canonical_json_sha256(receipt.get("rcmmc_scenes", {})) if arm == "G" else "",
        "rcmmc_clean_head_aux_vjp": "N_A_NONE_OR_ZERO_EXPECTED" if arm == "G" else "N_A",
        "rcmmc_leo_encoder_aux_vjp": "FINITE_NONZERO_REQUIRED" if arm == "G" else "N_A",
        "rcmmc_common_physical_order_bound": True,
        "rcmmc_common_scene_cycle_bound": True,
        "rcmmc_raw_vjp_required": True,
        "rcmmc_leo_encoder_vjp_finite_nonzero": True,
        "rcmmc_clean_head_vjp_na_none_or_zero": True,
        "proxy_selection_frozen_not_cli_tunable": True,
    }
    for field, wanted in expected.items():
        if manifest.get(field) != wanted or type(manifest.get(field)) is not type(wanted):
            raise RCMMCPostfreezePairError(f"{label} manifest {field} drifted")
    if str(manifest.get("candidate_id", "")) != str(checkpoint_info["candidate"]):
        raise RCMMCPostfreezePairError(f"{label} manifest RCMMC candidate binding drifted")
    if Path(str(manifest.get("checkpoint", ""))).resolve() != checkpoint_info["path"]:
        raise RCMMCPostfreezePairError(f"{label} manifest final checkpoint path drifted")
    if str(manifest.get("source_checkpoint_sha256", "")) != checkpoint_info["sha256"]:
        raise RCMMCPostfreezePairError(f"{label} manifest final checkpoint SHA256 drifted")


_ORIGINAL_VALIDATE_LV_PAYLOAD = _icmt._validate_lv_payload
_ORIGINAL_LOAD_LEO_BINDING = _icmt._load_icmt_leo_binding
_ORIGINAL_RECOMPUTE_PRIOR = _icmt._recompute_prior_pair_artifacts
_ORIGINAL_MATRIX_AGGREGATE = _icmt._matrix_aggregate
_ORIGINAL_FOLD_GATES = _icmt._fold_gates


def _validate_rcmmc_lv_payload(
    payload: Mapping[str, Any], source_tx_ids: Sequence[str], fold_index: int,
    expected_proxy_count: int, *, label: str,
) -> dict[str, Any]:
    manifest = payload.get("manifest")
    if not isinstance(manifest, Mapping):
        raise RCMMCPostfreezePairError(f"{label} lacks RCMMC L/V manifest")
    arm = "C" if label.startswith("C") else "G" if label.startswith("G") else ""
    if not arm:
        raise RCMMCPostfreezePairError(f"{label} does not identify a C/G arm")
    checkpoint_value = Path(str(manifest.get("checkpoint", ""))).resolve()
    if len(checkpoint_value.parents) < 2:
        raise RCMMCPostfreezePairError(f"{label} checkpoint layout is invalid")
    training_root = checkpoint_value.parents[1]
    candidate, checkpoint_path = _expected_final_checkpoint(training_root, fold_index, arm)
    if checkpoint_path != checkpoint_value:
        raise RCMMCPostfreezePairError(f"{label} checkpoint layout does not bind candidate")
    checkpoint_info = _strict_current_checkpoint(checkpoint_path, training_root=training_root, fold_index=fold_index, arm=arm, source_tx_ids=source_tx_ids)
    _require_exact_manifest_fields(manifest, checkpoint_info=checkpoint_info, label=label)
    compatibility_manifest = dict(manifest)
    compatibility_manifest.update(
        {
            "candidate_id": f"F{fold_index}{arm}_ICMT12",
            "icmt_receipt_schema": EXPECTED_RCMMC_RECEIPT_SCHEMA,
            "icmt_enabled": arm == "G",
            "icmt_source_labeled_indices_sha256": manifest["rcmmc_source_labeled_indices_sha256"],
            "icmt_source_split_manifest_sha256": manifest["rcmmc_source_split_manifest_sha256"],
        }
    )
    compatibility_payload = dict(payload)
    compatibility_payload["manifest"] = compatibility_manifest
    try:
        return _ORIGINAL_VALIDATE_LV_PAYLOAD(compatibility_payload, source_tx_ids, fold_index, expected_proxy_count, label=label)
    except _icmt.ICMTPostfreezePairError as exc:
        raise _translate(exc) from exc


def _load_rcmmc_leo_binding(
    path: str | Path, leo_payload: Mapping[str, Any], clean_payload: Mapping[str, Any], *,
    expected_npz: Path, expected_checkpoint: Path, expected_candidate: str, fold_index: int,
    arm: str, source_tx_ids: Sequence[str], training_root: Path, output_root: Path, label: str,
) -> dict[str, Any]:
    try:
        result = _ORIGINAL_LOAD_LEO_BINDING(
            path, leo_payload, clean_payload, expected_npz=expected_npz,
            expected_checkpoint=expected_checkpoint, expected_candidate=expected_candidate,
            fold_index=fold_index, arm=arm, source_tx_ids=source_tx_ids,
            training_root=training_root, output_root=output_root, label=label,
        )
    except _icmt.ICMTPostfreezePairError as exc:
        raise _translate(exc) from exc
    try:
        binding = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise RCMMCPostfreezePairError(f"{label} RCMMC LEO binding JSON is invalid") from exc
    if not isinstance(binding, Mapping):
        raise RCMMCPostfreezePairError(f"{label} RCMMC LEO binding must encode an object")
    _require_no_legacy_identity_fields(binding, label=f"{label} RCMMC LEO binding")
    checkpoint_info = _strict_current_checkpoint(expected_checkpoint, training_root=training_root, fold_index=fold_index, arm=arm, source_tx_ids=source_tx_ids)
    receipt = checkpoint_info["receipt"]
    expected = {
        "schema": EXPECTED_LEO_BINDING_SCHEMA, "method": "P1_RCMMC",
        "rcmmc_receipt_schema": EXPECTED_RCMMC_RECEIPT_SCHEMA,
        "rcmmc_receipt_sha256": checkpoint_info["raw_receipt_sha256"],
        "rcmmc_terminal_contract": str(receipt["rcmmc_terminal_contract"]),
        "rcmmc_terminal_contract_passed": True, "rcmmc_enabled": arm == "G",
        "rcmmc_lambda": _rcmmc_export.FROZEN_RCMMC_LAMBDA if arm == "G" else 0.0,
        "rcmmc_frozen_batch_size": _rcmmc.FROZEN_RCMMC_BATCH_SIZE,
        "rcmmc_feature_dim": _rcmmc.FROZEN_RCMMC_FEATURE_DIM,
        "rcmmc_local_class_count": len(_rcmmc.FROZEN_RCMMC_CLASS_IDS),
        "rcmmc_loss_global_denominator": _rcmmc.FROZEN_RCMMC_TERM_DIVISOR,
        "rcmmc_fixed_batch_size": _rcmmc.FROZEN_RCMMC_BATCH_SIZE,
        "rcmmc_fixed_feature_dim": _rcmmc.FROZEN_RCMMC_FEATURE_DIM,
        "rcmmc_fixed_local_class_count": len(_rcmmc.FROZEN_RCMMC_CLASS_IDS),
        "rcmmc_fixed_cells_per_scene": _rcmmc.FROZEN_RCMMC_TERM_DIVISOR,
        "rcmmc_source_receiver_count": _rcmmc.FROZEN_RCMMC_SOURCE_RECEIVER_COUNT,
        "rcmmc_source_receiver_order_sha256": str(receipt["source_receiver_order_sha256"]),
        "rcmmc_source_receiver_ids_sha256": str(receipt["source_receiver_ids_sha256"]),
        "rcmmc_source_labeled_indices_sha256": str(receipt["source_labeled_indices_sha256"]),
        "rcmmc_source_split_manifest_sha256": str(receipt["source_split_manifest_sha256"]),
        "rcmmc_source_partition_sha256": str(receipt["source_partition_sha256"]),
        "rcmmc_class_order_binding_sha256": str(receipt["class_order_binding_sha256"]),
        "rcmmc_common_batch_sequence_sha256": str(receipt["common_batch_sequence_sha256"]),
        "rcmmc_common_scenario_batches": {str(key): int(value) for key, value in dict(receipt["common_scenario_batches"]).items()},
        "rcmmc_common_cells_sha256": _rcmmc_export._canonical_json_sha256(receipt.get("rcmmc_common_cells", {})),
        "rcmmc_g_scenes_sha256": _rcmmc_export._canonical_json_sha256(receipt.get("rcmmc_scenes", {})) if arm == "G" else "",
        "rcmmc_clean_head_aux_vjp": "N_A_NONE_OR_ZERO_EXPECTED" if arm == "G" else "N_A",
        "rcmmc_leo_encoder_aux_vjp": "FINITE_NONZERO_REQUIRED" if arm == "G" else "N_A",
        "rcmmc_single_leo_forward_bound": True, "rcmmc_physical_tx_rx_day_binding_required": True,
        "rcmmc_common_physical_order_bound": True, "rcmmc_common_scene_cycle_bound": True,
        "rcmmc_raw_vjp_required": True, "rcmmc_leo_encoder_vjp_finite_nonzero": True,
        "rcmmc_clean_head_vjp_na_none_or_zero": True,
    }
    for field, wanted in expected.items():
        if binding.get(field) != wanted or type(binding.get(field)) is not type(wanted):
            raise RCMMCPostfreezePairError(f"{label} RCMMC LEO binding {field} drifted")
    return dict(result)


def _recompute_rcmmc_prior_pair_artifacts(
    record: Mapping[str, Any], *, output_root: Path, matrix_id: str,
    training_root: Path, expected_scenarios: Sequence[str],
) -> dict[str, Any]:
    """F6: reopen current raw RCMMC receipts/artifacts, never a prior summary."""

    fold_index = int(record.get("fold_index", -1))
    if fold_index not in range(1, 7):
        raise RCMMCPostfreezePairError("prior RCMMC pair fold is invalid")
    source_tx_ids = _icmt.FROZEN_FOLD_SOURCE_TX[fold_index]
    infos: dict[str, dict[str, Any]] = {}
    for arm in ("C", "G"):
        _, checkpoint = _expected_final_checkpoint(training_root, fold_index, arm)
        infos[arm] = _strict_current_checkpoint(checkpoint, training_root=training_root, fold_index=fold_index, arm=arm, source_tx_ids=source_tx_ids)
    common_binding = validate_rcmmc_common_training_binding(infos["C"]["receipt"], infos["G"]["receipt"])
    _validate_persisted_common_training_binding(record.get("rcmmc_common_training_binding"), common_binding, label=f"prior F{fold_index}")
    bindings = record.get("bindings")
    if not isinstance(bindings, Mapping):
        raise RCMMCPostfreezePairError("prior RCMMC pair lacks raw artifact bindings")
    for arm in ("c", "g"):
        _validate_proxy_logits_recompute(
            clean_npz=str(bindings.get(f"{arm}_clean_npz_path", "")),
            proxy_metrics_json=str(bindings.get(f"{arm}_proxy_metrics_json_path", "")),
            proxy_scores_csv=str(bindings.get(f"{arm}_proxy_scores_csv_path", "")),
            source_tx_ids=source_tx_ids, label=f"prior F{fold_index} {arm.upper()}",
        )
    try:
        result = dict(_ORIGINAL_RECOMPUTE_PRIOR(record, output_root=output_root, matrix_id=matrix_id, training_root=training_root, expected_scenarios=expected_scenarios))
    except _icmt.ICMTPostfreezePairError as exc:
        raise _translate(exc) from exc
    result["rcmmc_common_training_binding"] = common_binding
    result["rcmmc_raw_checkpoint_sha256"] = {arm: infos[arm]["sha256"] for arm in ("C", "G")}
    return _strip_legacy_identity_fields(result)


def _rcmmc_fold_gates(
    clean_delta: Mapping[str, Any], leo_scenarios: Mapping[str, Mapping[str, Any]],
    proxy_guardrail: Mapping[str, Any], expected_scenarios: Sequence[str],
) -> dict[str, Any]:
    result = dict(_ORIGINAL_FOLD_GATES(clean_delta, leo_scenarios, proxy_guardrail, expected_scenarios))
    if result.get("fold_verdict") == "REJECT_P1_ICMT_PERMANENT":
        result["fold_verdict"] = "REJECT_P1_RCMMC_PERMANENT"
    elif result.get("fold_verdict") == "PENDING_GLOBAL_18_GRID":
        result["fold_verdict"] = "PENDING_MAIN_REVIEW_FULL_6_FOLD"
    return result


def _rcmmc_matrix_aggregate(
    current: Mapping[str, Any], prior_paths: Sequence[str], *, expected_scenarios: Sequence[str],
    output_root: Path, matrix_id: str, training_root: Path,
) -> dict[str, Any]:
    fold_index = int(current.get("fold_index", -1))
    source_tx_ids = tuple(str(item) for item in current.get("source_tx_ids", []))
    infos: dict[str, dict[str, Any]] = {}
    for arm in ("C", "G"):
        _, checkpoint = _expected_final_checkpoint(training_root, fold_index, arm)
        infos[arm] = _strict_current_checkpoint(checkpoint, training_root=training_root, fold_index=fold_index, arm=arm, source_tx_ids=source_tx_ids)
    common_binding = validate_rcmmc_common_training_binding(infos["C"]["receipt"], infos["G"]["receipt"])
    if not isinstance(current, dict):
        raise RCMMCPostfreezePairError("current pair must be mutable for RCMMC binding receipt")
    current["rcmmc_common_training_binding"] = common_binding
    try:
        result = dict(_ORIGINAL_MATRIX_AGGREGATE(current, prior_paths, expected_scenarios=expected_scenarios, output_root=output_root, matrix_id=matrix_id, training_root=training_root))
    except _icmt.ICMTPostfreezePairError as exc:
        raise _translate(exc) from exc
    result["verdict"] = "REJECT_P1_RCMMC_PERMANENT" if result.get("verdict") == "REJECT_P1_ICMT_PERMANENT" else "PENDING_MAIN_REVIEW"
    return _strip_legacy_identity_fields(result)


@contextmanager
def _patched_signed_fairness_kernel() -> Iterator[None]:
    """Use unchanged numerical/floor mechanics while persisting only RCMMC identity."""

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
        "_fold_gates": _icmt._fold_gates, "_matrix_aggregate": _icmt._matrix_aggregate,
    }
    leo_saved = {
        "EXPECTED_TRAINING_RUN_LEAF": _icmt._icmt_leo.EXPECTED_TRAINING_RUN_LEAF,
        "EXPECTED_BINDING_SCHEMA": _icmt._icmt_leo.EXPECTED_BINDING_SCHEMA,
    }
    _icmt.EXPECTED_TRAINING_RUN_LEAF = EXPECTED_TRAINING_RUN_LEAF
    _icmt.EXPECTED_LV_EXPORT_SCHEMA = EXPECTED_LV_EXPORT_SCHEMA
    _icmt.EXPECTED_PAIR_SCHEMA = EXPECTED_PAIR_SCHEMA
    _icmt.EXPECTED_ICMT_RECEIPT_SCHEMA = EXPECTED_RCMMC_RECEIPT_SCHEMA
    _icmt.FROZEN_POSTFREEZE_CONTRACT = FROZEN_POSTFREEZE_CONTRACT
    _icmt._expected_final_checkpoint = _expected_final_checkpoint
    _icmt._validate_lv_payload = _validate_rcmmc_lv_payload
    _icmt._load_icmt_leo_binding = _load_rcmmc_leo_binding
    _icmt._recompute_prior_pair_artifacts = _recompute_rcmmc_prior_pair_artifacts
    _icmt._fold_gates = _rcmmc_fold_gates
    _icmt._matrix_aggregate = _rcmmc_matrix_aggregate
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
        raise RCMMCPostfreezePairError(f"postfreeze_matrix_id must be {EXPECTED_POSTFREEZE_MATRIX_ID}")
    output_root = Path(args.postfreeze_output_root).resolve()
    if output_root.name != EXPECTED_POSTFREEZE_MATRIX_ID or not output_root.is_dir():
        raise RCMMCPostfreezePairError("postfreeze output root does not bind frozen RCMMC matrix")
    training_root = _canonical_training_root(args.training_run_root)
    source_tx_ids = _icmt._cb._parse_items(args.source_tx_ids, field="source_tx_ids")
    fold_index = int(args.fold_index)
    if fold_index not in range(1, 7) or source_tx_ids != _icmt.FROZEN_FOLD_SOURCE_TX[fold_index]:
        raise RCMMCPostfreezePairError("RCMMC source TX/fold binding drifted")
    if str(args.candidate_pair) != f"F{fold_index}_C_vs_G":
        raise RCMMCPostfreezePairError("RCMMC candidate pair binding drifted")
    infos: dict[str, dict[str, Any]] = {}
    for arm, field in (("C", "c_final_checkpoint"), ("G", "g_final_checkpoint")):
        infos[arm] = _strict_current_checkpoint(getattr(args, field), training_root=training_root, fold_index=fold_index, arm=arm, source_tx_ids=source_tx_ids)
    common_binding = validate_rcmmc_common_training_binding(infos["C"]["receipt"], infos["G"]["receipt"])
    return training_root, source_tx_ids, fold_index, common_binding


def _atomic_rewrite_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".rcmmc.tmp")
    if temporary.exists():
        raise RCMMCPostfreezePairError(f"refusing to overwrite temporary pair JSON: {temporary}")
    temporary.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _pair_verdict(metrics: Mapping[str, Any]) -> str:
    aggregate = metrics.get("matrix_aggregate")
    if isinstance(aggregate, Mapping):
        return str(aggregate.get("verdict", "REJECT_P1_RCMMC_PERMANENT"))
    gates = metrics.get("postfreeze_gates")
    if not isinstance(gates, Mapping) or str(gates.get("fold_verdict", "")).startswith("REJECT"):
        return "REJECT_P1_RCMMC_PERMANENT"
    return "PENDING_MAIN_REVIEW_FULL_6_FOLD"


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate one same-fold RCMMC pair; F6 reopens raw sealed evidence."""

    training_root, source_tx_ids, fold_index, common_binding = _prevalidate_current_args(args)
    proxy_recomputation = {
        "C": _validate_proxy_logits_recompute(clean_npz=args.c_clean_npz, proxy_metrics_json=args.c_proxy_metrics_json, proxy_scores_csv=args.c_proxy_scores_csv, source_tx_ids=source_tx_ids, label="C"),
        "G": _validate_proxy_logits_recompute(clean_npz=args.g_clean_npz, proxy_metrics_json=args.g_proxy_metrics_json, proxy_scores_csv=args.g_proxy_scores_csv, source_tx_ids=source_tx_ids, label="G"),
    }
    try:
        with _patched_signed_fairness_kernel():
            metrics = dict(_icmt.evaluate(args))
    except (_icmt.ICMTPostfreezePairError, _icmt._cb.CBSFCEPostfreezePairError) as exc:
        raise _translate(exc) from exc
    receipt_binding: dict[str, Any] = {}
    for arm in ("C", "G"):
        info = _strict_current_checkpoint(getattr(args, f"{arm.lower()}_final_checkpoint"), training_root=training_root, fold_index=fold_index, arm=arm, source_tx_ids=source_tx_ids)
        receipt = info["receipt"]
        receipt_binding[arm] = {
            "candidate": info["candidate"], "final_checkpoint_sha256": info["sha256"],
            "raw_rcmmc_receipt_sha256": info["raw_receipt_sha256"],
            "terminal_contract": receipt["rcmmc_terminal_contract"], "terminal_contract_passed": True,
            "enabled": receipt["enabled"], "lambda": receipt["lambda"],
            "source_receiver_count": receipt["source_receiver_count"],
            "source_receiver_order_sha256": receipt["source_receiver_order_sha256"],
            "source_labeled_indices_sha256": receipt["source_labeled_indices_sha256"],
            "source_split_manifest_sha256": receipt["source_split_manifest_sha256"],
            "source_partition_sha256": receipt["source_partition_sha256"],
            "fixed_batch_size": receipt["frozen_batch_size"], "feature_dim": receipt["frozen_feature_dim"],
            "local_class_count": receipt["local_class_count"], "loss_global_denominator": receipt["loss_global_denominator"],
            "raw_vjp_once_required": arm == "G", "clean_and_exact_head_aux_vjp": "N_A_NONE_OR_ZERO_EXPECTED" if arm == "G" else "N_A",
            "leo_and_shared_encoder_aux_vjp": "FINITE_NONZERO_REQUIRED" if arm == "G" else "N_A",
        }
    metrics["rcmmc_training_receipt_revalidation"] = receipt_binding
    metrics["rcmmc_common_training_binding"] = common_binding
    metrics["rcmmc_proxy_logits_recomputation"] = proxy_recomputation
    metrics["rcmmc_f6_raw_reopen_required"] = True
    _validate_persisted_common_training_binding(metrics["rcmmc_common_training_binding"], common_binding, label=f"current F{fold_index}")
    metrics = dict(_strip_legacy_identity_fields(metrics))
    metrics["verdict"] = _pair_verdict(metrics)
    _require_no_legacy_identity_fields(metrics, label="RCMMC pair output")
    _atomic_rewrite_json(Path(args.output_metrics_json).resolve(), metrics)
    return metrics


def safe_totalized_l2_float64(features: Any, *, label: str = "RCMMC features") -> np.ndarray:
    """Pure evaluation map: mask before divide; zero rows remain exact zero."""

    try:
        return _icmt._normalize_float64(features, label=label)
    except _icmt.ICMTPostfreezePairError as exc:
        raise _translate(exc) from exc


normalize_rcmmc_float64 = safe_totalized_l2_float64


def fit_frozen_rcmmc_diagonal_gaussian(features: Any, labels: Any, source_tx_ids: Sequence[str]) -> dict[str, Any]:
    try:
        return _icmt.fit_frozen_diagonal_gaussian(features, labels, source_tx_ids)
    except _icmt.ICMTPostfreezePairError as exc:
        raise _translate(exc) from exc


def score_frozen_rcmmc_nll(features: Any, geometry: Mapping[str, Any]) -> np.ndarray:
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
