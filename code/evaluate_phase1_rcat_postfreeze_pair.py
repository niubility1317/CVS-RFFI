#!/usr/bin/env python
"""Final-only, source-paired P1-RCAT postfreeze closure.

The signed ICMT-v2 evaluator is reused only as a frozen fairness kernel for
float64 totalized-L2 Gaussian scoring, source/LEO metadata validation and F6
raw-artifact recomputation.  This facade owns every RCAT identity and reloads
the current raw RCAT terminal receipt for each C/G arm and every F1--F5 pair.
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

import export_phase1_rcat_features as _rcat_export
import export_phase1_rcat_leo_features as _rcat_leo
import evaluate_phase1_icmt_postfreeze_pair as _icmt


EXPECTED_TRAINING_RUN_LEAF = "phase1_rcat12_20260810_v1"
EXPECTED_POSTFREEZE_MATRIX_ID = "phase1_rcat_postfreeze_20260810_v1"
EXPECTED_LV_EXPORT_SCHEMA = "cvs.phase1.rcat_lv_export.v1"
EXPECTED_LEO_BINDING_SCHEMA = "cvs.phase1.rcat_leo_binding.v1"
EXPECTED_PAIR_SCHEMA = "cvs.phase1.rcat_postfreeze_pair.v1"
EXPECTED_RCAT_RECEIPT_SCHEMA = "cvs.phase1.rcat_receipt.v1"
EXPECTED_HEAD_CONTRACT = "dual_cvsincnet_tx_logits_v1"
EXPECTED_CLEAN_ARTIFACT = "icmt_clean_l_v_proxy_final_only.npz"

COMMON_TRAINING_BINDING_FIELDS = (
    "loss_rule",
    "loss_formula",
    "loss_global_denominator",
    "local_class_ids",
    "z_id_key",
    "feature_dimension_contract",
    "totalized_l2_rule",
    "training_accumulation_dtype",
    "postfreeze_totalized_l2_dtype",
    "clean_feature_detached",
    "same_physical_pairing",
    "common_lambda_sat_cons",
    "common_sat_kl",
    "head_input_path",
    "common_l_base_head_input_path_verified",
    "aux_gradient_scope",
    "uses_new_forward",
    "uses_resampling",
    "uses_rx_labels",
    "rx_permission",
    "rx_metadata_allowlist",
    "no_day_assertion",
    "uses_day_labels",
    "uses_domain_labels",
    "uses_target_rows",
    "uses_proxy_rows",
    "uses_held_rows",
    "uses_unlabeled_rows",
    "uses_ema_or_state",
    "uses_threshold",
    "uses_gradient_projection",
    "uses_cross_sample_pairing",
    "uses_cross_receiver_pairing",
    "warm_start_mode",
    "baseline_path",
    "baseline_sha256",
    "initial_checkpoint_sha256",
    "checkpoint_epoch",
    "checkpoint_role",
    "strict_model_keys",
    "missing_model_keys",
    "unexpected_model_keys",
    "optimizer_state_restored",
    "rng_state_restored",
    "amp_contract",
    "source_train_tx",
    "source_known_validation_tx",
    "source_proxy_unknown_tx",
    "source_partition_sha256",
    "class_order_binding_sha256",
    "source_labeled_indices_sha256",
    "source_split_manifest_sha256",
    "source_receiver_ids_sha256",
    "source_receiver_provenance",
    "source_receiver_ids",
    "source_receiver_count",
    "frozen_source_receiver_ids",
    "frozen_cells_per_scene",
    "dataset_tx_class_order",
    "local_tx_class_order",
    "checkpoint_train_tx_class_order",
    "local_to_dataset_class_ids",
    "local_to_head_class_ids",
    "expected_tx_class_ids",
    "dataset_class_count",
    "local_data_class_count",
    "checkpoint_head_class_count",
    "live_head_class_count",
    "optimizer_type",
    "optimizer_initial_state_sha256",
    "optimizer_initial_state_empty",
    "common_batch_sequence_sha256",
    "common_batch_sequence_batches",
    "common_batch_sequence_rows",
    "common_scenario_batches",
    "rcat_common_cells",
    "rcat_common_batch_cells",
)
COMMON_TRAINING_SHA_FIELDS = (
    "baseline_sha256",
    "initial_checkpoint_sha256",
    "source_partition_sha256",
    "class_order_binding_sha256",
    "source_labeled_indices_sha256",
    "source_split_manifest_sha256",
    "source_receiver_ids_sha256",
    "optimizer_initial_state_sha256",
    "common_batch_sequence_sha256",
)
G_ONLY_RECEIPT_FIELDS = (
    "rcat_batches",
    "rcat_total_rows",
    "rcat_positive_q",
    "rcat_loss_sum",
    "rcat_scenes",
    "rcat_g_batch_aux",
    "rcat_gradient_audit_attempted",
    "rcat_gradient_audit_completed",
    "rcat_gradient_audit",
)
FROZEN_POSTFREEZE_CONTRACT = {
    "RCAT-PF-01": "final-only RCAT z_id=feat_joint with L-only diagonal Gaussian fit",
    "RCAT-PF-02": "float64 totalized-L2 uses the training float32 piecewise rule without claiming byte identity, retains exact zero rows and rejects non-finite features",
    "RCAT-PF-03": "ddof1 class-equal pooled variance with 0.9/0.1 shrink, 1e-6 floor and stable full NLL",
    "RCAT-PF-04": "V/proxy contribute no fit rows while every L/V/proxy row remains in the evidence",
    "RCAT-PF-05": "fixed proxy days/RXs/seed/max-per-TX/total=400 binds NPZ, physical keys, JSON and CSV",
    "RCAT-PF-06": "raw C/G checkpoints bind warm-start/head/class/order/split/new AdamW and source-RX provenance; C aux is N/A-or-zero while G proves positive-q and feat_joint/shared-encoder VJP with exact-head aux None-or-zero",
    "RCAT-PF-07": "common physical/RX/class/scene n_rc and batch order close at 28 cells per scene and 84 terminal cells",
    "RCAT-PF-08": "F6 re-reads F1--F5 clean/LEO/binding/proxy JSON+CSV and recomputes summary, deltas and gates",
    "RCAT-PF-09": "clean6, LEO18, fold/global overall and two strict proxy gates are non-compensating",
}


class RCATPostfreezePairError(RuntimeError):
    """Raised when RCAT postfreeze evidence cannot close fail-closed."""


def _translate(error: BaseException) -> RCATPostfreezePairError:
    return RCATPostfreezePairError(str(error))


_LOGITS_REJECT_MODULE: Any | None = None


def _logits_reject_module() -> Any:
    """Load the frozen logits proxy scorer without adding an editable adapter."""

    global _LOGITS_REJECT_MODULE
    if _LOGITS_REJECT_MODULE is not None:
        return _LOGITS_REJECT_MODULE
    source = Path(__file__).resolve().parent / "scripts" / "eval_phase1_logits_open_set_reject.py"
    spec = importlib.util.spec_from_file_location("_rcat_frozen_logits_proxy", source)
    if spec is None or spec.loader is None:
        raise RCATPostfreezePairError("cannot load frozen logits proxy scorer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _LOGITS_REJECT_MODULE = module
    return module


def _close_proxy_value(actual: Any, expected: Any, *, field: str, label: str) -> None:
    if actual is None or expected is None:
        if actual is not expected:
            raise RCATPostfreezePairError(f"{label} proxy {field} does not match raw logits")
        return
    try:
        left = float(actual)
        right = float(expected)
    except (TypeError, ValueError) as exc:
        raise RCATPostfreezePairError(f"{label} proxy {field} is not numeric") from exc
    if not math.isfinite(left) or not math.isfinite(right) or not math.isclose(
        left, right, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise RCATPostfreezePairError(f"{label} proxy {field} does not match raw logits")


def _validate_proxy_logits_recompute(
    *,
    clean_npz: str | Path,
    proxy_metrics_json: str | Path,
    proxy_scores_csv: str | Path,
    source_tx_ids: Sequence[str],
    label: str,
) -> dict[str, Any]:
    """Recompute frozen logits JSON and CSV from the current clean NPZ bytes."""

    try:
        observed = json.loads(Path(proxy_metrics_json).read_text(encoding="utf-8"))
    except Exception as exc:
        raise RCATPostfreezePairError(f"{label} proxy diagnostic JSON is invalid") from exc
    if not isinstance(observed, Mapping):
        raise RCATPostfreezePairError(f"{label} proxy diagnostic JSON must encode an object")
    scorer = _logits_reject_module()
    with tempfile.TemporaryDirectory(prefix="rcat_proxy_recompute_") as temporary:
        expected_csv = Path(temporary) / "scores.csv"
        recomputed = scorer.evaluate(
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
            "AUROC_unknown",
            "unknown_FAR",
            "unknown_reject_rate",
            "known_closed_accuracy_no_reject",
            "known_coverage",
            "known_full_accuracy_after_reject",
            "known_accepted_accuracy",
            "old_retention_vs_closed",
        ):
            _close_proxy_value(
                observed.get(field), recomputed.get(field), field=field, label=label
            )
        for field in ("known_query_count", "unknown_query_count"):
            if type(observed.get(field)) is not int or observed.get(field) != recomputed.get(field):
                raise RCATPostfreezePairError(
                    f"{label} proxy {field} does not match raw logits"
                )
        if observed.get("gate_policy") != recomputed.get("gate_policy"):
            raise RCATPostfreezePairError(f"{label} proxy gate policy does not match raw logits")
        observed_calibration = observed.get("calibration")
        expected_calibration = recomputed.get("calibration")
        if not isinstance(observed_calibration, Mapping) or not isinstance(expected_calibration, Mapping):
            raise RCATPostfreezePairError(f"{label} proxy calibration is malformed")
        if set(observed_calibration) != set(expected_calibration):
            raise RCATPostfreezePairError(f"{label} proxy calibration keys drifted")
        for field in sorted(expected_calibration):
            _close_proxy_value(
                observed_calibration.get(field),
                expected_calibration.get(field),
                field=f"calibration.{field}",
                label=label,
            )
        try:
            with Path(proxy_scores_csv).open("r", encoding="utf-8", newline="") as handle:
                actual_reader = csv.DictReader(handle)
                actual_fields = tuple(actual_reader.fieldnames or ())
                actual_rows = list(actual_reader)
            with expected_csv.open("r", encoding="utf-8", newline="") as handle:
                expected_reader = csv.DictReader(handle)
                expected_fields = tuple(expected_reader.fieldnames or ())
                expected_rows = list(expected_reader)
        except Exception as exc:
            raise RCATPostfreezePairError(f"{label} proxy score CSV is invalid") from exc
        if actual_fields != expected_fields or actual_rows != expected_rows:
            raise RCATPostfreezePairError(f"{label} proxy score CSV does not match raw logits")
    return {
        "passed": True,
        "clean_npz_sha256": _icmt._cb._sha256_file(clean_npz),
        "proxy_metrics_json_sha256": _icmt._cb._sha256_file(proxy_metrics_json),
        "proxy_scores_csv_sha256": _icmt._cb._sha256_file(proxy_scores_csv),
    }


def _canonical_training_root(value: str | Path) -> Path:
    root = Path(value).resolve()
    if root.name != EXPECTED_TRAINING_RUN_LEAF or not root.is_dir():
        raise RCATPostfreezePairError(
            f"training run root must be existing {EXPECTED_TRAINING_RUN_LEAF}: {root}"
        )
    return root


def _expected_final_checkpoint(training_root: Path, fold_index: int, arm: str) -> tuple[str, Path]:
    if fold_index not in range(1, 7) or arm not in {"C", "G"}:
        raise RCATPostfreezePairError("unsupported frozen RCAT fold/arm")
    candidate = f"F{fold_index}{arm}_RCAT12"
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
        raise RCATPostfreezePairError(
            f"{arm} final checkpoint path does not bind frozen {candidate}"
        )
    checkpoint = torch.load(observed, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise RCATPostfreezePairError(f"{arm} checkpoint payload must be a mapping")
    try:
        _, receipt, observed_arm = _rcat_export.validate_rcat_training_checkpoint(
            checkpoint,
            checkpoint_path=observed,
            source_tx_ids=source_tx_ids,
            known_validation_tx_ids=(
                _icmt.FROZEN_FOLD_KNOWN_HELDOUT_TX[fold_index],
            ),
            proxy_unknown_tx_ids=(
                _icmt.FROZEN_FOLD_PROXY_TX[fold_index],
            ),
        )
    except _rcat_export.RCATSplitExportError as exc:
        raise _translate(exc) from exc
    if observed_arm != arm:
        raise RCATPostfreezePairError(f"{arm} checkpoint receipt arm drifted")
    raw_receipt = checkpoint.get("rcat_receipt")
    if not isinstance(raw_receipt, Mapping):
        raise RCATPostfreezePairError(f"{arm} checkpoint lacks raw rcat_receipt")
    return {
        "candidate": candidate,
        "path": observed,
        "sha256": _icmt._cb._sha256_file(observed),
        "receipt": dict(receipt),
        "raw_receipt_sha256": _rcat_export._canonical_json_sha256(dict(raw_receipt)),
    }


def _strict_common_training_projection(
    receipt: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    """Extract the exact C/G-common RCAT training projection."""

    if not isinstance(receipt, Mapping):
        raise RCATPostfreezePairError(f"{label} common training receipt is not a mapping")
    missing = [field for field in COMMON_TRAINING_BINDING_FIELDS if field not in receipt]
    if missing:
        raise RCATPostfreezePairError(
            f"{label} common training binding lacks fields: {','.join(missing)}"
        )
    projection = {field: receipt[field] for field in COMMON_TRAINING_BINDING_FIELDS}
    expected_literals = {
        "loss_rule": "SOURCE_L_RX_BY_LOCAL4_EQUAL_WEIGHT_STOPGRAD_CLEAN_TO_LEO_TOTALIZED_L2_feat_joint",
        "loss_formula": "T(z)=z/||z||2_if_norm_gt_0_else_0;q=||T(z_leo)-sg(T(z_clean))||2;g_rc=0_if_n_rc=0_else_mean_Irc(q);L=sum_rc(g_rc)/28",
        "z_id_key": "feat_joint",
        "feature_dimension_contract": "RAW_ENCODER_feat_joint_EXACT_HEAD_INPUT_DIMENSION_BOUND",
        "totalized_l2_rule": "T(z)=z/||z||2_IF_norm_gt_0_ELSE_0",
        "training_accumulation_dtype": "float32",
        "postfreeze_totalized_l2_dtype": "float64_SAME_PIECEWISE_RULE_NOT_BYTE_IDENTICAL",
        "same_physical_pairing": "SAME_SOURCE_L_PHYSICAL_ROW_COMMON_CLEAN_AND_SINGLE_LEO_FORWARD",
        "common_sat_kl": "sg(clean_tx_logits)_TO_leo_tx_logits",
        "head_input_path": "model_output.tx_logits_from_id_backbone.cls_head.head(feat_joint)",
        "aux_gradient_scope": "LEO_feat_joint_AND_SHARED_ENCODER_FINITE_NONZERO;EXACT_HEAD_AUX_VJP_NA_NONE_OR_ZERO",
        "rx_permission": "SOURCE_KNOWN_TRAIN_L_PHYSICAL_ID_BOUND_rx_i_ONLY",
        "no_day_assertion": "day_i_NOT_READ_BY_RCAT",
        "warm_start_mode": "MODEL_WEIGHTS_ONLY_NEW_ADAMW_AMP",
        "checkpoint_role": "training_final_only",
        "amp_contract": "COMMON_TRAINER_AMP_ENABLED",
    }
    for field, expected in expected_literals.items():
        if type(projection[field]) is not str or projection[field] != expected:
            raise RCATPostfreezePairError(
                f"{label} common training binding {field} drifted"
            )
    expected_bools = {
        "clean_feature_detached": True,
        "common_l_base_head_input_path_verified": True,
        "uses_new_forward": False,
        "uses_resampling": False,
        "uses_rx_labels": True,
        "uses_day_labels": False,
        "uses_domain_labels": False,
        "uses_target_rows": False,
        "uses_proxy_rows": False,
        "uses_held_rows": False,
        "uses_unlabeled_rows": False,
        "uses_ema_or_state": False,
        "uses_threshold": False,
        "uses_gradient_projection": False,
        "uses_cross_sample_pairing": False,
        "uses_cross_receiver_pairing": False,
        "strict_model_keys": True,
        "optimizer_state_restored": False,
        "rng_state_restored": False,
    }
    for field, expected in expected_bools.items():
        if type(projection[field]) is not bool or projection[field] is not expected:
            raise RCATPostfreezePairError(
                f"{label} common training binding {field} drifted"
            )
    if (
        type(projection["loss_global_denominator"]) is not int
        or projection["loss_global_denominator"] != _rcat_export.FROZEN_CELLS_PER_SCENE
    ):
        raise RCATPostfreezePairError(
            f"{label} common training binding fixed divisor drifted"
        )
    try:
        common_lambda_sat_cons = float(projection["common_lambda_sat_cons"])
    except (TypeError, ValueError) as exc:
        raise RCATPostfreezePairError(
            f"{label} common training binding common_lambda_sat_cons is invalid"
        ) from exc
    if not math.isfinite(common_lambda_sat_cons) or abs(common_lambda_sat_cons - 0.10) > 1e-12:
        raise RCATPostfreezePairError(
            f"{label} common training binding common_lambda_sat_cons drifted"
        )
    if projection["local_class_ids"] != [0, 1, 2, 3] or type(projection["local_class_ids"]) is not list:
        raise RCATPostfreezePairError(
            f"{label} common training binding local_class_ids drifted"
        )
    if projection["rx_metadata_allowlist"] != ["rx_i"] or type(projection["rx_metadata_allowlist"]) is not list:
        raise RCATPostfreezePairError(
            f"{label} common training binding RX metadata allowlist drifted"
        )
    if type(projection["baseline_path"]) is not str or not projection["baseline_path"]:
        raise RCATPostfreezePairError(
            f"{label} common training binding baseline_path is empty"
        )
    if type(projection["checkpoint_epoch"]) is not int or projection["checkpoint_epoch"] < 0:
        raise RCATPostfreezePairError(
            f"{label} common training binding checkpoint_epoch is invalid"
        )
    if projection["missing_model_keys"] != [] or projection["unexpected_model_keys"] != []:
        raise RCATPostfreezePairError(
            f"{label} common training binding strict model keys drifted"
        )
    for field in COMMON_TRAINING_SHA_FIELDS:
        if type(projection[field]) is not str:
            raise RCATPostfreezePairError(
                f"{label} common training binding {field} must be a string"
            )
        try:
            _rcat_export._require_sha256(
                projection[field], field=f"{label} common training binding {field}"
            )
        except _rcat_export.RCATSplitExportError as exc:
            raise _translate(exc) from exc
    if projection["source_receiver_provenance"] != _rcat_export.SOURCE_RECEIVER_PROVENANCE:
        raise RCATPostfreezePairError(f"{label} common training binding source receiver provenance drifted")
    for field in ("source_receiver_ids", "frozen_source_receiver_ids"):
        if type(projection[field]) is not list or tuple(projection[field]) != _rcat_export.FROZEN_SOURCE_RECEIVER_IDS:
            raise RCATPostfreezePairError(f"{label} common training binding {field} drifted")
    for field, expected in (
        ("source_receiver_count", _rcat_export.FROZEN_SOURCE_RECEIVER_COUNT),
        ("frozen_cells_per_scene", _rcat_export.FROZEN_CELLS_PER_SCENE),
    ):
        if type(projection[field]) is not int or projection[field] != expected:
            raise RCATPostfreezePairError(f"{label} common training binding {field} drifted")
    for field in (
        "dataset_tx_class_order",
        "local_tx_class_order",
        "checkpoint_train_tx_class_order",
        "source_train_tx",
        "source_known_validation_tx",
        "source_proxy_unknown_tx",
        "local_to_dataset_class_ids",
        "local_to_head_class_ids",
        "expected_tx_class_ids",
    ):
        if type(projection[field]) is not list:
            raise RCATPostfreezePairError(f"{label} common training binding {field} must be a list")
    for field in (
        "dataset_class_count",
        "local_data_class_count",
        "checkpoint_head_class_count",
        "live_head_class_count",
        "common_batch_sequence_batches",
        "common_batch_sequence_rows",
    ):
        if type(projection[field]) is not int or projection[field] <= 0:
            raise RCATPostfreezePairError(
                f"{label} common training binding {field} must be a positive integer"
            )
    if type(projection["optimizer_type"]) is not str or projection["optimizer_type"] != "AdamW":
        raise RCATPostfreezePairError(
            f"{label} common training binding optimizer_type must be literal AdamW"
        )
    if projection["optimizer_initial_state_empty"] is not True:
        raise RCATPostfreezePairError(
            f"{label} common training binding optimizer_initial_state_empty must be literal True"
        )
    scenarios = projection["common_scenario_batches"]
    expected_scenarios = tuple(_rcat_export._rcat.FROZEN_RCAT_SCENARIOS)
    if type(scenarios) is not dict or set(scenarios) != set(expected_scenarios):
        raise RCATPostfreezePairError(
            f"{label} common training binding common_scenario_batches keys drifted"
        )
    normalized_scenarios: dict[str, int] = {}
    for scenario in expected_scenarios:
        value = scenarios[scenario]
        if type(value) is not int or value <= 0:
            raise RCATPostfreezePairError(
                f"{label} common training binding scenario count must be a positive integer"
            )
        normalized_scenarios[scenario] = value
    if sum(normalized_scenarios.values()) != projection["common_batch_sequence_batches"]:
        raise RCATPostfreezePairError(
            f"{label} common training binding scenario batches do not close"
        )
    projection["common_scenario_batches"] = normalized_scenarios
    common_cells = projection["rcat_common_cells"]
    if type(common_cells) is not dict or set(common_cells) != set(expected_scenarios):
        raise RCATPostfreezePairError(f"{label} common training binding rcat_common_cells drifted")
    for scenario in expected_scenarios:
        cells = common_cells[scenario]
        if type(cells) is not dict or len(cells) != _rcat_export.FROZEN_CELLS_PER_SCENE:
            raise RCATPostfreezePairError(
                f"{label} common training binding lacks 28 receiver/class cells"
            )
    batch_cells = projection["rcat_common_batch_cells"]
    if type(batch_cells) is not list or len(batch_cells) != projection["common_batch_sequence_batches"]:
        raise RCATPostfreezePairError(
            f"{label} common training binding batch n_rc receipt is incomplete"
        )
    return projection


def validate_rcat_common_training_binding(
    c_receipt: Mapping[str, Any], g_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail unless C/G share exact warm-start, RX/class cells, data and AdamW."""

    c_projection = _strict_common_training_projection(c_receipt, label="C")
    g_projection = _strict_common_training_projection(g_receipt, label="G")
    if set(c_projection) != set(COMMON_TRAINING_BINDING_FIELDS) or set(g_projection) != set(
        COMMON_TRAINING_BINDING_FIELDS
    ):
        raise RCATPostfreezePairError("C/G common training binding key set drifted")
    for field in COMMON_TRAINING_BINDING_FIELDS:
        if type(c_projection[field]) is not type(g_projection[field]) or c_projection[field] != g_projection[field]:
            raise RCATPostfreezePairError(
                f"C/G common training binding {field} differs"
            )
    return {
        "passed": True,
        "fields": dict(c_projection),
        "sha256": _rcat_export._canonical_json_sha256(c_projection),
    }


def _validate_persisted_common_training_binding(
    observed: Any, expected: Mapping[str, Any], *, label: str
) -> None:
    """Compare a persisted pair receipt with a fresh raw C/G recomputation."""

    if type(observed) is not dict or set(observed) != {"passed", "fields", "sha256"}:
        raise RCATPostfreezePairError(
            f"{label} persisted common training binding key set drifted"
        )
    if observed["passed"] is not True:
        raise RCATPostfreezePairError(
            f"{label} persisted common training binding did not pass"
        )
    fields = observed["fields"]
    if type(fields) is not dict or set(fields) != set(COMMON_TRAINING_BINDING_FIELDS):
        raise RCATPostfreezePairError(
            f"{label} persisted common training binding fields drifted"
        )
    normalized = _strict_common_training_projection(fields, label=label)
    expected_fields = expected.get("fields")
    if type(expected_fields) is not dict:
        raise RCATPostfreezePairError("internal expected common training binding is malformed")
    for field in COMMON_TRAINING_BINDING_FIELDS:
        if type(normalized[field]) is not type(expected_fields.get(field)) or normalized[field] != expected_fields.get(field):
            raise RCATPostfreezePairError(
                f"{label} persisted common training binding {field} does not match raw receipts"
            )
    digest = _rcat_export._canonical_json_sha256(normalized)
    if type(observed["sha256"]) is not str or observed["sha256"] != digest or digest != expected.get("sha256"):
        raise RCATPostfreezePairError(
            f"{label} persisted common training binding SHA256 does not match raw receipts"
        )


def _require_exact_manifest_fields(
    manifest: Mapping[str, Any],
    *,
    checkpoint_info: Mapping[str, Any],
    label: str,
) -> None:
    legacy_identity_fields = sorted(
        str(field) for field in manifest if str(field).startswith("icmt_")
    )
    if legacy_identity_fields:
        raise RCATPostfreezePairError(
            f"{label} manifest contains legacy ICMT identity fields: "
            + ",".join(legacy_identity_fields)
        )
    receipt = checkpoint_info["receipt"]
    expected = {
        "schema": EXPECTED_LV_EXPORT_SCHEMA,
        "method": "P1_RCAT",
        "training_run_contract": EXPECTED_TRAINING_RUN_LEAF,
        "rcat_receipt_schema": EXPECTED_RCAT_RECEIPT_SCHEMA,
        "rcat_enabled": str(checkpoint_info["candidate"])[2] == "G",
        "rcat_receipt_sha256": checkpoint_info["raw_receipt_sha256"],
        "rcat_terminal_contract": str(receipt["rcat_terminal_contract"]),
        "rcat_terminal_contract_passed": True,
        "rcat_lambda": _rcat_export.FROZEN_RCAT_LAMBDA
        if str(checkpoint_info["candidate"])[2] == "G"
        else 0.0,
        "rcat_loss_global_denominator": _rcat_export.FROZEN_CELLS_PER_SCENE,
        "rcat_totalized_l2_rule": str(receipt["totalized_l2_rule"]),
        "rcat_training_accumulation_dtype": str(receipt["training_accumulation_dtype"]),
        "rcat_postfreeze_totalized_l2_dtype": str(receipt["postfreeze_totalized_l2_dtype"]),
        "rcat_clean_feature_detached": True,
        "rcat_common_l_base_head_input_path_verified": True,
        "rcat_exact_head_aux_vjp": "N_A_NONE_OR_ZERO_EXPECTED",
        "rcat_source_receiver_ids": list(_rcat_export.FROZEN_SOURCE_RECEIVER_IDS),
        "rcat_source_receiver_count": _rcat_export.FROZEN_SOURCE_RECEIVER_COUNT,
        "rcat_source_receiver_ids_sha256": str(receipt["source_receiver_ids_sha256"]),
        "rcat_source_receiver_provenance": _rcat_export.SOURCE_RECEIVER_PROVENANCE,
        "rcat_frozen_cells_per_scene": _rcat_export.FROZEN_CELLS_PER_SCENE,
        "rcat_common_physical_rx_class_scene_nrc_bound": True,
        "rcat_batch_order_bound": True,
        "proxy_selection_frozen_not_cli_tunable": True,
    }
    for field, wanted in expected.items():
        if manifest.get(field) != wanted or type(manifest.get(field)) is not type(wanted):
            raise RCATPostfreezePairError(f"{label} manifest {field} drifted")
    if str(manifest.get("candidate_id", "")) != str(checkpoint_info["candidate"]):
        raise RCATPostfreezePairError(f"{label} manifest RCAT candidate binding drifted")
    if Path(str(manifest.get("checkpoint", ""))).resolve() != checkpoint_info["path"]:
        raise RCATPostfreezePairError(f"{label} manifest final checkpoint path drifted")
    if str(manifest.get("source_checkpoint_sha256", "")) != checkpoint_info["sha256"]:
        raise RCATPostfreezePairError(f"{label} manifest final checkpoint SHA256 drifted")
    for field, expected_receipt in (
        ("rcat_source_labeled_indices_sha256", receipt["source_labeled_indices_sha256"]),
        ("rcat_source_split_manifest_sha256", receipt["source_split_manifest_sha256"]),
    ):
        if str(manifest.get(field, "")) != str(expected_receipt):
            raise RCATPostfreezePairError(f"{label} manifest {field} does not bind raw receipt")


_ORIGINAL_VALIDATE_LV_PAYLOAD = _icmt._validate_lv_payload
_ORIGINAL_LOAD_LEO_BINDING = _icmt._load_icmt_leo_binding
_ORIGINAL_RECOMPUTE_PRIOR = _icmt._recompute_prior_pair_artifacts
_ORIGINAL_MATRIX_AGGREGATE = _icmt._matrix_aggregate
_ORIGINAL_FOLD_GATES = _icmt._fold_gates


def _validate_rcat_lv_payload(
    payload: Mapping[str, Any],
    source_tx_ids: Sequence[str],
    fold_index: int,
    expected_proxy_count: int,
    *,
    label: str,
) -> dict[str, Any]:
    """Validate RCAT fields then invoke the signed generic L/V validator."""

    manifest = payload.get("manifest")
    if not isinstance(manifest, Mapping):
        raise RCATPostfreezePairError(f"{label} lacks RCAT L/V manifest")
    arm = "C" if label.startswith("C") else "G" if label.startswith("G") else ""
    if not arm:
        raise RCATPostfreezePairError(f"{label} does not identify a C/G arm")
    training_root_value = Path(str(manifest.get("checkpoint", ""))).resolve().parents[1]
    candidate, checkpoint_path = _expected_final_checkpoint(
        training_root_value, fold_index, arm
    )
    if checkpoint_path != Path(str(manifest.get("checkpoint", ""))).resolve():
        raise RCATPostfreezePairError(f"{label} checkpoint layout does not bind candidate")
    checkpoint_info = _strict_current_checkpoint(
        checkpoint_path,
        training_root=training_root_value,
        fold_index=fold_index,
        arm=arm,
        source_tx_ids=source_tx_ids,
    )
    _require_exact_manifest_fields(manifest, checkpoint_info=checkpoint_info, label=label)
    compatibility_manifest = dict(manifest)
    compatibility_manifest.update(
        {
            "candidate_id": f"F{fold_index}{arm}_ICMT12",
            "icmt_receipt_schema": EXPECTED_RCAT_RECEIPT_SCHEMA,
            "icmt_enabled": arm == "G",
            "icmt_source_labeled_indices_sha256": manifest[
                "rcat_source_labeled_indices_sha256"
            ],
            "icmt_source_split_manifest_sha256": manifest[
                "rcat_source_split_manifest_sha256"
            ],
        }
    )
    compatibility_payload = dict(payload)
    compatibility_payload["manifest"] = compatibility_manifest
    try:
        return _ORIGINAL_VALIDATE_LV_PAYLOAD(
            compatibility_payload,
            source_tx_ids,
            fold_index,
            expected_proxy_count,
            label=label,
        )
    except _icmt.ICMTPostfreezePairError as exc:
        raise _translate(exc) from exc


def _load_rcat_leo_binding(
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
        raise RCATPostfreezePairError(f"{label} RCAT LEO binding JSON is invalid") from exc
    if not isinstance(binding, Mapping):
        raise RCATPostfreezePairError(f"{label} RCAT LEO binding must encode an object")
    checkpoint_info = _strict_current_checkpoint(
        expected_checkpoint,
        training_root=training_root,
        fold_index=fold_index,
        arm=arm,
        source_tx_ids=source_tx_ids,
    )
    receipt = checkpoint_info["receipt"]
    expected_extra = {
        "schema": EXPECTED_LEO_BINDING_SCHEMA,
        "method": "P1_RCAT",
        "rcat_receipt_schema": EXPECTED_RCAT_RECEIPT_SCHEMA,
        "rcat_receipt_sha256": checkpoint_info["raw_receipt_sha256"],
        "rcat_terminal_contract": str(receipt["rcat_terminal_contract"]),
        "rcat_terminal_contract_passed": True,
        "rcat_lambda": _rcat_export.FROZEN_RCAT_LAMBDA if arm == "G" else 0.0,
        "rcat_loss_global_denominator": _rcat_export.FROZEN_CELLS_PER_SCENE,
        "rcat_totalized_l2_rule": str(receipt["totalized_l2_rule"]),
        "rcat_training_accumulation_dtype": str(receipt["training_accumulation_dtype"]),
        "rcat_postfreeze_totalized_l2_dtype": str(receipt["postfreeze_totalized_l2_dtype"]),
        "rcat_clean_feature_detached": True,
        "rcat_common_l_base_head_input_path_verified": True,
        "rcat_exact_head_aux_vjp": "N_A_NONE_OR_ZERO_EXPECTED",
        "rcat_source_receiver_ids": list(_rcat_export.FROZEN_SOURCE_RECEIVER_IDS),
        "rcat_source_receiver_count": _rcat_export.FROZEN_SOURCE_RECEIVER_COUNT,
        "rcat_source_receiver_ids_sha256": str(receipt["source_receiver_ids_sha256"]),
        "rcat_source_receiver_provenance": _rcat_export.SOURCE_RECEIVER_PROVENANCE,
        "rcat_frozen_cells_per_scene": _rcat_export.FROZEN_CELLS_PER_SCENE,
        "rcat_common_physical_rx_class_scene_nrc_bound": True,
        "rcat_batch_order_bound": True,
    }
    for field, wanted in expected_extra.items():
        if binding.get(field) != wanted or type(binding.get(field)) is not type(wanted):
            raise RCATPostfreezePairError(f"{label} RCAT LEO binding {field} drifted")
    return dict(result)


def _recompute_rcat_prior_pair_artifacts(
    record: Mapping[str, Any],
    *,
    output_root: Path,
    matrix_id: str,
    training_root: Path,
    expected_scenarios: Sequence[str],
) -> dict[str, Any]:
    """Rebuild F1--F5 evidence from raw current artifacts, not pair summaries."""

    fold_index = int(record.get("fold_index", -1))
    if fold_index not in range(1, 7):
        raise RCATPostfreezePairError("prior pair fold is invalid")
    source_tx_ids = _icmt.FROZEN_FOLD_SOURCE_TX[fold_index]
    checkpoint_infos: dict[str, dict[str, Any]] = {}
    for arm in ("C", "G"):
        _, checkpoint = _expected_final_checkpoint(training_root, fold_index, arm)
        checkpoint_infos[arm] = _strict_current_checkpoint(
            checkpoint,
            training_root=training_root,
            fold_index=fold_index,
            arm=arm,
            source_tx_ids=source_tx_ids,
        )
    common_binding = validate_rcat_common_training_binding(
        checkpoint_infos["C"]["receipt"], checkpoint_infos["G"]["receipt"]
    )
    _validate_persisted_common_training_binding(
        record.get("rcat_common_training_binding"),
        common_binding,
        label=f"prior F{fold_index}",
    )
    bindings = record.get("bindings")
    if not isinstance(bindings, Mapping):
        raise RCATPostfreezePairError("prior pair lacks raw artifact bindings")
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
    result["rcat_common_training_binding"] = common_binding
    return result


def _rcat_fold_gates(
    clean_delta: Mapping[str, Any],
    leo_scenarios: Mapping[str, Mapping[str, Any]],
    proxy_guardrail: Mapping[str, Any],
    expected_scenarios: Sequence[str],
) -> dict[str, Any]:
    result = dict(
        _ORIGINAL_FOLD_GATES(
            clean_delta, leo_scenarios, proxy_guardrail, expected_scenarios
        )
    )
    if result.get("fold_verdict") == "REJECT_P1_ICMT_PERMANENT":
        result["fold_verdict"] = "REJECT_P1_RCAT_PERMANENT"
    elif result.get("fold_verdict") == "PENDING_GLOBAL_18_GRID":
        result["fold_verdict"] = "PENDING_MAIN_REVIEW_FULL_6_FOLD"
    return result


def _rcat_matrix_aggregate(
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
    checkpoint_infos: dict[str, dict[str, Any]] = {}
    for arm in ("C", "G"):
        _, checkpoint = _expected_final_checkpoint(training_root, fold_index, arm)
        checkpoint_infos[arm] = _strict_current_checkpoint(
            checkpoint,
            training_root=training_root,
            fold_index=fold_index,
            arm=arm,
            source_tx_ids=source_tx_ids,
        )
    common_binding = validate_rcat_common_training_binding(
        checkpoint_infos["C"]["receipt"], checkpoint_infos["G"]["receipt"]
    )
    if not isinstance(current, dict):
        raise RCATPostfreezePairError("current pair must be mutable for RCAT binding receipt")
    current["rcat_common_training_binding"] = common_binding
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
    if result.get("verdict") == "REJECT_P1_ICMT_PERMANENT":
        result["verdict"] = "REJECT_P1_RCAT_PERMANENT"
    else:
        result["verdict"] = "PENDING_MAIN_REVIEW"
    return result


@contextmanager
def _patched_signed_fairness_kernel() -> Iterator[None]:
    """Retain signed math while replacing every ICMT identity with RCAT."""

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
    _icmt.EXPECTED_ICMT_RECEIPT_SCHEMA = EXPECTED_RCAT_RECEIPT_SCHEMA
    _icmt.FROZEN_POSTFREEZE_CONTRACT = FROZEN_POSTFREEZE_CONTRACT
    _icmt._expected_final_checkpoint = _expected_final_checkpoint
    _icmt._validate_lv_payload = _validate_rcat_lv_payload
    _icmt._load_icmt_leo_binding = _load_rcat_leo_binding
    _icmt._recompute_prior_pair_artifacts = _recompute_rcat_prior_pair_artifacts
    _icmt._fold_gates = _rcat_fold_gates
    _icmt._matrix_aggregate = _rcat_matrix_aggregate
    _icmt._icmt_leo.EXPECTED_TRAINING_RUN_LEAF = EXPECTED_TRAINING_RUN_LEAF
    _icmt._icmt_leo.EXPECTED_BINDING_SCHEMA = EXPECTED_LEO_BINDING_SCHEMA
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(_icmt, name, value)
        for name, value in leo_saved.items():
            setattr(_icmt._icmt_leo, name, value)


def _prevalidate_current_args(
    args: argparse.Namespace,
) -> tuple[Path, tuple[str, ...], int, dict[str, Any]]:
    matrix_id = str(args.postfreeze_matrix_id).strip()
    if matrix_id != EXPECTED_POSTFREEZE_MATRIX_ID:
        raise RCATPostfreezePairError(
            f"postfreeze_matrix_id must be {EXPECTED_POSTFREEZE_MATRIX_ID}"
        )
    output_root = Path(args.postfreeze_output_root).resolve()
    if output_root.name != EXPECTED_POSTFREEZE_MATRIX_ID or not output_root.is_dir():
        raise RCATPostfreezePairError("postfreeze output root does not bind frozen RCAT matrix")
    training_root = _canonical_training_root(args.training_run_root)
    source_tx_ids = _icmt._cb._parse_items(args.source_tx_ids, field="source_tx_ids")
    fold_index = int(args.fold_index)
    if fold_index not in range(1, 7) or source_tx_ids != _icmt.FROZEN_FOLD_SOURCE_TX[fold_index]:
        raise RCATPostfreezePairError("RCAT source TX/fold binding drifted")
    if str(args.candidate_pair) != f"F{fold_index}_C_vs_G":
        raise RCATPostfreezePairError("RCAT candidate pair binding drifted")
    checkpoint_infos: dict[str, dict[str, Any]] = {}
    for arm, field in (("C", "c_final_checkpoint"), ("G", "g_final_checkpoint")):
        checkpoint_infos[arm] = _strict_current_checkpoint(
            getattr(args, field),
            training_root=training_root,
            fold_index=fold_index,
            arm=arm,
            source_tx_ids=source_tx_ids,
        )
    common_binding = validate_rcat_common_training_binding(
        checkpoint_infos["C"]["receipt"], checkpoint_infos["G"]["receipt"]
    )
    return training_root, source_tx_ids, fold_index, common_binding


def _atomic_rewrite_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".rcat.tmp")
    if temporary.exists():
        raise RCATPostfreezePairError(f"refusing to overwrite temporary pair JSON: {temporary}")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _pair_verdict(metrics: Mapping[str, Any]) -> str:
    aggregate = metrics.get("matrix_aggregate")
    if isinstance(aggregate, Mapping):
        return str(aggregate.get("verdict", "REJECT_P1_RCAT_PERMANENT"))
    gates = metrics.get("postfreeze_gates")
    if not isinstance(gates, Mapping) or str(gates.get("fold_verdict", "")).startswith("REJECT"):
        return "REJECT_P1_RCAT_PERMANENT"
    return "PENDING_MAIN_REVIEW_FULL_6_FOLD"


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate one frozen RCAT pair; F6 seals only a pending-main matrix."""

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
            "raw_rcat_receipt_sha256": info["raw_receipt_sha256"],
            "terminal_contract": receipt["rcat_terminal_contract"],
            "terminal_contract_passed": True,
            "enabled": receipt["enabled"],
            "lambda": receipt["lambda"],
            "source_receiver_ids": receipt["source_receiver_ids"],
            "source_receiver_count": receipt["source_receiver_count"],
            "source_receiver_ids_sha256": receipt["source_receiver_ids_sha256"],
            "source_receiver_provenance": receipt["source_receiver_provenance"],
            "frozen_cells_per_scene": receipt["frozen_cells_per_scene"],
            "loss_global_denominator": receipt["loss_global_denominator"],
            "totalized_l2_rule": receipt["totalized_l2_rule"],
            "training_accumulation_dtype": receipt["training_accumulation_dtype"],
            "postfreeze_totalized_l2_dtype": receipt["postfreeze_totalized_l2_dtype"],
            "common_l_base_head_input_path_verified": receipt[
                "common_l_base_head_input_path_verified"
            ],
            "aux_status": "G_POSITIVE_Q_AND_VJP" if arm == "G" else "C_NA_OR_ZERO",
            "positive_q": int(receipt.get("rcat_positive_q", 0)),
            "gradient_audit_completed": bool(
                receipt.get("rcat_gradient_audit_completed", False)
            ),
            "exact_head_aux_vjp": (
                str(receipt.get("rcat_gradient_audit", {}).get("exact_head_aux_vjp", ""))
                if arm == "G"
                else "N_A_CONTROL_ARM"
            ),
        }
    metrics["rcat_training_receipt_revalidation"] = receipt_binding
    metrics["rcat_common_training_binding"] = common_binding
    metrics["rcat_proxy_logits_recomputation"] = proxy_recomputation
    _validate_persisted_common_training_binding(
        metrics["rcat_common_training_binding"],
        common_binding,
        label=f"current F{fold_index}",
    )
    metrics["verdict"] = _pair_verdict(metrics)
    output_path = Path(args.output_metrics_json).resolve()
    _atomic_rewrite_json(output_path, metrics)
    return metrics


fit_frozen_rcat_diagonal_gaussian = _icmt.fit_frozen_diagonal_gaussian
score_frozen_rcat_nll = _icmt.score_frozen_icmt_nll
normalize_rcat_float64 = _icmt._normalize_float64


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
