#!/usr/bin/env python
"""Final-only, source-paired P1-RECTE postfreeze closure.

The signed generic evaluator is reused only as a frozen fairness kernel for
float64 totalized-L2 Gaussian scoring, source/LEO metadata validation and F6
raw-artifact recomputation.  This facade owns every RECTE identity, reopens
the current raw RECTE receipts for both arms, and never accepts a prior pair's
self-reported summary as evidence.
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

import export_phase1_recte_features as _recte_export
import export_phase1_recte_leo_features as _recte_leo
import evaluate_phase1_icmt_postfreeze_pair as _icmt


EXPECTED_TRAINING_RUN_LEAF = "phase1_recte12_20260810_v1"
EXPECTED_POSTFREEZE_MATRIX_ID = "phase1_recte_postfreeze_20260810_v1"
EXPECTED_LV_EXPORT_SCHEMA = "cvs.phase1.recte_lv_export.v1"
EXPECTED_LEO_BINDING_SCHEMA = "cvs.phase1.recte_leo_binding.v1"
EXPECTED_PAIR_SCHEMA = "cvs.phase1.recte_postfreeze_pair.v1"
EXPECTED_RECTE_RECEIPT_SCHEMA = "cvs.phase1.recte_receipt.v1"
EXPECTED_HEAD_CONTRACT = "dual_cvsincnet_tx_logits_v1"
EXPECTED_CLEAN_ARTIFACT = "icmt_clean_l_v_proxy_final_only.npz"
LEGACY_IDENTITY_PREFIXES = ("icmt_", "rcat_", "rcrmd_", "cagm_")

COMMON_TRAINING_BINDING_FIELDS = (
    "baseline_sha256",
    "initial_checkpoint_sha256",
    "warm_start_mode",
    "baseline_path",
    "checkpoint_epoch",
    "checkpoint_role",
    "strict_model_keys",
    "missing_model_keys",
    "unexpected_model_keys",
    "optimizer_state_restored",
    "rng_state_restored",
    "optimizer_type",
    "optimizer_initial_state_sha256",
    "optimizer_initial_state_empty",
    "amp_contract",
    "source_partition_sha256",
    "class_order_binding_sha256",
    "source_labeled_indices_sha256",
    "source_split_manifest_sha256",
    "source_receiver_ids_sha256",
    "source_receiver_provenance",
    "source_receiver_ids",
    "source_receiver_count",
    "frozen_source_receiver_ids",
    "frozen_source_receiver_count",
    "frozen_cells_per_scene",
    "loss_global_denominator",
    "fixed_unordered_pair_count",
    "z_id_key",
    "head_input_path",
    "common_l_base_head_input_path_verified",
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
    "common_batch_sequence_sha256",
    "common_batch_sequence_batches",
    "common_batch_sequence_rows",
    "common_scenario_batches",
    "recte_common_cells",
    "recte_common_batch_cells",
)
COMMON_TRAINING_SHA_FIELDS = (
    "baseline_sha256",
    "initial_checkpoint_sha256",
    "optimizer_initial_state_sha256",
    "source_partition_sha256",
    "class_order_binding_sha256",
    "source_labeled_indices_sha256",
    "source_split_manifest_sha256",
    "source_receiver_ids_sha256",
    "common_batch_sequence_sha256",
)
G_ONLY_RECEIPT_FIELDS = (
    "recte_batches",
    "recte_total_rows",
    "recte_occupied_unordered_pair_count",
    "recte_positive_tail_pair_count",
    "recte_loss_sum",
    "recte_functional_head_readout_count",
    "recte_scenes",
    "recte_g_batch_aux",
    "recte_gradient_audit_attempted",
    "recte_gradient_audit_completed",
    "recte_gradient_audit_scenes",
)
FROZEN_POSTFREEZE_CONTRACT = {
    "RECTE-PF-01": "final-only RECTE z_id=feat_joint with L-only diagonal Gaussian fit",
    "RECTE-PF-02": "float64 totalized-L2 retains exact zero rows and rejects non-finite features",
    "RECTE-PF-03": "ddof1 class-equal pooled variance with 0.9/0.1 shrink, 1e-6 floor and stable full NLL",
    "RECTE-PF-04": "V/proxy contribute no fit rows while every L/V/proxy row remains in the evidence",
    "RECTE-PF-05": "fixed proxy days/RXs/seed/max-per-TX/total=400 binds NPZ, physical keys, JSON and CSV",
    "RECTE-PF-06": "raw C/G receipts bind warm-start/head/class/order/split/new AdamW and source-RX provenance",
    "RECTE-PF-07": "common physical/RX/class/scene n_rc and batch order close at 28 cells per scene and 84 terminal cells",
    "RECTE-PF-08": "each G scene has a positive-tail pair, exact functional logits equal live logits, and raw VJP with head None/zero",
    "RECTE-PF-09": "F6 re-reads F1--F5 clean/LEO/binding/proxy JSON+CSV/checkpoint and recomputes summary, deltas and gates",
    "RECTE-PF-10": "clean6, LEO18, four floors, fold/global overall and two strict proxy gates are non-compensating",
}


class RECTEPostfreezePairError(RuntimeError):
    """Raised when RECTE postfreeze evidence cannot close fail-closed."""


def _translate(error: BaseException) -> RECTEPostfreezePairError:
    return RECTEPostfreezePairError(str(error))


def _require_no_legacy_identity_fields(value: Mapping[str, Any], *, label: str) -> None:
    leaked = [
        str(field)
        for field in value
        if str(field).lower().startswith(LEGACY_IDENTITY_PREFIXES)
    ]
    if leaked:
        raise RECTEPostfreezePairError(
            f"{label} leaks historical method identity fields: {','.join(sorted(leaked))}"
        )


def _strip_legacy_identity_fields(value: Any) -> Any:
    """Remove compatibility-only keys before a RECTE artifact is persisted."""

    if isinstance(value, Mapping):
        return {
            str(key): _strip_legacy_identity_fields(item)
            for key, item in value.items()
            if not str(key).lower().startswith(LEGACY_IDENTITY_PREFIXES)
        }
    if isinstance(value, list):
        return [_strip_legacy_identity_fields(item) for item in value]
    return value


_LOGITS_REJECT_MODULE: Any | None = None


def _logits_reject_module() -> Any:
    """Load the frozen logits proxy scorer without an editable adapter."""

    global _LOGITS_REJECT_MODULE
    if _LOGITS_REJECT_MODULE is not None:
        return _LOGITS_REJECT_MODULE
    source = Path(__file__).resolve().parent / "scripts" / "eval_phase1_logits_open_set_reject.py"
    spec = importlib.util.spec_from_file_location("_recte_frozen_logits_proxy", source)
    if spec is None or spec.loader is None:
        raise RECTEPostfreezePairError("cannot load frozen logits proxy scorer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _LOGITS_REJECT_MODULE = module
    return module


def _close_proxy_value(actual: Any, expected: Any, *, field: str, label: str) -> None:
    if actual is None or expected is None:
        if actual is not expected:
            raise RECTEPostfreezePairError(f"{label} proxy {field} does not match raw logits")
        return
    try:
        left = float(actual)
        right = float(expected)
    except (TypeError, ValueError) as exc:
        raise RECTEPostfreezePairError(f"{label} proxy {field} is not numeric") from exc
    if not math.isfinite(left) or not math.isfinite(right) or not math.isclose(
        left, right, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise RECTEPostfreezePairError(f"{label} proxy {field} does not match raw logits")


def _validate_proxy_logits_recompute(
    *,
    clean_npz: str | Path,
    proxy_metrics_json: str | Path,
    proxy_scores_csv: str | Path,
    source_tx_ids: Sequence[str],
    label: str,
) -> dict[str, Any]:
    """Recompute frozen proxy JSON and CSV from the current clean NPZ bytes."""

    try:
        observed = json.loads(Path(proxy_metrics_json).read_text(encoding="utf-8"))
    except Exception as exc:
        raise RECTEPostfreezePairError(f"{label} proxy diagnostic JSON is invalid") from exc
    if not isinstance(observed, Mapping):
        raise RECTEPostfreezePairError(f"{label} proxy diagnostic JSON must encode an object")
    scorer = _logits_reject_module()
    with tempfile.TemporaryDirectory(prefix="recte_proxy_recompute_") as temporary:
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
                raise RECTEPostfreezePairError(
                    f"{label} proxy {field} does not match raw logits"
                )
        if observed.get("gate_policy") != recomputed.get("gate_policy"):
            raise RECTEPostfreezePairError(f"{label} proxy gate policy does not match raw logits")
        observed_calibration = observed.get("calibration")
        expected_calibration = recomputed.get("calibration")
        if not isinstance(observed_calibration, Mapping) or not isinstance(expected_calibration, Mapping):
            raise RECTEPostfreezePairError(f"{label} proxy calibration is malformed")
        if set(observed_calibration) != set(expected_calibration):
            raise RECTEPostfreezePairError(f"{label} proxy calibration keys drifted")
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
            raise RECTEPostfreezePairError(f"{label} proxy score CSV is invalid") from exc
        if actual_fields != expected_fields or actual_rows != expected_rows:
            raise RECTEPostfreezePairError(f"{label} proxy score CSV does not match raw logits")
    return {
        "passed": True,
        "clean_npz_sha256": _icmt._cb._sha256_file(clean_npz),
        "proxy_metrics_json_sha256": _icmt._cb._sha256_file(proxy_metrics_json),
        "proxy_scores_csv_sha256": _icmt._cb._sha256_file(proxy_scores_csv),
    }


def _canonical_training_root(value: str | Path) -> Path:
    root = Path(value).resolve()
    if root.name != EXPECTED_TRAINING_RUN_LEAF or not root.is_dir():
        raise RECTEPostfreezePairError(
            f"training run root must be existing {EXPECTED_TRAINING_RUN_LEAF}: {root}"
        )
    return root


def _expected_final_checkpoint(training_root: Path, fold_index: int, arm: str) -> tuple[str, Path]:
    if fold_index not in range(1, 7) or arm not in {"C", "G"}:
        raise RECTEPostfreezePairError("unsupported frozen RECTE fold/arm")
    candidate = f"F{fold_index}{arm}_RECTE12"
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
        raise RECTEPostfreezePairError(
            f"{arm} final checkpoint path does not bind frozen {candidate}"
        )
    checkpoint = torch.load(observed, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise RECTEPostfreezePairError(f"{arm} checkpoint payload must be a mapping")
    try:
        _, receipt, observed_arm = _recte_export.validate_recte_training_checkpoint(
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
    except _recte_export.RECTESplitExportError as exc:
        raise _translate(exc) from exc
    if observed_arm != arm:
        raise RECTEPostfreezePairError(f"{arm} checkpoint receipt arm drifted")
    raw_receipt = checkpoint.get("recte_receipt")
    if not isinstance(raw_receipt, Mapping):
        raise RECTEPostfreezePairError(f"{arm} checkpoint lacks raw recte_receipt")
    return {
        "candidate": candidate,
        "path": observed,
        "sha256": _icmt._cb._sha256_file(observed),
        "receipt": dict(receipt),
        "raw_receipt_sha256": _recte_export._canonical_json_sha256(dict(raw_receipt)),
    }


def _strict_common_training_projection(
    receipt: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    """Extract the exact C/G-common RECTE training projection."""

    if not isinstance(receipt, Mapping):
        raise RECTEPostfreezePairError(f"{label} common training receipt is not a mapping")
    missing = [field for field in COMMON_TRAINING_BINDING_FIELDS if field not in receipt]
    if missing:
        raise RECTEPostfreezePairError(
            f"{label} common training binding lacks fields: {','.join(missing)}"
        )
    projection = {field: receipt[field] for field in COMMON_TRAINING_BINDING_FIELDS}
    for field in COMMON_TRAINING_SHA_FIELDS:
        if type(projection[field]) is not str:
            raise RECTEPostfreezePairError(
                f"{label} common training binding {field} must be a string"
            )
        try:
            _recte_export._require_sha256(
                projection[field], field=f"{label} common training binding {field}"
            )
        except _recte_export.RECTESplitExportError as exc:
            raise _translate(exc) from exc
    if projection["warm_start_mode"] != "MODEL_WEIGHTS_ONLY_NEW_ADAMW_AMP":
        raise RECTEPostfreezePairError(f"{label} common warm-start mode drifted")
    if projection["checkpoint_role"] != "training_final_only":
        raise RECTEPostfreezePairError(f"{label} common checkpoint role drifted")
    if projection["strict_model_keys"] is not True or projection["missing_model_keys"] != [] or projection["unexpected_model_keys"] != []:
        raise RECTEPostfreezePairError(f"{label} common strict warm-start model keys drifted")
    if projection["optimizer_state_restored"] is not False or projection["rng_state_restored"] is not False:
        raise RECTEPostfreezePairError(f"{label} common optimizer/RNG restoration drifted")
    if projection["optimizer_type"] != "AdamW" or projection["optimizer_initial_state_empty"] is not True:
        raise RECTEPostfreezePairError(f"{label} common new AdamW receipt drifted")
    if projection["amp_contract"] != "COMMON_TRAINER_AMP_ENABLED":
        raise RECTEPostfreezePairError(f"{label} common AMP contract drifted")
    if projection["z_id_key"] != "feat_joint" or projection["common_l_base_head_input_path_verified"] is not True:
        raise RECTEPostfreezePairError(f"{label} common feat_joint/exact-head path drifted")
    if projection["source_receiver_provenance"] != _recte_export.SOURCE_RECEIVER_PROVENANCE:
        raise RECTEPostfreezePairError(f"{label} common source receiver provenance drifted")
    for field in ("source_receiver_ids", "frozen_source_receiver_ids"):
        if type(projection[field]) is not list or tuple(projection[field]) != _recte_export.FROZEN_SOURCE_RECEIVER_IDS:
            raise RECTEPostfreezePairError(f"{label} common training binding {field} drifted")
    for field, expected in (
        ("source_receiver_count", _recte_export.FROZEN_SOURCE_RECEIVER_COUNT),
        ("frozen_source_receiver_count", _recte_export.FROZEN_SOURCE_RECEIVER_COUNT),
        ("frozen_cells_per_scene", _recte_export.FROZEN_CELLS_PER_SCENE),
        ("loss_global_denominator", _recte_export.FROZEN_UNORDERED_PAIR_COUNT),
        ("fixed_unordered_pair_count", _recte_export.FROZEN_UNORDERED_PAIR_COUNT),
    ):
        if type(projection[field]) is not int or projection[field] != expected:
            raise RECTEPostfreezePairError(f"{label} common training binding {field} drifted")
    for field in (
        "dataset_tx_class_order",
        "local_tx_class_order",
        "checkpoint_train_tx_class_order",
        "local_to_dataset_class_ids",
        "local_to_head_class_ids",
        "expected_tx_class_ids",
    ):
        if type(projection[field]) is not list:
            raise RECTEPostfreezePairError(f"{label} common training binding {field} must be a list")
    for field in (
        "dataset_class_count",
        "local_data_class_count",
        "checkpoint_head_class_count",
        "live_head_class_count",
        "common_batch_sequence_batches",
        "common_batch_sequence_rows",
    ):
        if type(projection[field]) is not int or projection[field] <= 0:
            raise RECTEPostfreezePairError(
                f"{label} common training binding {field} must be a positive integer"
            )
    scenarios = projection["common_scenario_batches"]
    expected_scenarios = tuple(_recte_export._recte.FROZEN_RECTE_SCENARIOS)
    if type(scenarios) is not dict or set(scenarios) != set(expected_scenarios):
        raise RECTEPostfreezePairError(
            f"{label} common training binding common_scenario_batches keys drifted"
        )
    normalized_scenarios: dict[str, int] = {}
    for scenario in expected_scenarios:
        value = scenarios[scenario]
        if type(value) is not int or value <= 0:
            raise RECTEPostfreezePairError(
                f"{label} common training binding scenario count must be a positive integer"
            )
        normalized_scenarios[scenario] = value
    if sum(normalized_scenarios.values()) != projection["common_batch_sequence_batches"]:
        raise RECTEPostfreezePairError(
            f"{label} common training binding scenario batches do not close"
        )
    projection["common_scenario_batches"] = normalized_scenarios
    common_cells = projection["recte_common_cells"]
    if type(common_cells) is not dict or set(common_cells) != set(expected_scenarios):
        raise RECTEPostfreezePairError(f"{label} common training binding recte_common_cells drifted")
    for scenario in expected_scenarios:
        cells = common_cells[scenario]
        if type(cells) is not dict or len(cells) != _recte_export.FROZEN_CELLS_PER_SCENE:
            raise RECTEPostfreezePairError(
                f"{label} common training binding lacks 28 receiver/class cells"
            )
    batch_cells = projection["recte_common_batch_cells"]
    if type(batch_cells) is not list or len(batch_cells) != projection["common_batch_sequence_batches"]:
        raise RECTEPostfreezePairError(
            f"{label} common training binding batch n_rc receipt is incomplete"
        )
    return projection


def validate_recte_common_training_binding(
    c_receipt: Mapping[str, Any], g_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail unless C/G share exact warm-start, RX/class cells, data and AdamW."""

    c_projection = _strict_common_training_projection(c_receipt, label="C")
    g_projection = _strict_common_training_projection(g_receipt, label="G")
    if set(c_projection) != set(COMMON_TRAINING_BINDING_FIELDS) or set(g_projection) != set(
        COMMON_TRAINING_BINDING_FIELDS
    ):
        raise RECTEPostfreezePairError("C/G common training binding key set drifted")
    for field in COMMON_TRAINING_BINDING_FIELDS:
        if type(c_projection[field]) is not type(g_projection[field]) or c_projection[field] != g_projection[field]:
            raise RECTEPostfreezePairError(f"C/G common training binding {field} differs")
    return {
        "passed": True,
        "fields": dict(c_projection),
        "sha256": _recte_export._canonical_json_sha256(c_projection),
    }


def _validate_persisted_common_training_binding(
    observed: Any, expected: Mapping[str, Any], *, label: str
) -> None:
    """Compare a persisted pair receipt with a fresh raw C/G recomputation."""

    if type(observed) is not dict or set(observed) != {"passed", "fields", "sha256"}:
        raise RECTEPostfreezePairError(
            f"{label} persisted common training binding key set drifted"
        )
    if observed["passed"] is not True:
        raise RECTEPostfreezePairError(
            f"{label} persisted common training binding did not pass"
        )
    fields = observed["fields"]
    if type(fields) is not dict or set(fields) != set(COMMON_TRAINING_BINDING_FIELDS):
        raise RECTEPostfreezePairError(
            f"{label} persisted common training binding fields drifted"
        )
    normalized = _strict_common_training_projection(fields, label=label)
    expected_fields = expected.get("fields")
    if type(expected_fields) is not dict:
        raise RECTEPostfreezePairError("internal expected common training binding is malformed")
    for field in COMMON_TRAINING_BINDING_FIELDS:
        if type(normalized[field]) is not type(expected_fields.get(field)) or normalized[field] != expected_fields.get(field):
            raise RECTEPostfreezePairError(
                f"{label} persisted common training binding {field} does not match raw receipts"
            )
    digest = _recte_export._canonical_json_sha256(normalized)
    if type(observed["sha256"]) is not str or observed["sha256"] != digest or digest != expected.get("sha256"):
        raise RECTEPostfreezePairError(
            f"{label} persisted common training binding SHA256 does not match raw receipts"
        )


def _require_exact_manifest_fields(
    manifest: Mapping[str, Any],
    *,
    checkpoint_info: Mapping[str, Any],
    label: str,
) -> None:
    _require_no_legacy_identity_fields(manifest, label=f"{label} RECTE manifest")
    receipt = checkpoint_info["receipt"]
    arm = "G" if str(checkpoint_info["candidate"]).endswith("G_RECTE12") else "C"
    expected = {
        "schema": EXPECTED_LV_EXPORT_SCHEMA,
        "method": "P1_RECTE",
        "training_run_contract": EXPECTED_TRAINING_RUN_LEAF,
        "recte_receipt_schema": EXPECTED_RECTE_RECEIPT_SCHEMA,
        "recte_enabled": arm == "G",
        "recte_receipt_sha256": checkpoint_info["raw_receipt_sha256"],
        "recte_terminal_contract": str(receipt["recte_terminal_contract"]),
        "recte_terminal_contract_passed": True,
        "recte_lambda": _recte_export.FROZEN_RECTE_LAMBDA if arm == "G" else 0.0,
        "recte_loss_global_denominator": _recte_export.FROZEN_UNORDERED_PAIR_COUNT,
        "recte_fixed_unordered_pair_count": _recte_export.FROZEN_UNORDERED_PAIR_COUNT,
        "recte_source_receiver_ids": list(_recte_export.FROZEN_SOURCE_RECEIVER_IDS),
        "recte_source_receiver_count": _recte_export.FROZEN_SOURCE_RECEIVER_COUNT,
        "recte_source_receiver_ids_sha256": str(receipt["source_receiver_ids_sha256"]),
        "recte_source_receiver_provenance": _recte_export.SOURCE_RECEIVER_PROVENANCE,
        "recte_frozen_cells_per_scene": _recte_export.FROZEN_CELLS_PER_SCENE,
        "recte_common_physical_rx_class_scene_nrc_bound": True,
        "recte_batch_order_bound": True,
        "recte_functional_logits_live_equality_required": True,
        "recte_exact_head_aux_vjp_na_none_or_zero": True,
        "proxy_selection_frozen_not_cli_tunable": True,
    }
    for field, wanted in expected.items():
        if manifest.get(field) != wanted or type(manifest.get(field)) is not type(wanted):
            raise RECTEPostfreezePairError(f"{label} manifest {field} drifted")
    if str(manifest.get("candidate_id", "")) != str(checkpoint_info["candidate"]):
        raise RECTEPostfreezePairError(f"{label} manifest RECTE candidate binding drifted")
    if Path(str(manifest.get("checkpoint", ""))).resolve() != checkpoint_info["path"]:
        raise RECTEPostfreezePairError(f"{label} manifest final checkpoint path drifted")
    if str(manifest.get("source_checkpoint_sha256", "")) != checkpoint_info["sha256"]:
        raise RECTEPostfreezePairError(f"{label} manifest final checkpoint SHA256 drifted")
    for field, expected_receipt in (
        ("recte_source_labeled_indices_sha256", receipt["source_labeled_indices_sha256"]),
        ("recte_source_split_manifest_sha256", receipt["source_split_manifest_sha256"]),
    ):
        if str(manifest.get(field, "")) != str(expected_receipt):
            raise RECTEPostfreezePairError(f"{label} manifest {field} does not bind raw receipt")


_ORIGINAL_VALIDATE_LV_PAYLOAD = _icmt._validate_lv_payload
_ORIGINAL_LOAD_LEO_BINDING = _icmt._load_icmt_leo_binding
_ORIGINAL_RECOMPUTE_PRIOR = _icmt._recompute_prior_pair_artifacts
_ORIGINAL_MATRIX_AGGREGATE = _icmt._matrix_aggregate
_ORIGINAL_FOLD_GATES = _icmt._fold_gates


def _validate_recte_lv_payload(
    payload: Mapping[str, Any],
    source_tx_ids: Sequence[str],
    fold_index: int,
    expected_proxy_count: int,
    *,
    label: str,
) -> dict[str, Any]:
    """Validate RECTE identity then invoke the signed L/V Gaussian validator."""

    manifest = payload.get("manifest")
    if not isinstance(manifest, Mapping):
        raise RECTEPostfreezePairError(f"{label} lacks RECTE L/V manifest")
    arm = "C" if label.startswith("C") else "G" if label.startswith("G") else ""
    if not arm:
        raise RECTEPostfreezePairError(f"{label} does not identify a C/G arm")
    training_root_value = Path(str(manifest.get("checkpoint", ""))).resolve().parents[1]
    candidate, checkpoint_path = _expected_final_checkpoint(
        training_root_value, fold_index, arm
    )
    if checkpoint_path != Path(str(manifest.get("checkpoint", ""))).resolve():
        raise RECTEPostfreezePairError(f"{label} checkpoint layout does not bind candidate")
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
            "icmt_receipt_schema": EXPECTED_RECTE_RECEIPT_SCHEMA,
            "icmt_enabled": arm == "G",
            "icmt_source_labeled_indices_sha256": manifest[
                "recte_source_labeled_indices_sha256"
            ],
            "icmt_source_split_manifest_sha256": manifest[
                "recte_source_split_manifest_sha256"
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


def _load_recte_leo_binding(
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
        raise RECTEPostfreezePairError(f"{label} RECTE LEO binding JSON is invalid") from exc
    if not isinstance(binding, Mapping):
        raise RECTEPostfreezePairError(f"{label} RECTE LEO binding must encode an object")
    _require_no_legacy_identity_fields(binding, label=f"{label} RECTE LEO binding")
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
        "method": "P1_RECTE",
        "recte_receipt_schema": EXPECTED_RECTE_RECEIPT_SCHEMA,
        "recte_receipt_sha256": checkpoint_info["raw_receipt_sha256"],
        "recte_terminal_contract": str(receipt["recte_terminal_contract"]),
        "recte_terminal_contract_passed": True,
        "recte_lambda": _recte_export.FROZEN_RECTE_LAMBDA if arm == "G" else 0.0,
        "recte_loss_global_denominator": _recte_export.FROZEN_UNORDERED_PAIR_COUNT,
        "recte_fixed_unordered_pair_count": _recte_export.FROZEN_UNORDERED_PAIR_COUNT,
        "recte_source_receiver_ids": list(_recte_export.FROZEN_SOURCE_RECEIVER_IDS),
        "recte_source_receiver_count": _recte_export.FROZEN_SOURCE_RECEIVER_COUNT,
        "recte_source_receiver_ids_sha256": str(receipt["source_receiver_ids_sha256"]),
        "recte_source_receiver_provenance": _recte_export.SOURCE_RECEIVER_PROVENANCE,
        "recte_frozen_cells_per_scene": _recte_export.FROZEN_CELLS_PER_SCENE,
        "recte_common_physical_rx_class_scene_nrc_bound": True,
        "recte_batch_order_bound": True,
        "recte_functional_logits_live_equality_required": True,
        "recte_exact_head_aux_vjp_na_none_or_zero": True,
    }
    for field, wanted in expected_extra.items():
        if binding.get(field) != wanted or type(binding.get(field)) is not type(wanted):
            raise RECTEPostfreezePairError(f"{label} RECTE LEO binding {field} drifted")
    return dict(result)


def _recompute_recte_prior_pair_artifacts(
    record: Mapping[str, Any],
    *,
    output_root: Path,
    matrix_id: str,
    training_root: Path,
    expected_scenarios: Sequence[str],
) -> dict[str, Any]:
    """Rebuild F1--F5 from raw current artifacts, not any prior summary."""

    fold_index = int(record.get("fold_index", -1))
    if fold_index not in range(1, 7):
        raise RECTEPostfreezePairError("prior pair fold is invalid")
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
    common_binding = validate_recte_common_training_binding(
        checkpoint_infos["C"]["receipt"], checkpoint_infos["G"]["receipt"]
    )
    _validate_persisted_common_training_binding(
        record.get("recte_common_training_binding"),
        common_binding,
        label=f"prior F{fold_index}",
    )
    bindings = record.get("bindings")
    if not isinstance(bindings, Mapping):
        raise RECTEPostfreezePairError("prior pair lacks raw artifact bindings")
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
    result["recte_common_training_binding"] = common_binding
    result["recte_raw_checkpoint_sha256"] = {
        arm: checkpoint_infos[arm]["sha256"] for arm in ("C", "G")
    }
    return _strip_legacy_identity_fields(result)


def _recte_fold_gates(
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
        result["fold_verdict"] = "REJECT_P1_RECTE_PERMANENT"
    elif result.get("fold_verdict") == "PENDING_GLOBAL_18_GRID":
        result["fold_verdict"] = "PENDING_MAIN_REVIEW_FULL_6_FOLD"
    return result


def _recte_matrix_aggregate(
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
    common_binding = validate_recte_common_training_binding(
        checkpoint_infos["C"]["receipt"], checkpoint_infos["G"]["receipt"]
    )
    if not isinstance(current, dict):
        raise RECTEPostfreezePairError("current pair must be mutable for RECTE binding receipt")
    current["recte_common_training_binding"] = common_binding
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
        result["verdict"] = "REJECT_P1_RECTE_PERMANENT"
    else:
        result["verdict"] = "PENDING_MAIN_REVIEW"
    return _strip_legacy_identity_fields(result)


@contextmanager
def _patched_signed_fairness_kernel() -> Iterator[None]:
    """Retain signed math while replacing every persisted identity with RECTE."""

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
    _icmt.EXPECTED_ICMT_RECEIPT_SCHEMA = EXPECTED_RECTE_RECEIPT_SCHEMA
    _icmt.FROZEN_POSTFREEZE_CONTRACT = FROZEN_POSTFREEZE_CONTRACT
    _icmt._expected_final_checkpoint = _expected_final_checkpoint
    _icmt._validate_lv_payload = _validate_recte_lv_payload
    _icmt._load_icmt_leo_binding = _load_recte_leo_binding
    _icmt._recompute_prior_pair_artifacts = _recompute_recte_prior_pair_artifacts
    _icmt._fold_gates = _recte_fold_gates
    _icmt._matrix_aggregate = _recte_matrix_aggregate
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
        raise RECTEPostfreezePairError(
            f"postfreeze_matrix_id must be {EXPECTED_POSTFREEZE_MATRIX_ID}"
        )
    output_root = Path(args.postfreeze_output_root).resolve()
    if output_root.name != EXPECTED_POSTFREEZE_MATRIX_ID or not output_root.is_dir():
        raise RECTEPostfreezePairError("postfreeze output root does not bind frozen RECTE matrix")
    training_root = _canonical_training_root(args.training_run_root)
    source_tx_ids = _icmt._cb._parse_items(args.source_tx_ids, field="source_tx_ids")
    fold_index = int(args.fold_index)
    if fold_index not in range(1, 7) or source_tx_ids != _icmt.FROZEN_FOLD_SOURCE_TX[fold_index]:
        raise RECTEPostfreezePairError("RECTE source TX/fold binding drifted")
    if str(args.candidate_pair) != f"F{fold_index}_C_vs_G":
        raise RECTEPostfreezePairError("RECTE candidate pair binding drifted")
    checkpoint_infos: dict[str, dict[str, Any]] = {}
    for arm, field in (("C", "c_final_checkpoint"), ("G", "g_final_checkpoint")):
        checkpoint_infos[arm] = _strict_current_checkpoint(
            getattr(args, field),
            training_root=training_root,
            fold_index=fold_index,
            arm=arm,
            source_tx_ids=source_tx_ids,
        )
    common_binding = validate_recte_common_training_binding(
        checkpoint_infos["C"]["receipt"], checkpoint_infos["G"]["receipt"]
    )
    return training_root, source_tx_ids, fold_index, common_binding


def _atomic_rewrite_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".recte.tmp")
    if temporary.exists():
        raise RECTEPostfreezePairError(f"refusing to overwrite temporary pair JSON: {temporary}")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _pair_verdict(metrics: Mapping[str, Any]) -> str:
    aggregate = metrics.get("matrix_aggregate")
    if isinstance(aggregate, Mapping):
        return str(aggregate.get("verdict", "REJECT_P1_RECTE_PERMANENT"))
    gates = metrics.get("postfreeze_gates")
    if not isinstance(gates, Mapping) or str(gates.get("fold_verdict", "")).startswith("REJECT"):
        return "REJECT_P1_RECTE_PERMANENT"
    return "PENDING_MAIN_REVIEW_FULL_6_FOLD"


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate one same-fold RECTE pair; F6 seals only a pending-main matrix."""

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
            "raw_recte_receipt_sha256": info["raw_receipt_sha256"],
            "terminal_contract": receipt["recte_terminal_contract"],
            "terminal_contract_passed": True,
            "enabled": receipt["enabled"],
            "lambda": receipt["lambda"],
            "source_receiver_ids": receipt["source_receiver_ids"],
            "source_receiver_count": receipt["source_receiver_count"],
            "source_receiver_ids_sha256": receipt["source_receiver_ids_sha256"],
            "source_receiver_provenance": receipt["source_receiver_provenance"],
            "frozen_cells_per_scene": receipt["frozen_cells_per_scene"],
            "fixed_unordered_pair_count": receipt["fixed_unordered_pair_count"],
            "functional_logits_live_equality_required": True,
            "head_aux_vjp": "N_A_NONE_OR_ZERO_EXPECTED",
        }
    metrics["recte_training_receipt_revalidation"] = receipt_binding
    metrics["recte_common_training_binding"] = common_binding
    metrics["recte_proxy_logits_recomputation"] = proxy_recomputation
    metrics["recte_f6_raw_reopen_required"] = True
    _validate_persisted_common_training_binding(
        metrics["recte_common_training_binding"],
        common_binding,
        label=f"current F{fold_index}",
    )
    metrics = dict(_strip_legacy_identity_fields(metrics))
    metrics["verdict"] = _pair_verdict(metrics)
    _require_no_legacy_identity_fields(metrics, label="RECTE pair output")
    output_path = Path(args.output_metrics_json).resolve()
    _atomic_rewrite_json(output_path, metrics)
    return metrics


fit_frozen_recte_diagonal_gaussian = _icmt.fit_frozen_diagonal_gaussian
score_frozen_recte_nll = _icmt.score_frozen_icmt_nll
normalize_recte_float64 = _icmt._normalize_float64


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
