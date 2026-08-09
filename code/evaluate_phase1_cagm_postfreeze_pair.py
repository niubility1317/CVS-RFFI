#!/usr/bin/env python
"""Final-only, source-paired P1-CAGM postfreeze closure.

The signed ICMT-v2 evaluator is reused only as a frozen *fairness kernel*: it
provides the L-only totalized-L2 float64 diagonal-Gaussian score, source/LEO
metadata checks and F6 raw-artifact recomputation.  This facade replaces every
candidate/root/schema/receipt identity with P1-CAGM and independently reloads
the original CAGM terminal receipt for each current and prior arm.
"""

from __future__ import annotations

import argparse
import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

import export_phase1_cagm_features as _cagm_export
import export_phase1_cagm_leo_features as _cagm_leo
import evaluate_phase1_icmt_postfreeze_pair as _icmt


EXPECTED_TRAINING_RUN_LEAF = "phase1_cagm12_20260810_v2"
EXPECTED_POSTFREEZE_MATRIX_ID = "phase1_cagm_postfreeze_20260810_v2"
EXPECTED_LV_EXPORT_SCHEMA = "cvs.phase1.cagm_lv_export.v1"
EXPECTED_LEO_BINDING_SCHEMA = "cvs.phase1.cagm_leo_binding.v1"
EXPECTED_PAIR_SCHEMA = "cvs.phase1.cagm_postfreeze_pair.v1"
EXPECTED_CAGM_RECEIPT_SCHEMA = "cvs.phase1.cagm_receipt.v2"
EXPECTED_HEAD_CONTRACT = "dual_cvsincnet_tx_logits_v1"
EXPECTED_CLEAN_ARTIFACT = "icmt_clean_l_v_proxy_final_only.npz"
COMMON_TRAINING_BINDING_FIELDS = (
    "baseline_sha256",
    "initial_checkpoint_sha256",
    "class_order_binding_sha256",
    "source_labeled_indices_sha256",
    "source_split_manifest_sha256",
    "optimizer_type",
    "optimizer_initial_state_sha256",
    "optimizer_initial_state_empty",
    "common_batch_sequence_sha256",
    "common_batch_sequence_batches",
    "common_batch_sequence_rows",
    "common_scenario_batches",
)
COMMON_TRAINING_SHA_FIELDS = (
    "baseline_sha256",
    "initial_checkpoint_sha256",
    "class_order_binding_sha256",
    "source_labeled_indices_sha256",
    "source_split_manifest_sha256",
    "optimizer_initial_state_sha256",
    "common_batch_sequence_sha256",
)
FROZEN_POSTFREEZE_CONTRACT = {
    "CAGM-PF-01": "final-only CAGM z_id=feat_joint; diagonal Gaussian fits labelled L only",
    "CAGM-PF-02": "totalized float64 row L2: positive z/norm, exact zero maps to zero, non-finite is fatal",
    "CAGM-PF-03": "ddof=1, 0.9 class plus 0.1 pool variance, 1e-6 floor, complete NLL and stable logsumexp u",
    "CAGM-PF-04": "V/proxy have zero fit rows and every L/V/proxy row is retained",
    "CAGM-PF-05": "fixed signed proxy days/RXs/seed/max-per-TX/total=400 binds NPZ, manifest, JSON, CSV and physical receipt",
    "CAGM-PF-06": "CAGM training_final_only C/G checkpoint, class/head/root/arm and raw terminal receipt are revalidated",
    "CAGM-PF-07": "LEO sidecar repeats ManySig SHA, selected physical rows, NPZ SHA and full three-scenario TX/RX/day coverage",
    "CAGM-PF-08": "F6 re-reads F1--F5 clean/LEO/binding/proxy JSON+CSV and recomputes each summary, delta and gate",
    "CAGM-PF-09": "all clean/LEO floors, per-fold/global overall and six strict continuous-proxy gates are non-compensating",
}


class CAGMPostfreezePairError(RuntimeError):
    """Raised when CAGM postfreeze evidence cannot close fail-closed."""


def _translate(error: BaseException) -> CAGMPostfreezePairError:
    return CAGMPostfreezePairError(str(error))


def _canonical_training_root(value: str | Path) -> Path:
    root = Path(value).resolve()
    if root.name != EXPECTED_TRAINING_RUN_LEAF or not root.is_dir():
        raise CAGMPostfreezePairError(
            f"training run root must be existing {EXPECTED_TRAINING_RUN_LEAF}: {root}"
        )
    return root


def _expected_final_checkpoint(training_root: Path, fold_index: int, arm: str) -> tuple[str, Path]:
    if fold_index not in range(1, 7) or arm not in {"C", "G"}:
        raise CAGMPostfreezePairError("unsupported frozen CAGM fold/arm")
    candidate = f"F{fold_index}{arm}_CAGM12"
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
        raise CAGMPostfreezePairError(
            f"{arm} final checkpoint path does not bind frozen {candidate}"
        )
    checkpoint = torch.load(observed, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise CAGMPostfreezePairError(f"{arm} checkpoint payload must be a mapping")
    try:
        _, receipt, observed_arm = _cagm_export.validate_cagm_training_checkpoint(
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
    except _cagm_export.CAGMSplitExportError as exc:
        raise _translate(exc) from exc
    if observed_arm != arm:
        raise CAGMPostfreezePairError(f"{arm} checkpoint receipt arm drifted")
    raw_receipt = checkpoint.get("cagm_receipt")
    if not isinstance(raw_receipt, Mapping):
        raise CAGMPostfreezePairError(f"{arm} checkpoint lacks raw cagm_receipt")
    return {
        "candidate": candidate,
        "path": observed,
        "sha256": _icmt._cb._sha256_file(observed),
        "receipt": dict(receipt),
        "raw_receipt_sha256": _cagm_export._canonical_json_sha256(dict(raw_receipt)),
    }


def _strict_common_training_projection(
    receipt: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    """Extract one exact, type-strict common-training receipt projection."""

    if not isinstance(receipt, Mapping):
        raise CAGMPostfreezePairError(f"{label} common training receipt is not a mapping")
    missing = [field for field in COMMON_TRAINING_BINDING_FIELDS if field not in receipt]
    if missing:
        raise CAGMPostfreezePairError(
            f"{label} common training binding lacks fields: {','.join(missing)}"
        )
    projection = {field: receipt[field] for field in COMMON_TRAINING_BINDING_FIELDS}
    for field in COMMON_TRAINING_SHA_FIELDS:
        if type(projection[field]) is not str:
            raise CAGMPostfreezePairError(
                f"{label} common training binding {field} must be a string"
            )
        try:
            _cagm_export._require_sha256(
                projection[field], field=f"{label} common training binding {field}"
            )
        except _cagm_export.CAGMSplitExportError as exc:
            raise _translate(exc) from exc
    if type(projection["optimizer_type"]) is not str or projection["optimizer_type"] != "AdamW":
        raise CAGMPostfreezePairError(
            f"{label} common training binding optimizer_type must be literal AdamW"
        )
    if projection["optimizer_initial_state_empty"] is not True:
        raise CAGMPostfreezePairError(
            f"{label} common training binding optimizer_initial_state_empty must be literal True"
        )
    for field in ("common_batch_sequence_batches", "common_batch_sequence_rows"):
        value = projection[field]
        if type(value) is not int or value <= 0:
            raise CAGMPostfreezePairError(
                f"{label} common training binding {field} must be a positive integer"
            )
    scenarios = projection["common_scenario_batches"]
    if type(scenarios) is not dict or set(scenarios) != set(_cagm_export._cagm.FROZEN_CAGM_SCENARIOS):
        raise CAGMPostfreezePairError(
            f"{label} common training binding common_scenario_batches keys drifted"
        )
    normalized_scenarios: dict[str, int] = {}
    for scenario in _cagm_export._cagm.FROZEN_CAGM_SCENARIOS:
        value = scenarios[scenario]
        if type(value) is not int or value <= 0:
            raise CAGMPostfreezePairError(
                f"{label} common training binding scenario count must be a positive integer"
            )
        normalized_scenarios[scenario] = value
    if sum(normalized_scenarios.values()) != projection["common_batch_sequence_batches"]:
        raise CAGMPostfreezePairError(
            f"{label} common training binding scenario batches do not close"
        )
    projection["common_scenario_batches"] = normalized_scenarios
    return projection


def validate_cagm_common_training_binding(
    c_receipt: Mapping[str, Any], g_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail unless C/G share the exact frozen warm-start, data and optimizer binding."""

    c_projection = _strict_common_training_projection(c_receipt, label="C")
    g_projection = _strict_common_training_projection(g_receipt, label="G")
    if set(c_projection) != set(COMMON_TRAINING_BINDING_FIELDS) or set(g_projection) != set(
        COMMON_TRAINING_BINDING_FIELDS
    ):
        raise CAGMPostfreezePairError("C/G common training binding key set drifted")
    for field in COMMON_TRAINING_BINDING_FIELDS:
        if type(c_projection[field]) is not type(g_projection[field]) or c_projection[field] != g_projection[field]:
            raise CAGMPostfreezePairError(
                f"C/G common training binding {field} differs"
            )
    fields = dict(c_projection)
    return {
        "passed": True,
        "fields": fields,
        "sha256": _cagm_export._canonical_json_sha256(fields),
    }


def _validate_persisted_common_training_binding(
    observed: Any, expected: Mapping[str, Any], *, label: str
) -> None:
    """Compare a persisted pair receipt with a fresh raw C/G recomputation."""

    if type(observed) is not dict or set(observed) != {"passed", "fields", "sha256"}:
        raise CAGMPostfreezePairError(
            f"{label} persisted common training binding key set drifted"
        )
    if observed["passed"] is not True:
        raise CAGMPostfreezePairError(
            f"{label} persisted common training binding did not pass"
        )
    fields = observed["fields"]
    if type(fields) is not dict or set(fields) != set(COMMON_TRAINING_BINDING_FIELDS):
        raise CAGMPostfreezePairError(
            f"{label} persisted common training binding fields drifted"
        )
    normalized = _strict_common_training_projection(fields, label=label)
    expected_fields = expected.get("fields")
    if type(expected_fields) is not dict:
        raise CAGMPostfreezePairError("internal expected common training binding is malformed")
    for field in COMMON_TRAINING_BINDING_FIELDS:
        if type(normalized[field]) is not type(expected_fields.get(field)) or normalized[field] != expected_fields.get(field):
            raise CAGMPostfreezePairError(
                f"{label} persisted common training binding {field} does not match raw receipts"
            )
    digest = _cagm_export._canonical_json_sha256(normalized)
    if type(observed["sha256"]) is not str or observed["sha256"] != digest or digest != expected.get("sha256"):
        raise CAGMPostfreezePairError(
            f"{label} persisted common training binding SHA256 does not match raw receipts"
        )


def _require_exact_manifest_fields(
    manifest: Mapping[str, Any],
    *,
    checkpoint_info: Mapping[str, Any],
    label: str,
) -> None:
    receipt = checkpoint_info["receipt"]
    expected = {
        "schema": EXPECTED_LV_EXPORT_SCHEMA,
        "method": "P1_CAGM",
        "training_run_contract": EXPECTED_TRAINING_RUN_LEAF,
        "cagm_receipt_schema": EXPECTED_CAGM_RECEIPT_SCHEMA,
        "cagm_enabled": str(checkpoint_info["candidate"])[2] == "G",
        "cagm_receipt_sha256": checkpoint_info["raw_receipt_sha256"],
        "cagm_terminal_contract": str(receipt["cagm_terminal_contract"]),
        "cagm_terminal_contract_passed": True,
        "cagm_loss_divisor": _cagm_export.FROZEN_CAGM_DIVISOR,
        "cagm_clean_statistics_detached": True,
        "cagm_joint_zero_mask_aux_only": receipt[
            "joint_zero_mask_aux_only"
        ],
        "proxy_selection_frozen_not_cli_tunable": True,
    }
    for field, wanted in expected.items():
        if manifest.get(field) != wanted or type(manifest.get(field)) is not type(wanted):
            raise CAGMPostfreezePairError(f"{label} manifest {field} drifted")
    if str(manifest.get("candidate_id", "")) != str(checkpoint_info["candidate"]):
        raise CAGMPostfreezePairError(f"{label} manifest CAGM candidate binding drifted")
    if Path(str(manifest.get("checkpoint", ""))).resolve() != checkpoint_info["path"]:
        raise CAGMPostfreezePairError(f"{label} manifest final checkpoint path drifted")
    if str(manifest.get("source_checkpoint_sha256", "")) != checkpoint_info["sha256"]:
        raise CAGMPostfreezePairError(f"{label} manifest final checkpoint SHA256 drifted")
    for field, expected_receipt in (
        ("cagm_source_labeled_indices_sha256", receipt["source_labeled_indices_sha256"]),
        ("cagm_source_split_manifest_sha256", receipt["source_split_manifest_sha256"]),
    ):
        if str(manifest.get(field, "")) != str(expected_receipt):
            raise CAGMPostfreezePairError(f"{label} manifest {field} does not bind raw receipt")


_ORIGINAL_VALIDATE_LV_PAYLOAD = _icmt._validate_lv_payload
_ORIGINAL_LOAD_LEO_BINDING = _icmt._load_icmt_leo_binding
_ORIGINAL_RECOMPUTE_PRIOR = _icmt._recompute_prior_pair_artifacts
_ORIGINAL_MATRIX_AGGREGATE = _icmt._matrix_aggregate
_ORIGINAL_FOLD_GATES = _icmt._fold_gates


def _validate_cagm_lv_payload(
    payload: Mapping[str, Any],
    source_tx_ids: Sequence[str],
    fold_index: int,
    expected_proxy_count: int,
    *,
    label: str,
) -> dict[str, Any]:
    """Validate CAGM fields, then run the signed generic L/V validator in-memory."""

    manifest = payload.get("manifest")
    if not isinstance(manifest, Mapping):
        raise CAGMPostfreezePairError(f"{label} lacks CAGM L/V manifest")
    arm = "C" if label.startswith("C") else "G" if label.startswith("G") else ""
    if not arm:
        raise CAGMPostfreezePairError(f"{label} does not identify a C/G arm")
    training_root_value = Path(str(manifest.get("checkpoint", ""))).resolve().parents[1]
    # The actual root is checked by the evaluator before this function; derive
    # only the exact checkpoint path here to bind the manifest receipt.
    candidate, checkpoint_path = _expected_final_checkpoint(
        training_root_value, fold_index, arm
    )
    if checkpoint_path != Path(str(manifest.get("checkpoint", ""))).resolve():
        raise CAGMPostfreezePairError(f"{label} checkpoint layout does not bind candidate")
    checkpoint_info = _strict_current_checkpoint(
        checkpoint_path,
        training_root=training_root_value,
        fold_index=fold_index,
        arm=arm,
        source_tx_ids=source_tx_ids,
    )
    _require_exact_manifest_fields(manifest, checkpoint_info=checkpoint_info, label=label)
    compatibility_manifest = dict(manifest)
    # The imported signed function has one ICMT-only hard-coded candidate
    # string.  It receives a memory-only adapter; no artifact is modified.
    compatibility_manifest.update(
        {
            "candidate_id": f"F{fold_index}{arm}_ICMT12",
            "icmt_receipt_schema": EXPECTED_CAGM_RECEIPT_SCHEMA,
            "icmt_enabled": arm == "G",
            "icmt_source_labeled_indices_sha256": manifest[
                "cagm_source_labeled_indices_sha256"
            ],
            "icmt_source_split_manifest_sha256": manifest[
                "cagm_source_split_manifest_sha256"
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


def _load_cagm_leo_binding(
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
        raise CAGMPostfreezePairError(f"{label} CAGM LEO binding JSON is invalid") from exc
    if not isinstance(binding, Mapping):
        raise CAGMPostfreezePairError(f"{label} CAGM LEO binding must encode an object")
    checkpoint_info = _strict_current_checkpoint(
        expected_checkpoint,
        training_root=training_root,
        fold_index=fold_index,
        arm=arm,
        source_tx_ids=source_tx_ids,
    )
    expected_extra = {
        "schema": EXPECTED_LEO_BINDING_SCHEMA,
        "method": "P1_CAGM",
        "cagm_receipt_schema": EXPECTED_CAGM_RECEIPT_SCHEMA,
        "cagm_receipt_sha256": checkpoint_info["raw_receipt_sha256"],
        "cagm_terminal_contract": str(checkpoint_info["receipt"]["cagm_terminal_contract"]),
        "cagm_terminal_contract_passed": True,
        "cagm_loss_divisor": _cagm_export.FROZEN_CAGM_DIVISOR,
        "cagm_clean_statistics_detached": True,
        "cagm_joint_zero_mask_aux_only": checkpoint_info["receipt"][
            "joint_zero_mask_aux_only"
        ],
    }
    for field, wanted in expected_extra.items():
        if binding.get(field) != wanted or type(binding.get(field)) is not type(wanted):
            raise CAGMPostfreezePairError(f"{label} CAGM LEO binding {field} drifted")
    return dict(result)


def _recompute_cagm_prior_pair_artifacts(
    record: Mapping[str, Any],
    *,
    output_root: Path,
    matrix_id: str,
    training_root: Path,
    expected_scenarios: Sequence[str],
) -> dict[str, Any]:
    fold_index = int(record.get("fold_index", -1))
    if fold_index not in range(1, 7):
        raise CAGMPostfreezePairError("prior pair fold is invalid")
    source_tx_ids = _icmt.FROZEN_FOLD_SOURCE_TX[fold_index]
    # Re-read both original training checkpoints before trusting any F1--F5
    # derived JSON.  The signed raw-artifact recomputation below then reloads
    # clean/LEO/sidecar/proxy JSON+CSV and compares every recomputed field.
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
    common_binding = validate_cagm_common_training_binding(
        checkpoint_infos["C"]["receipt"], checkpoint_infos["G"]["receipt"]
    )
    _validate_persisted_common_training_binding(
        record.get("cagm_common_training_binding"),
        common_binding,
        label=f"prior F{fold_index}",
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
    result["cagm_common_training_binding"] = common_binding
    return result


def _cagm_fold_gates(
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
        result["fold_verdict"] = "REJECT_P1_CAGM_PERMANENT"
    return result


def _cagm_matrix_aggregate(
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
    common_binding = validate_cagm_common_training_binding(
        checkpoint_infos["C"]["receipt"], checkpoint_infos["G"]["receipt"]
    )
    if not isinstance(current, dict):
        raise CAGMPostfreezePairError("current pair must be mutable for CAGM binding receipt")
    current["cagm_common_training_binding"] = common_binding
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
        result["verdict"] = "REJECT_P1_CAGM_PERMANENT"
    return result


@contextmanager
def _patched_signed_fairness_kernel() -> Iterator[None]:
    """Temporarily replace ICMT identities while retaining its signed math/checks."""

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
    _icmt.EXPECTED_ICMT_RECEIPT_SCHEMA = EXPECTED_CAGM_RECEIPT_SCHEMA
    _icmt.FROZEN_POSTFREEZE_CONTRACT = FROZEN_POSTFREEZE_CONTRACT
    _icmt._expected_final_checkpoint = _expected_final_checkpoint
    _icmt._validate_lv_payload = _validate_cagm_lv_payload
    _icmt._load_icmt_leo_binding = _load_cagm_leo_binding
    _icmt._recompute_prior_pair_artifacts = _recompute_cagm_prior_pair_artifacts
    _icmt._fold_gates = _cagm_fold_gates
    _icmt._matrix_aggregate = _cagm_matrix_aggregate
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
        raise CAGMPostfreezePairError(
            f"postfreeze_matrix_id must be {EXPECTED_POSTFREEZE_MATRIX_ID}"
        )
    output_root = Path(args.postfreeze_output_root).resolve()
    if output_root.name != EXPECTED_POSTFREEZE_MATRIX_ID or not output_root.is_dir():
        raise CAGMPostfreezePairError("postfreeze output root does not bind frozen CAGM matrix")
    training_root = _canonical_training_root(args.training_run_root)
    source_tx_ids = _icmt._cb._parse_items(args.source_tx_ids, field="source_tx_ids")
    fold_index = int(args.fold_index)
    if fold_index not in range(1, 7) or source_tx_ids != _icmt.FROZEN_FOLD_SOURCE_TX[fold_index]:
        raise CAGMPostfreezePairError("CAGM source TX/fold binding drifted")
    if str(args.candidate_pair) != f"F{fold_index}_C_vs_G":
        raise CAGMPostfreezePairError("CAGM candidate pair binding drifted")
    checkpoint_infos: dict[str, dict[str, Any]] = {}
    for arm, field in (("C", "c_final_checkpoint"), ("G", "g_final_checkpoint")):
        checkpoint_infos[arm] = _strict_current_checkpoint(
            getattr(args, field),
            training_root=training_root,
            fold_index=fold_index,
            arm=arm,
            source_tx_ids=source_tx_ids,
        )
    common_binding = validate_cagm_common_training_binding(
        checkpoint_infos["C"]["receipt"], checkpoint_infos["G"]["receipt"]
    )
    return training_root, source_tx_ids, fold_index, common_binding


def _atomic_rewrite_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".cagm.tmp")
    if temporary.exists():
        raise CAGMPostfreezePairError(f"refusing to overwrite temporary pair JSON: {temporary}")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _pair_verdict(metrics: Mapping[str, Any]) -> str:
    aggregate = metrics.get("matrix_aggregate")
    if isinstance(aggregate, Mapping):
        return str(aggregate.get("verdict", "REJECT_P1_CAGM_PERMANENT"))
    gates = metrics.get("postfreeze_gates")
    if not isinstance(gates, Mapping) or str(gates.get("fold_verdict", "")).startswith("REJECT"):
        return "REJECT_P1_CAGM_PERMANENT"
    return "PENDING_GLOBAL_18_GRID"


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate one frozen CAGM pair; F6 seals the non-compensating six-fold gate."""

    training_root, source_tx_ids, fold_index, common_binding = _prevalidate_current_args(args)
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
        receipt_binding[arm] = {
            "candidate": info["candidate"],
            "final_checkpoint_sha256": info["sha256"],
            "raw_cagm_receipt_sha256": info["raw_receipt_sha256"],
            "terminal_contract": info["receipt"]["cagm_terminal_contract"],
            "terminal_contract_passed": True,
            "joint_zero_mask_aux_only": info["receipt"][
                "joint_zero_mask_aux_only"
            ],
        }
    metrics["cagm_training_receipt_revalidation"] = receipt_binding
    metrics["cagm_common_training_binding"] = common_binding
    _validate_persisted_common_training_binding(
        metrics["cagm_common_training_binding"],
        common_binding,
        label=f"current F{fold_index}",
    )
    metrics["verdict"] = _pair_verdict(metrics)
    output_path = Path(args.output_metrics_json).resolve()
    _atomic_rewrite_json(output_path, metrics)
    return metrics


# Public aliases make the exact frozen fairness math independently testable.
fit_frozen_cagm_diagonal_gaussian = _icmt.fit_frozen_diagonal_gaussian
score_frozen_cagm_nll = _icmt.score_frozen_icmt_nll
normalize_cagm_float64 = _icmt._normalize_float64


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
