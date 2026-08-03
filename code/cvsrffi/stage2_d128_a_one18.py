"""Truth-free, single-candidate D128-A-ONE18 prediction closure.

This is deliberately a small successor to the stopped D127 release path.  It
does not create a generic runner, merge A/B/C assets, or perform selection.
It consumes the already sealed D127 18-pair plan and target-package materializer,
loads exactly one A=`DA-A-FSRG-time_fuse` Phase1 bundle, and seals the four
same-row arms needed by the one-shot falsifier.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from . import stage2_d127_checkpoint_hooks as checkpoint_hooks
from . import stage2_d127_da_candidates as da
from . import stage2_d127_phase1_release as phase1_release
from . import stage2_d127_s0_entry as entry
from . import stage2_d127_s0_package_adapter as adapter


PREDICTION_SCHEMA = "cvs.stage2.d128.a.one18.paired_prediction.v1"
PAIR_MANIFEST_SCHEMA = "cvs.stage2.d128.a.one18.pair_manifest.v1"
PHASE1_ASSET_RECEIPT_SCHEMA = "cvs.stage2.d128.a.one18.phase1_asset_receipt.v1"
CANDIDATE_ID = da.CANDIDATE_A
ARM_IDS = entry.ARM_IDS
STATES = ("before", "after")
ROW_COUNT = adapter.S0_ROW_COUNT
FORBIDDEN_NORMALIZED_KEYS = frozenset(
    {"truth", "querytruth", "role", "roles", "queryrole", "quota", "classquota", "globalreassignment"}
)


class D128AOne18Error(ValueError):
    """Raised when the minimal A-only release closure drifts."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise D128AOne18Error(message)


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise D128AOne18Error("D128 canonical JSON value is invalid") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: Any, name: str) -> str:
    _require(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{name} must be a lowercase SHA256",
    )
    return value


def _text(value: Any, name: str) -> str:
    _require(type(value) is str and bool(value), f"{name} must be nonempty text")
    return value


def _normalized_key(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _reject_forbidden(value: Any, name: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _normalized_key(key) in FORBIDDEN_NORMALIZED_KEYS:
                raise D128AOne18Error(f"{name} contains forbidden truth/role/quota field: {key}")
            _reject_forbidden(item, f"{name}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _reject_forbidden(item, f"{name}[{index}]")


def _strings(value: Any, name: str, *, unique: bool = True) -> tuple[str, ...]:
    _require(isinstance(value, list) and bool(value), f"{name} must be a nonempty list")
    result = tuple(_text(item, f"{name}[{index}]") for index, item in enumerate(value))
    _require(not unique or len(result) == len(set(result)), f"{name} contains duplicates")
    return result


def _opaque_root(values: Sequence[str]) -> str:
    frozen = tuple(str(value) for value in values)
    _require(bool(frozen) and len(frozen) == len(set(frozen)) and all(frozen), "opaque root input drift")
    return canonical_sha256(sorted(frozen))


def _ordered_subset(left: Sequence[str], right: Sequence[str]) -> bool:
    iterator = iter(right)
    return all(any(candidate == value for candidate in iterator) for value in left)


def _validate_d127_plan(plan: Mapping[str, Any]) -> None:
    try:
        adapter._validate_prepared_plan(plan)
    except Exception as exc:
        raise D128AOne18Error("D128 prepared-plan closure drift") from exc


def _validate_asset_receipt(value: Any, *, plan: Mapping[str, Any]) -> None:
    expected_fields = {
        "schema", "manifest_sha256", "candidate_id", "method_lock_sha256", "checkpoint_sha256",
        "source_binding", "qknn_lock_binding", "episode_manifest_sha256", "episode_contract_sha256",
        "candidate_asset", "asset_receipt_sha256",
    }
    _require(isinstance(value, Mapping) and set(value) == expected_fields, "D128 A asset receipt closure drift")
    unsigned = dict(value)
    receipt = _sha(unsigned.pop("asset_receipt_sha256", None), "D128 A asset receipt")
    _require(canonical_sha256(unsigned) == receipt, "D128 A asset receipt digest drift")
    expected = plan["phase1_asset_expected_binding"]
    _require(
        value["schema"] == PHASE1_ASSET_RECEIPT_SCHEMA
        and value["candidate_id"] == CANDIDATE_ID
        and value["method_lock_sha256"] == expected["method_lock_sha256"]
        and value["checkpoint_sha256"] == expected["checkpoint_sha256"]
        and value["source_binding"] == expected["source_binding"]
        and value["qknn_lock_binding"] == expected["qknn_lock_binding"],
        "D128 A asset receipt lineage drift",
    )
    for field in ("manifest_sha256", "episode_manifest_sha256", "episode_contract_sha256"):
        _sha(value[field], f"D128 A asset receipt {field}")
    candidate_asset = value["candidate_asset"]
    _require(
        isinstance(candidate_asset, Mapping)
        and candidate_asset.get("candidate_id") == CANDIDATE_ID
        and candidate_asset.get("persistent_fp32_sidecar") is False,
        "D128 A asset receipt candidate closure drift",
    )


def load_d128_a_single_candidate_asset(
    *,
    bundle_dir: str | Path,
    expected_manifest_sha256: str,
    prepared_plan: Mapping[str, Any],
    device: torch.device | str,
) -> tuple[Any, dict[str, Any]]:
    """Read exactly one sealed A bundle; merged A/B/C bundles are rejected."""

    _validate_d127_plan(prepared_plan)
    _sha(expected_manifest_sha256, "D128 A expected asset manifest SHA256")
    try:
        _root, manifest, manifest_sha, _episode, assets_by_id = phase1_release._load_bundle_directory(
            bundle_dir, expected_manifest_sha256=expected_manifest_sha256
        )
    except Exception as exc:
        raise D128AOne18Error("D128 A single-candidate bundle load failed") from exc
    expected = prepared_plan["phase1_asset_expected_binding"]
    _require(
        manifest_sha == expected_manifest_sha256
        and manifest.get("bundle_kind") == "single_candidate"
        and manifest.get("candidate_ids") == [CANDIDATE_ID]
        and tuple(assets_by_id) == (CANDIDATE_ID,),
        "D128 requires exactly one A single-candidate bundle",
    )
    for field in ("method_lock_sha256", "checkpoint_sha256", "source_binding", "qknn_lock_binding"):
        _require(manifest.get(field) == expected[field], f"D128 A asset {field} lineage drift")
    try:
        asset = assets_by_id[CANDIDATE_ID].decode(device=torch.device(device))
    except Exception as exc:
        raise D128AOne18Error("D128 A asset decode failed") from exc
    receipt: dict[str, Any] = {
        "schema": PHASE1_ASSET_RECEIPT_SCHEMA,
        "manifest_sha256": manifest_sha,
        "candidate_id": CANDIDATE_ID,
        "method_lock_sha256": manifest["method_lock_sha256"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "source_binding": manifest["source_binding"],
        "qknn_lock_binding": manifest["qknn_lock_binding"],
        "episode_manifest_sha256": manifest["episode_manifest_sha256"],
        "episode_contract_sha256": manifest["episode_contract_sha256"],
        "candidate_asset": manifest["candidate_assets"][CANDIDATE_ID],
    }
    receipt["asset_receipt_sha256"] = canonical_sha256(receipt)
    _validate_asset_receipt(receipt, plan=prepared_plan)
    return asset, receipt


def materialize_d128_a_one18_rows(
    *,
    method_lock_path: str | Path,
    expected_method_lock_sha256: str,
    d106_context_path: str | Path,
    expected_d106_context_sha256: str,
    prepared_plan: Mapping[str, Any],
    device: torch.device | str = "cpu",
) -> adapter.D127S0PreparedPackageRows:
    """Re-open only the fixed D127 package materialization and bind it to its plan."""

    _validate_d127_plan(prepared_plan)
    try:
        prepared = adapter.materialize_d127_s0_package_rows(
            method_lock_path=method_lock_path,
            expected_method_lock_sha256=expected_method_lock_sha256,
            d106_context_path=d106_context_path,
            expected_d106_context_sha256=expected_d106_context_sha256,
            device=device,
        )
        adapter._assert_prepared_matches_plan(prepared, prepared_plan)
    except Exception as exc:
        raise D128AOne18Error("D128 fixed package materialization/plan binding failed") from exc
    return prepared


def load_d128_checkpoint(
    *, checkpoint_path: str | Path, prepared_plan: Mapping[str, Any], device: torch.device | str
) -> tuple[Any, Mapping[str, Any]]:
    """Load the frozen checkpoint and bind its content hash to the prepared plan."""

    _validate_d127_plan(prepared_plan)
    try:
        model, receipt = checkpoint_hooks.load_d127_frozen_checkpoint(checkpoint_path, device=device)
    except Exception as exc:
        raise D128AOne18Error("D128 frozen checkpoint load failed") from exc
    _require(
        isinstance(receipt, Mapping) and receipt.get("checkpoint_sha256") == prepared_plan["checkpoint_sha256"],
        "D128 checkpoint/prepared-plan SHA256 drift",
    )
    return model, receipt


def _validate_worker_payload(
    value: Any,
    *,
    state: str,
    plan: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], int, int]:
    expected_fields = {
        "schema", "candidate_id", "evaluation_scope", "truth_loaded", "row_count", "rows_complete",
        "query_rows_used_for_fit", "query_state_updates", "query_selection_count", "phase2_optimizer_steps",
        "resource", "rows", "prediction_sha256",
    }
    _require(isinstance(value, Mapping) and set(value) == expected_fields, f"D128 {state} worker field closure drift")
    _reject_forbidden(value, f"D128 {state} worker")
    unsigned = dict(value)
    digest = _sha(unsigned.pop("prediction_sha256", None), f"D128 {state} worker prediction SHA256")
    _require(entry._sha256(unsigned) == digest, f"D128 {state} worker digest drift")
    _require(
        value["schema"] == entry.LOCAL_WORKER_SCHEMA
        and value["candidate_id"] == CANDIDATE_ID
        and value["truth_loaded"] is False
        and value["row_count"] == ROW_COUNT
        and value["rows_complete"] is True
        and all(value[name] == 0 for name in ("query_rows_used_for_fit", "query_state_updates", "query_selection_count", "phase2_optimizer_steps")),
        f"D128 {state} worker access/count drift",
    )
    rows = value["rows"]
    _require(isinstance(rows, list) and len(rows) == ROW_COUNT, f"D128 {state} worker row coverage drift")
    resource = value["resource"]
    _require(isinstance(resource, Mapping), f"D128 {state} worker resource is missing")
    total_forwards = resource.get("total_id_backbone_forwards")
    total_query_rows = resource.get("total_query_rows")
    _require(type(total_forwards) is int and total_forwards >= 0, f"D128 {state} worker forward receipt drift")
    _require(type(total_query_rows) is int and total_query_rows > 0, f"D128 {state} worker query receipt drift")
    for index, (row, binding) in enumerate(zip(rows, plan["pair_bindings"], strict=True)):
        _require(isinstance(row, Mapping), f"D128 {state} row[{index}] must be an object")
        expected_row_fields = {
            "row_id", "receiver_id", "k_shot", "scene", "opaque_query_ids", "arms",
            "joint_receipt", "hook_receipt", "da_resource",
        }
        _require(set(row) == expected_row_fields, f"D128 {state} row[{index}] field closure drift")
        expected = binding[state]
        _require(
            row["row_id"] == binding["row_id"]
            and row["receiver_id"] == binding["receiver"]
            and row["k_shot"] == binding["k_shot"]
            and row["scene"] == binding["scene"],
            f"D128 {state} row[{index}] identity drift",
        )
        query_ids = _strings(row["opaque_query_ids"], f"D128 {state} row[{index}] opaque query IDs")
        _require(
            len(query_ids) == expected["query_token_count"]
            and canonical_sha256(list(query_ids)) == expected["query_token_ordered_sha256"]
            and _opaque_root(query_ids) == expected["query_token_root_sha256"],
            f"D128 {state} row[{index}] query-root drift",
        )
        arms = row["arms"]
        _require(
            isinstance(arms, Mapping) and len(arms) == len(ARM_IDS) and set(arms) == set(ARM_IDS),
            f"D128 {state} row[{index}] four-arm closure drift",
        )
        registry: tuple[str, ...] | None = None
        for arm_id in ARM_IDS:
            arm = arms[arm_id]
            expected_arm_fields = {"arm_id", "representation", "head", "classes", "logits", "predictions", "receipt"}
            _require(isinstance(arm, Mapping) and set(arm) == expected_arm_fields, f"D128 {state} row[{index}] {arm_id} field closure drift")
            _require(arm["arm_id"] == arm_id, f"D128 {state} row[{index}] {arm_id} identity drift")
            classes = _strings(arm["classes"], f"D128 {state} row[{index}] {arm_id} classes")
            registry = classes if registry is None else registry
            _require(classes == registry, f"D128 {state} row[{index}] registry drift")
            predictions = _strings(arm["predictions"], f"D128 {state} row[{index}] {arm_id} predictions", unique=False)
            _require(len(predictions) == len(query_ids) and all(label in classes for label in predictions), f"D128 {state} row[{index}] {arm_id} prediction drift")
            logits = arm["logits"]
            _require(isinstance(logits, list) and len(logits) == len(query_ids), f"D128 {state} row[{index}] {arm_id} logit row drift")
            for logit_row in logits:
                _require(isinstance(logit_row, list) and len(logit_row) == len(classes), f"D128 {state} row[{index}] {arm_id} logit class drift")
                _require(all(type(item) in (int, float) and math.isfinite(float(item)) for item in logit_row), f"D128 {state} row[{index}] {arm_id} nonfinite logits")
            receipt = arm["receipt"]
            _require(isinstance(receipt, Mapping), f"D128 {state} row[{index}] {arm_id} receipt drift")
            _require(
                all(receipt.get(name) == 0 for name in ("query_rows_used_for_fit", "query_state_updates", "query_selection_count")),
                f"D128 {state} row[{index}] {arm_id} query access drift",
            )
        _require(registry is not None and _opaque_root(registry) == expected["registered_class_root_sha256"], f"D128 {state} row[{index}] registry root drift")
        if binding["k_shot"] == 1:
            _require(
                arms["M0"]["predictions"] == arms["M_L92"]["predictions"]
                and arms["M0"]["logits"] == arms["M_L92"]["logits"]
                and arms["M_DA"]["predictions"] == arms["M_JOINT"]["predictions"]
                and arms["M_DA"]["logits"] == arms["M_JOINT"]["logits"],
                f"D128 {state} K1 alias drift",
            )
    return list(rows), total_forwards, total_query_rows


def _build_pair_manifest(
    *, plan: Mapping[str, Any], before_worker: Mapping[str, Any], after_worker: Mapping[str, Any]
) -> dict[str, Any]:
    before_rows, _, _ = _validate_worker_payload(before_worker, state="before", plan=plan)
    after_rows, _, _ = _validate_worker_payload(after_worker, state="after", plan=plan)
    rows: list[dict[str, Any]] = []
    for binding, before, after in zip(plan["pair_bindings"], before_rows, after_rows, strict=True):
        before_ids = _strings(before["opaque_query_ids"], "D128 before query IDs")
        after_ids = _strings(after["opaque_query_ids"], "D128 after query IDs")
        _require(_ordered_subset(before_ids, after_ids), "D128 before query IDs are not an ordered after subset")
        old_classes = _strings(before["arms"]["M0"]["classes"], "D128 old classes")
        after_classes = _strings(after["arms"]["M0"]["classes"], "D128 after classes")
        _require(after_classes[: len(old_classes)] == old_classes and len(after_classes) > len(old_classes), "D128 old/new registry prefix drift")
        formal = binding["formal_d92_reference"]
        rows.append(
            {
                "row_id": binding["row_id"],
                "receiver_id": binding["receiver"],
                "k_shot": binding["k_shot"],
                "scene": binding["scene"],
                "old_classes": list(old_classes),
                "new_classes": list(after_classes[len(old_classes) :]),
                "before_query_ids_sha256": canonical_sha256(list(before_ids)),
                "after_query_ids_sha256": canonical_sha256(list(after_ids)),
                "formal_d92_source_job_id": formal["source_d92_job_id"],
                "formal_d92_retry2_manifest_sha256": formal["d92_retry2_manifest_sha256"],
            }
        )
    manifest: dict[str, Any] = {
        "schema": PAIR_MANIFEST_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "arm_ids": list(ARM_IDS),
        "method_lock_sha256": plan["method_lock_sha256"],
        "checkpoint_sha256": plan["checkpoint_sha256"],
        "d106_context_sha256": plan["d106_context_sha256"],
        "qknn_lock_digests": plan["qknn_lock_digests"],
        "prepared_plan_sha256": plan["prepared_plan_sha256"],
        "prefix_receipt_sha256": plan["prefix_receipt_sha256"],
        "row_pair_count": ROW_COUNT,
        "rows": rows,
    }
    manifest["pair_manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def _validate_pair_manifest(value: Any, *, plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    expected_fields = {
        "schema", "candidate_id", "arm_ids", "method_lock_sha256", "checkpoint_sha256", "d106_context_sha256",
        "qknn_lock_digests", "prepared_plan_sha256", "prefix_receipt_sha256", "row_pair_count", "rows",
        "pair_manifest_sha256",
    }
    _require(isinstance(value, Mapping) and set(value) == expected_fields, "D128 pair manifest field closure drift")
    unsigned = dict(value)
    digest = _sha(unsigned.pop("pair_manifest_sha256", None), "D128 pair manifest SHA256")
    _require(canonical_sha256(unsigned) == digest, "D128 pair manifest digest drift")
    _require(
        value["schema"] == PAIR_MANIFEST_SCHEMA
        and value["candidate_id"] == CANDIDATE_ID
        and value["arm_ids"] == list(ARM_IDS)
        and value["method_lock_sha256"] == plan["method_lock_sha256"]
        and value["checkpoint_sha256"] == plan["checkpoint_sha256"]
        and value["d106_context_sha256"] == plan["d106_context_sha256"]
        and value["qknn_lock_digests"] == plan["qknn_lock_digests"]
        and value["prepared_plan_sha256"] == plan["prepared_plan_sha256"]
        and value["prefix_receipt_sha256"] == plan["prefix_receipt_sha256"]
        and value["row_pair_count"] == ROW_COUNT,
        "D128 pair manifest lineage/count drift",
    )
    rows = value["rows"]
    _require(isinstance(rows, list) and len(rows) == ROW_COUNT, "D128 pair manifest row coverage drift")
    expected_row_fields = {
        "row_id", "receiver_id", "k_shot", "scene", "old_classes", "new_classes", "before_query_ids_sha256",
        "after_query_ids_sha256", "formal_d92_source_job_id", "formal_d92_retry2_manifest_sha256",
    }
    for index, (row, binding) in enumerate(zip(rows, plan["pair_bindings"], strict=True)):
        _require(isinstance(row, Mapping) and set(row) == expected_row_fields, f"D128 pair manifest row[{index}] closure drift")
        _require(
            row["row_id"] == binding["row_id"]
            and row["receiver_id"] == binding["receiver"]
            and row["k_shot"] == binding["k_shot"]
            and row["scene"] == binding["scene"],
            f"D128 pair manifest row[{index}] identity drift",
        )
        _strings(row["old_classes"], f"D128 pair manifest row[{index}] old classes")
        _strings(row["new_classes"], f"D128 pair manifest row[{index}] new classes")
        _require(not set(row["old_classes"]).intersection(row["new_classes"]), f"D128 pair manifest row[{index}] old/new overlap")
        _sha(row["before_query_ids_sha256"], f"D128 pair manifest row[{index}] before query SHA256")
        _sha(row["after_query_ids_sha256"], f"D128 pair manifest row[{index}] after query SHA256")
        formal = binding["formal_d92_reference"]
        _require(
            row["formal_d92_source_job_id"] == formal["source_d92_job_id"]
            and row["formal_d92_retry2_manifest_sha256"] == formal["d92_retry2_manifest_sha256"],
            f"D128 pair manifest row[{index}] formal D92 locator drift",
        )
    return list(rows)


def build_d128_a_one18_prediction(
    *,
    prepared_plan: Mapping[str, Any],
    before_worker: Mapping[str, Any],
    after_worker: Mapping[str, Any],
    phase1_asset_manifest_sha256: str,
    phase1_asset_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind two A-only D127 local worker states into one D128 prediction file."""

    _validate_d127_plan(prepared_plan)
    _sha(phase1_asset_manifest_sha256, "D128 A asset manifest SHA256")
    _validate_asset_receipt(phase1_asset_receipt, plan=prepared_plan)
    _require(phase1_asset_receipt["manifest_sha256"] == phase1_asset_manifest_sha256, "D128 A asset receipt/manifest drift")
    _validate_worker_payload(before_worker, state="before", plan=prepared_plan)
    _validate_worker_payload(after_worker, state="after", plan=prepared_plan)
    pair_manifest = _build_pair_manifest(plan=prepared_plan, before_worker=before_worker, after_worker=after_worker)
    before_forwards = int(before_worker["resource"]["total_id_backbone_forwards"])
    after_forwards = int(after_worker["resource"]["total_id_backbone_forwards"])
    before_queries = int(before_worker["resource"]["total_query_rows"])
    after_queries = int(after_worker["resource"]["total_query_rows"])
    payload: dict[str, Any] = {
        "schema": PREDICTION_SCHEMA,
        "evaluation_scope": "D128_A_ONE18_FAST_FALSIFIER_NON_PROMOTABLE",
        "candidate_id": CANDIDATE_ID,
        "truth_loaded": False,
        "method_lock_sha256": prepared_plan["method_lock_sha256"],
        "checkpoint_sha256": prepared_plan["checkpoint_sha256"],
        "d106_context_sha256": prepared_plan["d106_context_sha256"],
        "qknn_lock_digests": prepared_plan["qknn_lock_digests"],
        "phase1_asset_manifest_sha256": phase1_asset_manifest_sha256,
        "phase1_asset_receipt": dict(phase1_asset_receipt),
        "prepared_plan_sha256": prepared_plan["prepared_plan_sha256"],
        "prefix_receipt_sha256": prepared_plan["prefix_receipt_sha256"],
        "row_pair_count": ROW_COUNT,
        "state_row_count": ROW_COUNT * len(STATES),
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "phase2_optimizer_steps": 0,
        "pair_manifest": pair_manifest,
        "states": {"before": dict(before_worker), "after": dict(after_worker)},
        "physical_execution": {
            "candidate_workers": 1,
            "candidate_id": CANDIDATE_ID,
            "state_workers": 2,
            "total_id_backbone_forwards": before_forwards + after_forwards,
            "total_query_rows": before_queries + after_queries,
        },
    }
    payload["prediction_sha256"] = canonical_sha256(payload)
    validate_d128_a_one18_prediction(payload, prepared_plan=prepared_plan)
    return payload


def run_d128_a_one18_prediction(
    *,
    model: Any,
    asset: Any,
    prepared: adapter.D127S0PreparedPackageRows,
    prepared_plan: Mapping[str, Any],
    phase1_asset_manifest_sha256: str,
    phase1_asset_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Run only A over both states; B/C workers and any merge are impossible here."""

    _validate_d127_plan(prepared_plan)
    try:
        adapter._assert_prepared_matches_plan(prepared, prepared_plan)
        before = entry._run_d127_s0_candidate_worker(
            model=model, candidate_id=CANDIDATE_ID, asset=asset, rows=tuple(item.row for item in prepared.before)
        )
        after = entry._run_d127_s0_candidate_worker(
            model=model, candidate_id=CANDIDATE_ID, asset=asset, rows=tuple(item.row for item in prepared.after)
        )
    except Exception as exc:
        raise D128AOne18Error("D128 A one-shot worker execution failed") from exc
    return build_d128_a_one18_prediction(
        prepared_plan=prepared_plan,
        before_worker=before,
        after_worker=after,
        phase1_asset_manifest_sha256=phase1_asset_manifest_sha256,
        phase1_asset_receipt=phase1_asset_receipt,
    )


def validate_d128_a_one18_prediction(
    prediction: Mapping[str, Any], *, prepared_plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail closed before a scorer may open any truth-side artifact."""

    _validate_d127_plan(prepared_plan)
    expected_fields = {
        "schema", "evaluation_scope", "candidate_id", "truth_loaded", "method_lock_sha256", "checkpoint_sha256",
        "d106_context_sha256", "qknn_lock_digests", "phase1_asset_manifest_sha256", "phase1_asset_receipt",
        "prepared_plan_sha256", "prefix_receipt_sha256", "row_pair_count", "state_row_count",
        "query_rows_used_for_fit", "query_state_updates", "query_selection_count", "phase2_optimizer_steps",
        "pair_manifest", "states", "physical_execution", "prediction_sha256",
    }
    _require(isinstance(prediction, Mapping) and set(prediction) == expected_fields, "D128 prediction field closure drift")
    _reject_forbidden(prediction, "D128 prediction")
    unsigned = dict(prediction)
    digest = _sha(unsigned.pop("prediction_sha256", None), "D128 prediction SHA256")
    _require(canonical_sha256(unsigned) == digest, "D128 prediction digest drift")
    _require(
        prediction["schema"] == PREDICTION_SCHEMA
        and prediction["candidate_id"] == CANDIDATE_ID
        and prediction["truth_loaded"] is False
        and prediction["method_lock_sha256"] == prepared_plan["method_lock_sha256"]
        and prediction["checkpoint_sha256"] == prepared_plan["checkpoint_sha256"]
        and prediction["d106_context_sha256"] == prepared_plan["d106_context_sha256"]
        and prediction["qknn_lock_digests"] == prepared_plan["qknn_lock_digests"]
        and prediction["prepared_plan_sha256"] == prepared_plan["prepared_plan_sha256"]
        and prediction["prefix_receipt_sha256"] == prepared_plan["prefix_receipt_sha256"]
        and prediction["row_pair_count"] == ROW_COUNT
        and prediction["state_row_count"] == ROW_COUNT * len(STATES)
        and all(prediction[name] == 0 for name in ("query_rows_used_for_fit", "query_state_updates", "query_selection_count", "phase2_optimizer_steps")),
        "D128 prediction lineage/access closure drift",
    )
    _sha(prediction["phase1_asset_manifest_sha256"], "D128 prediction asset manifest SHA256")
    _validate_asset_receipt(prediction["phase1_asset_receipt"], plan=prepared_plan)
    _require(prediction["phase1_asset_receipt"]["manifest_sha256"] == prediction["phase1_asset_manifest_sha256"], "D128 prediction asset receipt/manifest drift")
    manifest_rows = _validate_pair_manifest(prediction["pair_manifest"], plan=prepared_plan)
    states = prediction["states"]
    _require(isinstance(states, Mapping) and set(states) == set(STATES), "D128 prediction state closure drift")
    before_rows, before_forwards, before_queries = _validate_worker_payload(states["before"], state="before", plan=prepared_plan)
    after_rows, after_forwards, after_queries = _validate_worker_payload(states["after"], state="after", plan=prepared_plan)
    all_after: set[str] = set()
    for index, (manifest_row, before, after) in enumerate(zip(manifest_rows, before_rows, after_rows, strict=True)):
        before_ids = _strings(before["opaque_query_ids"], f"D128 before row[{index}] query IDs")
        after_ids = _strings(after["opaque_query_ids"], f"D128 after row[{index}] query IDs")
        _require(not all_after.intersection(after_ids), "D128 after query IDs are reused across rows")
        all_after.update(after_ids)
        _require(
            canonical_sha256(list(before_ids)) == manifest_row["before_query_ids_sha256"]
            and canonical_sha256(list(after_ids)) == manifest_row["after_query_ids_sha256"]
            and _ordered_subset(before_ids, after_ids),
            f"D128 paired row[{index}] query receipt drift",
        )
        old_classes = _strings(before["arms"]["M0"]["classes"], f"D128 before row[{index}] old classes")
        after_classes = _strings(after["arms"]["M0"]["classes"], f"D128 after row[{index}] classes")
        _require(
            tuple(manifest_row["old_classes"]) == old_classes
            and tuple(manifest_row["new_classes"]) == after_classes[len(old_classes) :]
            and after_classes[: len(old_classes)] == old_classes,
            f"D128 paired row[{index}] registry drift",
        )
    physical = prediction["physical_execution"]
    expected_physical_fields = {
        "candidate_workers", "candidate_id", "state_workers", "total_id_backbone_forwards", "total_query_rows",
    }
    _require(isinstance(physical, Mapping) and set(physical) == expected_physical_fields, "D128 physical execution closure drift")
    _require(
        physical["candidate_workers"] == 1
        and physical["candidate_id"] == CANDIDATE_ID
        and physical["state_workers"] == 2
        and physical["total_id_backbone_forwards"] == before_forwards + after_forwards
        and physical["total_query_rows"] == before_queries + after_queries,
        "D128 physical execution receipt drift",
    )
    return {
        "status": "D128_A_ONE18_TRUTH_FREE_PREDICTION_VALIDATED",
        "candidate_id": CANDIDATE_ID,
        "row_pair_count": ROW_COUNT,
        "state_row_count": ROW_COUNT * len(STATES),
        "truth_loaded": False,
        "prediction_sha256": prediction["prediction_sha256"],
        "pair_manifest_sha256": prediction["pair_manifest"]["pair_manifest_sha256"],
    }


def write_d128_a_one18_prediction_exclusive(
    path: str | Path, payload: Mapping[str, Any], *, prepared_plan: Mapping[str, Any]
) -> Path:
    validate_d128_a_one18_prediction(payload, prepared_plan=prepared_plan)
    target = Path(path)
    _require(not target.is_symlink(), "D128 prediction output cannot be a symlink")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as stream:
            stream.write(canonical_bytes(dict(payload)) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise D128AOne18Error("D128 prediction output already exists") from exc
    return target


def load_d128_a_one18_prediction(
    path: str | Path, *, expected_sha256: str, prepared_plan: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
    source = Path(path)
    _require(source.is_file() and not source.is_symlink(), "D128 prediction must be a regular file")
    expected = _sha(expected_sha256, "D128 expected prediction file SHA256")
    observed = _sha256_file(source)
    _require(observed == expected, "D128 prediction file SHA256 mismatch")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise D128AOne18Error("D128 prediction is not valid UTF-8 JSON") from exc
    _require(type(value) is dict, "D128 prediction must contain a JSON object")
    validate_d128_a_one18_prediction(value, prepared_plan=prepared_plan)
    return value, observed


__all__ = [
    "ARM_IDS", "CANDIDATE_ID", "D128AOne18Error", "FORBIDDEN_NORMALIZED_KEYS", "PAIR_MANIFEST_SCHEMA",
    "PHASE1_ASSET_RECEIPT_SCHEMA", "PREDICTION_SCHEMA", "ROW_COUNT", "STATES", "build_d128_a_one18_prediction",
    "canonical_bytes", "canonical_sha256", "load_d128_a_one18_prediction", "load_d128_a_single_candidate_asset",
    "load_d128_checkpoint", "materialize_d128_a_one18_rows", "run_d128_a_one18_prediction",
    "validate_d128_a_one18_prediction", "write_d128_a_one18_prediction_exclusive",
]
