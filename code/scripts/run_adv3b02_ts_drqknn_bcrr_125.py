#!/usr/bin/env python3
"""Authority-backed ADV3B02 full125 row and dynamic-matrix executor.

The runner opens only the existing p2 authority packages, performs the exact
head-bypass feature path, seals four immutable before/after predictions, and
hands truth joining to the existing independent scorer.  It contains no data
builder, query fit, query role input, or candidate-selection path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import re
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from cvsrffi.stage2_adv3b02_ts_drqknn_bcrr import (
    ADV3B02StateError, ARMS, CANDIDATE, SCENES, append_stage2_c, build_four_arm_states,
    build_four_arm_states_from_dual, domain_weight_audit,
    head_bypass_forward, predict_four_arms, repair_finite_exact_zero_singleton_class_medoid,
    state_receipt, verify_stage2_c_append_receipt, verify_zid_repair_receipt,
)


RECEIVERS = ("20-1", "3-19", "7-14", "7-7", "8-8")
SEEDS = (713102, 713103, 713104, 713105, 713106)
SLICES = ((10, 5), (10, 10), (10, 20), (5, 20), (1, 20))
MATRIX_COUNTS = {"jobs": 125, "scene_slices": 375, "score_rows": 1500, "arm_state_prediction_artifacts": 1000}
LAUNCHER_SCHEMA = "cvs.stage2.adv3b02.full125.artifact_validator.r4_q2f32_bcr3_zidtotal1"
FORMAL_GPU_IDS = tuple(range(8))
ROW_STATES = ("before", "after")
SCENE_METRIC_KEYS = {
    "query_count",
    "old_acc",
    "seen_new_acc",
    "h_old_new",
    "old_to_new_rate",
    "new_to_old_rate",
}


class ADV3B02LauncherError(ValueError):
    pass


class ADV3B02P0Error(ADV3B02LauncherError):
    """A structurally classified protocol or safety failure."""

    def __init__(self, failure_code: str, message: str):
        if failure_code not in P0_FAILURE_CODES:
            raise ValueError("unknown ADV3B02 P0 failure code")
        super().__init__(message)
        self.failure_code = failure_code


ROW_FAILURE_MARKER_PREFIX = "ADV3B02_ROW_FAILURE_JSON="
ROW_FAILURE_SCHEMA = "cvs.stage2.adv3b02.row_failure.v1"
P0_FAILURE_CODES = frozenset(
    {
        "AUTHORITY_OR_PACKAGE_BINDING_FAILURE",
        "CUDA_NAMESPACE_DRIFT",
        "INPUT_HASH_OR_CHECKOUT_DRIFT",
        "INPUT_MISSING_OR_CHECKOUT_DRIFT",
        "MATRIX_PROTOCOL_DRIFT",
        "OUTPUT_OVERWRITE",
        "QUERY_STATE_LEAKAGE",
        "REGISTRY_PROTOCOL_DRIFT",
        "RUN_OWNED_PROCESS_SAFETY_FAILURE",
        "ROW_PROTOCOL_OR_ARTIFACT_DRIFT",
        "STATE_PROTOCOL_DRIFT",
        "SUPPORT_PROTOCOL_DRIFT",
    }
)
TECHNICAL_FAILURE_CODE = "TECHNICAL_EXCEPTION"


def _canon(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")


def _classify_row_exception(exc: Exception) -> tuple[str, bool]:
    if isinstance(exc, ADV3B02P0Error):
        return exc.failure_code, True
    if isinstance(exc, FileExistsError):
        return "OUTPUT_OVERWRITE", True
    if isinstance(exc, FileNotFoundError):
        return "INPUT_MISSING_OR_CHECKOUT_DRIFT", True
    message = str(exc).lower()
    if isinstance(exc, ADV3B02StateError) and "query" in message:
        return "QUERY_STATE_LEAKAGE", True
    if isinstance(exc, ADV3B02LauncherError) and any(
        token in message
        for token in (
            "artifact", "authority", "hash", "overwrite", "protocol", "query",
            "registry", "sha", "unsafe",
        )
    ):
        return "ROW_PROTOCOL_OR_ARTIFACT_DRIFT", True
    return TECHNICAL_FAILURE_CODE, False


def _row_failure_marker_payload(
    *, job_id_value: str, exc: Exception, prediction_count: int
) -> dict[str, Any]:
    failure_code, p0 = _classify_row_exception(exc)
    body = {
        "schema": ROW_FAILURE_SCHEMA,
        "candidate": CANDIDATE,
        "job_id": str(job_id_value),
        "failure_code": failure_code,
        "p0_protocol_or_safety": p0,
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "prediction_artifact_count_at_failure": int(prediction_count),
        "query_rows_used_for_fit": 0,
    }
    return {**body, "receipt_sha256": hashlib.sha256(_canon(body)).hexdigest()}


def _validate_row_failure_marker(
    value: Any, *, expected_job_id: str
) -> dict[str, Any]:
    required = {
        "schema", "candidate", "job_id", "failure_code", "p0_protocol_or_safety",
        "exception_type", "exception_message", "prediction_artifact_count_at_failure",
        "query_rows_used_for_fit", "receipt_sha256",
    }
    payload = dict(value) if isinstance(value, Mapping) else {}
    body = {key: payload[key] for key in payload if key != "receipt_sha256"}
    code = payload.get("failure_code")
    if (
        set(payload) != required
        or payload.get("schema") != ROW_FAILURE_SCHEMA
        or payload.get("candidate") != CANDIDATE
        or payload.get("job_id") != expected_job_id
        or code not in P0_FAILURE_CODES | {TECHNICAL_FAILURE_CODE}
        or payload.get("p0_protocol_or_safety") != (code in P0_FAILURE_CODES)
        or type(payload.get("prediction_artifact_count_at_failure")) is not int
        or payload.get("prediction_artifact_count_at_failure", -1) < 0
        or payload.get("query_rows_used_for_fit") != 0
        or payload.get("receipt_sha256") != hashlib.sha256(_canon(body)).hexdigest()
    ):
        raise ADV3B02LauncherError("structured row failure marker drift")
    return payload


def _read_row_failure_marker(
    log_path: str | Path, *, expected_job_id: str
) -> dict[str, Any] | None:
    lines = Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()
    encoded = [line[len(ROW_FAILURE_MARKER_PREFIX):] for line in lines if line.startswith(ROW_FAILURE_MARKER_PREFIX)]
    if not encoded:
        return None
    if len(encoded) != 1:
        raise ADV3B02LauncherError("multiple structured row failure markers")
    try:
        value = json.loads(encoded[0])
    except json.JSONDecodeError as exc:
        raise ADV3B02LauncherError("structured row failure marker JSON drift") from exc
    return _validate_row_failure_marker(value, expected_job_id=expected_job_id)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
    try:
        os.write(descriptor, payload); os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, stat.S_IREAD)
    return hashlib.sha256(payload).hexdigest()


def write_json_new(path: str | Path, value: Mapping[str, Any]) -> str:
    return _write_new(Path(path), _canon(value) + b"\n")


def write_prediction_new(path: str | Path, *, query_tokens: np.ndarray,
                         scenarios: np.ndarray, predicted_class_handles: np.ndarray) -> str:
    token = np.asarray(query_tokens)
    scene = np.asarray(scenarios)
    predicted = np.asarray(predicted_class_handles)
    if token.ndim != scene.ndim or token.ndim != predicted.ndim or token.ndim != 1 or len(token) != len(scene) or len(token) != len(predicted):
        raise ADV3B02LauncherError("prediction artifact vector layout drift")
    if len(token) == 0 or len(set(token.astype(str).tolist())) != len(token) or any(item not in SCENES for item in scene.astype(str).tolist()):
        raise ADV3B02LauncherError("prediction artifact token/scenario drift")
    import io
    stream = io.BytesIO()
    np.savez_compressed(stream, query_tokens=token.astype(str), scenarios=scene.astype(str),
                        predicted_class_handles=predicted.astype(str))
    return _write_new(Path(path), stream.getvalue())


def job_id(receiver: str, seed: int, k_shot: int, new_class_count: int) -> str:
    return f"adv3b02_r4_q2f32_bcr3_rx_{receiver}_s_{seed}_k_{k_shot}_n_{new_class_count}"


def matrix_jobs(*, run_root: str | Path) -> list[dict[str, Any]]:
    root = Path(run_root)
    jobs = [{"job_id": job_id(receiver, seed, k, new), "receiver": receiver, "seed": seed,
             "k_shot": k, "new_class_count": new, "output_root": str(root / "jobs" / job_id(receiver, seed, k, new)),
             "prediction_artifact_count": len(ARMS) * 2, "scene_slice_count": len(SCENES),
             "score_row_count": len(ARMS) * len(SCENES)}
            for receiver in RECEIVERS for seed in SEEDS for k, new in SLICES]
    if len(jobs) != MATRIX_COUNTS["jobs"] or len({item["job_id"] for item in jobs}) != len(jobs):
        raise ADV3B02LauncherError("frozen full125 job cardinality drift")
    return sorted(jobs, key=lambda item: (-(item["k_shot"] * (10 + item["new_class_count"])), -item["new_class_count"], item["job_id"]))


def prediction_path(row_root: str | Path, state: str, arm: str) -> Path:
    if state not in ("before", "after") or arm not in ARMS:
        raise ADV3B02LauncherError("prediction artifact identity drift")
    return Path(row_root) / "predictions" / state / arm / "prediction_artifact.npz"


def score_path(row_root: str | Path, arm: str) -> Path:
    if arm not in ARMS:
        raise ADV3B02LauncherError("score artifact arm drift")
    return Path(row_root) / "scorer" / f"{arm}.score.json"


def _read_prediction(path: Path) -> Mapping[str, np.ndarray]:
    if not path.is_file() or path.is_symlink():
        raise ADV3B02LauncherError("prediction artifact is absent or unsafe")
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"query_tokens", "scenarios", "predicted_class_handles"}:
            raise ADV3B02LauncherError("prediction artifact allowlist drift")
        values = {name: np.asarray(payload[name]) for name in payload.files}
    token, scene, predicted = values["query_tokens"], values["scenarios"], values["predicted_class_handles"]
    if token.ndim != 1 or scene.shape != token.shape or predicted.shape != token.shape or not len(token):
        raise ADV3B02LauncherError("prediction artifact shape drift")
    if len(set(token.astype(str).tolist())) != len(token) or any(item not in SCENES for item in scene.astype(str).tolist()):
        raise ADV3B02LauncherError("prediction artifact token/scenario drift")
    return values


def _finite_number(value: Any) -> bool:
    return type(value) in (int, float) and np.isfinite(float(value))


def _validate_scene_metrics(
    value: Any, *, state: str, expected_query_count: int
) -> None:
    if type(value) is not dict or set(value) != SCENE_METRIC_KEYS:
        raise ADV3B02LauncherError("score scene metric schema/empty closure drift")
    if (
        type(value["query_count"]) is not int
        or value["query_count"] != expected_query_count
        or expected_query_count <= 0
        or not _finite_number(value["old_acc"])
        or not _finite_number(value["old_to_new_rate"])
    ):
        raise ADV3B02LauncherError("score scene query/old metric closure drift")
    if state == "before":
        if any(
            value[key] is not None
            for key in ("seen_new_acc", "h_old_new", "new_to_old_rate")
        ):
            raise ADV3B02LauncherError("before score unexpectedly covers new classes")
    elif any(
        not _finite_number(value[key])
        for key in ("seen_new_acc", "h_old_new", "new_to_old_rate")
    ):
        raise ADV3B02LauncherError("after score lacks complete paired metrics")


def _validate_runtime_scene_state_receipts(
    value: Any,
) -> dict[tuple[str, str], Mapping[str, Any]]:
    if type(value) is not list or len(value) != len(SCENES) * len(ROW_STATES):
        raise ADV3B02LauncherError("row scene/state runtime receipt closure drift")
    expected_pairs = {(scene, state) for scene in SCENES for state in ROW_STATES}
    seen: set[tuple[str, str]] = set()
    for item in value:
        if type(item) is not dict:
            raise ADV3B02LauncherError("row runtime receipt row drift")
        pair = (item.get("scene"), item.get("state"))
        if pair not in expected_pairs or pair in seen:
            raise ADV3B02LauncherError("row runtime scene/state identity drift")
        seen.add(pair)
        if (
            type(item.get("raw_qknn_sha256")) is not str
            or len(item["raw_qknn_sha256"]) != 64
            or type(item.get("dual_qknn_sha256")) is not str
            or len(item["dual_qknn_sha256"]) != 64
            or type(item.get("branch_qknn_wire_sha256")) is not str
            or item["branch_qknn_wire_sha256"] != item["raw_qknn_sha256"]
            or type(item.get("int8_audit_sha256")) is not str
            or len(item["int8_audit_sha256"]) != 64
            or any(
                type(item.get(name)) is not str
                or len(item[name]) != 64
                or item[name] != item[name].lower()
                or any(character not in "0123456789abcdef" for character in item[name])
                for name in (
                    "branch_actual_bank_binding_sha256",
                    "branch_teacher_support_sha256",
                    "branch_support_repair_receipt_sha256",
                )
            )
            or type(item.get("state_wire_bytes")) is not int
            or item["state_wire_bytes"] <= 0
            or type(item.get("raw_state_bytes")) is not int
            or item["raw_state_bytes"] <= 0
            or type(item.get("dual_domain_state_bytes")) is not int
            or item["dual_domain_state_bytes"] <= 0
            or item["state_wire_bytes"]
            != item["raw_state_bytes"] + item["dual_domain_state_bytes"]
            or type(item.get("fixed_rank")) is not int
            or item["fixed_rank"] != 2
            or type(item.get("active_rank")) is not int
            or not 0 <= item["active_rank"] <= 2
            or not _finite_number(item.get("alpha"))
            or not 0.0 <= float(item["alpha"]) < 0.5
            or not _finite_number(item.get("build_latency_ms"))
            or not _finite_number(item.get("predict_latency_ms"))
            or not _finite_number(item.get("feature_latency_ms"))
            or not _finite_number(item.get("total_latency_ms"))
            or float(item["build_latency_ms"]) < 0.0
            or float(item["predict_latency_ms"]) < 0.0
            or float(item["feature_latency_ms"]) < 0.0
            or not np.isclose(
                float(item["total_latency_ms"]),
                float(item["feature_latency_ms"])
                + float(item["build_latency_ms"])
                + float(item["predict_latency_ms"]),
                rtol=0.0,
                atol=1.0e-9,
            )
            or type(item.get("peak_cuda_memory_bytes")) is not int
            or item["peak_cuda_memory_bytes"] < 0
        ):
            raise ADV3B02LauncherError("row runtime state/resource receipt drift")
        try:
            repair = verify_zid_repair_receipt(item.get("support_repair_receipt"))
        except Exception as exc:
            raise ADV3B02LauncherError("row runtime z_id repair receipt drift") from exc
        if (
            repair["receipt_sha256"] != item["branch_support_repair_receipt_sha256"]
            or repair["unit_output_support_sha256"] != item["branch_teacher_support_sha256"]
        ):
            raise ADV3B02LauncherError("row runtime repair/teacher binding drift")
        delta = item.get("raw_vs_dual")
        weight = item.get("domain_weights")
        if (
            type(delta) is not dict
            or set(delta)
            != {
                "query_rows",
                "argmax_changed_count",
                "score_changed_count",
                "margin_changed_count",
                "max_abs_score_delta",
            }
            or type(weight) is not dict
            or set(weight)
            != {
                "query_class_rows",
                "nonuniform_rows",
                "max_weight_span",
                "mean_weight_span",
            }
            or any(
                type(delta[key]) is not int or delta[key] < 0
                for key in (
                    "query_rows",
                    "argmax_changed_count",
                    "score_changed_count",
                    "margin_changed_count",
                )
            )
            or not _finite_number(delta["max_abs_score_delta"])
            or any(
                type(weight[key]) is not int or weight[key] < 0
                for key in ("query_class_rows", "nonuniform_rows")
            )
            or not _finite_number(weight["max_weight_span"])
            or not _finite_number(weight["mean_weight_span"])
        ):
            raise ADV3B02LauncherError("row raw/dual audit receipt drift")
    if seen != expected_pairs:
        raise ADV3B02LauncherError("row runtime scene/state coverage drift")
    return {
        (str(item["scene"]), str(item["state"])): item for item in value
    }


def _validate_append_receipts(
    value: Any,
    *,
    runtime_by_scene_state: Mapping[
        tuple[str, str], Mapping[str, Any]
    ],
) -> None:
    if type(value) is not dict or set(value) != set(SCENES):
        raise ADV3B02LauncherError("row Stage2-C append receipt scene closure drift")
    for scene in SCENES:
        try:
            append = verify_stage2_c_append_receipt(value[scene])
        except Exception as exc:
            raise ADV3B02LauncherError(
                "row Stage2-C append receipt verification failed"
            ) from exc
        before = runtime_by_scene_state[(scene, "before")]
        after = runtime_by_scene_state[(scene, "after")]
        if (
            append["old_state_sha256"] != before["dual_qknn_sha256"]
            or append["after_state_sha256"] != after["dual_qknn_sha256"]
            or append["after_qknn_wire_sha256"]
            != after["branch_qknn_wire_sha256"]
            or append["after_branch_actual_bank_binding_sha256"]
            != after["branch_actual_bank_binding_sha256"]
            or append["after_branch_teacher_support_sha256"]
            != after["branch_teacher_support_sha256"]
            or append["after_support_repair_receipt_sha256"]
            != after["branch_support_repair_receipt_sha256"]
            or append["after_support_repair_unit_output_sha256"]
            != after["branch_teacher_support_sha256"]
            or append["after_int8_audit_sha256"]
            != after["int8_audit_sha256"]
        ):
            raise ADV3B02LauncherError(
                "row Stage2-C append/common-state binding drift"
            )


def validate_row_artifacts(job: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
    expected = {
        "schema": LAUNCHER_SCHEMA,
        "candidate": CANDIDATE,
        "job_id": job["job_id"],
        "status": "ROW_ARTIFACTS_COMPLETE",
        "query_truth_in_predictor": False,
        "query_rows_used_for_fit": 0,
    }
    if type(receipt) is not dict or any(
        receipt.get(key) != value for key, value in expected.items()
    ):
        raise ADV3B02LauncherError("row receipt identity/protocol closure drift")
    from scripts import run_dssc_zdom_jg_qknn_r4_bcrr_125 as dssc

    try:
        dssc._validate_formal_row_device_namespace_execution(
            receipt.get("device_namespace_execution")
        )
    except Exception as exc:
        raise ADV3B02LauncherError(
            "row CUDA namespace execution evidence drift"
        ) from exc
    runtime_by_scene_state = _validate_runtime_scene_state_receipts(
        receipt.get("scene_state_runtime_receipts")
    )
    _validate_append_receipts(
        receipt.get("append_receipts_by_scene"),
        runtime_by_scene_state=runtime_by_scene_state,
    )

    root = Path(str(job["output_root"]))
    published = receipt.get("prediction_sha256_by_state_arm")
    if type(published) is not dict or set(published) != set(ROW_STATES):
        raise ADV3B02LauncherError("row prediction publication schema drift")
    per_state_identity: dict[
        str, tuple[tuple[str, ...], tuple[str, ...]]
    ] = {}
    scene_counts: dict[str, dict[str, int]] = {}
    prediction_count = 0
    for state in ROW_STATES:
        arms = published.get(state)
        if type(arms) is not dict or set(arms) != set(ARMS):
            raise ADV3B02LauncherError("row arm-state prediction closure drift")
        for arm, digest in arms.items():
            path = prediction_path(root, state, arm)
            if (
                type(digest) is not str
                or len(digest) != 64
                or not path.is_file()
                or path.is_symlink()
                or sha256_file(path) != digest
            ):
                raise ADV3B02LauncherError("prediction artifact/hash closure drift")
            artifact = _read_prediction(path)
            identity = (
                tuple(artifact["query_tokens"].astype(str).tolist()),
                tuple(artifact["scenarios"].astype(str).tolist()),
            )
            prior = per_state_identity.setdefault(state, identity)
            if prior != identity:
                raise ADV3B02LauncherError(
                    "all arms must predict the same immutable query artifact"
                )
            counts = {
                scene: int(np.sum(artifact["scenarios"].astype(str) == scene))
                for scene in SCENES
            }
            if any(count <= 0 for count in counts.values()):
                raise ADV3B02LauncherError(
                    "prediction artifact lacks complete three-scene closure"
                )
            if state in scene_counts and scene_counts[state] != counts:
                raise ADV3B02LauncherError("prediction scene count drift")
            scene_counts[state] = counts
            prediction_count += 1

    scores = receipt.get("score_artifact_sha256")
    if type(scores) is not dict or set(scores) != set(ARMS):
        raise ADV3B02LauncherError("row score publication schema drift")
    score_rows = 0
    for arm, digest in scores.items():
        path = score_path(root, arm)
        if (
            type(digest) is not str
            or len(digest) != 64
            or not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != digest
        ):
            raise ADV3B02LauncherError("score artifact/hash closure drift")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            type(payload) is not dict
            or payload.get("candidate") != CANDIDATE
            or payload.get("arm") != arm
            or payload.get("query_truth_joined_after_prediction") is not True
            or payload.get("query_truth_joined_only_after_immutable_predictions")
            is not True
            or payload.get("query_truth_fed_back_to_predictor") is not False
            or payload.get("before_prediction_sha256")
            != published["before"][arm]
            or payload.get("after_prediction_sha256")
            != published["after"][arm]
        ):
            raise ADV3B02LauncherError("truth-side score artifact closure drift")
        for state in ROW_STATES:
            scored_state = payload.get(state)
            if (
                type(scored_state) is not dict
                or type(scored_state.get("by_scenario")) is not dict
                or set(scored_state["by_scenario"]) != set(SCENES)
                or type(scored_state.get("query_count")) is not int
                or scored_state["query_count"]
                != sum(scene_counts[state].values())
            ):
                raise ADV3B02LauncherError(
                    "score paired state/scenario closure drift"
                )
            for scene in SCENES:
                _validate_scene_metrics(
                    scored_state["by_scenario"][scene],
                    state=state,
                    expected_query_count=scene_counts[state][scene],
                )
        score_rows += len(SCENES)
    if (
        prediction_count != len(ROW_STATES) * len(ARMS)
        or score_rows != len(ARMS) * len(SCENES)
        or receipt.get("prediction_artifact_count") != prediction_count
        or receipt.get("scene_slice_count") != len(SCENES)
        or receipt.get("score_row_count") != score_rows
    ):
        raise ADV3B02LauncherError("row recomputed artifact cardinality drift")


def validate_matrix_artifacts(*, run_root: str | Path) -> Mapping[str, Any]:
    root = Path(run_root)
    runtime_path = root / "matrix_runtime_manifest.json"
    if runtime_path.is_file() and not runtime_path.is_symlink():
        manifest = json.loads(runtime_path.read_text(encoding="utf-8"))
        if (
            type(manifest) is not dict
            or manifest.get("schema") != LAUNCHER_SCHEMA
            or manifest.get("candidate") != CANDIDATE
            or manifest.get("counts") != MATRIX_COUNTS
            or manifest.get("gpu_ids") != list(FORMAL_GPU_IDS)
            or manifest.get("dynamic_workers") != len(FORMAL_GPU_IDS)
            or manifest.get("mapping_policy")
            != "dynamic_free_worker_physical_gpu_to_CUDA_VISIBLE_DEVICES_then_cuda:0"
            or manifest.get("query_truth_in_predictor") is not False
            or manifest.get("launch_capability") is not True
        ):
            raise ADV3B02LauncherError("matrix runtime manifest contract drift")
        gpu_audit = manifest.get("gpu_audit")
        if (
            type(gpu_audit) is not list
            or len(gpu_audit) != len(FORMAL_GPU_IDS)
            or tuple(item.get("physical_gpu_id") for item in gpu_audit)
            != FORMAL_GPU_IDS
            or any(
                type(item.get("device_name")) is not str
                or not item["device_name"]
                or type(item.get("total_memory_bytes")) is not int
                or item["total_memory_bytes"] <= 0
                for item in gpu_audit
            )
        ):
            raise ADV3B02LauncherError("matrix physical GPU audit drift")
        jobs = manifest.get("jobs")
        expected_jobs = {
            item["job_id"]: item for item in matrix_jobs(run_root=root)
        }
        if (
            type(jobs) is not list
            or len(jobs) != len(expected_jobs)
            or len({item.get("job_id") for item in jobs}) != len(jobs)
        ):
            raise ADV3B02LauncherError("matrix runtime job cardinality drift")
        runtime_jobs: dict[str, Mapping[str, Any]] = {}
        for item in jobs:
            ident = item.get("job_id")
            expected_job = expected_jobs.get(ident)
            if (
                expected_job is None
                or any(
                    item.get(key) != expected_job[key]
                    for key in (
                        "receiver",
                        "seed",
                        "k_shot",
                        "new_class_count",
                        "output_root",
                    )
                )
                or type(item.get("cache_manifest")) is not str
                or not item["cache_manifest"]
                or type(item.get("authority_bundle")) is not str
                or not item["authority_bundle"]
            ):
                raise ADV3B02LauncherError("matrix runtime job identity drift")
            runtime_jobs[str(ident)] = item
        completion_path = root / "matrix_runtime_completion.json"
        if not completion_path.is_file() or completion_path.is_symlink():
            raise ADV3B02LauncherError("matrix runtime completion is absent or unsafe")
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        mapping = completion.get("physical_gpu_by_job")
        returncodes = completion.get("returncodes")
        launcher_logs = completion.get("launcher_log_by_job")
        if (
            completion.get("candidate") != CANDIDATE
            or completion.get("status") != "ARTIFACTS_COMPLETE"
            or type(mapping) is not dict
            or set(mapping) != set(runtime_jobs)
            or any(
                type(value) is not int or value not in FORMAL_GPU_IDS
                for value in mapping.values()
            )
            or type(returncodes) is not dict
            or set(returncodes) != set(runtime_jobs)
            or any(type(value) is not int or value != 0 for value in returncodes.values())
            or type(launcher_logs) is not dict
            or set(launcher_logs) != set(runtime_jobs)
        ):
            raise ADV3B02LauncherError("matrix runtime mapping/exit closure drift")
        for ident, log_receipt in launcher_logs.items():
            expected_log = root / "launcher_logs" / f"{ident}.log"
            if (
                type(log_receipt) is not dict
                or log_receipt.get("path") != str(expected_log)
                or type(log_receipt.get("sha256")) is not str
                or len(log_receipt["sha256"]) != 64
                or not expected_log.is_file()
                or expected_log.is_symlink()
                or sha256_file(expected_log) != log_receipt["sha256"]
            ):
                raise ADV3B02LauncherError(
                    "matrix immutable per-row launcher log closure drift"
                )
        receipts = []
        for ident, job in runtime_jobs.items():
            path = Path(str(job["output_root"])) / "row_receipt.json"
            if not path.is_file() or path.is_symlink():
                raise ADV3B02LauncherError("full125 runtime row receipt is missing")
            receipt = json.loads(path.read_text(encoding="utf-8"))
            validate_row_artifacts(job, receipt)
            evidence = receipt["device_namespace_execution"]
            if (
                evidence.get("visible_physical_gpu_id") != mapping[ident]
                or evidence.get("cuda_visible_devices") != str(mapping[ident])
                or evidence.get("requested_logical_device") != "cuda:0"
                or evidence.get("torch_cuda_device_count") != 1
                or evidence.get("torch_cuda_current_device") != 0
            ):
                raise ADV3B02LauncherError(
                    "matrix launcher/row CUDA namespace binding drift"
                )
            receipts.append(receipt)
        aggregate = {
            "jobs": len(receipts),
            "scene_slices": sum(item["scene_slice_count"] for item in receipts),
            "score_rows": sum(item["score_row_count"] for item in receipts),
            "arm_state_prediction_artifacts": sum(
                item["prediction_artifact_count"] for item in receipts
            ),
        }
        if aggregate != MATRIX_COUNTS or completion.get("counts") != aggregate:
            raise ADV3B02LauncherError(
                "full125 runtime artifact cardinality closure failed"
            )
        return {
            "candidate": CANDIDATE,
            "status": "ARTIFACTS_COMPLETE",
            "counts": aggregate,
            "runtime_manifest_validated": True,
        }

    manifest_path = root / "matrix_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ADV3B02LauncherError("matrix manifest is absent or unsafe")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = matrix_jobs(run_root=root)
    if manifest != {"schema": LAUNCHER_SCHEMA, "candidate": CANDIDATE, "counts": MATRIX_COUNTS, "jobs": jobs,
                    "query_truth_in_predictor": False, "launch_capability": False}:
        raise ADV3B02LauncherError("matrix manifest/frozen-plan drift")
    receipts = []
    for job in jobs:
        path = Path(job["output_root"]) / "row_receipt.json"
        if not path.is_file() or path.is_symlink():
            raise ADV3B02LauncherError("full125 row receipt is missing")
        receipt = json.loads(path.read_text(encoding="utf-8"))
        validate_row_artifacts(job, receipt); receipts.append(receipt)
    aggregate = {"jobs": len(receipts), "scene_slices": sum(item["scene_slice_count"] for item in receipts),
                 "score_rows": sum(item["score_row_count"] for item in receipts),
                 "arm_state_prediction_artifacts": sum(item["prediction_artifact_count"] for item in receipts)}
    if aggregate != MATRIX_COUNTS:
        raise ADV3B02LauncherError("full125 artifact cardinality closure failed")
    return {"candidate": CANDIDATE, "status": "ARTIFACTS_COMPLETE", "counts": aggregate}


def write_plan_new(*, run_root: str | Path) -> str:
    root = Path(run_root)
    if root.exists():
        raise ADV3B02LauncherError("matrix plan root must be fresh")
    jobs = matrix_jobs(run_root=root)
    root.mkdir(parents=True)
    return write_json_new(root / "matrix_manifest.json", {"schema": LAUNCHER_SCHEMA, "candidate": CANDIDATE,
                          "counts": MATRIX_COUNTS, "jobs": jobs, "query_truth_in_predictor": False,
                          "launch_capability": False})


def _runtime_feature(model: Any, iq: np.ndarray, *, device: str, checkpoint_sha256: str) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
    """One exact, head-bypassed dual forward over received IQ only."""
    from scripts import run_dssc_zdom_jg_qknn_r4_bcrr_125 as dssc
    tensor = dssc._numpy_float32_tensor(np.asarray(iq, np.float32), device=device)
    z_id, z_dom, receipt = head_bypass_forward(model, tensor, checkpoint_sha256=checkpoint_sha256)
    return (np.asarray(z_id.detach().cpu().tolist(), np.float32),
            np.asarray(z_dom.detach().cpu().tolist(), np.float32), receipt)


def _publish_state_predictions(row_root: Path, state: str, rows: Mapping[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]) -> dict[str, str]:
    published: dict[str, str] = {}
    for arm in ARMS:
        parts = rows[arm]
        if not parts:
            raise ADV3B02LauncherError("state has no prediction rows")
        tokens = np.concatenate([part[0] for part in parts]).astype(str)
        scenes = np.concatenate([part[1] for part in parts]).astype(str)
        predicted = np.concatenate([part[2] for part in parts]).astype(str)
        published[arm] = write_prediction_new(prediction_path(row_root, state, arm), query_tokens=tokens, scenarios=scenes, predicted_class_handles=predicted)
    return published


def _top_margin(scores: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(scores, np.float64), axis=1)
    if ordered.ndim != 2 or ordered.shape[1] < 2:
        raise ADV3B02LauncherError("raw/dual margin audit class closure drift")
    return ordered[:, -1] - ordered[:, -2]


def _runtime_state_receipt(
    *,
    scene: str,
    state_name: str,
    states: Mapping[str, Any],
    logits: Mapping[str, np.ndarray],
    query_zdom: np.ndarray,
    feature_latency_ms: float,
    build_latency_ms: float,
    predict_latency_ms: float,
    peak_cuda_memory_bytes: int,
) -> dict[str, Any]:
    sealed = state_receipt(states)
    raw = np.asarray(logits["M0"], np.float64)
    dual = np.asarray(logits["M_DA"], np.float64)
    if raw.shape != dual.shape or raw.ndim != 2 or not len(raw):
        raise ADV3B02LauncherError("raw/dual runtime audit score closure drift")
    delta = np.abs(raw - dual)
    margin_delta = np.abs(_top_margin(raw) - _top_margin(dual))
    dual_state = states["M_DA"]
    return {
        "scene": scene,
        "state": state_name,
        "raw_qknn_sha256": sealed["raw_qknn_sha256"],
        "dual_qknn_sha256": sealed["dual_qknn_sha256"],
        "branch_qknn_wire_sha256": sealed["branch_qknn_wire_sha256"],
        "branch_actual_bank_binding_sha256": sealed[
            "branch_actual_bank_binding_sha256"
        ],
        "branch_teacher_support_sha256": sealed["branch_teacher_support_sha256"],
        "branch_support_repair_receipt_sha256": sealed[
            "branch_support_repair_receipt_sha256"
        ],
        "support_repair_receipt": sealed["support_repair_receipt"],
        "int8_audit_sha256": sealed["int8_audit_sha256"],
        "raw_state_bytes": int(sealed["raw_state_bytes"]),
        "dual_domain_state_bytes": int(sealed["dual_domain_state_bytes"]),
        "state_wire_bytes": int(sealed["wire_bytes"]),
        "alpha": float(dual_state.domain.alpha),
        "fixed_rank": 2,
        "active_rank": int(np.sum(dual_state.domain.rho > 0.0)),
        "feature_latency_ms": float(feature_latency_ms),
        "build_latency_ms": float(build_latency_ms),
        "predict_latency_ms": float(predict_latency_ms),
        "total_latency_ms": float(
            feature_latency_ms + build_latency_ms + predict_latency_ms
        ),
        "peak_cuda_memory_bytes": int(peak_cuda_memory_bytes),
        "raw_vs_dual": {
            "query_rows": len(raw),
            "argmax_changed_count": int(
                np.sum(np.argmax(raw, axis=1) != np.argmax(dual, axis=1))
            ),
            "score_changed_count": int(np.sum(np.any(delta > 0.0, axis=1))),
            "margin_changed_count": int(np.sum(margin_delta > 0.0)),
            "max_abs_score_delta": float(np.max(delta)),
        },
        "domain_weights": domain_weight_audit(dual_state, query_zdom),
    }


def _score_real_row(row_root: Path, *, truth_sidecar: str | Path, publications: Mapping[str, Mapping[str, str]]) -> dict[str, str]:
    """Open truth only after all eight prediction artifacts have been sealed."""
    from cvsrffi.stage2_diag_cosine_scorer import score_diag_cosine_pair
    scores: dict[str, str] = {}
    for arm in ARMS:
        base = row_root / "scorer" / f"{arm}.base_score.json"
        scored = score_diag_cosine_pair(before_prediction_path=prediction_path(row_root, "before", arm), after_prediction_path=prediction_path(row_root, "after", arm), truth_sidecar_path=truth_sidecar, output_path=base, candidate=CANDIDATE)
        scores[arm] = write_json_new(score_path(row_root, arm), {**scored, "arm": arm,
                                      "query_truth_joined_after_prediction": True,
                                      "before_prediction_sha256": publications["before"][arm],
                                      "after_prediction_sha256": publications["after"][arm]})
    return scores


def run_row(a: argparse.Namespace) -> Mapping[str, Any]:
    """Execute one authority-built row; no caller-supplied support/query NPZ is accepted."""
    from cvsrffi.somph_diagnostic_bundle_loader import load_verified_somph_predictor_bundle
    from scripts import run_dssc_zdom_jg_qknn_r4_bcrr_125 as dssc

    if a.k_shot not in (1, 5, 10) or (a.k_shot, a.new_class_count) not in SLICES:
        raise ADV3B02P0Error(
            "MATRIX_PROTOCOL_DRIFT", "row is outside the frozen full125 slice set"
        )
    try:
        input_hashes = {
            "phase1_checkpoint": sha256_file(a.phase1_checkpoint),
            "sealed_runtime": sha256_file(a.sealed_runtime),
            "package_method_lock": sha256_file(a.package_method_lock),
        }
    except FileNotFoundError as exc:
        raise ADV3B02P0Error(
            "INPUT_MISSING_OR_CHECKOUT_DRIFT", "frozen row input is absent"
        ) from exc
    if input_hashes != {
        "phase1_checkpoint": dssc.CHECKPOINT_SHA256,
        "sealed_runtime": dssc.SEALED_RUNTIME_SHA256,
        "package_method_lock": dssc.SOMPH_PACKAGE_LOCK_SHA256,
    }:
        raise ADV3B02P0Error(
            "INPUT_HASH_OR_CHECKOUT_DRIFT", "frozen row input SHA binding drift"
        )
    device_evidence = dssc._activate_row_device(a.device)
    try:
        dssc._validate_formal_row_device_namespace_execution(device_evidence)
    except Exception as exc:
        raise ADV3B02P0Error(
            "CUDA_NAMESPACE_DRIFT",
            "formal row CUDA namespace execution evidence drift",
        ) from exc
    import torch

    output = Path(a.output_root)
    if output.exists():
        raise ADV3B02P0Error(
            "OUTPUT_OVERWRITE", "row output root must be fresh"
        )
    # The existing authority builder reads only its presealed package lock.  It
    # has a legacy unused DSSC field; bind it to that same immutable lock rather
    # than introducing an ADV3B02 sidecar method-lock input.
    a.dssc_method_lock = a.package_method_lock
    output.mkdir(parents=True)
    try:
        build, runtime = dssc._build_finalized_packages(a, output)
    except Exception as exc:
        raise ADV3B02P0Error(
            "AUTHORITY_OR_PACKAGE_BINDING_FAILURE",
            "authority-backed row package construction failed",
        ) from exc
    be, bm, _ = load_verified_somph_predictor_bundle(runtime["before"]["enrollment"]["enrollment_package_root"], detached_seal_path=runtime["before"]["enrollment"]["enrollment_package_seal"], expected_seal_sha256=runtime["before"]["enrollment"]["enrollment_package_seal_sha256"])
    bq, _ba, _ = load_verified_somph_predictor_bundle(runtime["before"]["enrollment"]["apply_staging_root"], detached_seal_path=runtime["before"]["apply_seal"], expected_seal_sha256=runtime["before"]["apply"]["package_seal_sha256"])
    ae, am, _ = load_verified_somph_predictor_bundle(runtime["after"]["enrollment"]["enrollment_package_root"], detached_seal_path=runtime["after"]["enrollment"]["enrollment_package_seal"], expected_seal_sha256=runtime["after"]["enrollment"]["enrollment_package_seal_sha256"])
    aq, _aa, _ = load_verified_somph_predictor_bundle(runtime["after"]["enrollment"]["apply_staging_root"], detached_seal_path=runtime["after"]["apply_seal"], expected_seal_sha256=runtime["after"]["apply"]["package_seal_sha256"])
    old_registry, registry = dssc._registry(bm), dssc._registry(am)
    if tuple(registry[:len(old_registry)]) != tuple(old_registry):
        raise ADV3B02P0Error(
            "REGISTRY_PROTOCOL_DRIFT",
            "authority before/after registry append closure drift",
        )
    model, exact_receipt = dssc._exact_adv3b02(a.phase1_checkpoint, device=a.device)
    rows = {state: {arm: [] for arm in ARMS} for state in ("before", "after")}
    bypass_receipts: list[Mapping[str, Any]] = []
    runtime_receipts: list[Mapping[str, Any]] = []
    append_receipts: dict[str, Mapping[str, Any]] = {}
    for scene in SCENES:
        torch.cuda.reset_peak_memory_stats(torch.device(a.device))
        old_iq, old_labels, old_tokens = dssc._support(be[scene], tuple(old_registry), a.k_shot)
        before_query, before_query_tokens = dssc._query(bq[scene])
        feature_started = time.perf_counter()
        old_zid, old_zdom, bypass = _runtime_feature(model, old_iq, device=a.device, checkpoint_sha256=exact_receipt["checkpoint_sha256"])
        query_zid, query_zdom, query_bypass = _runtime_feature(model, before_query, device=a.device, checkpoint_sha256=exact_receipt["checkpoint_sha256"])
        before_feature_ms = 1000.0 * (time.perf_counter() - feature_started)
        started = time.perf_counter()
        repaired_old_zid, before_repair_receipt = repair_finite_exact_zero_singleton_class_medoid(
            old_zid, old_labels, tuple(old_registry), old_tokens
        )
        before_states = build_four_arm_states(support_zid=repaired_old_zid, support_zdom=old_zdom, support_labels=old_labels, registered_classes=tuple(old_registry), support_physical_tokens=old_tokens, support_repair_receipt=before_repair_receipt)
        before_build_ms = 1000.0 * (time.perf_counter() - started)
        started = time.perf_counter()
        before_logits = predict_four_arms(before_states, query_zid=query_zid, query_zdom=query_zdom)
        before_predict_ms = 1000.0 * (time.perf_counter() - started)
        runtime_receipts.append(
            _runtime_state_receipt(
                scene=scene,
                state_name="before",
                states=before_states,
                logits=before_logits,
                query_zdom=query_zdom,
                feature_latency_ms=before_feature_ms,
                build_latency_ms=before_build_ms,
                predict_latency_ms=before_predict_ms,
                peak_cuda_memory_bytes=int(
                    torch.cuda.max_memory_allocated(torch.device(a.device))
                ),
            )
        )
        bypass_receipts.extend((bypass, query_bypass))
        for arm in ARMS:
            rows["before"][arm].append((np.asarray(before_query_tokens), np.asarray([scene] * len(before_query_tokens)), np.asarray(old_registry)[np.argmax(before_logits[arm], axis=1)]))
        all_iq, all_labels, all_tokens = dssc._support(ae[scene], tuple(registry), a.k_shot)
        after_query, after_query_tokens = dssc._query(aq[scene])
        new_mask = np.asarray([label not in set(old_registry) for label in all_labels], bool)
        if int(new_mask.sum()) != a.new_class_count * a.k_shot:
            raise ADV3B02P0Error(
                "SUPPORT_PROTOCOL_DRIFT",
                "Stage2-C package lacks exact new-class support",
            )
        torch.cuda.reset_peak_memory_stats(torch.device(a.device))
        feature_started = time.perf_counter()
        all_zid, all_zdom, all_bypass = _runtime_feature(model, all_iq, device=a.device, checkpoint_sha256=exact_receipt["checkpoint_sha256"])
        after_zid, after_zdom, after_bypass = _runtime_feature(model, after_query, device=a.device, checkpoint_sha256=exact_receipt["checkpoint_sha256"])
        after_feature_ms = 1000.0 * (time.perf_counter() - feature_started)
        repaired_all_zid, after_repair_receipt = repair_finite_exact_zero_singleton_class_medoid(
            all_zid, all_labels, tuple(registry), all_tokens
        )
        dual_after, append_receipt = append_stage2_c(before_states["M_DA"], new_support_zid=repaired_all_zid[new_mask], new_support_zdom=all_zdom[new_mask], new_support_labels=tuple(label for label, keep in zip(all_labels, new_mask) if keep), new_registered_classes=tuple(label for label in registry if label not in set(old_registry)), new_support_physical_tokens=tuple(token for token, keep in zip(all_tokens, new_mask) if keep), after_full_teacher_zid=repaired_all_zid, after_full_teacher_physical_tokens=all_tokens, after_support_repair_receipt=after_repair_receipt)
        started = time.perf_counter()
        after_states = build_four_arm_states_from_dual(dual_after)
        after_build_ms = 1000.0 * (time.perf_counter() - started)
        started = time.perf_counter()
        after_logits = predict_four_arms(after_states, query_zid=after_zid, query_zdom=after_zdom)
        after_predict_ms = 1000.0 * (time.perf_counter() - started)
        runtime_receipts.append(
            _runtime_state_receipt(
                scene=scene,
                state_name="after",
                states=after_states,
                logits=after_logits,
                query_zdom=after_zdom,
                feature_latency_ms=after_feature_ms,
                build_latency_ms=after_build_ms,
                predict_latency_ms=after_predict_ms,
                peak_cuda_memory_bytes=int(
                    torch.cuda.max_memory_allocated(torch.device(a.device))
                ),
            )
        )
        append_receipts[scene] = append_receipt
        bypass_receipts.extend((all_bypass, after_bypass))
        for arm in ARMS:
            rows["after"][arm].append((np.asarray(after_query_tokens), np.asarray([scene] * len(after_query_tokens)), np.asarray(registry)[np.argmax(after_logits[arm], axis=1)]))
        if append_receipt["old_q_a_alpha_refit"] or not append_receipt["old_int8_codes_preserved"]:
            raise ADV3B02P0Error(
                "STATE_PROTOCOL_DRIFT", "Stage2-C append-only receipt drift"
            )
    publications = {state: _publish_state_predictions(output, state, rows[state]) for state in ("before", "after")}
    scores = _score_real_row(output, truth_sidecar=build["truth_sidecar"], publications=publications)
    receipt = {"schema": LAUNCHER_SCHEMA, "candidate": CANDIDATE, "job_id": job_id(a.receiver, a.seed, a.k_shot, a.new_class_count), "status": "ROW_ARTIFACTS_COMPLETE", "receiver": a.receiver, "seed": a.seed, "k_shot": a.k_shot, "new_class_count": a.new_class_count, "prediction_artifact_count": 8, "scene_slice_count": 3, "score_row_count": 12, "query_truth_in_predictor": False, "query_rows_used_for_fit": 0, "device_namespace_execution": device_evidence, "exact_checkpoint": exact_receipt, "head_bypass_receipts": bypass_receipts, "scene_state_runtime_receipts": runtime_receipts, "append_receipts_by_scene": append_receipts, "prediction_sha256_by_state_arm": publications, "score_artifact_sha256": scores}
    write_json_new(output / "row_receipt.json", receipt)
    validate_row_artifacts(
        {"job_id": receipt["job_id"], "output_root": str(output)}, receipt
    )
    return receipt


def _runtime_jobs(a: argparse.Namespace) -> list[dict[str, Any]]:
    jobs = []
    for receiver in RECEIVERS:
        leaf = f"rx_{receiver.replace('-', '_')}"
        for seed in SEEDS:
            for k_shot, new_count in SLICES:
                ident = job_id(receiver, seed, k_shot, new_count)
                jobs.append({"job_id": ident, "receiver": receiver, "seed": seed, "k_shot": k_shot, "new_class_count": new_count, "cache_manifest": str(Path(a.cache_root) / leaf / f"seed_{seed}" / "cache_set.json"), "authority_bundle": str(Path(a.authority_root) / f"authority_bundle_{leaf}_seed_{seed}"), "output_root": str(Path(a.run_root) / "jobs" / ident)})
    if len(jobs) != 125: raise ADV3B02LauncherError("runtime full125 job cardinality drift")
    return sorted(jobs, key=lambda item: (-(item["k_shot"] * (10 + item["new_class_count"])), item["job_id"]))


def _audit_formal_physical_gpus() -> list[dict[str, Any]]:
    import torch

    parent_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if parent_visible not in (None, ",".join(str(item) for item in FORMAL_GPU_IDS)):
        raise ADV3B02LauncherError(
            "matrix parent CUDA namespace does not expose physical GPUs 0-7 in order"
        )
    if not torch.cuda.is_available() or int(torch.cuda.device_count()) < len(
        FORMAL_GPU_IDS
    ):
        raise ADV3B02LauncherError(
            "formal matrix requires audited physical CUDA GPUs 0-7"
        )
    result = []
    for physical_gpu_id in FORMAL_GPU_IDS:
        properties = torch.cuda.get_device_properties(physical_gpu_id)
        result.append(
            {
                "physical_gpu_id": physical_gpu_id,
                "device_name": str(properties.name),
                "total_memory_bytes": int(properties.total_memory),
            }
        )
    return result


def _path_is_within(path: str | Path, root: str | Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except ValueError:
        return False
    return True


def _count_prediction_artifacts(output_root: str | Path) -> int:
    prediction_root = Path(output_root) / "predictions"
    if not prediction_root.is_dir():
        return 0
    return sum(
        1
        for item in prediction_root.rglob("prediction_artifact.npz")
        if item.is_file() and not item.is_symlink()
    )


def _normalized_exception_fingerprint(
    exc: Exception, *, prediction_count: int, prefix: str = "parent"
) -> str | None:
    """Build a row-invariant key for an exception with no sealed prediction."""
    if prediction_count:
        return None
    evidence = f"{type(exc).__name__}: {exc}"
    normalized = re.sub(
        r"(?:[A-Za-z]:)?[^\s:]+(?:\\|/)[^\s:]+", "<path>", evidence
    )
    normalized = re.sub(
        r"\b(?:seed|pid|row|job)?[_=-]?\d+\b",
        "#",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return f"{prefix}:" + normalized[:240]


def _preprediction_failure_fingerprint(
    log_path: str | Path, returncode: int, output_root: str | Path
) -> str | None:
    """Return a row-invariant failure key only before prediction publication."""
    output = Path(output_root)
    receipt_path = output / "row_receipt.json"
    prediction_count = _count_prediction_artifacts(output)
    if returncode == 0 and receipt_path.is_file():
        return None
    raw = Path(log_path).read_text(encoding="utf-8", errors="replace")[-12000:]
    if prediction_count:
        return None
    if returncode == 0:
        return "rc0_missing_row_receipt_or_prediction"
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    error_index = next(
        (
            index for index in range(len(lines) - 1, -1, -1)
            if any(
                token in lines[index].lower()
                for token in (
                    "error", "exception", "protocol", "security", "authority", "leak"
                )
            )
        ),
        len(lines) - 1,
    )
    error_line = lines[error_index] if lines else "empty_log"
    frame_line = next(
        (
            lines[index]
            for index in range(error_index - 1, -1, -1)
            if lines[index].lstrip().lower().startswith("file ")
        ),
        "no_stack_frame",
    )
    evidence = frame_line + " | " + error_line
    normalized = re.sub(r"(?:[A-Za-z]:)?[^\s:]+(?:\\|/)[^\s:]+", "<path>", evidence)
    normalized = re.sub(r"\b(?:seed|pid|row|job)?[_=-]?\d+\b", "#", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return "rc=" + str(returncode) + ":" + normalized[:240]


def _validate_run_owned_process_identity(
    *,
    pid: int,
    command: list[str],
    output_root: str,
    matrix_root: str | Path,
    expected_cwd: str | Path,
    observed_cwd: str | Path,
    observed_cmdline: list[str],
    observed_process_group_id: int | None,
) -> dict[str, Any]:
    """Validate immutable ownership evidence before a run-side termination."""
    if type(pid) is not int or pid <= 0:
        raise ADV3B02LauncherError("run-owned PID identity drift")
    if not _path_is_within(output_root, Path(matrix_root) / "jobs"):
        raise ADV3B02LauncherError("run-owned output root escapes matrix jobs root")
    if "--output-root" not in command:
        raise ADV3B02LauncherError("run-owned command lacks output root")
    index = command.index("--output-root")
    if index + 1 >= len(command) or command[index + 1] != output_root:
        raise ADV3B02LauncherError("run-owned command/output root binding drift")
    if Path(observed_cwd).resolve() != Path(expected_cwd).resolve():
        raise ADV3B02LauncherError("run-owned PID CWD binding drift")
    if tuple(observed_cmdline) != tuple(command):
        raise ADV3B02LauncherError("run-owned PID cmdline binding drift")
    if os.name != "nt" and observed_process_group_id != pid:
        raise ADV3B02LauncherError("run-owned PID process-group binding drift")
    return {
        "pid": pid,
        "cwd": str(Path(observed_cwd).resolve()),
        "cmdline": observed_cmdline,
        "cmdline_sha256": hashlib.sha256(_canon(observed_cmdline)).hexdigest(),
        "output_root": str(Path(output_root).resolve()),
        "matrix_root": str(Path(matrix_root).resolve()),
        "process_group_id": observed_process_group_id,
        "ownership_verified": True,
    }


def _process_group_is_alive(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    return True


def _capture_run_owned_process_identity(
    process: subprocess.Popen[bytes],
    *,
    command: list[str],
    output_root: str,
    matrix_root: str | Path,
    expected_cwd: str | Path,
) -> dict[str, Any]:
    """Seal PID/CWD/cmdline/session ownership immediately after Popen."""
    pid = int(process.pid)
    if os.name == "nt":
        return _validate_run_owned_process_identity(
            pid=pid,
            command=command,
            output_root=output_root,
            matrix_root=matrix_root,
            expected_cwd=expected_cwd,
            observed_cwd=expected_cwd,
            observed_cmdline=command,
            observed_process_group_id=None,
        )
    proc_cwd = Path(f"/proc/{pid}/cwd")
    proc_cmd = Path(f"/proc/{pid}/cmdline")
    if not proc_cwd.exists() or not proc_cmd.exists():
        raise ADV3B02LauncherError("run-owned PID evidence absent at launch")
    observed_cmdline = [
        item.decode("utf-8", errors="strict")
        for item in proc_cmd.read_bytes().split(b"\0")
        if item
    ]
    return _validate_run_owned_process_identity(
        pid=pid,
        command=command,
        output_root=output_root,
        matrix_root=matrix_root,
        expected_cwd=expected_cwd,
        observed_cwd=os.readlink(proc_cwd),
        observed_cmdline=observed_cmdline,
        observed_process_group_id=os.getpgid(pid),
    )


def _emergency_cleanup_unverified_spawn(
    process: subprocess.Popen[bytes], *, reason: str
) -> dict[str, Any]:
    """Best-effort cleanup for a Popen created by us before identity sealing."""
    pid = int(process.pid)
    cleanup_error: str | None = None
    escalated = False
    try:
        if os.name == "nt":
            if process.poll() is None:
                completed = subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    check=False,
                    capture_output=True,
                )
                escalated = True
                if completed.returncode != 0 and process.poll() is None:
                    cleanup_error = (
                        "taskkill_returncode=" + str(completed.returncode)
                    )
            if process.poll() is None:
                process.kill()
                escalated = True
        else:
            if _process_group_is_alive(pid):
                os.killpg(pid, signal.SIGKILL)
                escalated = True
        if process.poll() is None:
            process.wait(timeout=15)
    except (ProcessLookupError, ChildProcessError):
        pass
    except Exception as exc:  # retained in the fail-closed receipt
        cleanup_error = type(exc).__name__ + ": " + str(exc)
    group_alive = None if os.name == "nt" else _process_group_is_alive(pid)
    return {
        "pid": pid,
        "termination_strategy": "unverified_spawn_emergency_cleanup",
        "reason": reason,
        "root_exit_code": process.poll(),
        "tree_exit_confirmed": (
            process.poll() is not None
            if os.name == "nt"
            else not bool(group_alive)
        ),
        "escalated_to_kill": escalated,
        "cleanup_error": cleanup_error,
    }


def _terminate_run_owned_process_tree(
    process: subprocess.Popen[bytes],
    entry: Mapping[str, Any],
    *,
    matrix_root: str | Path,
    expected_cwd: str | Path,
) -> dict[str, Any]:
    """Terminate only the independently grouped row process and descendants."""
    pid = int(entry["pid"])
    command = [str(item) for item in entry["cmdline"]]
    output_root = str(entry["output_root"])
    if process.pid != pid:
        raise ADV3B02LauncherError("run-owned Popen/PID binding drift")
    captured = dict(entry.get("ownership_evidence", {}))
    if not captured.get("ownership_verified") or captured.get("pid") != pid:
        raise ADV3B02LauncherError("run-owned launch ownership evidence drift")
    evidence = _validate_run_owned_process_identity(
        pid=pid,
        command=command,
        output_root=output_root,
        matrix_root=matrix_root,
        expected_cwd=expected_cwd,
        observed_cwd=captured.get("cwd", ""),
        observed_cmdline=list(captured.get("cmdline", ())),
        observed_process_group_id=captured.get("process_group_id"),
    )
    if os.name == "nt":
        if process.poll() is not None:
            return {
                **evidence,
                "termination_strategy": "windows_root_already_exited",
                "tree_exit_confirmed": None,
                "escalated_to_kill": False,
            }
        completed = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            check=False,
            capture_output=True,
        )
        if completed.returncode != 0 and process.poll() is None:
            raise ADV3B02LauncherError("run-owned Windows process-tree termination failed")
        process.wait(timeout=15)
        return {
            **evidence,
            "termination_strategy": "windows_taskkill_tree",
            "taskkill_returncode": int(completed.returncode),
            "tree_exit_confirmed": process.poll() is not None,
            "escalated_to_kill": True,
        }
    process_group_id = int(evidence["process_group_id"])
    root_already_exited = process.poll() is not None
    if not root_already_exited:
        live = _capture_run_owned_process_identity(
            process,
            command=command,
            output_root=output_root,
            matrix_root=matrix_root,
            expected_cwd=expected_cwd,
        )
        if live != evidence:
            raise ADV3B02LauncherError("run-owned live/launch ownership evidence drift")
    if not _process_group_is_alive(process_group_id):
        if not root_already_exited:
            process.wait(timeout=15)
        return {
            **evidence,
            "termination_strategy": "posix_session_process_group",
            "root_already_exited": root_already_exited,
            "signal": None,
            "root_exit_code": int(process.returncode),
            "tree_exit_confirmed": True,
            "escalated_to_kill": False,
        }
    os.killpg(process_group_id, signal.SIGTERM)
    escalated = False
    deadline = time.monotonic() + 15.0
    while _process_group_is_alive(process_group_id) and time.monotonic() < deadline:
        time.sleep(0.1)
    if _process_group_is_alive(process_group_id):
        os.killpg(process_group_id, signal.SIGKILL)
        escalated = True
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired as exc:
        raise ADV3B02LauncherError("run-owned process root did not exit") from exc
    final_deadline = time.monotonic() + 5.0
    while _process_group_is_alive(process_group_id) and time.monotonic() < final_deadline:
        time.sleep(0.1)
    tree_alive = _process_group_is_alive(process_group_id)
    if tree_alive:
        raise ADV3B02LauncherError("run-owned process tree survived termination")
    return {
        **evidence,
        "termination_strategy": "posix_session_process_group",
        "root_already_exited": root_already_exited,
        "signal": "SIGKILL" if escalated else "SIGTERM",
        "root_exit_code": int(process.returncode),
        "tree_exit_confirmed": True,
        "escalated_to_kill": escalated,
    }


def _termination_receipt_confirmed(receipt: Mapping[str, Any] | None) -> bool:
    return bool(
        isinstance(receipt, Mapping)
        and receipt.get("tree_exit_confirmed") is True
        and not receipt.get("cleanup_error")
    )


def _best_effort_posix_sentinel_cleanup(
    *,
    process_group_ids: tuple[int | None, ...],
    processes: tuple[subprocess.Popen[bytes] | None, ...],
) -> tuple[str, ...]:
    """Clean self-created sentinel processes without masking later cleanup."""
    cleanup_errors: list[str] = []
    for process_group_id in process_group_ids:
        if process_group_id is None:
            continue
        try:
            if _process_group_is_alive(process_group_id):
                os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception as exc:
            cleanup_errors.append(
                "process_group_"
                + str(process_group_id)
                + ": "
                + type(exc).__name__
                + ": "
                + str(exc)
            )
    for process in processes:
        if process is None:
            continue
        try:
            if process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        except Exception as exc:
            cleanup_errors.append(
                "process_"
                + str(process.pid)
                + ": "
                + type(exc).__name__
                + ": "
                + str(exc)
            )
    return tuple(cleanup_errors)


def run_posix_root_grandchild_unrelated_sentinel() -> dict[str, Any]:
    """Prove POSIX cleanup kills only a run-owned root process group.

    This is a frozen-runner self-check for hosts that do not ship pytest.  It
    creates two independent process groups: a run-owned root which creates a
    grandchild and exits, and an unrelated sentinel.  The checked termination
    path must remove the surviving grandchild without affecting the sentinel.
    """
    if os.name == "nt":
        raise ADV3B02LauncherError("posix-sentinel requires a POSIX host")
    root_process: subprocess.Popen[bytes] | None = None
    sentinel: subprocess.Popen[bytes] | None = None
    target_process_group_id: int | None = None
    sentinel_process_group_id: int | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="adv3b02-posix-sentinel-") as temp_dir:
            temp_root = Path(temp_dir)
            matrix_root = temp_root / "matrix"
            output_root = matrix_root / "jobs" / "tree_row"
            child_pid_path = temp_root / "grandchild.pid"
            child_script = "import time;time.sleep(60)"
            root_script = (
                "import pathlib,subprocess,sys,time;"
                "child=subprocess.Popen([sys.executable,'-c',sys.argv[2]]);"
                "pathlib.Path(sys.argv[1]).write_text(str(child.pid),encoding='utf-8');"
                "time.sleep(1)"
            )
            command = [
                sys.executable,
                "-c",
                root_script,
                str(child_pid_path),
                child_script,
                "--output-root",
                str(output_root),
            ]
            sentinel = subprocess.Popen(
                [sys.executable, "-c", child_script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            sentinel_process_group_id = os.getpgid(sentinel.pid)
            root_process = subprocess.Popen(
                command,
                cwd=Path.cwd(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            # start_new_session makes the root PID its own process-group ID.
            # Keep that self-created group for finally cleanup before /proc
            # ownership capture, which can itself fail or race with root exit.
            target_process_group_id = root_process.pid
            ownership = _capture_run_owned_process_identity(
                root_process,
                command=command,
                output_root=str(output_root),
                matrix_root=matrix_root,
                expected_cwd=Path.cwd(),
            )
            if int(ownership["process_group_id"]) != target_process_group_id:
                raise ADV3B02LauncherError("posix-sentinel root lacks independent process group")
            if os.getpgid(sentinel.pid) != sentinel_process_group_id:
                raise ADV3B02LauncherError("posix-sentinel unrelated sentinel lacks independent process group")
            if sentinel_process_group_id == target_process_group_id:
                raise ADV3B02LauncherError("posix-sentinel process groups are not independent")
            deadline = time.monotonic() + 5.0
            while not child_pid_path.is_file() and time.monotonic() < deadline:
                time.sleep(0.05)
            if not child_pid_path.is_file():
                raise ADV3B02LauncherError("posix-sentinel grandchild PID was not published")
            grandchild_pid = int(child_pid_path.read_text(encoding="utf-8"))
            if os.getpgid(grandchild_pid) != target_process_group_id:
                raise ADV3B02LauncherError("posix-sentinel grandchild escaped target process group")
            if root_process.wait(timeout=5) != 0:
                raise ADV3B02LauncherError("posix-sentinel root did not exit cleanly")
            if not _process_group_is_alive(target_process_group_id):
                raise ADV3B02LauncherError("posix-sentinel grandchild did not survive root exit")
            receipt = _terminate_run_owned_process_tree(
                root_process,
                {
                    "pid": root_process.pid,
                    "cmdline": command,
                    "output_root": str(output_root),
                    "ownership_evidence": ownership,
                },
                matrix_root=matrix_root,
                expected_cwd=Path.cwd(),
            )
            if not receipt.get("root_already_exited"):
                raise ADV3B02LauncherError("posix-sentinel did not observe root already exited")
            if not receipt.get("tree_exit_confirmed"):
                raise ADV3B02LauncherError("posix-sentinel target process group survived cleanup")
            if _process_group_is_alive(target_process_group_id):
                raise ADV3B02LauncherError("posix-sentinel target process group remains alive")
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                try:
                    os.kill(grandchild_pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.05)
            else:
                raise ADV3B02LauncherError("posix-sentinel grandchild survived target cleanup")
            if sentinel.poll() is not None:
                raise ADV3B02LauncherError("posix-sentinel unrelated sentinel was terminated")
            return {
                "mode": "posix-sentinel",
                "root_already_exited": True,
                "tree_exit_confirmed": True,
                "target_process_group_id": target_process_group_id,
                "grandchild_pid": grandchild_pid,
                "unrelated_sentinel_alive": True,
            }
    finally:
        primary_exception_active = sys.exc_info()[0] is not None
        cleanup_errors = _best_effort_posix_sentinel_cleanup(
            process_group_ids=(target_process_group_id, sentinel_process_group_id),
            processes=(root_process, sentinel),
        )
        if cleanup_errors:
            cleanup_message = "posix-sentinel cleanup failed: " + "; ".join(
                cleanup_errors
            )
            if not primary_exception_active:
                raise ADV3B02LauncherError(cleanup_message)
            primary_exception = sys.exc_info()[1]
            try:
                add_note = getattr(primary_exception, "add_note", None)
                if callable(add_note):
                    add_note(cleanup_message)
                else:
                    sys.stderr.write(cleanup_message + "\n")
                    sys.stderr.flush()
            except Exception:
                # The primary failure must remain visible even if audit output fails.
                pass


def _terminate_run_owned_process_tree_checked(
    process: subprocess.Popen[bytes],
    entry: Mapping[str, Any],
    *,
    matrix_root: str | Path,
    expected_cwd: str | Path,
) -> dict[str, Any]:
    """Make every ownership or tree-termination failure an explicit P0."""
    try:
        receipt = _terminate_run_owned_process_tree(
            process,
            entry,
            matrix_root=matrix_root,
            expected_cwd=expected_cwd,
        )
    except Exception as exc:
        raise ADV3B02P0Error(
            "RUN_OWNED_PROCESS_SAFETY_FAILURE",
            "run-owned process-tree termination raised: "
            + type(exc).__name__
            + ": "
            + str(exc),
        ) from exc
    if not _termination_receipt_confirmed(receipt):
        raise ADV3B02P0Error(
            "RUN_OWNED_PROCESS_SAFETY_FAILURE",
            "run-owned process-tree termination was not confirmed",
        )
    return dict(receipt)


def run_matrix(a: argparse.Namespace) -> Mapping[str, Any]:
    root = Path(a.run_root)
    if root.exists(): raise ADV3B02LauncherError("matrix root must be fresh")
    try:
        gpu_ids = tuple(int(value) for value in a.gpu_ids.split(","))
    except ValueError as exc:
        raise ADV3B02LauncherError("GPU list drift") from exc
    if gpu_ids != FORMAL_GPU_IDS:
        raise ADV3B02LauncherError("formal matrix GPU list must be exactly 0-7")
    gpu_audit = _audit_formal_physical_gpus()
    jobs = _runtime_jobs(a); root.mkdir(parents=True)
    launcher_log_root = root / "launcher_logs"
    launcher_log_root.mkdir()
    parent_failure_root = root / "matrix_parent_failures"
    parent_failure_root.mkdir()
    write_json_new(root / "matrix_runtime_manifest.json", {"schema": LAUNCHER_SCHEMA, "candidate": CANDIDATE, "jobs": jobs, "counts": MATRIX_COUNTS, "gpu_ids": list(gpu_ids), "gpu_audit": gpu_audit, "dynamic_workers": len(gpu_ids), "mapping_policy": "dynamic_free_worker_physical_gpu_to_CUDA_VISIBLE_DEVICES_then_cuda:0", "query_truth_in_predictor": False, "launch_capability": True})
    available: queue.Queue[int] = queue.Queue()
    for gpu in gpu_ids: available.put(gpu)
    stop_event = threading.Event()
    active_children: dict[str, dict[str, Any]] = {}
    started_job_ids: set[str] = set()
    active_lock = threading.Lock()

    def persist_parent_failure(
        *,
        job: Mapping[str, Any],
        gpu: int | None,
        marker: Mapping[str, Any],
        fingerprint: str | None,
        child_pid: int | None,
        termination: Mapping[str, Any] | None,
        launcher_log: Mapping[str, Any] | None,
    ) -> dict[str, str]:
        path = parent_failure_root / f"{job['job_id']}.json"
        payload = {
            "schema": LAUNCHER_SCHEMA,
            "candidate": CANDIDATE,
            "job_id": str(job["job_id"]),
            "gpu": gpu,
            "failure_marker": dict(marker),
            "failure_fingerprint": fingerprint,
            "child_pid": child_pid,
            "termination": termination,
            "launcher_log": launcher_log,
        }
        return {"path": str(path), "sha256": write_json_new(path, payload)}

    def one(job: Mapping[str, Any]) -> tuple[
        str,
        int,
        int,
        Mapping[str, Any] | None,
        str | None,
        bool,
        int | None,
        Mapping[str, Any] | None,
        Mapping[str, Any] | None,
        Mapping[str, Any] | None,
    ]:
        gpu = available.get()
        process: subprocess.Popen[bytes] | None = None
        ownership: Mapping[str, Any] | None = None
        termination: Mapping[str, Any] | None = None
        log_created = False
        log_path = launcher_log_root / f"{job['job_id']}.log"
        try:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "row",
                "--cache-manifest",
                job["cache_manifest"],
                "--authority-bundle",
                job["authority_bundle"],
                "--authority-commit-sha256",
                sha256_file(Path(job["authority_bundle"]) / "COMMIT.json"),
                "--phase1-checkpoint",
                a.phase1_checkpoint,
                "--sealed-runtime",
                a.sealed_runtime,
                "--package-method-lock",
                a.package_method_lock,
                "--output-root",
                job["output_root"],
                "--receiver",
                job["receiver"],
                "--seed",
                str(job["seed"]),
                "--k-shot",
                str(job["k_shot"]),
                "--new-class-count",
                str(job["new_class_count"]),
                "--device",
                "cuda:0",
            ]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            descriptor = os.open(
                log_path,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_BINARY", 0),
                0o600,
            )
            log_created = True
            with os.fdopen(descriptor, "wb") as log_handle:
                startup = {"start_new_session": os.name != "nt"}
                if os.name == "nt": startup["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                process = subprocess.Popen(
                    command,
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    **startup,
                )
                with active_lock:
                    started_job_ids.add(str(job["job_id"]))
                try:
                    ownership = _capture_run_owned_process_identity(
                        process,
                        command=command,
                        output_root=str(job["output_root"]),
                        matrix_root=root,
                        expected_cwd=Path.cwd(),
                    )
                except Exception as exc:
                    termination = _emergency_cleanup_unverified_spawn(
                        process, reason="launch_ownership_capture_failed"
                    )
                    raise ADV3B02P0Error(
                        "RUN_OWNED_PROCESS_SAFETY_FAILURE",
                        "run-owned launch ownership capture failed",
                    ) from exc
                with active_lock:
                    active_children[str(job["job_id"])] = {
                        "pid": int(process.pid), "gpu": gpu, "cmdline": command,
                        "output_root": job["output_root"],
                        "ownership_evidence": ownership,
                    }
                while process.poll() is None:
                    if stop_event.wait(0.25):
                        with active_lock:
                            entry = dict(active_children[str(job["job_id"])])
                        termination = _terminate_run_owned_process_tree_checked(
                            process,
                            entry,
                            matrix_root=root,
                            expected_cwd=Path.cwd(),
                        )
                        break
                code = process.wait()
                if termination is None:
                    with active_lock:
                        entry = dict(active_children[str(job["job_id"])])
                    termination = _terminate_run_owned_process_tree_checked(
                        process,
                        entry,
                        matrix_root=root,
                        expected_cwd=Path.cwd(),
                    )
                if code == 0:
                    try:
                        receipt = json.loads((Path(job["output_root"]) / "row_receipt.json").read_text(encoding="utf-8"))
                        validate_row_artifacts(job, receipt)
                    except Exception as exc:
                        parent_exc = ADV3B02P0Error(
                            "ROW_PROTOCOL_OR_ARTIFACT_DRIFT",
                            "parent row artifact validation failed: "
                            + type(exc).__name__
                            + ": "
                            + str(exc),
                        )
                        marker = _row_failure_marker_payload(
                            job_id_value=str(job["job_id"]),
                            exc=parent_exc,
                            prediction_count=_count_prediction_artifacts(
                                job["output_root"]
                            ),
                        )
                        log_handle.write(
                            (
                                "\nROW_ARTIFACT_VALIDATION_ERROR: "
                                + repr(exc)
                                + "\n"
                                + ROW_FAILURE_MARKER_PREFIX
                                + _canon(marker).decode("utf-8")
                                + "\n"
                            ).encode("utf-8")
                        )
                        log_handle.flush()
                        code = 97
            os.chmod(log_path, stat.S_IREAD)
            with active_lock:
                active_children.pop(str(job["job_id"]), None)
            try:
                failure_marker = _read_row_failure_marker(
                    log_path, expected_job_id=str(job["job_id"])
                )
            except Exception as exc:
                raise ADV3B02P0Error(
                    "ROW_PROTOCOL_OR_ARTIFACT_DRIFT",
                    "parent could not validate the structured row failure marker",
                ) from exc
            fingerprint = _preprediction_failure_fingerprint(
                log_path, int(code), str(job["output_root"])
            )
            p0 = bool(
                failure_marker is not None
                and failure_marker["p0_protocol_or_safety"]
            )
            return (
                str(job["job_id"]),
                int(code),
                gpu,
                {"path": str(log_path), "sha256": sha256_file(log_path)},
                fingerprint,
                p0,
                int(process.pid),
                termination,
                failure_marker,
                None,
            )
        except Exception as exc:
            prediction_count = _count_prediction_artifacts(job["output_root"])
            cleanup_safety_failure: str | None = None
            needs_cleanup = process is not None and (
                process.poll() is None
                or (
                    os.name != "nt"
                    and _process_group_is_alive(int(process.pid))
                )
            )
            if needs_cleanup and process is not None:
                if ownership is not None:
                    entry = {
                        "pid": int(process.pid),
                        "gpu": gpu,
                        "cmdline": list(ownership["cmdline"]),
                        "output_root": str(job["output_root"]),
                        "ownership_evidence": ownership,
                    }
                    try:
                        termination = _terminate_run_owned_process_tree(
                            process,
                            entry,
                            matrix_root=root,
                            expected_cwd=Path.cwd(),
                        )
                        if not _termination_receipt_confirmed(termination):
                            cleanup_safety_failure = (
                                "verified termination receipt was not confirmed"
                            )
                    except Exception as cleanup_exc:
                        cleanup_safety_failure = (
                            "verified termination raised: "
                            + type(cleanup_exc).__name__
                            + ": "
                            + str(cleanup_exc)
                        )
                        termination = _emergency_cleanup_unverified_spawn(
                            process,
                            reason=(
                                "verified_tree_cleanup_failed:"
                                + type(cleanup_exc).__name__
                                + ":"
                                + str(cleanup_exc)
                            ),
                        )
                else:
                    termination = _emergency_cleanup_unverified_spawn(
                        process, reason="parent_worker_exception_before_ownership"
                    )
                    cleanup_safety_failure = (
                        "process ownership was unavailable during cleanup"
                    )
            if termination is not None and not _termination_receipt_confirmed(
                termination
            ):
                cleanup_safety_failure = (
                    cleanup_safety_failure
                    or "emergency termination receipt was not confirmed"
                )
            failure_exc: Exception = exc
            if cleanup_safety_failure is not None and not (
                isinstance(exc, ADV3B02P0Error)
                and exc.failure_code == "RUN_OWNED_PROCESS_SAFETY_FAILURE"
            ):
                failure_exc = ADV3B02P0Error(
                    "RUN_OWNED_PROCESS_SAFETY_FAILURE",
                    cleanup_safety_failure,
                )
            marker = _row_failure_marker_payload(
                job_id_value=str(job["job_id"]),
                exc=failure_exc,
                prediction_count=prediction_count,
            )
            fingerprint = _normalized_exception_fingerprint(
                failure_exc, prediction_count=prediction_count
            )
            log_receipt: Mapping[str, Any] | None = None
            if log_created and log_path.is_file() and not log_path.is_symlink():
                os.chmod(log_path, stat.S_IREAD)
                log_receipt = {
                    "path": str(log_path),
                    "sha256": sha256_file(log_path),
                }
            parent_receipt = persist_parent_failure(
                job=job,
                gpu=gpu,
                marker=marker,
                fingerprint=fingerprint,
                child_pid=None if process is None else int(process.pid),
                termination=termination,
                launcher_log=log_receipt,
            )
            return (
                str(job["job_id"]),
                98,
                gpu,
                log_receipt,
                fingerprint,
                bool(marker["p0_protocol_or_safety"]),
                None if process is None else int(process.pid),
                termination,
                marker,
                parent_receipt,
            )
        finally:
            with active_lock:
                active_children.pop(str(job["job_id"]), None)
            available.put(gpu)
    result: dict[str, int] = {}
    physical_gpu_by_job: dict[str, int] = {}
    launcher_log_by_job: dict[str, Mapping[str, Any]] = {}
    fingerprints: dict[str, list[str]] = {}
    child_pid_by_job: dict[str, int] = {}
    termination_receipts: dict[str, Mapping[str, Any]] = {}
    row_failure_markers: dict[str, Mapping[str, Any]] = {}
    parent_failure_receipts: dict[str, Mapping[str, Any]] = {}
    launched = completed = succeeded = failed = cancelled_pending = 0
    systemic: str | None = None
    systemic_detected_after_submitted: int | None = None
    iterator = iter(jobs)
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        pending: dict[Any, Mapping[str, Any]] = {}
        for _ in range(min(len(gpu_ids), len(jobs))):
            item = next(iterator)
            pending[pool.submit(one, item)] = item
        launched = len(pending)
        while pending:
            done, _ = wait(set(pending), return_when=FIRST_COMPLETED)
            for future in done:
                job = pending.pop(future)
                if future.cancelled():
                    continue
                try:
                    (
                        ident,
                        returncode,
                        gpu,
                        log_receipt,
                        fingerprint,
                        p0,
                        child_pid,
                        termination,
                        failure_marker,
                        parent_failure_receipt,
                    ) = future.result()
                except Exception as exc:
                    ident = str(job["job_id"])
                    returncode = 99
                    gpu = -1
                    log_receipt = None
                    fingerprint = _normalized_exception_fingerprint(
                        exc,
                        prediction_count=_count_prediction_artifacts(
                            job["output_root"]
                        ),
                        prefix="coordinator",
                    )
                    coordinator_exc = ADV3B02P0Error(
                        "ROW_PROTOCOL_OR_ARTIFACT_DRIFT",
                        "matrix worker escaped structured result: "
                        + type(exc).__name__
                        + ": "
                        + str(exc),
                    )
                    failure_marker = _row_failure_marker_payload(
                        job_id_value=ident,
                        exc=coordinator_exc,
                        prediction_count=_count_prediction_artifacts(
                            job["output_root"]
                        ),
                    )
                    p0 = True
                    child_pid = None
                    termination = None
                    parent_failure_receipt = persist_parent_failure(
                        job=job,
                        gpu=None,
                        marker=failure_marker,
                        fingerprint=fingerprint,
                        child_pid=None,
                        termination=None,
                        launcher_log=None,
                    )
                completed += 1; result[ident] = returncode
                if gpu >= 0:
                    physical_gpu_by_job[ident] = gpu
                if log_receipt is not None:
                    launcher_log_by_job[ident] = log_receipt
                if child_pid is not None: child_pid_by_job[ident] = child_pid
                if termination is not None: termination_receipts[ident] = termination
                if failure_marker is not None: row_failure_markers[ident] = failure_marker
                if parent_failure_receipt is not None:
                    parent_failure_receipts[ident] = parent_failure_receipt
                if returncode == 0:
                    succeeded += 1
                else:
                    failed += 1
                    if p0 and systemic is None:
                        systemic = "P0:" + str(
                            failure_marker["failure_code"]
                            if failure_marker is not None
                            else "ROW_PROTOCOL_OR_ARTIFACT_DRIFT"
                        )
                        systemic_detected_after_submitted = launched
                        stop_event.set()
                    elif fingerprint is not None:
                        fingerprints.setdefault(fingerprint, []).append(ident)
                        if len(set(fingerprints[fingerprint])) >= 2 and systemic is None:
                            systemic = fingerprint
                            systemic_detected_after_submitted = launched
                            stop_event.set()
            if systemic is None:
                while len(pending) < len(gpu_ids):
                    try:
                        item = next(iterator)
                        pending[pool.submit(one, item)] = item
                        launched += 1
                    except StopIteration:
                        break
            else:
                # Futures not yet running have no process and are safe to cancel.
                for future in pending:
                    if future.cancel():
                        cancelled_pending += 1
    partial_counts = {
        "jobs": 0,
        "scene_slices": 0,
        "score_rows": 0,
        "arm_state_prediction_artifacts": 0,
    }
    for job in jobs:
        if result.get(str(job["job_id"])) != 0:
            continue
        receipt_path = Path(job["output_root"]) / "row_receipt.json"
        if not receipt_path.is_file():
            continue
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        partial_counts["jobs"] += 1
        partial_counts["scene_slices"] += int(receipt["scene_slice_count"])
        partial_counts["score_rows"] += int(receipt["score_row_count"])
        partial_counts["arm_state_prediction_artifacts"] += int(receipt["prediction_artifact_count"])
    health = {"submitted": launched, "launched": len(started_job_ids), "completed": completed, "succeeded": succeeded, "failed": failed,
              "prediction_count": partial_counts["arm_state_prediction_artifacts"],
              "fingerprints": fingerprints, "systemic_fingerprint": systemic,
              "systemic_detected_after_submitted": systemic_detected_after_submitted,
              "child_pid_by_job": child_pid_by_job, "termination_receipts": termination_receipts,
              "row_failure_markers": row_failure_markers,
              "parent_failure_receipts": parent_failure_receipts,
              "cancelled_pending": cancelled_pending,
              "never_submitted": len(jobs) - launched}
    if systemic is not None:
        write_json_new(root / "matrix_runtime_completion.json", {"candidate": CANDIDATE, "status": "STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE", "performance_status": "NO_PERFORMANCE_RESULT", "counts": partial_counts, "returncodes": result, "physical_gpu_by_job": physical_gpu_by_job, "launcher_log_by_job": launcher_log_by_job, "health": health})
        raise ADV3B02LauncherError("systemic pre-prediction row failure; dispatch stopped")
    if any(result.values()):
        write_json_new(root / "matrix_runtime_completion.json", {"candidate": CANDIDATE, "status": "TECHNICAL_FAILURE", "performance_status": "NO_PERFORMANCE_RESULT", "counts": partial_counts, "returncodes": result, "physical_gpu_by_job": physical_gpu_by_job, "launcher_log_by_job": launcher_log_by_job, "health": health})
        raise ADV3B02LauncherError("one or more full125 rows failed")
    receipts = []
    for job in jobs:
        receipt = json.loads((Path(job["output_root"]) / "row_receipt.json").read_text(encoding="utf-8")); validate_row_artifacts(job, receipt); receipts.append(receipt)
    counts = {"jobs": len(receipts), "scene_slices": sum(item["scene_slice_count"] for item in receipts), "score_rows": sum(item["score_row_count"] for item in receipts), "arm_state_prediction_artifacts": sum(item["prediction_artifact_count"] for item in receipts)}
    if counts != MATRIX_COUNTS: raise ADV3B02LauncherError("full125 runtime artifact cardinality drift")
    write_json_new(root / "matrix_runtime_completion.json", {"candidate": CANDIDATE, "status": "ARTIFACTS_COMPLETE", "counts": counts, "returncodes": result, "physical_gpu_by_job": physical_gpu_by_job, "launcher_log_by_job": launcher_log_by_job, "health": health})
    return validate_matrix_artifacts(run_root=root)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    plan = sub.add_parser("plan"); plan.add_argument("--run-root", required=True)
    validate = sub.add_parser("validate"); validate.add_argument("--run-root", required=True)
    sub.add_parser("posix-sentinel")
    def common(item: argparse.ArgumentParser) -> None:
        item.add_argument("--phase1-checkpoint", required=True); item.add_argument("--sealed-runtime", required=True)
        item.add_argument("--package-method-lock", required=True)
    row = sub.add_parser("row"); common(row)
    row.add_argument("--cache-manifest", required=True); row.add_argument("--authority-bundle", required=True)
    row.add_argument("--authority-commit-sha256", required=True); row.add_argument("--output-root", required=True)
    row.add_argument("--receiver", required=True); row.add_argument("--seed", type=int, required=True)
    row.add_argument("--k-shot", type=int, required=True); row.add_argument("--new-class-count", type=int, required=True)
    row.add_argument("--device", required=True)
    matrix = sub.add_parser("matrix"); common(matrix)
    matrix.add_argument("--cache-root", required=True); matrix.add_argument("--authority-root", required=True)
    matrix.add_argument("--run-root", required=True); matrix.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()
    if args.mode == "plan": result = {"manifest_sha256": write_plan_new(run_root=args.run_root), "counts": MATRIX_COUNTS}
    elif args.mode == "validate": result = validate_matrix_artifacts(run_root=args.run_root)
    elif args.mode == "posix-sentinel": result = run_posix_root_grandchild_unrelated_sentinel()
    elif args.mode == "row":
        try:
            result = run_row(args)
        except Exception as exc:
            prediction_root = Path(args.output_root) / "predictions"
            prediction_count = sum(
                1
                for item in prediction_root.rglob("prediction_artifact.npz")
                if item.is_file() and not item.is_symlink()
            ) if prediction_root.is_dir() else 0
            marker = _row_failure_marker_payload(
                job_id_value=job_id(
                    args.receiver, args.seed, args.k_shot, args.new_class_count
                ),
                exc=exc,
                prediction_count=prediction_count,
            )
            sys.stderr.write(
                ROW_FAILURE_MARKER_PREFIX + _canon(marker).decode("utf-8") + "\n"
            )
            sys.stderr.flush()
            raise
    else: result = run_matrix(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
