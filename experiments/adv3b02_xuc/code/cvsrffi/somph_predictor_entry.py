"""Fixed isolated-process entry logic for SOMP-H enrollment and apply."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from cvsrffi.phase2_runtime_contract import PHASE2_FULL_CONTRACT
from cvsrffi.somph_head_artifact import publish_somph_head_artifact
from cvsrffi.somph_prediction_artifact import publish_somph_prediction_artifact
from cvsrffi.somph_diagnostic_bundle_loader import (
    load_verified_somph_head_capsule,
    load_verified_somph_predictor_bundle,
    preflight_somph_predictor_bundle,
)
from cvsrffi.somph_predictor_bundle import (
    APPLY_ONLY,
    ENROLLMENT_ONLY,
)
from cvsrffi.somph_predictor_runtime import (
    apply_somph_heads,
    canonical_sha256,
    enroll_somph_heads,
)
from cvsrffi.somph_runtime_request import (
    SOMPH_APPLY_BATCH_SIZE,
    validate_somph_apply_request,
    validate_somph_enrollment_request,
)
from cvsrffi.stage2_predictor_runtime import (
    load_json_artifact_same_fd,
    load_torchscript_backbone_same_fd,
)


ENROLLMENT_RECEIPT_NAME = "enrollment_resource_receipt.json"
APPLY_RECEIPT_NAME = "apply_resource_receipt.json"


class SomphPredictorEntryError(ValueError):
    """Raised when one fixed SOMP-H process entry fails closed."""


def _read_request_same_fd(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    before = os.lstat(source)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise SomphPredictorEntryError(
            "SOMP-H request must be a regular non-symlink file"
        )
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    descriptor = os.open(source, flags)
    try:
        opened = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
        ):
            raise SomphPredictorEntryError(
                "SOMP-H request identity changed before open"
            )
        raw = b""
        while len(raw) < opened.st_size:
            chunk = os.read(descriptor, opened.st_size - len(raw))
            if not chunk:
                raise SomphPredictorEntryError("SOMP-H request was truncated")
            raw += chunk
    finally:
        os.close(descriptor)
    try:
        payload = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SomphPredictorEntryError("SOMP-H request is not UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise SomphPredictorEntryError("SOMP-H request root must be an object")
    return payload, hashlib.sha256(raw).hexdigest()


def _output_root(path: str | Path) -> Path:
    raw = Path(path)
    resolved = raw.resolve(strict=True)
    if raw.is_symlink() or not resolved.is_dir():
        raise SomphPredictorEntryError(
            "SOMP-H output root must be a non-symlink directory"
        )
    return resolved


def _device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise SomphPredictorEntryError(
            "SOMP-H request requires CUDA but CUDA is unavailable"
        )
    index = int(value.split(":", 1)[1])
    if index >= torch.cuda.device_count():
        raise SomphPredictorEntryError("SOMP-H CUDA device index is unavailable")
    return torch.device(value)


def _prepare_cuda_memory_audit(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.cuda.set_device(device)
    torch.empty(0, device=device)
    torch.cuda.reset_peak_memory_stats(device)


def _descriptor(manifest: Mapping[str, Any], kind: str) -> dict[str, Any]:
    matches = [dict(item) for item in manifest["members"] if item["kind"] == kind]
    if len(matches) != 1:
        raise SomphPredictorEntryError(
            f"SOMP-H package descriptor missing or duplicated: {kind}"
        )
    return matches[0]


def _write_readonly_json_new(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (
        json.dumps(
            dict(payload),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o444)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(f"short write: {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, 0o444)
    return hashlib.sha256(raw).hexdigest()


def _load_fixed_runtime(
    package_root: str | Path,
    manifest: Mapping[str, Any],
    *,
    device: torch.device,
) -> tuple[torch.jit.ScriptModule, dict[str, Any]]:
    method_lock = load_json_artifact_same_fd(
        package_root, _descriptor(manifest, "method_lock")
    )
    model = load_torchscript_backbone_same_fd(
        package_root,
        _descriptor(manifest, "feature_runtime"),
        device=device,
    )
    return model, method_lock


def run_somph_enrollment(
    *,
    request_json: str | Path,
    package_root: str | Path,
    detached_seal_path: str | Path,
    expected_seal_sha256: str,
    output_root: str | Path,
) -> dict[str, Any]:
    raw_request, request_sha256 = _read_request_same_fd(request_json)
    request = validate_somph_enrollment_request(raw_request)
    if request["package_seal_sha256"] != str(expected_seal_sha256).lower():
        raise SomphPredictorEntryError(
            "SOMP-H enrollment request/package seal trust root mismatch"
        )
    output = _output_root(output_root)
    if request["head_output_leaf"] == ENROLLMENT_RECEIPT_NAME:
        raise SomphPredictorEntryError("SOMP-H enrollment output leaf collision")
    payloads, manifest, preopen = load_verified_somph_predictor_bundle(
        package_root,
        detached_seal_path=detached_seal_path,
        expected_seal_sha256=request["package_seal_sha256"],
    )
    if manifest["profile"] != ENROLLMENT_ONLY:
        raise SomphPredictorEntryError(
            "SOMP-H enrollment entry requires enrollment_only package"
        )
    device = _device(request["device"])
    _prepare_cuda_memory_audit(device)
    model, method_lock = _load_fixed_runtime(
        package_root, manifest, device=device
    )
    enrollment_input = {
        "schema": "cvs.phase2.somph_enrollment_binding.v1",
        "stage": manifest["stage"],
        "registration_state": manifest["registration_state"],
        "receiver": manifest["receiver"],
        "seed": manifest["seed"],
        "k_shot": manifest["k_shot"],
        "registered_class_handles": [
            item["class_handle"] for item in manifest["registered_classes"]
        ],
        "enrollment_package_root_sha256": manifest["package_root_sha256"],
        "enrollment_package_seal_sha256": request["package_seal_sha256"],
        "phase1_checkpoint_sha256": manifest[
            "phase1_checkpoint_sha256"
        ],
        "feature_runtime_sha256": manifest["feature_runtime_sha256"],
        "method_lock_sha256": manifest["method_lock_sha256"],
    }
    capsule, resource = enroll_somph_heads(
        model,
        payloads,
        enrollment_binding=enrollment_input,
        method_lock=method_lock,
        device=device,
        batch_size=request["support_batch_size"],
    )
    published = publish_somph_head_artifact(
        output / request["head_output_leaf"],
        capsule=capsule,
        method_lock=method_lock,
        expected_enrollment_binding_sha256=resource[
            "enrollment_binding_sha256"
        ],
    )
    receipt = {
        "schema": "cvs.phase2.somph_enrollment_execution_receipt.v1",
        "diagnostic_only": True,
        "status": "LOCAL_PROTOCOL_REPAIR_REQUIRED",
        "formal_launch_authority": False,
        "formal_metric_claim_allowed": False,
        "request_sha256": request_sha256,
        "package_root_sha256": manifest["package_root_sha256"],
        "package_seal_sha256": request["package_seal_sha256"],
        "phase1_checkpoint_sha256": manifest[
            "phase1_checkpoint_sha256"
        ],
        "feature_runtime_sha256": manifest["feature_runtime_sha256"],
        "method_lock_sha256": manifest["method_lock_sha256"],
        "overlay_provenance_sha256": manifest["overlay_provenance_sha256"],
        "head_capsule_sha256": published["head_capsule_sha256"],
        "enrollment_binding_sha256": published[
            "enrollment_binding_sha256"
        ],
        "preopen_audit": preopen,
        "resource": resource,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
    }
    receipt_sha256 = _write_readonly_json_new(
        output / ENROLLMENT_RECEIPT_NAME, receipt
    )
    return {
        "schema": "cvs.phase2.somph_enrollment_stdout.v1",
        "diagnostic_only": True,
        "profile": ENROLLMENT_ONLY,
        "request_sha256": request_sha256,
        "head_output_leaf": request["head_output_leaf"],
        "head_capsule_sha256": published["head_capsule_sha256"],
        "enrollment_binding_sha256": published[
            "enrollment_binding_sha256"
        ],
        "execution_receipt_sha256": receipt_sha256,
        "formal_launch_authority": False,
    }


def run_somph_apply(
    *,
    request_json: str | Path,
    package_root: str | Path,
    detached_seal_path: str | Path,
    expected_seal_sha256: str,
    output_root: str | Path,
) -> dict[str, Any]:
    raw_request, request_sha256 = _read_request_same_fd(request_json)
    request = validate_somph_apply_request(raw_request)
    if request["package_seal_sha256"] != str(expected_seal_sha256).lower():
        raise SomphPredictorEntryError(
            "SOMP-H apply request/package seal trust root mismatch"
        )
    output = _output_root(output_root)
    if request["prediction_output_leaf"] == APPLY_RECEIPT_NAME:
        raise SomphPredictorEntryError("SOMP-H apply output leaf collision")
    manifest, _seal, _preflight = preflight_somph_predictor_bundle(
        package_root,
        detached_seal_path=detached_seal_path,
        expected_seal_sha256=request["package_seal_sha256"],
    )
    if manifest["profile"] != APPLY_ONLY:
        raise SomphPredictorEntryError(
            "SOMP-H apply entry requires apply_only package"
        )
    if manifest["head_capsule_sha256"] != request["head_capsule_sha256"]:
        raise SomphPredictorEntryError(
            "SOMP-H apply request/head capsule trust root mismatch"
        )
    if (
        manifest["head_enrollment_binding_sha256"]
        != request["head_enrollment_binding_sha256"]
    ):
        raise SomphPredictorEntryError(
            "SOMP-H apply request/head binding trust root mismatch"
        )
    if manifest["row_handle"] != request["row_handle"]:
        raise SomphPredictorEntryError(
            "SOMP-H apply request/manifest row handle mismatch"
        )
    if manifest["row_manifest_sha256"] != request["row_manifest_sha256"]:
        raise SomphPredictorEntryError(
            "SOMP-H apply request/manifest row SHA256 mismatch"
        )
    payloads, loaded_manifest, preopen = load_verified_somph_predictor_bundle(
        package_root,
        detached_seal_path=detached_seal_path,
        expected_seal_sha256=request["package_seal_sha256"],
    )
    if loaded_manifest != manifest:
        raise SomphPredictorEntryError(
            "SOMP-H apply manifest changed between preflight and materialization"
        )
    capsule, _binding, binding_sha256 = load_verified_somph_head_capsule(
        package_root,
        detached_seal_path=detached_seal_path,
        expected_seal_sha256=request["package_seal_sha256"],
    )
    if binding_sha256 != request["head_enrollment_binding_sha256"]:
        raise SomphPredictorEntryError(
            "SOMP-H loaded head binding does not match request"
        )
    device = _device(request["device"])
    _prepare_cuda_memory_audit(device)
    model, method_lock = _load_fixed_runtime(
        package_root, manifest, device=device
    )
    handles = [item["class_handle"] for item in manifest["registered_classes"]]
    payload, resource = apply_somph_heads(
        model,
        payloads,
        capsule,
        registered_class_handles=handles,
        expected_enrollment_binding_sha256=binding_sha256,
        method_lock=method_lock,
        device=device,
        batch_size=SOMPH_APPLY_BATCH_SIZE,
    )
    indices = np.asarray(payload["predicted_class_indices"])
    if (
        indices.dtype.kind not in {"i", "u"}
        or indices.ndim != 1
        or len(indices) != len(payload["query_tokens"])
        or (len(indices) and int(indices.max()) >= len(handles))
        or (len(indices) and int(indices.min()) < 0)
    ):
        raise SomphPredictorEntryError(
            "SOMP-H apply produced an invalid class-index vector"
        )
    predicted_handles = np.asarray(handles)[indices.astype(np.int64)]
    stage = "Stage2-B" if manifest["stage"] == "stage2b" else "Stage2-C"
    registration_state = (
        "before_registration"
        if manifest["registration_state"] == "before"
        else "after_registration"
    )
    published = publish_somph_prediction_artifact(
        output / request["prediction_output_leaf"],
        query_tokens=payload["query_tokens"],
        scenarios=payload["scenarios"],
        predicted_class_handles=predicted_handles,
        backbone_forward_counts=payload["backbone_forward_counts"],
        stage=stage,
        registration_state=registration_state,
        row_id=manifest["row_handle"],
        receiver=manifest["receiver"],
        seed=manifest["seed"],
        k_shot=manifest["k_shot"],
        registered_class_count=len(handles),
        registry_snapshot_sha256=hashlib.sha256(
            json.dumps(
                handles,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
        method_lock_sha256=manifest["method_lock_sha256"],
        row_manifest_sha256=manifest["row_manifest_sha256"],
        stage_input_binding_sha256=binding_sha256,
        package_root_sha256=manifest["package_root_sha256"],
        package_seal_sha256=request["package_seal_sha256"],
        feature_runtime_sha256=manifest["feature_runtime_sha256"],
        head_capsule_sha256=request["head_capsule_sha256"],
        protocol_policy_sha256=canonical_sha256(PHASE2_FULL_CONTRACT),
    )
    receipt = {
        "schema": "cvs.phase2.somph_apply_execution_receipt.v1",
        "diagnostic_only": True,
        "status": "LOCAL_PROTOCOL_REPAIR_REQUIRED",
        "formal_launch_authority": False,
        "formal_metric_claim_allowed": False,
        "request_sha256": request_sha256,
        "package_root_sha256": manifest["package_root_sha256"],
        "package_seal_sha256": request["package_seal_sha256"],
        "phase1_checkpoint_sha256": manifest[
            "phase1_checkpoint_sha256"
        ],
        "feature_runtime_sha256": manifest["feature_runtime_sha256"],
        "method_lock_sha256": manifest["method_lock_sha256"],
        "overlay_provenance_sha256": manifest["overlay_provenance_sha256"],
        "head_capsule_sha256": request["head_capsule_sha256"],
        "enrollment_binding_sha256": binding_sha256,
        "prediction_artifact_sha256": published["artifact_sha256"],
        "prediction_seal_sha256": published["seal_sha256"],
        "preopen_audit": preopen,
        "resource": resource,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
    }
    receipt_sha256 = _write_readonly_json_new(
        output / APPLY_RECEIPT_NAME, receipt
    )
    return {
        "schema": "cvs.phase2.somph_apply_stdout.v1",
        "diagnostic_only": True,
        "profile": APPLY_ONLY,
        "request_sha256": request_sha256,
        "prediction_output_leaf": request["prediction_output_leaf"],
        "artifact_sha256": published["artifact_sha256"],
        "seal_sha256": published["seal_sha256"],
        "execution_receipt_sha256": receipt_sha256,
        "formal_launch_authority": False,
    }


__all__ = [
    "APPLY_RECEIPT_NAME",
    "ENROLLMENT_RECEIPT_NAME",
    "SomphPredictorEntryError",
    "run_somph_apply",
    "run_somph_enrollment",
]
