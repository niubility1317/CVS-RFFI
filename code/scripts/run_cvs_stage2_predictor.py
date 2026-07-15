#!/usr/bin/env python
"""Run one sealed, truth-free Stage2 request and atomically publish predictions."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.phase2_runtime_contract import validate_predictor_request  # noqa: E402
from cvsrffi.phase2_memfd_snapshot import (  # noqa: E402
    open_pinned_special,
    pinned_input_mode_active,
)
from cvsrffi.stage2_prediction_artifact import (  # noqa: E402
    publish_prediction_artifact,
    verify_prediction_artifact,
)
from cvsrffi.stage2_predictor_bundle import sha256_bytes, sha256_file  # noqa: E402
from cvsrffi.stage2_predictor_entry import prepare_role_blind_prediction  # noqa: E402


def _read_request(path: Path) -> tuple[dict[str, Any], str]:
    if pinned_input_mode_active():
        with open_pinned_special("request") as handle:
            raw = handle.read()
        payload = json.loads(raw.decode("utf-8-sig"))
        if not isinstance(payload, dict):
            raise ValueError("predictor request root must be an object")
        validate_predictor_request(payload)
        return payload, sha256_bytes(raw)
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError("predictor request must be a regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
        ):
            raise ValueError("predictor request identity changed before open")
        raw = b""
        while len(raw) < opened.st_size:
            chunk = os.read(descriptor, opened.st_size - len(raw))
            if not chunk:
                raise ValueError("predictor request was truncated")
            raw += chunk
    finally:
        os.close(descriptor)
    payload = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("predictor request root must be an object")
    # The request is the only file read before this validation.  Package/seal,
    # runtime artifacts and IQ remain untouched until it passes.
    validate_predictor_request(payload)
    return payload, sha256_bytes(raw)


def _class_handle_predictions(
    predictions: Mapping[str, np.ndarray], registered_classes: list[Mapping[str, Any]]
) -> dict[str, np.ndarray]:
    handles = np.asarray([item["class_handle"] for item in registered_classes])
    result: dict[str, np.ndarray] = {}
    for stream in (
        "candidate_after",
        "candidate_before",
        "identity_after",
        "identity_before",
        "direct",
    ):
        indices = np.asarray(predictions[stream])
        if indices.ndim != 1 or indices.dtype.kind not in {"i", "u"}:
            raise ValueError(f"predictor stream is not a class-index vector: {stream}")
        if len(indices) and (int(indices.min()) < 0 or int(indices.max()) >= len(handles)):
            raise ValueError(f"predictor stream class index is outside registry: {stream}")
        result[stream] = handles[indices.astype(np.int64)]
    return result


def _write_readonly_json_new(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write for predictor receipt")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, 0o444)
    return sha256_file(path)


def _prepare_device(requested: str) -> torch.device:
    if torch.cuda.is_available() and requested.startswith("cuda"):
        device = torch.device(requested)
        torch.cuda.init()
        torch.cuda.set_device(device)
        # torch.cuda.init() initializes PyTorch's CUDA state but does not
        # guarantee that the caching allocator has created the selected-device
        # context.  Peak-memory APIs require that allocator context, so create
        # and immediately release one scalar before resetting the counters.
        warmup = torch.empty(1, device=device)
        del warmup
        torch.cuda.reset_peak_memory_stats(device)
        return device
    return torch.device("cpu")


def run(args: argparse.Namespace) -> dict[str, Any]:
    request_path = Path(args.request_json)
    request, request_sha256 = _read_request(request_path)
    device = _prepare_device(str(args.device))
    payload, metadata, audit = prepare_role_blind_prediction(
        request,
        predictor_package_root=args.predictor_package_root,
        detached_seal_path=args.detached_seal_path,
        expected_seal_sha256=str(args.expected_seal_sha256).lower(),
        device=device,
        batch_size=int(args.batch_size),
    )
    handles = _class_handle_predictions(payload, request["registered_classes"])
    output_root = Path(args.output_root).resolve(strict=True)
    if output_root.is_symlink() or not output_root.is_dir():
        raise ValueError("predictor output root must be a non-symlink directory")
    relative = Path(request["output_contract"]["relative_path"])
    if len(relative.parts) != 1:
        raise ValueError("predictor output artifact must be a direct child of output root")
    target = output_root / relative
    stage = "Stage2-B" if request["stage"] == "stage2b" else "Stage2-C"
    published = publish_prediction_artifact(
        target,
        stage=stage,
        row_id=str(request["row_id"]),
        receiver=str(request["receiver"]),
        k_shot=int(request["k_shot"]),
        candidate_lock_sha256=str(request["candidate_lock_sha256"]),
        package_root_sha256=str(request["package_root_sha256"]),
        package_seal_sha256=str(args.expected_seal_sha256).lower(),
        query_tokens=payload["query_tokens"],
        scenarios=payload["scenarios"],
        candidate_after=handles["candidate_after"],
        candidate_before=handles["candidate_before"],
        identity_after=handles["identity_after"],
        identity_before=handles["identity_before"],
        direct=handles["direct"],
        shared_view_counts=payload["shared_view_counts"],
    )
    verified = verify_prediction_artifact(
        target,
        expected_artifact_sha256=published["artifact_sha256"],
        expected_seal_sha256=published["seal_sha256"],
    )
    resource = dict(audit["resource_receipt"])
    resource.update(
        {
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
            "request_sha256": request_sha256,
            "prediction_artifact_sha256": published["artifact_sha256"],
            "prediction_seal_sha256": published["seal_sha256"],
        }
    )
    resource_path = output_root / "predictor_resource_receipt.json"
    resource_sha256 = _write_readonly_json_new(resource_path, resource)
    audit_receipt = {
        "schema": "cvs.phase2.predictor_execution_audit.v2",
        "status": "PASS",
        "request_sha256": sha256_file(request_path),
        "prediction_artifact_sha256": published["artifact_sha256"],
        "prediction_seal_sha256": published["seal_sha256"],
        "predictor_resource_receipt_sha256": resource_sha256,
        "preopen_audit": audit["preopen"],
        "materialization_audit": audit["materialization"],
        "query_truth_access": False,
        "query_role_access": False,
        "query_true_batch_class_count_access": False,
        "query_class_quota_access": False,
        "query_batch_global_assignment": False,
    }
    audit_path = output_root / "predictor_execution_audit.json"
    audit_sha256 = _write_readonly_json_new(audit_path, audit_receipt)
    return {
        **published,
        "verified_artifact_sha256": verified["artifact_sha256"],
        "request_sha256": sha256_file(request_path),
        "predictor_resource_receipt": str(resource_path),
        "predictor_resource_receipt_sha256": resource_sha256,
        "predictor_execution_audit": str(audit_path),
        "predictor_execution_audit_sha256": audit_sha256,
        "metadata": metadata,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--predictor-package-root", type=Path, required=True)
    parser.add_argument("--detached-seal-path", type=Path, required=True)
    parser.add_argument("--expected-seal-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
