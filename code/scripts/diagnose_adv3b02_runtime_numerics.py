#!/usr/bin/env python
"""Diagnose ADV3B02 eager/TorchScript numerical drift without authority.

This utility deliberately cannot emit a parity receipt or select a runtime.  It
uses one frozen probe matrix to compare repeatability, the existing runtime,
and a fresh trace.  Its orchestrator starts baseline and deterministic modes
as separate Python workers so the deterministic CUBLAS environment exists
before that worker imports Torch.  The JSON output is a development-only
mechanism diagnostic.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess
import sys
import tempfile
import time
import tracemalloc
from typing import Any, Iterator, Mapping
import zipfile


CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
for value in (str(REPO_ROOT), str(CODE_ROOT)):
    while value in sys.path:
        sys.path.remove(value)
for value in (str(REPO_ROOT), str(CODE_ROOT)):
    sys.path.insert(0, value)

DIAGNOSTIC_SCHEMA = "cvs.development.adv3b02_runtime_numerical_diagnostic.v1"
DIAGNOSTIC_STATUS = "DEVELOPMENT_NUMERICAL_DIAGNOSTIC_NO_AUTHORITY"
BLOCKED_SOURCE_STATUS = "BLOCKED_MISSING_SIGNED_SOURCE_RECEIPT"
WORKER_SCHEMA = "cvs.development.adv3b02_runtime_numerical_worker.v1"
WORKER_STATUS = "DEVELOPMENT_NUMERICAL_DIAGNOSTIC_WORKER_NO_AUTHORITY"
TRACE_BUILDER_SCHEMA = "cvs.development.adv3b02_fresh_trace_builder.v1"
TRACE_BUILDER_STATUS = "DEVELOPMENT_FRESH_TRACE_BUILDER_NO_AUTHORITY"
INPUT_LEN = 256
PROBE_SEED = 20260720
BATCH_SIZES = (1, 8, 256)
REFERENCE_THRESHOLD = 1.0e-5
SOURCE_RELEASE_ISSUER = "qknnv42_stage2bc_extreme_light_route_20260716"
SOURCE_RELEASE_KEY_ID = "somph-authority-ed25519-20260716"
SOURCE_RELEASE_PUBLIC_KEY_HEX = (
    "ec301433b5a625f8e34f887f5aeea664e809236d1b871fcc0ffeb47cb540bdc1"
)
SOURCE_RELEASE_PUBLIC_KEY_SHA256 = (
    "52944e59ec99d360e227cbe78e84efeca6db3ebca3d9698f5d567270c37a9444"
)
EXPECTED_PROBE_ROOT_SHA256 = (
    "3382c7381a3e6a3cb0311da567bad1d3de3dd242cef24466db9350c100ace6bb"
)
BASE_CHECKPOINT_SHA256 = (
    "2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98"
)
RUNTIME_ARMS: dict[str, dict[str, str]] = {
    "b202": {
        "sha256": "b2021ca1ac97848a8cfda353a4070530bfa41bc08a711f746f329bd2d8d870d9",
        "canonical_remote_path": "/home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_ci_strict_matrix_20260716_v1/runtime_artifacts_v2/adv3b02_base_runtime.ts",
        "lineage_evidence_path": "/home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_ci_strict_matrix_20260716_v1/runtime_artifacts_v2/runtime_parity_receipt.json",
        "lineage_evidence_sha256": "db8635b986bcaea6cbe6f954e90e5ed37b9fb6042876628392db96fe82be42f4",
        "lineage_scope": (
            "historical_generation_log_reports_same_checkpoint_no_formal_lineage"
        ),
        "artifact_origin_receipt_sha256": "",
    },
    "f119": {
        "sha256": "f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a",
        "canonical_remote_path": "/home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/sealed_feature_runtime.pt",
        "lineage_evidence_path": "/home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/method_lock.json",
        "lineage_evidence_sha256": "0496594db4a82efbbf17ec3d67ebc3fb1f0c7ced41b542a5a0bde3482e704523",
        "lineage_scope": (
            "historical_runtime_without_independent_checkpoint_lineage_receipt"
        ),
        "artifact_origin_receipt_sha256": "",
    },
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ADV3B02NumericalDiagnosticError(ValueError):
    """Raised when diagnostic inputs or execution violate the fixed contract."""


class MissingSignedSourceReceiptError(ADV3B02NumericalDiagnosticError):
    """Raised when the external signed source archive/receipt does not exist."""


def _load_worker_dependencies() -> None:
    """Import Torch and project modules only inside trace/compare workers."""

    if "torch" in globals():
        return
    import numpy as numpy_module
    import torch as torch_module
    from cvsrffi.checkpoint_loading import (
        build_exact_ssdg_model_from_checkpoint as build_model,
        strip_module_prefix as strip_prefix,
    )
    from cvsrffi import phase1_adv3b02_deployment_bundle as bundle_module
    from scripts.export_adv3b02_effective8_torchscript import (
        ADV3B02IdentityRuntime as identity_runtime,
    )

    globals().update(
        {
            "np": numpy_module,
            "torch": torch_module,
            "build_exact_ssdg_model_from_checkpoint": build_model,
            "strip_module_prefix": strip_prefix,
            "deployment_bundle": bundle_module,
            "ADV3B02IdentityRuntime": identity_runtime,
        }
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_bytes_new(path: Path, payload: bytes, name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ADV3B02NumericalDiagnosticError(
            f"refusing to overwrite {name}"
        ) from exc


def _read_regular_bytes(path: Path, name: str) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise ADV3B02NumericalDiagnosticError(f"{name} must be a regular file")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ADV3B02NumericalDiagnosticError(f"failed to read {name}") from exc


def _json_from_snapshot(value: bytes, name: str) -> Any:
    try:
        return json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ADV3B02NumericalDiagnosticError(f"{name} is invalid JSON") from exc


def _file_record_from_snapshot(path: Path, value: bytes) -> dict[str, Any]:
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "bytes": len(value),
        "sha256": _sha256_bytes(value),
    }


def _source_manifest_root(rows: list[dict[str, Any]]) -> str:
    return _sha256_bytes(_canonical_json_bytes({"source_members": rows}))


def _load_checkpoint_bytes(value: bytes) -> Any:
    try:
        return torch.load(io.BytesIO(value), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(io.BytesIO(value), map_location="cpu")


def _validate_sha256(value: str, name: str) -> str:
    normalized = str(value).strip().lower()
    if _SHA256_RE.fullmatch(normalized) is None:
        raise ADV3B02NumericalDiagnosticError(f"{name} must be lowercase SHA256")
    return normalized


def _arm_contract(arm_id: str) -> dict[str, str]:
    value = str(arm_id).strip().lower()
    if value not in RUNTIME_ARMS:
        raise ADV3B02NumericalDiagnosticError(
            "arm_id must be one of the internally preregistered runtime arms"
        )
    contract = dict(RUNTIME_ARMS[value])
    contract["sha256"] = _validate_sha256(contract.get("sha256", ""), "arm SHA256")
    contract["lineage_evidence_sha256"] = _validate_sha256(
        contract.get("lineage_evidence_sha256", ""), "lineage evidence SHA256"
    )
    for field in ("canonical_remote_path", "lineage_evidence_path", "lineage_scope"):
        if not str(contract.get(field, "")).strip():
            raise ADV3B02NumericalDiagnosticError(f"arm contract missing {field}")
    origin_sha = str(contract.get("artifact_origin_receipt_sha256", "")).strip()
    if origin_sha:
        contract["artifact_origin_receipt_sha256"] = _validate_sha256(
            origin_sha, "artifact origin receipt SHA256"
        )
    contract["arm_id"] = value
    return contract


def _validate_asset_lineage(
    *,
    runtime: Path,
    runtime_sha256: str,
    lineage_evidence_path: str | Path,
    artifact_origin_receipt_path: str | Path | None,
    contract: Mapping[str, str],
) -> dict[str, Any]:
    evidence = Path(lineage_evidence_path).resolve()
    if str(evidence) != str(Path(contract["lineage_evidence_path"])):
        raise ADV3B02NumericalDiagnosticError(
            "lineage evidence path does not match preregistered arm path"
        )
    evidence_bytes = _read_regular_bytes(evidence, "lineage evidence")
    evidence_sha = _sha256_bytes(evidence_bytes)
    if evidence_sha != contract["lineage_evidence_sha256"]:
        raise ADV3B02NumericalDiagnosticError("lineage evidence SHA256 mismatch")
    canonical_runtime = str(Path(contract["canonical_remote_path"]))
    base_origin: dict[str, Any] = {
        "existing_runtime_path": str(runtime),
        "canonical_origin_path": canonical_runtime,
        "runtime_sha256": runtime_sha256,
        "lineage_scope": contract["lineage_scope"],
    }
    if str(runtime) == canonical_runtime:
        origin = {
            **base_origin,
            "scope": "canonical_remote_asset_path",
            "receipt_required": False,
            "receipt_path": None,
            "receipt_sha256": None,
            "receipt_content": None,
        }
    else:
        locked_receipt_sha = str(
            contract.get("artifact_origin_receipt_sha256", "")
        ).strip()
        if not locked_receipt_sha or artifact_origin_receipt_path is None:
            raise ADV3B02NumericalDiagnosticError(
                "noncanonical runtime path requires an internally preregistered origin receipt"
            )
        receipt_path = Path(artifact_origin_receipt_path).resolve()
        receipt_bytes = _read_regular_bytes(receipt_path, "artifact origin receipt")
        receipt_sha = _sha256_bytes(receipt_bytes)
        if receipt_sha != locked_receipt_sha:
            raise ADV3B02NumericalDiagnosticError(
                "artifact origin receipt SHA256 mismatch"
            )
        try:
            receipt = json.loads(receipt_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ADV3B02NumericalDiagnosticError(
                "artifact origin receipt is invalid JSON"
            ) from exc
        expected = {
            "schema": "cvs.development.runtime_artifact_origin_path.v1",
            "arm_id": contract["arm_id"],
            "canonical_origin_path": canonical_runtime,
            "diagnostic_runtime_path": str(runtime),
            "runtime_sha256": runtime_sha256,
            "formal_authority": False,
        }
        if receipt != expected:
            raise ADV3B02NumericalDiagnosticError(
                "artifact origin receipt content drift"
            )
        origin = {
            **base_origin,
            "scope": "preregistered_read_only_copy",
            "receipt_required": True,
            "receipt_path": str(receipt_path),
            "receipt_sha256": receipt_sha,
            "receipt_content": expected,
        }
    return {
        "lineage_evidence_path": str(evidence),
        "lineage_evidence_sha256": evidence_sha,
        "lineage_scope": contract["lineage_scope"],
        "runtime_origin": origin,
    }


def _expected_runtime_origin(
    *,
    contract: Mapping[str, str],
    runtime_path: str | Path,
    runtime_sha256: str,
    artifact_origin_receipt_path: str | Path | None,
) -> dict[str, Any]:
    runtime = Path(runtime_path).resolve()
    canonical = str(Path(contract["canonical_remote_path"]))
    base: dict[str, Any] = {
        "existing_runtime_path": str(runtime),
        "canonical_origin_path": canonical,
        "runtime_sha256": runtime_sha256,
        "lineage_scope": contract["lineage_scope"],
    }
    if str(runtime) == canonical:
        if artifact_origin_receipt_path is not None:
            raise ADV3B02NumericalDiagnosticError(
                "canonical runtime must not supply an origin-copy receipt"
            )
        return {
            **base,
            "scope": "canonical_remote_asset_path",
            "receipt_required": False,
            "receipt_path": None,
            "receipt_sha256": None,
            "receipt_content": None,
        }
    locked_sha = str(contract.get("artifact_origin_receipt_sha256", "")).strip()
    if not locked_sha or artifact_origin_receipt_path is None:
        raise ADV3B02NumericalDiagnosticError(
            "noncanonical runtime path requires an internally preregistered origin receipt"
        )
    receipt_path = Path(artifact_origin_receipt_path).resolve()
    content = {
        "schema": "cvs.development.runtime_artifact_origin_path.v1",
        "arm_id": contract["arm_id"],
        "canonical_origin_path": canonical,
        "diagnostic_runtime_path": str(runtime),
        "runtime_sha256": runtime_sha256,
        "formal_authority": False,
    }
    return {
        **base,
        "scope": "preregistered_read_only_copy",
        "receipt_required": True,
        "receipt_path": str(receipt_path),
        "receipt_sha256": locked_sha,
        "receipt_content": content,
    }


def _verify_source_release_signature(message: bytes, signature: bytes) -> None:
    from cvsrffi.somph_runtime_trust import verify_ed25519

    verify_ed25519(bytes.fromhex(SOURCE_RELEASE_PUBLIC_KEY_HEX), message, signature)


def _validate_source_release(
    *,
    source_archive_path: str | Path,
    source_release_receipt_path: str | Path,
    _unit_test_fixture: bool = False,
) -> dict[str, Any]:
    archive = Path(source_archive_path).resolve()
    receipt_path = Path(source_release_receipt_path).resolve()
    if not archive.exists() or not receipt_path.exists():
        raise MissingSignedSourceReceiptError(BLOCKED_SOURCE_STATUS)
    archive_bytes = _read_regular_bytes(archive, "source archive")
    receipt_bytes = _read_regular_bytes(receipt_path, "source release receipt")
    receipt = _json_from_snapshot(receipt_bytes, "source release receipt")
    if not isinstance(receipt, dict):
        raise ADV3B02NumericalDiagnosticError(
            "source release receipt must be a JSON object"
        )
    signature_hex = receipt.pop("signature_hex", None)
    if not isinstance(signature_hex, str) or re.fullmatch(
        r"[0-9a-f]{128}", signature_hex
    ) is None:
        raise ADV3B02NumericalDiagnosticError(
            "source release receipt signature is invalid"
        )
    archive_sha = _sha256_bytes(archive_bytes)
    required = {
        "schema": "cvs.development.source_archive_commit_receipt.v1",
        "issuer": SOURCE_RELEASE_ISSUER,
        "key_id": SOURCE_RELEASE_KEY_ID,
        "public_key_sha256": SOURCE_RELEASE_PUBLIC_KEY_SHA256,
        "source_archive_path": str(archive),
        "source_archive_sha256": archive_sha,
    }
    if any(receipt.get(key) != value for key, value in required.items()):
        raise ADV3B02NumericalDiagnosticError(
            "source release receipt archive/issuer binding drift"
        )
    commit = receipt.get("source_git_commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ADV3B02NumericalDiagnosticError(
            "source release receipt Git commit is invalid"
        )
    source_members = receipt.get("source_members")
    if not isinstance(source_members, list) or not source_members:
        raise ADV3B02NumericalDiagnosticError(
            "signed source member manifest is missing"
        )
    normalized_members: list[dict[str, Any]] = []
    seen_members: set[str] = set()
    for row in source_members:
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise ADV3B02NumericalDiagnosticError("signed source member row drift")
        member_path = str(row.get("path", ""))
        pure = PurePosixPath(member_path)
        if (
            not member_path
            or pure.is_absolute()
            or "\\" in member_path
            or any(part in {"", ".", ".."} for part in pure.parts)
            or member_path in seen_members
        ):
            raise ADV3B02NumericalDiagnosticError(
                "signed source member path is unsafe or duplicated"
            )
        member_bytes = row.get("bytes")
        if not isinstance(member_bytes, int) or isinstance(member_bytes, bool) or member_bytes < 0:
            raise ADV3B02NumericalDiagnosticError("signed source member size is invalid")
        member_sha = _validate_sha256(row.get("sha256", ""), "source member SHA256")
        normalized_members.append(
            {"path": member_path, "bytes": member_bytes, "sha256": member_sha}
        )
        seen_members.add(member_path)
    if normalized_members != sorted(normalized_members, key=lambda row: row["path"]):
        raise ADV3B02NumericalDiagnosticError("signed source members must be sorted")
    manifest_root = _source_manifest_root(normalized_members)
    if receipt.get("source_manifest_root_sha256") != manifest_root:
        raise ADV3B02NumericalDiagnosticError("signed source manifest root mismatch")
    git_policy = receipt.get("git_policy")
    if not isinstance(git_policy, dict):
        raise ADV3B02NumericalDiagnosticError("signed source Git policy is missing")
    if git_policy.get("mode") == "git_exact":
        required_git_fields = {
            "mode",
            "commit",
            "dirty",
            "status_root_sha256",
            "diff_root_sha256",
            "cached_diff_root_sha256",
            "untracked_root_sha256",
        }
        if (
            set(git_policy) != required_git_fields
            or git_policy.get("commit") != commit
            or not isinstance(git_policy.get("dirty"), bool)
        ):
            raise ADV3B02NumericalDiagnosticError("signed Git exact policy drift")
        for field in required_git_fields - {"mode", "commit", "dirty"}:
            _validate_sha256(git_policy.get(field, ""), f"signed Git {field}")
    elif git_policy != {"mode": "signed_manifest_only_no_git"}:
        raise ADV3B02NumericalDiagnosticError("signed source Git policy mode drift")
    if set(receipt) != {
        "schema",
        "issuer",
        "key_id",
        "public_key_sha256",
        "source_archive_path",
        "source_archive_sha256",
        "source_git_commit",
        "source_members",
        "source_manifest_root_sha256",
        "git_policy",
    }:
        raise ADV3B02NumericalDiagnosticError(
            "source release receipt fields drift"
        )
    _verify_source_release_signature(
        _canonical_json_bytes(receipt), bytes.fromhex(signature_hex)
    )
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes), mode="r") as bundle:
            archive_rows: list[dict[str, Any]] = []
            for info in bundle.infolist():
                if info.is_dir():
                    raise ADV3B02NumericalDiagnosticError(
                        "source archive directories are not manifest members"
                    )
                unix_mode = (int(info.external_attr) >> 16) & 0o170000
                if unix_mode == 0o120000:
                    raise ADV3B02NumericalDiagnosticError(
                        "source archive symlink member is forbidden"
                    )
                member = bundle.read(info)
                archive_rows.append(
                    {
                        "path": info.filename,
                        "bytes": len(member),
                        "sha256": _sha256_bytes(member),
                    }
                )
    except zipfile.BadZipFile as exc:
        raise ADV3B02NumericalDiagnosticError(
            "source archive is not a valid ZIP"
        ) from exc
    archive_rows.sort(key=lambda row: row["path"])
    if archive_rows != normalized_members:
        raise ADV3B02NumericalDiagnosticError(
            "source archive members do not close against signed manifest"
        )
    return {
        **receipt,
        "signature_hex": signature_hex,
        "receipt_path": str(receipt_path),
        "receipt_sha256": _sha256_bytes(receipt_bytes),
        "source_archive_bytes": len(archive_bytes),
        "signature_verified": True,
        "acceptance": (
            "UNIT_TEST_SIGNATURE_FIXTURE_NOT_AUTHORIZED"
            if _unit_test_fixture
            else "EXTERNAL_SIGNED_SOURCE_RECEIPT_VERIFIED"
        ),
    }


def _resolve_primary_device(
    requested: str, *, _allow_cpu_for_tests: bool = False
) -> torch.device:
    value = str(requested).strip().lower()
    if value == "cpu" and _allow_cpu_for_tests:
        return torch.device("cpu")
    if not value.startswith("cuda:"):
        raise ADV3B02NumericalDiagnosticError(
            "primary device must be explicit cuda:<index>; CPU is control-only"
        )
    try:
        index = int(value.split(":", 1)[1])
    except (TypeError, ValueError) as exc:
        raise ADV3B02NumericalDiagnosticError("CUDA device index is invalid") from exc
    if (
        index < 0
        or not torch.cuda.is_available()
        or index >= int(torch.cuda.device_count())
    ):
        raise ADV3B02NumericalDiagnosticError(
            "requested CUDA device is unavailable; CPU fallback is forbidden"
        )
    return torch.device(f"cuda:{index}")


def _validate_cuda_request_without_initializing(requested: str) -> str:
    value = str(requested).strip().lower()
    match = re.fullmatch(r"cuda:([0-9]+)", value)
    if match is None:
        raise ADV3B02NumericalDiagnosticError(
            "primary device must be explicit cuda:<index>"
        )
    return f"cuda:{int(match.group(1))}"


def _flag_snapshot() -> dict[str, Any]:
    warn_only_getter = getattr(
        torch, "is_deterministic_algorithms_warn_only_enabled", None
    )
    return {
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_algorithms_warn_only": bool(warn_only_getter())
        if warn_only_getter is not None
        else False,
    }


def _apply_flags(flags: Mapping[str, Any]) -> None:
    torch.backends.cuda.matmul.allow_tf32 = bool(flags["cuda_matmul_allow_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(flags["cudnn_allow_tf32"])
    torch.backends.cudnn.benchmark = bool(flags["cudnn_benchmark"])
    torch.backends.cudnn.deterministic = bool(flags["cudnn_deterministic"])
    try:
        torch.use_deterministic_algorithms(
            bool(flags["deterministic_algorithms"]),
            warn_only=bool(flags["deterministic_algorithms_warn_only"]),
        )
    except TypeError:
        torch.use_deterministic_algorithms(bool(flags["deterministic_algorithms"]))


def _deterministic_flags() -> dict[str, Any]:
    return {
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "deterministic_algorithms": True,
        "deterministic_algorithms_warn_only": False,
    }


@contextmanager
def _restore_global_flags() -> Iterator[dict[str, Any]]:
    original = _flag_snapshot()
    try:
        yield original
    finally:
        _apply_flags(original)


def _probes() -> dict[int, torch.Tensor]:
    result: dict[int, torch.Tensor] = {}
    for batch_size in BATCH_SIZES:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(PROBE_SEED + int(batch_size))
        result[int(batch_size)] = torch.randn(
            int(batch_size),
            2,
            INPUT_LEN,
            generator=generator,
            dtype=torch.float32,
        )
    return result


def _tensor_audit(value: torch.Tensor) -> dict[str, Any]:
    array = np.ascontiguousarray(value.detach().float().cpu().numpy())
    return {
        "shape": [int(item) for item in array.shape],
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(array.view(np.uint8)).hexdigest(),
    }


def _tensor_record(name: str, value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().cpu().contiguous()
    quantization: dict[str, Any] | None = None
    quantization_parameter_bytes = 0
    if tensor.is_quantized:
        raw_tensor = tensor.int_repr().contiguous()
        quantization = {"qscheme": str(tensor.qscheme())}
        if tensor.qscheme() in (torch.per_tensor_affine, torch.per_tensor_symmetric):
            quantization.update(
                {
                    "scale": float(tensor.q_scale()),
                    "zero_point": int(tensor.q_zero_point()),
                    "scale_bytes": 8,
                    "zero_point_bytes": 8,
                }
            )
            quantization_parameter_bytes = 16
        elif tensor.qscheme() in (
            torch.per_channel_affine,
            torch.per_channel_symmetric,
            torch.per_channel_affine_float_qparams,
        ):
            scales = tensor.q_per_channel_scales().detach().cpu().contiguous()
            zero_points = (
                tensor.q_per_channel_zero_points().detach().cpu().contiguous()
            )
            scale_raw = scales.reshape(-1).view(torch.uint8).numpy().tobytes()
            zero_raw = zero_points.reshape(-1).view(torch.uint8).numpy().tobytes()
            quantization.update(
                {
                    "axis": int(tensor.q_per_channel_axis()),
                    "scale_dtype": str(scales.dtype),
                    "scale_count": int(scales.numel()),
                    "scale_bytes": len(scale_raw),
                    "scale_sha256": _sha256_bytes(scale_raw),
                    "zero_point_dtype": str(zero_points.dtype),
                    "zero_point_count": int(zero_points.numel()),
                    "zero_point_bytes": len(zero_raw),
                    "zero_point_sha256": _sha256_bytes(zero_raw),
                }
            )
            quantization_parameter_bytes = len(scale_raw) + len(zero_raw)
    else:
        raw_tensor = tensor
    raw = raw_tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    result: dict[str, Any] = {
        "name": str(name),
        "shape": [int(item) for item in tensor.shape],
        "dtype": str(tensor.dtype),
        "numel": int(tensor.numel()),
        "bytes": len(raw),
        "data_bytes": len(raw),
        "quantization_parameter_bytes": quantization_parameter_bytes,
        "tensor_bytes": len(raw) + quantization_parameter_bytes,
        "sha256": _sha256_bytes(raw),
    }
    if quantization is not None:
        result["quantization"] = quantization
    return result


def _tensor_registry(values: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    rows = [
        _tensor_record(str(name), value)
        for name, value in sorted(values.items(), key=lambda item: str(item[0]))
        if torch.is_tensor(value)
    ]
    preimage = {"tensors": rows}
    return {
        "tensor_count": len(rows),
        "tensor_data_bytes": sum(int(row["data_bytes"]) for row in rows),
        "quantization_parameter_bytes": sum(
            int(row["quantization_parameter_bytes"]) for row in rows
        ),
        "tensor_bytes": sum(int(row["tensor_bytes"]) for row in rows),
        "root_sha256": _sha256_bytes(_canonical_json_bytes(preimage)),
        "tensors": rows,
    }


def _module_state_audit(module: torch.nn.Module) -> dict[str, Any]:
    return {
        "state": _tensor_registry(dict(module.state_dict())),
        "parameters": _tensor_registry(dict(module.named_parameters())),
        "buffers": _tensor_registry(dict(module.named_buffers())),
    }


def _probe_spec(probes: Mapping[int, torch.Tensor]) -> dict[str, Any]:
    spec = {
        "batch_sizes": list(BATCH_SIZES),
        "input_len": INPUT_LEN,
        "probe_seed": PROBE_SEED,
        "rows": [
            {
                "batch_size": int(batch_size),
                "generator_seed": PROBE_SEED + int(batch_size),
                "input": _tensor_audit(probes[int(batch_size)]),
            }
            for batch_size in BATCH_SIZES
        ],
    }
    root = _sha256_bytes(_canonical_json_bytes(spec).rstrip(b"\n"))
    if root != EXPECTED_PROBE_ROOT_SHA256:
        raise ADV3B02NumericalDiagnosticError(
            "fixed probe root drifted from preregistered probe specification"
        )
    return {**spec, "root_sha256": root}


def _outputs(value: Any, *, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ADV3B02NumericalDiagnosticError(
            "runtime output must be (z160, logits)"
        )
    feature, logits = value
    if (
        not torch.is_tensor(feature)
        or not torch.is_tensor(logits)
        or tuple(feature.shape) != (int(rows), 160)
        or logits.ndim != 2
        or int(logits.shape[0]) != int(rows)
        or int(logits.shape[1]) < 2
        or not bool(torch.isfinite(feature).all())
        or not bool(torch.isfinite(logits).all())
    ):
        raise ADV3B02NumericalDiagnosticError(
            "runtime output shape/finite contract drift"
        )
    return feature.detach().float().cpu(), logits.detach().float().cpu()


def _difference(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if tuple(left.shape) != tuple(right.shape):
        raise ADV3B02NumericalDiagnosticError("comparison tensor shape mismatch")
    delta = torch.abs(left - right)
    maximum = float(delta.max().item())
    mean = float(delta.mean().item())
    if not np.isfinite(maximum) or not np.isfinite(mean):
        raise ADV3B02NumericalDiagnosticError("comparison difference is non-finite")
    return {
        "max_abs": maximum,
        "mean_abs": mean,
        "exceeds_reference_threshold": bool(maximum > REFERENCE_THRESHOLD),
    }


def _logit_difference(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    result = _difference(left, right)
    left_top = torch.argmax(left, dim=1)
    right_top = torch.argmax(right, dim=1)
    disagreement = left_top != right_top
    rows = int(left.shape[0])
    mask = torch.nn.functional.one_hot(
        left_top, num_classes=int(left.shape[1])
    ).bool()
    left_other = left.masked_fill(mask, -torch.inf).max(dim=1).values
    right_other = right.masked_fill(mask, -torch.inf).max(dim=1).values
    left_margin = left.gather(1, left_top[:, None]).squeeze(1) - left_other
    right_margin = right.gather(1, left_top[:, None]).squeeze(1) - right_other
    margin_delta = torch.abs(left_margin - right_margin)
    flips = (left_margin > 0) != (right_margin > 0)
    result.update(
        {
            "top1_disagreement_count": int(disagreement.sum().item()),
            "top1_disagreement_rate": float(disagreement.float().mean().item()),
            "reference_margin_max_abs_change": float(margin_delta.max().item()),
            "reference_margin_mean_abs_change": float(margin_delta.mean().item()),
            "reference_margin_sign_flip_count": int(flips.sum().item()),
            "reference_margin_sign_flip_rate": float(flips.float().mean().item()),
            "rows": rows,
        }
    )
    return result


def _compare_outputs(
    left: tuple[torch.Tensor, torch.Tensor],
    right: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, Any]:
    return {
        "feature": _difference(left[0], right[0]),
        "logits": _logit_difference(left[1], right[1]),
    }


def _load_runtime(runtime_bytes: bytes, device: torch.device) -> torch.jit.ScriptModule:
    try:
        return torch.jit.load(io.BytesIO(runtime_bytes), map_location=device).eval()
    except Exception as exc:
        raise ADV3B02NumericalDiagnosticError(
            "existing runtime is not loadable TorchScript"
        ) from exc


def _build_fresh_runtime(
    eager: torch.nn.Module,
    trace_example: torch.Tensor,
    device: torch.device,
    output: Path,
) -> dict[str, Any]:
    eager.eval()
    if output.exists():
        raise ADV3B02NumericalDiagnosticError(
            "fresh runtime path unexpectedly already exists"
        )
    traced = torch.jit.trace(eager, trace_example, strict=False, check_trace=False)
    buffer = io.BytesIO()
    torch.jit.save(traced, buffer)
    runtime_bytes = buffer.getvalue()
    _write_bytes_new(output, runtime_bytes, "fresh trace runtime")
    fresh = _load_runtime(runtime_bytes, device)
    structure = deployment_bundle._runtime_structure_from_bytes(runtime_bytes)[0]
    return {
        "sha256": _sha256_bytes(runtime_bytes),
        "bytes": len(runtime_bytes),
        "path": str(output),
        "storage_scope": "immutable_unique_trace_builder_output",
        "selected_for_runtime": False,
        "runtime_structure": structure,
        "tensor_registry": _module_state_audit(fresh),
    }


def _run_mode(
    *,
    eager: torch.nn.Module,
    existing: torch.jit.ScriptModule,
    fresh: torch.jit.ScriptModule,
    fresh_audit: Mapping[str, Any],
    probes_cpu: Mapping[int, torch.Tensor],
    device: torch.device,
    mode: str,
    requested_flags: Mapping[str, Any],
) -> dict[str, Any]:
    _apply_flags(requested_flags)
    applied = _flag_snapshot()
    rows: dict[str, Any] = {}
    output_tensor_bytes = 0
    for batch_size in BATCH_SIZES:
        probe_cpu = probes_cpu[int(batch_size)]
        probe = probe_cpu.to(device)
        eager_a = _outputs(eager(probe), rows=int(batch_size))
        eager_b = _outputs(eager(probe), rows=int(batch_size))
        existing_a = _outputs(existing(probe), rows=int(batch_size))
        existing_b = _outputs(existing(probe), rows=int(batch_size))
        fresh_a = _outputs(fresh(probe), rows=int(batch_size))
        fresh_b = _outputs(fresh(probe), rows=int(batch_size))
        output_tensor_bytes += sum(
            int(tensor.numel()) * int(tensor.element_size())
            for pair in (
                eager_a,
                eager_b,
                existing_a,
                existing_b,
                fresh_a,
                fresh_b,
            )
            for tensor in pair
        )
        rows[str(batch_size)] = {
            "batch_size": int(batch_size),
            "generator_seed": PROBE_SEED + int(batch_size),
            "input": _tensor_audit(probe_cpu),
            "output_audits": {
                "eager_a_feature": _tensor_audit(eager_a[0]),
                "eager_a_logits": _tensor_audit(eager_a[1]),
                "existing_runtime_a_feature": _tensor_audit(existing_a[0]),
                "existing_runtime_a_logits": _tensor_audit(existing_a[1]),
                "fresh_trace_a_feature": _tensor_audit(fresh_a[0]),
                "fresh_trace_a_logits": _tensor_audit(fresh_a[1]),
            },
            "comparisons": {
                "eager_a_vs_eager_b": _compare_outputs(eager_a, eager_b),
                "existing_runtime_a_vs_existing_runtime_b": _compare_outputs(
                    existing_a, existing_b
                ),
                "fresh_trace_a_vs_fresh_trace_b": _compare_outputs(
                    fresh_a, fresh_b
                ),
                "eager_a_vs_existing_runtime_a": _compare_outputs(
                    eager_a, existing_a
                ),
                "eager_a_vs_fresh_trace_a": _compare_outputs(eager_a, fresh_a),
                "fresh_trace_a_vs_existing_runtime_a": _compare_outputs(
                    fresh_a, existing_a
                ),
            },
        }
    return {
        "mode": mode,
        "flags": applied,
        "fresh_trace": fresh_audit,
        "batches": rows,
        "output_tensor_bytes": int(output_tensor_bytes),
    }


def _build_eager(
    checkpoint_bytes: bytes, *, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    checkpoint = _load_checkpoint_bytes(checkpoint_bytes)
    if not isinstance(checkpoint, Mapping) or not isinstance(
        checkpoint.get("model"), Mapping
    ):
        raise ADV3B02NumericalDiagnosticError(
            "checkpoint must contain a tensor model state mapping"
        )
    checkpoint_state = _tensor_registry(strip_module_prefix(checkpoint["model"]))
    model, audit = build_exact_ssdg_model_from_checkpoint(
        checkpoint, input_len=INPUT_LEN, device=device
    )
    eager = ADV3B02IdentityRuntime(model).to(device).eval()
    return eager, dict(audit), checkpoint_state


def _device_audit(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "resolved_device": str(device),
        "device_type": str(device.type),
    }
    if device.type == "cuda":
        index = int(device.index or 0)
        properties = torch.cuda.get_device_properties(index)
        result.update(
            {
                "device_index": index,
                "device_name": str(properties.name),
                "compute_capability": [int(properties.major), int(properties.minor)],
                "total_memory_bytes": int(properties.total_memory),
            }
        )
    return result


def _dependency_audit() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(),
        CODE_ROOT / "cvsrffi" / "checkpoint_loading.py",
        CODE_ROOT / "cvsrffi" / "identity_only_forward.py",
        CODE_ROOT / "cvsrffi" / "phase1_adv3b02_deployment_bundle.py",
        CODE_ROOT / "cvsrffi" / "phase2_candidate_capsule.py",
        CODE_ROOT / "scripts" / "export_adv3b02_effective8_torchscript.py",
        CODE_ROOT / "SSDG" / "train_ssdg.py",
        CODE_ROOT / "model_dual_cvsincnet.py",
        CODE_ROOT / "cvsrffi" / "somph_runtime_trust.py",
        CODE_ROOT / "cvsrffi" / "phase1_center_lowrank_prototype_bundle.py",
        CODE_ROOT / "cvsrffi" / "stage2_predictor_bundle.py",
        REPO_ROOT
        / "paper_reproduction"
        / "scripts"
        / "benchmark_cvs_adaptive_rxlight_tta.py",
        REPO_ROOT
        / "paper_reproduction"
        / "scripts"
        / "train_export_cvs_support_lora_adapter.py",
    )
    entries = []
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise ADV3B02NumericalDiagnosticError(
                f"diagnostic code dependency is missing: {path}"
            )
        dependency_bytes = _read_regular_bytes(path, "diagnostic code dependency")
        entries.append(_file_record_from_snapshot(path, dependency_bytes))
    preimage = {"entries": entries}
    loaded_entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name, module in sorted(sys.modules.items()):
        source = getattr(module, "__file__", None)
        if not source:
            continue
        try:
            path = Path(source).resolve()
            relative = path.relative_to(REPO_ROOT).as_posix()
        except (OSError, ValueError):
            continue
        if relative in seen or not path.is_file() or path.is_symlink():
            continue
        seen.add(relative)
        module_bytes = _read_regular_bytes(path, "loaded project module")
        loaded_entries.append(
            {
                "module": str(name),
                **_file_record_from_snapshot(path, module_bytes),
            }
        )
    loaded_preimage = {"loaded_project_modules": loaded_entries}
    return {
        "root_sha256": _sha256_bytes(_canonical_json_bytes(preimage)),
        "entries": entries,
        "loaded_project_module_root_sha256": _sha256_bytes(
            _canonical_json_bytes(loaded_preimage)
        ),
        "loaded_project_modules": loaded_entries,
    }


def _source_execution_contract(
    source_release: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "cvs.development.source_execution_member_contract.v1",
        "source_git_commit": source_release["source_git_commit"],
        "source_archive_sha256": source_release["source_archive_sha256"],
        "source_release_receipt_sha256": source_release["receipt_sha256"],
        "source_manifest_root_sha256": source_release[
            "source_manifest_root_sha256"
        ],
        "source_members": source_release["source_members"],
        "git_policy": source_release["git_policy"],
    }


def _load_source_execution_contract(
    path: str | Path, expected_sha256: str
) -> dict[str, Any]:
    contract_path = Path(path).resolve()
    contract_bytes = _read_regular_bytes(
        contract_path, "source execution member contract"
    )
    contract_sha = _sha256_bytes(contract_bytes)
    if contract_sha != _validate_sha256(
        expected_sha256, "source execution member contract SHA256"
    ):
        raise ADV3B02NumericalDiagnosticError(
            "source execution member contract SHA256 mismatch"
        )
    contract = _json_from_snapshot(
        contract_bytes, "source execution member contract"
    )
    if not isinstance(contract, dict) or contract.get("schema") != (
        "cvs.development.source_execution_member_contract.v1"
    ):
        raise ADV3B02NumericalDiagnosticError(
            "source execution member contract schema drift"
        )
    return {**contract, "contract_path": str(contract_path), "contract_sha256": contract_sha}


def _close_execution_contract_release(
    contract: Mapping[str, Any],
    *,
    source_git_commit: str,
    source_archive_sha256: str,
    source_release_receipt_sha256: str,
) -> None:
    if (
        contract.get("source_git_commit") != source_git_commit
        or contract.get("source_archive_sha256") != source_archive_sha256
        or contract.get("source_release_receipt_sha256")
        != source_release_receipt_sha256
    ):
        raise ADV3B02NumericalDiagnosticError(
            "source execution contract does not bind the signed release"
        )


def _validate_execution_source_binding(
    *,
    dependencies: Mapping[str, Any],
    git: Mapping[str, Any],
    source_git_commit: str,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    if contract.get("source_git_commit") != source_git_commit:
        raise ADV3B02NumericalDiagnosticError(
            "execution source contract Git commit drift"
        )
    signed_members = contract.get("source_members")
    if not isinstance(signed_members, list) or _source_manifest_root(
        signed_members
    ) != contract.get("source_manifest_root_sha256"):
        raise ADV3B02NumericalDiagnosticError(
            "execution source member manifest drift"
        )
    observed_members = sorted(
        (
            {
                "path": row["path"],
                "bytes": row["bytes"],
                "sha256": row["sha256"],
            }
            for row in dependencies["loaded_project_modules"]
        ),
        key=lambda row: row["path"],
    )
    if observed_members != signed_members:
        raise ADV3B02NumericalDiagnosticError(
            "actual imported project dependencies do not exactly match signed archive manifest"
        )
    policy = contract.get("git_policy")
    if not isinstance(policy, Mapping):
        raise ADV3B02NumericalDiagnosticError("execution source Git policy is missing")
    if policy.get("mode") == "git_exact":
        expected = {
            "commit": policy.get("commit"),
            "dirty": policy.get("dirty"),
            "status_root_sha256": policy.get("status_root_sha256"),
            "diff_root_sha256": policy.get("diff_root_sha256"),
            "cached_diff_root_sha256": policy.get("cached_diff_root_sha256"),
            "untracked_root_sha256": policy.get("untracked_root_sha256"),
        }
        observed = {key: git.get(key) for key in expected}
        if git.get("git_available") is not True or observed != expected:
            raise ADV3B02NumericalDiagnosticError(
                "observed Git state does not match signed exact policy"
            )
    elif policy.get("mode") == "signed_manifest_only_no_git":
        if git.get("git_available") is not False:
            raise ADV3B02NumericalDiagnosticError(
                "no-Git source policy cannot authorize a Git worktree"
            )
    else:
        raise ADV3B02NumericalDiagnosticError("execution source Git policy drift")
    return {
        "status": "SIGNED_MEMBER_MANIFEST_EXECUTION_CLOSED",
        "contract_path": contract.get("contract_path"),
        "contract_sha256": contract.get("contract_sha256"),
        "source_manifest_root_sha256": contract["source_manifest_root_sha256"],
        "observed_member_count": len(observed_members),
        "git_policy_mode": policy["mode"],
    }


def _git_audit() -> dict[str, Any]:
    try:
        head_result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            timeout=10,
        )
        status_result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain=v1", "-z"],
            check=True,
            capture_output=True,
            timeout=30,
        )
        diff_result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "diff", "--binary", "HEAD", "--"],
            check=True,
            capture_output=True,
            timeout=60,
        )
        cached_result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "diff", "--cached", "--binary", "--"],
            check=True,
            capture_output=True,
            timeout=60,
        )
        commit = head_result.stdout.decode("ascii").strip()
    except (OSError, subprocess.SubprocessError):
        return {
            "repository_root": str(REPO_ROOT),
            "git_available": False,
            "commit": None,
            "dirty": None,
            "status_root_sha256": None,
            "diff_root_sha256": None,
            "cached_diff_root_sha256": None,
            "untracked": [],
            "untracked_root_sha256": None,
        }
    status_bytes = status_result.stdout
    status_rows = [row for row in status_bytes.split(b"\0") if row]
    untracked = sorted(
        row[3:].decode("utf-8", errors="surrogateescape")
        for row in status_rows
        if row.startswith(b"?? ")
    )
    untracked_entries = []
    for relative in untracked:
        path = REPO_ROOT / relative
        if path.is_file() and not path.is_symlink():
            untracked_bytes = _read_regular_bytes(path, "untracked Git file")
            untracked_entries.append(
                {
                    "path": relative.replace("\\", "/"),
                    "bytes": len(untracked_bytes),
                    "sha256": _sha256_bytes(untracked_bytes),
                }
            )
        else:
            untracked_entries.append(
                {"path": relative.replace("\\", "/"), "kind": "non_regular_or_directory"}
            )
    return {
        "repository_root": str(REPO_ROOT),
        "git_available": True,
        "commit": commit,
        "dirty": bool(status_rows),
        "status_root_sha256": _sha256_bytes(status_bytes),
        "diff_root_sha256": _sha256_bytes(diff_result.stdout),
        "cached_diff_root_sha256": _sha256_bytes(cached_result.stdout),
        "untracked": untracked_entries,
        "untracked_root_sha256": _sha256_bytes(
            _canonical_json_bytes({"untracked": untracked_entries})
        ),
    }


def _driver_version(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {"status": "NOT_APPLICABLE_CPU_CONTROL", "version": None}
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        values = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if not values:
            return {"status": "INCOMPLETE", "version": None, "reason": "empty_output"}
        return {"status": "COMPLETE", "version": values[int(device.index or 0)]}
    except (OSError, subprocess.SubprocessError, IndexError) as exc:
        return {
            "status": "INCOMPLETE",
            "version": None,
            "reason": type(exc).__name__,
        }


def _process_peak_rss_bytes() -> dict[str, Any]:
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS_EX()
            counters.cb = ctypes.sizeof(counters)
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                ctypes.windll.kernel32.GetCurrentProcess(),
                ctypes.byref(counters),
                counters.cb,
            )
            if not ok:
                raise OSError("GetProcessMemoryInfo failed")
            return {
                "status": "COMPLETE",
                "peak_rss_bytes": int(counters.PeakWorkingSetSize),
                "source": "GetProcessMemoryInfo.PeakWorkingSetSize",
            }
        except (AttributeError, OSError, TypeError) as exc:
            return {
                "status": "INCOMPLETE",
                "peak_rss_bytes": None,
                "reason": type(exc).__name__,
            }
    try:
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform != "darwin":
            peak *= 1024
        return {
            "status": "COMPLETE",
            "peak_rss_bytes": peak,
            "source": "resource.getrusage.ru_maxrss",
        }
    except (ImportError, OSError, ValueError) as exc:
        return {
            "status": "INCOMPLETE",
            "peak_rss_bytes": None,
            "reason": type(exc).__name__,
        }


def _software_audit(
    device: torch.device,
    source_git_commit: str,
    *,
    source_execution_contract: Mapping[str, Any] | None,
    allow_unsigned_unit_fixture: bool,
) -> dict[str, Any]:
    git = _git_audit()
    dependencies = _dependency_audit()
    if source_execution_contract is None:
        if not allow_unsigned_unit_fixture:
            raise ADV3B02NumericalDiagnosticError(
                "production worker requires signed source execution member contract"
            )
        source_binding = {
            "status": "UNIT_TEST_FIXTURE_UNBOUND_NOT_AUTHORIZED",
            "formal_authority": False,
        }
    else:
        source_binding = _validate_execution_source_binding(
            dependencies=dependencies,
            git=git,
            source_git_commit=source_git_commit,
            contract=source_execution_contract,
        )
    observed_commit = git.get("commit")
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "cudnn": int(torch.backends.cudnn.version())
        if torch.backends.cudnn.version() is not None
        else None,
        "nvidia_driver": _driver_version(device),
        "git": {
            **git,
            "declared_source_commit": source_git_commit,
            "observed_matches_declared": observed_commit == source_git_commit
            if observed_commit is not None
            else None,
        },
        "code_dependencies": dependencies,
        "source_execution_binding": source_binding,
    }


def _run_device_suite(
    *,
    checkpoint_bytes: bytes,
    runtime_bytes: bytes,
    fresh_runtime_bytes: bytes,
    fresh_runtime_audit: Mapping[str, Any],
    probes_cpu: Mapping[int, torch.Tensor],
    device: torch.device,
    scope: str,
    mode: str,
    requested_flags: Mapping[str, Any],
) -> dict[str, Any]:
    _apply_flags(requested_flags)
    eager, checkpoint_audit, checkpoint_state = _build_eager(
        checkpoint_bytes, device=device
    )
    existing = _load_runtime(runtime_bytes, device)
    fresh = _load_runtime(fresh_runtime_bytes, device)
    result = _run_mode(
        eager=eager,
        existing=existing,
        fresh=fresh,
        fresh_audit=fresh_runtime_audit,
        probes_cpu=probes_cpu,
        device=device,
        mode=mode,
        requested_flags=requested_flags,
    )
    return {
        "scope": scope,
        "authority_scope": "mechanism_comparison_only",
        "device": _device_audit(device),
        "checkpoint_load_audit": checkpoint_audit,
        "checkpoint_model_state": checkpoint_state,
        "eager_tensor_registry": _module_state_audit(eager),
        "existing_runtime_tensor_registry": _module_state_audit(existing),
        "existing_runtime_structure": deployment_bundle._runtime_structure_from_bytes(
            runtime_bytes
        )[0],
        "mode_result": result,
    }


def _trace_builder_diagnostic(
    *,
    checkpoint_path: str | Path,
    fresh_runtime_out: str | Path,
    trace_builder_artifact_out: str | Path,
    device: str,
    source_git_commit: str,
    source_archive_sha256: str,
    source_release_receipt_sha256: str,
    source_execution_contract_path: str | Path | None = None,
    source_execution_contract_sha256: str | None = None,
    _allow_cpu_for_tests: bool = False,
) -> dict[str, Any]:
    _load_worker_dependencies()
    torch.set_grad_enabled(False)
    started = time.perf_counter()
    checkpoint = Path(checkpoint_path).resolve()
    runtime_output = Path(fresh_runtime_out).resolve()
    artifact_output = Path(trace_builder_artifact_out).resolve()
    if runtime_output.exists() or artifact_output.exists():
        raise ADV3B02NumericalDiagnosticError(
            "refusing to overwrite trace builder runtime/artifact"
        )
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") is not None:
        raise ADV3B02NumericalDiagnosticError(
            "trace builder baseline environment must not set CUBLAS_WORKSPACE_CONFIG"
        )
    resolved_device = _resolve_primary_device(
        device, _allow_cpu_for_tests=_allow_cpu_for_tests
    )
    checkpoint_bytes = _read_regular_bytes(checkpoint, "checkpoint")
    checkpoint_sha = _sha256_bytes(checkpoint_bytes)
    if checkpoint_sha != BASE_CHECKPOINT_SHA256:
        raise ADV3B02NumericalDiagnosticError("checkpoint is not strict ADV3B02")
    source_commit = str(source_git_commit).strip().lower()
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ADV3B02NumericalDiagnosticError("source_git_commit must be lowercase SHA1")
    source_archive_sha = _validate_sha256(
        source_archive_sha256, "source archive SHA256"
    )
    source_receipt_sha = _validate_sha256(
        source_release_receipt_sha256, "source release receipt SHA256"
    )
    execution_contract = None
    if source_execution_contract_path is not None or source_execution_contract_sha256 is not None:
        if source_execution_contract_path is None or source_execution_contract_sha256 is None:
            raise ADV3B02NumericalDiagnosticError(
                "source execution contract path/SHA must be supplied together"
            )
        execution_contract = _load_source_execution_contract(
            source_execution_contract_path, source_execution_contract_sha256
        )
        _close_execution_contract_release(
            execution_contract,
            source_git_commit=source_commit,
            source_archive_sha256=source_archive_sha,
            source_release_receipt_sha256=source_receipt_sha,
        )
    probes_cpu = _probes()
    probe_spec = _probe_spec(probes_cpu)
    with _restore_global_flags() as baseline_flags:
        _apply_flags(baseline_flags)
        eager, checkpoint_audit, checkpoint_state = _build_eager(
            checkpoint_bytes, device=resolved_device
        )
        fresh_audit = _build_fresh_runtime(
            eager,
            probes_cpu[8][:2].to(resolved_device),
            resolved_device,
            runtime_output,
        )
    software = _software_audit(
        resolved_device,
        source_commit,
        source_execution_contract=execution_contract,
        allow_unsigned_unit_fixture=_allow_cpu_for_tests,
    )
    payload = {
        "schema": TRACE_BUILDER_SCHEMA,
        "status": TRACE_BUILDER_STATUS,
        "formal_authority": False,
        "parity_receipt_emitted": False,
        "runtime_selection_performed": False,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_load_audit": checkpoint_audit,
        "checkpoint_model_state": checkpoint_state,
        "eager_tensor_registry": _module_state_audit(eager),
        "fresh_runtime": fresh_audit,
        "device": _device_audit(resolved_device),
        "fixed_probe_spec": probe_spec,
        "flags": baseline_flags,
        "source_release_binding": {
            "source_git_commit": source_commit,
            "source_archive_sha256": source_archive_sha,
            "source_release_receipt_sha256": source_receipt_sha,
            "source_execution_contract_sha256": (
                execution_contract.get("contract_sha256")
                if execution_contract is not None
                else None
            ),
            "source_manifest_root_sha256": (
                execution_contract.get("source_manifest_root_sha256")
                if execution_contract is not None
                else None
            ),
        },
        "software": software,
        "diagnostic_completeness": (
            "COMPLETE"
            if software["nvidia_driver"]["status"]
            in {"COMPLETE", "NOT_APPLICABLE_CPU_CONTROL"}
            else "INCOMPLETE_NVIDIA_DRIVER_AUDIT"
        ),
        "resources": {
            "wall_time_seconds": float(time.perf_counter() - started),
            "checkpoint_file_bytes": len(checkpoint_bytes),
            "fresh_runtime_file_bytes": int(fresh_audit["bytes"]),
            "process_peak_rss": _process_peak_rss_bytes(),
        },
    }
    serialized = _canonical_json_bytes(payload)
    _write_bytes_new(artifact_output, serialized, "trace builder artifact")
    return {
        "status": TRACE_BUILDER_STATUS,
        "artifact_path": str(artifact_output),
        "artifact_sha256": _sha256_bytes(serialized),
        "artifact_bytes": len(serialized),
        "fresh_runtime_path": str(runtime_output),
        "fresh_runtime_sha256": fresh_audit["sha256"],
        "fresh_runtime_bytes": fresh_audit["bytes"],
        "formal_authority": False,
    }


def _worker_diagnostic(
    *,
    checkpoint_path: str | Path,
    runtime_path: str | Path,
    lineage_evidence_path: str | Path,
    artifact_origin_receipt_path: str | Path | None,
    fresh_runtime_path: str | Path,
    expected_fresh_runtime_sha256: str,
    trace_builder_artifact_sha256: str,
    arm_id: str,
    source_git_commit: str,
    source_archive_sha256: str,
    source_release_receipt_sha256: str,
    worker_artifact_out: str | Path,
    device: str,
    worker_mode: str,
    worker_scope: str,
    source_execution_contract_path: str | Path | None = None,
    source_execution_contract_sha256: str | None = None,
    _allow_cpu_primary_for_tests: bool = False,
) -> dict[str, Any]:
    """Run exactly one numerical mode in one isolated worker process."""

    _load_worker_dependencies()
    torch.set_grad_enabled(False)
    started = time.perf_counter()
    checkpoint = Path(checkpoint_path).resolve()
    runtime = Path(runtime_path).resolve()
    fresh_runtime = Path(fresh_runtime_path).resolve()
    output = Path(worker_artifact_out).resolve()
    if output.exists():
        raise ADV3B02NumericalDiagnosticError(
            "refusing to overwrite numerical worker artifact"
        )
    contract = _arm_contract(arm_id)
    source_commit = str(source_git_commit).strip().lower()
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ADV3B02NumericalDiagnosticError("source_git_commit must be lowercase SHA1")
    if worker_mode not in {"baseline", "deterministic"}:
        raise ADV3B02NumericalDiagnosticError("worker_mode must be baseline or deterministic")
    if worker_scope == "optional_cpu_control":
        if str(device).strip().lower() != "cpu":
            raise ADV3B02NumericalDiagnosticError("CPU control worker must use cpu")
        resolved_device = torch.device("cpu")
    elif worker_scope == "required_primary_cuda":
        resolved_device = _resolve_primary_device(
            device, _allow_cpu_for_tests=_allow_cpu_primary_for_tests
        )
    elif worker_scope == "unit_test_only_cpu_primary" and _allow_cpu_primary_for_tests:
        resolved_device = torch.device("cpu")
    else:
        raise ADV3B02NumericalDiagnosticError("worker_scope is not allowed")
    if worker_mode == "deterministic" and os.environ.get(
        "CUBLAS_WORKSPACE_CONFIG"
    ) != ":4096:8":
        raise ADV3B02NumericalDiagnosticError(
            "deterministic worker requires preregistered CUBLAS_WORKSPACE_CONFIG"
        )
    checkpoint_bytes = _read_regular_bytes(checkpoint, "checkpoint")
    runtime_bytes = _read_regular_bytes(runtime, "existing runtime")
    fresh_runtime_bytes = _read_regular_bytes(fresh_runtime, "fresh trace runtime")
    checkpoint_sha = _sha256_bytes(checkpoint_bytes)
    runtime_sha = _sha256_bytes(runtime_bytes)
    fresh_runtime_sha = _sha256_bytes(fresh_runtime_bytes)
    if checkpoint_sha != BASE_CHECKPOINT_SHA256:
        raise ADV3B02NumericalDiagnosticError("checkpoint is not strict ADV3B02")
    if runtime_sha != contract["sha256"]:
        raise ADV3B02NumericalDiagnosticError(
            "runtime SHA256 does not close against the preregistered arm"
        )
    expected_fresh_sha = _validate_sha256(
        expected_fresh_runtime_sha256, "expected fresh runtime SHA256"
    )
    if fresh_runtime_sha != expected_fresh_sha:
        raise ADV3B02NumericalDiagnosticError("fresh runtime SHA256 mismatch")
    trace_builder_sha = _validate_sha256(
        trace_builder_artifact_sha256, "trace builder artifact SHA256"
    )
    source_archive_sha = _validate_sha256(
        source_archive_sha256, "source archive SHA256"
    )
    source_receipt_sha = _validate_sha256(
        source_release_receipt_sha256, "source release receipt SHA256"
    )
    execution_contract = None
    if source_execution_contract_path is not None or source_execution_contract_sha256 is not None:
        if source_execution_contract_path is None or source_execution_contract_sha256 is None:
            raise ADV3B02NumericalDiagnosticError(
                "source execution contract path/SHA must be supplied together"
            )
        execution_contract = _load_source_execution_contract(
            source_execution_contract_path, source_execution_contract_sha256
        )
        _close_execution_contract_release(
            execution_contract,
            source_git_commit=source_commit,
            source_archive_sha256=source_archive_sha,
            source_release_receipt_sha256=source_receipt_sha,
        )
    lineage_audit = _validate_asset_lineage(
        runtime=runtime,
        runtime_sha256=runtime_sha,
        lineage_evidence_path=lineage_evidence_path,
        artifact_origin_receipt_path=artifact_origin_receipt_path,
        contract=contract,
    )
    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)
    probes_cpu = _probes()
    probe_spec = _probe_spec(probes_cpu)
    fresh_module = _load_runtime(fresh_runtime_bytes, resolved_device)
    fresh_audit = {
        "path": str(fresh_runtime),
        "sha256": fresh_runtime_sha,
        "bytes": len(fresh_runtime_bytes),
        "storage_scope": "immutable_trace_builder_output_read_only",
        "selected_for_runtime": False,
        "runtime_structure": deployment_bundle._runtime_structure_from_bytes(
            fresh_runtime_bytes
        )[0],
        "tensor_registry": _module_state_audit(fresh_module),
        "trace_builder_artifact_sha256": trace_builder_sha,
    }
    with _restore_global_flags() as baseline_flags:
        requested_flags = (
            baseline_flags if worker_mode == "baseline" else _deterministic_flags()
        )
        suite = _run_device_suite(
            checkpoint_bytes=checkpoint_bytes,
            runtime_bytes=runtime_bytes,
            fresh_runtime_bytes=fresh_runtime_bytes,
            fresh_runtime_audit=fresh_audit,
            probes_cpu=probes_cpu,
            device=resolved_device,
            scope=worker_scope,
            mode=worker_mode,
            requested_flags=requested_flags,
        )
    peak_allocated = None
    peak_reserved = None
    if resolved_device.type == "cuda":
        peak_allocated = int(torch.cuda.max_memory_allocated(resolved_device))
        peak_reserved = int(torch.cuda.max_memory_reserved(resolved_device))
    eager_state_bytes = int(suite["eager_tensor_registry"]["state"]["tensor_bytes"])
    existing_state_bytes = int(
        suite["existing_runtime_tensor_registry"]["state"]["tensor_bytes"]
    )
    fresh_state_bytes = int(
        suite["mode_result"]["fresh_trace"]["tensor_registry"]["state"][
            "tensor_bytes"
        ]
    )
    software = _software_audit(
        resolved_device,
        source_commit,
        source_execution_contract=execution_contract,
        allow_unsigned_unit_fixture=(
            worker_scope == "unit_test_only_cpu_primary"
            and _allow_cpu_primary_for_tests
        ),
    )
    payload = {
        "schema": WORKER_SCHEMA,
        "status": WORKER_STATUS,
        "formal_authority": False,
        "parity_receipt_emitted": False,
        "target_access": False,
        "source_cache_access": False,
        "runtime_selection_performed": False,
        "threshold_can_authorize_or_select": False,
        "arm": contract,
        "asset_lineage": lineage_audit,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "existing_runtime_path": str(runtime),
        "existing_runtime_sha256": runtime_sha,
        "runtime_lineage_scope": contract["lineage_scope"],
        "source_release_binding": {
            "source_git_commit": source_commit,
            "source_archive_sha256": source_archive_sha,
            "source_release_receipt_sha256": source_receipt_sha,
            "source_execution_contract_sha256": (
                execution_contract.get("contract_sha256")
                if execution_contract is not None
                else None
            ),
            "source_manifest_root_sha256": (
                execution_contract.get("source_manifest_root_sha256")
                if execution_contract is not None
                else None
            ),
        },
        "fresh_runtime_binding": {
            "path": str(fresh_runtime),
            "sha256": fresh_runtime_sha,
            "trace_builder_artifact_sha256": trace_builder_sha,
        },
        "worker": {
            "mode": worker_mode,
            "scope": worker_scope,
            "pid": int(os.getpid()),
            "startup_environment": {
                "CUBLAS_WORKSPACE_CONFIG": os.environ.get(
                    "CUBLAS_WORKSPACE_CONFIG"
                )
            },
        },
        "fixed_probe_spec": probe_spec,
        "software": software,
        "diagnostic_completeness": (
            "COMPLETE"
            if software["nvidia_driver"]["status"]
            in {"COMPLETE", "NOT_APPLICABLE_CPU_CONTROL"}
            else "INCOMPLETE_NVIDIA_DRIVER_AUDIT"
        ),
        "suite": suite,
        "resources": {
            "wall_time_seconds": float(time.perf_counter() - started),
            "peak_gpu_memory_allocated_bytes": peak_allocated,
            "peak_gpu_memory_reserved_bytes": peak_reserved,
            "checkpoint_file_bytes": len(checkpoint_bytes),
            "runtime_file_bytes": len(runtime_bytes),
            "fresh_runtime_file_bytes": len(fresh_runtime_bytes),
            "checkpoint_model_state_bytes": int(
                suite["checkpoint_model_state"]["tensor_bytes"]
            ),
            "eager_state_bytes": eager_state_bytes,
            "existing_runtime_state_bytes": existing_state_bytes,
            "fresh_trace_state_bytes": fresh_state_bytes,
            "probe_tensor_bytes": sum(
                int(value.numel()) * int(value.element_size())
                for value in probes_cpu.values()
            ),
            "output_tensor_bytes": int(
                suite["mode_result"]["output_tensor_bytes"]
            ),
            "process_peak_rss": _process_peak_rss_bytes(),
        },
    }
    serialized = _canonical_json_bytes(payload)
    _write_bytes_new(output, serialized, "numerical worker artifact")
    return {
        "status": WORKER_STATUS,
        "artifact_path": str(output),
        "artifact_sha256": _sha256_bytes(serialized),
        "artifact_bytes": len(serialized),
        "arm_id": contract["arm_id"],
        "checkpoint_sha256": checkpoint_sha,
        "existing_runtime_path": str(runtime),
        "existing_runtime_sha256": runtime_sha,
        "resolved_device": str(resolved_device),
        "batch_sizes": list(BATCH_SIZES),
        "parity_receipt_emitted": False,
        "formal_authority": False,
    }


def _launch_trace_builder(
    *,
    checkpoint: Path,
    fresh_runtime_output: Path,
    trace_builder_output: Path,
    device: str,
    source_git_commit: str,
    source_archive_sha256: str,
    source_release_receipt_sha256: str,
    source_execution_contract: Path,
    source_execution_contract_sha256: str,
) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.pop("CUBLAS_WORKSPACE_CONFIG", None)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--checkpoint",
        str(checkpoint),
        "--device",
        device,
        "--source-git-commit",
        source_git_commit,
        "--source-archive-sha256",
        source_archive_sha256,
        "--source-release-receipt-sha256",
        source_release_receipt_sha256,
        "--source-execution-contract",
        str(source_execution_contract),
        "--source-execution-contract-sha256",
        source_execution_contract_sha256,
        "--trace-builder",
        "--fresh-runtime-out",
        str(fresh_runtime_output),
        "--trace-builder-artifact-out",
        str(trace_builder_output),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=environment,
        timeout=900,
    )
    if completed.returncode != 0:
        raise ADV3B02NumericalDiagnosticError(
            "fresh trace builder failed: " + completed.stderr[-2000:]
        )
    if (
        not fresh_runtime_output.is_file()
        or fresh_runtime_output.is_symlink()
        or not trace_builder_output.is_file()
        or trace_builder_output.is_symlink()
    ):
        raise ADV3B02NumericalDiagnosticError(
            "fresh trace builder outputs are missing"
        )
    artifact_snapshot = _read_regular_bytes(
        trace_builder_output, "fresh trace builder artifact"
    )
    fresh_snapshot = _read_regular_bytes(
        fresh_runtime_output, "fresh trace builder runtime"
    )
    payload = _json_from_snapshot(
        artifact_snapshot, "fresh trace builder artifact"
    )
    try:
        summary = json.loads(completed.stdout.strip())
    except json.JSONDecodeError as exc:
        raise ADV3B02NumericalDiagnosticError(
            "fresh trace builder stdout is invalid"
        ) from exc
    artifact_sha = _sha256_bytes(artifact_snapshot)
    artifact_bytes = len(artifact_snapshot)
    fresh_sha = _sha256_bytes(fresh_snapshot)
    fresh_bytes = len(fresh_snapshot)
    if (
        payload.get("schema") != TRACE_BUILDER_SCHEMA
        or payload.get("status") != TRACE_BUILDER_STATUS
        or summary.get("status") != TRACE_BUILDER_STATUS
        or summary.get("artifact_path") != str(trace_builder_output)
        or summary.get("artifact_sha256") != artifact_sha
        or summary.get("artifact_bytes") != artifact_bytes
        or summary.get("fresh_runtime_path") != str(fresh_runtime_output)
        or summary.get("fresh_runtime_sha256") != fresh_sha
        or summary.get("fresh_runtime_bytes") != fresh_bytes
    ):
        raise ADV3B02NumericalDiagnosticError(
            "fresh trace builder stdout/artifact/runtime closure failed"
        )
    payload["orchestrator_launch_audit"] = {
        "returncode": int(completed.returncode),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "stdout_summary": summary,
        "trace_builder_artifact_sha256": artifact_sha,
        "trace_builder_artifact_bytes": artifact_bytes,
        "trace_builder_artifact_path": str(trace_builder_output),
        "fresh_runtime_sha256": fresh_sha,
        "fresh_runtime_bytes": fresh_bytes,
        "fresh_runtime_path": str(fresh_runtime_output),
        "environment_applied_before_worker_python_start": True,
        "cublas_removed_before_spawn": True,
    }
    return payload


def _launch_worker(
    *,
    checkpoint: Path,
    runtime: Path,
    lineage_evidence: Path,
    artifact_origin_receipt: Path | None,
    fresh_runtime: Path,
    fresh_runtime_sha256: str,
    trace_builder_artifact_sha256: str,
    arm_id: str,
    source_git_commit: str,
    source_archive_sha256: str,
    source_release_receipt_sha256: str,
    source_execution_contract: Path,
    source_execution_contract_sha256: str,
    worker_output: Path,
    device: str,
    mode: str,
    scope: str,
) -> dict[str, Any]:
    environment = os.environ.copy()
    if mode == "deterministic":
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        environment.pop("CUBLAS_WORKSPACE_CONFIG", None)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--checkpoint",
        str(checkpoint),
        "--runtime",
        str(runtime),
        "--lineage-evidence",
        str(lineage_evidence),
        "--fresh-runtime",
        str(fresh_runtime),
        "--expected-fresh-runtime-sha256",
        fresh_runtime_sha256,
        "--trace-builder-artifact-sha256",
        trace_builder_artifact_sha256,
        "--arm-id",
        arm_id,
        "--source-git-commit",
        source_git_commit,
        "--source-archive-sha256",
        source_archive_sha256,
        "--source-release-receipt-sha256",
        source_release_receipt_sha256,
        "--source-execution-contract",
        str(source_execution_contract),
        "--source-execution-contract-sha256",
        source_execution_contract_sha256,
        "--device",
        device,
        "--worker-mode",
        mode,
        "--worker-scope",
        scope,
        "--worker-artifact-out",
        str(worker_output),
    ]
    if artifact_origin_receipt is not None:
        command.extend(["--artifact-origin-receipt", str(artifact_origin_receipt)])
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=environment,
        timeout=900,
    )
    if completed.returncode != 0:
        raise ADV3B02NumericalDiagnosticError(
            "numerical diagnostic worker failed: "
            f"mode={mode} scope={scope} stderr={completed.stderr[-2000:]}"
        )
    if not worker_output.is_file() or worker_output.is_symlink():
        raise ADV3B02NumericalDiagnosticError(
            "numerical diagnostic worker artifact is missing"
        )
    artifact_snapshot = _read_regular_bytes(
        worker_output, "numerical diagnostic worker artifact"
    )
    payload = _json_from_snapshot(
        artifact_snapshot, "numerical diagnostic worker artifact"
    )
    try:
        summary = json.loads(completed.stdout.strip())
    except json.JSONDecodeError as exc:
        raise ADV3B02NumericalDiagnosticError(
            "numerical worker stdout summary is invalid"
        ) from exc
    artifact_sha = _sha256_bytes(artifact_snapshot)
    artifact_bytes = len(artifact_snapshot)
    if (
        summary.get("status") != WORKER_STATUS
        or summary.get("artifact_path") != str(worker_output)
        or summary.get("artifact_sha256") != artifact_sha
        or summary.get("artifact_bytes") != artifact_bytes
    ):
        raise ADV3B02NumericalDiagnosticError(
            "numerical worker stdout/artifact closure failed"
        )
    payload["orchestrator_launch_audit"] = {
        "returncode": int(completed.returncode),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "stdout_summary": summary,
        "worker_artifact_sha256": artifact_sha,
        "worker_artifact_bytes": artifact_bytes,
        "worker_artifact_path": str(worker_output),
        "environment_applied_before_worker_python_start": True,
        "deterministic_cublas_set_before_spawn": mode == "deterministic",
    }
    return payload


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise ADV3B02NumericalDiagnosticError(
                f"worker artifact closure field is missing: {'.'.join(keys)}"
            )
        value = value[key]
    return value


def _validate_workers(
    workers: Mapping[str, Mapping[str, Any]],
    *,
    contract: Mapping[str, str],
    checkpoint_sha: str,
    runtime_sha: str,
    runtime_path: str | Path,
    artifact_origin_receipt_path: str | Path | None,
    normalized_device: str,
    source_git_commit: str,
    source_archive_sha256: str,
    source_release_receipt_sha256: str,
    source_execution_contract_sha256: str,
    source_manifest_root_sha256: str,
    fresh_runtime_sha256: str,
    trace_builder_artifact_sha256: str,
) -> dict[str, Any]:
    if not workers:
        raise ADV3B02NumericalDiagnosticError("no numerical workers were launched")
    invariant_paths = (
        ("fixed_probe_spec", "root_sha256"),
        ("software", "code_dependencies", "root_sha256"),
        ("software", "code_dependencies", "loaded_project_module_root_sha256"),
        ("software", "git", "commit"),
        ("software", "git", "status_root_sha256"),
        ("software", "git", "diff_root_sha256"),
        ("software", "git", "cached_diff_root_sha256"),
        ("software", "git", "untracked_root_sha256"),
        ("software", "git", "dirty"),
        ("software", "source_execution_binding", "contract_sha256"),
        ("software", "source_execution_binding", "source_manifest_root_sha256"),
        ("suite", "checkpoint_model_state", "root_sha256"),
        ("suite", "eager_tensor_registry", "state", "root_sha256"),
        ("suite", "existing_runtime_tensor_registry", "state", "root_sha256"),
        ("suite", "existing_runtime_structure", "runtime_structure_sha256"),
        ("suite", "mode_result", "fresh_trace", "sha256"),
        (
            "suite",
            "mode_result",
            "fresh_trace",
            "runtime_structure",
            "runtime_structure_sha256",
        ),
        (
            "suite",
            "mode_result",
            "fresh_trace",
            "tensor_registry",
            "state",
            "root_sha256",
        ),
    )
    closure: dict[str, Any] = {}
    expected_runtime_path = str(Path(runtime_path).resolve())
    expected_origin = _expected_runtime_origin(
        contract=contract,
        runtime_path=runtime_path,
        runtime_sha256=runtime_sha,
        artifact_origin_receipt_path=artifact_origin_receipt_path,
    )
    for name, worker in workers.items():
        if name.startswith("primary_cuda_"):
            expected_scope = "required_primary_cuda"
            expected_device = normalized_device
            expected_mode = name.removeprefix("primary_cuda_")
        elif name.startswith("optional_cpu_control_"):
            expected_scope = "optional_cpu_control"
            expected_device = "cpu"
            expected_mode = name.removeprefix("optional_cpu_control_")
        else:
            raise ADV3B02NumericalDiagnosticError(
                f"unexpected numerical worker key: {name}"
            )
        launch = worker.get("orchestrator_launch_audit")
        if (
            worker.get("schema") != WORKER_SCHEMA
            or worker.get("status") != WORKER_STATUS
            or worker.get("formal_authority") is not False
            or worker.get("parity_receipt_emitted") is not False
            or _nested(worker, "arm", "arm_id") != contract["arm_id"]
            or _nested(worker, "asset_lineage", "lineage_evidence_path")
            != str(Path(contract["lineage_evidence_path"]))
            or _nested(worker, "asset_lineage", "lineage_evidence_sha256")
            != contract["lineage_evidence_sha256"]
            or _nested(worker, "asset_lineage", "lineage_scope")
            != contract["lineage_scope"]
            or _nested(worker, "asset_lineage", "runtime_origin")
            != expected_origin
            or worker.get("checkpoint_sha256") != checkpoint_sha
            or worker.get("existing_runtime_path") != expected_runtime_path
            or worker.get("existing_runtime_sha256") != runtime_sha
            or _nested(worker, "fixed_probe_spec", "root_sha256")
            != EXPECTED_PROBE_ROOT_SHA256
            or _nested(worker, "worker", "mode") != expected_mode
            or _nested(worker, "worker", "scope") != expected_scope
            or _nested(worker, "suite", "mode_result", "mode") != expected_mode
            or _nested(worker, "suite", "device", "resolved_device")
            != expected_device
            or _nested(worker, "source_release_binding", "source_git_commit")
            != source_git_commit
            or _nested(worker, "source_release_binding", "source_archive_sha256")
            != source_archive_sha256
            or _nested(
                worker, "source_release_binding", "source_release_receipt_sha256"
            )
            != source_release_receipt_sha256
            or _nested(
                worker,
                "source_release_binding",
                "source_execution_contract_sha256",
            )
            != source_execution_contract_sha256
            or _nested(
                worker,
                "source_release_binding",
                "source_manifest_root_sha256",
            )
            != source_manifest_root_sha256
            or _nested(
                worker, "software", "source_execution_binding", "status"
            )
            != "SIGNED_MEMBER_MANIFEST_EXECUTION_CLOSED"
            or _nested(worker, "fresh_runtime_binding", "sha256")
            != fresh_runtime_sha256
            or _nested(
                worker, "fresh_runtime_binding", "trace_builder_artifact_sha256"
            )
            != trace_builder_artifact_sha256
            or not isinstance(launch, Mapping)
            or launch.get("returncode") != 0
            or launch.get("environment_applied_before_worker_python_start") is not True
            or not isinstance(launch.get("stdout"), str)
            or not isinstance(launch.get("stderr"), str)
            or not isinstance(launch.get("stdout_summary"), Mapping)
            or launch["stdout_summary"].get("artifact_sha256")
            != launch.get("worker_artifact_sha256")
            or launch["stdout_summary"].get("artifact_bytes")
            != launch.get("worker_artifact_bytes")
            or launch["stdout_summary"].get("artifact_path")
            != launch.get("worker_artifact_path")
        ):
            raise ADV3B02NumericalDiagnosticError(
                f"numerical worker authority/identity closure failed: {name}"
            )
        expected_cublas = ":4096:8" if expected_mode == "deterministic" else None
        if (
            _nested(
                worker, "worker", "startup_environment", "CUBLAS_WORKSPACE_CONFIG"
            )
            != expected_cublas
            or _nested(
                worker,
                "suite",
                "mode_result",
                "flags",
                "cublas_workspace_config",
            )
            != expected_cublas
            or launch.get("deterministic_cublas_set_before_spawn")
            is not (expected_mode == "deterministic")
        ):
            raise ADV3B02NumericalDiagnosticError(
                f"numerical worker CUBLAS closure failed: {name}"
            )
    first = next(iter(workers.values()))
    for path in invariant_paths:
        expected = _nested(first, *path)
        values = {name: _nested(worker, *path) for name, worker in workers.items()}
        if any(value != expected for value in values.values()):
            raise ADV3B02NumericalDiagnosticError(
                f"cross-worker invariant drift: {'.'.join(path)}"
            )
        closure[".".join(path)] = expected
    return closure


def _validate_trace_builder(
    trace_builder: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    normalized_device: str,
    source_git_commit: str,
    source_archive_sha256: str,
    source_release_receipt_sha256: str,
    source_execution_contract_sha256: str,
    source_manifest_root_sha256: str,
    fresh_runtime_sha256: str,
    trace_builder_artifact_sha256: str,
    worker_closure: Mapping[str, Any],
) -> None:
    launch = trace_builder.get("orchestrator_launch_audit")
    if (
        trace_builder.get("schema") != TRACE_BUILDER_SCHEMA
        or trace_builder.get("status") != TRACE_BUILDER_STATUS
        or trace_builder.get("formal_authority") is not False
        or trace_builder.get("parity_receipt_emitted") is not False
        or trace_builder.get("checkpoint_sha256") != checkpoint_sha256
        or _nested(trace_builder, "fixed_probe_spec", "root_sha256")
        != EXPECTED_PROBE_ROOT_SHA256
        or _nested(trace_builder, "fresh_runtime", "sha256")
        != fresh_runtime_sha256
        or _nested(trace_builder, "device", "resolved_device") != normalized_device
    ):
        raise ADV3B02NumericalDiagnosticError(
            "fresh trace builder identity closure failed"
        )
    if (
        _nested(trace_builder, "source_release_binding", "source_git_commit")
        != source_git_commit
        or _nested(
            trace_builder, "source_release_binding", "source_archive_sha256"
        )
        != source_archive_sha256
        or _nested(
            trace_builder,
            "source_release_binding",
            "source_release_receipt_sha256",
        )
        != source_release_receipt_sha256
        or _nested(
            trace_builder,
            "source_release_binding",
            "source_execution_contract_sha256",
        )
        != source_execution_contract_sha256
        or _nested(
            trace_builder,
            "source_release_binding",
            "source_manifest_root_sha256",
        )
        != source_manifest_root_sha256
        or _nested(
            trace_builder, "software", "source_execution_binding", "status"
        )
        != "SIGNED_MEMBER_MANIFEST_EXECUTION_CLOSED"
        or _nested(trace_builder, "flags", "cublas_workspace_config") is not None
        or not isinstance(launch, Mapping)
        or launch.get("returncode") != 0
        or launch.get("trace_builder_artifact_sha256")
        != trace_builder_artifact_sha256
        or launch.get("fresh_runtime_sha256") != fresh_runtime_sha256
        or launch.get("stdout_summary", {}).get("artifact_sha256")
        != trace_builder_artifact_sha256
        or launch.get("stdout_summary", {}).get("fresh_runtime_sha256")
        != fresh_runtime_sha256
        or launch.get("stdout_summary", {}).get("artifact_path")
        != launch.get("trace_builder_artifact_path")
        or launch.get("stdout_summary", {}).get("fresh_runtime_path")
        != launch.get("fresh_runtime_path")
        or launch.get("environment_applied_before_worker_python_start") is not True
        or launch.get("cublas_removed_before_spawn") is not True
    ):
        raise ADV3B02NumericalDiagnosticError(
            "fresh trace builder launch/source closure failed"
        )
    comparisons = {
        "software.code_dependencies.root_sha256": _nested(
            trace_builder, "software", "code_dependencies", "root_sha256"
        ),
        "software.code_dependencies.loaded_project_module_root_sha256": _nested(
            trace_builder,
            "software",
            "code_dependencies",
            "loaded_project_module_root_sha256",
        ),
        "suite.checkpoint_model_state.root_sha256": _nested(
            trace_builder, "checkpoint_model_state", "root_sha256"
        ),
        "suite.eager_tensor_registry.state.root_sha256": _nested(
            trace_builder, "eager_tensor_registry", "state", "root_sha256"
        ),
        "suite.mode_result.fresh_trace.sha256": _nested(
            trace_builder, "fresh_runtime", "sha256"
        ),
        "suite.mode_result.fresh_trace.runtime_structure.runtime_structure_sha256": _nested(
            trace_builder,
            "fresh_runtime",
            "runtime_structure",
            "runtime_structure_sha256",
        ),
        "suite.mode_result.fresh_trace.tensor_registry.state.root_sha256": _nested(
            trace_builder, "fresh_runtime", "tensor_registry", "state", "root_sha256"
        ),
    }
    for key, value in comparisons.items():
        if worker_closure.get(key) != value:
            raise ADV3B02NumericalDiagnosticError(
                f"trace-builder/worker invariant drift: {key}"
            )


def _nvidia_completeness_summary(
    trace_builder: Mapping[str, Any],
    workers: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    incomplete = []
    if _nested(trace_builder, "software", "nvidia_driver", "status") == "INCOMPLETE":
        incomplete.append("trace_builder")
    incomplete.extend(
        name
        for name, worker in workers.items()
        if _nested(worker, "software", "nvidia_driver", "status") == "INCOMPLETE"
    )
    return {
        "diagnostic_completeness": (
            "INCOMPLETE_NVIDIA_DRIVER_AUDIT"
            if incomplete
            else "COMPLETE_DEVELOPMENT_DIAGNOSTIC"
        ),
        "incomplete_nvidia_smi_components": incomplete,
    }


def diagnose_runtime_numerics(
    *,
    checkpoint_path: str | Path,
    runtime_path: str | Path,
    lineage_evidence_path: str | Path,
    artifact_origin_receipt_path: str | Path | None,
    arm_id: str,
    source_archive_path: str | Path,
    source_release_receipt_path: str | Path,
    artifact_out: str | Path,
    device: str,
    include_cpu_control: bool = False,
    _allow_parent_torch_for_tests: bool = False,
    _allow_unit_source_fixture: bool = False,
) -> dict[str, Any]:
    """Orchestrate isolated workers and write one non-authority artifact."""

    started = time.perf_counter()
    torch_present_at_entry = "torch" in sys.modules
    if torch_present_at_entry and not _allow_parent_torch_for_tests:
        raise ADV3B02NumericalDiagnosticError(
            "parent orchestrator must start before Torch is imported"
        )
    checkpoint = Path(checkpoint_path).resolve()
    runtime = Path(runtime_path).resolve()
    output = Path(artifact_out).resolve()
    if output.exists():
        raise ADV3B02NumericalDiagnosticError(
            "refusing to overwrite numerical diagnostic artifact"
        )
    contract = _arm_contract(arm_id)
    checkpoint_bytes = _read_regular_bytes(checkpoint, "checkpoint")
    runtime_bytes = _read_regular_bytes(runtime, "existing runtime")
    checkpoint_sha = _sha256_bytes(checkpoint_bytes)
    runtime_sha = _sha256_bytes(runtime_bytes)
    if checkpoint_sha != BASE_CHECKPOINT_SHA256:
        raise ADV3B02NumericalDiagnosticError("checkpoint is not strict ADV3B02")
    if runtime_sha != contract["sha256"]:
        raise ADV3B02NumericalDiagnosticError(
            "runtime SHA256 does not close against the preregistered arm"
        )
    source_release = _validate_source_release(
        source_archive_path=source_archive_path,
        source_release_receipt_path=source_release_receipt_path,
        _unit_test_fixture=_allow_unit_source_fixture,
    )
    source_commit = source_release["source_git_commit"]
    source_archive_sha = source_release["source_archive_sha256"]
    source_receipt_sha = source_release["receipt_sha256"]
    execution_contract = _source_execution_contract(source_release)
    execution_contract_bytes = _canonical_json_bytes(execution_contract)
    execution_contract_sha = _sha256_bytes(execution_contract_bytes)
    source_manifest_root = source_release["source_manifest_root_sha256"]
    tracemalloc.start()
    lineage_audit = _validate_asset_lineage(
        runtime=runtime,
        runtime_sha256=runtime_sha,
        lineage_evidence_path=lineage_evidence_path,
        artifact_origin_receipt_path=artifact_origin_receipt_path,
        contract=contract,
    )
    normalized_device = _validate_cuda_request_without_initializing(device)
    workers: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="adv3b02_numerical_workers_") as directory:
        worker_root = Path(directory)
        fresh_runtime = worker_root / "fresh_trace_runtime.ts"
        trace_builder_output = worker_root / "trace_builder.json"
        execution_contract_path = worker_root / "source_execution_contract.json"
        _write_bytes_new(
            execution_contract_path,
            execution_contract_bytes,
            "source execution member contract",
        )
        trace_builder = _launch_trace_builder(
            checkpoint=checkpoint,
            fresh_runtime_output=fresh_runtime,
            trace_builder_output=trace_builder_output,
            device=normalized_device,
            source_git_commit=source_commit,
            source_archive_sha256=source_archive_sha,
            source_release_receipt_sha256=source_receipt_sha,
            source_execution_contract=execution_contract_path,
            source_execution_contract_sha256=execution_contract_sha,
        )
        trace_builder_sha = _nested(
            trace_builder,
            "orchestrator_launch_audit",
            "trace_builder_artifact_sha256",
        )
        fresh_runtime_sha = _nested(
            trace_builder, "orchestrator_launch_audit", "fresh_runtime_sha256"
        )
        for mode in ("baseline", "deterministic"):
            key = f"primary_cuda_{mode}"
            workers[key] = _launch_worker(
                checkpoint=checkpoint,
                runtime=runtime,
                lineage_evidence=Path(lineage_evidence_path).resolve(),
                artifact_origin_receipt=Path(artifact_origin_receipt_path).resolve()
                if artifact_origin_receipt_path is not None
                else None,
                fresh_runtime=fresh_runtime,
                fresh_runtime_sha256=fresh_runtime_sha,
                trace_builder_artifact_sha256=trace_builder_sha,
                arm_id=contract["arm_id"],
                source_git_commit=source_commit,
                source_archive_sha256=source_archive_sha,
                source_release_receipt_sha256=source_receipt_sha,
                source_execution_contract=execution_contract_path,
                source_execution_contract_sha256=execution_contract_sha,
                worker_output=worker_root / f"{key}.json",
                device=normalized_device,
                mode=mode,
                scope="required_primary_cuda",
            )
        if include_cpu_control:
            for mode in ("baseline", "deterministic"):
                key = f"optional_cpu_control_{mode}"
                workers[key] = _launch_worker(
                    checkpoint=checkpoint,
                    runtime=runtime,
                    lineage_evidence=Path(lineage_evidence_path).resolve(),
                    artifact_origin_receipt=Path(artifact_origin_receipt_path).resolve()
                    if artifact_origin_receipt_path is not None
                    else None,
                    fresh_runtime=fresh_runtime,
                    fresh_runtime_sha256=fresh_runtime_sha,
                    trace_builder_artifact_sha256=trace_builder_sha,
                    arm_id=contract["arm_id"],
                    source_git_commit=source_commit,
                    source_archive_sha256=source_archive_sha,
                    source_release_receipt_sha256=source_receipt_sha,
                    source_execution_contract=execution_contract_path,
                    source_execution_contract_sha256=execution_contract_sha,
                    worker_output=worker_root / f"{key}.json",
                    device="cpu",
                    mode=mode,
                    scope="optional_cpu_control",
                )
        closure = _validate_workers(
            workers,
            contract=contract,
            checkpoint_sha=checkpoint_sha,
            runtime_sha=runtime_sha,
            runtime_path=runtime,
            artifact_origin_receipt_path=artifact_origin_receipt_path,
            normalized_device=normalized_device,
            source_git_commit=source_commit,
            source_archive_sha256=source_archive_sha,
            source_release_receipt_sha256=source_receipt_sha,
            source_execution_contract_sha256=execution_contract_sha,
            source_manifest_root_sha256=source_manifest_root,
            fresh_runtime_sha256=fresh_runtime_sha,
            trace_builder_artifact_sha256=trace_builder_sha,
        )
        _validate_trace_builder(
            trace_builder,
            checkpoint_sha256=checkpoint_sha,
            normalized_device=normalized_device,
            source_git_commit=source_commit,
            source_archive_sha256=source_archive_sha,
            source_release_receipt_sha256=source_receipt_sha,
            source_execution_contract_sha256=execution_contract_sha,
            source_manifest_root_sha256=source_manifest_root,
            fresh_runtime_sha256=fresh_runtime_sha,
            trace_builder_artifact_sha256=trace_builder_sha,
            worker_closure=closure,
        )
    parent_current, parent_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    completeness = _nvidia_completeness_summary(trace_builder, workers)
    payload = {
        "schema": DIAGNOSTIC_SCHEMA,
        "status": DIAGNOSTIC_STATUS,
        "formal_authority": False,
        "parity_receipt_emitted": False,
        "target_access": False,
        "source_cache_access": False,
        "runtime_selection_performed": False,
        "threshold_can_authorize_or_select": False,
        "cpu_control_can_substitute_cuda_or_authorize": False,
        "cpu_control_scope": "optional_mechanism_comparison_only",
        "arm": contract,
        "asset_lineage": lineage_audit,
        "source_release": source_release,
        "source_execution_contract": {
            "sha256": execution_contract_sha,
            "source_manifest_root_sha256": source_manifest_root,
            "member_count": len(source_release["source_members"]),
        },
        **completeness,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "existing_runtime_path": str(runtime),
        "existing_runtime_sha256": runtime_sha,
        "fixed_contract": {
            "input_len": INPUT_LEN,
            "probe_seed": PROBE_SEED,
            "batch_sizes": list(BATCH_SIZES),
            "probe_root_sha256": EXPECTED_PROBE_ROOT_SHA256,
            "reference_threshold_annotation_only": REFERENCE_THRESHOLD,
        },
        "cross_worker_closure": closure,
        "trace_builder": trace_builder,
        "workers": workers,
        "parent_orchestrator_audit": {
            "torch_in_sys_modules_at_entry": torch_present_at_entry,
            "cuda_not_initialized_parent": not torch_present_at_entry,
            "parent_model_forward_count": 0,
            "parent_cuda_api_call_count": 0,
            "wall_time_seconds": float(time.perf_counter() - started),
            "python_tracemalloc_current_bytes": int(parent_current),
            "python_tracemalloc_peak_bytes": int(parent_peak),
            "process_peak_rss": _process_peak_rss_bytes(),
        },
    }
    serialized = _canonical_json_bytes(payload)
    _write_bytes_new(output, serialized, "numerical diagnostic artifact")
    return {
        "status": DIAGNOSTIC_STATUS,
        "artifact_path": str(output),
        "artifact_sha256": _sha256_bytes(serialized),
        "artifact_bytes": len(serialized),
        "arm_id": contract["arm_id"],
        "checkpoint_sha256": checkpoint_sha,
        "existing_runtime_path": str(runtime),
        "existing_runtime_sha256": runtime_sha,
        "resolved_device": normalized_device,
        "batch_sizes": list(BATCH_SIZES),
        "parity_receipt_emitted": False,
        "formal_authority": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--lineage-evidence", type=Path)
    parser.add_argument("--artifact-origin-receipt", type=Path)
    parser.add_argument("--arm-id", choices=sorted(RUNTIME_ARMS))
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--source-release-receipt", type=Path)
    parser.add_argument("--artifact-out", type=Path)
    parser.add_argument("--device", required=True, help="Explicit cuda:<index>")
    parser.add_argument("--include-cpu-control", action="store_true")
    parser.add_argument(
        "--worker-mode", choices=("baseline", "deterministic"), help=argparse.SUPPRESS
    )
    parser.add_argument("--worker-scope", help=argparse.SUPPRESS)
    parser.add_argument("--worker-artifact-out", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--fresh-runtime", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--expected-fresh-runtime-sha256", help=argparse.SUPPRESS
    )
    parser.add_argument("--trace-builder-artifact-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--source-git-commit", help=argparse.SUPPRESS)
    parser.add_argument("--source-archive-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--source-release-receipt-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--source-execution-contract", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--source-execution-contract-sha256", help=argparse.SUPPRESS
    )
    parser.add_argument("--trace-builder", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--fresh-runtime-out", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--trace-builder-artifact-out", type=Path, help=argparse.SUPPRESS
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.trace_builder:
        required = (
            args.fresh_runtime_out,
            args.trace_builder_artifact_out,
            args.source_git_commit,
            args.source_archive_sha256,
            args.source_release_receipt_sha256,
            args.source_execution_contract,
            args.source_execution_contract_sha256,
        )
        if any(value is None for value in required):
            raise ADV3B02NumericalDiagnosticError(
                "trace builder hidden arguments are incomplete"
            )
        summary = _trace_builder_diagnostic(
            checkpoint_path=args.checkpoint,
            fresh_runtime_out=args.fresh_runtime_out,
            trace_builder_artifact_out=args.trace_builder_artifact_out,
            device=args.device,
            source_git_commit=args.source_git_commit,
            source_archive_sha256=args.source_archive_sha256,
            source_release_receipt_sha256=args.source_release_receipt_sha256,
            source_execution_contract_path=args.source_execution_contract,
            source_execution_contract_sha256=args.source_execution_contract_sha256,
        )
    elif args.worker_mode is not None:
        required = (
            args.runtime,
            args.lineage_evidence,
            args.arm_id,
            args.worker_scope,
            args.worker_artifact_out,
            args.fresh_runtime,
            args.expected_fresh_runtime_sha256,
            args.trace_builder_artifact_sha256,
            args.source_git_commit,
            args.source_archive_sha256,
            args.source_release_receipt_sha256,
            args.source_execution_contract,
            args.source_execution_contract_sha256,
        )
        if any(value is None for value in required):
            raise ADV3B02NumericalDiagnosticError(
                "worker hidden arguments are incomplete"
            )
        summary = _worker_diagnostic(
            checkpoint_path=args.checkpoint,
            runtime_path=args.runtime,
            lineage_evidence_path=args.lineage_evidence,
            artifact_origin_receipt_path=args.artifact_origin_receipt,
            fresh_runtime_path=args.fresh_runtime,
            expected_fresh_runtime_sha256=args.expected_fresh_runtime_sha256,
            trace_builder_artifact_sha256=args.trace_builder_artifact_sha256,
            arm_id=args.arm_id,
            source_git_commit=args.source_git_commit,
            source_archive_sha256=args.source_archive_sha256,
            source_release_receipt_sha256=args.source_release_receipt_sha256,
            source_execution_contract_path=args.source_execution_contract,
            source_execution_contract_sha256=args.source_execution_contract_sha256,
            worker_artifact_out=args.worker_artifact_out,
            device=args.device,
            worker_mode=args.worker_mode,
            worker_scope=args.worker_scope,
        )
    else:
        if args.source_archive is None or args.source_release_receipt is None:
            print(
                json.dumps(
                    {
                        "status": BLOCKED_SOURCE_STATUS,
                        "artifact_emitted": False,
                        "formal_authority": False,
                        "reason": "external signed source archive and receipt are required",
                    },
                    ensure_ascii=True,
                    allow_nan=False,
                    sort_keys=True,
                )
            )
            return 2
        required = (
            args.runtime,
            args.lineage_evidence,
            args.arm_id,
            args.source_archive,
            args.source_release_receipt,
            args.artifact_out,
        )
        if any(value is None for value in required):
            raise ADV3B02NumericalDiagnosticError(
                "orchestrator arguments are incomplete"
            )
        try:
            summary = diagnose_runtime_numerics(
                checkpoint_path=args.checkpoint,
                runtime_path=args.runtime,
                lineage_evidence_path=args.lineage_evidence,
                artifact_origin_receipt_path=args.artifact_origin_receipt,
                arm_id=args.arm_id,
                source_archive_path=args.source_archive,
                source_release_receipt_path=args.source_release_receipt,
                artifact_out=args.artifact_out,
                device=args.device,
                include_cpu_control=bool(args.include_cpu_control),
            )
        except MissingSignedSourceReceiptError:
            print(
                json.dumps(
                    {
                        "status": BLOCKED_SOURCE_STATUS,
                        "artifact_emitted": False,
                        "formal_authority": False,
                        "reason": "external signed source archive or receipt is missing",
                    },
                    ensure_ascii=True,
                    allow_nan=False,
                    sort_keys=True,
                )
            )
            return 2
    print(json.dumps(summary, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
