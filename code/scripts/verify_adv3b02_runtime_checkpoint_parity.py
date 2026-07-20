"""Verify an existing ADV3B02 base runtime against its checkpoint.

This utility creates a development parity receipt from a fresh numerical
comparison.  It does not create an external signature or formal deployment
authority.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
for value in (str(REPO_ROOT), str(CODE_ROOT)):
    while value in sys.path:
        sys.path.remove(value)
for value in (str(REPO_ROOT), str(CODE_ROOT)):
    sys.path.insert(0, value)

from cvsrffi.checkpoint_loading import build_exact_ssdg_model_from_checkpoint  # noqa: E402
from cvsrffi.identity_only_forward import identity_only_feature_forward  # noqa: E402
from cvsrffi import phase1_adv3b02_deployment_bundle as deployment_bundle  # noqa: E402
from cvsrffi.phase2_candidate_capsule import BASE_CHECKPOINT_SHA256  # noqa: E402
from scripts.export_adv3b02_effective8_torchscript import (  # noqa: E402
    ADV3B02IdentityRuntime,
)


RECEIPT_SCHEMA = "cvs.phase1.runtime_checkpoint_parity_receipt.v1"


class ADV3B02RuntimeParityError(ValueError):
    """Raised when checkpoint/runtime identity or numerical parity drifts."""


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_regular_bytes(path: Path, name: str) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise ADV3B02RuntimeParityError(f"{name} must be a regular file")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ADV3B02RuntimeParityError(f"failed to read {name}") from exc


def _load_checkpoint_bytes(value: bytes) -> Any:
    try:
        return torch.load(io.BytesIO(value), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(io.BytesIO(value), map_location="cpu")


def _array_receipt(value: torch.Tensor) -> dict[str, Any]:
    array = np.ascontiguousarray(value.detach().float().cpu().numpy())
    return {
        "shape": [int(item) for item in array.shape],
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(array.view(np.uint8)).hexdigest(),
    }


def _runtime_outputs(value: Any, *, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ADV3B02RuntimeParityError("runtime output must be (z160, logits)")
    feature, logits = value
    if (
        not torch.is_tensor(feature)
        or not torch.is_tensor(logits)
        or tuple(feature.shape) != (rows, 160)
        or logits.ndim != 2
        or logits.shape[0] != rows
        or logits.shape[1] < 2
        or not torch.isfinite(feature).all()
        or not torch.isfinite(logits).all()
    ):
        raise ADV3B02RuntimeParityError("runtime output shape/finite contract drift")
    return feature.detach().float().cpu(), logits.detach().float().cpu()


def _max_abs(left: torch.Tensor, right: torch.Tensor, field: str) -> float:
    if tuple(left.shape) != tuple(right.shape):
        raise ADV3B02RuntimeParityError(f"parity shape mismatch: {field}")
    value = float(torch.max(torch.abs(left - right)).item())
    if not np.isfinite(value):
        raise ADV3B02RuntimeParityError(f"parity is non-finite: {field}")
    return value


def _resolve_device(requested: str) -> torch.device:
    value = str(requested).strip().lower()
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        value = "cuda:0"
    if not value.startswith("cuda:"):
        raise ADV3B02RuntimeParityError("device must be cpu or cuda:<index>")
    try:
        index = int(value.split(":", 1)[1])
    except (TypeError, ValueError) as exc:
        raise ADV3B02RuntimeParityError("CUDA device index is invalid") from exc
    if (
        index < 0
        or not torch.cuda.is_available()
        or index >= int(torch.cuda.device_count())
    ):
        raise ADV3B02RuntimeParityError(
            "requested CUDA device is unavailable; CPU fallback is forbidden"
        )
    return torch.device(f"cuda:{index}")


@torch.no_grad()
def verify_runtime_checkpoint(
    *,
    checkpoint_path: str | Path,
    runtime_path: str | Path,
    receipt_out: str | Path,
    vector_audit_out: str | Path,
    input_len: int,
    parity_seed: int,
    parity_rows: int,
    device: str,
    max_abs_tolerance: float = 1.0e-5,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path).resolve()
    runtime_file = Path(runtime_path).resolve()
    output = Path(receipt_out).resolve()
    vector_output = Path(vector_audit_out).resolve()
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise ADV3B02RuntimeParityError("checkpoint must be a regular file")
    if not runtime_file.is_file() or runtime_file.is_symlink():
        raise ADV3B02RuntimeParityError("runtime must be a regular file")
    if isinstance(input_len, bool) or int(input_len) <= 0:
        raise ADV3B02RuntimeParityError("input_len must be positive")
    if isinstance(parity_rows, bool) or not 2 <= int(parity_rows) <= 255:
        raise ADV3B02RuntimeParityError(
            "parity_rows must be an intermediate batch in [2,255]"
        )
    if not 0.0 < float(max_abs_tolerance) <= 1.0e-5:
        raise ADV3B02RuntimeParityError("tolerance must be in (0,1e-5]")
    if output == vector_output or output.exists() or vector_output.exists():
        raise ADV3B02RuntimeParityError(
            "refusing to overwrite or alias parity receipt/vector audit"
        )
    checkpoint_bytes = _read_regular_bytes(checkpoint, "checkpoint")
    runtime_bytes = _read_regular_bytes(runtime_file, "runtime")
    checkpoint_sha = hashlib.sha256(checkpoint_bytes).hexdigest()
    runtime_sha = hashlib.sha256(runtime_bytes).hexdigest()
    if checkpoint_sha != BASE_CHECKPOINT_SHA256:
        raise ADV3B02RuntimeParityError("checkpoint is not strict ADV3B02")

    runtime_device = _resolve_device(device)
    checkpoint_value = _load_checkpoint_bytes(checkpoint_bytes)
    model, checkpoint_audit = build_exact_ssdg_model_from_checkpoint(
        checkpoint_value,
        input_len=int(input_len),
        device=runtime_device,
    )
    eager = ADV3B02IdentityRuntime(model).to(runtime_device).eval()
    try:
        scripted = torch.jit.load(
            io.BytesIO(runtime_bytes), map_location=runtime_device
        ).eval()
    except Exception as exc:
        raise ADV3B02RuntimeParityError("runtime is not loadable TorchScript") from exc
    batch_sizes = tuple(sorted({1, int(parity_rows), 256}))
    diagnostics: dict[str, dict[str, float]] = {}
    vector_rows: list[dict[str, Any]] = []
    maximum = 0.0
    for batch_size in batch_sizes:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(parity_seed) + int(batch_size))
        probes_cpu = torch.randn(
            int(batch_size),
            2,
            int(input_len),
            generator=generator,
            dtype=torch.float32,
        )
        probes = probes_cpu.to(runtime_device)
        eager_feature, eager_logits = _runtime_outputs(
            eager(probes), rows=int(batch_size)
        )
        script_feature, script_logits = _runtime_outputs(
            scripted(probes), rows=int(batch_size)
        )
        batch_diagnostics = {
            "feature": _max_abs(
                eager_feature, script_feature, f"z160.batch{batch_size}"
            ),
            "logits": _max_abs(
                eager_logits, script_logits, f"logits.batch{batch_size}"
            ),
        }
        diagnostics[str(batch_size)] = batch_diagnostics
        maximum = max(maximum, *batch_diagnostics.values())
        vector_rows.append(
            {
                "batch_size": int(batch_size),
                "generator_seed": int(parity_seed) + int(batch_size),
                "input": _array_receipt(probes_cpu),
                "checkpoint_z160": _array_receipt(eager_feature),
                "checkpoint_logits": _array_receipt(eager_logits),
                "runtime_z160": _array_receipt(script_feature),
                "runtime_logits": _array_receipt(script_logits),
                "max_abs_delta": batch_diagnostics,
            }
        )
    if maximum > float(max_abs_tolerance):
        raise ADV3B02RuntimeParityError(
            f"runtime/checkpoint parity exceeds tolerance: {diagnostics}"
        )
    parity_vector = {
        "schema": "cvs.phase1.runtime_checkpoint_parity_vector_audit.v1",
        "generator": "torch.Generator(cpu).manual_seed",
        "parity_seed": int(parity_seed),
        "batch_sizes": list(batch_sizes),
        "input_len": int(input_len),
        "resolved_device": str(runtime_device),
        "rows": vector_rows,
    }
    parity_vector_root = deployment_bundle.sha256_bytes(
        deployment_bundle.canonical_json_bytes(parity_vector)
    )
    structure = deployment_bundle._runtime_structure_from_bytes(runtime_bytes)[0]
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "checkpoint_lineage_sha256": checkpoint_sha,
        "runtime_sha256": runtime_sha,
        "parity_status": "PASS",
        "max_abs_output_delta": maximum,
        "parity_vector_root_sha256": parity_vector_root,
        **structure,
    }
    vector_audit = {
        **parity_vector,
        "parity_vector_root_sha256": parity_vector_root,
        "authority_scope": "development_non_authority_recomputation_audit",
    }
    if (
        _sha256_file(checkpoint) != checkpoint_sha
        or _sha256_file(runtime_file) != runtime_sha
    ):
        raise ADV3B02RuntimeParityError(
            "checkpoint/runtime changed during parity verification"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    vector_output.parent.mkdir(parents=True, exist_ok=True)
    with vector_output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(
            vector_audit,
            handle,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(receipt, handle, ensure_ascii=True, allow_nan=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "status": "PASS",
        "authority_scope": "development_numerical_parity_not_external_signature",
        "resolved_device": str(runtime_device),
        "batch_sizes": list(batch_sizes),
        "receipt_path": str(output),
        "receipt_sha256": _sha256_file(output),
        "vector_audit_path": str(vector_output),
        "vector_audit_sha256": _sha256_file(vector_output),
        "checkpoint_sha256": checkpoint_sha,
        "runtime_sha256": runtime_sha,
        "max_abs_output_delta": maximum,
        "diagnostics": diagnostics,
        "parity_vector_root_sha256": receipt["parity_vector_root_sha256"],
        "parity_vector": parity_vector,
        "runtime_structure": structure,
        "checkpoint_load_audit": checkpoint_audit,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--receipt-out", type=Path, required=True)
    parser.add_argument("--vector-audit-out", type=Path, required=True)
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--parity-seed", type=int, default=20260720)
    parser.add_argument("--parity-rows", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-abs-tolerance", type=float, default=1.0e-5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(
        json.dumps(
            verify_runtime_checkpoint(
                checkpoint_path=args.checkpoint,
                runtime_path=args.runtime,
                receipt_out=args.receipt_out,
                vector_audit_out=args.vector_audit_out,
                input_len=args.input_len,
                parity_seed=args.parity_seed,
                parity_rows=args.parity_rows,
                device=args.device,
                max_abs_tolerance=args.max_abs_tolerance,
            ),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
