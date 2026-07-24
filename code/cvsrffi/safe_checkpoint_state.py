"""Build and verify a pure-tensor Phase1 checkpoint for old PyTorch runtimes."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch


SCHEMA = "cvs.safe-phase1-model-ema.v1"
RECEIPT_SCHEMA = SCHEMA + ".receipt.v1"


class SafeCheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be proven safe and lineage-bound."""


def _sha_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _require_sha(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise SafeCheckpointError(f"{name} must be lowercase SHA256")
    return value


def _regular_file(path: str | Path, name: str) -> Path:
    value = Path(path)
    if value.is_symlink() or not value.is_file():
        raise SafeCheckpointError(f"{name} must be a regular file")
    return value.resolve()


def _state_mapping(value: Any, name: str) -> OrderedDict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise SafeCheckpointError(f"{name} must be a non-empty mapping")
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key in sorted(value):
        tensor = value[key]
        if not isinstance(key, str) or not key:
            raise SafeCheckpointError(f"{name} keys must be non-empty strings")
        if not isinstance(tensor, torch.Tensor):
            raise SafeCheckpointError(f"{name}.{key} must be a tensor")
        if tensor.layout != torch.strided:
            raise SafeCheckpointError(f"{name}.{key} must be strided")
        tensor = tensor.detach().cpu().contiguous()
        if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(
            tensor
        ).all():
            raise SafeCheckpointError(f"{name}.{key} is non-finite")
        result[key] = tensor
    return result


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def _state_digest(
    model: Mapping[str, torch.Tensor], ema_model: Mapping[str, torch.Tensor]
) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    tensor_count = 0
    parameter_count = 0
    for state_name, state in (("model", model), ("ema_model", ema_model)):
        for key in sorted(state):
            tensor = state[key]
            descriptor = {
                "state": state_name,
                "key": key,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            }
            payload = _canonical(descriptor)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
            raw = _tensor_bytes(tensor)
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
            tensor_count += 1
            parameter_count += int(tensor.numel())
    return digest.hexdigest(), tensor_count, parameter_count


def _load_source_weights_only(path: Path) -> Mapping[str, Any]:
    safe_globals = getattr(torch.serialization, "safe_globals", None)
    if safe_globals is None:
        raise SafeCheckpointError(
            "source export requires torch.serialization.safe_globals"
        )
    from baseline_origin_sat_view import SatViewStage

    with safe_globals([SatViewStage]):
        value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping):
        raise SafeCheckpointError("source checkpoint must be a mapping")
    return value


def export_safe_checkpoint(
    *,
    source_path: str | Path,
    source_sha256: str,
    output_path: str | Path,
    receipt_path: str | Path,
) -> dict[str, Any]:
    source = _regular_file(source_path, "source checkpoint")
    expected_source = _require_sha(source_sha256, "source checkpoint SHA256")
    if _sha_file(source) != expected_source:
        raise SafeCheckpointError("source checkpoint SHA256 drift")
    output = Path(output_path)
    receipt_output = Path(receipt_path)
    if (
        output.exists()
        or output.is_symlink()
        or receipt_output.exists()
        or receipt_output.is_symlink()
    ):
        raise FileExistsError("refusing to overwrite safe checkpoint output")

    loaded = _load_source_weights_only(source)
    model = _state_mapping(loaded.get("model"), "model")
    ema_model = _state_mapping(loaded.get("ema_model"), "ema_model")
    if tuple(model) != tuple(ema_model):
        raise SafeCheckpointError("model/ema_model key drift")
    state_sha256, tensor_count, parameter_count = _state_digest(model, ema_model)
    payload = {
        "schema": SCHEMA,
        "source_checkpoint_sha256": expected_source,
        "state_sha256": state_sha256,
        "model": model,
        "ema_model": ema_model,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    artifact_sha256 = _sha_file(output)
    verified, audit = load_safe_checkpoint(
        checkpoint_path=output,
        checkpoint_sha256=artifact_sha256,
        expected_source_sha256=expected_source,
    )
    del verified
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "source_checkpoint_sha256": expected_source,
        "safe_checkpoint_sha256": artifact_sha256,
        "state_sha256": state_sha256,
        "tensor_count": tensor_count,
        "parameter_count_model_plus_ema": parameter_count,
        "model_key_count": len(model),
        "torch_export_version": str(torch.__version__),
        "weights_only_roundtrip": True,
        "roundtrip_audit": audit,
    }
    receipt["receipt_sha256"] = hashlib.sha256(_canonical(receipt)).hexdigest()
    receipt_output.parent.mkdir(parents=True, exist_ok=True)
    with receipt_output.open("xb") as handle:
        handle.write(_canonical(receipt) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return receipt


def load_safe_checkpoint(
    *,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    expected_source_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = _regular_file(checkpoint_path, "safe checkpoint")
    expected_checkpoint = _require_sha(
        checkpoint_sha256, "safe checkpoint SHA256"
    )
    expected_source = _require_sha(
        expected_source_sha256, "source checkpoint SHA256"
    )
    if _sha_file(checkpoint) != expected_checkpoint:
        raise SafeCheckpointError("safe checkpoint SHA256 drift")
    try:
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise SafeCheckpointError("runtime lacks weights_only checkpoint load") from exc
    if not isinstance(loaded, Mapping) or set(loaded) != {
        "schema",
        "source_checkpoint_sha256",
        "state_sha256",
        "model",
        "ema_model",
    }:
        raise SafeCheckpointError("safe checkpoint schema drift")
    if (
        loaded["schema"] != SCHEMA
        or loaded["source_checkpoint_sha256"] != expected_source
    ):
        raise SafeCheckpointError("safe checkpoint source lineage drift")
    model = _state_mapping(loaded["model"], "model")
    ema_model = _state_mapping(loaded["ema_model"], "ema_model")
    if tuple(model) != tuple(ema_model):
        raise SafeCheckpointError("safe checkpoint model/ema key drift")
    state_sha256, tensor_count, parameter_count = _state_digest(model, ema_model)
    if loaded["state_sha256"] != state_sha256:
        raise SafeCheckpointError("safe checkpoint tensor digest drift")
    payload = {
        "schema": SCHEMA,
        "source_checkpoint_sha256": expected_source,
        "state_sha256": state_sha256,
        "model": model,
        "ema_model": ema_model,
    }
    audit = {
        "safe_checkpoint_sha256": expected_checkpoint,
        "source_checkpoint_sha256": expected_source,
        "state_sha256": state_sha256,
        "tensor_count": tensor_count,
        "parameter_count_model_plus_ema": parameter_count,
        "model_key_count": len(model),
        "torch_load_version": str(torch.__version__),
        "weights_only": True,
    }
    return payload, audit


def verify_safe_checkpoint_receipt(
    *,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    receipt_path: str | Path,
    receipt_sha256: str,
    expected_source_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt_file = _regular_file(receipt_path, "safe checkpoint receipt")
    expected_receipt = _require_sha(
        receipt_sha256, "safe checkpoint receipt SHA256"
    )
    if _sha_file(receipt_file) != expected_receipt:
        raise SafeCheckpointError("safe checkpoint receipt file SHA256 drift")
    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise SafeCheckpointError("safe checkpoint receipt must be an object")
    internal_receipt = receipt.pop("receipt_sha256", None)
    if internal_receipt != hashlib.sha256(_canonical(receipt)).hexdigest():
        raise SafeCheckpointError("safe checkpoint receipt content drift")
    receipt["receipt_sha256"] = internal_receipt
    payload, audit = load_safe_checkpoint(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        expected_source_sha256=expected_source_sha256,
    )
    expected = {
        "schema": RECEIPT_SCHEMA,
        "source_checkpoint_sha256": audit["source_checkpoint_sha256"],
        "safe_checkpoint_sha256": audit["safe_checkpoint_sha256"],
        "state_sha256": audit["state_sha256"],
        "tensor_count": audit["tensor_count"],
        "parameter_count_model_plus_ema": audit[
            "parameter_count_model_plus_ema"
        ],
        "model_key_count": audit["model_key_count"],
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise SafeCheckpointError("safe checkpoint receipt lineage drift")
    audit["safe_checkpoint_receipt_sha256"] = expected_receipt
    audit["receipt_schema"] = receipt["schema"]
    return payload, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    export = sub.add_parser("export")
    for name in ("source", "source-sha256", "output", "receipt"):
        export.add_argument("--" + name, required=True)
    verify = sub.add_parser("verify")
    for name in (
        "checkpoint",
        "checkpoint-sha256",
        "receipt",
        "receipt-sha256",
        "source-sha256",
    ):
        verify.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if args.cmd == "export":
        result = export_safe_checkpoint(
            source_path=args.source,
            source_sha256=args.source_sha256,
            output_path=args.output,
            receipt_path=args.receipt,
        )
    else:
        _, result = verify_safe_checkpoint_receipt(
            checkpoint_path=args.checkpoint,
            checkpoint_sha256=args.checkpoint_sha256,
            receipt_path=args.receipt,
            receipt_sha256=args.receipt_sha256,
            expected_source_sha256=args.source_sha256,
        )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
