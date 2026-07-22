"""Verify numerical parity of a nonformal ADV3B02 dual-feature runtime.

The verifier compares ``z_id160``, ``z_dom160``, and TX logits at batch sizes
1, ``parity_rows``, and 256.  Its receipt is deliberately not accepted by the
existing v1 deployment-bundle authority and grants no Phase2 permission.
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

from cvsrffi import phase1_adv3b02_deployment_bundle as deployment_bundle  # noqa: E402
from cvsrffi.checkpoint_loading import build_exact_ssdg_model_from_checkpoint  # noqa: E402
from cvsrffi.phase2_candidate_capsule import (  # noqa: E402
    BASE_CHECKPOINT_SHA256,
    EFFECTIVE8_TARGET_MODULES,
)
from paper_reproduction.scripts.benchmark_cvs_adaptive_rxlight_tta import (  # noqa: E402
    apply_fp16_lora_state,
)
from paper_reproduction.scripts.train_export_cvs_support_lora_adapter import (  # noqa: E402
    merge_feat_joint_lora,
)
from scripts.export_adv3b02_dual_feature_torchscript import (  # noqa: E402
    ADV3B02DualFeatureRuntime,
    EXPORT_SCHEMA,
    FEATURE_DIM,
    FRESH_PARITY_BATCH_SIZES,
    FORBIDDEN_RUNTIME_TOKENS,
    MAX_ABS_TOLERANCE,
    RUNTIME_BATCH_CAPACITY,
    RUNTIME_COMPONENT_ALLOWLIST,
    RUNTIME_OUTPUT_SCHEMA,
    _recheck_execution_contract,
    _seal_graph_executor_optimize_false,
    _validate_execution_contract,
)


RECEIPT_SCHEMA = "cvs.phase1.adv3b02_dual_runtime_checkpoint_parity_receipt.v2"
VECTOR_SCHEMA = "cvs.phase1.adv3b02_dual_runtime_parity_vector_audit.v2"
RUNTIME_ROLES = ("base", "candidate")


class ADV3B02DualRuntimeParityError(ValueError):
    """Raised when checkpoint/runtime identity or any dual output drifts."""


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_regular_bytes(path: Path, name: str) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise ADV3B02DualRuntimeParityError(f"{name} must be a regular file")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ADV3B02DualRuntimeParityError(f"failed to read {name}") from exc


def _load_checkpoint_bytes(value: bytes) -> Any:
    try:
        return torch.load(io.BytesIO(value), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(io.BytesIO(value), map_location="cpu")


def _load_tensor_state_bytes(value: bytes) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(io.BytesIO(value), map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(io.BytesIO(value), map_location="cpu")
    if not isinstance(state, dict) or not all(
        torch.is_tensor(item) for item in state.values()
    ):
        raise ADV3B02DualRuntimeParityError(
            "adapter state must be a tensor dictionary"
        )
    return {str(key): tensor for key, tensor in state.items()}


def _parse_export_receipt(value: bytes) -> dict[str, Any]:
    def _reject_constant(token: str) -> None:
        raise ValueError(f"non-finite JSON constant: {token}")

    try:
        payload = json.loads(
            value.decode("utf-8"),
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ADV3B02DualRuntimeParityError(
            "export receipt must be strict UTF-8 JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise ADV3B02DualRuntimeParityError("export receipt must be a JSON object")
    return payload


def _validate_export_binding(
    payload: dict[str, Any],
    *,
    runtime_role: str,
    checkpoint_sha: str,
    adapter_sha: str,
    runtime_sha: str,
    input_len: int,
    device: torch.device,
) -> tuple[int, dict[str, Any]]:
    runtime_key = f"{runtime_role}_runtime_sha256"
    dimensions = payload.get("feature_dimensions")
    try:
        execution_contract = _validate_execution_contract(payload.get("execution_contract"), device=device)
    except (TypeError, ValueError) as exc:
        raise ADV3B02DualRuntimeParityError("export execution contract drift") from exc
    if (
        payload.get("schema") != EXPORT_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("runtime_output_schema") != RUNTIME_OUTPUT_SCHEMA
        or payload.get("feature_keys")
        != {"z_id": "feat_joint", "z_dom": "feat_imp"}
        or payload.get("checkpoint_sha256") != checkpoint_sha
        or payload.get("adapter_state_sha256") != adapter_sha
        or payload.get(runtime_key) != runtime_sha
        or payload.get("expected_input_len") != int(input_len)
        or payload.get("runtime_batch_capacity") != RUNTIME_BATCH_CAPACITY
        or payload.get("formal_phase2_eligible") is not False
        or payload.get("bundle_created") is not False
        or payload.get("runtime_component_allowlist")
        != list(RUNTIME_COMPONENT_ALLOWLIST)
        or payload.get("forbidden_runtime_tokens")
        != list(FORBIDDEN_RUNTIME_TOKENS)
        or payload.get("effective8_target_modules")
        != list(EFFECTIVE8_TARGET_MODULES)
        or payload.get("execution_contract_sha256") != execution_contract["contract_sha256"]
        or payload.get("max_abs_tolerance") != MAX_ABS_TOLERANCE
        or not isinstance(dimensions, dict)
        or dimensions.get("z_id") != FEATURE_DIM
        or dimensions.get("z_dom") != FEATURE_DIM
        or not isinstance(dimensions.get("tx_logits"), int)
        or isinstance(dimensions.get("tx_logits"), bool)
        or int(dimensions["tx_logits"]) < 2
    ):
        raise ADV3B02DualRuntimeParityError(
            "export receipt/runtime/checkpoint/adapter/allowlist binding drift"
        )
    return int(dimensions["tx_logits"]), execution_contract


def _audit_runtime_components(runtime: torch.jit.ScriptModule) -> dict[str, Any]:
    module_names = tuple(name for name, _ in runtime.named_modules())
    parameter_names = tuple(name for name, _ in runtime.named_parameters())
    buffer_names = tuple(name for name, _ in runtime.named_buffers())
    all_names = module_names + parameter_names + buffer_names
    forbidden = tuple(
        name
        for name in all_names
        if any(token in name.lower() for token in FORBIDDEN_RUNTIME_TOKENS)
    )
    if forbidden:
        raise ADV3B02DualRuntimeParityError(
            f"forbidden runtime component/state detected: {forbidden}"
        )

    required = {
        "runtime.id_backbone": False,
        "runtime.dom_backbone": False,
        "runtime.dom_enhancer": False,
    }
    unexpected = []
    for name in module_names:
        if name in ("", "runtime"):
            continue
        matched = False
        for prefix in RUNTIME_COMPONENT_ALLOWLIST:
            if name == prefix or name.startswith(prefix + "."):
                required[prefix] = True
                matched = True
                break
        if not matched:
            unexpected.append(name)
    for name in parameter_names + buffer_names:
        if not any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in RUNTIME_COMPONENT_ALLOWLIST
        ):
            unexpected.append(name)
    if unexpected or not all(required.values()):
        raise ADV3B02DualRuntimeParityError(
            f"runtime component allowlist drift: unexpected={unexpected}, required={required}"
        )
    return {
        "module_name_root_sha256": hashlib.sha256(
            "\n".join(module_names).encode("utf-8")
        ).hexdigest(),
        "parameter_name_root_sha256": hashlib.sha256(
            "\n".join(parameter_names).encode("utf-8")
        ).hexdigest(),
        "buffer_name_root_sha256": hashlib.sha256(
            "\n".join(buffer_names).encode("utf-8")
        ).hexdigest(),
        "module_count": len(module_names),
        "parameter_tensor_count": len(parameter_names),
        "buffer_tensor_count": len(buffer_names),
        "forbidden_runtime_components_absent": True,
        "runtime_component_allowlist": list(RUNTIME_COMPONENT_ALLOWLIST),
    }


def _array_receipt(value: torch.Tensor) -> dict[str, Any]:
    array = np.ascontiguousarray(value.detach().float().cpu().numpy())
    return {
        "shape": [int(item) for item in array.shape],
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(array.view(np.uint8)).hexdigest(),
    }


def _runtime_outputs(
    value: Any, *, rows: int, expected_tx_classes: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ADV3B02DualRuntimeParityError(
            "runtime output must be (z_id160,z_dom160,tx_logits)"
        )
    z_id, z_dom, logits = value
    for name, tensor, width in (
        ("z_id", z_id, FEATURE_DIM),
        ("z_dom", z_dom, FEATURE_DIM),
        ("tx_logits", logits, None),
    ):
        if (
            not torch.is_tensor(tensor)
            or tensor.dtype != torch.float32
            or tensor.ndim != 2
            or int(tensor.shape[0]) != int(rows)
            or (width is not None and int(tensor.shape[1]) != int(width))
            or (
                width is None
                and int(tensor.shape[1]) != int(expected_tx_classes)
            )
            or not bool(torch.isfinite(tensor).all().item())
        ):
            raise ADV3B02DualRuntimeParityError(
                f"runtime {name} shape/dtype/finite contract drift"
            )
    return (
        z_id.detach().float().cpu(),
        z_dom.detach().float().cpu(),
        logits.detach().float().cpu(),
    )


def _max_abs(left: torch.Tensor, right: torch.Tensor, field: str) -> float:
    if tuple(left.shape) != tuple(right.shape):
        raise ADV3B02DualRuntimeParityError(f"parity shape mismatch: {field}")
    value = float(torch.max(torch.abs(left - right)).item())
    if not np.isfinite(value):
        raise ADV3B02DualRuntimeParityError(f"parity is non-finite: {field}")
    return value


def _resolve_device(requested: str) -> torch.device:
    value = str(requested).strip().lower()
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        value = "cuda:0"
    if not value.startswith("cuda:"):
        raise ADV3B02DualRuntimeParityError("device must be cpu or cuda:<index>")
    try:
        index = int(value.split(":", 1)[1])
    except (TypeError, ValueError) as exc:
        raise ADV3B02DualRuntimeParityError("CUDA device index is invalid") from exc
    if (
        index < 0
        or not torch.cuda.is_available()
        or index >= int(torch.cuda.device_count())
    ):
        raise ADV3B02DualRuntimeParityError(
            "requested CUDA device is unavailable; CPU fallback is forbidden"
        )
    return torch.device(f"cuda:{index}")


def _write_json_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


@torch.no_grad()
def verify_dual_runtime_checkpoint(
    *,
    checkpoint_path: str | Path,
    adapter_state_path: str | Path,
    runtime_path: str | Path,
    export_receipt_path: str | Path,
    expected_export_receipt_sha256: str,
    runtime_role: str,
    receipt_out: str | Path,
    vector_audit_out: str | Path,
    input_len: int,
    parity_seed: int,
    parity_rows: int,
    device: str,
    max_abs_tolerance: float = 1.0e-5,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path).resolve()
    adapter_file = Path(adapter_state_path).resolve()
    runtime_file = Path(runtime_path).resolve()
    export_receipt_file = Path(export_receipt_path).resolve()
    receipt_path = Path(receipt_out).resolve()
    vector_path = Path(vector_audit_out).resolve()
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise ADV3B02DualRuntimeParityError("checkpoint must be a regular file")
    if runtime_role not in RUNTIME_ROLES:
        raise ADV3B02DualRuntimeParityError("runtime_role must be base or candidate")
    if isinstance(input_len, bool) or int(input_len) <= 0:
        raise ADV3B02DualRuntimeParityError("input_len must be positive")
    if isinstance(parity_rows, bool) or int(parity_rows) != 8:
        raise ADV3B02DualRuntimeParityError(
            "parity_rows must be exactly 8 for the frozen fresh-batch audit"
        )
    if not 0.0 < float(max_abs_tolerance) <= MAX_ABS_TOLERANCE:
        raise ADV3B02DualRuntimeParityError("tolerance must be in (0,1e-5]")
    if (
        receipt_path == vector_path
        or receipt_path.exists()
        or vector_path.exists()
    ):
        raise ADV3B02DualRuntimeParityError(
            "refusing to overwrite or alias parity receipt/vector audit"
        )

    checkpoint_bytes = _read_regular_bytes(checkpoint, "checkpoint")
    adapter_bytes = _read_regular_bytes(adapter_file, "adapter state")
    runtime_bytes = _read_regular_bytes(runtime_file, "runtime")
    export_receipt_bytes = _read_regular_bytes(
        export_receipt_file, "export receipt"
    )
    checkpoint_sha = hashlib.sha256(checkpoint_bytes).hexdigest()
    adapter_sha = hashlib.sha256(adapter_bytes).hexdigest()
    runtime_sha = hashlib.sha256(runtime_bytes).hexdigest()
    export_receipt_sha = hashlib.sha256(export_receipt_bytes).hexdigest()
    if export_receipt_sha != str(expected_export_receipt_sha256):
        raise ADV3B02DualRuntimeParityError("export receipt SHA256 binding drift")
    if checkpoint_sha != BASE_CHECKPOINT_SHA256:
        raise ADV3B02DualRuntimeParityError("checkpoint is not strict ADV3B02")
    export_receipt = _parse_export_receipt(export_receipt_bytes)
    runtime_device = _resolve_device(device)
    live_execution_contract = _seal_graph_executor_optimize_false(runtime_device)
    expected_tx_classes, export_execution_contract = _validate_export_binding(
        export_receipt,
        runtime_role=runtime_role,
        checkpoint_sha=checkpoint_sha,
        adapter_sha=adapter_sha,
        runtime_sha=runtime_sha,
        input_len=int(input_len),
        device=runtime_device,
    )
    if export_execution_contract != live_execution_contract:
        raise ADV3B02DualRuntimeParityError("export execution contract does not close to live contract")

    checkpoint_value = _load_checkpoint_bytes(checkpoint_bytes)
    model, checkpoint_audit = build_exact_ssdg_model_from_checkpoint(
        checkpoint_value,
        input_len=int(input_len),
        device=runtime_device,
    )
    adapter_audit: dict[str, Any] | None = None
    merge_audit: dict[str, Any] | None = None
    if runtime_role == "candidate":
        adapter_state = _load_tensor_state_bytes(adapter_bytes)
        adapter_audit = apply_fp16_lora_state(
            model,
            adapter_state,
            scope="effective_feature",
            rank=16,
            alpha=16.0,
        )
        if (
            adapter_audit.get("element_count") != 44_048
            or adapter_audit.get("tensor_bytes_fp16") != 88_096
            or tuple(adapter_audit.get("target_modules", ()))
            != EFFECTIVE8_TARGET_MODULES
        ):
            raise ADV3B02DualRuntimeParityError(
                "effective8 delta resource/target contract drift"
            )
        merge_audit = merge_feat_joint_lora(model)
        if (
            merge_audit.get("merged_module_count") != 8
            or merge_audit.get("remaining_lora_wrappers") != []
            or merge_audit.get("algebraic_probe_parity_pass") is not True
        ):
            raise ADV3B02DualRuntimeParityError("effective8 merge audit failed")
    eager = ADV3B02DualFeatureRuntime(
        model,
        expected_input_len=int(input_len),
        expected_tx_classes=expected_tx_classes,
        runtime_batch_size=RUNTIME_BATCH_CAPACITY,
    ).to(runtime_device).eval()
    try:
        _recheck_execution_contract(live_execution_contract, device=runtime_device)
        scripted = torch.jit.load(
            io.BytesIO(runtime_bytes), map_location=runtime_device
        ).eval()
    except Exception as exc:
        raise ADV3B02DualRuntimeParityError(
            "runtime is not loadable TorchScript"
        ) from exc
    component_audit = _audit_runtime_components(scripted)

    batch_sizes = FRESH_PARITY_BATCH_SIZES
    diagnostics: dict[str, dict[str, dict[str, float]]] = {}
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
        eager_values = _runtime_outputs(
            eager(probes),
            rows=int(batch_size),
            expected_tx_classes=expected_tx_classes,
        )
        runtime_calls: list[dict[str, Any]] = []
        batch_diagnostics: dict[str, dict[str, float]] = {}
        for call_index in (1, 2, 3):
            runtime_values = _runtime_outputs(scripted(probes), rows=int(batch_size), expected_tx_classes=expected_tx_classes)
            call_diagnostics = {
                "z_id": _max_abs(eager_values[0], runtime_values[0], f"z_id.batch{batch_size}.call{call_index}"),
                "z_dom": _max_abs(eager_values[1], runtime_values[1], f"z_dom.batch{batch_size}.call{call_index}"),
                "tx_logits": _max_abs(eager_values[2], runtime_values[2], f"tx_logits.batch{batch_size}.call{call_index}"),
            }
            batch_diagnostics[str(call_index)] = call_diagnostics
            maximum = max(maximum, *call_diagnostics.values())
            runtime_calls.append({"call_index": call_index, "runtime_z_id": _array_receipt(runtime_values[0]), "runtime_z_dom": _array_receipt(runtime_values[1]), "runtime_tx_logits": _array_receipt(runtime_values[2]), "max_abs_delta": call_diagnostics})
        diagnostics[str(batch_size)] = batch_diagnostics
        vector_rows.append(
            {
                "batch_size": int(batch_size),
                "generator_seed": int(parity_seed) + int(batch_size),
                "input": _array_receipt(probes_cpu),
                "checkpoint_z_id": _array_receipt(eager_values[0]),
                "checkpoint_z_dom": _array_receipt(eager_values[1]),
                "checkpoint_tx_logits": _array_receipt(eager_values[2]),
                "runtime_calls": runtime_calls,
                "max_abs_delta_by_runtime_call": batch_diagnostics,
            }
        )
    if maximum > float(max_abs_tolerance):
        raise ADV3B02DualRuntimeParityError(
            f"runtime/checkpoint parity exceeds tolerance: {diagnostics}"
        )

    parity_vector = {
        "schema": VECTOR_SCHEMA,
        "runtime_output_schema": RUNTIME_OUTPUT_SCHEMA,
        "generator": "torch.Generator(cpu).manual_seed",
        "parity_seed": int(parity_seed),
        "batch_sizes": list(batch_sizes),
        "input_len": int(input_len),
        "expected_tx_classes": expected_tx_classes,
        "runtime_batch_capacity": RUNTIME_BATCH_CAPACITY,
        "resolved_device": str(runtime_device),
        "runtime_role": runtime_role,
        "execution_contract": live_execution_contract,
        "execution_contract_sha256": live_execution_contract["contract_sha256"],
        "max_abs_tolerance": MAX_ABS_TOLERANCE,
        "runtime_calls_per_batch": 3,
        "rows": vector_rows,
    }
    parity_vector_root = deployment_bundle.sha256_bytes(
        deployment_bundle.canonical_json_bytes(parity_vector)
    )
    structure = deployment_bundle._runtime_structure_from_bytes(runtime_bytes)[0]
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "PASS",
        "artifact_stage": "phase1_offline_dual_runtime_parity_without_bundle",
        "runtime_output_schema": RUNTIME_OUTPUT_SCHEMA,
        "checkpoint_lineage_sha256": checkpoint_sha,
        "adapter_state_sha256": adapter_sha,
        "runtime_sha256": runtime_sha,
        "export_receipt_sha256": export_receipt_sha,
        "runtime_role": runtime_role,
        "execution_contract": live_execution_contract,
        "execution_contract_sha256": live_execution_contract["contract_sha256"],
        "max_abs_tolerance": MAX_ABS_TOLERANCE,
        "expected_input_len": int(input_len),
        "expected_tx_classes": expected_tx_classes,
        "runtime_batch_capacity": RUNTIME_BATCH_CAPACITY,
        "max_abs_output_delta": maximum,
        "parity_vector_root_sha256": parity_vector_root,
        "runtime_invocations_per_parity_batch": 3,
        "runtime_calls_per_batch": 3,
        "runtime_component_audit": component_audit,
        "formal_phase2_eligible": False,
        "bundle_created": False,
        "bundle_id": None,
        **structure,
    }
    vector_audit = {
        **parity_vector,
        "parity_vector_root_sha256": parity_vector_root,
        "authority_scope": "development_dual_runtime_parity_not_phase2_authority",
    }
    _write_json_new(vector_path, vector_audit)
    _write_json_new(receipt_path, receipt)
    return {
        "status": "PASS",
        "authority_scope": "development_dual_runtime_parity_not_phase2_authority",
        "formal_phase2_eligible": False,
        "bundle_created": False,
        "resolved_device": str(runtime_device),
        "batch_sizes": list(batch_sizes),
        "receipt_path": str(receipt_path),
        "receipt_sha256": _sha256_file(receipt_path),
        "vector_audit_path": str(vector_path),
        "vector_audit_sha256": _sha256_file(vector_path),
        "checkpoint_sha256": checkpoint_sha,
        "adapter_state_sha256": adapter_sha,
        "runtime_sha256": runtime_sha,
        "export_receipt_sha256": export_receipt_sha,
        "max_abs_output_delta": maximum,
        "diagnostics": diagnostics,
        "parity_vector_root_sha256": parity_vector_root,
        "runtime_structure": structure,
        "checkpoint_load_audit": checkpoint_audit,
        "effective8_delta_audit": adapter_audit,
        "effective8_merge_audit": merge_audit,
        "runtime_component_audit": component_audit,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-state", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--export-receipt", type=Path, required=True)
    parser.add_argument("--expected-export-receipt-sha256", required=True)
    parser.add_argument("--runtime-role", choices=RUNTIME_ROLES, required=True)
    parser.add_argument("--receipt-out", type=Path, required=True)
    parser.add_argument("--vector-audit-out", type=Path, required=True)
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--parity-seed", type=int, default=20260721)
    parser.add_argument("--parity-rows", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-abs-tolerance", type=float, default=1.0e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            verify_dual_runtime_checkpoint(
                checkpoint_path=args.checkpoint,
                adapter_state_path=args.adapter_state,
                runtime_path=args.runtime,
                export_receipt_path=args.export_receipt,
                expected_export_receipt_sha256=args.expected_export_receipt_sha256,
                runtime_role=args.runtime_role,
                receipt_out=args.receipt_out,
                vector_audit_out=args.vector_audit_out,
                input_len=args.input_len,
                parity_seed=args.parity_seed,
                parity_rows=args.parity_rows,
                device=args.device,
                max_abs_tolerance=args.max_abs_tolerance,
            ),
            ensure_ascii=True,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
