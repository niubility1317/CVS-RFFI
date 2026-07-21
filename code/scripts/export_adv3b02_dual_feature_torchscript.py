"""Export nonformal ADV3B02 dual-feature TorchScript runtimes.

The artifact exposes ``(z_id160, z_dom160, tx_logits)`` from one fixed received
IQ call.  It creates neither a deployment bundle nor Phase2 authority; a
separate bundle/version review is required before any target runtime can use
the result.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
for value in (str(REPO_ROOT), str(CODE_ROOT)):
    while value in sys.path:
        sys.path.remove(value)
for value in (str(REPO_ROOT), str(CODE_ROOT)):
    sys.path.insert(0, value)

from cvsrffi.checkpoint_loading import build_exact_ssdg_model_from_checkpoint  # noqa: E402
from cvsrffi.dual_feature_forward import (  # noqa: E402
    DOM_FEATURE_KEY,
    ID_FEATURE_KEY,
    dual_feature_forward,
    dual_feature_components_forward,
)
from cvsrffi.phase2_candidate_capsule import (  # noqa: E402
    BASE_CHECKPOINT_SHA256,
    EFFECTIVE8_TARGET_MODULES,
    sha256_file,
)
from paper_reproduction.scripts.benchmark_cvs_adaptive_rxlight_tta import (  # noqa: E402
    apply_fp16_lora_state,
)
from paper_reproduction.scripts.train_export_cvs_support_lora_adapter import (  # noqa: E402
    merge_feat_joint_lora,
)


EXPORT_SCHEMA = "cvs.phase1.adv3b02_dual_feature_torchscript_export.v1"
RUNTIME_OUTPUT_SCHEMA = "adv3b02.dual_feature_runtime.zid160_zdom160_txlogits.v1"
FEATURE_DIM = 160
RUNTIME_BATCH_CAPACITY = 256
FORBIDDEN_RUNTIME_TOKENS = ("dom_head", "adv_head", "tx_adv_head")
RUNTIME_COMPONENT_ALLOWLIST = (
    "runtime.id_backbone",
    "runtime.dom_backbone",
    "runtime.dom_enhancer",
)


class ADV3B02DualFeatureRuntime(nn.Module):
    """Fixed-capacity wrapper for one same-IQ dual-feature forward."""

    def __init__(
        self,
        model: nn.Module,
        *,
        expected_input_len: int,
        expected_tx_classes: int,
        runtime_batch_size: int = RUNTIME_BATCH_CAPACITY,
    ) -> None:
        super().__init__()
        if (
            isinstance(runtime_batch_size, bool)
            or int(runtime_batch_size) != RUNTIME_BATCH_CAPACITY
        ):
            raise ValueError("runtime_batch_size must be exactly 256")
        if isinstance(expected_input_len, bool) or int(expected_input_len) < 1:
            raise ValueError("expected_input_len must be positive")
        if isinstance(expected_tx_classes, bool) or int(expected_tx_classes) < 2:
            raise ValueError("expected_tx_classes must be at least two")
        if str(getattr(model, "id_feature_key", "")) != ID_FEATURE_KEY:
            raise ValueError("ADV3B02 z_id feature key must be feat_joint")
        if str(getattr(model, "dom_feature_key", "")) != DOM_FEATURE_KEY:
            raise ValueError("ADV3B02 z_dom feature key must be feat_imp")
        for name in ("id_backbone", "dom_backbone", "dom_enhancer"):
            if not isinstance(getattr(model, name, None), nn.Module):
                raise ValueError(f"ADV3B02 dual feature module missing: {name}")
        # Intentionally retain only the three allowed components.  dom_head,
        # adv_head, tx_adv_head, and the full model object cannot be serialized.
        self.id_backbone = model.id_backbone.eval()
        self.dom_backbone = model.dom_backbone.eval()
        self.dom_enhancer = model.dom_enhancer.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.runtime_batch_size = int(runtime_batch_size)
        self.expected_input_len = int(expected_input_len)
        self.expected_tx_classes = int(expected_tx_classes)

    def forward(
        self, rows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if rows.dtype != torch.float32:
            raise RuntimeError("dual runtime input must use float32")
        if rows.dim() != 3 or rows.size(0) < 1 or rows.size(1) != 2:
            raise RuntimeError("dual runtime input must have shape [N,2,T]")
        if rows.size(2) != self.expected_input_len:
            raise RuntimeError("dual runtime input length drift")
        count = rows.size(0)
        if count > self.runtime_batch_size:
            raise RuntimeError("dual runtime batch exceeds fixed capacity")
        padded = rows.new_zeros(
            (self.runtime_batch_size, rows.size(1), rows.size(2))
        )
        padded[:count].copy_(rows)
        z_id, z_dom, logits = dual_feature_components_forward(
            self.id_backbone,
            self.dom_backbone,
            self.dom_enhancer,
            padded,
        )
        if logits.size(1) != self.expected_tx_classes:
            raise RuntimeError("dual runtime TX class width drift")
        return z_id[:count], z_dom[:count], logits[:count]


class _InputValidatedRuntime(nn.Module):
    """Keep input guards dynamic around the traced model graph."""

    def __init__(
        self,
        runtime: nn.Module,
        *,
        expected_input_len: int,
        expected_tx_classes: int,
    ) -> None:
        super().__init__()
        self.runtime = runtime
        self.expected_input_len = int(expected_input_len)
        self.expected_tx_classes = int(expected_tx_classes)
        self.runtime_batch_capacity = int(RUNTIME_BATCH_CAPACITY)
        self.feature_dim = int(FEATURE_DIM)

    def forward(
        self, rows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if rows.dtype != torch.float32:
            raise RuntimeError("dual runtime input must use float32")
        if rows.dim() != 3 or rows.size(0) < 1 or rows.size(1) != 2:
            raise RuntimeError("dual runtime input must have shape [N,2,T]")
        if rows.size(2) != self.expected_input_len:
            raise RuntimeError("dual runtime input length drift")
        if rows.size(0) > self.runtime_batch_capacity:
            raise RuntimeError("dual runtime batch exceeds fixed capacity")
        if not torch.isfinite(rows).all():
            raise RuntimeError("dual runtime input must be finite")
        z_id, z_dom, logits = self.runtime(rows)
        if (
            z_id.dtype != torch.float32
            or z_id.dim() != 2
            or z_id.size(0) != rows.size(0)
            or z_id.size(1) != self.feature_dim
            or not torch.isfinite(z_id).all()
        ):
            raise RuntimeError("dual runtime z_id output contract drift")
        if (
            z_dom.dtype != torch.float32
            or z_dom.dim() != 2
            or z_dom.size(0) != rows.size(0)
            or z_dom.size(1) != self.feature_dim
            or not torch.isfinite(z_dom).all()
        ):
            raise RuntimeError("dual runtime z_dom output contract drift")
        if (
            logits.dtype != torch.float32
            or logits.dim() != 2
            or logits.size(0) != rows.size(0)
            or logits.size(1) != self.expected_tx_classes
            or not torch.isfinite(logits).all()
        ):
            raise RuntimeError("dual runtime TX logits output contract drift")
        return z_id, z_dom, logits


def _resolve_device(requested: str) -> torch.device:
    value = str(requested).strip().lower()
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        value = "cuda:0"
    if not value.startswith("cuda:"):
        raise ValueError("device must be cpu or cuda:<index>")
    try:
        index = int(value.split(":", 1)[1])
    except (TypeError, ValueError) as exc:
        raise ValueError("CUDA device index is invalid") from exc
    if (
        index < 0
        or not torch.cuda.is_available()
        or index >= int(torch.cuda.device_count())
    ):
        raise ValueError("requested CUDA device is unavailable; CPU fallback is forbidden")
    return torch.device(f"cuda:{index}")


def _read_regular_bytes(path: Path, name: str) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{name} must be a regular file")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ValueError(f"failed to read {name}") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
        raise TypeError("adapter state must be a tensor dictionary")
    return {str(key): tensor for key, tensor in state.items()}


def _write_json_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _state_bytes(module: nn.Module) -> int:
    return int(
        sum(
            int(tensor.numel()) * int(tensor.element_size())
            for _, tensor in tuple(module.named_parameters())
            + tuple(module.named_buffers())
        )
    )


def _validate_outputs(
    value: Any, *, rows: int, expected_tx_classes: int | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise RuntimeError("dual runtime must return z_id, z_dom, and tx_logits")
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
                and (
                    int(tensor.shape[1]) < 2
                    or (
                        expected_tx_classes is not None
                        and int(tensor.shape[1]) != int(expected_tx_classes)
                    )
                )
            )
            or not bool(torch.isfinite(tensor).all().item())
        ):
            raise RuntimeError(f"dual runtime {name} shape/dtype/finite drift")
    return z_id, z_dom, logits


@torch.no_grad()
def _run(
    wrapper: nn.Module,
    rows: torch.Tensor,
    *,
    expected_tx_classes: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    wrapper.eval()
    values = _validate_outputs(
        wrapper(rows),
        rows=int(rows.shape[0]),
        expected_tx_classes=expected_tx_classes,
    )
    return tuple(value.detach().float().cpu() for value in values)  # type: ignore[return-value]


def _max_abs(left: torch.Tensor, right: torch.Tensor, *, field: str) -> float:
    if tuple(left.shape) != tuple(right.shape):
        raise ValueError(f"parity tensor shape mismatch: {field}")
    value = float(torch.max(torch.abs(left - right)).item())
    if not torch.isfinite(torch.tensor(value)):
        raise ValueError(f"non-finite parity error: {field}")
    return value


def _trace_and_save(
    wrapper: nn.Module, example: torch.Tensor, output: Path
) -> torch.jit.ScriptModule:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite TorchScript runtime: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    wrapper.eval()
    traced = torch.jit.trace(wrapper, example, strict=False, check_trace=False)
    if not isinstance(wrapper, ADV3B02DualFeatureRuntime):
        raise TypeError("dual trace requires the exact sealed runtime wrapper")
    validated = torch.jit.script(
        _InputValidatedRuntime(
            traced,
            expected_input_len=wrapper.expected_input_len,
            expected_tx_classes=wrapper.expected_tx_classes,
        ).eval()
    )
    torch.jit.save(validated, output)
    if not output.is_file() or output.stat().st_size < 1:
        raise RuntimeError(f"TorchScript export is empty: {output}")
    return torch.jit.load(str(output), map_location=example.device).eval()


def _compare_triplet(
    left: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    right: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    prefix: str,
) -> dict[str, float]:
    return {
        f"{prefix}_z_id": _max_abs(left[0], right[0], field=f"{prefix}.z_id"),
        f"{prefix}_z_dom": _max_abs(left[1], right[1], field=f"{prefix}.z_dom"),
        f"{prefix}_tx_logits": _max_abs(
            left[2], right[2], field=f"{prefix}.tx_logits"
        ),
    }


def export(args: argparse.Namespace) -> dict[str, Any]:
    if (
        isinstance(args.runtime_batch_size, bool)
        or int(args.runtime_batch_size) != RUNTIME_BATCH_CAPACITY
    ):
        raise ValueError(
            "runtime_batch_size must be exactly 256 for the sealed dual runtime"
        )
    if isinstance(args.input_len, bool) or int(args.input_len) <= 0:
        raise ValueError("input_len must be a positive integer")
    checkpoint_path = Path(args.checkpoint)
    adapter_path = Path(args.adapter_state)
    outputs = (
        Path(args.base_runtime_out),
        Path(args.candidate_runtime_out),
        Path(args.export_receipt_out),
    )
    if len({path.resolve() for path in outputs}) != len(outputs):
        raise ValueError("dual export outputs must not alias")
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite dual export artifact")
    checkpoint_bytes = _read_regular_bytes(checkpoint_path, "checkpoint")
    adapter_bytes = _read_regular_bytes(adapter_path, "adapter state")
    checkpoint_sha = _sha256_bytes(checkpoint_bytes)
    adapter_sha = _sha256_bytes(adapter_bytes)
    if checkpoint_sha != BASE_CHECKPOINT_SHA256:
        raise ValueError("checkpoint is not the strict ADV3B02 base")
    if isinstance(args.parity_rows, bool) or not 2 <= int(args.parity_rows) <= 255:
        raise ValueError("parity_rows must be in [2,255]")
    if not 0.0 < float(args.max_abs_tolerance) <= 1.0e-4:
        raise ValueError("TorchScript parity tolerance must be in (0,1e-4]")

    device = _resolve_device(str(args.device))
    checkpoint = _load_checkpoint_bytes(checkpoint_bytes)
    base_model, base_audit = build_exact_ssdg_model_from_checkpoint(
        checkpoint, input_len=int(args.input_len), device=device
    )
    candidate_model, candidate_audit = build_exact_ssdg_model_from_checkpoint(
        checkpoint, input_len=int(args.input_len), device=device
    )
    if base_audit != candidate_audit:
        raise ValueError("base/candidate strict checkpoint reconstruction drift")
    base_model.to(device).eval()
    candidate_model.to(device).eval()
    adapter_state = _load_tensor_state_bytes(adapter_bytes)
    delta_audit = apply_fp16_lora_state(
        candidate_model,
        adapter_state,
        scope="effective_feature",
        rank=16,
        alpha=16.0,
    )
    if (
        delta_audit.get("element_count") != 44_048
        or delta_audit.get("tensor_bytes_fp16") != 88_096
        or tuple(delta_audit.get("target_modules", ())) != EFFECTIVE8_TARGET_MODULES
    ):
        raise ValueError("effective8 delta resource/target contract drift")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.parity_seed))
    probes = torch.randn(
        int(args.parity_rows),
        2,
        int(args.input_len),
        generator=generator,
        dtype=torch.float32,
    ).to(device)
    trace_example = probes[:2]
    base_contract = _validate_outputs(
        dual_feature_forward(base_model, probes[:1]), rows=1
    )
    candidate_contract = _validate_outputs(
        dual_feature_forward(candidate_model, probes[:1]), rows=1
    )
    tx_classes = int(base_contract[2].shape[1])
    if int(candidate_contract[2].shape[1]) != tx_classes:
        raise ValueError("base/candidate TX class width drift")
    base_wrapper = ADV3B02DualFeatureRuntime(
        base_model,
        expected_input_len=int(args.input_len),
        expected_tx_classes=tx_classes,
        runtime_batch_size=int(args.runtime_batch_size),
    ).to(device).eval()
    injected_wrapper = ADV3B02DualFeatureRuntime(
        candidate_model,
        expected_input_len=int(args.input_len),
        expected_tx_classes=tx_classes,
        runtime_batch_size=int(args.runtime_batch_size),
    ).to(device).eval()
    injected = _run(
        injected_wrapper, probes, expected_tx_classes=tx_classes
    )
    merge_audit = merge_feat_joint_lora(candidate_model)
    if (
        merge_audit.get("merged_module_count") != 8
        or merge_audit.get("remaining_lora_wrappers") != []
        or merge_audit.get("algebraic_probe_parity_pass") is not True
    ):
        raise ValueError("effective8 merge audit failed")
    merged_wrapper = ADV3B02DualFeatureRuntime(
        candidate_model,
        expected_input_len=int(args.input_len),
        expected_tx_classes=tx_classes,
        runtime_batch_size=int(args.runtime_batch_size),
    ).to(device).eval()
    merged = _run(merged_wrapper, probes, expected_tx_classes=tx_classes)

    base_script = _trace_and_save(base_wrapper, trace_example, outputs[0])
    candidate_script = _trace_and_save(merged_wrapper, trace_example, outputs[1])
    base_eager = _run(base_wrapper, probes, expected_tx_classes=tx_classes)
    base_loaded = _run(base_script, probes, expected_tx_classes=tx_classes)
    candidate_loaded = _run(
        candidate_script, probes, expected_tx_classes=tx_classes
    )
    diagnostics = {
        **_compare_triplet(injected, merged, prefix="injected_vs_merged"),
        **_compare_triplet(base_eager, base_loaded, prefix="base_eager_vs_runtime"),
        **_compare_triplet(merged, candidate_loaded, prefix="merged_vs_runtime"),
    }
    failed = {
        key: value
        for key, value in diagnostics.items()
        if value > float(args.max_abs_tolerance)
    }
    if failed:
        raise ValueError(f"ADV3B02 dual TorchScript parity failed: {failed}")
    receipt = {
        "schema": EXPORT_SCHEMA,
        "status": "PASS",
        "artifact_stage": "phase1_offline_dual_runtime_export_without_bundle",
        "runtime_output_schema": RUNTIME_OUTPUT_SCHEMA,
        "checkpoint_sha256": checkpoint_sha,
        "adapter_state_sha256": adapter_sha,
        "base_runtime_sha256": sha256_file(outputs[0]),
        "candidate_runtime_sha256": sha256_file(outputs[1]),
        "base_runtime_size_bytes": outputs[0].stat().st_size,
        "candidate_runtime_size_bytes": outputs[1].stat().st_size,
        "feature_keys": {"z_id": "feat_joint", "z_dom": "feat_imp"},
        "feature_dimensions": {
            "z_id": FEATURE_DIM,
            "z_dom": FEATURE_DIM,
            "tx_logits": tx_classes,
        },
        "expected_input_len": int(args.input_len),
        "runtime_batch_capacity": RUNTIME_BATCH_CAPACITY,
        "runtime_invocations_per_prediction": 1,
        "component_forward_counts_per_invocation": {
            "id_backbone": 1,
            "dom_backbone": 1,
            "dom_enhancer": 1,
        },
        "forbidden_runtime_tokens": list(FORBIDDEN_RUNTIME_TOKENS),
        "runtime_component_allowlist": list(RUNTIME_COMPONENT_ALLOWLIST),
        "effective8_target_modules": list(EFFECTIVE8_TARGET_MODULES),
        "formal_phase2_eligible": False,
        "bundle_created": False,
        "bundle_id": None,
        "diagnostics": diagnostics,
        "checkpoint_load_audit": base_audit,
        "effective8_delta_audit": delta_audit,
        "effective8_merge_audit": merge_audit,
        "resource_audit": {
            "trainable_parameters": 0,
            "optimizer_steps": 0,
            "query_state_updates": 0,
            "runtime_batch_size": int(args.runtime_batch_size),
            "base_runtime_state_bytes": _state_bytes(base_script),
            "candidate_runtime_state_bytes": _state_bytes(candidate_script),
        },
    }
    _write_json_new(outputs[2], receipt)
    return {
        "status": "PASS",
        "formal_phase2_eligible": False,
        "bundle_created": False,
        "base_runtime": str(outputs[0]),
        "candidate_runtime": str(outputs[1]),
        "export_receipt": str(outputs[2]),
        "base_runtime_sha256": receipt["base_runtime_sha256"],
        "candidate_runtime_sha256": receipt["candidate_runtime_sha256"],
        "export_receipt_sha256": sha256_file(outputs[2]),
        "diagnostics": diagnostics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-state", type=Path, required=True)
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--base-runtime-out", type=Path, required=True)
    parser.add_argument("--candidate-runtime-out", type=Path, required=True)
    parser.add_argument("--export-receipt-out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--parity-seed", type=int, default=20260721)
    parser.add_argument("--parity-rows", type=int, default=8)
    parser.add_argument("--runtime-batch-size", type=int, default=256)
    parser.add_argument("--max-abs-tolerance", type=float, default=1.0e-4)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(export(parse_args()), ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
