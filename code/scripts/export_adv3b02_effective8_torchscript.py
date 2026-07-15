#!/usr/bin/env python
"""Export base and merged-effective8 ADV3B02 TorchScript runtimes with parity proof."""

from __future__ import annotations

import argparse
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
from cvsrffi.identity_only_forward import identity_only_feature_forward  # noqa: E402
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


PARITY_SCHEMA = "cvs.adv3b02_effective8_torchscript_parity.v1"


class ADV3B02IdentityRuntime(nn.Module):
    """Expose only the z_id feature and old-class logits used by strict qKNN."""

    def __init__(self, model: nn.Module, *, runtime_batch_size: int = 256) -> None:
        super().__init__()
        self.model = model
        self.runtime_batch_size = int(runtime_batch_size)

    def forward(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # ADV3B02 converts the batch dimension to a Python integer internally,
        # which would otherwise freeze the traced runtime to the example batch.
        # Present the model with one fixed deployment batch and slice the public
        # result back to the request size; the traced outer slices remain dynamic.
        count = rows.size(0)
        padded = rows.new_zeros(
            (self.runtime_batch_size, rows.size(1), rows.size(2))
        )
        padded[:count].copy_(rows)
        result = identity_only_feature_forward(self.model, padded, "z_id")
        if result is None:
            raise RuntimeError("ADV3B02 checkpoint does not support identity-only z_id export")
        features, logits = result
        return features[:count], logits[:count]


def _load_checkpoint(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_tensor_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict) or not all(torch.is_tensor(item) for item in value.values()):
        raise TypeError("adapter state must be a tensor dictionary")
    return {str(key): tensor for key, tensor in value.items()}


def _write_json_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


@torch.no_grad()
def _run(wrapper: nn.Module, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    wrapper.eval()
    features, logits = wrapper(rows)
    return features.detach().float().cpu(), logits.detach().float().cpu()


def _max_abs(left: torch.Tensor, right: torch.Tensor, *, field: str) -> float:
    if tuple(left.shape) != tuple(right.shape):
        raise ValueError(f"parity tensor shape mismatch: {field}")
    value = float(torch.max(torch.abs(left - right)).item())
    if not torch.isfinite(torch.tensor(value)):
        raise ValueError(f"non-finite parity error: {field}")
    return value


def _trace_and_save(wrapper: nn.Module, example: torch.Tensor, output: Path) -> torch.jit.ScriptModule:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite TorchScript runtime: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    wrapper.eval()
    # ADV3B02's FFT path can constant-fold an equivalent complex tensor with a
    # different internal dtype on the tracer's second graph construction.  A
    # graph-text sanity comparison therefore rejects a numerically identical
    # runtime.  The export path below performs stronger eager/injected/merged/
    # reloaded-TorchScript feature and logit parity on independent probes.
    traced = torch.jit.trace(wrapper, example, strict=False, check_trace=False)
    torch.jit.save(traced, output)
    if not output.is_file() or output.stat().st_size < 1:
        raise RuntimeError(f"TorchScript export is empty: {output}")
    return torch.jit.load(str(output), map_location=example.device).eval()


def export(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = Path(args.checkpoint)
    adapter_path = Path(args.adapter_state)
    if sha256_file(checkpoint_path) != BASE_CHECKPOINT_SHA256:
        raise ValueError("checkpoint is not the strict ADV3B02 base")
    device = torch.device(str(args.device) if torch.cuda.is_available() else "cpu")
    checkpoint = _load_checkpoint(checkpoint_path)
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
    adapter_state = _load_tensor_state(adapter_path)
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
    trace_example = probes[: min(2, len(probes))]
    base_wrapper = ADV3B02IdentityRuntime(base_model).to(device).eval()
    candidate_wrapper = ADV3B02IdentityRuntime(candidate_model).to(device).eval()
    injected_feature, injected_logit = _run(candidate_wrapper, probes)
    merge_audit = merge_feat_joint_lora(candidate_model)
    if (
        merge_audit.get("merged_module_count") != 8
        or merge_audit.get("remaining_lora_wrappers") != []
        or merge_audit.get("algebraic_probe_parity_pass") is not True
    ):
        raise ValueError("effective8 merge audit failed")
    merged_wrapper = ADV3B02IdentityRuntime(candidate_model).to(device).eval()
    merged_feature, merged_logit = _run(merged_wrapper, probes)

    base_output = Path(args.base_runtime_out)
    candidate_output = Path(args.candidate_runtime_out)
    base_script = _trace_and_save(base_wrapper, trace_example, base_output)
    candidate_script = _trace_and_save(merged_wrapper, trace_example, candidate_output)
    base_eager_feature, base_eager_logit = _run(base_wrapper, probes)
    base_script_feature, base_script_logit = _run(base_script, probes)
    candidate_script_feature, candidate_script_logit = _run(candidate_script, probes)

    tolerance = float(args.max_abs_tolerance)
    if not 0.0 < tolerance <= 1.0e-4:
        raise ValueError("TorchScript parity tolerance must be in (0,1e-4]")
    diagnostics = {
        "base_eager_vs_torchscript_feature": _max_abs(
            base_eager_feature, base_script_feature, field="base_feature"
        ),
        "base_eager_vs_torchscript_logit": _max_abs(
            base_eager_logit, base_script_logit, field="base_logit"
        ),
        "injected_vs_merged_feature": _max_abs(
            injected_feature, merged_feature, field="injected_merged_feature"
        ),
        "injected_vs_merged_logit": _max_abs(
            injected_logit, merged_logit, field="injected_merged_logit"
        ),
        "merged_vs_torchscript_feature": _max_abs(
            merged_feature, candidate_script_feature, field="merged_script_feature"
        ),
        "merged_vs_torchscript_logit": _max_abs(
            merged_logit, candidate_script_logit, field="merged_script_logit"
        ),
    }
    failed = {key: value for key, value in diagnostics.items() if value > tolerance}
    if failed:
        raise ValueError(f"ADV3B02 TorchScript parity failed: {failed}")
    receipt = {
        "schema": PARITY_SCHEMA,
        "status": "PASS",
        "base_runtime_sha256": sha256_file(base_output),
        "candidate_runtime_sha256": sha256_file(candidate_output),
        "adapter_state_sha256": sha256_file(adapter_path),
        "target_modules": list(EFFECTIVE8_TARGET_MODULES),
        "lora_tensor_keys": sorted(adapter_state),
        "max_abs_injected_vs_merged_feature": diagnostics[
            "injected_vs_merged_feature"
        ],
        "max_abs_injected_vs_merged_logit": diagnostics["injected_vs_merged_logit"],
        "max_abs_merged_vs_torchscript_feature": diagnostics[
            "merged_vs_torchscript_feature"
        ],
        "max_abs_merged_vs_torchscript_logit": diagnostics[
            "merged_vs_torchscript_logit"
        ],
    }
    parity_output = Path(args.parity_receipt_out)
    _write_json_new(parity_output, receipt)
    return {
        "status": "PASS",
        "base_runtime": str(base_output),
        "base_runtime_sha256": receipt["base_runtime_sha256"],
        "candidate_runtime": str(candidate_output),
        "candidate_runtime_sha256": receipt["candidate_runtime_sha256"],
        "parity_receipt": str(parity_output),
        "parity_receipt_sha256": sha256_file(parity_output),
        "diagnostics": diagnostics,
        "checkpoint_load_audit": base_audit,
        "delta_audit": delta_audit,
        "merge_audit": merge_audit,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-state", type=Path, required=True)
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--base-runtime-out", type=Path, required=True)
    parser.add_argument("--candidate-runtime-out", type=Path, required=True)
    parser.add_argument("--parity-receipt-out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--parity-seed", type=int, default=20260715)
    parser.add_argument("--parity-rows", type=int, default=8)
    parser.add_argument("--max-abs-tolerance", type=float, default=1.0e-4)
    return parser.parse_args()


def main() -> int:
    print(json.dumps(export(parse_args()), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
