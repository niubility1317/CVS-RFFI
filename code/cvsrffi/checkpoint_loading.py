"""Fail-closed reconstruction for Phase1 SSDG checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch


def strip_module_prefix(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        (str(key)[7:] if str(key).startswith("module.") else str(key)): value
        for key, value in state.items()
    }


def infer_num_domains_from_state(state: Mapping[str, torch.Tensor]) -> int:
    for key in (
        "dom_head.net.3.bias",
        "dom_head.net.3.weight",
        "adv_head.net.3.bias",
        "adv_head.net.3.weight",
    ):
        value = state.get(key)
        if torch.is_tensor(value) and value.ndim >= 1 and int(value.shape[0]) > 0:
            return int(value.shape[0])
    raise ValueError("cannot infer num_domains from checkpoint state")


def build_exact_ssdg_model_from_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    input_len: int,
    device: torch.device,
    ssdg_module=None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Rebuild the training-time SSDG model and require a complete state load.

    This intentionally avoids rebuilding the historical source data split. The
    checkpoint state determines the domain-head width, while checkpoint args
    determine every architecture option used by ``build_baseline_model``.
    """

    if "args" not in checkpoint or "model" not in checkpoint:
        raise KeyError("checkpoint must contain 'args' and 'model'")
    checkpoint_args = dict(checkpoint.get("args") or {})
    state = strip_module_prefix(checkpoint["model"])
    num_domains = infer_num_domains_from_state(state)
    if ssdg_module is None:
        from SSDG import train_ssdg as ssdg_module

    parser = ssdg_module.build_arg_parser()
    parsed = parser.parse_args(
        ["--output_dir", str(Path(".tmp_exact_ssdg_checkpoint_rebuild"))]
    )
    for key, value in checkpoint_args.items():
        setattr(parsed, key, value)
    parsed.device = str(device)
    merged = ssdg_module.merge_checkpoint_args(
        checkpoint,
        parsed,
        input_len=int(input_len),
        num_domains=int(num_domains),
    )
    merged = ssdg_module._apply_model_cli_args(merged, parsed)
    model = ssdg_module.build_baseline_model(merged, device)
    try:
        incompatible = model.load_state_dict(state, strict=False)
    except RuntimeError as exc:
        raise ValueError(f"strict checkpoint reconstruction shape mismatch: {exc}") from exc
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        raise ValueError(
            "strict checkpoint reconstruction failed: "
            f"missing={missing} unexpected={unexpected}"
        )
    audit = {
        "loader": "exact_ssdg_training_architecture_v1",
        "checkpoint_load_strict": True,
        "crra_enabled": bool(checkpoint_args.get("use_crra", False)),
        "missing_keys": 0,
        "unexpected_keys": 0,
        "skipped_mismatch": 0,
        "state_tensor_count": len(state),
        "num_domains_from_state": int(num_domains),
        "input_len": int(input_len),
    }
    return model, audit
