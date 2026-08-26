"""Truth-blind Phase2 runner for the int8 CVS-FSFA-V2 bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .factored_slow_fast import apply_factored_context, solve_factored_context, support_safety_diagnostics
from .factored_slow_fast_bundle import load_factored_bundle_strict
from .stage2_meta_adapter_runner import (
    _integer_tensor,
    _load_npz,
    _prototype_tensors,
    _query_ids,
    _received_iq_tensor,
    _support_physical_ids,
    _write_json_atomically,
    _write_prediction,
)
from .stage2_slow_fast_runner import _extract_features, _predict
from .stage2_structured_late_block_runner import _load_frozen_checkpoint


_CONFIG_KEYS = frozenset(
    {
        "adaptation_mode", "candidate_id", "bundle_id", "protocol_schema", "phase2_data_status",
        "capsule_id", "split_id", "base_checkpoint_path", "bundle_path", "support_path", "query_path",
        "prototype_path", "receiver", "scenario", "operating_point", "seed", "k_shot",
    }
)
_SUPPORT_KEYS = frozenset({"received_iq", "support_labels", "support_physical_ids"})
_QUERY_KEYS = frozenset({"received_iq", "query_ids"})
_PROTOTYPE_KEYS = frozenset({"prototypes", "class_ids"})


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping) or set(config) != set(_CONFIG_KEYS):
        raise ValueError("factored config allowlist mismatch")
    cfg = dict(config)
    if cfg["adaptation_mode"] != "FACTORED_CONTEXT_ADAPT":
        raise ValueError("factored runner requires FACTORED_CONTEXT_ADAPT")
    if cfg["candidate_id"] not in {"CVS_FSFA_V2_B3", "CVS_FSFA_V2_B5"}:
        raise ValueError("factored candidate_id mismatch")
    if cfg["protocol_schema"] != "p2_min_v1" or cfg["phase2_data_status"] != "VALIDATED_ONCE":
        raise ValueError("factored runner requires p2_min_v1/VALIDATED_ONCE")
    for key in ("bundle_id", "capsule_id", "split_id", "receiver", "scenario", "operating_point"):
        if not isinstance(cfg[key], str) or not cfg[key].strip():
            raise ValueError(f"{key} must be nonempty")
    cfg["seed"] = int(cfg["seed"])
    cfg["k_shot"] = int(cfg["k_shot"])
    if cfg["k_shot"] < 1:
        raise ValueError("k_shot must be positive")
    return cfg


def _safe_context(support: Tensor, labels: Tensor, state) -> tuple[Tensor, dict[str, Any]]:
    context, context_audit = solve_factored_context(support, labels, state)
    row_by_id = {int(value): row for row, value in enumerate(state.class_ids.tolist())}
    label_rows = torch.tensor([row_by_id[int(value)] for value in labels.tolist()])
    scales = (0.0,) if float(context_audit["support_shift_norm"]) <= 0.01 else (1.0, 0.75, 0.5, 0.25, 0.0)
    for scale in scales:
        scaled = context * scale
        safety = support_safety_diagnostics(
            F.normalize(support, dim=1),
            apply_factored_context(support, state, scaled),
            label_rows,
            state.decision_prototypes,
            coverage=float(context_audit["coverage"]) * scale * scale,
            disagreement=float(context_audit["class_code_disagreement"]) * scale,
            min_coverage=0.05 if scale > 0.0 else 0.0,
            max_disagreement=1.0,
            min_correct_margin_q10=0.5,
            min_wrong_margin_median=0.0,
            min_class_margin_cvar=-0.05,
        )
        if scale == 0.0 or safety["safe_to_commit"]:
            serializable_context = {
                key: value for key, value in context_audit.items() if key != "per_class_codes"
            }
            return scaled, {**serializable_context, **safety, "selected_context_scale": scale}
    raise AssertionError("zero factored context must be a safe fallback")


def run_factored_slow_fast_stage2_row(
    config: Mapping[str, Any],
    output_dir: str | Path,
    *,
    device: str | torch.device,
) -> dict[str, Any]:
    cfg = _validate_config(config)
    prototype_payload = _load_npz(cfg["prototype_path"], allowed=_PROTOTYPE_KEYS, label="prototype")
    prototypes, class_ids = _prototype_tensors(prototype_payload)
    state, bundle_audit = load_factored_bundle_strict(cfg["bundle_path"], decision_prototypes=prototypes)
    expected_candidate = "B3" if cfg["candidate_id"].endswith("B3") else "B5"
    if bundle_audit["candidate"] != expected_candidate or not torch.equal(state.class_ids, class_ids.cpu()):
        raise ValueError("factored bundle candidate/class IDs do not match row config")
    model = _load_frozen_checkpoint(cfg["base_checkpoint_path"], device=device)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("base checkpoint must remain frozen")

    support_payload = _load_npz(cfg["support_path"], allowed=_SUPPORT_KEYS, label="support")
    support_iq = _received_iq_tensor(support_payload["received_iq"], label="support")
    support_labels = _integer_tensor(support_payload["support_labels"], label="support_labels")
    if support_labels.numel() != support_iq.shape[0]:
        raise ValueError("support labels and IQ rows must align")
    support_ids = _support_physical_ids(support_payload["support_physical_ids"], expected_rows=int(support_iq.shape[0]))
    if not set(int(value) for value in support_labels.tolist()) <= set(int(value) for value in state.class_ids.tolist()):
        raise ValueError("factored context may use registered old-class support only")
    support_features = _extract_features(model, support_iq)
    context, support_audit = _safe_context(support_features, support_labels, state)

    query_payload = _load_npz(cfg["query_path"], allowed=_QUERY_KEYS, label="query")
    query_iq = _received_iq_tensor(query_payload["received_iq"], label="query")
    query_ids = _query_ids(query_payload["query_ids"], expected_rows=int(query_iq.shape[0]))
    query_features = torch.cat(
        [_extract_features(model, query_iq[row : row + 1]) for row in range(query_iq.shape[0])], dim=0
    )
    da0 = _predict(query_features, prototypes, class_ids)
    da1 = _predict(apply_factored_context(query_features, state, context), prototypes, class_ids)

    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"non-overwriting output already exists: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    prediction_paths: dict[str, str] = {}
    for name, values in (("DA0_REG0", da0), ("DA1_REG0", da1)):
        path = destination / f"predictions_{name}.npz"
        _write_prediction(path, query_ids=query_ids, predicted_class_ids=values[0], scores=values[1])
        prediction_paths[name] = str(path.resolve())
    receipt_path = destination / "receipt.json"
    receipt = {
        "status": "PREDICTIONS_COMPLETE",
        "adaptation_mode": cfg["adaptation_mode"],
        "support_adapter_opened": True,
        "states": ["DA0_REG0", "DA1_REG0"],
        "candidate_id": cfg["candidate_id"],
        "bundle_id": cfg["bundle_id"],
        "base_checkpoint_id": bundle_audit["base_checkpoint_id"],
        "protocol_schema": cfg["protocol_schema"],
        "phase2_data_status": cfg["phase2_data_status"],
        "capsule_id": cfg["capsule_id"],
        "split_id": cfg["split_id"],
        "receiver": cfg["receiver"],
        "scenario": cfg["scenario"],
        "operating_point": cfg["operating_point"],
        "seed": cfg["seed"],
        "k_shot": cfg["k_shot"],
        "support_physical_sample_count": len(support_ids),
        "query_count": int(query_iq.shape[0]),
        "fast_parameter_count": state.fast_parameter_count,
        "optimizer_state_bytes": 0,
        "aggregate_storage_dtype": bundle_audit["aggregate_storage_dtype"],
        "support_selection": support_audit,
        "query_state_update_count": 0,
        "query_truth_opened": False,
        "query_role_opened": False,
        "source_opened": False,
        "states_same_row": True,
        "query_ids": query_ids.tolist(),
        "prediction_paths": prediction_paths,
        "receipt_path": str(receipt_path.resolve()),
    }
    _write_json_atomically(receipt_path, receipt)
    return receipt


__all__ = ["run_factored_slow_fast_stage2_row"]
