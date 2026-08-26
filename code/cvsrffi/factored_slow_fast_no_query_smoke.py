"""Real-checkpoint support smoke for CVS-FSFA-V2 with no query capability."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from .factored_slow_fast_bundle import load_factored_bundle_strict
from .stage2_factored_slow_fast_runner import _extract_features, _safe_context
from .stage2_meta_adapter_runner import (
    _integer_tensor,
    _load_npz,
    _prototype_tensors,
    _received_iq_tensor,
    _support_physical_ids,
    _write_json_atomically,
)
from .stage2_structured_late_block_runner import _load_frozen_checkpoint


_KEYS = frozenset(
    {
        "candidate_id", "bundle_id", "protocol_schema", "phase2_data_status", "capsule_id", "split_id",
        "base_checkpoint_path", "bundle_path", "support_path", "prototype_path", "receiver", "scenario",
        "operating_point", "seed", "k_shot",
    }
)


def run_factored_slow_fast_no_query_smoke(
    config: Mapping[str, Any],
    output_path: str | Path,
    *,
    device: str | torch.device,
) -> dict[str, Any]:
    if not isinstance(config, Mapping) or set(config) != set(_KEYS):
        raise ValueError("factored no-query smoke allowlist mismatch")
    if config["protocol_schema"] != "p2_min_v1" or config["phase2_data_status"] != "VALIDATED_ONCE":
        raise ValueError("factored no-query smoke requires validated p2_min_v1 support")
    prototype_payload = _load_npz(
        config["prototype_path"], allowed=frozenset({"prototypes", "class_ids"}), label="prototype"
    )
    prototypes, class_ids = _prototype_tensors(prototype_payload)
    state, bundle_audit = load_factored_bundle_strict(config["bundle_path"], decision_prototypes=prototypes)
    expected = "B3" if str(config["candidate_id"]).endswith("B3") else "B5"
    if bundle_audit["candidate"] != expected or not torch.equal(state.class_ids, class_ids.cpu()):
        raise ValueError("factored smoke bundle mismatch")
    model = _load_frozen_checkpoint(config["base_checkpoint_path"], device=device)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("factored smoke base checkpoint is not frozen")
    support = _load_npz(
        config["support_path"],
        allowed=frozenset({"received_iq", "support_labels", "support_physical_ids"}),
        label="support",
    )
    support_iq = _received_iq_tensor(support["received_iq"], label="support")
    labels = _integer_tensor(support["support_labels"], label="support_labels")
    physical_ids = _support_physical_ids(
        support["support_physical_ids"], expected_rows=int(support_iq.shape[0])
    )
    if labels.numel() != support_iq.shape[0] or not set(int(value) for value in labels.tolist()) <= set(int(value) for value in state.class_ids.tolist()):
        raise ValueError("factored smoke support must contain registered old classes only")
    context, selection = _safe_context(_extract_features(model, support_iq), labels, state)
    receipt = {
        "status": "SMOKE_PASS",
        "candidate_id": config["candidate_id"],
        "bundle_id": config["bundle_id"],
        "base_checkpoint_id": bundle_audit["base_checkpoint_id"],
        "support_physical_sample_count": len(physical_ids),
        "fast_parameter_count": int(context.numel()),
        "selected_context_scale": float(selection["selected_context_scale"]),
        "aggregate_storage_dtype": bundle_audit["aggregate_storage_dtype"],
        "query_input_capability": False,
        "query_opened": False,
        "query_truth_opened": False,
        "query_role_opened": False,
        "source_opened": False,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomically(destination, receipt)
    return receipt


__all__ = ["run_factored_slow_fast_no_query_smoke"]
