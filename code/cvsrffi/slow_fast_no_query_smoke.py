"""Real-checkpoint support smoke that has no query input capability."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from .slow_fast_bundle import load_slow_fast_bundle_strict
from .slow_fast_selection import select_support_only_state
from .stage2_meta_adapter_runner import (
    _integer_tensor,
    _load_npz,
    _prototype_tensors,
    _received_iq_tensor,
    _support_physical_ids,
    _write_json_atomically,
)
from .stage2_slow_fast_runner import _extract_features, _row_labels
from .stage2_structured_late_block_runner import _load_frozen_checkpoint


_KEYS = frozenset(
    {
        "candidate_id", "bundle_id", "protocol_schema", "phase2_data_status",
        "capsule_id", "split_id", "base_checkpoint_path", "bundle_path",
        "support_path", "prototype_path", "receiver", "scenario",
        "operating_point", "seed", "k_shot", "steps",
    }
)


def run_slow_fast_no_query_smoke(
    config: Mapping[str, Any],
    output_path: str | Path,
    *,
    device: str | torch.device,
) -> dict[str, Any]:
    if not isinstance(config, Mapping) or frozenset(config) != _KEYS:
        actual = frozenset(config) if isinstance(config, Mapping) else frozenset()
        raise ValueError(
            "no-query smoke allowlist mismatch: "
            f"missing={sorted(_KEYS - actual)} extra={sorted(actual - _KEYS)}"
        )
    if config["protocol_schema"] != "p2_min_v1" or config["phase2_data_status"] != "VALIDATED_ONCE":
        raise ValueError("no-query smoke requires validated p2_min_v1 support")
    if int(config["k_shot"]) < 1 or int(config["steps"]) != 3:
        raise ValueError("no-query smoke requires positive K and three steps")

    state, audit = load_slow_fast_bundle_strict(config["bundle_path"])
    if state.candidate.value != config["candidate_id"]:
        raise ValueError("smoke candidate and bundle mismatch")
    prototype_payload = _load_npz(
        config["prototype_path"],
        allowed=frozenset({"prototypes", "class_ids"}),
        label="prototype",
    )
    prototypes, class_ids = _prototype_tensors(prototype_payload)
    if not torch.equal(class_ids, audit["class_ids"]) or not torch.allclose(
        prototypes, audit["prototypes"], atol=0.0, rtol=0.0
    ):
        raise ValueError("smoke prototypes do not match bundle")
    model = _load_frozen_checkpoint(config["base_checkpoint_path"], device=device)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("smoke base checkpoint is not frozen")
    support = _load_npz(
        config["support_path"],
        allowed=frozenset({"received_iq", "support_labels", "support_physical_ids"}),
        label="support",
    )
    support_iq = _received_iq_tensor(support["received_iq"], label="support")
    labels = _integer_tensor(support["support_labels"], label="support_labels")
    if labels.numel() != support_iq.shape[0]:
        raise ValueError("smoke support labels and IQ must align")
    physical_ids = _support_physical_ids(
        support["support_physical_ids"], expected_rows=int(support_iq.shape[0])
    )
    selected, selection = select_support_only_state(
        _extract_features(model, support_iq),
        _row_labels(labels, class_ids),
        prototypes,
        state,
        k_shot=int(config["k_shot"]),
        logit_scale=float(audit["support_logit_scale"]),
        steps=3,
        step_size=float(audit["fast_step_size"]),
        trust_radius=float(audit["trust_radius"]),
    )
    receipt = {
        "status": "SMOKE_PASS",
        "candidate_id": config["candidate_id"],
        "bundle_id": config["bundle_id"],
        "base_checkpoint_id": audit["base_checkpoint_id"],
        "support_physical_sample_count": len(physical_ids),
        "selected_lambda": float(selection["selected_lambda"]),
        "selection_reason": str(selection["reason"]),
        "selected_fast_parameter_count": selected.fast_parameter_count,
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


__all__ = ["run_slow_fast_no_query_smoke"]
