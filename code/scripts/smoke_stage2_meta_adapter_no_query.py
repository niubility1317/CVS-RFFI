"""Run the real meta-checkpoint, support-only Phase2 smoke.

The smoke intentionally has no query path in its allowlist.  It proves strict
bundle loading, the real three-step support update and the frozen post-update
state; it never opens a query payload and emits no performance result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch
from torch import nn

from cvsrffi import stage2_meta_adapter_runner as _runner

# Kept as a module-level alias so a smoke harness can replace the strict
# loader without introducing a second checkpoint-loading path.
load_meta_bundle_strict = _runner.load_meta_bundle_strict


def run_meta_adapter_no_query_smoke(
    config: Mapping[str, Any],
    output_dir: str | Path,
    device: str | torch.device,
) -> Mapping[str, Any]:
    """Run one no-query smoke and write only a compact receipt."""

    resolved = _runner._validate_config(config, require_query=False)
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"output directory already exists: {destination}")
    target_device = torch.device(device)
    seed = int(resolved["seed"])
    torch.manual_seed(seed)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Read only checkpoint, support and frozen prototypes.  There is no query
    # argument in this path, so it cannot accidentally open a query file.
    model, raw_bundle_audit = load_meta_bundle_strict(
        resolved["checkpoint_path"], target_device
    )
    bundle_audit = _runner._require_strict_audit(raw_bundle_audit)
    if not isinstance(model, nn.Module):
        raise _runner.MetaAdapterStage2RunnerError(
            "strict meta bundle did not return a model"
        )
    support_payload = _runner._load_npz(
        resolved["support_path"],
        allowed=_runner._SUPPORT_KEYS,
        label="support",
    )
    support_iq = _runner._received_iq_tensor(
        support_payload["received_iq"], label="support"
    )
    support_labels = _runner._integer_tensor(
        support_payload["support_labels"], label="support_labels"
    )
    prototype_payload = _runner._load_npz(
        resolved["prototype_path"],
        allowed=_runner._PROTOTYPE_KEYS,
        label="prototype",
    )
    prototypes, class_ids = _runner._prototype_tensors(prototype_payload)
    _runner._validate_support(
        support_iq,
        support_labels,
        class_ids,
        k_shot=int(resolved["k_shot"]),
    )
    physical_ids = tuple(
        f"receiver={resolved['receiver']};support_physical_index={index:08d}"
        for index in range(int(support_iq.size(0)))
    )
    support_batch = _runner.ValidatedTargetSupportBatch(
        received_iq=support_iq,
        labels=support_labels,
        support_physical_ids=physical_ids,
        receiver_id=resolved["receiver"],
        context={key: resolved[key] for key in _runner._CONTEXT_KEYS},
    )
    handle = _runner._adapt(
        model, support_batch, prototypes, class_ids, resolved
    )
    audit = getattr(handle, "audit", None)
    backward_count = int(_runner._audit_value(audit, "gradient_updates", -1))
    if backward_count != 3:
        raise _runner.MetaAdapterStage2RunnerError(
            "no-query smoke requires exactly three support updates"
        )
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise _runner.MetaAdapterStage2RunnerError(
            "no-query smoke model is not fully frozen after support adaptation"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=False, exist_ok=False)
    receipt_path = destination / "smoke_receipt.json"
    receipt: dict[str, Any] = {
        "status": "REAL_META_CHECKPOINT_NO_QUERY_SMOKE_PASS",
        "protocol_schema": resolved["protocol_schema"],
        "phase2_data_status": resolved["phase2_data_status"],
        "capsule_id": resolved["capsule_id"],
        "split_id": resolved["split_id"],
        "receiver": resolved["receiver"],
        "scenario": resolved["scenario"],
        "operating_point": resolved["operating_point"],
        "seed": seed,
        "k_shot": int(resolved["k_shot"]),
        "steps": 3,
        "query_opened": False,
        "source_opened": False,
        "backward_count": backward_count,
        "checkpoint_load_strict": bool(bundle_audit["checkpoint_load_strict"]),
        "trainable_fraction": float(
            _runner._audit_value(
                audit, "trainable_fraction", bundle_audit["trainable_fraction"]
            )
        ),
        "query_state_update_count": 0,
        "performance_result": None,
        "receipt_path": str(receipt_path),
    }
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return receipt


run_no_query_smoke = run_meta_adapter_no_query_smoke


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = run_meta_adapter_no_query_smoke(config, args.output_dir, args.device)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
