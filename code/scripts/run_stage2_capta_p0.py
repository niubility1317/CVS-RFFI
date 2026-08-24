"""Run protocol-safe CAPTA-P0 on an existing VALIDATED_ONCE Stage2-B row.

The row/package readers are reused from the previously verified Stage2-B
launcher.  CAPTA owns only support-state construction and read-only scoring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
for _path in (str(CODE_ROOT), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import scripts.run_stage2_structured_late_block_no_query_smoke as late
from cvsrffi.stage2_capta.prototype_transport import (
    A1_SUPPORT_SHRINK,
    A2_SHARED_SHIFT,
    A3_R4_SUPPORT_SHIFT,
)
from cvsrffi.stage2_capta.runtime import (
    CaptaConfig,
    CaptaPhase2Context,
    adapt_on_target_support,
    predict_query_read_only,
)


def _context(value: dict[str, Any]) -> CaptaPhase2Context:
    return CaptaPhase2Context(
        protocol_schema=str(value["protocol_schema"]),
        phase2_data_status=str(value["phase2_data_status"]),
        capsule_id=str(value["capsule_id"]),
        split_id=str(value["split_id"]),
    )


def _adapt_from_whitelist(
    args: argparse.Namespace,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any], Any]:
    context_value = late._read_context(Path(args.context).resolve())
    received_iq, support_class_indices = late._load_support_only(
        Path(args.support_only).resolve()
    )
    prototypes, class_ids, prototype_checkpoint_sha256 = late._load_prototypes(
        Path(args.frozen_prototypes).resolve()
    )
    if (
        len(received_iq) != int(context_value["support_input_count"])
        or np.any(support_class_indices < 0)
        or np.any(support_class_indices >= len(class_ids))
    ):
        raise ValueError("support-only input does not match preregistered context")
    model, checkpoint_info = late._exact_adv3b02(
        Path(args.checkpoint).resolve(), device=torch.device(args.device)
    )
    if (
        str(checkpoint_info["checkpoint_sha256"]) != prototype_checkpoint_sha256
        or str(context_value["checkpoint_sha256"]) != prototype_checkpoint_sha256
    ):
        raise ValueError("checkpoint/prototype/context binding mismatch")
    support_labels = tuple(class_ids[int(index)] for index in support_class_indices)
    state = adapt_on_target_support(
        model,
        received_iq,
        support_labels,
        prototypes,
        class_ids,
        context=_context(context_value),
        config=CaptaConfig(
            candidate_id=str(args.candidate_id),
            rank=int(args.rank),
            prior_strength=float(args.prior_strength),
        ),
        device=args.device,
    )
    return model, checkpoint_info, context_value, state


def smoke(args: argparse.Namespace) -> dict[str, Any]:
    _, checkpoint_info, context_value, state = _adapt_from_whitelist(args)
    result = {
        "status": "PASS",
        "checkpoint_load_strict": bool(checkpoint_info["checkpoint_load_strict"]),
        "checkpoint_load_audit": checkpoint_info["checkpoint_load_audit"],
        "support_input_count": int(context_value["support_input_count"]),
        "source_input_count": 0,
        "query_input_count": 0,
        "query_loaded": False,
        "audit": state.audit,
    }
    late._write_json(Path(args.output).resolve(), result)
    return result


def run_baseline(args: argparse.Namespace) -> dict[str, Any]:
    """Reuse the exact frozen DA0_REG0 decision path."""

    return late.run_baseline(args)


def run_row(args: argparse.Namespace) -> dict[str, Any]:
    model, checkpoint_info, context_value, state = _adapt_from_whitelist(args)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("CAPTA state did not freeze before query open")

    query_row = late._read_row_binding(Path(args.row_binding).resolve())
    row_identity_fields = (
        "protocol_schema",
        "phase2_data_status",
        "capsule_id",
        "split_id",
    )
    if any(
        str(query_row[field]) != str(context_value[field])
        for field in row_identity_fields
    ):
        raise ValueError("support/query row binding mismatch")

    query_received_iq, query_tokens = late._load_query_received_iq(
        Path(args.query_package).resolve(),
        Path(args.package_manifest).resolve(),
        Path(args.validated_row_manifest).resolve(),
        Path(args.row_binding).resolve(),
    )
    prediction = predict_query_read_only(
        model,
        query_received_iq,
        state,
        context=_context(context_value),
    )
    rows = [
        {
            "sample_index": index,
            "query_token": str(query_tokens[index]),
            "predicted_class_id": prediction.predicted_class_ids[index],
            "scores": prediction.mixed_scores[index].tolist(),
            "source_scores": prediction.source_scores[index].tolist(),
            "target_scores": prediction.target_scores[index].tolist(),
            "source_weight": float(state.source_weight),
        }
        for index in range(len(prediction.predicted_class_ids))
    ]
    result = {
        "status": "PREDICTIONS_COMPLETE",
        "state": "DA1_REG0",
        "checkpoint_load_strict": bool(checkpoint_info["checkpoint_load_strict"]),
        "source_input_count": 0,
        "support_input_count": int(context_value["support_input_count"]),
        "query_input_count": len(rows),
        "query_truth_loaded": False,
        "query_role_loaded": False,
        "query_batch_state_updated": bool(prediction.query_batch_state_updated),
        "audit": state.audit,
        "predictions": rows,
    }
    late._write_json(Path(args.output).resolve(), result)
    return result


def _add_capta_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--support-only", required=True)
    parser.add_argument("--frozen-prototypes", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--candidate-id",
        choices=(A1_SUPPORT_SHRINK, A2_SHARED_SHIFT, A3_R4_SUPPORT_SHIFT),
        default=A3_R4_SUPPORT_SHIFT,
    )
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--prior-strength", type=float, default=3.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke_parser = subparsers.add_parser("smoke")
    _add_capta_inputs(smoke_parser)
    smoke_parser.set_defaults(handler=smoke)

    row_parser = subparsers.add_parser("run-row")
    _add_capta_inputs(row_parser)
    row_parser.add_argument("--query-package", required=True)
    row_parser.add_argument("--package-manifest", required=True)
    row_parser.add_argument("--validated-row-manifest", required=True)
    row_parser.add_argument("--row-binding", required=True)
    row_parser.set_defaults(handler=run_row)

    baseline_parser = subparsers.add_parser("run-baseline")
    baseline_parser.add_argument("--checkpoint", required=True)
    baseline_parser.add_argument("--frozen-prototypes", required=True)
    baseline_parser.add_argument("--context", required=True)
    baseline_parser.add_argument("--output", required=True)
    baseline_parser.add_argument("--device", default="cpu")
    baseline_parser.add_argument("--query-package", required=True)
    baseline_parser.add_argument("--package-manifest", required=True)
    baseline_parser.add_argument("--validated-row-manifest", required=True)
    baseline_parser.add_argument("--row-binding", required=True)
    baseline_parser.set_defaults(handler=run_baseline)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = args.handler(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
