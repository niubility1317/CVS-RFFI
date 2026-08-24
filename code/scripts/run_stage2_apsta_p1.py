"""Run protocol-safe APSTA-P1 on one existing VALIDATED_ONCE Stage2-B row."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
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

import scripts.run_stage2_structured_late_block_no_query_smoke as late  # noqa: E402
from cvsrffi.stage2_apsta_time_robust import (  # noqa: E402
    ApstaConfig,
    ApstaPhase2Context,
    adapt_on_target_support,
    predict_query_read_only,
)


def _context(value: dict[str, Any]) -> ApstaPhase2Context:
    return ApstaPhase2Context(
        protocol_schema=str(value["protocol_schema"]),
        phase2_data_status=str(value["phase2_data_status"]),
        capsule_id=str(value["capsule_id"]),
        split_id=str(value["split_id"]),
    )


def _audit_dict(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    return dict(vars(value))


def _config(args: argparse.Namespace) -> ApstaConfig:
    return ApstaConfig(
        checkpoints=tuple(int(value) for value in args.checkpoints),
        learning_rate=float(args.learning_rate),
        anchor_strength=float(args.anchor_strength),
        head_ce_weight=float(args.head_ce_weight),
        loo_mean_weight=float(args.loo_mean_weight),
        tail_weight=float(args.tail_weight),
        tail_temperature=float(args.tail_temperature),
        topology_weight=float(args.topology_weight),
        l2sp_weight=float(args.l2sp_weight),
        margin_epsilon=float(args.margin_epsilon),
    )


def _adapt_from_whitelist(
    args: argparse.Namespace,
) -> tuple[
    torch.nn.Module,
    torch.nn.Module,
    dict[str, Any],
    dict[str, Any],
    Any,
    tuple[str, ...],
    np.ndarray,
]:
    context_value = late._read_context(Path(args.context).resolve())
    received_iq, support_indices = late._load_support_only(
        Path(args.support_only).resolve()
    )
    prototypes, class_ids, prototype_checkpoint_sha256 = late._load_prototypes(
        Path(args.frozen_prototypes).resolve()
    )
    if (
        len(received_iq) != int(context_value["support_input_count"])
        or np.any(support_indices < 0)
        or np.any(support_indices >= len(class_ids))
    ):
        raise ValueError("support-only input does not match preregistered context")
    student, checkpoint_info = late._exact_adv3b02(
        Path(args.checkpoint).resolve(), device=torch.device(args.device)
    )
    teacher, teacher_info = late._exact_adv3b02(
        Path(args.checkpoint).resolve(), device=torch.device(args.device)
    )
    observed_sha = str(checkpoint_info["checkpoint_sha256"])
    if (
        observed_sha != str(teacher_info["checkpoint_sha256"])
        or observed_sha != prototype_checkpoint_sha256
        or str(context_value["checkpoint_sha256"]) != prototype_checkpoint_sha256
    ):
        raise ValueError("checkpoint/prototype/context binding mismatch")
    labels = tuple(class_ids[int(index)] for index in support_indices)
    audit = adapt_on_target_support(
        student,
        received_iq,
        labels,
        prototypes,
        class_ids,
        context=_context(context_value),
        config=_config(args),
        device=args.device,
    )
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    teacher.eval()
    return student, teacher, checkpoint_info, context_value, audit, class_ids, prototypes


def smoke(args: argparse.Namespace) -> dict[str, Any]:
    _, _, checkpoint_info, context_value, audit, _, _ = _adapt_from_whitelist(args)
    result = {
        "status": "PASS",
        "checkpoint_load_strict": bool(checkpoint_info["checkpoint_load_strict"]),
        "checkpoint_load_audit": checkpoint_info["checkpoint_load_audit"],
        "support_input_count": int(context_value["support_input_count"]),
        "source_input_count": 0,
        "query_input_count": 0,
        "query_loaded": False,
        "audit": _audit_dict(audit),
    }
    late._write_json(Path(args.output).resolve(), result)
    return result


def run_baseline(args: argparse.Namespace) -> dict[str, Any]:
    return late.run_baseline(args)


def run_row(args: argparse.Namespace) -> dict[str, Any]:
    student, teacher, checkpoint_info, context_value, audit, class_ids, prototypes = (
        _adapt_from_whitelist(args)
    )
    if any(
        parameter.requires_grad
        for model in (student, teacher)
        for parameter in model.parameters()
    ):
        raise ValueError("APSTA did not freeze models before query open")

    query_row = late._read_row_binding(Path(args.row_binding).resolve())
    identity_fields = (
        "protocol_schema",
        "phase2_data_status",
        "capsule_id",
        "split_id",
    )
    if any(
        str(query_row[field]) != str(context_value[field])
        for field in identity_fields
    ):
        raise ValueError("support/query row binding mismatch")

    query_iq, query_tokens = late._load_query_received_iq(
        Path(args.query_package).resolve(),
        Path(args.package_manifest).resolve(),
        Path(args.validated_row_manifest).resolve(),
        Path(args.row_binding).resolve(),
    )
    prediction = predict_query_read_only(
        student,
        teacher,
        query_iq,
        prototypes,
        class_ids,
        context=_context(context_value),
    )
    rows = [
        {
            "sample_index": index,
            "query_token": str(query_tokens[index]),
            "predicted_class_id": prediction.predicted_class_ids[index],
            "scores": prediction.student_scores[index].tolist(),
            "student_scores": prediction.student_scores[index].tolist(),
            "teacher_scores": prediction.teacher_scores[index].tolist(),
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
        "query_batch_state_updated": bool(prediction.query_state_updated),
        "audit": _audit_dict(audit),
        "predictions": rows,
    }
    late._write_json(Path(args.output).resolve(), result)
    return result


def _add_adaptation_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--support-only", required=True)
    parser.add_argument("--frozen-prototypes", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoints", nargs="+", type=int, default=[0, 10, 30, 100, 300])
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--anchor-strength", type=float, default=3.0)
    parser.add_argument("--head-ce-weight", type=float, default=0.25)
    parser.add_argument("--loo-mean-weight", type=float, default=1.0)
    parser.add_argument("--tail-weight", type=float, default=0.50)
    parser.add_argument("--tail-temperature", type=float, default=0.50)
    parser.add_argument("--topology-weight", type=float, default=0.25)
    parser.add_argument("--l2sp-weight", type=float, default=1.0e-3)
    parser.add_argument("--margin-epsilon", type=float, default=0.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke_parser = subparsers.add_parser("smoke")
    _add_adaptation_inputs(smoke_parser)
    smoke_parser.set_defaults(handler=smoke)
    row_parser = subparsers.add_parser("run-row")
    _add_adaptation_inputs(row_parser)
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
