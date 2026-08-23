#!/usr/bin/env python3
"""Train a support-only sparse ADV3B02 encoder state (no query input)."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence

import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.checkpoint_loading import build_exact_ssdg_model_from_checkpoint
from cvsrffi.stage2_support_sparse_encoder_adaptation import (
    SparseEncoderAdaptationConfig,
    adapt_on_target_support,
    load_validated_enrollment_support,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_checkpoint(path: Path, *, input_len: int, device: torch.device):
    try:
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    model, audit = build_exact_ssdg_model_from_checkpoint(
        checkpoint,
        input_len=int(input_len),
        device=device,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, audit


def _validate_output_paths(output_state: Path, output_audit: Path) -> None:
    if output_state.resolve(strict=False) == output_audit.resolve(strict=False):
        raise ValueError("output-state and output-audit must be different paths")
    for output in (output_state, output_audit):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite output: {output}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--enrollment-root", type=Path, required=True)
    parser.add_argument("--context-json", type=Path, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--k-shot", type=int, required=True)
    parser.add_argument(
        "--candidate",
        choices=(
            "c1_norm_affine",
            "c2_norm_gates",
            "c3_norm_gates_fproj",
        ),
        default="c1_norm_affine",
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--feature-anchor-weight", type=float, default=0.05)
    parser.add_argument("--parameter-anchor-weight", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-state", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_output_paths(args.output_state, args.output_audit)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.context_json.is_file():
        raise FileNotFoundError(args.context_json)
    context = json.loads(args.context_json.read_text(encoding="utf-8-sig"))
    support = load_validated_enrollment_support(
        args.enrollment_root,
        scenario=str(args.scenario),
        k_shot=int(args.k_shot),
        context=context,
    )
    checkpoint_sha256 = _sha256_file(args.checkpoint)
    if checkpoint_sha256 != support.checkpoint_sha256:
        raise ValueError(
            "eager checkpoint SHA256 does not match the validated enrollment package"
        )
    if str(args.device).lower() == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(str(args.device))
    model, checkpoint_load_audit = _load_checkpoint(
        args.checkpoint,
        input_len=int(support.iq.shape[-1]),
        device=device,
    )
    audit = adapt_on_target_support(
        model,
        support.iq,
        support.class_indices,
        context=context,
        config=SparseEncoderAdaptationConfig(
            candidate=str(args.candidate),
            steps=int(args.steps),
            learning_rate=float(args.learning_rate),
            feature_anchor_weight=float(args.feature_anchor_weight),
            parameter_anchor_weight=float(args.parameter_anchor_weight),
            gradient_clip=float(args.gradient_clip),
        ),
    )
    named_parameters = dict(model.named_parameters())
    sparse_state = {
        name: named_parameters[name].detach().to(device="cpu", dtype=torch.float32)
        for name in audit.trainable_parameter_names
    }
    state_payload = {
        "schema": "cvs.stage2.sofesa_sparse_encoder_state.v1",
        "method_id": audit.method_id,
        "candidate": audit.candidate,
        "checkpoint_sha256": checkpoint_sha256,
        "scenario": support.scenario,
        "k_shot": int(args.k_shot),
        "protocol_schema": str(context["protocol_schema"]),
        "phase2_data_status": str(context["phase2_data_status"]),
        "capsule_id": str(context["capsule_id"]),
        "split_id": str(context["split_id"]),
        "encoder_state_dict": sparse_state,
    }
    args.output_state.parent.mkdir(parents=True, exist_ok=True)
    args.output_audit.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_payload, args.output_state)
    audit_payload = {
        "schema": "cvs.stage2.sofesa_adaptation_audit.v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_load_audit": checkpoint_load_audit,
        "enrollment_root": str(args.enrollment_root),
        "scenario": support.scenario,
        "k_shot": int(args.k_shot),
        "context": context,
        "adaptation": asdict(audit),
        "query_input_opened": False,
        "source_or_clean_input_opened": False,
        "classification_head_trained": False,
    }
    args.output_audit.write_text(
        json.dumps(audit_payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(audit_payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
