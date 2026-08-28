#!/usr/bin/env python3
"""Run head-free t3.norm adaptation and fixed D92 four-state evaluation."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.stage2_sf_erbt_four_state import (  # noqa: E402
    run_four_state_prediction,
    score_four_state_predictions,
)
from cvsrffi.stage2_sf_erbt_oldonly import (  # noqa: E402
    _extract_identity160,
    make_fft96,
)
from cvsrffi.stage2_sf_t3_d92_combinations import (  # noqa: E402
    CANDIDATES,
    FORMAL_SCENES,
    NEW_CLASS_COUNTS,
    build_candidate_config,
    build_experiment_rows,
    validate_combo_plan,
)
from cvsrffi.stage2_sf_t3_d92_r3 import crossfit_d92_support_risk  # noqa: E402
from cvsrffi.stage2_sf_t3_delta_bundle import (  # noqa: E402
    load_t3_only_delta_bundle_strict,
    resolve_t3_parameter_names,
    write_t3_only_delta_bundle,
)
from cvsrffi.target_only_progressive_adapt import (  # noqa: E402
    ensure_time_adapter,
    fit_sf_tapft_inplace,
    fit_sf_tapft_rse_delta_ensemble,
    select_sf_tapft_by_grouped_cv,
)
from cvsrffi.target_only_progressive_runner import (  # noqa: E402
    _default_checkpoint_loader,
    _load_target_support,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser("inspect-plan")
    inspect.add_argument("--plan", type=Path, required=True)

    adapt = commands.add_parser("adapt")
    adapt.add_argument("--plan", type=Path, required=True)
    adapt.add_argument("--candidate", choices=CANDIDATES, required=True)
    adapt.add_argument("--scenario", choices=FORMAL_SCENES, required=True)
    adapt.add_argument("--new-count", choices=NEW_CLASS_COUNTS, type=int)
    adapt.add_argument("--output-root", type=Path, required=True)
    adapt.add_argument("--device", required=True)

    predict = commands.add_parser("predict")
    predict.add_argument("--base-checkpoint", type=Path, required=True)
    predict.add_argument("--t3-delta", type=Path, required=True)
    predict.add_argument("--old-support", type=Path, required=True)
    predict.add_argument("--registered-support", type=Path, required=True)
    predict.add_argument("--query", type=Path, required=True)
    predict.add_argument("--data-handle", type=Path, required=True)
    predict.add_argument("--output-root", type=Path, required=True)
    predict.add_argument("--seed", type=int, required=True)
    predict.add_argument("--device", required=True)

    score = commands.add_parser("score")
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--truth", type=Path, required=True)
    score.add_argument("--prediction-receipt", type=Path, required=True)
    score.add_argument("--data-handle", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    return parser


def _load_plan(path: Path) -> Mapping[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"cannot load combo plan: {path}") from exc
    return validate_combo_plan(raw)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _t3_deltas(result: Any) -> dict[str, torch.Tensor]:
    named = dict(result.model.named_parameters())
    anchor_names = tuple(
        str(name) for name in result.base_parameter_anchors
        if str(name).endswith(("t3.norm.weight", "t3.norm.bias"))
        and "dom_backbone." not in str(name)
    )
    try:
        resolved_names = resolve_t3_parameter_names(anchor_names)
    except ValueError as exc:
        raise RuntimeError("adapted result lacks one unambiguous identity t3 anchor pair") from exc
    output: dict[str, torch.Tensor] = {}
    for short_name in resolved_names:
        anchor_name = f"model.{short_name}"
        anchor = result.base_parameter_anchors.get(anchor_name)
        if not torch.is_tensor(anchor) or short_name not in named:
            raise RuntimeError(f"adapted result lacks t3 anchor: {anchor_name}")
        output[anchor_name] = named[short_name].detach().cpu() - anchor.detach().cpu()
    return output


def _zero_t3_deltas(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    named = dict(model.named_parameters())
    candidates = tuple(
        name for name in named
        if name.endswith(("t3.norm.weight", "t3.norm.bias"))
        and "dom_backbone." not in name
    )
    resolved_names = resolve_t3_parameter_names(candidates)
    return {
        f"model.{name}": torch.zeros_like(named[name], device="cpu")
        for name in resolved_names
    }


def _resource_audit(result: Any | None) -> dict[str, Any]:
    if result is None:
        return {
            "total_steps": None,
            "trainable_parameter_elements": None,
            "actual_changed_elements": 0,
            "backbone_train_forward_steps": None,
            "prefix_cache_build_forward_steps": None,
            "cached_suffix_forward_steps": None,
            "snapshot_tensor_bytes": None,
            "selection_compute_completed": True,
        }
    return {
        "total_steps": result.audit.total_steps,
        "trainable_parameter_elements": result.audit.trainable_parameter_elements,
        "actual_changed_elements": result.audit.actual_changed_elements,
        "backbone_train_forward_steps": result.audit.backbone_train_forward_steps,
        "prefix_cache_build_forward_steps": result.audit.prefix_cache_build_forward_steps,
        "cached_suffix_forward_steps": result.audit.cached_suffix_forward_steps,
        "snapshot_tensor_bytes": result.audit.snapshot_tensor_bytes,
        "selection_compute_completed": True,
    }


def _load_registered_support(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    source = Path(path)
    try:
        with np.load(source, allow_pickle=False) as payload:
            if frozenset(payload.files) != frozenset(
                {"received_iq", "support_labels", "support_physical_ids"}
            ):
                raise ValueError("registered support allowlist mismatch")
            iq = np.asarray(payload["received_iq"], dtype=np.float32)
            labels = np.asarray(payload["support_labels"], dtype=np.int64)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load registered support: {source}") from exc
    if (
        iq.ndim != 3
        or iq.shape[1:] != (2, 256)
        or labels.shape != (len(iq),)
        or not np.isfinite(iq).all()
    ):
        raise ValueError("registered support geometry drift")
    return iq, labels


def _r3_d92_strength_selection(
    base_checkpoint_path: str,
    fitted: Any,
    registered_support_path: str,
    *,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    raw_delta = _t3_deltas(fitted)
    iq, labels = _load_registered_support(registered_support_path)
    class_ids = tuple(int(value) for value in np.unique(labels).tolist())
    if class_ids[:6] != tuple(range(6)) or len(class_ids) <= 6:
        raise ValueError("R3 D92-in-loop requires the six-class old prefix and new classes")
    rows = []
    for alpha in (0.0, 0.5, 0.75, 1.0):
        candidate_model = _default_checkpoint_loader(base_checkpoint_path, device=device)
        ensure_time_adapter(candidate_model, rank=16)
        named = dict(candidate_model.named_parameters())
        with torch.no_grad():
            for canonical_name, delta in raw_delta.items():
                short_name = canonical_name.removeprefix("model.")
                named[short_name].add_(
                    float(alpha) * delta.to(device=named[short_name].device, dtype=named[short_name].dtype)
                )
        identity = _extract_identity160(candidate_model, iq, device)
        fft = make_fft96(iq)
        risk = crossfit_d92_support_risk(
            identity,
            fft,
            labels,
            class_ids=class_ids,
            old_class_ids=class_ids[:6],
            folds=2,
            seed=int(seed),
            device=device,
        )
        rows.append((float(risk.total), float(alpha), risk))
    _, selected_alpha, selected_risk = min(rows, key=lambda row: (row[0], row[1]))
    selected = {
        name: value * float(selected_alpha) for name, value in raw_delta.items()
    }
    return selected, {
        "selection": "support_only_d92_crossfit_strength",
        "selected_alpha": selected_alpha,
        "candidate_risks": [
            {
                "alpha": alpha,
                "total": risk.total,
                "macro_nll": risk.macro_nll,
                "class_tail_nll": risk.class_tail_nll,
                "class_floor_error": risk.class_floor_error,
                "old_new_balance": risk.old_new_balance,
            }
            for _total, alpha, risk in rows
        ],
        "selected_risk": {
            "total": selected_risk.total,
            "macro_nll": selected_risk.macro_nll,
            "class_tail_nll": selected_risk.class_tail_nll,
            "class_floor_error": selected_risk.class_floor_error,
            "old_new_balance": selected_risk.old_new_balance,
        },
        "d92_method_lock": "D92-E0-NORF32",
        "rf32_used": False,
        "query_rows_used": 0,
    }


def _adapt(args: argparse.Namespace) -> dict[str, Any]:
    plan = _load_plan(args.plan)
    scene = plan["scenes"][args.scenario]
    output_root = Path(args.output_root)
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"adapt output root already exists: {output_root}")
    if args.candidate == "R3_DUALDELTA_T3_D92_INLOOP":
        if args.new_count not in NEW_CLASS_COUNTS:
            raise ValueError("R3 D92-in-loop requires --new-count")
    elif args.new_count is not None:
        raise ValueError("D0/S02 adaptation is shared across new-class counts")

    target_device = torch.device(args.device)
    model = _default_checkpoint_loader(plan["base_checkpoint_path"], device=target_device)
    support = _load_target_support(scene["old_support"])
    config = build_candidate_config(args.candidate)
    method_receipt: dict[str, Any]
    if args.candidate == "D0_T3_D92":
        result = fit_sf_tapft_inplace(
            model,
            support,
            config,
            checkpoint_selection_mode="final_step",
        )
        deltas = _t3_deltas(result)
        method_receipt = {
            "route": "D0_H6_COMPACT_FIXED",
            "phase_steps": list(config.phase_steps),
        }
    elif args.candidate == "S02_T3_D92":
        selection = select_sf_tapft_by_grouped_cv(
            model,
            support,
            config,
            folds=4,
            full_support_refit=True,
        )
        result = selection.full_support_result
        selected = str(selection.selected)
        if result is None and selected in {"zero_adapt", "frozen"}:
            deltas = _zero_t3_deltas(model)
        elif result is not None and selection.adapted_result is not None:
            deltas = _t3_deltas(result)
        else:
            raise RuntimeError("S02 grouped selection result geometry drift")
        method_receipt = {
            "route": "S02_GROUPED_OOF_FULL_SUPPORT_REFIT",
            "folds": 4,
            "selected": selected,
            "selected_phase_steps": list(selection.selected_phase_steps),
            "frozen_metrics": asdict(selection.frozen_metrics),
            "adapted_metrics": asdict(selection.adapted_metrics),
            "zero_delta_fallback": result is None,
            "full_support_refit": result is not None,
        }
    else:
        ensemble = fit_sf_tapft_rse_delta_ensemble(
            model,
            support,
            config,
            ensemble_count=2,
            per_class=8,
            polish_steps=30,
        )
        result = ensemble.result
        registered_path = scene["registered_support_pattern"].format(
            new_count=args.new_count
        )
        deltas, d92_receipt = _r3_d92_strength_selection(
            plan["base_checkpoint_path"],
            result,
            registered_path,
            device=target_device,
            seed=config.seed,
        )
        method_receipt = {
            "route": "R3_DUALDELTA_T3_D92_INLOOP",
            "subset_fit_count": ensemble.subset_fit_count,
            "polish_steps": ensemble.polish_steps,
            "new_class_count": args.new_count,
            **d92_receipt,
        }

    output_root.mkdir(parents=True, exist_ok=False)
    delta_path = output_root / "sf_t3_only_delta_bundle.pt"
    delta_receipt = write_t3_only_delta_bundle(
        delta_path,
        model_deltas=deltas,
        protocol_schema=plan["protocol_schema"],
        phase2_data_status=plan["phase2_data_status"],
        capsule_id=plan["capsule_id"],
        split_id=scene["split_id"],
        base_checkpoint_path=plan["base_checkpoint_path"],
        candidate_id=args.candidate,
        support_count=len(support.physical_ids),
        d92_method_lock=plan["d92_method_lock"],
        adapter_rank=config.adapter_rank,
    )
    receipt = {
        "status": "DEPLOY_ADAPT_COMPLETE",
        "candidate_id": args.candidate,
        "scenario": args.scenario,
        "protocol_schema": plan["protocol_schema"],
        "phase2_data_status": plan["phase2_data_status"],
        "capsule_id": plan["capsule_id"],
        "split_id": scene["split_id"],
        "support_count": len(support.physical_ids),
        "persistent_parameter_names": [
            "model.t3.norm.weight",
            "model.t3.norm.bias",
        ],
        "temporary_target_head_persisted": False,
        "d92_method_lock": "D92-E0-NORF32",
        "rf32_used": False,
        "query_rows_used": 0,
        "query_truth_opened": False,
        "query_role_opened": False,
        "delta": delta_receipt,
        "method": method_receipt,
        "resource_audit": _resource_audit(result),
    }
    _write_json(output_root / "selection.json", receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "inspect-plan":
        plan = _load_plan(args.plan)
        result = {"row_count": len(build_experiment_rows(plan)), "run_id": plan["run_id"]}
    elif args.command == "adapt":
        result = _adapt(args)
    elif args.command == "predict":
        result = run_four_state_prediction(
            base_checkpoint_path=args.base_checkpoint,
            d3_delta_path=args.t3_delta,
            old_support_path=args.old_support,
            registered_support_path=args.registered_support,
            query_path=args.query,
            data_handle_path=args.data_handle,
            output_root=args.output_root,
            seed=args.seed,
            device=args.device,
            delta_bundle_loader=load_t3_only_delta_bundle_strict,
        )
    else:
        result = score_four_state_predictions(
            args.predictions,
            args.truth,
            args.prediction_receipt,
            args.data_handle,
            args.output,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
