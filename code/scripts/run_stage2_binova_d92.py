#!/usr/bin/env python3
"""Run support-only BiNOVA-D92 Stage A, conditional Stage B, predict, and score."""

from __future__ import annotations

import argparse
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

from cvsrffi.stage2_binova_da import (  # noqa: E402
    NOVA_DA_Config,
    NOVA_DA_Module,
    NOVA_DA_State,
    evaluate_nova_da_crossfit,
    fit_nova_da,
)
from cvsrffi.stage2_binova_features import (  # noqa: E402
    BiNOVAQuery,
    BiNOVASupport,
    class_balanced_domain_context,
    extract_binova_features,
)
from cvsrffi.stage2_binova_lifecycle import (  # noqa: E402
    evaluate_stage_a_continuation_gate,
    freeze_binova_support_states,
    predict_binova_query_read_only,
    select_binova_mode,
)
from cvsrffi.stage2_binova_reg import (  # noqa: E402
    NOVA_REG_Config,
    NOVA_REG_Module,
    NOVA_REG_State,
    fit_nova_reg,
)
from cvsrffi.stage2_sf_erbt_four_state import score_four_state_predictions  # noqa: E402
from cvsrffi.target_only_progressive_runner import (  # noqa: E402
    _default_checkpoint_loader,
    _load_target_support,
)


PLAN_SCHEMA = "cvs.binova_d92.plan.v1"
BUNDLE_A_SCHEMA = "cvs.binova_d92.stage_a.v1"
BUNDLE_B_SCHEMA = "cvs.binova_d92.stage_b.v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect-plan")
    inspect.add_argument("--plan", type=Path, required=True)
    for name in ("adapt-a", "adapt-b", "run-auto", "predict"):
        command = commands.add_parser(name)
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--device", required=True)
        if name in {"adapt-b", "predict"}:
            command.add_argument("--stage-a-root", type=Path, required=True)
        if name == "predict":
            command.add_argument("--stage-b-root", type=Path)
    score = commands.add_parser("score")
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--truth", type=Path, required=True)
    score.add_argument("--prediction-receipt", type=Path, required=True)
    score.add_argument("--data-handle", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    return parser


def _load_plan(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"cannot load BiNOVA plan: {path}") from exc
    required = {
        "schema", "run_id", "protocol_schema", "phase2_data_status", "capsule_id",
        "base_checkpoint_path", "old_class_count", "k_shot", "new_class_count",
        "seed", "stage_a", "stage_b", "scene",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("BiNOVA plan allowlist mismatch")
    if payload["schema"] != PLAN_SCHEMA or payload["protocol_schema"] != "p2_min_v1":
        raise ValueError("BiNOVA plan schema mismatch")
    if payload["phase2_data_status"] != "VALIDATED_ONCE":
        raise ValueError("BiNOVA plan requires VALIDATED_ONCE data")
    if int(payload["old_class_count"]) != 6 or int(payload["k_shot"]) != 10:
        raise ValueError("BiNOVA minimal plan requires six old classes and K=10")
    scene_required = {
        "scenario", "split_id", "old_support", "registered_support", "query", "data_handle"
    }
    if not isinstance(payload["scene"], dict) or set(payload["scene"]) != scene_required:
        raise ValueError("BiNOVA scene allowlist mismatch")
    for section in ("stage_a", "stage_b"):
        if set(payload[section]) != {"steps", "learning_rate"}:
            raise ValueError(f"BiNOVA {section} allowlist mismatch")
    return payload


def _new_root(path: Path) -> Path:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"output root already exists: {path}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _save_torch_new(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("xb") as handle:
        torch.save(dict(payload), handle)


def _context(plan: Mapping[str, Any]) -> dict[str, str]:
    return {
        "protocol_schema": str(plan["protocol_schema"]),
        "phase2_data_status": str(plan["phase2_data_status"]),
        "capsule_id": str(plan["capsule_id"]),
        "split_id": str(plan["scene"]["split_id"]),
    }


def _ranks(labels: np.ndarray) -> np.ndarray:
    counts: dict[int, int] = {}
    output = []
    for value in labels.tolist():
        class_id = int(value)
        output.append(counts.get(class_id, 0))
        counts[class_id] = counts.get(class_id, 0) + 1
    return np.asarray(output, dtype=np.int64)


def _frozen_model(plan: Mapping[str, Any], device: torch.device) -> torch.nn.Module:
    model = _default_checkpoint_loader(plan["base_checkpoint_path"], device=device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _support_from_path(
    path: str | Path,
    model: torch.nn.Module,
    plan: Mapping[str, Any],
    device: torch.device,
) -> BiNOVASupport:
    loaded = _load_target_support(path)
    iq = loaded.received_iq.detach().cpu().numpy()
    labels = loaded.labels.detach().cpu().numpy().astype(np.int64, copy=False)
    features = extract_binova_features(
        model, iq, physical_ids=loaded.physical_ids, device=device
    )
    return BiNOVASupport(
        features=features, labels=labels, ranks=_ranks(labels), context=_context(plan)
    )


def _query_from_path(
    path: str | Path,
    model: torch.nn.Module,
    plan: Mapping[str, Any],
    device: torch.device,
) -> BiNOVAQuery:
    source = Path(path)
    try:
        with np.load(source, allow_pickle=False) as payload:
            if frozenset(payload.files) != frozenset({"received_iq", "query_ids"}):
                raise ValueError("query allowlist mismatch")
            iq = np.asarray(payload["received_iq"], dtype=np.float32)
            ids = tuple(np.asarray(payload["query_ids"]).astype(str).tolist())
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load label-free query: {source}") from exc
    features = extract_binova_features(model, iq, physical_ids=ids, device=device)
    return BiNOVAQuery(features=features, context=_context(plan))


def _baseline_stage_a(support: BiNOVASupport) -> NOVA_DA_State:
    module = NOVA_DA_Module()
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    context = class_balanced_domain_context(
        np.concatenate([support.features.domain160, support.features.physical6], axis=1),
        support.labels,
    )
    return NOVA_DA_State(
        module=module, config=NOVA_DA_Config(steps=1), domain_context166=context,
        audit={"query_rows_used": 0, "non_affine_fraction": 0.0, "baseline": True},
    )


def _stage_a_payload(state: NOVA_DA_State, candidate: str) -> dict[str, Any]:
    return {
        "schema": BUNDLE_A_SCHEMA,
        "candidate": candidate,
        "module_state": {name: value.detach().cpu() for name, value in state.module.state_dict().items()},
        "config": asdict(state.config),
        "domain_context166": state.domain_context166,
        "audit": dict(state.audit),
    }


def _load_stage_a(path: Path, device: torch.device) -> NOVA_DA_State:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != BUNDLE_A_SCHEMA:
        raise ValueError("Stage A bundle schema mismatch")
    config = NOVA_DA_Config(**payload["config"])
    module = NOVA_DA_Module(config.late_rank, config.identity_rank)
    module.load_state_dict(payload["module_state"], strict=True)
    module.to(device).eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return NOVA_DA_State(
        module=module, config=config, domain_context166=payload["domain_context166"], audit=payload["audit"]
    )


def _adapt_a(plan: Mapping[str, Any], output_root: Path, device: torch.device) -> Mapping[str, Any]:
    destination = _new_root(output_root)
    model = _frozen_model(plan, device)
    support = _support_from_path(plan["scene"]["old_support"], model, plan, device)
    common = {
        "steps": int(plan["stage_a"]["steps"]),
        "learning_rate": float(plan["stage_a"]["learning_rate"]),
        "seed": int(plan["seed"]),
    }
    candidates = {
        "A0": _baseline_stage_a(support),
        "A2": fit_nova_da(support, NOVA_DA_Config(**common, pseudo_registration=False), device=device),
        "A3": fit_nova_da(support, NOVA_DA_Config(**common), device=device),
        "A4": fit_nova_da(support, NOVA_DA_Config(**common, weight_affine_leak=0.0), device=device),
    }
    metrics = {name: dict(evaluate_nova_da_crossfit(state, support, device=device)) for name, state in candidates.items()}
    gate = evaluate_stage_a_continuation_gate(
        metrics["A2"], metrics["A3"],
        non_affine_fraction=float(candidates["A3"].audit["non_affine_fraction"]),
    )
    for name, state in candidates.items():
        _save_torch_new(destination / f"{name}.pt", _stage_a_payload(state, name))
    receipt = {
        "schema": "cvs.binova_d92.stage_a.selection.v1",
        "status": "STAGE_A_SUPPORT_CROSSFIT_COMPLETE",
        "run_id": plan["run_id"],
        "A1": "NOT_RUN_NON_GATE_LEGACY_T3_REFERENCE",
        "metrics": metrics,
        "gate": asdict(gate),
        "continue_stage_b": gate.passed,
        "query_rows_used": 0,
    }
    _write_json(destination / "stage_a_selection.json", receipt)
    return receipt


def _load_gate(stage_a_root: Path) -> Mapping[str, Any]:
    payload = json.loads((stage_a_root / "stage_a_selection.json").read_text(encoding="utf-8"))
    if payload.get("schema") != "cvs.binova_d92.stage_a.selection.v1":
        raise ValueError("Stage A selection schema mismatch")
    return payload


def _stage_b_payload(state: NOVA_REG_State, candidate: str) -> dict[str, Any]:
    conditioning = state.conditioning_d92
    return {
        "schema": BUNDLE_B_SCHEMA,
        "candidate": candidate,
        "module_state": {name: value.detach().cpu() for name, value in state.module.state_dict().items()},
        "config": asdict(state.config),
        "old_class_count": state.old_class_count,
        "condition_mean6": state.condition_mean6.detach().cpu(),
        "condition_scale6": state.condition_scale6.detach().cpu(),
        "audit": dict(state.audit),
        "conditioning": {
            "class_ids": conditioning.class_ids,
            "old_class_count": conditioning.old_class_count,
            "centers": conditioning.centers.detach().cpu(),
            "covariance": conditioning.covariance.detach().cpu(),
            "cholesky": conditioning.cholesky.detach().cpu(),
            "coefficient": conditioning.coefficient.detach().cpu(),
            "intercept": conditioning.intercept.detach().cpu(),
            "audit": dict(conditioning.audit),
        },
    }


def _load_stage_b(path: Path, stage_a: NOVA_DA_State, device: torch.device) -> NOVA_REG_State:
    from cvsrffi.stage2_binova_d92 import DifferentiableD92State

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != BUNDLE_B_SCHEMA:
        raise ValueError("Stage B bundle schema mismatch")
    config = NOVA_REG_Config(**payload["config"])
    module = NOVA_REG_Module(config.rank)
    module.load_state_dict(payload["module_state"], strict=True)
    module.to(device).eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    condition = payload["conditioning"]
    conditioning = DifferentiableD92State(
        class_ids=tuple(condition["class_ids"]), old_class_count=int(condition["old_class_count"]),
        centers=condition["centers"].to(device), covariance=condition["covariance"].to(device),
        cholesky=condition["cholesky"].to(device), coefficient=condition["coefficient"].to(device),
        intercept=condition["intercept"].to(device), audit=condition["audit"],
    )
    return NOVA_REG_State(
        module=module, stage_a=stage_a, conditioning_d92=conditioning,
        condition_mean6=payload["condition_mean6"].to(device),
        condition_scale6=payload["condition_scale6"].to(device),
        old_class_count=int(payload["old_class_count"]), config=config, audit=payload["audit"],
    )


def _adapt_b(
    plan: Mapping[str, Any], stage_a_root: Path, output_root: Path, device: torch.device
) -> Mapping[str, Any]:
    selection = _load_gate(stage_a_root)
    if selection.get("continue_stage_b") is not True:
        raise RuntimeError("Stage B blocked: Stage A support-only continuation gate not met")
    destination = _new_root(output_root)
    model = _frozen_model(plan, device)
    support = _support_from_path(plan["scene"]["registered_support"], model, plan, device)
    stage_a = _load_stage_a(stage_a_root / "A3.pt", device)
    baseline_a = _load_stage_a(stage_a_root / "A0.pt", device)
    common = {
        "steps": int(plan["stage_b"]["steps"]),
        "learning_rate": float(plan["stage_b"]["learning_rate"]),
        "seed": int(plan["seed"]),
    }
    candidates = {
        "B2": fit_nova_reg(baseline_a, support, old_class_count=6, config=NOVA_REG_Config(**common), device=device),
        "B3": fit_nova_reg(stage_a, support, old_class_count=6, config=NOVA_REG_Config(**common), device=device),
        "B4": fit_nova_reg(stage_a, support, old_class_count=6, config=NOVA_REG_Config(**common, weight_forgetting=0.0), device=device),
        "B5": fit_nova_reg(stage_a, support, old_class_count=6, config=NOVA_REG_Config(**common, weight_topology=0.0), device=device),
    }
    for name, state in candidates.items():
        _save_torch_new(destination / f"{name}.pt", _stage_b_payload(state, name))
    receipt = {
        "schema": "cvs.binova_d92.stage_b.selection.v1",
        "status": "STAGE_B_SUPPORT_FIT_COMPLETE",
        "run_id": plan["run_id"],
        "rows": {"B0": "DIRECT_D92", "B1": "A3_TO_D92", **{name: dict(state.audit) for name, state in candidates.items()}, "B6": "DEFERRED_UPPER_BOUND"},
        "selected": "B3",
        "query_rows_used": 0,
    }
    _write_json(destination / "stage_b_selection.json", receipt)
    return receipt


def _run_auto(
    plan: Mapping[str, Any], output_root: Path, device: torch.device
) -> Mapping[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"automatic run root already exists: {output_root}")
    stage_a_root = output_root / "stage_a"
    stage_a = _adapt_a(plan, stage_a_root, device)
    if stage_a.get("continue_stage_b") is not True:
        result = {
            "status": "STAGE_A_COMPLETE_STAGE_B_NOT_RUN_GATE_NOT_MET",
            "stage_a": stage_a,
            "stage_b": "NOT_RUN_GATE_NOT_MET",
        }
    else:
        stage_b = _adapt_b(plan, stage_a_root, output_root / "stage_b", device)
        result = {
            "status": "STAGE_A_COMPLETE_STAGE_B_AUTO_CONTINUED",
            "stage_a": stage_a,
            "stage_b": stage_b,
        }
    _write_json(output_root / "automatic_stage_result.json", result)
    return result


def _predict(
    plan: Mapping[str, Any], stage_a_root: Path, stage_b_root: Path | None,
    output_root: Path, device: torch.device,
) -> Mapping[str, Any]:
    destination = _new_root(output_root)
    model = _frozen_model(plan, device)
    old_support = _support_from_path(plan["scene"]["old_support"], model, plan, device)
    registered_support = _support_from_path(plan["scene"]["registered_support"], model, plan, device)
    stage_a = _load_stage_a(stage_a_root / "A3.pt", device)
    gate = _load_gate(stage_a_root)
    stage_b = None
    stage_b_passed = False
    if stage_b_root is not None:
        stage_b = _load_stage_b(stage_b_root / "B3.pt", stage_a, device)
        stage_b_passed = True
    selected_mode = select_binova_mode(
        stage_a_gate_passed=bool(gate.get("continue_stage_b")),
        stage_b_gate_passed=stage_b_passed,
    )
    frozen = freeze_binova_support_states(
        old_support, registered_support, stage_a=stage_a, stage_b=stage_b,
        selected_mode=selected_mode, seed=int(plan["seed"]), device=device,
    )
    query = _query_from_path(plan["scene"]["query"], model, plan, device)
    predictions = predict_binova_query_read_only(frozen, query)
    npz_payload: dict[str, np.ndarray] = {"query_ids": np.asarray(query.features.physical_ids)}
    for state_name, row in predictions.items():
        prefix = state_name.lower()
        for key in ("class_ids", "logits", "predictions"):
            npz_payload[f"{prefix}_{key}"] = np.asarray(row[key])
    with (destination / "predictions.npz").open("xb") as handle:
        np.savez(handle, **npz_payload)
    receipt = {
        "schema": "cvs.sf_erbt_four_state.prediction.v1",
        "status": "PREDICTIONS_COMPLETE",
        "capsule_id": plan["capsule_id"], "split_id": plan["scene"]["split_id"],
        "scenario": plan["scene"]["scenario"], "k_shot": 10,
        "old_class_count": 6, "new_class_count": int(plan["new_class_count"]),
        "query_rows": query.features.row_count,
        "four_states": ["DA0_REG0", "DA1_REG0", "DA0_REG1", "DA1_REG1"],
        "selected_mode": selected_mode,
        "support_states_frozen_before_query_open": True,
        "query_truth_opened": False, "query_role_opened": False, "source_opened": False,
    }
    _write_json(destination / "prediction_receipt.json", receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "score":
        result = score_four_state_predictions(
            args.predictions, args.truth, args.prediction_receipt, args.data_handle, args.output
        )
    else:
        plan = _load_plan(args.plan)
        if args.command == "inspect-plan":
            result = {"run_id": plan["run_id"], "stage_order": ["A", "B_IF_A_GATE_PASS"]}
        else:
            device = torch.device(args.device)
            if args.command == "adapt-a":
                result = _adapt_a(plan, args.output_root, device)
            elif args.command == "adapt-b":
                result = _adapt_b(plan, args.stage_a_root, args.output_root, device)
            elif args.command == "run-auto":
                result = _run_auto(plan, args.output_root, device)
            else:
                result = _predict(plan, args.stage_a_root, args.stage_b_root, args.output_root, device)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
