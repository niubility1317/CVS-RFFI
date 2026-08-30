#!/usr/bin/env python3
"""Run the truth-blind WISER-RF A/B/C historical pilot and score it later."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.stage2_bisage_runner import frozen_checkpoint  # noqa: E402
from cvsrffi.stage2_wiser_pilot import (  # noqa: E402
    ARMS,
    SCENARIOS,
    formal_promotion_decision,
    load_query_package,
    load_support_package,
)
from cvsrffi.stage2_wiser_runner import (  # noqa: E402
    WISERTrainingConfig,
    predict_wiser_representation_probes,
    train_wiser_arm,
)
from cvsrffi.stage2_wiser_scoring import score_wiser_predictions  # noqa: E402
from cvsrffi.wiser_source_summary import load_quantized_source_summary  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_phase1_binding(
    checkpoint: Path,
    source_summary: Path,
    binding_path: Path,
) -> Mapping[str, Any]:
    binding = _load_json(binding_path)
    if binding.get("schema") != "cvs.phase1.wiser_rf.source_binding.v1":
        raise ValueError("WISER Phase1 source binding schema drift")
    if _sha256(checkpoint) != binding.get("checkpoint_sha256"):
        raise ValueError("WISER checkpoint/source-summary binding drift")
    if _sha256(source_summary) != binding.get("source_summary_sha256"):
        raise ValueError("WISER source-summary artifact binding drift")
    registry = binding.get("class_registry")
    if (
        binding.get("checkpoint_id") != "ADV3B02_CORE90_SOFT_E200"
        or binding.get("feature_schema") != "ADV3B02:z_id:unit_l2:160:v1"
        or binding.get("feature_dim") != 160
        or not isinstance(registry, list)
        or len(registry) != 6
        or len(set(map(str, registry))) != 6
    ):
        raise ValueError("WISER Phase1 semantic binding drift")
    return binding


def _write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _new_root(path: Path) -> Path:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"immutable output root exists: {path}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def _pilot_job(manifest: Mapping[str, Any], outer_key: str) -> Mapping[str, Any]:
    if manifest.get("protocol_schema") != "p2_min_v1":
        raise ValueError("WISER pilot requires p2_min_v1")
    rows = [row for row in manifest.get("jobs", []) if row.get("outer_key") == outer_key]
    if len(rows) != 1:
        raise ValueError("WISER pilot outer-key coverage drift")
    row = rows[0]
    if row.get("protocol_schema", manifest.get("protocol_schema")) != "p2_min_v1":
        raise ValueError("WISER pilot job protocol binding drift")
    if (
        row.get("phase2_data_status", manifest.get("phase2_data_status"))
        != "VALIDATED_ONCE"
    ):
        raise ValueError("WISER pilot requires VALIDATED_ONCE data")
    if not row.get("capsule_id") or not row.get("split_id"):
        raise ValueError("WISER pilot capsule/split binding missing")
    return row


def _package_root(job: Mapping[str, Any]) -> Path:
    packages = job.get("packages")
    if not isinstance(packages, Mapping):
        raise ValueError("WISER package registry missing")
    before = packages.get("before_enrollment")
    if not isinstance(before, Mapping) or not before.get("package_root"):
        raise ValueError("WISER before-enrollment package root missing")
    return Path(str(before["package_root"]))


def _support_path(job: Mapping[str, Any], scenario: str) -> Path:
    return _package_root(job) / f"support_{scenario}.npz"


def _query_path(job: Mapping[str, Any], scenario: str) -> Path:
    return _package_root(job) / f"query_{scenario}.npz"


def _training_config(args: argparse.Namespace) -> WISERTrainingConfig:
    return WISERTrainingConfig(
        stage_steps=tuple(int(value) for value in args.stage_steps),
        lambda_proto=float(args.lambda_proto),
        lambda_sp=float(args.lambda_sp),
        lambda_vsw=float(args.lambda_vsw),
        lambda_inversion=float(args.lambda_inversion),
        num_vsw_projections=int(args.num_vsw_projections),
        inversion_steps=int(args.inversion_steps),
        inversion_samples_per_class=int(args.inversion_samples_per_class),
        seed=int(args.seed),
    )


def _tensor(values: np.ndarray, device: str, *, labels: bool = False) -> torch.Tensor:
    dtype = torch.long if labels else torch.float32
    contiguous = np.ascontiguousarray(values)
    try:
        return torch.from_numpy(contiguous).to(device=torch.device(device), dtype=dtype)
    except TypeError:
        return torch.tensor(contiguous.tolist(), dtype=dtype, device=torch.device(device))


def _save_adapted_state_new(
    path: Path,
    model: torch.nn.Module,
    audit: Mapping[str, Any],
) -> None:
    changed_names = {
        str(name)
        for stage in audit.get("stage_audits", [])
        for name in stage.get("trainable_parameter_names", [])
    }
    state = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if name in changed_names
    }
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"immutable WISER state exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def _load_adapted_state(path: Path, model: torch.nn.Module, device: str) -> None:
    state = torch.load(path, map_location=torch.device(device))
    if not isinstance(state, Mapping) or not all(
        isinstance(name, str) and torch.is_tensor(value) for name, value in state.items()
    ):
        raise ValueError("WISER adapted state payload drift")
    known = dict(model.named_parameters())
    if not set(state).issubset(known):
        raise ValueError("WISER adapted state parameter registry drift")
    with torch.no_grad():
        for name, value in state.items():
            if known[name].shape != value.shape:
                raise ValueError(f"WISER adapted state shape drift: {name}")
            known[name].copy_(value.to(known[name].device, known[name].dtype))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _smoke(args: argparse.Namespace) -> Mapping[str, Any]:
    destination = _new_root(args.output_root)
    manifest = _load_json(args.manifest)
    job = _pilot_job(manifest, args.pilot_outer_key)
    binding = _validate_phase1_binding(
        args.checkpoint, args.source_summary, args.source_binding
    )
    summary = load_quantized_source_summary(args.source_summary)
    if (
        tuple(summary.class_registry) != tuple(binding["class_registry"])
        or summary.feature_schema != binding["feature_schema"]
        or tuple(summary.centers.shape) != (6, int(binding["feature_dim"]))
    ):
        raise ValueError("WISER loaded summary semantic binding drift")
    support = load_support_package(_support_path(job, args.scenario))
    model = frozen_checkpoint(args.checkpoint, args.device)
    audit = train_wiser_arm(
        model,
        _tensor(support.iq, args.device),
        _tensor(support.labels, args.device, labels=True),
        source_summary=summary,
        arm=args.arm,
        config=_training_config(args),
    )
    result = {
        "schema": "cvs.phase2.wiser_rf.no_query_smoke.v1",
        "status": "PASS",
        "arm": args.arm,
        "scenario": args.scenario,
        "outer_key": job["outer_key"],
        "capsule_id": job["capsule_id"],
        "split_id": job["split_id"],
        "query_opened": False,
        "source_summary_class_count": len(summary.class_registry),
        "source_summary_feature_dim": int(summary.centers.shape[1]),
        "training_audit": asdict(audit),
    }
    _write_json_new(destination / "smoke_result.json", result)
    return result


def _pilot(args: argparse.Namespace) -> Mapping[str, Any]:
    destination = _new_root(args.output_root)
    manifest = _load_json(args.manifest)
    job = _pilot_job(manifest, args.pilot_outer_key)
    binding = _validate_phase1_binding(
        args.checkpoint, args.source_summary, args.source_binding
    )
    summary = load_quantized_source_summary(args.source_summary)
    if (
        tuple(summary.class_registry) != tuple(binding["class_registry"])
        or summary.feature_schema != binding["feature_schema"]
        or tuple(summary.centers.shape) != (6, int(binding["feature_dim"]))
    ):
        raise ValueError("WISER loaded summary semantic binding drift")
    config = _training_config(args)
    support_stage: list[dict[str, Any]] = []
    # Phase 1: freeze every scenario/arm support state before opening any query NPZ.
    for scenario in SCENARIOS:
        support = load_support_package(_support_path(job, scenario))
        support_iq = _tensor(support.iq, args.device)
        support_labels = _tensor(support.labels, args.device, labels=True)
        for arm in ARMS:
            unit = destination / scenario / arm
            unit.mkdir(parents=True, exist_ok=False)
            model = frozen_checkpoint(args.checkpoint, args.device)
            audit: Mapping[str, Any]
            if arm == "B0":
                audit = {
                    "arm": "B0",
                    "optimizer_steps": 0,
                    "query_rows_used": 0,
                    "vsw_enabled": False,
                    "model_inversion_enabled": False,
                    "stage_audits": [],
                    "config": asdict(config),
                }
            else:
                audit = asdict(
                    train_wiser_arm(
                        model,
                        support_iq,
                        support_labels,
                        source_summary=summary,
                        arm=arm,
                        config=config,
                    )
                )
            _save_adapted_state_new(unit / "adapted_state.pt", model, audit)
            _write_json_new(unit / "training_audit.json", audit)
            support_stage.append(
                {
                    "scenario": scenario,
                    "arm": arm,
                    "status": "SUPPORT_STATE_FROZEN",
                    "query_opened": False,
                }
            )

    completed: list[dict[str, Any]] = []
    # Phase 2: all fitting is over; only frozen inference is allowed from here on.
    for scenario in SCENARIOS:
        support = load_support_package(_support_path(job, scenario))
        support_iq = _tensor(support.iq, args.device)
        support_labels = _tensor(support.labels, args.device, labels=True)
        query = load_query_package(_query_path(job, scenario))
        query_iq = _tensor(query.iq, args.device)
        for arm in ARMS:
            unit = destination / scenario / arm
            prediction_root = unit / "prediction"
            prediction_root.mkdir(parents=True, exist_ok=False)
            model = frozen_checkpoint(args.checkpoint, args.device)
            _load_adapted_state(unit / "adapted_state.pt", model, args.device)
            audit = _load_json(unit / "training_audit.json")
            predictions = predict_wiser_representation_probes(
                model,
                support_iq,
                support_labels,
                query_iq,
                query_tokens=query.tokens,
                source_summary=summary,
                seed=int(job.get("seed", args.seed)),
            )
            np.savez_compressed(prediction_root / "predictions.npz", **predictions)
            receipt = {
                "schema": "cvs.phase2.wiser_rf.prediction_receipt.v1",
                "status": "PREDICTIONS_COMPLETE",
                "outer_key": job["outer_key"],
                "receiver": str(job["receiver"]),
                "scenario": scenario,
                "arm": arm,
                "capsule_id": job["capsule_id"],
                "split_id": job["split_id"],
                "query_rows": len(query.tokens),
                "expected_query_tokens": list(query.tokens),
                "query_truth_opened": False,
                "query_role_opened": False,
                "support_state_frozen_before_query": True,
                "formal_protocol_eligible": arm in {"B0", "A", "B"},
                "claim_scope": (
                    "FORMAL_P2_MIN_V1"
                    if arm in {"B0", "A", "B"}
                    else "DIAGNOSTIC_MODEL_INVERSION_NON_FORMAL"
                ),
                "training_audit": audit,
            }
            _write_json_new(prediction_root / "prediction_receipt.json", receipt)
            completed.append(
                {
                    "scenario": scenario,
                    "arm": arm,
                    "query_rows": len(query.tokens),
                    "status": "PREDICTIONS_COMPLETE",
                }
            )
    result = {
        "schema": "cvs.phase2.wiser_rf.pilot.v1",
        "status": "ARTIFACTS_COMPLETE",
        "pilot_outer_key": job["outer_key"],
        "scene_arm_unit_count": len(completed),
        "units": completed,
        "truth_opened": False,
        "scoring_required": True,
    }
    _write_json_new(destination / "pilot_result.json", result)
    return result


def _score_pilot(args: argparse.Namespace) -> Mapping[str, Any]:
    destination = _new_root(args.output_root)
    manifest = _load_json(args.manifest)
    job = _pilot_job(manifest, args.pilot_outer_key)
    truth = Path(str(job["truth_sidecar"]))
    rows: list[Mapping[str, Any]] = []
    for scenario in SCENARIOS:
        for arm in ARMS:
            source = args.prediction_root / scenario / arm / "prediction"
            score = score_wiser_predictions(
                source / "predictions.npz",
                source / "prediction_receipt.json",
                truth,
            )
            output = destination / scenario / arm / "score.json"
            _write_json_new(output, score)
            rows.append(score)
    decisions = {
        arm: formal_promotion_decision(rows, arm=arm) for arm in ("A", "B")
    }
    best_formal_arm = next(
        (arm for arm in ("B", "A") if decisions[arm]["passed"]), None
    )
    result = {
        "schema": "cvs.phase2.wiser_rf.score_collection.v1",
        "status": "ANALYZED",
        "scene_arm_unit_count": len(rows),
        "rows": rows,
        "formal_decisions": decisions,
        "best_formal_arm": best_formal_arm,
        "next_experiment_authorized": best_formal_arm is not None,
        "c_diagnostic_rows_used_for_promotion": 0,
        "truth_join_after_prediction_only": True,
    }
    _write_json_new(destination / "score_collection.json", result)
    return result


def _add_common(command: argparse.ArgumentParser) -> None:
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument(
        "--pilot-outer-key", default="rx_3_19__seed_713102__k_10__new_5"
    )


def _add_training(command: argparse.ArgumentParser) -> None:
    _add_common(command)
    command.add_argument("--checkpoint", type=Path, required=True)
    command.add_argument("--source-summary", type=Path, required=True)
    command.add_argument("--source-binding", type=Path, required=True)
    command.add_argument("--output-root", type=Path, required=True)
    command.add_argument("--device", required=True)
    command.add_argument("--stage-steps", type=int, nargs=3, default=(1500, 2500, 4000))
    command.add_argument("--lambda-proto", type=float, default=0.5)
    command.add_argument("--lambda-sp", type=float, default=1.0)
    command.add_argument("--lambda-vsw", type=float, default=0.5)
    command.add_argument("--lambda-inversion", type=float, default=0.25)
    command.add_argument("--num-vsw-projections", type=int, default=32)
    command.add_argument("--inversion-steps", type=int, default=300)
    command.add_argument("--inversion-samples-per-class", type=int, default=2)
    command.add_argument("--seed", type=int, default=713102)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke")
    _add_training(smoke)
    smoke.set_defaults(stage_steps=(1, 1, 1), inversion_steps=1)
    smoke.add_argument("--arm", choices=("A", "B", "C", "ABC"), default="ABC")
    smoke.add_argument("--scenario", choices=SCENARIOS, default=SCENARIOS[0])
    pilot = commands.add_parser("pilot")
    _add_training(pilot)
    score = commands.add_parser("score-pilot")
    _add_common(score)
    score.add_argument("--prediction-root", type=Path, required=True)
    score.add_argument("--output-root", type=Path, required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    result = {
        "smoke": _smoke,
        "pilot": _pilot,
        "score-pilot": _score_pilot,
    }[args.command](args)
    print(json.dumps(dict(result), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
