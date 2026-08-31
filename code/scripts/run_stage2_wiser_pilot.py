#!/usr/bin/env python3
"""Run the truth-blind WISER-RF A/B/C historical pilot and score it later."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
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
    P3_ARMS,
    SCENARIOS,
    formal_p3_primary_decision,
    formal_promotion_decision,
    load_query_package,
    load_support_package,
    normalize_p3_arms,
    select_p3_primary_champion,
)
from cvsrffi.stage2_wiser_runner import (  # noqa: E402
    WISERP3TrainingConfig,
    WISERTrainingConfig,
    predict_wiser_representation_probes,
    train_wiser_arm,
    train_wiser_p3_arm,
)
from cvsrffi.stage2_wiser_scoring import (  # noqa: E402
    compare_wiser_score_rows,
    score_wiser_predictions,
)
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


def _p3_runtime_identity(
    *,
    runtime_commit: str,
    p3_config: Path,
    checkpoint: Path,
    source_summary: Path,
    source_binding: Path,
    job: Mapping[str, Any],
    checkpoint_id: str,
) -> dict[str, Any]:
    """Bind a P3 prediction root to the exact runtime bytes and data row."""

    commit = str(runtime_commit).strip()
    if not commit:
        raise ValueError("P3 runtime commit is required")
    for path, label in ((p3_config, "P3 config"), (checkpoint, "checkpoint"), (source_summary, "source summary"), (source_binding, "source binding")):
        if not path.is_file():
            raise ValueError(f"P3 {label} is missing")
    result = {
        "runtime_commit": commit,
        "p3_config_sha256": _sha256(p3_config),
        "checkpoint_id": str(checkpoint_id),
        "checkpoint_sha256": _sha256(checkpoint),
        "source_summary_sha256": _sha256(source_summary),
        "source_binding_sha256": _sha256(source_binding),
    }
    for field in ("outer_key", "capsule_id", "split_id", "receiver", "seed", "k_shot", "new_class_count"):
        if field not in job or job[field] in (None, ""):
            raise ValueError(f"P3 runtime identity {field} is missing")
        result[field] = job[field]
    return result


def _validate_p3_runtime_identity(
    identity: Any, *, runtime_commit: str, p3_config: Path, job: Mapping[str, Any]
) -> Mapping[str, Any]:
    if not isinstance(identity, Mapping):
        raise ValueError("P3 runtime identity is missing")
    if str(runtime_commit).strip() != identity.get("runtime_commit"):
        raise ValueError("P3 runtime commit identity drift")
    if not p3_config.is_file() or _sha256(p3_config) != identity.get("p3_config_sha256"):
        raise ValueError("P3 config runtime identity drift")
    for field in ("outer_key", "capsule_id", "split_id", "receiver", "seed", "k_shot", "new_class_count"):
        if identity.get(field) != job.get(field):
            raise ValueError("P3 runtime data binding drift")
    for field in ("checkpoint_id", "checkpoint_sha256", "source_summary_sha256", "source_binding_sha256"):
        if not isinstance(identity.get(field), str) or not identity[field]:
            raise ValueError("P3 runtime artifact identity drift")
    return identity


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


def _normalize_arms(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    selected = tuple(str(value).upper() for value in values)
    if not selected or selected[0] != "B0" or "B0" not in selected:
        raise ValueError("WISER bounded arm subset must start with B0")
    if len(set(selected)) != len(selected) or any(value not in ARMS for value in selected):
        raise ValueError("WISER bounded arm subset is invalid")
    if len(selected) < 2:
        raise ValueError("WISER bounded arm subset needs one candidate after B0")
    return selected


def _frozen_pilot_arms(
    prediction_root: Path,
    requested: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    pilot_result = _load_json(prediction_root / "pilot_result.json")
    frozen = _normalize_arms(pilot_result.get("arms", ()))
    if requested is not None and _normalize_arms(requested) != frozen:
        raise ValueError("WISER score-pilot arm registry mismatch")
    return frozen


def _package_root(job: Mapping[str, Any], package_name: str) -> Path:
    packages = job.get("packages")
    if not isinstance(packages, Mapping):
        raise ValueError("WISER package registry missing")
    package = packages.get(package_name)
    if not isinstance(package, Mapping) or not package.get("package_root"):
        raise ValueError(f"WISER {package_name} package root missing")
    return Path(str(package["package_root"]))


def _support_path(job: Mapping[str, Any], scenario: str) -> Path:
    return _package_root(job, "before_enrollment") / f"support_{scenario}.npz"


def _query_path(job: Mapping[str, Any], scenario: str) -> Path:
    return _package_root(job, "before_apply") / f"query_{scenario}.npz"


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


_P3_CONFIG_SCHEMA = "cvs.phase2.wiser_rf.p3_primary.config.v1"
_P3_PRIMARY_IDENTITY = {
    "protocol_schema": "p2_min_v1",
    "phase2_data_status": "VALIDATED_ONCE",
    "pilot_outer_key": "rx_3_19__seed_713102__k_10__new_5",
    "query_policy": "full_package_read_only_after_support_freeze",
    "capsule_id": "d92-e0-full-target125:5910674066e8bbf93684fddd6af6fd2cef7e8f208d64e403ac7e58030a2a8cc5",
    "split_id": "d92-e0-full-target125:rx_3_19__seed_713102__k_10__new_5",
    "receiver": "3-19",
    "seed": 713102,
    "k_shot": 10,
    "new_class_count": 5,
    "checkpoint_id": "ADV3B02_CORE90_SOFT_E200",
}
_P3_CONFIG_KEYS = frozenset(
    {
        "schema", "protocol_schema", "phase2_data_status", "pilot_outer_key",
        "arms", "scenarios", "fold_count", "query_policy", "capsule_id", "split_id",
        "receiver", "seed", "k_shot", "new_class_count", "checkpoint_id",
        "source_binding", "n1_training", "p3_training",
    }
)


def _default_p3_config_payload() -> dict[str, Any]:
    """Return the complete frozen P3 configuration shape for strict validation."""

    return {
        "schema": _P3_CONFIG_SCHEMA,
        **_P3_PRIMARY_IDENTITY,
        "arms": list(P3_ARMS),
        "scenarios": list(SCENARIOS),
        "fold_count": 5,
        "source_binding": {
            "schema": "cvs.phase1.wiser_rf.source_binding.v1",
            "checkpoint_id": _P3_PRIMARY_IDENTITY["checkpoint_id"],
            "feature_schema": "ADV3B02:z_id:unit_l2:160:v1",
            "feature_dim": 160,
            "class_registry": ["14-10", "14-7", "20-15", "20-19", "6-15", "8-20"],
        },
        "n1_training": asdict(WISERTrainingConfig(seed=_P3_PRIMARY_IDENTITY["seed"])),
        "p3_training": asdict(WISERP3TrainingConfig(seed=_P3_PRIMARY_IDENTITY["seed"])),
    }


def _strict_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"P3 config {field} must be an integer")
    return int(value)


def _strict_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"P3 config {field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"P3 config {field} must be finite")
    return result


def _strict_training_mapping(
    payload: Any, reference: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"P3 config {label} must be an object")
    keys = set(payload)
    expected = set(reference)
    if keys != expected:
        missing = sorted(expected - keys)
        unknown = sorted(keys - expected)
        if unknown:
            raise ValueError(f"P3 config {label} has unknown keys: {', '.join(unknown)}")
        raise ValueError(f"P3 config {label} is missing keys: {', '.join(missing)}")
    result: dict[str, Any] = {}
    for key, default in reference.items():
        value = payload[key]
        if isinstance(default, tuple):
            if not isinstance(value, list) or len(value) != len(default):
                raise ValueError(f"P3 config {label}.{key} must be a fixed-length list")
            converted = []
            for index, item in enumerate(value):
                if isinstance(default[index], int) and not isinstance(default[index], bool):
                    converted.append(_strict_int(item, f"{label}.{key}"))
                else:
                    converted.append(_strict_float(item, f"{label}.{key}"))
            result[key] = tuple(converted)
        elif isinstance(default, int) and not isinstance(default, bool):
            result[key] = _strict_int(value, f"{label}.{key}")
        elif isinstance(default, float):
            result[key] = _strict_float(value, f"{label}.{key}")
        else:
            raise ValueError(f"P3 config {label}.{key} has unsupported schema")
    return result


def _load_p3_config(path: Path) -> Mapping[str, Any]:
    payload = _load_json(path)
    keys = set(payload)
    if keys != set(_P3_CONFIG_KEYS):
        unknown = sorted(keys - _P3_CONFIG_KEYS)
        if unknown:
            raise ValueError(f"P3 config has unknown keys: {', '.join(unknown)}")
        raise ValueError(f"P3 config is missing keys: {', '.join(sorted(_P3_CONFIG_KEYS - keys))}")
    template = _default_p3_config_payload()
    for field in ("schema", "protocol_schema", "phase2_data_status", "pilot_outer_key", "query_policy", "capsule_id", "split_id", "receiver", "checkpoint_id"):
        if not isinstance(payload[field], str) or not payload[field]:
            raise ValueError(f"P3 config {field} must be a nonempty string")
    if payload["schema"] != _P3_CONFIG_SCHEMA:
        raise ValueError("P3 config schema drift")
    for field, expected in _P3_PRIMARY_IDENTITY.items():
        if payload[field] != expected:
            raise ValueError(f"P3 config fixed identity drift: {field}")
    if not isinstance(payload["arms"], list) or tuple(payload["arms"]) != P3_ARMS:
        raise ValueError("P3 config arm registry drift")
    if not isinstance(payload["scenarios"], list) or tuple(payload["scenarios"]) != SCENARIOS:
        raise ValueError("P3 config scenario registry drift")
    if _strict_int(payload["fold_count"], "fold_count") != 5:
        raise ValueError("P3 config fold count drift")
    for field, expected in (("seed", 713102), ("k_shot", 10), ("new_class_count", 5)):
        if _strict_int(payload[field], field) != expected:
            raise ValueError(f"P3 config {field} drift")
    source_binding = payload["source_binding"]
    if not isinstance(source_binding, Mapping) or set(source_binding) != set(template["source_binding"]):
        raise ValueError("P3 config source binding schema drift")
    if (
        source_binding.get("schema") != "cvs.phase1.wiser_rf.source_binding.v1"
        or source_binding.get("checkpoint_id") != _P3_PRIMARY_IDENTITY["checkpoint_id"]
        or source_binding.get("feature_schema") != "ADV3B02:z_id:unit_l2:160:v1"
        or _strict_int(source_binding.get("feature_dim"), "source_binding.feature_dim") != 160
        or not isinstance(source_binding.get("class_registry"), list)
        or len(source_binding["class_registry"]) != 6
        or len(set(map(str, source_binding["class_registry"]))) != 6
    ):
        raise ValueError("P3 config source binding drift")
    n1 = _strict_training_mapping(payload["n1_training"], template["n1_training"], label="n1_training")
    p3 = _strict_training_mapping(payload["p3_training"], template["p3_training"], label="p3_training")
    if n1["seed"] != payload["seed"] or p3["seed"] != payload["seed"] or p3["fold_count"] != payload["fold_count"]:
        raise ValueError("P3 config nested training binding drift")
    return {**payload, "n1_training": n1, "p3_training": p3}


def _validate_p3_job_binding(
    config: Mapping[str, Any], job: Mapping[str, Any], binding: Mapping[str, Any]
) -> None:
    for config_field, job_field in (
        ("pilot_outer_key", "outer_key"),
        ("capsule_id", "capsule_id"),
        ("split_id", "split_id"),
        ("receiver", "receiver"),
        ("seed", "seed"),
        ("k_shot", "k_shot"),
        ("new_class_count", "new_class_count"),
    ):
        if job_field not in job or config[config_field] != job[job_field]:
            raise ValueError(f"P3 config/manifest {config_field} binding drift")
    source = config["source_binding"]
    if (
        config["checkpoint_id"] != binding.get("checkpoint_id")
        or source["checkpoint_id"] != binding.get("checkpoint_id")
        or source["feature_schema"] != binding.get("feature_schema")
        or source["feature_dim"] != binding.get("feature_dim")
        or tuple(source["class_registry"]) != tuple(binding.get("class_registry", ()))
    ):
        raise ValueError("P3 config/Phase1 source binding drift")


def _p3_n1_training_config(config: Mapping[str, Any]) -> WISERTrainingConfig:
    return WISERTrainingConfig(**dict(config["n1_training"]))


def _p3_training_config(config: Mapping[str, Any]) -> WISERP3TrainingConfig:
    return WISERP3TrainingConfig(**dict(config["p3_training"]))


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
    arms = _normalize_arms(getattr(args, "arms", ARMS))
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
        for arm in arms:
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
        for arm in arms:
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
        "arms": list(arms),
        "scene_arm_unit_count": len(completed),
        "units": completed,
        "truth_opened": False,
        "scoring_required": True,
    }
    _write_json_new(destination / "pilot_result.json", result)
    return result


def _score_pilot(args: argparse.Namespace) -> Mapping[str, Any]:
    arms = _frozen_pilot_arms(
        args.prediction_root,
        getattr(args, "arms", None),
    )
    destination = _new_root(args.output_root)
    manifest = _load_json(args.manifest)
    job = _pilot_job(manifest, args.pilot_outer_key)
    truth = Path(str(job["truth_sidecar"]))
    rows: list[Mapping[str, Any]] = []
    for scenario in SCENARIOS:
        for arm in arms:
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
        arm: formal_promotion_decision(rows, arm=arm)
        for arm in ("A", "B")
        if arm in arms
    }
    best_formal_arm = next(
        (arm for arm in ("B", "A") if arm in decisions and decisions[arm]["passed"]),
        None,
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


def _p3_context(args: argparse.Namespace) -> tuple[Mapping[str, Any], Mapping[str, Any], Any, Mapping[str, Any]]:
    """Validate all P3 bindings before a mutable output directory exists."""

    config = _load_p3_config(args.p3_config)
    manifest = _load_json(args.manifest)
    job = _pilot_job(manifest, config["pilot_outer_key"])
    if args.pilot_outer_key != config["pilot_outer_key"]:
        raise ValueError("P3 CLI/config pilot outer binding drift")
    binding = _validate_phase1_binding(args.checkpoint, args.source_summary, args.source_binding)
    _validate_p3_job_binding(config, job, binding)
    summary = load_quantized_source_summary(args.source_summary)
    if (
        tuple(summary.class_registry) != tuple(binding["class_registry"])
        or summary.feature_schema != binding["feature_schema"]
        or tuple(summary.centers.shape) != (6, int(binding["feature_dim"]))
    ):
        raise ValueError("P3 loaded source summary semantic binding drift")
    return config, job, summary, binding


def _p3_n0_audit(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "outer_arm": "N0",
        "trainer_arm": None,
        "optimizer_steps": 0,
        "query_rows_used": 0,
        "stage_audits": [],
        "support_state_frozen": True,
        "config": {"p3_training": dict(config["p3_training"])},
    }


def _p3_train_one(
    arm: str,
    model: torch.nn.Module,
    support: WISERSupportPackage,
    *,
    device: str,
    summary: Any,
    binding: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    if arm == "N0":
        return _p3_n0_audit(config)
    support_iq = _tensor(support.iq, device)
    support_labels = _tensor(support.labels, device, labels=True)
    if arm == "N1":
        audit = asdict(
            train_wiser_arm(
                model, support_iq, support_labels, source_summary=summary,
                arm="A", config=_p3_n1_training_config(config),
            )
        )
        return {**audit, "outer_arm": "N1", "trainer_arm": "A", "support_state_frozen": True}
    audit = asdict(
        train_wiser_p3_arm(
            model, support_iq, support_labels, support_tokens=support.tokens,
            source_summary=summary,
            expected_source_class_registry=tuple(binding["class_registry"]),
            expected_source_feature_schema=str(binding["feature_schema"]),
            arm=arm, config=_p3_training_config(config),
        )
    )
    return {**audit, "outer_arm": arm, "trainer_arm": arm, "support_state_frozen": True}


def _p3_smoke(args: argparse.Namespace) -> Mapping[str, Any]:
    config, job, summary, binding = _p3_context(args)
    arm = str(args.arm).upper()
    if arm not in P3_ARMS:
        raise ValueError("P3 smoke arm registry drift")
    destination = _new_root(args.output_root)
    support = load_support_package(_support_path(job, args.scenario))
    model = frozen_checkpoint(args.checkpoint, args.device)
    audit = _p3_train_one(
        arm, model, support, device=args.device, summary=summary, binding=binding, config=config
    )
    result = {
        "schema": "cvs.phase2.wiser_rf.p3_primary.no_query_smoke.v1",
        "status": "PASS", "outer_key": job["outer_key"], "capsule_id": job["capsule_id"],
        "split_id": job["split_id"], "receiver": job["receiver"], "scenario": args.scenario,
        "arm": arm, "query_opened": False, "query_rows_used": 0,
        "support_state_frozen_before_query": True, "training_audit": audit,
    }
    _write_json_new(destination / "smoke_result.json", result)
    return result


def _p3_pilot(args: argparse.Namespace) -> Mapping[str, Any]:
    config, job, summary, binding = _p3_context(args)
    arms = normalize_p3_arms(tuple(getattr(args, "arms", P3_ARMS)))
    runtime_identity = _p3_runtime_identity(
        runtime_commit=args.runtime_commit, p3_config=args.p3_config, checkpoint=args.checkpoint,
        source_summary=args.source_summary, source_binding=args.source_binding, job=job,
        checkpoint_id=str(binding["checkpoint_id"]),
    )
    destination = _new_root(args.output_root)
    support_cache: dict[str, WISERSupportPackage] = {}
    frozen_models: dict[tuple[str, str], torch.nn.Module] = {}
    support_units: list[dict[str, Any]] = []
    # Persist every support-only state before loading any query package.
    for scenario in SCENARIOS:
        support = load_support_package(_support_path(job, scenario))
        support_cache[scenario] = support
        for arm in arms:
            unit = destination / scenario / arm
            unit.mkdir(parents=True, exist_ok=False)
            model = frozen_checkpoint(args.checkpoint, args.device)
            audit = _p3_train_one(
                arm, model, support, device=args.device, summary=summary, binding=binding, config=config
            )
            if int(audit.get("query_rows_used", -1)) != 0:
                raise ValueError("P3 support training used query rows")
            _save_adapted_state_new(unit / "adapted_state.pt", model, audit)
            _write_json_new(unit / "training_audit.json", audit)
            frozen_models[(scenario, arm)] = model
            support_units.append(
                {"scenario": scenario, "arm": arm, "status": "SUPPORT_STATE_FROZEN", "query_opened": False}
            )
    support_audit = {
        "schema": "cvs.phase2.wiser_rf.p3_primary.support_audit.v1",
        "status": "SUPPORT_STATES_COMPLETE", "outer_key": job["outer_key"],
        "capsule_id": job["capsule_id"], "split_id": job["split_id"],
        "receiver": job["receiver"], "arms": list(arms), "scenarios": list(SCENARIOS),
        "expected_scene_arm_unit_count": len(SCENARIOS) * len(arms),
        "units": support_units, "query_opened": False,
        "all_support_states_frozen": len(support_units) == len(SCENARIOS) * len(arms),
        "runtime_identity": runtime_identity,
    }
    _write_json_new(destination / "support_audit.json", support_audit)
    if not support_audit["all_support_states_frozen"]:
        raise RuntimeError("P3 root support audit is incomplete")

    completed: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        support = support_cache[scenario]
        support_iq = _tensor(support.iq, args.device)
        support_labels = _tensor(support.labels, args.device, labels=True)
        query = load_query_package(_query_path(job, scenario))
        query_iq = _tensor(query.iq, args.device)
        for arm in arms:
            unit = destination / scenario / arm
            prediction_root = unit / "prediction"
            prediction_root.mkdir(parents=True, exist_ok=False)
            model = frozen_models[(scenario, arm)]
            _load_adapted_state(unit / "adapted_state.pt", model, args.device)
            audit = _load_json(unit / "training_audit.json")
            predictions = predict_wiser_representation_probes(
                model, support_iq, support_labels, query_iq, query_tokens=query.tokens,
                source_summary=summary, seed=int(job["seed"]),
            )
            np.savez_compressed(prediction_root / "predictions.npz", **predictions)
            receipt = {
                "schema": "cvs.phase2.wiser_rf.p3_primary.prediction_receipt.v1",
                "status": "PREDICTIONS_COMPLETE", "outer_key": job["outer_key"],
                "capsule_id": job["capsule_id"], "split_id": job["split_id"],
                "receiver": job["receiver"], "scenario": scenario, "arm": arm,
                "query_rows": len(query.tokens), "expected_query_tokens": list(query.tokens),
                "support_audit_reference": "support_audit.json", "query_truth_opened": False,
                "query_role_opened": False, "support_state_frozen_before_query": True,
                "training_audit": audit, "runtime_identity": runtime_identity,
            }
            _write_json_new(prediction_root / "prediction_receipt.json", receipt)
            completed.append({"scenario": scenario, "arm": arm, "status": "PREDICTIONS_COMPLETE", "query_rows": len(query.tokens)})
    result = {
        "schema": "cvs.phase2.wiser_rf.p3_primary.pilot.v1", "status": "ARTIFACTS_COMPLETE",
        "pilot_outer_key": job["outer_key"], "arms": list(arms), "scenarios": list(SCENARIOS),
        "scene_arm_unit_count": len(completed), "units": completed,
        "support_audit_reference": "support_audit.json", "truth_opened": False, "scoring_required": True,
        "runtime_identity": runtime_identity,
    }
    _write_json_new(destination / "pilot_result.json", result)
    return result


def _p3_frozen_pilot_arms(prediction_root: Path, requested: Any) -> tuple[str, ...]:
    pilot = _load_json(prediction_root / "pilot_result.json")
    if pilot.get("schema") != "cvs.phase2.wiser_rf.p3_primary.pilot.v1" or pilot.get("status") != "ARTIFACTS_COMPLETE":
        raise ValueError("P3 score-pilot root schema/status drift")
    frozen = normalize_p3_arms(tuple(pilot.get("arms", ())))
    if requested is not None and normalize_p3_arms(tuple(requested)) != frozen:
        raise ValueError("P3 score-pilot arm registry mismatch")
    return frozen


def _p3_validate_prediction_registry(
    prediction_root: Path, *, arms: tuple[str, ...], job: Mapping[str, Any], runtime_identity: Mapping[str, Any]
) -> None:
    support_audit = _load_json(prediction_root / "support_audit.json")
    if support_audit.get("schema") != "cvs.phase2.wiser_rf.p3_primary.support_audit.v1" or support_audit.get("all_support_states_frozen") is not True:
        raise ValueError("P3 support audit is not complete")
    for field in ("outer_key", "capsule_id", "split_id", "receiver"):
        if support_audit.get(field) != job[field]:
            raise ValueError("P3 support audit binding drift")
    if support_audit.get("runtime_identity") != runtime_identity:
        raise ValueError("P3 support audit runtime identity drift")
    if support_audit.get("arms") != list(arms) or support_audit.get("scenarios") != list(SCENARIOS):
        raise ValueError("P3 support audit registry drift")
    if int(support_audit.get("expected_scene_arm_unit_count", -1)) != len(SCENARIOS) * len(arms):
        raise ValueError("P3 support audit unit coverage drift")
    units = support_audit.get("units")
    if not isinstance(units, list):
        raise ValueError("P3 support audit units are missing")
    observed_units = {
        (str(unit.get("scenario")), str(unit.get("arm")))
        for unit in units if isinstance(unit, Mapping)
        and unit.get("status") == "SUPPORT_STATE_FROZEN" and unit.get("query_opened") is False
    }
    expected_units = {(scenario, arm) for scenario in SCENARIOS for arm in arms}
    if len(units) != len(expected_units) or observed_units != expected_units:
        raise ValueError("P3 support audit unit coverage drift")
    for scenario in SCENARIOS:
        for arm in arms:
            source = prediction_root / scenario / arm / "prediction"
            if not (source / "predictions.npz").is_file() or not (source / "prediction_receipt.json").is_file():
                raise ValueError("P3 prediction registry is incomplete")
            receipt = _load_json(source / "prediction_receipt.json")
            if receipt.get("schema") != "cvs.phase2.wiser_rf.p3_primary.prediction_receipt.v1" or receipt.get("status") != "PREDICTIONS_COMPLETE":
                raise ValueError("P3 prediction receipt schema/status drift")
            for field, expected in (("outer_key", job["outer_key"]), ("capsule_id", job["capsule_id"]), ("split_id", job["split_id"]), ("receiver", job["receiver"]), ("scenario", scenario), ("arm", arm)):
                if receipt.get(field) != expected:
                    raise ValueError("P3 prediction receipt binding drift")
            if receipt.get("query_truth_opened") is not False or receipt.get("query_role_opened") is not False or receipt.get("support_state_frozen_before_query") is not True or receipt.get("support_audit_reference") != "support_audit.json":
                raise ValueError("P3 prediction receipt truth-last drift")
            if receipt.get("runtime_identity") != runtime_identity:
                raise ValueError("P3 prediction receipt runtime identity drift")
            _p3_unit_training_audit(prediction_root / scenario / arm, receipt, arm=arm)
            _p3_validate_prediction_npz(source / "predictions.npz", receipt)


def _p3_validate_prediction_npz(path: Path, receipt: Mapping[str, Any]) -> None:
    """Validate one complete truth-blind P3 prediction NPZ without opening truth."""

    required = {
        "query_tokens", "query_z_id",
        "p1_predictions", "p1_logits",
        "p2_predictions", "p2_logits",
        "p3_predictions", "p3_logits",
    }
    try:
        with np.load(path, allow_pickle=False) as arrays:
            if set(arrays.files) != required:
                raise ValueError("P3 prediction NPZ member registry drift")
            expected = receipt.get("expected_query_tokens")
            if not isinstance(expected, list):
                raise ValueError("P3 prediction token registry is missing")
            expected_tokens = tuple(str(token) for token in expected)
            row_count = _strict_int(receipt.get("query_rows"), "prediction_receipt.query_rows")
            tokens = np.asarray(arrays["query_tokens"]).astype(str)
            if (
                tokens.ndim != 1 or len(tokens) != row_count
                or tuple(tokens.tolist()) != expected_tokens
                or len(set(expected_tokens)) != len(expected_tokens)
                or len(set(tokens.tolist())) != len(tokens)
            ):
                raise ValueError("P3 prediction token registry drift")
            identity = np.asarray(arrays["query_z_id"], dtype=np.float64)
            if identity.shape != (row_count, 160) or not np.isfinite(identity).all():
                raise ValueError("P3 prediction identity feature drift")
            for prefix in ("p1", "p2", "p3"):
                prediction = np.asarray(arrays[f"{prefix}_predictions"])
                if (
                    prediction.ndim != 1 or prediction.shape[0] != row_count
                    or not np.issubdtype(prediction.dtype, np.integer)
                ):
                    raise ValueError("P3 prediction index geometry drift")
                indices = prediction.astype(np.int64, copy=False)
                if bool(((indices < 0) | (indices >= 6)).any()):
                    raise ValueError("P3 prediction index range drift")
                logits = np.asarray(arrays[f"{prefix}_logits"], dtype=np.float64)
                if logits.shape != (row_count, 6) or not np.isfinite(logits).all():
                    raise ValueError("P3 prediction logit geometry drift")
                if not np.array_equal(indices, logits.argmax(axis=1).astype(np.int64)):
                    raise ValueError("P3 prediction/logit argmax drift")
    except OSError as error:
        raise ValueError("P3 prediction NPZ cannot be opened") from error


def _p3_unit_training_audit(
    unit: Path, receipt: Mapping[str, Any], *, arm: str
) -> Mapping[str, Any]:
    """Read the frozen unit audit and reject a receipt/audit identity mismatch."""

    audit = _load_json(unit / "training_audit.json")
    if receipt.get("training_audit") != audit:
        raise ValueError("P3 receipt/training audit binding drift")
    if audit.get("outer_arm") != arm or _strict_int(audit.get("query_rows_used"), "training_audit.query_rows_used") != 0:
        raise ValueError("P3 unit training audit binding drift")
    expected_trainer = None if arm == "N0" else "A" if arm == "N1" else arm
    if audit.get("trainer_arm") != expected_trainer or audit.get("support_state_frozen") is not True:
        raise ValueError("P3 unit training audit binding drift")
    if arm in {"N2", "N3", "N4", "N5", "N6"}:
        for field in (
            "baseline_joint_condition_number",
            "final_joint_condition_number",
            "final_zero_identity_count",
        ):
            if field not in audit:
                raise ValueError("P3 candidate training audit diagnostics are missing")
        baseline = _strict_float(audit["baseline_joint_condition_number"], "training_audit.baseline_joint_condition_number")
        final = _strict_float(audit["final_joint_condition_number"], "training_audit.final_joint_condition_number")
        zero_count = _strict_int(audit["final_zero_identity_count"], "training_audit.final_zero_identity_count")
        if baseline <= 0.0 or final < 0.0 or zero_count < 0:
            raise ValueError("P3 candidate training audit diagnostics are invalid")
    return audit


def _p3_score_pilot(args: argparse.Namespace) -> Mapping[str, Any]:
    config = _load_p3_config(args.p3_config)
    manifest = _load_json(args.manifest)
    job = _pilot_job(manifest, config["pilot_outer_key"])
    if args.pilot_outer_key != config["pilot_outer_key"]:
        raise ValueError("P3 CLI/config pilot outer binding drift")
    _validate_p3_job_binding(config, job, config["source_binding"])
    arms = _p3_frozen_pilot_arms(args.prediction_root, getattr(args, "arms", None))
    pilot = _load_json(args.prediction_root / "pilot_result.json")
    runtime_identity = _validate_p3_runtime_identity(
        pilot.get("runtime_identity"), runtime_commit=args.runtime_commit,
        p3_config=args.p3_config, job=job,
    )
    _p3_validate_prediction_registry(args.prediction_root, arms=arms, job=job, runtime_identity=runtime_identity)
    destination = _new_root(args.output_root)
    truth = Path(str(job["truth_sidecar"]))
    scores: dict[tuple[str, str], Mapping[str, Any]] = {}
    for scenario in SCENARIOS:
        for arm in arms:
            source = args.prediction_root / scenario / arm / "prediction"
            score = score_wiser_predictions(source / "predictions.npz", source / "prediction_receipt.json", truth)
            _write_json_new(destination / scenario / arm / "score.json", score)
            scores[(scenario, arm)] = score
    paired: list[Mapping[str, Any]] = []
    for scenario in SCENARIOS:
        for arm in arms:
            if arm == "N0":
                continue
            comparison = dict(compare_wiser_score_rows(scores[(scenario, "N0")], scores[(scenario, arm)]))
            receipt = _load_json(args.prediction_root / scenario / arm / "prediction" / "prediction_receipt.json")
            audit = _p3_unit_training_audit(args.prediction_root / scenario / arm, receipt, arm=arm)
            comparison["candidate_training_audit"] = audit
            baseline_receipt = _load_json(args.prediction_root / scenario / "N0" / "prediction" / "prediction_receipt.json")
            comparison["baseline_training_audit"] = _p3_unit_training_audit(
                args.prediction_root / scenario / "N0", baseline_receipt, arm="N0"
            )
            paired.append(comparison)
    decisions = {arm: formal_p3_primary_decision(paired, arm=arm) for arm in arms if arm != "N0"}
    champion = select_p3_primary_champion(decisions)
    result = {
        "schema": "cvs.phase2.wiser_rf.p3_primary.score_collection.v1", "status": "ANALYZED",
        "scene_arm_unit_count": len(scores), "rows": list(scores.values()), "paired_rows": paired,
        "formal_p3_primary_decisions": decisions, "p3_primary_champion": champion,
        "full_target25_authorized": champion is not None, "truth_join_after_prediction_only": True,
        "champion_identity": ({"arm": champion, **dict(runtime_identity)} if champion is not None else None),
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


def _add_p3_training(command: argparse.ArgumentParser) -> None:
    _add_common(command)
    command.add_argument("--p3-config", type=Path, required=True)
    command.add_argument("--checkpoint", type=Path, required=True)
    command.add_argument("--source-summary", type=Path, required=True)
    command.add_argument("--source-binding", type=Path, required=True)
    command.add_argument("--output-root", type=Path, required=True)
    command.add_argument("--device", required=True)
    command.add_argument("--runtime-commit", required=True)


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
    pilot.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
    score = commands.add_parser("score-pilot")
    _add_common(score)
    score.add_argument("--prediction-root", type=Path, required=True)
    score.add_argument("--output-root", type=Path, required=True)
    score.add_argument("--arms", nargs="+", choices=ARMS)
    p3_smoke = commands.add_parser("p3-smoke")
    _add_p3_training(p3_smoke)
    p3_smoke.add_argument("--arm", choices=P3_ARMS, default="N6")
    p3_smoke.add_argument("--scenario", choices=SCENARIOS, default=SCENARIOS[0])
    p3_pilot = commands.add_parser("p3-pilot")
    _add_p3_training(p3_pilot)
    p3_pilot.add_argument("--arms", nargs="+", choices=P3_ARMS, default=P3_ARMS)
    p3_score = commands.add_parser("p3-score-pilot")
    _add_common(p3_score)
    p3_score.add_argument("--p3-config", type=Path, required=True)
    p3_score.add_argument("--prediction-root", type=Path, required=True)
    p3_score.add_argument("--output-root", type=Path, required=True)
    p3_score.add_argument("--arms", nargs="+", choices=P3_ARMS)
    p3_score.add_argument("--runtime-commit", required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    result = {
        "smoke": _smoke,
        "pilot": _pilot,
        "score-pilot": _score_pilot,
        "p3-smoke": _p3_smoke,
        "p3-pilot": _p3_pilot,
        "p3-score-pilot": _p3_score_pilot,
    }[args.command](args)
    print(json.dumps(dict(result), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
