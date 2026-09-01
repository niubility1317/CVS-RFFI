#!/usr/bin/env python3
"""Run MARC-OT no-query smoke, frozen pilot, or independent score."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from statistics import median
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.meta_weight_bank import BlockSpec, parameter_block_key  # noqa: E402
from cvsrffi.marc_ot_support_features import (  # noqa: E402
    MARC_OT_SUPPORT_FEATURE_CONFIG,
    MARC_OT_SUPPORT_ROW_DIM,
    MARC_OT_SUPPORT_ROW_SCHEMA,
    build_marc_ot_support_features,
)
from cvsrffi.meta_weight_bank_checkpoint import load_meta_weight_bundle  # noqa: E402
from cvsrffi.meta_weight_calibrator import calibrate_weight_plan  # noqa: E402
from cvsrffi.stage2_bisage_runner import frozen_checkpoint  # noqa: E402
from cvsrffi.stage2_marc_ot_pilot import (  # noqa: E402
    FORMAL_ARMS,
    SCENARIOS,
    load_query_package,
    load_support_package,
    run_support_then_query,
    validate_manifest_job,
    validate_pilot_config,
)
from cvsrffi.stage2_marc_ot_runner import (  # noqa: E402
    MARCOTRunnerConfig,
    predict_marc_ot_probes,
    train_marc_ot_arm,
)
from cvsrffi.stage2_marc_ot_scoring import (  # noqa: E402
    compare_marc_ot_score_rows,
    load_marc_ot_truth,
    preflight_marc_ot_prediction,
    score_preflighted_marc_ot_prediction,
)


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return value


def _json_feature_config() -> Mapping[str, Any]:
    return json.loads(json.dumps(dict(MARC_OT_SUPPORT_FEATURE_CONFIG)))


def _validate_support_feature_binding(value: Any) -> Mapping[str, Any]:
    expected_config = _json_feature_config()
    if not isinstance(value, Mapping) or set(value) != {"schema", "dim", "config"}:
        raise ValueError("support feature binding field set drift")
    if value["schema"] != MARC_OT_SUPPORT_ROW_SCHEMA:
        raise ValueError("support feature schema mismatch")
    if value["dim"] != MARC_OT_SUPPORT_ROW_DIM:
        raise ValueError("support feature dimension mismatch")
    if not isinstance(value["config"], Mapping) or dict(value["config"]) != dict(
        expected_config
    ):
        raise ValueError("support feature config mismatch")
    return {
        "schema": MARC_OT_SUPPORT_ROW_SCHEMA,
        "dim": MARC_OT_SUPPORT_ROW_DIM,
        "config": dict(expected_config),
    }


def _validate_config_payload(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or "support_feature" not in value:
        raise ValueError("support feature binding is required")
    feature_binding = _validate_support_feature_binding(value["support_feature"])
    legacy_payload = dict(value)
    legacy_payload.pop("support_feature")
    validated = dict(validate_pilot_config(legacy_payload))
    validated["support_feature"] = feature_binding
    return validated


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def create_immutable_output_root(path: str | Path) -> Path:
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"immutable output root exists: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def _tensor(values: np.ndarray, device: str, *, labels: bool = False) -> torch.Tensor:
    dtype = torch.long if labels else torch.float32
    contiguous = np.ascontiguousarray(values)
    target = torch.device(device)
    try:
        return torch.from_numpy(contiguous).to(device=target, dtype=dtype)
    except TypeError:
        if contiguous.size > 10_000_000:
            raise ValueError("NumPy bridge fallback exceeds bounded tensor size")
        return torch.tensor(contiguous.tolist(), device=target, dtype=dtype)


def _peak_rss_bytes() -> int | None:
    """Return the process peak resident set using only the standard library."""

    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024
    except (ImportError, AttributeError, OSError, ValueError):
        pass
    if sys.platform != "win32":
        return None
    try:
        import ctypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = (
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            )

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        measured = ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        )
        return int(counters.PeakWorkingSetSize) if measured else None
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _resource_receipt(
    *,
    training_seconds: float,
    inference_seconds: float,
    peak_rss_bytes: int | None,
    peak_cuda_bytes: int | None,
    peak_cuda_status: str,
    trainable_parameter_count: int,
) -> Mapping[str, Any]:
    rss_status = "MEASURED" if peak_rss_bytes is not None else "UNAVAILABLE"
    return {
        "training_seconds": float(training_seconds),
        "inference_seconds": float(inference_seconds),
        "peak_rss_bytes": int(peak_rss_bytes) if peak_rss_bytes is not None else "N/A",
        "peak_cuda_bytes": int(peak_cuda_bytes) if peak_cuda_bytes is not None else "N/A",
        "peak_rss_status": rss_status,
        "peak_cuda_status": str(peak_cuda_status),
        "trainable_parameter_count": int(trainable_parameter_count),
    }


def _runner_config(config: Mapping[str, Any]) -> MARCOTRunnerConfig:
    return MARCOTRunnerConfig(
        fold_count=int(config["fold_count"]),
        stage_steps=tuple(int(value) for value in config["stage_steps"]),
        learning_rate_min=float(config["learning_rate_bounds"]["min"]),
        learning_rate_max=float(config["learning_rate_bounds"]["max"]),
        ot_epsilon=float(config["ot"]["epsilon"]),
        ot_iterations=int(config["ot"]["iterations"]),
        ratio_cap=float(config["ratio_cap"]),
        interpolation_grid=tuple(float(value) for value in config["interpolation_grid"]),
        seed=int(config["seed"]),
    )


def _package_root(job: Mapping[str, Any], name: str) -> Path:
    packages = job.get("packages")
    if not isinstance(packages, Mapping):
        raise ValueError("MARC-OT package registry is missing")
    row = packages.get(name)
    if not isinstance(row, Mapping) or not isinstance(row.get("package_root"), str):
        raise ValueError(f"MARC-OT {name} package root is missing")
    return Path(row["package_root"])


def _support_path(job: Mapping[str, Any], scenario: str) -> Path:
    return _package_root(job, "before_enrollment") / f"support_{scenario}.npz"


def _query_path(job: Mapping[str, Any], scenario: str) -> Path:
    return _package_root(job, "before_apply") / f"query_{scenario}.npz"


def _expected_block_specs(model: torch.nn.Module) -> tuple[BlockSpec, ...]:
    grouped: dict[str, list[tuple[str, torch.Tensor]]] = {}
    for name, value in model.named_parameters():
        block = parameter_block_key(name)
        if block is not None:
            grouped.setdefault(block, []).append((name, value))
    if not grouped:
        raise ValueError("ADV3B02 checkpoint has no canonical MARC-OT blocks")
    return tuple(
        BlockSpec(
            name=block,
            parameter_names=tuple(name for name, _ in sorted(rows)),
            shapes=tuple(tuple(value.shape) for _, value in sorted(rows)),
            dtypes=tuple(str(value.dtype) for _, value in sorted(rows)),
        )
        for block, rows in sorted(grouped.items())
    )


def _validate_checkpoint_identity(path: Path, expected_checkpoint_id: str) -> str:
    """Fail closed unless the checkpoint embeds the configured method identity."""

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"cannot load checkpoint identity: {path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint identity root must be a mapping")
    args = payload.get("args")
    sources = (payload, args) if isinstance(args, Mapping) else (payload,)
    identities = {
        str(source[key])
        for source in sources
        for key in ("candidate_id", "checkpoint_id", "method_id")
        if isinstance(source.get(key), str) and str(source[key]).strip()
    }
    if len(identities) != 1 or identities != {str(expected_checkpoint_id)}:
        raise ValueError("checkpoint embedded identity does not match config checkpoint_id")
    return next(iter(identities))


def _load_model_and_bundle(args: argparse.Namespace, config: Mapping[str, Any]):
    _validate_checkpoint_identity(args.checkpoint, str(config["checkpoint_id"]))
    model = frozen_checkpoint(args.checkpoint, args.device)
    base_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    bundle = load_meta_weight_bundle(
        args.bundle,
        expected_base_checkpoint_id=str(config["checkpoint_id"]),
        base_state=base_state,
        expected_block_specs=_expected_block_specs(model),
    )
    feature_binding = _validate_support_feature_binding(config.get("support_feature"))
    if (
        bundle.feature_schema != feature_binding["schema"]
        or bundle.feature_dim != feature_binding["dim"]
        or json.loads(json.dumps(dict(bundle.feature_config)))
        != dict(feature_binding["config"])
    ):
        raise ValueError("bundle/config support feature ABI mismatch")
    bundle.support_encoder.to(torch.device(args.device)).eval()
    for parameter in bundle.support_encoder.parameters():
        parameter.requires_grad_(False)
    return model, bundle, base_state


def _rows_per_class(labels: torch.Tensor) -> int:
    _, counts = torch.unique(labels, sorted=True, return_counts=True)
    if counts.numel() == 0 or bool((counts != counts[0]).any()):
        raise ValueError("MARC-OT support class K mismatch before feature building")
    return int(counts[0].item())


def _build_support_features(
    model: torch.nn.Module,
    values: torch.Tensor,
    labels: torch.Tensor,
    tokens: tuple[str, ...],
    *,
    nominal_k: int,
):
    row_k = _rows_per_class(labels)
    return build_marc_ot_support_features(
        model,
        values,
        labels,
        tokens,
        nominal_k=nominal_k,
        effective_mask=(
            None
            if row_k == nominal_k
            else torch.ones(len(labels), device=values.device, dtype=values.dtype)
        ),
        validated_unpadded=row_k == nominal_k,
    )


def _bank_task_features(bundle: Any, device: torch.device) -> torch.Tensor:
    task_count = len(bundle.bank.task_keys)
    members = []
    for entry in bundle.bank.entries:
        coefficients = entry.task_coefficients.detach().to(device=device, dtype=torch.float32)
        if coefficients.shape != (task_count, entry.effective_rank):
            raise ValueError("MARC-OT bank task coefficient geometry drift")
        members.append(coefficients)
    result = torch.cat(members, dim=1)
    if result.shape[1] == 0 or not bool(torch.isfinite(result).all()):
        raise ValueError("MARC-OT bank task coefficient space is empty or nonfinite")
    return result


def _calibration_transform(
    audit_sink: list[Mapping[str, Any]], *, nominal_k: int
):
    def transform(
        model: torch.nn.Module,
        values: torch.Tensor,
        labels: torch.Tensor,
        tokens: tuple[str, ...],
    ):
        built = _build_support_features(
            model, values, labels, tokens, nominal_k=nominal_k
        )
        audit_sink.append(dict(built.audit))
        return built.rows

    return transform


def _adapt_unit(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    support: Any,
    arm: str,
    *,
    smoke: bool,
) -> Mapping[str, Any]:
    unit_device = torch.device(args.device)
    if unit_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(unit_device)
    model, bundle, base_state = _load_model_and_bundle(args, config)
    support_iq = _tensor(support.iq, args.device)
    support_labels = _tensor(support.labels, args.device, labels=True)
    full_support_plans: list[Any] = []
    scoped_plans: dict[tuple[str, tuple[str, ...]], Any] = {}
    support_feature_audits: list[Mapping[str, Any]] = []

    def plan_for_fit(
        fit_iq: torch.Tensor,
        fit_labels: torch.Tensor,
        fit_tokens: tuple[str, ...],
        fit_scope: str,
    ) -> Any:
        key = (str(fit_scope), tuple(fit_tokens))
        if key in scoped_plans:
            return scoped_plans[key]
        previous_training = bool(model.training)
        try:
            model.eval()
            built = _build_support_features(
                model,
                fit_iq,
                fit_labels,
                fit_tokens,
                nominal_k=int(config["k_shot"]),
            )
            support_feature_audits.append(dict(built.audit))
            support_state = bundle.support_encoder(
                built.rows,
                built.labels,
                built.physical_tokens,
                built.effective_mask,
            )
            plan = calibrate_weight_plan(
                base_state,
                str(config["checkpoint_id"]),
                bundle.bank,
                support_state,
                lr_min=float(config["learning_rate_bounds"]["min"]),
                lr_max=float(config["learning_rate_bounds"]["max"]),
            )
        finally:
            model.train(previous_training)
        if fit_scope == "full_support":
            full_support_plans.append(plan)
        scoped_plans[key] = plan
        return plan

    def initial_state_factory(
        fit_iq: torch.Tensor,
        fit_labels: torch.Tensor,
        fit_tokens: tuple[str, ...],
        fit_scope: str,
    ) -> Mapping[str, torch.Tensor]:
        plan = plan_for_fit(fit_iq, fit_labels, fit_tokens, fit_scope)
        return plan.state_dict

    def block_learning_rate_factory(
        fit_iq: torch.Tensor,
        fit_labels: torch.Tensor,
        fit_tokens: tuple[str, ...],
        fit_scope: str,
    ) -> Mapping[str, float]:
        plan = plan_for_fit(fit_iq, fit_labels, fit_tokens, fit_scope)
        if len(plan.block_lrs) != len(bundle.bank.entries):
            raise ValueError("fit-scope MARC-OT block learning-rate geometry drift")
        return {
            entry.spec.name: float(plan.block_lrs[index])
            for index, entry in enumerate(bundle.bank.entries)
        }

    runner = _runner_config(config)
    if smoke:
        runner = replace(runner, stage_steps=(1, 1, 1, 1))
    audit = train_marc_ot_arm(
        model,
        support_iq,
        support_labels,
        support.tokens,
        arm=arm,
        config=runner,
        bank_task_features=(
            _bank_task_features(bundle, support_iq.device) if arm in {"R6", "R8"} else None
        ),
        calibration_feature_transform=(
            _calibration_transform(
                support_feature_audits, nominal_k=int(config["k_shot"])
            )
            if arm in {"R2", "R4", "R6", "R8"}
            else None
        ),
        initial_state_factory=(
            initial_state_factory if arm in {"R4", "R6", "R8"} else None
        ),
        block_learning_rate_factory=(
            block_learning_rate_factory if arm in {"R4", "R6", "R8"} else None
        ),
    )
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    reached = set(audit.reached_parameter_names)
    trainable_count = sum(
        int(parameter.numel()) for name, parameter in model.named_parameters() if name in reached
    )
    if full_support_plans:
        plan = full_support_plans[0]
        bank_initialization = {
            "applied": bool(plan.applied),
            "reason": str(plan.reason),
            "uncertainty": float(plan.uncertainty),
            "block_gates": list(plan.block_gates),
            "block_lrs": list(plan.block_lrs),
        }
    else:
        bank_initialization = {
            "applied": False,
            "reason": (
                "SUPPORT_CROSSFIT_REJECTED"
                if arm in {"R4", "R6", "R8"}
                else "NOT_APPLICABLE"
            ),
            "uncertainty": None,
            "block_gates": [],
            "block_lrs": [],
        }
    return {
        "model_state": state,
        "audit": asdict(audit),
        "trainable_parameter_count": trainable_count,
        "peak_rss_bytes": _peak_rss_bytes(),
        "bank_initialization": bank_initialization,
        "support_feature_abi": {
            "schema": MARC_OT_SUPPORT_ROW_SCHEMA,
            "dim": MARC_OT_SUPPORT_ROW_DIM,
            "config": _json_feature_config(),
        },
        "support_feature_audits": support_feature_audits,
    }


def _save_frozen_unit(destination: Path, scenario: str, arm: str, state: Mapping[str, Any]) -> None:
    unit = destination / scenario / arm
    unit.mkdir(parents=True, exist_ok=False)
    model_path = unit / "support_frozen_state.pt"
    if model_path.exists() or model_path.is_symlink():
        raise FileExistsError(f"immutable support state exists: {model_path}")
    torch.save(state["model_state"], model_path)
    _write_json_new(
        unit / "support_state_receipt.json",
        {
            "schema": "cvs.phase2.marc_ot.support_state_receipt.v1",
            "status": "SUPPORT_STATE_FROZEN",
            "scenario": scenario,
            "arm": arm,
            "query_opened": False,
            "query_rows_used": 0,
            "training_audit": state["audit"],
            "bank_initialization": state["bank_initialization"],
            "support_feature_abi": state["support_feature_abi"],
            "support_feature_audits": state["support_feature_audits"],
            "trainable_parameter_count": state["trainable_parameter_count"],
        },
    )


def _predict_unit(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    job: Mapping[str, Any],
    destination: Path,
    scenario: str,
    arm: str,
    support: Any,
    query: Any,
    state: Mapping[str, Any],
) -> None:
    unit_device = torch.device(args.device)
    if unit_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(unit_device)
    model = frozen_checkpoint(args.checkpoint, args.device)
    model.load_state_dict(state["model_state"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    class_count = len(set(int(value) for value in support.labels.tolist()))
    registry = tuple(str(index) for index in range(class_count))
    started = time.perf_counter()
    predictions = predict_marc_ot_probes(
        model,
        _tensor(support.iq, args.device),
        _tensor(support.labels, args.device, labels=True),
        _tensor(query.iq, args.device),
        support_tokens=support.tokens,
        query_tokens=query.tokens,
        class_registry=registry,
        seed=int(config["seed"]),
        batch_size=int(args.batch_size),
    )
    inference_seconds = float(time.perf_counter() - started)
    inference_cuda_peak = (
        int(torch.cuda.max_memory_allocated(unit_device))
        if unit_device.type == "cuda"
        else None
    )
    prediction_root = destination / scenario / arm / "prediction"
    prediction_root.mkdir(parents=True, exist_ok=False)
    prediction_path = prediction_root / "predictions.npz"
    if prediction_path.exists() or prediction_path.is_symlink():
        raise FileExistsError(f"immutable prediction exists: {prediction_path}")
    np.savez_compressed(prediction_path, **predictions)
    audit = state["audit"]
    _write_json_new(
        prediction_root / "prediction_receipt.json",
        {
            "schema": "cvs.phase2.marc_ot.prediction_receipt.v1",
            "status": "PREDICTIONS_COMPLETE",
            "protocol_schema": "p2_min_v1",
            "phase2_data_status": "VALIDATED_ONCE",
            "outer_key": str(job["outer_key"]),
            "capsule_id": str(job["capsule_id"]),
            "split_id": str(job["split_id"]),
            "receiver": str(job["receiver"]),
            "scenario": scenario,
            "arm": arm,
            "query_rows": len(query.tokens),
            "expected_query_tokens": list(query.tokens),
            "class_registry": list(registry),
            "query_truth_opened": False,
            "query_role_opened": False,
            "support_state_frozen_before_query": True,
            "query_decision_policy": "per_sample_all_registered_classes",
            "resources": _resource_receipt(
                training_seconds=float(audit["training_seconds"]),
                inference_seconds=inference_seconds,
                peak_rss_bytes=_peak_rss_bytes(),
                peak_cuda_bytes=(
                    max(int(audit["peak_cuda_bytes"] or 0), inference_cuda_peak or 0)
                    if unit_device.type == "cuda"
                    else None
                ),
                peak_cuda_status=(
                    "MEASURED" if unit_device.type == "cuda" else "NOT_APPLICABLE"
                ),
                trainable_parameter_count=int(state["trainable_parameter_count"]),
            ),
        },
    )


def _context(args: argparse.Namespace):
    config = _validate_config_payload(_load_json(args.config))
    manifest = _load_json(args.manifest)
    job = validate_manifest_job(
        manifest, outer_key=str(config["pilot_outer_key"]), config=config
    )
    return config, job


def _smoke(args: argparse.Namespace) -> Mapping[str, Any]:
    destination = create_immutable_output_root(args.output_root)
    config, job = _context(args)
    support = load_support_package(_support_path(job, args.scenario))
    state = _adapt_unit(args, config, support, args.arm, smoke=True)
    result = {
        "schema": "cvs.phase2.marc_ot.no_query_smoke.v1",
        "status": "PASS",
        "arm": args.arm,
        "scenario": args.scenario,
        "outer_key": str(job["outer_key"]),
        "capsule_id": str(job["capsule_id"]),
        "split_id": str(job["split_id"]),
        "query_opened": False,
        "query_rows_used": 0,
        "training_audit": state["audit"],
        "support_feature_abi": state["support_feature_abi"],
        "support_feature_audits": state["support_feature_audits"],
    }
    _write_json_new(destination / "smoke_result.json", result)
    return result


def _pilot(args: argparse.Namespace) -> Mapping[str, Any]:
    destination = create_immutable_output_root(args.output_root)
    config, job = _context(args)

    def support_loader(scenario: str):
        return load_support_package(_support_path(job, scenario))

    def adapt(scenario: str, arm: str, support: Any):
        del scenario
        return _adapt_unit(args, config, support, arm, smoke=False)

    def write_state(scenario: str, arm: str, state: Mapping[str, Any]):
        _save_frozen_unit(destination, scenario, arm, state)

    def query_loader(scenario: str):
        return load_query_package(_query_path(job, scenario))

    def predict(scenario: str, arm: str, support: Any, query: Any, state: Mapping[str, Any]):
        _predict_unit(args, config, job, destination, scenario, arm, support, query, state)

    lifecycle = run_support_then_query(
        scenarios=SCENARIOS,
        arms=FORMAL_ARMS,
        support_loader=support_loader,
        adapt_and_freeze=adapt,
        support_state_writer=write_state,
        query_loader=query_loader,
        predict_and_write=predict,
    )
    result = {
        **lifecycle,
        "pilot_outer_key": str(job["outer_key"]),
        "capsule_id": str(job["capsule_id"]),
        "split_id": str(job["split_id"]),
        "receiver": str(job["receiver"]),
        "promotion_gates": dict(config["promotion_gates"]),
        "scoring_required": True,
    }
    _write_json_new(destination / "pilot_result.json", result)
    return result


def _promotion_decision(
    paired_rows: list[Mapping[str, Any]], gates: Mapping[str, Any], arm: str
) -> Mapping[str, Any]:
    selected = [row for row in paired_rows if row["candidate_arm"] == arm]
    if len(selected) != len(SCENARIOS):
        raise ValueError("MARC-OT promotion requires three-scene paired evidence")
    p3_ba = [row["probes"]["P3_OLD_D92"]["balanced_accuracy_delta_pp"] for row in selected]
    p3_floor = [row["probes"]["P3_OLD_D92"]["floor_delta_pp"] for row in selected]
    p1_p2 = [
        row["probes"][probe]["balanced_accuracy_delta_pp"]
        for row in selected
        for probe in ("P1_SOURCE_HEAD", "P2_SUPPORT_PROTOTYPE")
    ]
    help_scenes = sum(
        row["probes"]["P3_OLD_D92"]["help_count"]
        > row["probes"]["P3_OLD_D92"]["harm_count"]
        for row in selected
    )
    low_elev = next(row for row in selected if row["scenario"] == "leo_low_elev_weak")
    checks = {
        "median_p3_ba": median(p3_ba) >= float(gates["median_p3_ba_delta_pp"]),
        "worst_scene_p3_ba": min(p3_ba) >= float(gates["worst_scene_p3_ba_delta_pp"]),
        "median_p3_floor": median(p3_floor) >= float(gates["median_p3_floor_delta_pp"]),
        "low_elev_p3_floor": low_elev["probes"]["P3_OLD_D92"]["floor_delta_pp"]
        >= float(gates["low_elev_p3_floor_delta_pp"]),
        "p1_p2_scene_drop": min(p1_p2) >= -float(gates["max_p1_p2_scene_drop_pp"]),
        "help_gt_harm_scenes": help_scenes >= int(gates["minimum_help_gt_harm_scenes"]),
    }
    passed = all(checks.values())
    return {
        "arm": arm,
        "status": "PROMOTE_TO_TARGET25" if passed else "NO_PROMOTION_TO_TARGET25",
        "passed": passed,
        "gates": checks,
    }


def _score(args: argparse.Namespace) -> Mapping[str, Any]:
    destination = create_immutable_output_root(args.output_root)
    pilot = _load_json(args.prediction_root / "pilot_result.json")
    if (
        pilot.get("status") != "ARTIFACTS_COMPLETE"
        or pilot.get("support_frozen_unit_count") != len(SCENARIOS) * len(FORMAL_ARMS)
        or pilot.get("prediction_unit_count") != len(SCENARIOS) * len(FORMAL_ARMS)
        or tuple(pilot.get("arms", ())) != FORMAL_ARMS
        or tuple(pilot.get("scenarios", ())) != SCENARIOS
        or pilot.get("truth_opened") is not False
    ):
        raise ValueError("MARC-OT prediction root is incomplete")
    preflighted = {
        (scenario, arm): preflight_marc_ot_prediction(
            args.prediction_root / scenario / arm / "prediction"
        )
        for scenario in SCENARIOS
        for arm in FORMAL_ARMS
    }
    truth_payload = load_marc_ot_truth(args.truth_sidecar)
    rows: list[Mapping[str, Any]] = []
    paired: list[Mapping[str, Any]] = []
    for scenario in SCENARIOS:
        scores: dict[str, Mapping[str, Any]] = {}
        for arm in FORMAL_ARMS:
            score = score_preflighted_marc_ot_prediction(
                preflighted[(scenario, arm)], truth_payload
            )
            scores[arm] = score
            rows.append(score)
            _write_json_new(destination / scenario / arm / "score.json", score)
        for arm in FORMAL_ARMS[1:]:
            comparison = compare_marc_ot_score_rows(scores["R0"], scores[arm])
            paired.append(comparison)
            _write_json_new(destination / scenario / arm / "paired_vs_r0.json", comparison)
    gates = pilot.get("promotion_gates")
    if not isinstance(gates, Mapping):
        raise ValueError("MARC-OT frozen promotion gates are missing")
    decisions = {arm: _promotion_decision(paired, gates, arm) for arm in FORMAL_ARMS[1:]}
    promotable = [arm for arm in FORMAL_ARMS[1:] if decisions[arm]["passed"]]
    result = {
        "schema": "cvs.phase2.marc_ot.score_collection.v1",
        "status": "ANALYZED",
        "rows": rows,
        "paired_rows": paired,
        "decisions": decisions,
        "best_promotable_arm": promotable[-1] if promotable else None,
        "next_state": "PROMOTE_TO_TARGET25" if promotable else "NO_PROMOTION_TO_TARGET25",
        "truth_join_after_prediction_only": True,
    }
    _write_json_new(destination / "score_collection.json", result)
    return result


def _add_execution(command: argparse.ArgumentParser) -> None:
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--checkpoint", type=Path, required=True)
    command.add_argument("--bundle", type=Path, required=True)
    command.add_argument("--output-root", type=Path, required=True)
    command.add_argument("--device", required=True)
    command.add_argument("--batch-size", type=int, default=128)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke", help="one support-only no-query smoke")
    _add_execution(smoke)
    smoke.add_argument("--arm", choices=FORMAL_ARMS, default="R8")
    smoke.add_argument("--scenario", choices=SCENARIOS, default=SCENARIOS[0])
    pilot = commands.add_parser("pilot", help="freeze all support states, then predict")
    _add_execution(pilot)
    score = commands.add_parser("score", help="independent truth-last scoring")
    score.add_argument("--prediction-root", type=Path, required=True)
    score.add_argument("--truth-sidecar", type=Path, required=True)
    score.add_argument("--output-root", type=Path, required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    result = {"smoke": _smoke, "pilot": _pilot, "score": _score}[args.command](args)
    print(json.dumps(dict(result), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
