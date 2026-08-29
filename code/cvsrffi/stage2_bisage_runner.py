"""Truth-blind BiSAGE-D92 job execution and independent truth-last scoring."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import torch

from cvsrffi.stage2_binova_features import (
    BiNOVAQuery,
    BiNOVASupport,
    extract_binova_features,
)
from cvsrffi.stage2_bisage_da import (
    SAGEDConfig,
    SAGEDModule,
    SAGEDState,
    evaluate_sage_d_crossfit,
    fit_sage_d,
    stage_a_gate,
)
from cvsrffi.stage2_bisage_lifecycle import (
    evaluate_registered_modes,
    freeze_bisage_support_states,
    predict_bisage_query_read_only,
)
from cvsrffi.stage2_bisage_reg import SAGERConfig, fit_sage_r
from cvsrffi.target_only_progressive_runner import _default_checkpoint_loader


class BiSAGERunnerError(ValueError):
    """Raised when a package, lifecycle, or truth-last binding drifts."""


FOUR_STATES = ("DA0_REG0", "DA1_REG0", "DA0_REG1", "DA1_REG1")
SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")


def joint_stage_a_gate(receipts: Mapping[str, Mapping[str, Any]]) -> bool:
    if tuple(receipts) != SCENARIOS:
        raise BiSAGERunnerError("joint Stage A gate requires all three ordered scenarios")
    return all(
        row.get("scenario") == scenario
        and row.get("gate", {}).get("stage_a_gate_passed") is True
        for scenario, row in receipts.items()
    )


def _write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def frozen_checkpoint(path: str | Path, device: str | torch.device) -> torch.nn.Module:
    model = _default_checkpoint_loader(path, device=device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _context(job: Mapping[str, Any]) -> dict[str, str]:
    required = ("protocol_schema", "phase2_data_status", "capsule_id", "split_id")
    if any(not str(job.get(key, "")).strip() for key in required):
        raise BiSAGERunnerError("job Phase2 binding is incomplete")
    return {key: str(job[key]) for key in required}


def _package_root(job: Mapping[str, Any], stage: str) -> Path:
    packages = job.get("packages")
    row = packages.get(stage) if isinstance(packages, Mapping) else None
    root = row.get("package_root") if isinstance(row, Mapping) else None
    if not isinstance(root, str) or not root:
        raise BiSAGERunnerError(f"sealed package root missing: {stage}")
    return Path(root)


def _load_npz_subset(path: Path, required: frozenset[str], forbidden: tuple[str, ...]) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            names = frozenset(payload.files)
            if not required.issubset(names):
                raise BiSAGERunnerError(f"package members missing: {path}")
            lowered = tuple(name.lower() for name in names)
            if any(token in name for name in lowered for token in forbidden):
                raise BiSAGERunnerError(f"forbidden package member exposed: {path}")
            return {name: np.asarray(payload[name]) for name in required}
    except (OSError, ValueError) as exc:
        raise BiSAGERunnerError(f"cannot load sealed package: {path}") from exc


def load_support(
    job: Mapping[str, Any], scenario: str, stage: str, model: torch.nn.Module,
    device: str | torch.device,
) -> BiNOVASupport:
    if scenario not in SCENARIOS or stage not in {"before_enrollment", "after_enrollment"}:
        raise BiSAGERunnerError("support scenario or registration state drift")
    path = _package_root(job, stage) / f"support_{scenario}.npz"
    data = _load_npz_subset(
        path,
        frozenset({
            "support_leo_weak_iq", "support_class_indices",
            "support_rank_within_class", "support_tokens",
        }),
        ("query", "truth", "role", "quota"),
    )
    labels = np.asarray(data["support_class_indices"], dtype=np.int64)
    ranks = np.asarray(data["support_rank_within_class"], dtype=np.int64)
    tokens = tuple(np.asarray(data["support_tokens"]).astype(str).tolist())
    features = extract_binova_features(
        model,
        np.asarray(data["support_leo_weak_iq"], dtype=np.float32),
        physical_ids=tokens,
        device=device,
    )
    return BiNOVASupport(features=features, labels=labels, ranks=ranks, context=_context(job))


def load_query(
    job: Mapping[str, Any], scenario: str, model: torch.nn.Module,
    device: str | torch.device,
) -> BiNOVAQuery:
    if scenario not in SCENARIOS:
        raise BiSAGERunnerError("query scenario drift")
    path = _package_root(job, "after_apply") / f"query_{scenario}.npz"
    data = _load_npz_subset(
        path,
        frozenset({"query_leo_weak_iq", "query_tokens"}),
        ("label", "truth", "role", "quota", "class_count"),
    )
    tokens = tuple(np.asarray(data["query_tokens"]).astype(str).tolist())
    features = extract_binova_features(
        model,
        np.asarray(data["query_leo_weak_iq"], dtype=np.float32),
        physical_ids=tokens,
        device=device,
    )
    return BiNOVAQuery(features=features, context=_context(job))


def class_handle_registry(job: Mapping[str, Any]) -> dict[int, str]:
    path = _package_root(job, "after_enrollment") / "package_manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload["registered_classes"]
        result = {int(row["class_index"]): str(row["class_handle"]) for row in rows}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise BiSAGERunnerError("cannot load registered opaque class handles") from exc
    expected = set(range(6 + int(job["new_class_count"])))
    if set(result) != expected or len(set(result.values())) != len(result):
        raise BiSAGERunnerError("registered class handle coverage drift")
    return result


def save_stage_a(path: Path, state: SAGEDState, metrics: Mapping[str, Any], gate: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "cvs.phase2.bisage_d92.stage_a.v1",
        "module_state": {name: value.detach().cpu() for name, value in state.module.state_dict().items()},
        "config": asdict(state.config),
        "domain_context166": state.domain_context166,
        "audit": dict(state.audit),
        "metrics": dict(metrics),
        "gate": _jsonable(gate),
    }
    with path.open("xb") as handle:
        torch.save(payload, handle)


def load_stage_a(path: Path, device: str | torch.device) -> tuple[SAGEDState, Mapping[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != "cvs.phase2.bisage_d92.stage_a.v1":
        raise BiSAGERunnerError("Stage A bundle schema drift")
    config = SAGEDConfig(**payload["config"])
    module = SAGEDModule(config.late_rank, config.identity_rank, config.context_dim)
    module.load_state_dict(payload["module_state"], strict=True)
    module.to(device=device, dtype=torch.float64).eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    state = SAGEDState(
        module=module,
        config=config,
        domain_context166=payload["domain_context166"],
        audit=payload["audit"],
    )
    return state, MappingProxyType(dict(payload["gate"]))


def adapt_stage_a(
    job: Mapping[str, Any], scenario: str, model: torch.nn.Module,
    output_root: Path, device: str | torch.device, *, steps: int,
) -> Mapping[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"Stage A output root exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    support = load_support(job, scenario, "before_enrollment", model, device)
    state = fit_sage_d(
        support,
        SAGEDConfig(steps=int(steps), seed=int(job["seed"])),
        device=device,
    )
    if state.audit.get("selected_mode") == "S0":
        metrics: Mapping[str, Any] = MappingProxyType({"query_rows_used": 0, "k1_fallback": True})
        gate: Mapping[str, Any] = MappingProxyType({
            "stage_a_gate_passed": False,
            "status": "LOW_K_S0_FALLBACK",
            "checks": MappingProxyType({}),
        })
    else:
        metrics = evaluate_sage_d_crossfit(state, support, device=device)
        gate = stage_a_gate(metrics)
    save_stage_a(output_root / "stage_a.pt", state, metrics, gate)
    receipt = {
        "schema": "cvs.phase2.bisage_d92.stage_a.receipt.v1",
        "status": str(gate["status"]),
        "outer_key": job["outer_key"],
        "receiver": job["receiver"],
        "scenario": scenario,
        "metrics": _jsonable(metrics),
        "gate": _jsonable(gate),
        "query_rows_used": 0,
        "actual_new_class_rows_used": 0,
    }
    _write_json_new(output_root / "stage_a_receipt.json", receipt)
    return receipt


def _save_predictions(
    path: Path, query: BiNOVAQuery, predictions: Mapping[str, Mapping[str, Any]],
    handles: Mapping[int, str],
) -> None:
    payload: dict[str, np.ndarray] = {"query_tokens": np.asarray(query.features.physical_ids)}
    for state, row in predictions.items():
        prefix = state.lower()
        class_ids = np.asarray(row["class_ids"], dtype=np.int64)
        predicted = np.asarray(row["predictions"], dtype=np.int64)
        payload[f"{prefix}_class_ids"] = class_ids
        payload[f"{prefix}_class_handles"] = np.asarray([handles[int(value)] for value in class_ids])
        payload[f"{prefix}_logits"] = np.asarray(row["logits"], dtype=np.float32)
        payload[f"{prefix}_predictions"] = predicted
        payload[f"{prefix}_prediction_handles"] = np.asarray([handles[int(value)] for value in predicted])
    with path.open("xb") as handle:
        np.savez(handle, **payload)


def build_prediction_receipt(
    job: Mapping[str, Any], scenario: str, *, query_rows: int, selected_mode: str,
    stage_b_ran: bool, mode_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "cvs.phase2.bisage_d92.prediction.v1",
        "status": "PREDICTIONS_COMPLETE",
        "outer_key": job["outer_key"],
        "receiver": job["receiver"],
        "scenario": scenario,
        "capsule_id": job["capsule_id"],
        "split_id": job["split_id"],
        "k_shot": int(job["k_shot"]),
        "old_class_count": 6,
        "new_class_count": int(job["new_class_count"]),
        "query_rows": int(query_rows),
        "selected_mode": selected_mode,
        "stage_b_ran": bool(stage_b_ran),
        "support_mode_metrics": _jsonable(mode_metrics),
        "four_states": list(FOUR_STATES),
        "support_states_frozen_before_query_open": True,
        "query_truth_opened": False,
        "query_role_opened": False,
        "query_fit_update_selection": False,
    }


def adapt_stage_b_and_predict(
    job: Mapping[str, Any], scenario: str, model: torch.nn.Module,
    stage_a_root: Path, output_root: Path, device: str | torch.device, *,
    steps: int, enable_stage_b: bool,
) -> Mapping[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"prediction output root exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    stage_a, gate = load_stage_a(stage_a_root / "stage_a.pt", device)
    old_support = load_support(job, scenario, "before_enrollment", model, device)
    registered = load_support(job, scenario, "after_enrollment", model, device)
    stage_b = None
    mode_metrics: Mapping[str, Any] = MappingProxyType({"selected_mode": "S0", "query_rows_used": 0})
    if enable_stage_b:
        if gate.get("stage_a_gate_passed") is not True:
            raise BiSAGERunnerError("Stage B requested without a passing Stage A gate")
        stage_b = fit_sage_r(
            stage_a,
            registered,
            old_class_count=6,
            config=SAGERConfig(steps=int(steps), seed=int(job["seed"])),
            device=device,
        )
        mode_metrics = evaluate_registered_modes(stage_a, stage_b, registered, old_class_count=6)
    selected_mode = str(mode_metrics["selected_mode"])
    frozen = freeze_bisage_support_states(
        old_support,
        registered,
        stage_a=stage_a,
        stage_b=stage_b,
        selected_mode=selected_mode,
        seed=int(job["seed"]),
        device=device,
    )
    query = load_query(job, scenario, model, device)
    predictions = predict_bisage_query_read_only(frozen, query)
    handles = class_handle_registry(job)
    _save_predictions(output_root / "predictions.npz", query, predictions, handles)
    if stage_b is not None:
        with (output_root / "stage_b.pt").open("xb") as handle:
            torch.save({
                "schema": "cvs.phase2.bisage_d92.stage_b.v1",
                "module_state": {name: value.detach().cpu() for name, value in stage_b.module.state_dict().items()},
                "config": asdict(stage_b.config),
                "audit": dict(stage_b.audit),
            }, handle)
    receipt = build_prediction_receipt(
        job,
        scenario,
        query_rows=query.features.row_count,
        selected_mode=selected_mode,
        stage_b_ran=stage_b is not None,
        mode_metrics=mode_metrics,
    )
    _write_json_new(output_root / "prediction_receipt.json", receipt)
    return receipt


def _accuracy(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(prediction[mask] == truth[mask])) if bool(mask.any()) else float("nan")


def _floor(prediction: np.ndarray, truth: np.ndarray, class_ids: np.ndarray) -> float:
    return min(float(np.mean(prediction[truth == class_id] == class_id)) for class_id in class_ids)


def score_truth_last(
    predictions_path: Path, receipt_path: Path, truth_path: Path, output_path: Path,
) -> Mapping[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "PREDICTIONS_COMPLETE" or receipt.get("query_truth_opened") is not False:
        raise BiSAGERunnerError("prediction is not truth-last eligible")
    with np.load(predictions_path, allow_pickle=False) as payload:
        predictions = {name: np.asarray(payload[name]) for name in payload.files}
    truth_payload = json.loads(truth_path.read_text(encoding="utf-8"))
    if str(truth_payload.get("receiver")) != str(receipt.get("receiver")):
        raise BiSAGERunnerError("truth receiver binding drift")
    rows = truth_payload.get("rows")
    if not isinstance(rows, list):
        raise BiSAGERunnerError("truth sidecar rows missing")
    query_tokens = np.asarray(predictions["query_tokens"]).astype(str)
    truth_lookup: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        token = str(row["query_token"])
        previous = truth_lookup.get(token)
        if previous is not None and (
            int(previous["true_class_index"]) != int(row["true_class_index"])
            or str(previous["true_class_handle"]) != str(row["true_class_handle"])
        ):
            raise BiSAGERunnerError("truth duplicate token binding drift")
        truth_lookup[token] = row
    if (
        len(set(query_tokens.tolist())) != len(query_tokens)
        or not set(query_tokens.tolist()).issubset(truth_lookup)
    ):
        raise BiSAGERunnerError("truth token join drift")
    aligned = [truth_lookup[token] for token in query_tokens]
    truth = np.asarray([int(row["true_class_index"]) for row in aligned], dtype=np.int64)
    truth_handles = np.asarray([str(row["true_class_handle"]) for row in aligned])
    old = truth < 6
    new = ~old
    states: dict[str, Any] = {}
    numeric_predictions: dict[str, np.ndarray] = {}
    for state in FOUR_STATES:
        prefix = state.lower()
        class_ids = np.asarray(predictions[f"{prefix}_class_ids"], dtype=np.int64)
        class_handles = np.asarray(predictions[f"{prefix}_class_handles"]).astype(str)
        logits = np.asarray(predictions[f"{prefix}_logits"], dtype=np.float64)
        predicted = np.asarray(predictions[f"{prefix}_predictions"], dtype=np.int64)
        predicted_handles = np.asarray(predictions[f"{prefix}_prediction_handles"]).astype(str)
        if logits.shape != (len(truth), len(class_ids)) or not np.isfinite(logits).all():
            raise BiSAGERunnerError(f"{state} prediction geometry drift")
        if not np.array_equal(predicted, class_ids[np.argmax(logits, axis=1)]):
            raise BiSAGERunnerError(f"{state} argmax drift")
        handle_map = dict(zip(class_ids.tolist(), class_handles.tolist()))
        for class_id in class_ids.tolist():
            observed = set(truth_handles[truth == int(class_id)].tolist())
            if observed != {handle_map[int(class_id)]}:
                raise BiSAGERunnerError(f"{state} truth/class handle binding drift")
        if not np.array_equal(predicted_handles, np.asarray([handle_map[int(value)] for value in predicted])):
            raise BiSAGERunnerError(f"{state} opaque class handle drift")
        numeric_predictions[state] = predicted
        per_class = {
            str(class_id): float(np.mean(predicted[truth == class_id] == class_id))
            for class_id in np.unique(truth)
            if class_id in set(class_ids.tolist())
        }
        row: dict[str, Any] = {
            "old_accuracy": _accuracy(predicted, truth, old),
            "old_floor": _floor(predicted, truth, np.arange(6)),
            "per_class_accuracy": per_class,
        }
        if state.endswith("REG0"):
            row.update({"new_accuracy": "N/A", "h": "N/A", "new_floor": "N/A", "old_to_new": "N/A", "new_to_old": "N/A"})
        else:
            old_accuracy = float(row["old_accuracy"])
            new_accuracy = _accuracy(predicted, truth, new)
            row.update({
                "new_accuracy": new_accuracy,
                "h": 2.0 * old_accuracy * new_accuracy / max(old_accuracy + new_accuracy, 1.0e-12),
                "new_floor": _floor(predicted, truth, np.arange(6, 6 + int(receipt["new_class_count"]))),
                "old_to_new": float(np.mean(predicted[old] >= 6)),
                "new_to_old": float(np.mean(predicted[new] < 6)),
            })
        states[state] = row
    effects = {
        "da_before_registration": states["DA1_REG0"]["old_accuracy"] - states["DA0_REG0"]["old_accuracy"],
        "da_after_registration": states["DA1_REG1"]["old_accuracy"] - states["DA0_REG1"]["old_accuracy"],
        "registration_without_da": states["DA0_REG1"]["old_accuracy"] - states["DA0_REG0"]["old_accuracy"],
        "registration_with_da": states["DA1_REG1"]["old_accuracy"] - states["DA1_REG0"]["old_accuracy"],
    }
    effects["interaction"] = effects["da_after_registration"] - effects["da_before_registration"]
    result = {
        "schema": "cvs.phase2.bisage_d92.score.v1",
        "status": "ANALYZED",
        "outer_key": receipt["outer_key"],
        "scenario": receipt["scenario"],
        "capsule_id": receipt["capsule_id"],
        "split_id": receipt["split_id"],
        "states": states,
        "effects_old_accuracy": effects,
        "forgetting_without_da": states["DA0_REG0"]["old_accuracy"] - states["DA0_REG1"]["old_accuracy"],
        "forgetting_with_da": states["DA1_REG0"]["old_accuracy"] - states["DA1_REG1"]["old_accuracy"],
        "truth_join_after_prediction_only": True,
        "truth_handle_alignment_verified": bool(all(truth_handles)),
    }
    _write_json_new(output_path, result)
    return MappingProxyType(result)


__all__ = [
    "BiSAGERunnerError", "FOUR_STATES", "SCENARIOS", "adapt_stage_a",
    "adapt_stage_b_and_predict", "class_handle_registry", "frozen_checkpoint",
    "build_prediction_receipt", "joint_stage_a_gate", "load_query", "load_stage_a",
    "load_support", "save_stage_a", "score_truth_last",
]
