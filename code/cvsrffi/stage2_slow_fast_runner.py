"""Truth-blind Phase2 runner for cached slow/fast adapter rows."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .slow_fast_adapter import SlowFastAdapterState, apply_slow_fast
from .slow_fast_bundle import load_slow_fast_bundle_strict
from .slow_fast_selection import (
    SupportTrustPolicy,
    evaluate_frozen_support_state,
    fit_support_candidate_states,
    select_support_only_state,
    select_support_only_state_legacy,
)
from .stage2_meta_adapter_runner import (
    _integer_tensor,
    _load_npz,
    _prototype_tensors,
    _query_ids,
    _received_iq_tensor,
    _support_physical_ids,
    _write_json_atomically,
    _write_prediction,
)
from .stage2_structured_late_block_adaptation import _identity_features
from .stage2_structured_late_block_runner import _load_frozen_checkpoint


_CONFIG_KEYS = frozenset(
    {
        "candidate_id",
        "bundle_id",
        "protocol_schema",
        "phase2_data_status",
        "capsule_id",
        "split_id",
        "base_checkpoint_path",
        "bundle_path",
        "support_path",
        "query_path",
        "prototype_path",
        "receiver",
        "scenario",
        "operating_point",
        "seed",
        "k_shot",
        "steps",
    }
)
_SHADOW_KEYS = frozenset(
    {"shadow_steps", "shadow_step_multipliers", "shadow_lambdas", "crossfit_repeats", "crossfit_seed"}
)
_P05_KEYS = frozenset(
    {
        "crossfit_seed",
        "p05_policy",
        "p05_lambda_grid",
        "p05_crossfit_repeats",
        "p05_step_size",
    }
)
_DA0_ONLY_KEYS = frozenset({"adaptation_mode"})
_SUPPORT_KEYS = frozenset(
    {"received_iq", "support_labels", "support_physical_ids"}
)
_QUERY_KEYS = frozenset({"received_iq", "query_ids"})
_PROTOTYPE_KEYS = frozenset({"prototypes", "class_ids"})


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping) or any(not isinstance(key, str) for key in config):
        raise ValueError("config must be a string-keyed mapping")
    actual = frozenset(config)
    allowed_shapes = (
        _CONFIG_KEYS,
        _CONFIG_KEYS | _SHADOW_KEYS,
        _CONFIG_KEYS | _P05_KEYS,
        _CONFIG_KEYS | _DA0_ONLY_KEYS,
    )
    if actual not in allowed_shapes:
        raise ValueError(
            "config allowlist mismatch: "
            f"missing={sorted(_CONFIG_KEYS - actual)} "
            f"extra={sorted(actual - (_CONFIG_KEYS | _SHADOW_KEYS | _P05_KEYS))}"
        )
    validated = dict(config)
    if validated["protocol_schema"] != "p2_min_v1":
        raise ValueError("protocol_schema must be p2_min_v1")
    if validated["phase2_data_status"] != "VALIDATED_ONCE":
        raise ValueError("phase2_data_status must be VALIDATED_ONCE")
    for key in (
        "candidate_id", "bundle_id", "capsule_id", "split_id", "receiver",
        "scenario", "operating_point",
    ):
        if not isinstance(validated[key], str) or not validated[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    if int(validated["k_shot"]) < 1 or int(validated["steps"]) != 3:
        raise ValueError("k_shot must be positive and steps must equal 3")
    validated["seed"] = int(validated["seed"])
    validated["k_shot"] = int(validated["k_shot"])
    validated["steps"] = int(validated["steps"])
    if actual == _CONFIG_KEYS | _DA0_ONLY_KEYS:
        if validated["adaptation_mode"] != "DA0_ONLY":
            raise ValueError("adaptation_mode must be DA0_ONLY")
        validated["adaptation_mode"] = "DA0_ONLY"
    if actual == _CONFIG_KEYS | _SHADOW_KEYS:
        steps = tuple(int(value) for value in validated["shadow_steps"])
        multipliers = tuple(float(value) for value in validated["shadow_step_multipliers"])
        lambdas = tuple(float(value) for value in validated["shadow_lambdas"])
        if not steps or len(set(steps)) != len(steps) or not set(steps) <= {1, 3, 5, 10}:
            raise ValueError("shadow_steps must be unique values from {1,3,5,10}")
        if (
            not multipliers
            or len(set(multipliers)) != len(multipliers)
            or not set(multipliers) <= {0.5, 1.0, 2.0, 4.0}
        ):
            raise ValueError("shadow_step_multipliers must use {0.5,1,2,4}")
        if not lambdas or len(set(lambdas)) != len(lambdas) or any(
            value <= 0.0 or value > 1.0 for value in lambdas
        ):
            raise ValueError("shadow_lambdas must be unique values in (0,1]")
        repeats = int(validated["crossfit_repeats"])
        if repeats < 1:
            raise ValueError("crossfit_repeats must be positive")
        validated["shadow_steps"] = steps
        validated["shadow_step_multipliers"] = multipliers
        validated["shadow_lambdas"] = lambdas
        validated["crossfit_repeats"] = repeats
        validated["crossfit_seed"] = int(validated["crossfit_seed"])
    if actual == _CONFIG_KEYS | _P05_KEYS:
        if validated["candidate_id"] != "FAST_FILM_R8":
            raise ValueError("formal P0.5 config requires FAST_FILM_R8")
        validated["crossfit_seed"] = int(validated["crossfit_seed"])
        policy = validated["p05_policy"]
        policy_keys = {
            "q90_move",
            "hard_move",
            "q90_relative_move",
            "minimum_positive_folds",
            "lcb_z",
            "require_fold_lcb",
        }
        if not isinstance(policy, Mapping) or set(policy) != policy_keys:
            raise ValueError("p05_policy allowlist mismatch")
        validated["p05_policy"] = {
            "q90_move": float(policy["q90_move"]),
            "hard_move": float(policy["hard_move"]),
            "q90_relative_move": float(policy["q90_relative_move"]),
            "minimum_positive_folds": int(policy["minimum_positive_folds"]),
            "lcb_z": float(policy["lcb_z"]),
            "require_fold_lcb": policy["require_fold_lcb"],
        }
        SupportTrustPolicy(**validated["p05_policy"])
        grid = tuple(float(value) for value in validated["p05_lambda_grid"])
        if not grid or tuple(sorted(set(grid))) != grid or grid[0] != 0.0 or grid[-1] > 1.0:
            raise ValueError("p05_lambda_grid must be sorted, unique and include zero")
        validated["p05_lambda_grid"] = grid
        validated["p05_crossfit_repeats"] = int(validated["p05_crossfit_repeats"])
        validated["p05_step_size"] = float(validated["p05_step_size"])
        if validated["p05_crossfit_repeats"] < 1 or validated["p05_step_size"] <= 0.0:
            raise ValueError("P0.5 repeat and step-size settings must be positive")
    return validated


def _extract_features(model: nn.Module, rows: Tensor) -> Tensor:
    try:
        device = next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("frozen base model has no parameters") from exc
    with torch.no_grad():
        features = _identity_features(model, rows.to(device))
    if features.ndim != 2 or features.shape[0] != rows.shape[0]:
        raise ValueError("frozen checkpoint returned invalid row-aligned z_id")
    return features.detach().cpu().float()


def _row_labels(class_labels: Tensor, class_ids: Tensor) -> Tensor:
    index = {int(class_id): row for row, class_id in enumerate(class_ids.tolist())}
    try:
        rows = [index[int(label)] for label in class_labels.tolist()]
    except KeyError as exc:
        raise ValueError("support contains a class absent from frozen prototypes") from exc
    return torch.tensor(rows, dtype=torch.long)


def _predict(features: Tensor, prototypes: Tensor, class_ids: Tensor) -> tuple[Tensor, Tensor]:
    scores = F.normalize(features.float(), dim=1) @ F.normalize(prototypes.float(), dim=1).T
    predictions = class_ids[scores.argmax(dim=1)]
    return predictions, scores


def _computation_accounting(
    *,
    candidate_id: str,
    selection: Mapping[str, Any],
    legacy_selection: Mapping[str, Any] | None,
    shadow_steps: tuple[int, ...] = (),
    shadow_step_multipliers: tuple[float, ...] = (),
) -> dict[str, int]:
    deployment = int(selection.get("deployment_candidate_updates", 0))
    crossfit = int(selection.get("crossfit_updates", 0))
    full_support = int(selection.get("full_support_fit_updates", deployment))
    committed = int(selection.get("committed_gradient_updates", 0))
    total_selection = int(selection.get("total_selection_updates", crossfit + full_support))
    query_inference = int(selection.get("query_inference_updates", 0))
    is_fast = candidate_id in {"FAST_FILM_R8", "FAST_LOWRANK_R8"}
    legacy = int(legacy_selection.get("attempted_gradient_updates", 0)) if is_fast and legacy_selection else 0
    grid = len(shadow_step_multipliers) * sum(int(value) for value in shadow_steps) if is_fast else 0
    total_deployment = total_selection
    return {
        "deployment_candidate_updates": deployment,
        "crossfit_updates": crossfit,
        "full_support_fit_updates": full_support,
        "committed_updates": committed,
        "total_selection_updates": total_selection,
        "query_inference_updates": query_inference,
        "total_deployment_updates": total_deployment,
        "legacy_diagnostic_updates": legacy,
        "shadow_grid_updates": grid,
        "total_diagnostic_updates": total_deployment + legacy + grid,
    }


def run_slow_fast_stage2_row(
    config: Mapping[str, Any],
    output_dir: str | Path,
    *,
    device: str | torch.device,
) -> dict[str, Any]:
    """Produce DA0_REG0 and DA1_REG0 predictions without opening query truth."""

    cfg = _validate_config(config)
    da0_only_mode = cfg.get("adaptation_mode") == "DA0_ONLY"
    state, audit = load_slow_fast_bundle_strict(cfg["bundle_path"])
    if state.candidate.value != cfg["candidate_id"]:
        raise ValueError("candidate_id does not match the frozen bundle")
    p05_mode = all(key in cfg for key in _P05_KEYS)
    calibration_policy = SupportTrustPolicy(**cfg["p05_policy"]) if p05_mode else None

    prototype_payload = _load_npz(
        cfg["prototype_path"], allowed=_PROTOTYPE_KEYS, label="prototype"
    )
    prototypes, class_ids = _prototype_tensors(prototype_payload)
    bundle_class_ids = audit["class_ids"]
    bundle_prototypes = audit["prototypes"]
    if not torch.equal(class_ids.cpu(), bundle_class_ids.cpu()) or not torch.allclose(
        prototypes.cpu(), bundle_prototypes.cpu(), atol=0.0, rtol=0.0
    ):
        raise ValueError("external prototypes do not exactly match the frozen bundle")
    if state.feature_dim != int(prototypes.shape[1]):
        raise ValueError("adapter and prototype feature widths differ")

    model = _load_frozen_checkpoint(cfg["base_checkpoint_path"], device=device)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("base checkpoint must remain fully frozen")

    logit_scale = float(audit["support_logit_scale"])
    trust_radius = float(audit["trust_radius"])
    named_adapter_states: dict[str, SlowFastAdapterState] = {}
    legacy_selection: dict[str, Any] | None = None
    shadow_state_specs: dict[str, dict[str, float | int | str]] = {}
    shadow_support_diagnostics: dict[str, Any] = {}
    shadow_mode = False
    support_sample_count = 0
    if da0_only_mode:
        selection_seed = int(cfg["seed"])
        selection = {
            "adaptation_mode": "DA0_ONLY",
            "selected_lambda": 0.0,
            "reason": "DA0_ONLY",
            "gradient_updates": 0,
            "deployment_candidate_updates": 0,
            "crossfit_updates": 0,
            "full_support_fit_updates": 0,
            "committed_gradient_updates": 0,
            "total_selection_updates": 0,
            "query_inference_updates": 0,
        }
    else:
        support_payload = _load_npz(
            cfg["support_path"], allowed=_SUPPORT_KEYS, label="support"
        )
        support_iq = _received_iq_tensor(
            support_payload["received_iq"], label="support"
        )
        support_labels = _integer_tensor(
            support_payload["support_labels"], label="support_labels"
        )
        if support_labels.numel() != support_iq.shape[0]:
            raise ValueError("support labels and IQ rows must align")
        support_ids = _support_physical_ids(
            support_payload["support_physical_ids"], expected_rows=int(support_iq.shape[0])
        )
        support_sample_count = len(support_ids)
        label_rows = _row_labels(support_labels, class_ids)
        support_features = _extract_features(model, support_iq)
        if calibration_policy is not None and calibration_policy.hard_move > trust_radius + 1.0e-8:
            raise ValueError("p05_policy hard_move exceeds the frozen bundle trust radius")
        shadow_mode = all(key in cfg for key in _SHADOW_KEYS)
        selection_steps = cfg["steps"]
        selection_step_size = float(cfg["p05_step_size"]) if p05_mode else float(audit["fast_step_size"])
        selection_grid = (
            tuple(cfg["p05_lambda_grid"])
            if p05_mode
            else ((0.0, *cfg["shadow_lambdas"]) if shadow_mode else (0.0, 0.125, 0.25, 0.5, 0.75, 1.0))
        )
        selection_repeats = (
            int(cfg["p05_crossfit_repeats"])
            if p05_mode
            else (cfg["crossfit_repeats"] if shadow_mode else 3)
        )
        selection_seed = int(cfg["crossfit_seed"]) if (p05_mode or shadow_mode) else int(cfg["seed"])
        selected_state, selection = select_support_only_state(
            support_features,
            label_rows,
            prototypes,
            state,
            k_shot=cfg["k_shot"],
            logit_scale=logit_scale,
            steps=selection_steps,
            step_size=selection_step_size,
            trust_radius=trust_radius,
            lambda_grid=selection_grid,
            crossfit_seed=selection_seed,
            repeats=selection_repeats,
            physical_ids=support_ids,
            trust_policy=calibration_policy,
        )

        if shadow_mode:
            legacy_state, legacy_selection = select_support_only_state_legacy(
                support_features,
                label_rows,
                prototypes,
                state,
                k_shot=cfg["k_shot"],
                logit_scale=logit_scale,
                trust_radius=trust_radius,
                steps=cfg["steps"],
                step_size=float(audit["fast_step_size"]),
            )
            if state.candidate.value == "COMMON_SHIFT_R4":
                fixed = fit_support_candidate_states(
                    support_features,
                    label_rows,
                    prototypes,
                    state,
                    steps=cfg["steps"],
                    step_size=float(audit["fast_step_size"]),
                    logit_scale=logit_scale,
                    lambda_grid=cfg["shadow_lambdas"],
                )
                for strength, fixed_state in fixed.items():
                    name = f"DA1_L{int(round(strength * 1000)):04d}_REG0"
                    named_adapter_states[name] = fixed_state
                    shadow_state_specs[name] = {
                        "selection": "fixed_support_only",
                        "steps": cfg["steps"],
                        "step_multiplier": 1.0,
                        "lambda": strength,
                    }
            else:
                for steps in cfg["shadow_steps"]:
                    for multiplier in cfg["shadow_step_multipliers"]:
                        fixed = fit_support_candidate_states(
                            support_features,
                            label_rows,
                            prototypes,
                            state,
                            steps=steps,
                            step_size=float(audit["fast_step_size"]) * multiplier,
                            logit_scale=logit_scale,
                            lambda_grid=cfg["shadow_lambdas"],
                        )
                        for strength, fixed_state in fixed.items():
                            name = (
                                f"DA1_J{steps:02d}_A{int(round(multiplier * 100)):03d}_"
                                f"L{int(round(strength * 1000)):04d}_REG0"
                            )
                            named_adapter_states[name] = fixed_state
                            shadow_state_specs[name] = {
                                "selection": "fixed_support_only",
                                "steps": steps,
                                "step_multiplier": multiplier,
                                "lambda": strength,
                            }
            named_adapter_states["DA1_GATE_LEGACY_REG0"] = legacy_state
            named_adapter_states["DA1_GATE_CF_REG0"] = selected_state
        else:
            named_adapter_states["DA1_REG0"] = selected_state

        support_diagnostic_states = {
            "DA0_REG0": replace(state, rho=0.0),
            **named_adapter_states,
        }
        shadow_support_diagnostics = {
            name: evaluate_frozen_support_state(
                support_features,
                label_rows,
                prototypes,
                adapter_state,
                logit_scale=logit_scale,
            )
            for name, adapter_state in support_diagnostic_states.items()
        }

    computation_accounting = _computation_accounting(
        candidate_id=cfg["candidate_id"],
        selection=selection,
        legacy_selection=legacy_selection,
        shadow_steps=tuple(cfg["shadow_steps"]) if shadow_mode else (),
        shadow_step_multipliers=tuple(cfg["shadow_step_multipliers"]) if shadow_mode else (),
    )

    # Query is opened only after the support-only state is final and immutable.
    query_payload = _load_npz(cfg["query_path"], allowed=_QUERY_KEYS, label="query")
    query_iq = _received_iq_tensor(query_payload["received_iq"], label="query")
    query_ids = _query_ids(
        query_payload["query_ids"], expected_rows=int(query_iq.shape[0])
    )
    query_features = torch.cat(
        [_extract_features(model, query_iq[row : row + 1]) for row in range(query_iq.shape[0])],
        dim=0,
    )
    named_predictions: dict[str, tuple[Tensor, Tensor]] = {
        "DA0_REG0": _predict(query_features, prototypes, class_ids)
    }
    if da0_only_mode:
        named_predictions["DA1_REG0"] = named_predictions["DA0_REG0"]
    else:
        for name, adapter_state in named_adapter_states.items():
            adapted = apply_slow_fast(query_features, adapter_state)
            named_predictions[name] = _predict(adapted, prototypes, class_ids)

    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"non-overwriting output already exists: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    prediction_paths: dict[str, str] = {}
    for state_name, (predicted, scores) in named_predictions.items():
        path = destination / f"predictions_{state_name}.npz"
        _write_prediction(path, query_ids=query_ids, predicted_class_ids=predicted, scores=scores)
        prediction_paths[state_name] = str(path.resolve())

    receipt_path = destination / "receipt.json"
    receipt = {
        "status": "PREDICTIONS_COMPLETE",
        "adaptation_mode": "DA0_ONLY" if da0_only_mode else "SUPPORT_ONLY",
        "support_adapter_opened": not da0_only_mode,
        "states": list(named_predictions),
        "candidate_id": cfg["candidate_id"],
        "bundle_id": cfg["bundle_id"],
        "base_checkpoint_id": audit["base_checkpoint_id"],
        "protocol_schema": cfg["protocol_schema"],
        "phase2_data_status": cfg["phase2_data_status"],
        "capsule_id": cfg["capsule_id"],
        "split_id": cfg["split_id"],
        "receiver": cfg["receiver"],
        "scenario": cfg["scenario"],
        "operating_point": cfg["operating_point"],
        "seed": cfg["seed"],
        "k_shot": cfg["k_shot"],
        "steps": cfg["steps"],
        "support_physical_sample_count": support_sample_count,
        "query_count": int(query_iq.shape[0]),
        "support_logit_scale": logit_scale,
        "score_type": "raw_cosine",
        "trust_radius": trust_radius,
        "registered_class_ids": [int(value) for value in class_ids.tolist()],
        "decision_rule": "frozen_prototype_cosine_slow_fast_v2" if p05_mode else "frozen_prototype_cosine_slow_fast_v1",
        "adapter_schema": str(audit["schema"]),
        "selection_schema": "cvs.slow_fast.p05.selection.v1" if p05_mode else "cvs.slow_fast.selection.v2",
        "prediction_schema": "raw_cosine.v1",
        "crossfit_seed": selection_seed,
        "selected_lambda": float(selection["selected_lambda"]),
        "selection_reason": str(selection["reason"]),
        "support_gradient_updates": int(selection["gradient_updates"]),
        "support_selection": selection,
        "legacy_selection": legacy_selection,
        "shadow_state_specs": shadow_state_specs,
        "shadow_support_diagnostics": shadow_support_diagnostics,
        "computation_accounting": computation_accounting,
        "query_state_update_count": 0,
        "query_truth_opened": False,
        "query_role_opened": False,
        "source_opened": False,
        "states_same_row": True,
        "query_ids": query_ids.tolist(),
        "prediction_paths": prediction_paths,
        "receipt_path": str(receipt_path.resolve()),
    }
    _write_json_atomically(receipt_path, receipt)
    return receipt


__all__ = ["run_slow_fast_stage2_row"]
