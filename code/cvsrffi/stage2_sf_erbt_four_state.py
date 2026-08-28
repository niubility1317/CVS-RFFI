"""Four-state SF-TAPFT plus ERBT-IDR Stage2-C helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from cvsrffi.stage2_sf_erbt_oldonly import (
    ARM,
    OldOnlyERBTError,
    OldOnlyERBTState,
    _d92_geometry,
    _extract_identity160,
    _features,
    fit_old_only_erbt,
    make_fft96,
)


def fit_registered_erbt(
    identity160: Any,
    fft96: Any,
    labels: Any,
    *,
    class_ids: Sequence[int],
    old_class_count: int,
    seed: int,
    device: Any = "cpu",
) -> OldOnlyERBTState:
    """Fit the no-RF32 ERBT state from old/new K-shot support only."""

    from cvsrffi import stage2_ablation_executors as executors
    from cvsrffi import stage2_d42_unified_shrinkage_lda as d42

    registry = tuple(int(value) for value in class_ids)
    old_count = int(old_class_count)
    if old_count != 6 or len(registry) <= old_count or len(set(registry)) != len(registry):
        raise OldOnlyERBTError("REG1 requires six old classes and at least one new class")
    label_rows = np.asarray(labels, dtype=np.int64)
    lookup = {value: index for index, value in enumerate(registry)}
    if set(label_rows.tolist()) != set(registry):
        raise OldOnlyERBTError("registered support registry drift")
    targets = np.asarray([lookup[int(value)] for value in label_rows], dtype=np.int64)
    counts = np.bincount(targets, minlength=len(registry))
    if np.any(counts <= 0) or len(set(counts.tolist())) != 1:
        raise OldOnlyERBTError("registered support must be balanced K-shot")
    features = _features(identity160, fft96)
    if len(features) != len(targets):
        raise OldOnlyERBTError("registered support feature/label row drift")
    old_mask = targets < old_count
    if int(old_mask.sum()) != old_count * int(counts[0]):
        raise OldOnlyERBTError("old support prefix drift")

    with _d92_geometry():
        log_diag, trace, _ = executors._metric(
            features[old_mask],
            targets[old_mask],
            old_count,
            enabled=True,
            seed=int(seed),
            device=device,
        )
        transformed = d42._transform(features, log_diag)
        fit, method = executors._component_builder(
            "P2-B0",
            ground_basis=np.empty((160, 0), dtype=np.float64),
            ground_weights=np.empty(0, dtype=np.float64),
            ground_audit={},
        )
        coefficient, intercept, fit_audit = executors._fit_with_fp32_centering_audit(
            fit,
            transformed,
            targets,
            len(registry),
            int(counts[0]),
        )
    audit = {
        **dict(fit_audit),
        "arm": ARM,
        "method_lock": "D92-E0-NORF32",
        "rf32_used": False,
        "registration_state": "REG1",
        "support_only": True,
        "query_rows_used": 0,
        "metric_support_rows": int(old_mask.sum()),
        "metric_new_support_rows": 0,
        "numerical_method": method,
        "optimizer_steps": len(trace),
    }
    return OldOnlyERBTState(
        class_ids=registry,
        log_diag=np.asarray(log_diag, dtype=np.float32),
        coefficient=np.asarray(coefficient, dtype=np.float32),
        intercept=np.asarray(intercept, dtype=np.float32),
        audit=audit,
    )


def fit_erbt_registration_pair(
    old_identity160: Any,
    old_fft96: Any,
    old_labels: Any,
    registered_identity160: Any,
    registered_fft96: Any,
    registered_labels: Any,
    *,
    old_class_ids: Sequence[int],
    registered_class_ids: Sequence[int],
    seed: int,
    device: Any = "cpu",
) -> tuple[OldOnlyERBTState, OldOnlyERBTState, dict[str, Any]]:
    """Fit REG0/REG1 heads while sharing one old-support domain metric."""

    from cvsrffi import stage2_ablation_executors as executors
    from cvsrffi import stage2_d42_unified_shrinkage_lda as d42
    from cvsrffi.stage2_d92_registration_balanced_covariance import _group_covariance

    old_registry = tuple(int(value) for value in old_class_ids)
    registered_registry = tuple(int(value) for value in registered_class_ids)
    if old_registry != registered_registry[:6] or len(old_registry) != 6:
        raise OldOnlyERBTError("registration pair old registry drift")
    old_targets = np.asarray(old_labels, dtype=np.int64)
    registered_targets = np.asarray(registered_labels, dtype=np.int64)
    if set(old_targets.tolist()) != set(old_registry):
        raise OldOnlyERBTError("REG0 support registry drift")
    if set(registered_targets.tolist()) != set(registered_registry):
        raise OldOnlyERBTError("REG1 support registry drift")
    old_lookup = {value: index for index, value in enumerate(old_registry)}
    registered_lookup = {
        value: index for index, value in enumerate(registered_registry)
    }
    old_indices = np.asarray([old_lookup[int(value)] for value in old_targets])
    registered_indices = np.asarray(
        [registered_lookup[int(value)] for value in registered_targets]
    )
    old_counts = np.bincount(old_indices, minlength=6)
    registered_counts = np.bincount(
        registered_indices, minlength=len(registered_registry)
    )
    k_shot = int(old_counts[0]) if len(old_counts) else 0
    if (
        k_shot < 3
        or old_counts.tolist() != [k_shot] * 6
        or registered_counts.tolist() != [k_shot] * len(registered_registry)
    ):
        raise OldOnlyERBTError("registration pair requires balanced K>=3 support")
    old_features = _features(old_identity160, old_fft96)
    registered_features = _features(registered_identity160, registered_fft96)
    if len(old_features) != k_shot * 6 or len(registered_features) != k_shot * len(
        registered_registry
    ):
        raise OldOnlyERBTError("registration pair support row drift")

    with _d92_geometry():
        log_diag, trace, _ = executors._metric(
            old_features,
            old_indices,
            6,
            enabled=True,
            seed=int(seed),
            device=device,
        )

        def fit_head(
            features: np.ndarray,
            targets: np.ndarray,
            registry: tuple[int, ...],
            registration_state: str,
        ) -> OldOnlyERBTState:
            transformed = d42._transform(features, log_diag)
            fit, method = executors._component_builder(
                "P2-B0",
                ground_basis=np.empty((160, 0), dtype=np.float64),
                ground_weights=np.empty(0, dtype=np.float64),
                ground_audit={},
            )
            coefficient, intercept, fit_audit = executors._fit_with_fp32_centering_audit(
                fit,
                transformed,
                targets,
                len(registry),
                k_shot,
            )
            return OldOnlyERBTState(
                class_ids=registry,
                log_diag=np.asarray(log_diag, dtype=np.float32),
                coefficient=np.asarray(coefficient, dtype=np.float32),
                intercept=np.asarray(intercept, dtype=np.float32),
                audit={
                    **dict(fit_audit),
                    "arm": ARM,
                    "method_lock": "D92-E0-NORF32",
                    "rf32_used": False,
                    "registration_state": registration_state,
                    "support_only": True,
                    "query_rows_used": 0,
                    "metric_support_rows": int(k_shot * 6),
                    "metric_new_support_rows": 0,
                    "numerical_method": method,
                    "optimizer_steps": len(trace),
                },
            )

        reg0 = fit_head(old_features, old_indices, old_registry, "REG0")
        reg1 = fit_head(
            registered_features,
            registered_indices,
            registered_registry,
            "REG1",
        )
        registered_transformed = d42._transform(registered_features, log_diag)
        old_covariance = _group_covariance(
            d42,
            registered_transformed,
            registered_indices,
            np.arange(6, dtype=np.int64),
        )
        new_covariance = _group_covariance(
            d42,
            registered_transformed,
            registered_indices,
            np.arange(6, len(registered_registry), dtype=np.int64),
        )
        balanced_covariance = 0.5 * (old_covariance + new_covariance)
        balanced_covariance = 0.5 * (
            balanced_covariance + balanced_covariance.T
        )
        eigenvalues = np.linalg.eigvalsh(balanced_covariance)
        block_traces = [
            float(np.trace(balanced_covariance[block, block]))
            for block in d42.BLOCK_SLICES
        ]
    return reg0, reg1, {
        "metric_fit_count": 1,
        "metric_support_rows": int(k_shot * 6),
        "metric_new_support_rows": 0,
        "metric_optimizer_steps": len(trace),
        "k_shot": k_shot,
        "reg0_d92_audit": {
            key: value for key, value in reg0.audit.items() if key.startswith("d92_")
        },
        "reg1_d92_audit": {
            key: value for key, value in reg1.audit.items() if key.startswith("d92_")
        },
        "reg1_balanced_covariance_audit": {
            "positive_definite": bool(float(np.min(eigenvalues)) > 0.0),
            "eigenvalue_min": float(np.min(eigenvalues)),
            "eigenvalue_max": float(np.max(eigenvalues)),
            "condition_number": float(np.max(eigenvalues) / np.min(eigenvalues)),
            "block_traces": block_traces,
        },
    }


_SUPPORT_KEYS = frozenset(
    {"received_iq", "support_labels", "support_physical_ids"}
)
_QUERY_KEYS = frozenset({"received_iq", "query_ids"})


def _load_npz(path: str | Path, expected: frozenset[str], label: str) -> dict[str, np.ndarray]:
    try:
        with np.load(Path(path), allow_pickle=False) as payload:
            if frozenset(payload.files) != expected:
                raise OldOnlyERBTError(f"{label} allowlist mismatch")
            return {name: np.asarray(payload[name]) for name in payload.files}
    except OldOnlyERBTError:
        raise
    except (OSError, ValueError) as exc:
        raise OldOnlyERBTError(f"cannot load {label}") from exc


def _load_query(path: str | Path) -> dict[str, np.ndarray]:
    return _load_npz(path, _QUERY_KEYS, "query")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def run_four_state_prediction(
    *,
    base_checkpoint_path: str | Path,
    d3_delta_path: str | Path,
    old_support_path: str | Path,
    registered_support_path: str | Path,
    query_path: str | Path,
    data_handle_path: str | Path,
    output_root: str | Path,
    seed: int,
    device: str | torch.device,
    base_model_loader: Any | None = None,
    delta_bundle_loader: Any | None = None,
    query_loader: Any | None = None,
) -> dict[str, Any]:
    """Freeze all four support states, then open one unlabeled query stream."""

    destination = Path(output_root)
    if destination.exists() or destination.is_symlink():
        raise OldOnlyERBTError(f"output root already exists: {destination}")
    try:
        with Path(data_handle_path).open("r", encoding="utf-8") as handle:
            binding = json.load(handle)
    except (OSError, ValueError, TypeError) as exc:
        raise OldOnlyERBTError("cannot load four-state data handle") from exc
    expected_handle = {
        "schema": "cvs.sf_erbt_four_state.handle.v1",
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "k_shot": 10,
        "old_class_count": 6,
        "old_support_rows": 60,
    }
    if not isinstance(binding, dict) or any(
        binding.get(name) != value for name, value in expected_handle.items()
    ):
        raise OldOnlyERBTError("four-state data handle binding mismatch")
    new_class_count = int(binding.get("new_class_count", -1))
    registered_class_count = 6 + new_class_count
    registered_support_rows = registered_class_count * 10
    if (
        new_class_count not in {1, 2, 3, 5, 10, 15, 20}
        or binding.get("registered_support_rows") != registered_support_rows
    ):
        raise OldOnlyERBTError("four-state registered class geometry drift")
    if str(binding.get("base_checkpoint_path")) != str(base_checkpoint_path):
        raise OldOnlyERBTError("base checkpoint binding mismatch")
    if not str(binding.get("capsule_id", "")).strip() or not str(
        binding.get("split_id", "")
    ).strip() or not str(binding.get("da_split_id", "")).strip():
        raise OldOnlyERBTError("capsule or split binding is empty")

    old_support = _load_npz(old_support_path, _SUPPORT_KEYS, "old support")
    registered_support = _load_npz(
        registered_support_path, _SUPPORT_KEYS, "registered support"
    )
    old_iq = np.asarray(old_support["received_iq"], dtype=np.float32)
    registered_iq = np.asarray(registered_support["received_iq"], dtype=np.float32)
    old_labels = np.asarray(old_support["support_labels"], dtype=np.int64)
    registered_labels = np.asarray(
        registered_support["support_labels"], dtype=np.int64
    )
    old_ids = np.asarray(old_support["support_physical_ids"]).astype(str)
    registered_ids = np.asarray(
        registered_support["support_physical_ids"]
    ).astype(str)
    if (
        old_iq.shape != (60, 2, 256)
        or registered_iq.shape != (registered_support_rows, 2, 256)
        or old_labels.shape != (60,)
        or registered_labels.shape != (registered_support_rows,)
        or np.bincount(old_labels, minlength=6).tolist() != [10] * 6
        or np.bincount(
            registered_labels, minlength=registered_class_count
        ).tolist()
        != [10] * registered_class_count
        or old_ids.shape != (60,)
        or registered_ids.shape != (registered_support_rows,)
    ):
        raise OldOnlyERBTError("four-state support geometry drift")
    if not np.array_equal(old_ids, registered_ids[:60]) or not np.array_equal(
        old_iq, registered_iq[:60]
    ):
        raise OldOnlyERBTError("REG0/REG1 old support is not the same row")

    target_device = torch.device(device)
    if base_model_loader is None:
        from cvsrffi.target_only_progressive_adapt import ensure_time_adapter
        from cvsrffi.target_only_progressive_runner import _default_checkpoint_loader

        da0_model = _default_checkpoint_loader(base_checkpoint_path, device=target_device)
        ensure_time_adapter(da0_model, rank=16)
    else:
        da0_model = base_model_loader(base_checkpoint_path, device=target_device)
    if delta_bundle_loader is None:
        from cvsrffi.target_only_progressive_runner import load_sf_tapft_delta_bundle_strict

        delta_bundle_loader = load_sf_tapft_delta_bundle_strict
    expected_target_binding = {
        "protocol_schema": binding["protocol_schema"],
        "phase2_data_status": binding["phase2_data_status"],
        "capsule_id": binding["capsule_id"],
        "split_id": binding["da_split_id"],
        "support_count": 60,
    }
    da1_model, _da1_head, delta_audit = delta_bundle_loader(
        d3_delta_path,
        device=target_device,
        expected_target_binding=expected_target_binding,
    )
    for model in (da0_model, da1_model):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    states: dict[str, OldOnlyERBTState] = {}
    state_audits: dict[str, Any] = {}
    for da_name, model in (("DA0", da0_model), ("DA1", da1_model)):
        old_identity = _extract_identity160(model, old_iq, target_device)
        registered_identity = _extract_identity160(
            model, registered_iq, target_device
        )
        reg0, reg1, pair_audit = fit_erbt_registration_pair(
            old_identity,
            make_fft96(old_iq),
            old_labels,
            registered_identity,
            make_fft96(registered_iq),
            registered_labels,
            old_class_ids=tuple(range(6)),
            registered_class_ids=tuple(range(registered_class_count)),
            seed=int(seed),
            device=target_device,
        )
        states[f"{da_name}_REG0"] = reg0
        states[f"{da_name}_REG1"] = reg1
        state_audits[da_name] = pair_audit

    destination.mkdir(parents=True, exist_ok=False)
    support_receipt = {
        "schema": "cvs.sf_erbt_four_state.support_states.v1",
        "capsule_id": binding["capsule_id"],
        "split_id": binding["split_id"],
        "scenario": binding["scenario"],
        "support_states": ["DA0_REG0", "DA1_REG0", "DA0_REG1", "DA1_REG1"],
        "support_states_frozen": True,
        "query_opened": False,
        "query_truth_opened": False,
        "query_role_opened": False,
        "state_audits": state_audits,
        "delta_audit": dict(delta_audit),
    }
    _write_json(destination / "support_state_receipt.json", support_receipt)

    load_query = query_loader or _load_query
    query = load_query(query_path)
    query_iq = np.asarray(query["received_iq"], dtype=np.float32)
    query_ids = np.asarray(query["query_ids"]).astype(str)
    if (
        query_iq.ndim != 3
        or query_iq.shape[1:] != (2, 256)
        or query_ids.shape != (len(query_iq),)
        or len(query_iq) != int(binding["query_rows"])
        or set(query_ids.tolist()) & set(registered_ids.tolist())
    ):
        raise OldOnlyERBTError("four-state query geometry or separation drift")

    query_fft = make_fft96(query_iq)
    prediction_payload: dict[str, np.ndarray] = {"query_ids": query_ids}
    for da_name, model in (("DA0", da0_model), ("DA1", da1_model)):
        identity = _extract_identity160(model, query_iq, target_device)
        for reg_name in ("REG0", "REG1"):
            key = f"{da_name}_{reg_name}"
            state = states[key]
            logits = state.score(identity, query_fft)
            predictions = np.asarray(state.class_ids, dtype=np.int64)[
                np.argmax(logits, axis=1)
            ]
            prefix = key.lower()
            prediction_payload[f"{prefix}_class_ids"] = np.asarray(
                state.class_ids, dtype=np.int64
            )
            prediction_payload[f"{prefix}_logits"] = logits
            prediction_payload[f"{prefix}_predictions"] = predictions
    with (destination / "predictions.npz").open("xb") as handle:
        np.savez(handle, **prediction_payload)
    receipt = {
        "schema": "cvs.sf_erbt_four_state.prediction.v1",
        "status": "PREDICTIONS_COMPLETE",
        "capsule_id": binding["capsule_id"],
        "split_id": binding["split_id"],
        "scenario": binding["scenario"],
        "k_shot": 10,
        "old_class_count": 6,
        "new_class_count": new_class_count,
        "query_rows": len(query_iq),
        "four_states": ["DA0_REG0", "DA1_REG0", "DA0_REG1", "DA1_REG1"],
        "support_states_frozen_before_query_open": True,
        "query_truth_opened": False,
        "query_role_opened": False,
        "source_opened": False,
    }
    _write_json(destination / "prediction_receipt.json", receipt)
    return receipt


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=1, keepdims=True)


def _ece(probabilities: np.ndarray, predictions: np.ndarray, truth: np.ndarray) -> float:
    confidence = np.max(probabilities, axis=1)
    correct = predictions == truth
    result = 0.0
    for index in range(10):
        lower = index / 10.0
        upper = (index + 1) / 10.0
        mask = (confidence >= lower) & (
            confidence <= upper if index == 9 else confidence < upper
        )
        if np.any(mask):
            result += float(np.mean(mask)) * abs(
                float(np.mean(correct[mask])) - float(np.mean(confidence[mask]))
            )
    return float(result)


def _state_metrics(
    *,
    class_ids: np.ndarray,
    logits: np.ndarray,
    predictions: np.ndarray,
    truth: np.ndarray,
    registration_state: str,
) -> dict[str, Any]:
    probabilities = _softmax(logits)
    lookup = {int(value): index for index, value in enumerate(class_ids.tolist())}
    old_mask = truth < 6

    def subset_metrics(mask: np.ndarray) -> dict[str, Any]:
        selected_truth = truth[mask]
        selected_predictions = predictions[mask]
        selected_probabilities = probabilities[mask]
        true_columns = np.asarray([lookup[int(value)] for value in selected_truth])
        nll_rows = -np.log(
            np.clip(
                selected_probabilities[np.arange(len(true_columns)), true_columns],
                1.0e-12,
                1.0,
            )
        )
        classes = sorted(set(selected_truth.tolist()))
        class_accuracy = {
            str(value): float(
                np.mean(selected_predictions[selected_truth == value] == value)
            )
            for value in classes
        }
        class_nll = {
            str(value): float(np.mean(nll_rows[selected_truth == value]))
            for value in classes
        }
        return {
            "accuracy": float(np.mean(selected_predictions == selected_truth)),
            "floor": float(min(class_accuracy.values())),
            "nll": float(np.mean(nll_rows)),
            "ece10": _ece(
                selected_probabilities, selected_predictions, selected_truth
            ),
            "class_accuracy": class_accuracy,
            "class_nll": class_nll,
            "rows": int(np.sum(mask)),
        }

    old = subset_metrics(old_mask)
    result: dict[str, Any] = {
        "old_acc": old["accuracy"],
        "old_floor": old["floor"],
        "old_nll": old["nll"],
        "old_ece10": old["ece10"],
        "old_class_accuracy": old["class_accuracy"],
        "old_class_nll": old["class_nll"],
        "old_query_rows": old["rows"],
        "seen_new_acc": None,
        "new_floor": None,
        "H_old_new": None,
        "registered_acc": old["accuracy"],
        "registered_floor": old["floor"],
        "registered_nll": old["nll"],
        "registered_ece10": old["ece10"],
        "seen_new_class_accuracy": None,
        "seen_new_class_nll": None,
    }
    if registration_state == "REG1":
        new_mask = ~old_mask
        new = subset_metrics(new_mask)
        registered = subset_metrics(np.ones(len(truth), dtype=bool))
        denominator = old["accuracy"] + new["accuracy"]
        result.update(
            {
                "seen_new_acc": new["accuracy"],
                "new_floor": new["floor"],
                "H_old_new": (
                    0.0
                    if denominator <= 0.0
                    else float(2.0 * old["accuracy"] * new["accuracy"] / denominator)
                ),
                "registered_acc": registered["accuracy"],
                "registered_floor": registered["floor"],
                "registered_nll": registered["nll"],
                "registered_ece10": registered["ece10"],
                "seen_new_class_accuracy": new["class_accuracy"],
                "seen_new_class_nll": new["class_nll"],
            }
        )
    return result


def score_four_state_predictions(
    predictions_path: str | Path,
    truth_path: str | Path,
    prediction_receipt_path: str | Path,
    data_handle_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Join truth after prediction closure and report four-state effects."""

    states = ("DA0_REG0", "DA1_REG0", "DA0_REG1", "DA1_REG1")
    expected_prediction_keys = {"query_ids"}
    for state in states:
        prefix = state.lower()
        expected_prediction_keys.update(
            {
                f"{prefix}_class_ids",
                f"{prefix}_logits",
                f"{prefix}_predictions",
            }
        )
    predictions = _load_npz(
        predictions_path,
        frozenset(expected_prediction_keys),
        "four-state predictions",
    )
    truth_payload = _load_npz(
        truth_path, frozenset({"query_ids", "query_labels"}), "truth"
    )
    try:
        with Path(prediction_receipt_path).open("r", encoding="utf-8") as handle:
            receipt = json.load(handle)
        with Path(data_handle_path).open("r", encoding="utf-8") as handle:
            binding = json.load(handle)
    except (OSError, ValueError, TypeError) as exc:
        raise OldOnlyERBTError("cannot load four-state scoring bindings") from exc
    required_receipt = {
        "schema": "cvs.sf_erbt_four_state.prediction.v1",
        "status": "PREDICTIONS_COMPLETE",
        "query_truth_opened": False,
        "query_role_opened": False,
    }
    if not isinstance(receipt, dict) or any(
        receipt.get(name) != value for name, value in required_receipt.items()
    ):
        raise OldOnlyERBTError("prediction receipt is not truth-last eligible")
    for name in (
        "capsule_id",
        "split_id",
        "scenario",
        "k_shot",
        "old_class_count",
        "new_class_count",
        "query_rows",
    ):
        if receipt.get(name) != binding.get(name):
            raise OldOnlyERBTError("prediction/data scoring binding mismatch")
    query_ids = np.asarray(predictions["query_ids"]).astype(str)
    truth_ids = np.asarray(truth_payload["query_ids"]).astype(str)
    truth_labels = np.asarray(truth_payload["query_labels"], dtype=np.int64)
    if (
        query_ids.shape != truth_ids.shape
        or truth_labels.shape != query_ids.shape
        or len(set(query_ids.tolist())) != len(query_ids)
        or set(query_ids.tolist()) != set(truth_ids.tolist())
    ):
        raise OldOnlyERBTError("truth ID join drift")
    truth_lookup = {value: int(label) for value, label in zip(truth_ids, truth_labels)}
    aligned_truth = np.asarray([truth_lookup[value] for value in query_ids], dtype=np.int64)
    new_class_count = int(binding["new_class_count"])
    registered_class_count = 6 + new_class_count
    if set(aligned_truth.tolist()) != set(range(registered_class_count)):
        raise OldOnlyERBTError("truth registered class coverage drift")

    state_results: dict[str, Any] = {}
    for state in states:
        prefix = state.lower()
        class_ids = np.asarray(predictions[f"{prefix}_class_ids"], dtype=np.int64)
        logits = np.asarray(predictions[f"{prefix}_logits"], dtype=np.float64)
        predicted = np.asarray(
            predictions[f"{prefix}_predictions"], dtype=np.int64
        )
        expected_classes = np.arange(
            6 if state.endswith("REG0") else registered_class_count
        )
        if (
            not np.array_equal(class_ids, expected_classes)
            or logits.shape != (len(query_ids), len(class_ids))
            or predicted.shape != (len(query_ids),)
            or not np.isfinite(logits).all()
            or not np.array_equal(
                predicted, class_ids[np.argmax(logits, axis=1)]
            )
        ):
            raise OldOnlyERBTError(f"{state} prediction geometry drift")
        state_results[state] = _state_metrics(
            class_ids=class_ids,
            logits=logits,
            predictions=predicted,
            truth=aligned_truth,
            registration_state="REG0" if state.endswith("REG0") else "REG1",
        )

    effect_metrics = ("old_acc", "old_floor", "old_nll", "old_ece10")

    def effect(new_state: str, old_state: str) -> dict[str, float]:
        return {
            name: float(
                state_results[new_state][name] - state_results[old_state][name]
            )
            for name in effect_metrics
        }

    da_before = effect("DA1_REG0", "DA0_REG0")
    da_after = effect("DA1_REG1", "DA0_REG1")
    effects = {
        "da_before_registration": da_before,
        "da_after_registration": da_after,
        "registration_without_da": effect("DA0_REG1", "DA0_REG0"),
        "registration_with_da": effect("DA1_REG1", "DA1_REG0"),
        "interaction": {
            name: float(da_after[name] - da_before[name])
            for name in effect_metrics
        },
    }
    result = {
        "schema": "cvs.sf_erbt_four_state.score.v1",
        "status": "ANALYZED",
        "capsule_id": receipt["capsule_id"],
        "split_id": receipt["split_id"],
        "scenario": receipt["scenario"],
        "k_shot": 10,
        "old_class_count": 6,
        "new_class_count": new_class_count,
        "states": state_results,
        "effects": effects,
        "truth_join_after_prediction_only": True,
    }
    _write_json(Path(output_path), result)
    return result


__all__ = [
    "fit_erbt_registration_pair",
    "fit_registered_erbt",
    "run_four_state_prediction",
    "score_four_state_predictions",
]
