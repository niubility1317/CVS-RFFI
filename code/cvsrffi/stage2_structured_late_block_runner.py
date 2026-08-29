"""One-row Phase2-B runner for structured ADV3B02 late-block adaptation.

The runtime surface is deliberately closed: a frozen checkpoint, one fixed
target-support payload, one fixed target-query payload, immutable two-
dimensional class prototypes and a preregistered algorithm configuration.
Query data is not opened until support adaptation has returned a fully frozen
model, and prediction artifacts contain no truth or role information.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

from cvsrffi.stage2_structured_late_block_adaptation import (
    MAX_ADAPTATION_STEPS,
    StructuredLateBlockConfig,
    adapt_on_target_support_with_frozen_prototypes,
    predict_query_with_frozen_prototypes,
)


_CONFIG_ALLOWLIST = frozenset(
    {
        "protocol_schema",
        "phase2_data_status",
        "capsule_id",
        "split_id",
        "row_id",
        "receiver",
        "scenario",
        "seed",
        "k_shot",
        "checkpoint_path",
        "support_path",
        "query_path",
        "prototype_path",
        "candidate",
        "steps",
        "learning_rate",
        "decision_rule",
    }
)
_BOUNDED_CONFIG_ALLOWLIST = _CONFIG_ALLOWLIST | frozenset(
    {"min_trainable_fraction", "max_trainable_fraction"}
)
_CONTEXT_KEYS = (
    "protocol_schema",
    "phase2_data_status",
    "capsule_id",
    "split_id",
)
_SUPPORT_PAYLOAD_ALLOWLIST = frozenset({"received_iq", "support_labels"})
_QUERY_PAYLOAD_ALLOWLIST = frozenset({"received_iq", "query_ids"})
_PROTOTYPE_PAYLOAD_ALLOWLIST = frozenset({"prototypes", "class_ids"})
_DECISION_RULE = "frozen_prototype_cosine_v1"


class StructuredLateBlockRunnerError(ValueError):
    """Raised when a row violates the closed Phase2 runner contract."""


def _validate_exact_keys(
    payload: Mapping[str, Any],
    allowed: frozenset[str],
    *,
    label: str,
) -> None:
    actual = frozenset(str(key) for key in payload)
    if actual != allowed:
        raise StructuredLateBlockRunnerError(
            f"{label} payload allowlist mismatch: "
            f"missing={sorted(allowed - actual)}, extra={sorted(actual - allowed)}"
        )


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping) or any(
        not isinstance(key, str) for key in config
    ):
        raise StructuredLateBlockRunnerError(
            "runner config must be a string-keyed allowlist mapping"
        )
    actual = frozenset(config)
    if actual not in {_CONFIG_ALLOWLIST, _BOUNDED_CONFIG_ALLOWLIST}:
        raise StructuredLateBlockRunnerError(
            "runner config allowlist mismatch: "
            f"missing={sorted(_CONFIG_ALLOWLIST - actual)}, "
            f"extra={sorted(actual - _BOUNDED_CONFIG_ALLOWLIST)}"
        )
    resolved = dict(config)
    if str(resolved["protocol_schema"]) != "p2_min_v1":
        raise StructuredLateBlockRunnerError("protocol_schema must be p2_min_v1")
    if str(resolved["phase2_data_status"]) != "VALIDATED_ONCE":
        raise StructuredLateBlockRunnerError(
            "phase2_data_status must be VALIDATED_ONCE"
        )
    for key in (
        "capsule_id",
        "split_id",
        "row_id",
        "receiver",
        "scenario",
        "checkpoint_path",
        "support_path",
        "query_path",
        "prototype_path",
    ):
        if not str(resolved[key]).strip():
            raise StructuredLateBlockRunnerError(f"{key} must be nonempty")
    if str(resolved["decision_rule"]) != _DECISION_RULE:
        raise StructuredLateBlockRunnerError(
            f"decision_rule must remain {_DECISION_RULE}"
        )
    if str(resolved["candidate"]) not in {"freq_f3_proj", "time_t3"}:
        raise StructuredLateBlockRunnerError("candidate is not preregistered")
    steps = int(resolved["steps"])
    if steps < 1 or steps > MAX_ADAPTATION_STEPS:
        raise StructuredLateBlockRunnerError(
            f"steps must be in [1, {MAX_ADAPTATION_STEPS}]"
        )
    if float(resolved["learning_rate"]) <= 0.0:
        raise StructuredLateBlockRunnerError("learning_rate must be positive")
    if int(resolved["k_shot"]) < 1:
        raise StructuredLateBlockRunnerError("k_shot must be positive")
    minimum = float(resolved.get("min_trainable_fraction", 0.05))
    maximum = float(resolved.get("max_trainable_fraction", 0.15))
    if not 0.0 < minimum <= maximum <= 1.0:
        raise StructuredLateBlockRunnerError(
            "trainable fraction bounds must satisfy 0 < min <= max <= 1"
        )
    resolved["min_trainable_fraction"] = minimum
    resolved["max_trainable_fraction"] = maximum
    int(resolved["seed"])
    return resolved


def _load_npz(
    path: str | Path,
    *,
    allowed: frozenset[str] | None = None,
    label: str = "Phase2",
) -> dict[str, np.ndarray]:
    """Load one explicitly addressed NPZ without pickle/object support."""

    resolved = Path(path)
    if not resolved.is_file() or resolved.suffix.lower() != ".npz":
        raise StructuredLateBlockRunnerError(
            f"Phase2 NPZ input is missing or invalid: {resolved}"
        )
    try:
        with np.load(resolved, allow_pickle=False) as archive:
            names = tuple(str(name) for name in archive.files)
            if allowed is not None:
                actual = frozenset(names)
                if actual != allowed:
                    raise StructuredLateBlockRunnerError(
                        f"{label} payload allowlist mismatch: "
                        f"missing={sorted(allowed - actual)}, "
                        f"extra={sorted(actual - allowed)}"
                    )
            return {name: np.asarray(archive[name]).copy() for name in names}
    except StructuredLateBlockRunnerError:
        raise
    except (OSError, ValueError) as exc:
        raise StructuredLateBlockRunnerError(
            f"cannot load Phase2 NPZ input: {resolved}"
        ) from exc


def _strip_module_prefix(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        (key[7:] if str(key).startswith("module.") else str(key)): value
        for key, value in state.items()
    }


def _checkpoint_args(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw = payload.get("args")
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return dict(raw)
    try:
        return dict(vars(raw))
    except TypeError as exc:
        raise StructuredLateBlockRunnerError(
            "ADV3B02 checkpoint args are not reconstructable"
        ) from exc


def _infer_num_domains(state: Mapping[str, torch.Tensor]) -> int:
    for key in (
        "dom_head.net.3.bias",
        "dom_head.net.3.weight",
        "adv_head.net.3.bias",
        "adv_head.net.3.weight",
    ):
        value = state.get(key)
        if torch.is_tensor(value) and value.ndim >= 1:
            return int(value.shape[0])
    raise StructuredLateBlockRunnerError(
        "strict ADV3B02 reconstruction cannot infer num_domains"
    )


def _infer_num_classes(
    checkpoint_args: Mapping[str, Any],
    state: Mapping[str, torch.Tensor],
) -> int:
    if "num_classes" in checkpoint_args:
        value = int(checkpoint_args["num_classes"])
        if value > 0:
            return value
    for key in (
        "id_backbone.cls_head.head.weight",
        "id_backbone.cls_head.head.W",
    ):
        value = state.get(key)
        if torch.is_tensor(value) and value.ndim == 2:
            return int(value.shape[0])
    raise StructuredLateBlockRunnerError(
        "strict ADV3B02 reconstruction cannot infer num_classes"
    )


def _checkpoint_state(payload: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    for key in ("model", "model_state_dict", "state_dict"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping) and candidate:
            state = _strip_module_prefix(candidate)
            if not all(torch.is_tensor(value) for value in state.values()):
                raise StructuredLateBlockRunnerError(
                    "ADV3B02 checkpoint state contains non-tensor values"
                )
            return state
    raise StructuredLateBlockRunnerError(
        "ADV3B02 checkpoint has no model state_dict"
    )


def _load_frozen_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device,
) -> nn.Module:
    """Rebuild ADV3B02 and require zero missing or unexpected state keys."""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise StructuredLateBlockRunnerError(
            f"ADV3B02 checkpoint is missing: {checkpoint_path}"
        )
    try:
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise StructuredLateBlockRunnerError(
            f"cannot load ADV3B02 checkpoint: {checkpoint_path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise StructuredLateBlockRunnerError("ADV3B02 checkpoint must be a mapping")
    state = _checkpoint_state(payload)
    args = _checkpoint_args(payload)
    num_classes = _infer_num_classes(args, state)
    num_domains = _infer_num_domains(state)
    sample_rate_hz = float(args.get("sample_rate_hz", 0.0))
    if sample_rate_hz <= 0.0:
        sample_rate_hz = 25e6

    # Imported only on the real checkpoint path so unit tests can replace this
    # loader without importing any legacy supervised or predictor runtime.
    from model_dual_cvsincnet import build_dual_model

    model = build_dual_model(
        num_classes,
        num_domains,
        model_size=str(args.get("model_size", "M")),
        dataset=str(args.get("dataset", "wisig")),
        input_len=int(args.get("input_len", 256)),
        sample_rate_hz=sample_rate_hz,
        id_feature_key=str(args.get("id_feature_key", "feat_joint")),
        dom_feature_key=str(args.get("dom_feature_key", "feat_imp")),
        model_variant=str(args.get("model_variant", "lite_c")),
        branch_ablation=str(args.get("branch_ablation", "none")),
        mixstyle_on=bool(args.get("use_mixstyle", False)),
        mixstyle_p=float(args.get("mixstyle_p", 0.3)),
        mixstyle_alpha=float(args.get("mixstyle_alpha", 0.1)),
        mixstyle_eps=float(args.get("mixstyle_eps", 1e-6)),
        mixstyle_layers=str(args.get("mixstyle_layers", "time_down,t1")),
        mixstyle_use_domain_label=bool(args.get("mixstyle_use_domain_label", True)),
        mixstyle_mix=str(args.get("mixstyle_mix", "crossdomain")),
        mixstyle_strength=float(args.get("mixstyle_strength", 1.0)),
        mixstyle_fallback=str(args.get("mixstyle_fallback", "random")),
        domain_branch_ablation=str(args.get("domain_branch_ablation", "same")),
        domain_enhancer=str(args.get("domain_enhancer", "rcn_stats")),
        domain_enhancer_strength=float(args.get("domain_enhancer_strength", 0.35)),
        id_time_stability_mode=str(args.get("id_time_stability_mode", "off")),
        id_freq_stability_mode=str(args.get("id_freq_stability_mode", "off")),
        domain_time_stability_mode=str(
            args.get("domain_time_stability_mode", "off")
        ),
        domain_freq_stability_mode=str(
            args.get("domain_freq_stability_mode", "off")
        ),
        time_stability_channels=int(args.get("time_stability_channels", 8)),
        freq_stability_channels=int(args.get("freq_stability_channels", 4)),
        fast_infer_when_no_aux=bool(args.get("fast_infer_when_no_aux", True)),
        arch_family=str(args.get("arch_family", "cvsincnet")),
    )
    try:
        incompatible = model.load_state_dict(state, strict=False)
    except RuntimeError as exc:
        raise StructuredLateBlockRunnerError(
            "strict ADV3B02 reconstruction failed on tensor shape"
        ) from exc
    missing = tuple(incompatible.missing_keys)
    unexpected = tuple(incompatible.unexpected_keys)
    if missing or unexpected:
        raise StructuredLateBlockRunnerError(
            "strict ADV3B02 reconstruction failed: "
            f"missing={list(missing)}, unexpected={list(unexpected)}"
        )
    if not hasattr(model, "id_backbone") or not callable(
        getattr(model, "_pick_z_id", None)
    ):
        raise StructuredLateBlockRunnerError(
            "ADV3B02 checkpoint lacks the identity backbone/z_id interface"
        )
    model.to(torch.device(device))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model


def _received_iq_tensor(value: np.ndarray, *, label: str) -> torch.Tensor:
    array = np.asarray(value)
    if (
        array.ndim != 3
        or array.shape[0] < 1
        or array.shape[1] != 2
        or array.shape[2] < 1
        or not np.issubdtype(array.dtype, np.number)
        or not np.isfinite(array).all()
    ):
        raise StructuredLateBlockRunnerError(
            f"{label} received_iq must be finite nonempty [N,2,L]"
        )
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    return torch.frombuffer(
        bytearray(contiguous.tobytes(order="C")), dtype=torch.float32
    ).clone().reshape(contiguous.shape)


def _integer_tensor(value: np.ndarray, *, label: str) -> torch.Tensor:
    array = np.asarray(value)
    if (
        array.ndim != 1
        or array.shape[0] < 1
        or not np.issubdtype(array.dtype, np.integer)
    ):
        raise StructuredLateBlockRunnerError(
            f"{label} must be a nonempty integer vector"
        )
    contiguous = np.ascontiguousarray(array, dtype=np.int64)
    return torch.frombuffer(
        bytearray(contiguous.tobytes(order="C")), dtype=torch.int64
    ).clone().reshape(contiguous.shape)


def _prototype_tensors(
    payload: Mapping[str, np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_exact_keys(
        payload,
        _PROTOTYPE_PAYLOAD_ALLOWLIST,
        label="prototype",
    )
    array = np.asarray(payload["prototypes"])
    if (
        array.ndim != 2
        or array.shape[0] < 1
        or array.shape[1] < 1
        or not np.issubdtype(array.dtype, np.number)
        or not np.isfinite(array).all()
    ):
        raise StructuredLateBlockRunnerError(
            "prototype array must be a finite nonempty 2D matrix"
        )
    class_ids = _integer_tensor(payload["class_ids"], label="prototype class_ids")
    if class_ids.shape[0] != array.shape[0]:
        raise StructuredLateBlockRunnerError(
            "prototype matrix and class_ids must align"
        )
    if torch.unique(class_ids).numel() != class_ids.numel():
        raise StructuredLateBlockRunnerError("prototype class_ids must be unique")
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    prototypes = torch.frombuffer(
        bytearray(contiguous.tobytes(order="C")), dtype=torch.float32
    ).clone().reshape(contiguous.shape)
    prototypes.requires_grad_(False)
    return prototypes, class_ids


def _audit_mapping(audit: Any) -> Mapping[str, Any]:
    if isinstance(audit, Mapping):
        return audit
    if is_dataclass(audit):
        return asdict(audit)
    raise StructuredLateBlockRunnerError("adaptation audit must be a mapping/dataclass")


def _state_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if torch.is_tensor(value)
    }


def _assert_state_unchanged(
    model: nn.Module,
    before: Mapping[str, torch.Tensor],
) -> None:
    after = model.state_dict()
    changed = [
        name
        for name, value in before.items()
        if name not in after or not torch.equal(value, after[name].detach())
    ]
    if changed:
        raise StructuredLateBlockRunnerError(
            f"query inference mutated frozen model state: {changed}"
        )


def run_stage2_row(
    config: Mapping[str, Any],
    *,
    output_dir: str | Path,
    device: str | torch.device,
) -> dict[str, Any]:
    """Adapt and predict one fixed Stage2-B row without opening query early."""

    resolved = _validate_config(config)
    destination = Path(output_dir)
    if destination.exists():
        raise StructuredLateBlockRunnerError(
            f"output directory already exists: {destination}"
        )
    target_device = torch.device(device)
    seed = int(resolved["seed"])
    torch.manual_seed(seed)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = _load_frozen_checkpoint(
        resolved["checkpoint_path"],
        device=target_device,
    )

    # Support is opened first.  Query remains unopened throughout adaptation.
    support_payload = _load_npz(
        resolved["support_path"],
        allowed=_SUPPORT_PAYLOAD_ALLOWLIST,
        label="support",
    )
    _validate_exact_keys(
        support_payload,
        _SUPPORT_PAYLOAD_ALLOWLIST,
        label="support",
    )
    support_iq = _received_iq_tensor(
        support_payload["received_iq"], label="support"
    )
    support_labels = _integer_tensor(
        support_payload["support_labels"], label="support_labels"
    )
    if support_labels.shape[0] != support_iq.shape[0]:
        raise StructuredLateBlockRunnerError("support IQ and labels must align")
    _support_classes, support_counts = torch.unique(
        support_labels, sorted=True, return_counts=True
    )
    if torch.any(support_counts != int(resolved["k_shot"])):
        raise StructuredLateBlockRunnerError(
            "support payload must contain exactly K-shot rows per class"
        )

    prototype_payload = _load_npz(
        resolved["prototype_path"],
        allowed=_PROTOTYPE_PAYLOAD_ALLOWLIST,
        label="prototype",
    )
    prototypes, prototype_class_ids = _prototype_tensors(prototype_payload)
    prototype_before = prototypes.clone()
    adaptation_config = StructuredLateBlockConfig(
        candidate=str(resolved["candidate"]),
        steps=int(resolved["steps"]),
        learning_rate=float(resolved["learning_rate"]),
        min_trainable_fraction=float(resolved["min_trainable_fraction"]),
        max_trainable_fraction=float(resolved["max_trainable_fraction"]),
    )
    context = {key: resolved[key] for key in _CONTEXT_KEYS}
    audit = adapt_on_target_support_with_frozen_prototypes(
        model,
        support_iq,
        support_labels,
        frozen_prototypes=prototypes,
        prototype_class_ids=prototype_class_ids,
        context=context,
        config=adaptation_config,
    )
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise StructuredLateBlockRunnerError(
            "adaptation did not return a fully frozen model"
        )
    if not torch.equal(prototypes, prototype_before) or prototypes.requires_grad:
        raise StructuredLateBlockRunnerError(
            "adaptation changed immutable class prototypes"
        )

    # This is deliberately the first query open in the entire row lifecycle.
    query_payload = _load_npz(
        resolved["query_path"],
        allowed=_QUERY_PAYLOAD_ALLOWLIST,
        label="query",
    )
    _validate_exact_keys(
        query_payload,
        _QUERY_PAYLOAD_ALLOWLIST,
        label="query",
    )
    query_iq = _received_iq_tensor(query_payload["received_iq"], label="query")
    query_ids = np.asarray(query_payload["query_ids"])
    if query_ids.ndim != 1 or query_ids.shape[0] != query_iq.shape[0]:
        raise StructuredLateBlockRunnerError("query IDs and IQ rows must align")
    if query_ids.dtype.kind == "O":
        raise StructuredLateBlockRunnerError("query_ids cannot use object/pickle data")
    query_ids = query_ids.astype(str)
    if len(set(query_ids.tolist())) != len(query_ids):
        raise StructuredLateBlockRunnerError("query_ids must be unique")

    query_state_before = _state_snapshot(model)
    predictions, scores = predict_query_with_frozen_prototypes(
        model,
        query_iq,
        frozen_prototypes=prototypes,
        prototype_class_ids=prototype_class_ids,
    )
    _assert_state_unchanged(model, query_state_before)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise StructuredLateBlockRunnerError("query inference unfroze model state")
    if not torch.equal(prototypes, prototype_before) or prototypes.requires_grad:
        raise StructuredLateBlockRunnerError(
            "query inference changed immutable class prototypes"
        )
    if (
        not torch.is_tensor(predictions)
        or predictions.ndim != 1
        or predictions.shape[0] != query_iq.shape[0]
    ):
        raise StructuredLateBlockRunnerError("predictions must align with query rows")
    if (
        not torch.is_tensor(scores)
        or scores.ndim != 2
        or scores.shape[0] != query_iq.shape[0]
        or scores.shape[1] != prototype_class_ids.shape[0]
        or not torch.isfinite(scores).all()
    ):
        raise StructuredLateBlockRunnerError(
            "prediction scores must be finite [queries,classes]"
        )

    destination.mkdir(parents=True, exist_ok=False)
    prediction_path = destination / "predictions.npz"
    partial_path = destination / "predictions.npz.partial"
    with partial_path.open("xb") as handle:
        np.savez(
            handle,
            query_ids=query_ids,
            predicted_class_ids=predictions.detach().cpu().numpy().astype(np.int64),
            scores=scores.detach().cpu().numpy().astype(np.float32),
        )
        handle.flush()
        os.fsync(handle.fileno())
    partial_path.replace(prediction_path)
    audit_values = _audit_mapping(audit)
    return {
        "status": "PREDICTIONS_COMPLETE",
        "row_id": str(resolved["row_id"]),
        "candidate": str(resolved["candidate"]),
        "gradient_updates": int(audit_values["gradient_updates"]),
        "trainable_fraction": float(audit_values.get("trainable_fraction", 0.0)),
        "support_samples": int(support_iq.shape[0]),
        "query_samples": int(query_iq.shape[0]),
        "query_truth_opened": False,
        "query_role_opened": False,
        "checkpoint_load_strict": True,
        "prediction_path": str(prediction_path),
    }


__all__ = [
    "StructuredLateBlockRunnerError",
    "run_stage2_row",
]
