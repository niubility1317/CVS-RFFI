"""Truth-blind, same-row Phase2-B runner for the tri-R4 meta adapter.

The runner owns one immutable row.  It loads the strict meta bundle, keeps a
frozen DA0 snapshot, adapts only from the validated received-IQ support
carrier, and opens the query payload exactly once after DA1 is frozen.  The
two prediction artifacts use that one query ID vector and contain no truth or
role fields.
"""

from __future__ import annotations

import copy
import inspect
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .meta_checkpoint import load_meta_bundle_strict
from .stage2_meta_adapter_adaptation import (
    MetaAdapterPhase2Config,
    ValidatedTargetSupportBatch,
    adapt_meta_adapter_on_support,
    predict_with_frozen_meta_adapter,
)


_CONFIG_ALLOWLIST = frozenset(
    {
        "candidate_id",
        "bundle_id",
        "protocol_schema",
        "phase2_data_status",
        "capsule_id",
        "split_id",
        "checkpoint_path",
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
_SMOKE_CONFIG_ALLOWLIST = _CONFIG_ALLOWLIST - {"query_path"}
_CONTEXT_KEYS = ("protocol_schema", "phase2_data_status", "capsule_id", "split_id")
_SUPPORT_KEYS = frozenset(
    {"received_iq", "support_labels", "support_physical_ids"}
)
_QUERY_KEYS = frozenset({"received_iq", "query_ids"})
_PROTOTYPE_KEYS = frozenset({"prototypes", "class_ids"})
_DECISION_RULE = "frozen_prototype_cosine_v1"


class MetaAdapterStage2RunnerError(ValueError):
    """Raised when one Phase2 row violates the closed runner contract."""


def _validate_config(
    config: Mapping[str, Any], *, require_query: bool = True
) -> dict[str, Any]:
    if not isinstance(config, Mapping) or any(not isinstance(key, str) for key in config):
        raise MetaAdapterStage2RunnerError(
            "runner config must be a string-keyed allowlist mapping"
        )
    allowed = _CONFIG_ALLOWLIST if require_query else _SMOKE_CONFIG_ALLOWLIST
    actual = frozenset(config)
    if actual != allowed:
        raise MetaAdapterStage2RunnerError(
            "runner config allowlist mismatch: "
            f"missing={sorted(allowed - actual)} extra={sorted(actual - allowed)}"
        )
    resolved = dict(config)
    for key in ("candidate_id", "bundle_id"):
        if not isinstance(resolved[key], str) or not resolved[key].strip():
            raise MetaAdapterStage2RunnerError(f"{key} must be a nonempty string")
    if resolved["protocol_schema"] != "p2_min_v1":
        raise MetaAdapterStage2RunnerError("protocol_schema must be p2_min_v1")
    if resolved["phase2_data_status"] != "VALIDATED_ONCE":
        raise MetaAdapterStage2RunnerError(
            "phase2_data_status must be VALIDATED_ONCE"
        )
    for key in (
        "capsule_id",
        "split_id",
        "checkpoint_path",
        "support_path",
        "prototype_path",
        "receiver",
        "scenario",
        "operating_point",
    ):
        if not str(resolved[key]).strip():
            raise MetaAdapterStage2RunnerError(f"{key} must be nonempty")
    if require_query and not str(resolved["query_path"]).strip():
        raise MetaAdapterStage2RunnerError("query_path must be nonempty")
    steps = resolved["steps"]
    if isinstance(steps, bool) or not isinstance(steps, int) or steps != 3:
        raise MetaAdapterStage2RunnerError(
            "formal Phase2 meta adapter steps must be exactly 3"
        )
    k_shot = resolved["k_shot"]
    if isinstance(k_shot, bool) or not isinstance(k_shot, int) or k_shot < 1:
        raise MetaAdapterStage2RunnerError("k_shot must be a positive integer")
    seed = resolved["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise MetaAdapterStage2RunnerError("seed must be an integer")
    return resolved


def _validate_exact_keys(
    payload: Mapping[str, Any], allowed: frozenset[str], *, label: str
) -> None:
    actual = frozenset(payload)
    if actual != allowed:
        raise MetaAdapterStage2RunnerError(
            f"{label} payload allowlist mismatch: "
            f"missing={sorted(allowed - actual)} extra={sorted(actual - allowed)}"
        )


def _load_npz(
    path: str | Path,
    *,
    allowed: frozenset[str],
    label: str,
) -> dict[str, np.ndarray]:
    resolved = Path(path)
    if resolved.is_symlink() or not resolved.is_file() or resolved.suffix.lower() != ".npz":
        raise MetaAdapterStage2RunnerError(
            f"{label} NPZ input is missing or invalid: {resolved}"
        )
    try:
        with np.load(resolved, allow_pickle=False) as archive:
            names = tuple(str(name) for name in archive.files)
            _validate_exact_keys(dict.fromkeys(names), allowed, label=label)
            return {name: np.asarray(archive[name]).copy() for name in names}
    except MetaAdapterStage2RunnerError:
        raise
    except (OSError, ValueError) as exc:
        raise MetaAdapterStage2RunnerError(
            f"cannot load {label} NPZ input: {resolved}"
        ) from exc


def _received_iq_tensor(value: np.ndarray, *, label: str) -> Tensor:
    array = np.asarray(value)
    if (
        array.ndim != 3
        or array.shape[0] < 1
        or array.shape[1] != 2
        or array.shape[2] < 1
        or not np.issubdtype(array.dtype, np.number)
        or not np.isfinite(array).all()
    ):
        raise MetaAdapterStage2RunnerError(
            f"{label} received_iq must be finite nonempty [N,2,L]"
        )
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)).clone()


def _integer_tensor(value: np.ndarray, *, label: str) -> Tensor:
    array = np.asarray(value)
    if (
        array.ndim != 1
        or array.shape[0] < 1
        or not np.issubdtype(array.dtype, np.integer)
    ):
        raise MetaAdapterStage2RunnerError(
            f"{label} must be a nonempty integer vector"
        )
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.int64)).clone()


def _prototype_tensors(
    payload: Mapping[str, np.ndarray],
) -> tuple[Tensor, Tensor]:
    _validate_exact_keys(payload, _PROTOTYPE_KEYS, label="prototype")
    array = np.asarray(payload["prototypes"])
    if (
        array.ndim != 2
        or array.shape[0] < 1
        or array.shape[1] < 1
        or not np.issubdtype(array.dtype, np.number)
        or not np.isfinite(array).all()
    ):
        raise MetaAdapterStage2RunnerError(
            "prototype array must be a finite nonempty 2D matrix"
        )
    class_ids = _integer_tensor(payload["class_ids"], label="prototype class_ids")
    if class_ids.shape[0] != array.shape[0]:
        raise MetaAdapterStage2RunnerError(
            "prototype matrix and class_ids must align"
        )
    if torch.unique(class_ids).numel() != class_ids.numel():
        raise MetaAdapterStage2RunnerError("prototype class_ids must be unique")
    prototypes = torch.from_numpy(
        np.ascontiguousarray(array, dtype=np.float32)
    ).clone()
    prototypes.requires_grad_(False)
    if bool((torch.linalg.vector_norm(prototypes, dim=1) <= 0).any()):
        raise MetaAdapterStage2RunnerError(
            "prototype rows must be non-zero frozen class means"
        )
    return prototypes, class_ids


def _bundle_prototype_tensors(audit: Any) -> tuple[Tensor, Tensor]:
    raw_prototypes = _audit_value(audit, "prototypes", None)
    raw_mapping = _audit_value(audit, "class_mapping", None)
    if raw_prototypes is None or not isinstance(raw_mapping, Mapping) or not raw_mapping:
        raise MetaAdapterStage2RunnerError(
            "strict bundle audit must expose class_mapping and prototypes"
        )
    if torch.is_tensor(raw_prototypes):
        try:
            class_ids = torch.tensor(
                sorted(int(key) for key in raw_mapping), dtype=torch.long
            )
        except (TypeError, ValueError) as exc:
            raise MetaAdapterStage2RunnerError(
                "strict bundle class_mapping keys must be integer class IDs"
            ) from exc
        prototypes = raw_prototypes.detach().cpu().to(dtype=torch.float32).clone()
    elif isinstance(raw_prototypes, Mapping):
        try:
            class_ids_list = sorted(int(key) for key in raw_prototypes)
            prototypes = torch.stack(
                [torch.as_tensor(raw_prototypes[str(class_id)]) for class_id in class_ids_list]
            ).detach().cpu().to(dtype=torch.float32)
            class_ids = torch.tensor(class_ids_list, dtype=torch.long)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise MetaAdapterStage2RunnerError(
                "strict bundle prototypes are not a valid class mapping"
            ) from exc
        if len(raw_mapping) != len(class_ids_list):
            raise MetaAdapterStage2RunnerError(
                "strict bundle class_mapping and prototypes must align"
            )
    else:
        raise MetaAdapterStage2RunnerError(
            "strict bundle prototypes must be a tensor or class mapping"
        )
    if prototypes.ndim != 2 or prototypes.size(0) != class_ids.numel():
        raise MetaAdapterStage2RunnerError(
            "strict bundle class_mapping and prototypes must align"
        )
    if not bool(torch.isfinite(prototypes).all()) or bool(
        (torch.linalg.vector_norm(prototypes, dim=1) <= 0).any()
    ):
        raise MetaAdapterStage2RunnerError(
            "strict bundle prototypes must be finite non-zero class means"
        )
    return prototypes, class_ids


def _query_ids(value: np.ndarray, *, expected_rows: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.shape[0] != expected_rows:
        raise MetaAdapterStage2RunnerError(
            "query_ids must be a one-dimensional vector aligned with query IQ"
        )
    if array.dtype.kind not in {"U", "S"}:
        raise MetaAdapterStage2RunnerError(
            "query_ids must use a non-object string dtype"
        )
    result = array.astype(str)
    if any(not item.strip() for item in result.tolist()):
        raise MetaAdapterStage2RunnerError("query_ids must be nonempty strings")
    if len(set(result.tolist())) != result.shape[0]:
        raise MetaAdapterStage2RunnerError("query_ids must be unique")
    return result


def _support_physical_ids(
    value: np.ndarray, *, expected_rows: int
) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.ndim != 1 or array.shape[0] != expected_rows:
        raise MetaAdapterStage2RunnerError(
            "support_physical_ids must be a one-dimensional vector aligned "
            "with support IQ"
        )
    if array.dtype.kind not in {"U", "S"}:
        raise MetaAdapterStage2RunnerError(
            "support_physical_ids must use a non-object string dtype"
        )
    result = array.astype(str)
    values = result.tolist()
    if any(not item.strip() for item in values):
        raise MetaAdapterStage2RunnerError(
            "support_physical_ids must contain non-empty physical IDs"
        )
    if len(set(values)) != len(values):
        raise MetaAdapterStage2RunnerError(
            "support_physical_ids must be non-empty and unique"
        )
    return tuple(values)


def _audit_value(audit: Any, key: str, default: Any = None) -> Any:
    if isinstance(audit, Mapping):
        return audit.get(key, default)
    return getattr(audit, key, default)


def _require_strict_audit(audit: Any) -> dict[str, Any]:
    strict = _audit_value(audit, "checkpoint_load_strict", False)
    if strict is not True:
        raise MetaAdapterStage2RunnerError(
            "meta bundle must report checkpoint_load_strict=true"
        )
    fraction = float(_audit_value(audit, "trainable_fraction", 0.0))
    if not np.isfinite(fraction) or fraction > 0.01:
        raise MetaAdapterStage2RunnerError(
            "meta bundle trainable parameter fraction exceeds 1%"
        )
    return {
        "checkpoint_load_strict": True,
        "trainable_fraction": fraction,
        "base_checkpoint_id": _audit_value(audit, "base_checkpoint_id"),
        "class_mapping": _audit_value(audit, "class_mapping"),
        "prototypes": _audit_value(audit, "prototypes"),
    }


def _state_snapshot(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if torch.is_tensor(value)
    }


def _state_diff_count(
    before: Mapping[str, Tensor], after_model: nn.Module
) -> int:
    after = after_model.state_dict()
    return sum(
        1
        for name, value in before.items()
        if name not in after or not torch.equal(value, after[name].detach())
    )


def _freeze_model(model: nn.Module) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _model_device_dtype(model: nn.Module) -> tuple[torch.device, torch.dtype]:
    for parameter in model.parameters():
        if parameter.is_floating_point():
            return parameter.device, parameter.dtype
    for buffer in model.buffers():
        if buffer.is_floating_point():
            return buffer.device, buffer.dtype
    raise MetaAdapterStage2RunnerError("model has no floating-point state")


def _forward_kwargs(model: nn.Module) -> dict[str, object]:
    try:
        parameters = inspect.signature(model.forward).parameters
    except (TypeError, ValueError) as exc:
        raise MetaAdapterStage2RunnerError(
            "cannot inspect meta bundle model.forward signature"
        ) from exc
    kwargs: dict[str, object] = {}
    if "return_aux" in parameters:
        kwargs["return_aux"] = True
    label_names = [name for name in ("y", "y_tx") if name in parameters]
    if len(label_names) > 1:
        raise MetaAdapterStage2RunnerError(
            "model.forward exposes ambiguous label arguments"
        )
    if label_names:
        kwargs[label_names[0]] = None
    return kwargs


def _extract_embedding(outputs: Any, *, batch_size: int) -> Tensor:
    if torch.is_tensor(outputs):
        embedding = outputs
    elif isinstance(outputs, Mapping):
        preferred: list[str] = []
        if "z_id" in outputs:
            z_id_key = outputs.get("z_id_key")
            if isinstance(z_id_key, str) and f"id_{z_id_key}" in outputs:
                preferred.append(f"id_{z_id_key}")
            preferred.append("z_id")
        preferred.extend(
            key
            for key in (
                "feat_cls",
                "id_feat_joint",
                "embedding",
                "features",
                "feature",
                "feat_joint",
                "base",
            )
            if key not in preferred
        )
        embedding = None
        for key in preferred:
            if key in outputs:
                value = outputs[key]
                if not torch.is_tensor(value):
                    raise MetaAdapterStage2RunnerError(
                        f"model embedding key {key!r} must be a tensor"
                    )
                embedding = value
                break
        if embedding is None:
            raise MetaAdapterStage2RunnerError(
                "model output lacks a supported identity embedding"
            )
    else:
        raise MetaAdapterStage2RunnerError(
            "meta bundle model must return a tensor or mapping"
        )
    if (
        embedding.ndim != 2
        or embedding.size(0) != batch_size
        or not embedding.is_floating_point()
        or not bool(torch.isfinite(embedding).all())
    ):
        raise MetaAdapterStage2RunnerError(
            "model embedding must be finite floating-point [batch, dimension]"
        )
    return embedding


def _cosine_logits(embedding: Tensor, prototypes: Tensor) -> Tensor:
    if embedding.size(1) != prototypes.size(1):
        raise MetaAdapterStage2RunnerError(
            "model embedding dimension must match frozen prototypes"
        )
    return F.normalize(embedding, dim=1) @ F.normalize(
        prototypes.to(device=embedding.device, dtype=embedding.dtype), dim=1
    ).transpose(0, 1)


@torch.no_grad()
def _predict_with_scores(
    model: nn.Module,
    query_iq: Tensor,
    prototypes: Tensor,
    class_ids: Tensor,
) -> tuple[Tensor, Tensor]:
    inference_model = copy.deepcopy(model)
    _freeze_model(inference_model)
    device, dtype = _model_device_dtype(inference_model)
    query = query_iq.detach().to(device=device, dtype=dtype)
    outputs = inference_model(query, **_forward_kwargs(inference_model))
    embedding = _extract_embedding(outputs, batch_size=query.size(0))
    scores = _cosine_logits(embedding, prototypes.detach())
    predicted = class_ids.to(device=scores.device, dtype=torch.long)[
        scores.argmax(dim=1)
    ]
    return predicted.detach().cpu(), scores.detach().cpu()


def _validate_support(
    support_iq: Tensor,
    support_labels: Tensor,
    class_ids: Tensor,
    *,
    k_shot: int,
) -> None:
    if support_iq.size(0) != support_labels.size(0):
        raise MetaAdapterStage2RunnerError("support IQ and labels must align")
    if torch.unique(support_labels).numel() != class_ids.numel():
        raise MetaAdapterStage2RunnerError(
            "support labels must cover exactly the registered prototype classes"
        )
    for class_id in class_ids.tolist():
        count = int((support_labels == int(class_id)).sum().item())
        if count != k_shot:
            raise MetaAdapterStage2RunnerError(
                "support payload must contain exactly K-shot rows per class"
            )


def _load_bundle(
    resolved: Mapping[str, Any],
    *,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    model, bundle_audit = load_meta_bundle_strict(
        resolved["checkpoint_path"], device
    )
    audit = _require_strict_audit(bundle_audit)
    if not isinstance(model, nn.Module):
        raise MetaAdapterStage2RunnerError("strict meta bundle did not return a model")
    return model, audit


def _snapshot_frozen_model(model: nn.Module) -> nn.Module:
    snapshot = copy.deepcopy(model)
    _freeze_model(snapshot)
    return snapshot


def _load_support_and_prototypes(
    resolved: Mapping[str, Any],
    bundle_audit: Any | None = None,
) -> tuple[ValidatedTargetSupportBatch, Tensor, Tensor]:

    if bundle_audit is None:
        _unused_model, raw_audit = load_meta_bundle_strict(
            resolved["checkpoint_path"], torch.device("cpu")
        )
        del _unused_model
        bundle_audit = _require_strict_audit(raw_audit)

    support_payload = _load_npz(
        resolved["support_path"], allowed=_SUPPORT_KEYS, label="support"
    )
    support_iq = _received_iq_tensor(support_payload["received_iq"], label="support")
    support_labels = _integer_tensor(
        support_payload["support_labels"], label="support_labels"
    )
    prototype_payload = _load_npz(
        resolved["prototype_path"], allowed=_PROTOTYPE_KEYS, label="prototype"
    )
    prototypes, class_ids = _prototype_tensors(prototype_payload)
    bundle_prototypes, bundle_class_ids = _bundle_prototype_tensors(bundle_audit)
    if not torch.equal(class_ids, bundle_class_ids) or not torch.equal(
        prototypes, bundle_prototypes
    ):
        raise MetaAdapterStage2RunnerError(
            "external class_mapping/prototypes must exactly match strict bundle prototypes"
        )
    _validate_support(
        support_iq,
        support_labels,
        class_ids,
        k_shot=int(resolved["k_shot"]),
    )
    physical_ids = _support_physical_ids(
        support_payload["support_physical_ids"],
        expected_rows=int(support_iq.size(0)),
    )
    receiver = resolved["receiver"]
    context = {key: resolved[key] for key in _CONTEXT_KEYS}
    support_batch = ValidatedTargetSupportBatch(
        received_iq=support_iq,
        labels=support_labels,
        support_physical_ids=physical_ids,
        receiver_id=receiver,
        context=context,
    )
    return support_batch, prototypes, class_ids


def _load_bundle_and_support(
    resolved: Mapping[str, Any],
    *,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any], ValidatedTargetSupportBatch, Tensor, Tensor]:
    """Compatibility wrapper for callers that need the combined inputs."""

    model, audit = _load_bundle(resolved, device=device)
    support_batch, prototypes, class_ids = _load_support_and_prototypes(resolved, audit)
    return model, audit, support_batch, prototypes, class_ids


def _adapt(
    model: nn.Module,
    support_batch: ValidatedTargetSupportBatch,
    prototypes: Tensor,
    class_ids: Tensor,
    resolved: Mapping[str, Any],
) -> Any:
    prototype_before = prototypes.detach().clone()
    handle = adapt_meta_adapter_on_support(
        model,
        support_batch,
        prototypes,
        class_ids,
        MetaAdapterPhase2Config(
            expected_capsule_id=str(resolved["capsule_id"]),
            expected_split_id=str(resolved["split_id"]),
        ),
    )
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise MetaAdapterStage2RunnerError(
            "support adaptation did not return a fully frozen DA1 model"
        )
    if not torch.equal(prototypes, prototype_before) or prototypes.requires_grad:
        raise MetaAdapterStage2RunnerError(
            "support adaptation changed immutable prototypes"
        )
    audit = getattr(handle, "audit", None)
    updates = int(_audit_value(audit, "gradient_updates", -1))
    if updates != 3:
        raise MetaAdapterStage2RunnerError(
            "formal Phase2 support adaptation must perform exactly three updates"
        )
    return handle


def _write_prediction(
    path: Path,
    *,
    query_ids: np.ndarray,
    predicted_class_ids: Tensor,
    scores: Tensor,
) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"prediction artifact already exists: {path}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"prediction temporary artifact already exists: {temporary}")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            query_ids=query_ids,
            predicted_class_ids=predicted_class_ids.detach()
            .cpu()
            .numpy()
            .astype(np.int64),
            scores=scores.detach().cpu().numpy().astype(np.float32),
        )
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"prediction artifact appeared during write: {path}")
    os.replace(temporary, path)


def _write_json_atomically(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"JSON artifact already exists: {path}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"JSON temporary artifact already exists: {temporary}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"JSON artifact appeared during write: {path}")
    os.replace(temporary, path)


def _write_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    _write_json_atomically(path, receipt)


def _safe_failure_message(error: BaseException) -> str:
    message = str(error)
    return re.sub(r"truth|role", "[redacted]", message, flags=re.IGNORECASE)


def _write_failure_receipt(
    destination: Path,
    error: BaseException,
    completed_stages: list[str],
) -> None:
    payload = {
        "status": "FAILED",
        "error_type": type(error).__name__,
        "error_message": _safe_failure_message(error),
        "completed_stages": list(completed_stages),
    }
    _write_json_atomically(destination / "failure_receipt.json", payload)


def run_meta_adapter_stage2_row(
    config: Mapping[str, Any],
    output_dir: str | Path,
    device: str | torch.device,
) -> Mapping[str, Any]:
    """Run one DA0_REG0/DA1_REG0 row with query opened only after DA1 freeze."""

    resolved = _validate_config(config, require_query=True)
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"output directory already exists: {destination}")

    target_device = torch.device(device)
    seed = int(resolved["seed"])
    torch.manual_seed(seed)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Keep the protocol order closed: strict bundle, frozen DA0 snapshot,
    # support/prototype opening, formal adaptation, frozen DA1, then query.
    model, bundle_audit = _load_bundle(resolved, device=target_device)
    da0_model = _snapshot_frozen_model(model)
    support_batch, prototypes, class_ids = _load_support_and_prototypes(resolved, bundle_audit)
    handle = _adapt(model, support_batch, prototypes, class_ids, resolved)

    # The query payload is intentionally opened once, at this point only.
    query_payload = _load_npz(
        resolved["query_path"], allowed=_QUERY_KEYS, label="query"
    )
    query_iq = _received_iq_tensor(query_payload["received_iq"], label="query")
    query_ids = _query_ids(
        query_payload["query_ids"], expected_rows=int(query_iq.size(0))
    )
    query_state_before_da0 = _state_snapshot(da0_model)
    query_state_before_da1 = _state_snapshot(handle.model)
    prototype_before = prototypes.detach().clone()

    da0_predictions, da0_scores = _predict_with_scores(
        da0_model, query_iq, prototypes, class_ids
    )
    da1_predictions = predict_with_frozen_meta_adapter(
        handle,
        query_iq,
        prototypes,
        class_ids,
    )
    if not torch.is_tensor(da1_predictions):
        raise MetaAdapterStage2RunnerError(
            "DA1 prediction must be a tensor of class IDs"
        )
    _, da1_scores = _predict_with_scores(
        handle.model, query_iq, prototypes, class_ids
    )
    da1_predictions = da1_predictions.detach().cpu().to(dtype=torch.long)
    if da1_predictions.ndim != 1 or da1_predictions.numel() != query_iq.size(0):
        raise MetaAdapterStage2RunnerError(
            "DA1 predictions must align with query rows"
        )
    if da0_predictions.shape != da1_predictions.shape:
        raise MetaAdapterStage2RunnerError(
            "DA0/DA1 predictions must have the same query row shape"
        )
    if da0_scores.shape != da1_scores.shape or not bool(torch.isfinite(da1_scores).all()):
        raise MetaAdapterStage2RunnerError(
            "DA0/DA1 score matrices must be finite and shape-aligned"
        )

    query_state_update_count = _state_diff_count(
        query_state_before_da0, da0_model
    ) + _state_diff_count(query_state_before_da1, handle.model)
    if query_state_update_count:
        raise MetaAdapterStage2RunnerError(
            "query inference mutated frozen model state"
        )
    if not torch.equal(prototypes, prototype_before) or prototypes.requires_grad:
        raise MetaAdapterStage2RunnerError(
            "query inference changed immutable prototypes"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=False, exist_ok=False)
    da0_path = destination / "predictions_DA0_REG0.npz"
    da1_path = destination / "predictions_DA1_REG0.npz"
    receipt_path = destination / "receipt.json"
    completed_stages: list[str] = []
    try:
        # All predictions are already complete in memory.  Each artifact is
        # written to a sibling temporary file and atomically renamed; a
        # failed write leaves the partial/temp artifact for diagnosis.
        _write_prediction(
            da0_path,
            query_ids=query_ids,
            predicted_class_ids=da0_predictions,
            scores=da0_scores,
        )
        completed_stages.append("DA0_REG0")
        _write_prediction(
            da1_path,
            query_ids=query_ids,
            predicted_class_ids=da1_predictions,
            scores=da1_scores,
        )
        completed_stages.append("DA1_REG0")
        adaptation_audit = getattr(handle, "audit", None)
        receipt: dict[str, Any] = {
            "status": "PREDICTIONS_COMPLETE",
            "states": ["DA0_REG0", "DA1_REG0"],
            "candidate_id": resolved["candidate_id"],
            "bundle_id": resolved["bundle_id"],
            "registered_class_ids": class_ids.detach().cpu().tolist(),
            "protocol_schema": resolved["protocol_schema"],
            "phase2_data_status": resolved["phase2_data_status"],
            "capsule_id": resolved["capsule_id"],
            "split_id": resolved["split_id"],
            "receiver": resolved["receiver"],
            "scenario": resolved["scenario"],
            "operating_point": resolved["operating_point"],
            "seed": int(resolved["seed"]),
            "k_shot": int(resolved["k_shot"]),
            "steps": 3,
            "checkpoint_load_strict": bundle_audit["checkpoint_load_strict"],
            "trainable_fraction": float(
                _audit_value(
                    adaptation_audit,
                    "trainable_fraction",
                    bundle_audit["trainable_fraction"],
                )
            ),
            "backward_count": int(
                _audit_value(adaptation_audit, "gradient_updates", 3)
            ),
            "support_samples": int(support_batch.received_iq.size(0)),
            "query_samples": int(query_iq.size(0)),
            "query_opened_before_adaptation": False,
            "query_opened": True,
            "source_opened": False,
            "query_truth_opened": False,
            "query_role_opened": False,
            "query_state_update_count": 0,
            "decision_rule": _DECISION_RULE,
            "states_same_row": True,
            "query_ids": query_ids.tolist(),
            "prediction_paths": {
                "DA0_REG0": str(da0_path),
                "DA1_REG0": str(da1_path),
            },
            "receipt_path": str(receipt_path),
        }
        _write_receipt(receipt_path, receipt)
        completed_stages.append("receipt")
    except Exception as exc:
        try:
            _write_failure_receipt(destination, exc, completed_stages)
        except Exception:
            # Preserve the original artifact failure; a filesystem failure
            # while recording diagnostics cannot be turned into completion.
            pass
        raise
    return receipt


__all__ = [
    "MetaAdapterStage2RunnerError",
    "run_meta_adapter_stage2_row",
]
