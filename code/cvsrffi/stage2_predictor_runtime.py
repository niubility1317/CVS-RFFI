"""Truth-free Stage2 predictor core for a sealed TorchScript ADV3B02 runtime."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from cvsrffi.stage2_predictor_bundle import open_regular_member_same_fd


ADAPTER_SCHEMA = "cvs.phase2.feature_adapter.v1"
HEAD_SCHEMA = "cvs.phase2.prototype_head.v1"
TTA_SCHEMA = "cvs.phase2.adaptive_rxlight_tta.v1"
RX_LIGHT5_ORDER = (
    "rx_base",
    "rx_shift_m2",
    "rx_shift_p2",
    "rx_cfo_m1e4",
    "rx_cfo_p1e4",
)


class Stage2PredictorRuntimeError(ValueError):
    pass


def _hash_handle(handle) -> tuple[str, int]:
    handle.seek(0)
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
        size += len(chunk)
    handle.seek(0)
    return digest.hexdigest(), size


def _verify_descriptor_handle(handle, descriptor: Mapping[str, Any]) -> None:
    digest, size = _hash_handle(handle)
    if digest != descriptor["sha256"] or size != int(descriptor["size_bytes"]):
        raise Stage2PredictorRuntimeError("sealed runtime artifact digest mismatch")


def load_json_artifact_same_fd(
    package_root: str | Path, descriptor: Mapping[str, Any]
) -> dict[str, Any]:
    with open_regular_member_same_fd(
        Path(package_root), str(descriptor["relative_path"])
    ) as handle:
        _verify_descriptor_handle(handle, descriptor)
        raw = handle.read()
    try:
        value = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage2PredictorRuntimeError("sealed runtime JSON artifact is invalid") from exc
    if not isinstance(value, dict):
        raise Stage2PredictorRuntimeError("sealed runtime JSON root must be an object")
    return value


def load_torchscript_backbone_same_fd(
    package_root: str | Path,
    descriptor: Mapping[str, Any],
    *,
    device: torch.device,
) -> torch.jit.ScriptModule:
    if descriptor.get("schema") != "adv3b02.torchscript_identity_runtime.v1":
        raise Stage2PredictorRuntimeError("checkpoint is not a sealed ADV3B02 TorchScript runtime")
    with open_regular_member_same_fd(
        Path(package_root), str(descriptor["relative_path"])
    ) as handle:
        _verify_descriptor_handle(handle, descriptor)
        handle.seek(0)
        try:
            model = torch.jit.load(handle, map_location=device)
        except Exception as exc:  # pragma: no cover - Torch wraps backend exceptions
            raise Stage2PredictorRuntimeError("failed to load sealed TorchScript runtime") from exc
    model.eval()
    return model


def _validate_resource_bounds(config: Mapping[str, Any]) -> None:
    for key, upper in (
        ("trainable_parameters", 100_000),
        ("adapt_epochs", 40),
        ("persistent_state_bytes", 512 * 1024),
    ):
        value = config.get(key)
        if not isinstance(value, int) or value < 0 or value > upper:
            raise Stage2PredictorRuntimeError(f"adapter resource bound invalid: {key}")


def validate_adapter_config(config: Mapping[str, Any], *, feature_dim: int) -> None:
    common = {
        "schema",
        "mode",
        "trainable_parameters",
        "adapt_epochs",
        "persistent_state_bytes",
        "fft_dim",
        "fft_weight",
    }
    mode = config.get("mode")
    extras: set[str]
    if mode == "identity":
        extras = set()
    elif mode == "diag_affine":
        extras = {"feature_mean", "feature_inv_std", "feature_scale", "feature_bias"}
    elif mode == "low_rank_residual":
        extras = {
            "feature_mean",
            "feature_inv_std",
            "low_rank_down",
            "low_rank_up",
            "residual_scale",
        }
    else:
        raise Stage2PredictorRuntimeError("unsupported feature adapter mode")
    if set(config) != common | extras or config.get("schema") != ADAPTER_SCHEMA:
        raise Stage2PredictorRuntimeError("feature adapter exact schema drift")
    _validate_resource_bounds(config)
    fft_dim = config.get("fft_dim")
    fft_weight = config.get("fft_weight")
    if not isinstance(fft_dim, int) or fft_dim < 0 or fft_dim > 96:
        raise Stage2PredictorRuntimeError("fft_dim must be in [0,96]")
    if not isinstance(fft_weight, (int, float)) or not np.isfinite(float(fft_weight)):
        raise Stage2PredictorRuntimeError("fft_weight must be finite")

    def vector(name: str) -> np.ndarray:
        value = np.asarray(config[name], dtype=np.float32)
        if value.shape != (feature_dim,) or not np.isfinite(value).all():
            raise Stage2PredictorRuntimeError(f"adapter vector drift: {name}")
        return value

    if mode == "diag_affine":
        for name in extras:
            vector(name)
    elif mode == "low_rank_residual":
        vector("feature_mean")
        vector("feature_inv_std")
        down = np.asarray(config["low_rank_down"], dtype=np.float32)
        up = np.asarray(config["low_rank_up"], dtype=np.float32)
        if down.ndim != 2 or down.shape[0] != feature_dim or up.shape != (
            down.shape[1],
            feature_dim,
        ):
            raise Stage2PredictorRuntimeError("low-rank adapter matrix drift")
        if not np.isfinite(down).all() or not np.isfinite(up).all():
            raise Stage2PredictorRuntimeError("low-rank adapter contains non-finite values")
        if not isinstance(config["residual_scale"], (int, float)):
            raise Stage2PredictorRuntimeError("low-rank residual scale drift")


def validate_head_config(config: Mapping[str, Any]) -> None:
    if set(config) != {"schema", "metric", "temperature"}:
        raise Stage2PredictorRuntimeError("prototype head exact schema drift")
    if config.get("schema") != HEAD_SCHEMA or config.get("metric") != "cosine":
        raise Stage2PredictorRuntimeError("only the sealed cosine prototype head is allowed")
    value = config.get("temperature")
    if not isinstance(value, (int, float)) or not np.isfinite(float(value)) or value <= 0:
        raise Stage2PredictorRuntimeError("prototype temperature must be positive and finite")


def validate_tta_config(config: Mapping[str, Any]) -> None:
    common = {"schema", "mode", "base_views", "max_views"}
    mode = config.get("mode")
    if mode == "base_only":
        expected = common
        if config.get("base_views") != 1 or config.get("max_views") != 1:
            raise Stage2PredictorRuntimeError("base-only TTA count drift")
    elif mode == "adaptive_1_3_5":
        expected = common | {
            "base_stop_margin",
            "shift3_stop_margin",
            "shift3_max_disagreement",
            "calibration_scope",
            "uses_query_labels",
            "uses_query_role",
            "uses_class_quota",
        }
        if config.get("base_views") != 1 or config.get("max_views") != 5:
            raise Stage2PredictorRuntimeError("adaptive TTA count drift")
        if config.get("calibration_scope") not in {
            "source_validation",
            "registered_support",
            "source_validation_or_registered_support_only",
        }:
            raise Stage2PredictorRuntimeError("adaptive TTA calibration scope is not legal")
        for key in ("uses_query_labels", "uses_query_role", "uses_class_quota"):
            if config.get(key) is not False:
                raise Stage2PredictorRuntimeError(f"adaptive TTA forbidden access: {key}")
        for key in (
            "base_stop_margin",
            "shift3_stop_margin",
            "shift3_max_disagreement",
        ):
            if not isinstance(config.get(key), (int, float)) or not np.isfinite(
                float(config[key])
            ):
                raise Stage2PredictorRuntimeError(f"adaptive TTA threshold drift: {key}")
    else:
        raise Stage2PredictorRuntimeError("unsupported TTA mode")
    if set(config) != expected or config.get("schema") != TTA_SCHEMA:
        raise Stage2PredictorRuntimeError("adaptive TTA exact schema drift")


def select_nested_support_prefix(
    support_arrays: Mapping[str, np.ndarray], *, k_shot: int, class_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(support_arrays["support_pool_class_indices"], dtype=np.int64)
    ranks = np.asarray(support_arrays["support_pool_rank_within_class"], dtype=np.int64)
    iq = np.asarray(support_arrays["support_pool_leo_weak_iq"], dtype=np.float32)
    tokens = np.asarray(support_arrays["support_pool_tokens"]).astype(str)
    mask = ranks < int(k_shot)
    selected_labels = labels[mask]
    selected_ranks = ranks[mask]
    expected = [(class_index, rank) for class_index in range(class_count) for rank in range(k_shot)]
    if list(zip(selected_labels.tolist(), selected_ranks.tolist())) != expected:
        raise Stage2PredictorRuntimeError("support pool is not the exact nested K prefix")
    if iq[mask].shape[0] != len(expected) or tokens[mask].shape[0] != len(expected):
        raise Stage2PredictorRuntimeError("support nested K tensor count drift")
    return iq[mask], selected_labels, tokens[mask]


def _runtime_forward(
    model: torch.nn.Module,
    rows: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), int(batch_size)):
            batch = torch.from_numpy(np.asarray(rows[start : start + batch_size], dtype=np.float32)).to(device)
            output = model(batch)
            if isinstance(output, dict):
                feature_value = output.get("features")
                logit_value = output.get("logits")
            elif isinstance(output, (tuple, list)) and len(output) == 2:
                feature_value, logit_value = output
            else:
                raise Stage2PredictorRuntimeError(
                    "TorchScript runtime must return {features,logits} or a two-tensor tuple"
                )
            if not torch.is_tensor(feature_value) or not torch.is_tensor(logit_value):
                raise Stage2PredictorRuntimeError("TorchScript feature/logit output drift")
            features.append(feature_value.detach().float().cpu().numpy())
            logits.append(logit_value.detach().float().cpu().numpy())
    feature_array = np.concatenate(features, axis=0).astype(np.float32)
    logit_array = np.concatenate(logits, axis=0).astype(np.float32)
    if feature_array.ndim != 2 or logit_array.ndim != 2:
        raise Stage2PredictorRuntimeError("TorchScript outputs must be rank-2")
    if not np.isfinite(feature_array).all() or not np.isfinite(logit_array).all():
        raise Stage2PredictorRuntimeError("TorchScript outputs contain non-finite values")
    return feature_array, logit_array


def spectral_logmag_sketch(rows: np.ndarray, *, dim: int) -> np.ndarray:
    raw = np.asarray(rows, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[1] != 2 or dim < 1:
        raise Stage2PredictorRuntimeError("FFT descriptor input drift")
    target_x = np.linspace(0.0, 1.0, int(dim), dtype=np.float64)
    result: list[np.ndarray] = []
    for row in raw:
        value = row[0].astype(np.float64) + 1j * row[1].astype(np.float64)
        value -= np.mean(value)
        rms = float(np.sqrt(np.mean(np.abs(value) ** 2)))
        if rms > 1.0e-8:
            value /= rms
        window = np.hanning(value.size)
        if float(np.max(window)) <= 0.0:
            window = np.ones(value.size, dtype=np.float64)
        spectrum = np.fft.fftshift(np.fft.fft(value * window))
        logmag = np.log1p(np.abs(spectrum))
        source_x = np.linspace(0.0, 1.0, logmag.size, dtype=np.float64)
        sketch = np.interp(target_x, source_x, logmag).astype(np.float32)
        sketch -= np.mean(sketch, dtype=np.float64).astype(np.float32)
        sketch /= max(float(np.linalg.norm(sketch)), 1.0e-8)
        result.append(sketch)
    return np.stack(result).astype(np.float32)


def apply_feature_adapter(
    features: np.ndarray, rows: np.ndarray, config: Mapping[str, Any]
) -> np.ndarray:
    value = np.asarray(features, dtype=np.float32)
    validate_adapter_config(config, feature_dim=value.shape[1])
    mode = config["mode"]
    if mode == "identity":
        adapted = value
    else:
        mean = np.asarray(config["feature_mean"], dtype=np.float32)
        inv_std = np.asarray(config["feature_inv_std"], dtype=np.float32)
        normalized = (value - mean) * inv_std
        if mode == "diag_affine":
            adapted = normalized * np.asarray(config["feature_scale"], dtype=np.float32)
            adapted += np.asarray(config["feature_bias"], dtype=np.float32)
        else:
            down = np.asarray(config["low_rank_down"], dtype=np.float32)
            up = np.asarray(config["low_rank_up"], dtype=np.float32)
            adapted = normalized + float(config["residual_scale"]) * (normalized @ down @ up)
    fft_dim = int(config["fft_dim"])
    if fft_dim:
        fft = spectral_logmag_sketch(rows, dim=fft_dim) * float(config["fft_weight"])
        adapted = np.concatenate([adapted, fft], axis=1)
    if not np.isfinite(adapted).all():
        raise Stage2PredictorRuntimeError("adapted feature contains non-finite values")
    return np.asarray(adapted, dtype=np.float32)


def _normalize(value: np.ndarray) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float32)
    return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1.0e-8)


def _prototypes(features: np.ndarray, labels: np.ndarray, class_count: int) -> np.ndarray:
    result = []
    for class_index in range(class_count):
        rows = features[labels == class_index]
        if len(rows) == 0:
            raise Stage2PredictorRuntimeError("registered class lacks support")
        result.append(_normalize(rows).mean(axis=0))
    return _normalize(np.stack(result))


def _scores(features: np.ndarray, prototypes: np.ndarray, temperature: float) -> np.ndarray:
    return (_normalize(features) @ prototypes.T * float(temperature)).astype(np.float32)


def _margin(scores: np.ndarray) -> np.ndarray:
    if scores.shape[1] < 2:
        return np.full(scores.shape[0], np.inf, dtype=np.float32)
    top2 = np.partition(scores, scores.shape[1] - 2, axis=1)[:, -2:]
    return (np.max(top2, axis=1) - np.min(top2, axis=1)).astype(np.float32)


def _receive_views(rows: np.ndarray, names: Sequence[str]) -> dict[str, np.ndarray]:
    raw = np.asarray(rows, dtype=np.float32)
    result: dict[str, np.ndarray] = {}
    for name in names:
        if name == "rx_base":
            result[name] = raw
        elif name == "rx_shift_m2":
            result[name] = np.roll(raw, -2, axis=-1).copy()
        elif name == "rx_shift_p2":
            result[name] = np.roll(raw, 2, axis=-1).copy()
        elif name in {"rx_cfo_m1e4", "rx_cfo_p1e4"}:
            delta = -1.0e-4 if name.endswith("m1e4") else 1.0e-4
            complex_iq = raw[:, 0].astype(np.float64) + 1j * raw[:, 1].astype(np.float64)
            steps = np.arange(raw.shape[-1], dtype=np.float64)[None, :]
            shifted = complex_iq * np.exp(1j * (2.0 * np.pi * delta * steps))
            result[name] = np.stack([shifted.real, shifted.imag], axis=1).astype(np.float32)
        else:
            raise Stage2PredictorRuntimeError(f"unknown receive view: {name}")
    return result


def predict_all_streams(
    model: torch.nn.Module,
    support_arrays: Mapping[str, np.ndarray],
    query_arrays: Mapping[str, np.ndarray],
    *,
    k_shot: int,
    registered_class_count: int,
    new_class_count: int,
    adapter_config: Mapping[str, Any],
    head_config: Mapping[str, Any],
    tta_config: Mapping[str, Any],
    device: torch.device,
    batch_size: int = 256,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    validate_head_config(head_config)
    validate_tta_config(tta_config)
    old_class_count = int(registered_class_count) - int(new_class_count)
    if old_class_count < 1 or registered_class_count < old_class_count:
        raise Stage2PredictorRuntimeError("registered old/new class count drift")
    support_iq, support_y, _support_tokens = select_nested_support_prefix(
        support_arrays, k_shot=k_shot, class_count=registered_class_count
    )
    query_iq = np.asarray(query_arrays["query_leo_weak_iq"], dtype=np.float32)
    started = time.perf_counter()
    support_raw, _support_logits = _runtime_forward(
        model, support_iq, device=device, batch_size=batch_size
    )
    query_raw, direct_logits = _runtime_forward(
        model, query_iq, device=device, batch_size=batch_size
    )
    if direct_logits.shape[1] < old_class_count:
        raise Stage2PredictorRuntimeError("direct ADV3B02 logits do not cover old classes")
    validate_adapter_config(adapter_config, feature_dim=support_raw.shape[1])
    support_candidate = apply_feature_adapter(support_raw, support_iq, adapter_config)
    query_candidate = apply_feature_adapter(query_raw, query_iq, adapter_config)
    identity_prototypes = _prototypes(support_raw, support_y, registered_class_count)
    candidate_prototypes = _prototypes(
        support_candidate, support_y, registered_class_count
    )
    temperature = float(head_config["temperature"])
    stream_sums = {
        "candidate_after": _scores(query_candidate, candidate_prototypes, temperature),
        "candidate_before": _scores(
            query_candidate, candidate_prototypes[:old_class_count], temperature
        ),
        "identity_after": _scores(query_raw, identity_prototypes, temperature),
        "identity_before": _scores(
            query_raw, identity_prototypes[:old_class_count], temperature
        ),
    }
    counts = np.ones(len(query_iq), dtype=np.int64)
    if tta_config["mode"] == "adaptive_1_3_5":
        shift_indices = np.flatnonzero(
            _margin(stream_sums["candidate_after"])
            < float(tta_config["base_stop_margin"])
        ).astype(np.int64)
        if len(shift_indices):
            shift_views = _receive_views(
                query_iq[shift_indices], ("rx_shift_m2", "rx_shift_p2")
            )
            shift_candidate_view_scores: list[np.ndarray] = []
            for name in ("rx_shift_m2", "rx_shift_p2"):
                rows = shift_views[name]
                raw, _logits = _runtime_forward(
                    model, rows, device=device, batch_size=batch_size
                )
                candidate = apply_feature_adapter(raw, rows, adapter_config)
                values = {
                    "candidate_after": _scores(candidate, candidate_prototypes, temperature),
                    "candidate_before": _scores(
                        candidate, candidate_prototypes[:old_class_count], temperature
                    ),
                    "identity_after": _scores(raw, identity_prototypes, temperature),
                    "identity_before": _scores(
                        raw, identity_prototypes[:old_class_count], temperature
                    ),
                }
                shift_candidate_view_scores.append(values["candidate_after"])
                for stream, score in values.items():
                    stream_sums[stream][shift_indices] += score
            counts[shift_indices] = 3
            shift_mean = stream_sums["candidate_after"][shift_indices] / 3.0
            shift_predictions = np.stack(
                [
                    np.argmax(stream_sums["candidate_after"][shift_indices] - sum(shift_candidate_view_scores), axis=1),
                    np.argmax(shift_candidate_view_scores[0], axis=1),
                    np.argmax(shift_candidate_view_scores[1], axis=1),
                ],
                axis=1,
            )
            consensus = np.argmax(shift_mean, axis=1)
            disagreement = np.mean(
                shift_predictions != consensus[:, None], axis=1
            ).astype(np.float32)
            needs_cfo = (
                _margin(shift_mean) < float(tta_config["shift3_stop_margin"])
            ) | (
                disagreement > float(tta_config["shift3_max_disagreement"])
            )
            cfo_indices = shift_indices[needs_cfo]
            if len(cfo_indices):
                cfo_views = _receive_views(
                    query_iq[cfo_indices], ("rx_cfo_m1e4", "rx_cfo_p1e4")
                )
                for name in ("rx_cfo_m1e4", "rx_cfo_p1e4"):
                    rows = cfo_views[name]
                    raw, _logits = _runtime_forward(
                        model, rows, device=device, batch_size=batch_size
                    )
                    candidate = apply_feature_adapter(raw, rows, adapter_config)
                    values = {
                        "candidate_after": _scores(candidate, candidate_prototypes, temperature),
                        "candidate_before": _scores(
                            candidate, candidate_prototypes[:old_class_count], temperature
                        ),
                        "identity_after": _scores(raw, identity_prototypes, temperature),
                        "identity_before": _scores(
                            raw, identity_prototypes[:old_class_count], temperature
                        ),
                    }
                    for stream, score in values.items():
                        stream_sums[stream][cfo_indices] += score
                counts[cfo_indices] = 5

    predictions = {
        stream: np.argmax(scores / counts[:, None], axis=1).astype(np.int64)
        for stream, scores in stream_sums.items()
    }
    predictions["direct"] = np.argmax(
        direct_logits[:, :old_class_count], axis=1
    ).astype(np.int64)
    predictions["shared_view_counts"] = counts
    resource = {
        "schema": "cvs.phase2.predictor_resource_receipt.v2",
        "candidate_query_latency_ms": float((time.perf_counter() - started) * 1000.0 / max(len(query_iq), 1)),
        "mean_backbone_forwards": float(np.mean(counts)),
        "p95_backbone_forwards": float(np.percentile(counts, 95, method="higher")),
        "view1_rate": float(np.mean(counts == 1)),
        "view3_rate": float(np.mean(counts == 3)),
        "view5_rate": float(np.mean(counts == 5)),
        "support_enrollment_backbone_forwards": int(len(support_iq)),
        "query_backbone_forwards": int(np.sum(counts)),
        "fft_descriptor_count": int(
            (len(support_iq) + int(np.sum(counts))) if int(adapter_config["fft_dim"]) else 0
        ),
        "trainable_parameters": int(adapter_config["trainable_parameters"]),
        "adapt_epochs": int(adapter_config["adapt_epochs"]),
        "persistent_state_bytes": int(adapter_config["persistent_state_bytes"]),
        "shared_view_budget_for_all_streams": True,
        "direct_uses_base_view_only": True,
    }
    return predictions, resource
