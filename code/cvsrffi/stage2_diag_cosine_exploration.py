"""Strict target-support-only diag-cosine exploration on sealed LEO_weak packages.

This module deliberately has no dataset selector, raw/clean loader, source
bank, source-logit, truth-sidecar, role-oracle, quota, or scorer interface.
It consumes one verified enrollment package and its matched apply package,
fits on registered support only, freezes the head, and then emits an immutable
unlabeled prediction artifact.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from cvsrffi.phase2_runtime_contract import PHASE2_FULL_CONTRACT
from cvsrffi.somph_diagnostic_bundle_loader import (
    load_verified_somph_predictor_bundle,
)
from cvsrffi.somph_predictor_bundle import (
    APPLY_ONLY,
    ENROLLMENT_ONLY,
    FORMAL_LEO_WEAK_SCENARIOS,
)
from cvsrffi.stage2_predictor_runtime import load_torchscript_backbone_same_fd


EPS = 1.0e-8
FFT_DIM = 96
RF_STAT_DIM = 32
AUXILIARY_WEIGHT = 4.0
ADAPTATION_EPOCHS = 20
LEARNING_RATE = 0.01
BATCH_SIZE = 32
TEMPERATURE = 18.0
PROTOTYPE_ANCHOR_WEIGHT = 0.05
FEATURE_NOISE_STD = 0.01
WEIGHT_DECAY = 0.002
MAX_TRAINABLE_PARAMETERS = 50_000
MAX_PERSISTENT_STATE_BYTES = 256 * 1024
FEATURE_DIM = 160
LOG_SCALE_LIMIT = 1.5
FFT_LOG_SCALE_LIMIT = math.log(1.5)
CANDIDATE_D1 = "d1_historical_diag_fftrf"
CANDIDATE_D1_B0_CAP = "d1_b0_cap_diag_fftrf"
CANDIDATE_D2 = "d2_fixed_proto_diag_fftrf"
CANDIDATE_D3_SCENARIO_OLDLOCK_NEWFIT = "d3_scenario_oldlock_newfit"
CANDIDATES = (
    CANDIDATE_D1,
    CANDIDATE_D1_B0_CAP,
    CANDIDATE_D2,
    CANDIDATE_D3_SCENARIO_OLDLOCK_NEWFIT,
)


class DiagCosineExplorationError(ValueError):
    """Raised when the strict exploration route fails closed."""


@dataclass(frozen=True)
class DiagCosineState:
    candidate: str
    classes: np.ndarray
    log_scale: np.ndarray
    weights: np.ndarray
    bias: np.ndarray
    trace: tuple[dict[str, Any], ...]
    resource: dict[str, Any]


@dataclass(frozen=True)
class ScenarioDiagCosineState:
    candidate: str
    scenarios: np.ndarray
    classes: np.ndarray
    log_scale: np.ndarray
    weights: np.ndarray
    bias: np.ndarray
    new_offset: np.ndarray
    old_class_count: int
    trace: tuple[dict[str, Any], ...]
    resource: dict[str, Any]


def _canonical_json_bytes(value: Mapping[str, Any] | list[Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), EPS)


def log_scale_bounds(
    candidate: str, feature_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return locked per-feature log-scale bounds for one candidate."""

    if candidate not in CANDIDATES:
        raise DiagCosineExplorationError(
            f"unsupported diag-cosine candidate: {candidate}"
        )
    dimension = int(feature_dim)
    if dimension < 1:
        raise DiagCosineExplorationError("diag-cosine feature dimension is invalid")
    lower = np.full(dimension, -LOG_SCALE_LIMIT, dtype=np.float32)
    upper = np.full(dimension, LOG_SCALE_LIMIT, dtype=np.float32)
    if candidate == CANDIDATE_D1_B0_CAP:
        expected = FEATURE_DIM + FFT_DIM + RF_STAT_DIM
        if dimension != expected:
            raise DiagCosineExplorationError(
                "D1-B0-Cap requires z_id160 plus FFT96 plus RF32"
            )
        fft_start = FEATURE_DIM
        fft_stop = FEATURE_DIM + FFT_DIM
        lower[fft_start:fft_stop] = -FFT_LOG_SCALE_LIMIT
        upper[fft_start:fft_stop] = FFT_LOG_SCALE_LIMIT
    return lower, upper


def spectral_logmag_sketch(rows: np.ndarray, *, dim: int = FFT_DIM) -> np.ndarray:
    """Compute the historical FFT descriptor from already-overlaid IQ only."""

    raw = np.asarray(rows, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[1] != 2 or int(dim) < 1:
        raise DiagCosineExplorationError("FFT descriptor expects finite [N,2,T] IQ")
    if not np.isfinite(raw).all():
        raise DiagCosineExplorationError("FFT descriptor input contains non-finite IQ")
    target_x = np.linspace(0.0, 1.0, int(dim), dtype=np.float64)
    result: list[np.ndarray] = []
    for row in raw:
        value = row[0].astype(np.float64) + 1j * row[1].astype(np.float64)
        value -= np.mean(value)
        rms = float(np.sqrt(np.mean(np.abs(value) ** 2)))
        if rms > EPS:
            value /= rms
        window = np.hanning(value.size)
        if float(np.max(window)) <= 0.0:
            window = np.ones(value.size, dtype=np.float64)
        spectrum = np.fft.fftshift(np.fft.fft(value * window))
        logmag = np.log1p(np.abs(spectrum))
        source_x = np.linspace(0.0, 1.0, logmag.size, dtype=np.float64)
        sketch = np.interp(target_x, source_x, logmag).astype(np.float32)
        sketch -= np.mean(sketch, dtype=np.float64).astype(np.float32)
        sketch /= max(float(np.linalg.norm(sketch)), EPS)
        result.append(sketch)
    return np.stack(result).astype(np.float32)


def rf_statistics(rows: np.ndarray) -> np.ndarray:
    """Compute the historical 32-D gain-normalized descriptor from LEO_weak IQ."""

    raw = np.asarray(rows, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[1] != 2 or not np.isfinite(raw).all():
        raise DiagCosineExplorationError("RF descriptor expects finite [N,2,T] IQ")
    result: list[np.ndarray] = []
    for row in raw:
        z = row[0].astype(np.float64) + 1j * row[1].astype(np.float64)
        rms = float(np.sqrt(np.mean(np.abs(z) ** 2)))
        if rms > EPS:
            z /= rms
        centered = z - np.mean(z)
        centered_rms = float(np.sqrt(np.mean(np.abs(centered) ** 2)))
        if centered_rms > EPS:
            centered /= centered_rms

        i = z.real
        q = z.imag
        std_i = float(np.std(i))
        std_q = float(np.std(q))
        iq_corr = (
            float(np.mean((i - np.mean(i)) * (q - np.mean(q))) / (std_i * std_q))
            if std_i > EPS and std_q > EPS
            else 0.0
        )
        amp = np.abs(z)
        amp_mean = float(np.mean(amp))
        amp_std = float(np.std(amp))
        amp_centered = amp - amp_mean
        amp_skew = float(np.mean(amp_centered**3) / max(amp_std**3, EPS))
        amp_kurt = float(np.mean(amp_centered**4) / max(amp_std**4, EPS))
        values: list[float] = [
            float(np.mean(i)),
            float(np.mean(q)),
            std_i,
            std_q,
            iq_corr,
            amp_mean,
            amp_std,
            *[float(value) for value in np.quantile(amp, [0.10, 0.25, 0.50, 0.75, 0.90])],
            float(np.max(amp)),
            amp_skew,
            amp_kurt,
        ]
        for order, include_abs in ((2, True), (3, False), (4, True)):
            moment = complex(np.mean(centered**order))
            values.extend([float(moment.real), float(moment.imag)])
            if include_abs:
                values.append(float(abs(moment)))
        for lag in (1, 2, 4, 8):
            correlation = (
                0.0j
                if centered.size <= lag
                else complex(np.mean(centered[lag:] * np.conj(centered[:-lag])))
            )
            values.extend([float(correlation.real), float(correlation.imag)])
        amp_corr1 = (
            float(
                np.mean(amp_centered[1:] * amp_centered[:-1])
                / max(amp_std**2, EPS)
            )
            if amp.size > 1 and amp_std > EPS
            else 0.0
        )
        values.append(amp_corr1)
        descriptor = np.asarray(values, dtype=np.float32)
        if descriptor.shape != (RF_STAT_DIM,) or not np.isfinite(descriptor).all():
            raise DiagCosineExplorationError("RF descriptor dimension/value drift")
        descriptor /= max(float(np.linalg.norm(descriptor)), EPS)
        result.append(descriptor)
    return np.stack(result).astype(np.float32)


def registered_feature(rows: np.ndarray, zid160: np.ndarray) -> np.ndarray:
    """Concatenate normalized z_id160 and same-row FFT-RF with locked weight 4."""

    primary = _normalize(np.asarray(zid160, dtype=np.float32))
    auxiliary = np.concatenate(
        [spectral_logmag_sketch(rows), rf_statistics(rows)], axis=1
    )
    if primary.ndim != 2 or primary.shape[1] != FEATURE_DIM:
        raise DiagCosineExplorationError("sealed runtime feature schema is not z_id160")
    if len(primary) != len(auxiliary) or auxiliary.shape[1] != FFT_DIM + RF_STAT_DIM:
        raise DiagCosineExplorationError("FFT-RF auxiliary feature alignment drift")
    return _normalize(
        np.concatenate([primary, AUXILIARY_WEIGHT * _normalize(auxiliary)], axis=1)
    )


def _tensor_from_numpy(value: np.ndarray, *, device: torch.device) -> torch.Tensor:
    rows = np.ascontiguousarray(value, dtype=np.float32)
    return (
        torch.frombuffer(rows, dtype=torch.float32)
        .reshape(rows.shape)
        .clone()
        .to(device)
    )


def forward_zid160(
    model: torch.nn.Module,
    rows: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Run the sealed runtime; query callers use batch_size=1."""

    source = np.asarray(rows, dtype=np.float32)
    if (
        source.ndim != 3
        or source.shape[1] != 2
        or not np.isfinite(source).all()
        or int(batch_size) < 1
    ):
        raise DiagCosineExplorationError("sealed runtime input drift")
    features: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(source), int(batch_size)):
            output = model(
                _tensor_from_numpy(
                    source[start : start + int(batch_size)], device=device
                )
            )
            if isinstance(output, dict):
                feature_value = output.get("features")
            elif isinstance(output, (tuple, list)) and len(output) == 2:
                feature_value = output[0]
            else:
                raise DiagCosineExplorationError(
                    "sealed runtime must return features and logits"
                )
            if (
                not torch.is_tensor(feature_value)
                or feature_value.dtype != torch.float32
                or feature_value.ndim != 2
                or int(feature_value.shape[1]) != FEATURE_DIM
            ):
                raise DiagCosineExplorationError(
                    "sealed runtime output is not finite z_id160"
                )
            values = np.asarray(feature_value.detach().cpu().tolist(), dtype=np.float32)
            if not np.isfinite(values).all():
                raise DiagCosineExplorationError(
                    "sealed runtime output contains non-finite values"
                )
            features.append(values)
    return np.concatenate(features, axis=0)


def fit_diag_cosine_state(
    support_x: np.ndarray,
    support_y: np.ndarray,
    *,
    seed: int,
    device: torch.device,
    candidate: str = CANDIDATE_D1,
) -> DiagCosineState:
    """Fit only on registered support; no query argument exists by design."""

    support = np.asarray(support_x, dtype=np.float32)
    labels = np.asarray(support_y).astype(str)
    if (
        support.ndim != 2
        or len(support) != len(labels)
        or len(support) == 0
        or not np.isfinite(support).all()
    ):
        raise DiagCosineExplorationError("support feature/label alignment drift")
    classes = np.asarray(sorted(set(labels.tolist())))
    if len(classes) < 2:
        raise DiagCosineExplorationError("at least two registered classes are required")
    if candidate not in CANDIDATES:
        raise DiagCosineExplorationError(f"unsupported diag-cosine candidate: {candidate}")
    if candidate == CANDIDATE_D3_SCENARIO_OLDLOCK_NEWFIT:
        raise DiagCosineExplorationError(
            "D3 requires scenario-specific before/after orchestration"
        )
    class_to_index = {label: index for index, label in enumerate(classes.tolist())}
    targets_np = np.asarray([class_to_index[label] for label in labels], dtype=np.int64)
    feature_dim = int(support.shape[1])
    class_count = int(len(classes))
    trainable_head = candidate in {CANDIDATE_D1, CANDIDATE_D1_B0_CAP}
    if candidate == CANDIDATE_D1:
        trainable_parameters = int(
            feature_dim + class_count * feature_dim + class_count
        )
        stored_float_count = trainable_parameters
    elif candidate == CANDIDATE_D1_B0_CAP:
        trainable_parameters = int(feature_dim + class_count * feature_dim)
        stored_float_count = trainable_parameters
    else:
        trainable_parameters = feature_dim
        # Preserve the existing D2 fixed-prototype and zero-bias state layout.
        stored_float_count = int(
            feature_dim + class_count * feature_dim + class_count
        )
    parameter_state_bytes = int(stored_float_count * np.dtype(np.float32).itemsize)
    registry_state_bytes = len(_canonical_json_bytes(classes.tolist()))
    persistent_state_bytes = parameter_state_bytes + registry_state_bytes
    if trainable_parameters > MAX_TRAINABLE_PARAMETERS:
        raise DiagCosineExplorationError("diag-cosine trainable parameter cap exceeded")
    if persistent_state_bytes > MAX_PERSISTENT_STATE_BYTES:
        raise DiagCosineExplorationError("diag-cosine persistent state cap exceeded")

    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(int(seed))
        torch.empty(0, device=device)
        torch.cuda.reset_peak_memory_stats(device)
    x = _tensor_from_numpy(support, device=device)
    targets = torch.as_tensor(targets_np, dtype=torch.long, device=device)
    prototypes = torch.stack(
        [
            F.normalize(x[targets == index].mean(dim=0), dim=0)
            for index in range(class_count)
        ]
    )
    log_scale = torch.nn.Parameter(torch.zeros(feature_dim, device=device))
    lower_np, upper_np = log_scale_bounds(candidate, feature_dim)
    lower_bound = _tensor_from_numpy(lower_np, device=device)
    upper_bound = _tensor_from_numpy(upper_np, device=device)
    if trainable_head:
        weights: torch.Tensor = torch.nn.Parameter(prototypes.detach().clone())
        if candidate == CANDIDATE_D1:
            bias: torch.Tensor = torch.nn.Parameter(
                torch.zeros(class_count, device=device)
            )
            parameters = [log_scale, weights, bias]
        else:
            bias = torch.empty(0, dtype=x.dtype, device=device)
            parameters = [log_scale, weights]
    else:
        weights = prototypes.detach().clone()
        bias = torch.zeros(class_count, device=device)
        parameters = [log_scale]
    optimizer = torch.optim.AdamW(
        parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    generator = torch.Generator(device=device).manual_seed(int(seed))
    trace: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, ADAPTATION_EPOCHS + 1):
        permutation = torch.randperm(len(x), generator=generator, device=device)
        total_sum = 0.0
        ce_sum = 0.0
        anchor_sum = 0.0
        correct = 0
        seen = 0
        grad_sum = 0.0
        batches = 0
        for positions in permutation.split(min(BATCH_SIZE, len(x))):
            optimizer.zero_grad(set_to_none=True)
            rows = x[positions]
            rows = rows + FEATURE_NOISE_STD * torch.randn(
                rows.shape, generator=generator, device=device, dtype=rows.dtype
            )
            effective_log_scale = torch.minimum(
                torch.maximum(log_scale, lower_bound), upper_bound
            )
            scaled = rows * torch.exp(effective_log_scale)
            logits = TEMPERATURE * (
                F.normalize(scaled, dim=1) @ F.normalize(weights, dim=1).T
            )
            if bias.numel():
                logits = logits + bias
            ce_loss = F.cross_entropy(logits, targets[positions])
            anchor_loss = (
                torch.mean((F.normalize(weights, dim=1) - prototypes) ** 2)
                if trainable_head
                else torch.zeros((), dtype=x.dtype, device=device)
            )
            loss = (
                ce_loss + PROTOTYPE_ANCHOR_WEIGHT * anchor_loss
                if trainable_head
                else ce_loss
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
            optimizer.step()
            if candidate == CANDIDATE_D1_B0_CAP:
                with torch.no_grad():
                    log_scale.copy_(
                        torch.minimum(
                            torch.maximum(log_scale, lower_bound), upper_bound
                        )
                    )
            count = int(len(positions))
            total_sum += float(loss.detach()) * count
            ce_sum += float(ce_loss.detach()) * count
            anchor_sum += float(anchor_loss.detach()) * count
            correct += int((logits.argmax(dim=1) == targets[positions]).sum().item())
            seen += count
            grad_sum += float(grad_norm.detach())
            batches += 1
        row = {
            "phase": "target_support_only_diag_cosine_fit",
            "epoch": epoch,
            "step": epoch,
            "total_steps": ADAPTATION_EPOCHS,
            "loss": total_sum / max(1, seen),
            "total_loss": total_sum / max(1, seen),
            "ce_loss": ce_sum / max(1, seen),
            "prototype_anchor_loss": anchor_sum / max(1, seen),
            "learning_rate": LEARNING_RATE,
            "gradient_norm": grad_sum / max(1, batches),
            "support_accuracy": correct / max(1, seen),
        }
        if not all(
            math.isfinite(float(value))
            for key, value in row.items()
            if key != "phase"
        ):
            raise DiagCosineExplorationError("non-finite diag-cosine loss trace")
        trace.append(row)
    elapsed = time.perf_counter() - started
    scale_np = np.asarray(log_scale.detach().cpu().tolist(), dtype=np.float32)
    weights_np = np.asarray(weights.detach().cpu().tolist(), dtype=np.float32)
    bias_np = np.asarray(bias.detach().cpu().tolist(), dtype=np.float32)
    macs_per_sample = int(feature_dim + class_count * feature_dim)
    resource = {
        "schema": "cvs.phase2.diag_cosine_resource.v1",
        "adaptation_objective": "target_support_only_diag_metric_cosine_ce",
        "candidate": candidate,
        "classifier_state_policy": (
            "free_diag_weights_and_bias"
            if candidate == CANDIDATE_D1
            else (
                "free_diag_weights_zero_class_bias_fft96_capped"
                if candidate == CANDIDATE_D1_B0_CAP
                else "current_registry_fixed_support_prototypes_zero_class_bias_shared_diag_only"
            )
        ),
        "class_bias_enabled": candidate == CANDIDATE_D1,
        "class_bias_trainable_parameters": (
            class_count if candidate == CANDIDATE_D1 else 0
        ),
        "log_scale_bounds": {
            "z_id160": [-LOG_SCALE_LIMIT, LOG_SCALE_LIMIT],
            "fft96": [
                (
                    -FFT_LOG_SCALE_LIMIT
                    if candidate == CANDIDATE_D1_B0_CAP
                    else -LOG_SCALE_LIMIT
                ),
                (
                    FFT_LOG_SCALE_LIMIT
                    if candidate == CANDIDATE_D1_B0_CAP
                    else LOG_SCALE_LIMIT
                ),
            ],
            "rf32": [-LOG_SCALE_LIMIT, LOG_SCALE_LIMIT],
        },
        "support_only": True,
        "query_rows_used_for_fit": 0,
        "query_labels_used_for_fit": False,
        "query_features_used_for_fit": False,
        "query_role_oracle_access": False,
        "query_true_batch_class_count_access": False,
        "query_class_quota_access": False,
        "query_batch_global_assignment": False,
        "query_query_graph_used": False,
        "source_sample_access": False,
        "source_cache_access": False,
        "source_derived_signal_access": False,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "trainable_parameters": trainable_parameters,
        "persistent_state_bytes": persistent_state_bytes,
        "parameter_state_bytes": parameter_state_bytes,
        "registry_state_bytes": registry_state_bytes,
        "feature_dim": feature_dim,
        "class_count": class_count,
        "adaptation_epochs": ADAPTATION_EPOCHS,
        "optimizer_steps": ADAPTATION_EPOCHS
        * math.ceil(len(support) / min(BATCH_SIZE, len(support))),
        "support_enrollment_rows": int(len(support)),
        "support_view_count": len(FORMAL_LEO_WEAK_SCENARIOS),
        "query_view_count": 1,
        "estimated_adaptation_macs": int(
            ADAPTATION_EPOCHS * len(support) * macs_per_sample * 3
        ),
        "estimated_macs_per_query": macs_per_sample,
        "dense_query_graph_bytes": 0,
        "adaptation_latency_sec": float(elapsed),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "runtime_device": str(device),
        "loss_initial": float(trace[0]["loss"]),
        "loss_final": float(trace[-1]["loss"]),
        "support_accuracy_final": float(trace[-1]["support_accuracy"]),
    }
    return DiagCosineState(
        candidate=candidate,
        classes=classes,
        log_scale=scale_np,
        weights=weights_np,
        bias=bias_np,
        trace=tuple(trace),
        resource=resource,
    )


def predict_diag_cosine(state: DiagCosineState, query_x: np.ndarray) -> np.ndarray:
    """Score each query independently against every registered class."""

    query = np.asarray(query_x, dtype=np.float32)
    if (
        query.ndim != 2
        or query.shape[1] != state.weights.shape[1]
        or not np.isfinite(query).all()
    ):
        raise DiagCosineExplorationError("query feature schema drift")
    lower, upper = log_scale_bounds(state.candidate, query.shape[1])
    effective_log_scale = np.minimum(
        np.maximum(state.log_scale, lower), upper
    )
    scaled = query * np.exp(effective_log_scale)[None, :]
    logits = TEMPERATURE * (_normalize(scaled) @ _normalize(state.weights).T)
    if state.bias.size:
        logits += state.bias[None, :]
    return state.classes[np.argmax(logits, axis=1)]


def _scenario_predict(
    state: ScenarioDiagCosineState,
    scenario: str,
    query_x: np.ndarray,
) -> np.ndarray:
    query = np.asarray(query_x, dtype=np.float32)
    matches = np.flatnonzero(state.scenarios.astype(str) == str(scenario))
    if len(matches) != 1:
        raise DiagCosineExplorationError("scenario-specific state lookup drift")
    index = int(matches[0])
    if (
        query.ndim != 2
        or query.shape[1] != state.weights.shape[2]
        or not np.isfinite(query).all()
    ):
        raise DiagCosineExplorationError("scenario query feature schema drift")
    lower, upper = log_scale_bounds(CANDIDATE_D1, query.shape[1])
    scale = np.minimum(np.maximum(state.log_scale[index], lower), upper)
    logits = TEMPERATURE * (
        _normalize(query * np.exp(scale)[None, :])
        @ _normalize(state.weights[index]).T
    )
    logits += state.bias[index][None, :]
    if state.old_class_count < len(state.classes):
        logits[:, state.old_class_count :] -= float(state.new_offset[index])
    return state.classes[np.argmax(logits, axis=1)]


def _read_readonly_regular_bytes(path: Path) -> bytes:
    source = path.resolve(strict=True)
    if path.is_symlink() or source.is_symlink() or not source.is_file():
        raise DiagCosineExplorationError("parent closure member must be a regular file")
    descriptor = os.open(
        source,
        os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise DiagCosineExplorationError(
                "parent closure member must remain a regular file"
            )
        if metadata.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise DiagCosineExplorationError(
                "parent closure member must be read-only"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _parent_member(
    commit: Mapping[str, Any],
    relative_path: str,
) -> dict[str, Any]:
    matches = [
        item
        for item in commit.get("members", [])
        if isinstance(item, dict) and item.get("relative_path") == relative_path
    ]
    if len(matches) != 1:
        raise DiagCosineExplorationError(
            f"parent COMMIT member missing or duplicated: {relative_path}"
        )
    member = dict(matches[0])
    if (
        not isinstance(member.get("sha256"), str)
        or len(member["sha256"]) != 64
        or not isinstance(member.get("size_bytes"), int)
        or member["size_bytes"] < 1
    ):
        raise DiagCosineExplorationError("parent COMMIT member metadata drift")
    return member


def _load_parent_scenario_state(
    *,
    parent_diag_root: str | Path,
    expected_parent_commit_sha256: str,
    enrollment_manifest: Mapping[str, Any],
    apply_manifest: Mapping[str, Any],
) -> tuple[ScenarioDiagCosineState, dict[str, str]]:
    raw_root = Path(parent_diag_root)
    root = raw_root.resolve(strict=True)
    if raw_root.is_symlink() or not root.is_dir():
        raise DiagCosineExplorationError("parent diag root must be a regular directory")
    commit_raw = _read_readonly_regular_bytes(root / "COMMIT.json")
    commit_sha256 = hashlib.sha256(commit_raw).hexdigest()
    if commit_sha256 != str(expected_parent_commit_sha256).lower():
        raise DiagCosineExplorationError("parent COMMIT SHA256 mismatch")
    try:
        commit = json.loads(commit_raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DiagCosineExplorationError("parent COMMIT JSON drift") from exc
    if (
        not isinstance(commit, dict)
        or commit.get("schema")
        != "cvs.phase2.diag_cosine_exploration_commit.v1"
    ):
        raise DiagCosineExplorationError("parent COMMIT schema drift")
    receipt_member = _parent_member(commit, "execution_receipt.json")
    state_member = _parent_member(commit, "diag_cosine_state.npz")
    receipt_raw = _read_readonly_regular_bytes(root / "execution_receipt.json")
    if (
        len(receipt_raw) != receipt_member["size_bytes"]
        or hashlib.sha256(receipt_raw).hexdigest() != receipt_member["sha256"]
        or commit.get("execution_receipt_sha256") != receipt_member["sha256"]
    ):
        raise DiagCosineExplorationError("parent execution receipt binding drift")
    try:
        receipt = json.loads(receipt_raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DiagCosineExplorationError("parent execution receipt JSON drift") from exc
    required_matches = {
        "stage": "stage2b",
        "registration_state": "before",
        "receiver": enrollment_manifest["receiver"],
        "seed": enrollment_manifest["seed"],
        "k_shot": enrollment_manifest["k_shot"],
        "row_handle": apply_manifest["row_handle"],
        "row_manifest_sha256": apply_manifest["row_manifest_sha256"],
        "phase1_checkpoint_sha256": enrollment_manifest[
            "phase1_checkpoint_sha256"
        ],
        "feature_runtime_sha256": enrollment_manifest["feature_runtime_sha256"],
        "method_lock_sha256": enrollment_manifest["method_lock_sha256"],
    }
    resource = receipt.get("resource", {}) if isinstance(receipt, dict) else {}
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema")
        != "cvs.phase2.diag_cosine_exploration_receipt.v1"
        or receipt.get("candidate", {}).get("name")
        != CANDIDATE_D3_SCENARIO_OLDLOCK_NEWFIT
        or any(receipt.get(key) != value for key, value in required_matches.items())
        or any(
            receipt.get(key) != value
            for key, value in PHASE2_FULL_CONTRACT.items()
        )
        or receipt.get("query_truth_present_in_predictor") is not False
        or not isinstance(resource, dict)
        or resource.get("query_rows_used_for_fit") != 0
        or resource.get("query_labels_used_for_fit") is not False
        or resource.get("query_features_used_for_fit") is not False
        or resource.get("query_role_oracle_access") is not False
        or resource.get("query_true_batch_class_count_access") is not False
        or resource.get("query_class_quota_access") is not False
        or resource.get("query_batch_global_assignment") is not False
        or receipt.get("artifacts", {}).get("diag_cosine_state.npz")
        != state_member["sha256"]
    ):
        raise DiagCosineExplorationError("parent execution receipt lineage drift")
    state_raw = _read_readonly_regular_bytes(root / "diag_cosine_state.npz")
    if (
        len(state_raw) != state_member["size_bytes"]
        or hashlib.sha256(state_raw).hexdigest() != state_member["sha256"]
    ):
        raise DiagCosineExplorationError("parent scenario state binding drift")
    try:
        with np.load(io.BytesIO(state_raw), allow_pickle=False) as archive:
            expected_members = (
                "scenarios",
                "classes",
                "log_scale",
                "weights",
                "bias",
                "new_offset",
                "old_class_count",
            )
            if tuple(archive.files) != expected_members:
                raise DiagCosineExplorationError(
                    "parent scenario state exact schema drift"
                )
            scenarios = archive["scenarios"].astype(str)
            classes = archive["classes"].astype(str)
            log_scale = archive["log_scale"].astype(np.float32)
            weights = archive["weights"].astype(np.float32)
            bias = archive["bias"].astype(np.float32)
            new_offset = archive["new_offset"].astype(np.float32)
            old_class_count = int(archive["old_class_count"].reshape(-1)[0])
    except (OSError, ValueError, KeyError) as exc:
        raise DiagCosineExplorationError("parent scenario state NPZ drift") from exc
    scenario_count = len(FORMAL_LEO_WEAK_SCENARIOS)
    if (
        scenarios.tolist() != list(FORMAL_LEO_WEAK_SCENARIOS)
        or classes.ndim != 1
        or len(classes) < 2
        or log_scale.shape != (scenario_count, weights.shape[2])
        or weights.shape != (scenario_count, len(classes), log_scale.shape[1])
        or bias.shape != (scenario_count, len(classes))
        or new_offset.shape != (scenario_count,)
        or old_class_count != len(classes)
        or not all(
            np.isfinite(value).all()
            for value in (log_scale, weights, bias, new_offset)
        )
    ):
        raise DiagCosineExplorationError("parent scenario state shape/value drift")
    state = ScenarioDiagCosineState(
        candidate=CANDIDATE_D3_SCENARIO_OLDLOCK_NEWFIT,
        scenarios=scenarios,
        classes=classes,
        log_scale=log_scale,
        weights=weights,
        bias=bias,
        new_offset=new_offset,
        old_class_count=old_class_count,
        trace=(),
        resource={},
    )
    return state, {
        "parent_diag_commit_sha256": commit_sha256,
        "parent_execution_receipt_sha256": receipt_member["sha256"],
        "parent_state_sha256": state_member["sha256"],
        "parent_old_classes_sha256": hashlib.sha256(
            _canonical_json_bytes(classes.tolist())
        ).hexdigest(),
    }


def _d3_common_resource(
    *,
    class_count: int,
    old_class_count: int,
    feature_dim: int,
    support_rows: int,
    trainable_parameters: int,
    parameter_state_bytes: int,
    registry_state_bytes: int,
    estimated_adaptation_macs: int,
    optimizer_steps: int,
    adaptation_latency_sec: float,
    peak_cuda_memory_bytes: int,
    trace: list[dict[str, Any]],
    phase: str,
    audits: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    persistent_state_bytes = parameter_state_bytes + registry_state_bytes
    if trainable_parameters > MAX_TRAINABLE_PARAMETERS:
        raise DiagCosineExplorationError("D3 trainable parameter cap exceeded")
    if persistent_state_bytes > MAX_PERSISTENT_STATE_BYTES:
        raise DiagCosineExplorationError("D3 persistent state cap exceeded")
    return {
        "schema": "cvs.phase2.diag_cosine_resource.v1",
        "adaptation_objective": phase,
        "candidate": CANDIDATE_D3_SCENARIO_OLDLOCK_NEWFIT,
        "classifier_state_policy": (
            "scenario_specific_old_head_fit"
            if phase == "target_support_only_scenario_old_head_fit"
            else "scenario_specific_old_head_bitwise_locked_new_weights_only"
        ),
        "class_bias_enabled": True,
        "class_bias_trainable_parameters": (
            len(FORMAL_LEO_WEAK_SCENARIOS) * old_class_count
            if phase == "target_support_only_scenario_old_head_fit"
            else 0
        ),
        "new_class_bias_policy": "zero_per_class_bias_shared_nonnegative_offset",
        "log_scale_bounds": {
            "z_id160": [-LOG_SCALE_LIMIT, LOG_SCALE_LIMIT],
            "fft96": [-LOG_SCALE_LIMIT, LOG_SCALE_LIMIT],
            "rf32": [-LOG_SCALE_LIMIT, LOG_SCALE_LIMIT],
        },
        "support_only": True,
        "query_rows_used_for_fit": 0,
        "query_labels_used_for_fit": False,
        "query_features_used_for_fit": False,
        "query_role_oracle_access": False,
        "query_true_batch_class_count_access": False,
        "query_class_quota_access": False,
        "query_batch_global_assignment": False,
        "query_query_graph_used": False,
        "source_sample_access": False,
        "source_cache_access": False,
        "source_derived_signal_access": False,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "trainable_parameters": int(trainable_parameters),
        "persistent_state_bytes": int(persistent_state_bytes),
        "parameter_state_bytes": int(parameter_state_bytes),
        "registry_state_bytes": int(registry_state_bytes),
        "feature_dim": int(feature_dim),
        "class_count": int(class_count),
        "old_class_count": int(old_class_count),
        "new_class_count": int(class_count - old_class_count),
        "adaptation_epochs": ADAPTATION_EPOCHS,
        "epochs_per_scenario": ADAPTATION_EPOCHS,
        "total_epoch_passes": (
            ADAPTATION_EPOCHS * len(FORMAL_LEO_WEAK_SCENARIOS)
        ),
        "optimizer_steps": int(optimizer_steps),
        "support_enrollment_rows": int(support_rows),
        "support_view_count": len(FORMAL_LEO_WEAK_SCENARIOS),
        "query_view_count": 1,
        "scenario_specific_state_count": len(FORMAL_LEO_WEAK_SCENARIOS),
        "estimated_adaptation_macs": int(estimated_adaptation_macs),
        "estimated_adaptation_macs_scope": (
            "registered_head_plus_feature_scaling_excludes_backbone_fft_rf"
        ),
        "estimated_macs_per_query": int(
            feature_dim + class_count * feature_dim
        ),
        "estimated_head_macs_per_query": int(
            feature_dim + class_count * feature_dim
        ),
        "estimated_macs_per_query_scope": (
            "registered_head_plus_feature_scaling_excludes_backbone_fft_rf"
        ),
        "dense_query_graph_bytes": 0,
        "adaptation_latency_sec": float(adaptation_latency_sec),
        "peak_cuda_memory_bytes": int(peak_cuda_memory_bytes),
        "runtime_device": (
            str(trace[0].get("runtime_device", "unknown")) if trace else "unknown"
        ),
        "floor_promotion_gate_requires_post_prediction_scorer": True,
        "scenario_old_support_intrusion_audit": dict(audits or {}),
    }


def _fit_d3_before(
    support_by_scenario: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    seed: int,
    device: torch.device,
) -> ScenarioDiagCosineState:
    states: list[DiagCosineState] = []
    trace: list[dict[str, Any]] = []
    started = time.perf_counter()
    for scenario_index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS):
        support_x, support_y = support_by_scenario[scenario]
        fitted = fit_diag_cosine_state(
            support_x,
            support_y,
            seed=int(seed) + scenario_index,
            device=device,
            candidate=CANDIDATE_D1,
        )
        states.append(fitted)
        trace.extend(
            {
                **row,
                "scenario": scenario,
                "runtime_device": str(device),
            }
            for row in fitted.trace
        )
    classes = states[0].classes
    if any(not np.array_equal(state.classes, classes) for state in states[1:]):
        raise DiagCosineExplorationError("D3 before scenario class registry drift")
    log_scale = np.stack([state.log_scale for state in states]).astype(np.float32)
    weights = np.stack([state.weights for state in states]).astype(np.float32)
    bias = np.stack([state.bias for state in states]).astype(np.float32)
    new_offset = np.zeros(len(states), dtype=np.float32)
    feature_dim = int(weights.shape[2])
    parameter_state_bytes = int(
        (log_scale.size + weights.size + bias.size + new_offset.size)
        * np.dtype(np.float32).itemsize
        + np.dtype(np.int64).itemsize
    )
    registry_state_bytes = len(_canonical_json_bytes(classes.tolist()))
    resource = _d3_common_resource(
        class_count=len(classes),
        old_class_count=len(classes),
        feature_dim=feature_dim,
        support_rows=sum(len(value[0]) for value in support_by_scenario.values()),
        trainable_parameters=sum(
            int(state.resource["trainable_parameters"]) for state in states
        ),
        parameter_state_bytes=parameter_state_bytes,
        registry_state_bytes=registry_state_bytes,
        estimated_adaptation_macs=sum(
            int(state.resource["estimated_adaptation_macs"]) for state in states
        ),
        optimizer_steps=sum(
            int(state.resource["optimizer_steps"]) for state in states
        ),
        adaptation_latency_sec=time.perf_counter() - started,
        peak_cuda_memory_bytes=max(
            int(state.resource["peak_cuda_memory_bytes"]) for state in states
        ),
        trace=trace,
        phase="target_support_only_scenario_old_head_fit",
    )
    return ScenarioDiagCosineState(
        candidate=CANDIDATE_D3_SCENARIO_OLDLOCK_NEWFIT,
        scenarios=np.asarray(FORMAL_LEO_WEAK_SCENARIOS),
        classes=classes.copy(),
        log_scale=log_scale,
        weights=weights,
        bias=bias,
        new_offset=new_offset,
        old_class_count=len(classes),
        trace=tuple(trace),
        resource=resource,
    )


def _fit_d3_after(
    parent: ScenarioDiagCosineState,
    support_by_scenario: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    seed: int,
    device: torch.device,
) -> ScenarioDiagCosineState:
    current_sets = [
        set(np.asarray(support_by_scenario[scenario][1]).astype(str).tolist())
        for scenario in FORMAL_LEO_WEAK_SCENARIOS
    ]
    if any(values != current_sets[0] for values in current_sets[1:]):
        raise DiagCosineExplorationError("D3 after scenario class registry drift")
    old_classes = parent.classes.astype(str)
    old_set = set(old_classes.tolist())
    if not old_set < current_sets[0]:
        raise DiagCosineExplorationError(
            "D3 parent classes must be a strict subset of after registry"
        )
    new_classes = np.asarray(sorted(current_sets[0] - old_set))
    classes = np.concatenate([old_classes, new_classes])
    old_class_count = len(old_classes)
    feature_dim = int(parent.weights.shape[2])
    class_to_index = {
        label: index for index, label in enumerate(classes.tolist())
    }
    all_log_scale: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []
    all_bias: list[np.ndarray] = []
    all_offsets: list[float] = []
    trace: list[dict[str, Any]] = []
    audits: dict[str, Any] = {}
    estimated_adaptation_macs = 0
    optimizer_steps = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    for scenario_index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS):
        support_x, support_y = support_by_scenario[scenario]
        support = np.asarray(support_x, dtype=np.float32)
        labels = np.asarray(support_y).astype(str)
        targets_np = np.asarray(
            [class_to_index[label] for label in labels], dtype=np.int64
        )
        if support.ndim != 2 or support.shape[1] != feature_dim:
            raise DiagCosineExplorationError("D3 after support feature drift")
        x = _tensor_from_numpy(support, device=device)
        targets = torch.as_tensor(targets_np, dtype=torch.long, device=device)
        scale = _tensor_from_numpy(
            parent.log_scale[scenario_index], device=device
        )
        old_weights = _tensor_from_numpy(
            parent.weights[scenario_index], device=device
        )
        old_bias = _tensor_from_numpy(parent.bias[scenario_index], device=device)
        transformed = F.normalize(x * torch.exp(scale)[None, :], dim=1)
        prototypes = torch.stack(
            [
                F.normalize(
                    transformed[targets == class_index].mean(dim=0), dim=0
                )
                for class_index in range(old_class_count, len(classes))
            ]
        )
        new_weights = torch.nn.Parameter(prototypes.detach().clone())
        optimizer = torch.optim.AdamW(
            [new_weights], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        generator = torch.Generator(device=device).manual_seed(
            int(seed) + 1009 + scenario_index
        )
        for epoch in range(1, ADAPTATION_EPOCHS + 1):
            permutation = torch.randperm(
                len(x), generator=generator, device=device
            )
            total_sum = 0.0
            ce_sum = 0.0
            anchor_sum = 0.0
            correct = 0
            seen = 0
            grad_sum = 0.0
            batches = 0
            for positions in permutation.split(min(BATCH_SIZE, len(x))):
                optimizer.zero_grad(set_to_none=True)
                rows = x[positions]
                rows = rows + FEATURE_NOISE_STD * torch.randn(
                    rows.shape,
                    generator=generator,
                    device=device,
                    dtype=rows.dtype,
                )
                z = F.normalize(rows * torch.exp(scale)[None, :], dim=1)
                old_logits = TEMPERATURE * (
                    z @ F.normalize(old_weights, dim=1).T
                ) + old_bias[None, :]
                new_logits = TEMPERATURE * (
                    z @ F.normalize(new_weights, dim=1).T
                )
                logits = torch.cat([old_logits, new_logits], dim=1)
                ce_loss = F.cross_entropy(logits, targets[positions])
                anchor_loss = torch.mean(
                    (F.normalize(new_weights, dim=1) - prototypes) ** 2
                )
                loss = ce_loss + PROTOTYPE_ANCHOR_WEIGHT * anchor_loss
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [new_weights], max_norm=5.0
                )
                optimizer.step()
                count = int(len(positions))
                total_sum += float(loss.detach()) * count
                ce_sum += float(ce_loss.detach()) * count
                anchor_sum += float(anchor_loss.detach()) * count
                correct += int(
                    (logits.argmax(dim=1) == targets[positions]).sum().item()
                )
                seen += count
                grad_sum += float(grad_norm.detach())
                batches += 1
            row = {
                "phase": "target_support_only_scenario_new_weight_fit",
                "scenario": scenario,
                "epoch": epoch,
                "step": epoch,
                "total_steps": ADAPTATION_EPOCHS,
                "loss": total_sum / max(1, seen),
                "total_loss": total_sum / max(1, seen),
                "ce_loss": ce_sum / max(1, seen),
                "prototype_anchor_loss": anchor_sum / max(1, seen),
                "learning_rate": LEARNING_RATE,
                "gradient_norm": grad_sum / max(1, batches),
                "support_accuracy": correct / max(1, seen),
                "runtime_device": str(device),
            }
            if not all(
                math.isfinite(float(value))
                for key, value in row.items()
                if key not in {"phase", "scenario", "runtime_device"}
            ):
                raise DiagCosineExplorationError("non-finite D3 loss trace")
            trace.append(row)
        with torch.no_grad():
            old_mask = targets < old_class_count
            old_z = transformed[old_mask]
            old_targets = targets[old_mask]
            old_logits = TEMPERATURE * (
                old_z @ F.normalize(old_weights, dim=1).T
            ) + old_bias[None, :]
            new_logits = TEMPERATURE * (
                old_z @ F.normalize(new_weights, dim=1).T
            )
            pre_margin = old_logits.max(dim=1).values - new_logits.max(dim=1).values
            offset = max(
                0.0,
                float((-pre_margin.min() + 1.0e-6).detach().cpu()),
            )
            post_margin = pre_margin + float(offset)
            before_pred = old_logits.argmax(dim=1)
            combined = torch.cat(
                [old_logits, new_logits - float(offset)], dim=1
            )
            after_pred = combined.argmax(dim=1)
            per_class: dict[str, Any] = {}
            intrusion_count = 0
            for class_index, handle in enumerate(old_classes.tolist()):
                mask = old_targets == class_index
                class_pre = pre_margin[mask]
                class_post = post_margin[mask]
                class_before = before_pred[mask]
                class_after = after_pred[mask]
                class_intrusions = int(
                    (class_after >= old_class_count).sum().item()
                )
                intrusion_count += class_intrusions
                per_class[handle] = {
                    "support_count": int(mask.sum().item()),
                    "pre_support_margin_min": float(class_pre.min().cpu()),
                    "pre_support_margin_mean": float(class_pre.mean().cpu()),
                    "post_support_margin_min": float(class_post.min().cpu()),
                    "post_support_margin_mean": float(class_post.mean().cpu()),
                    "pre_support_accuracy": float(
                        (class_before == old_targets[mask]).float().mean().cpu()
                    ),
                    "post_support_accuracy": float(
                        (class_after == old_targets[mask]).float().mean().cpu()
                    ),
                    "old_class_intrusion_count": class_intrusions,
                }
            worst_old_class_margin = min(
                row["post_support_margin_min"] for row in per_class.values()
            )
            if intrusion_count != 0 or worst_old_class_margin < -1.0e-6:
                raise DiagCosineExplorationError(
                    "D3 old-class support intrusion protection failed"
                )
        audits[scenario] = {
            "new_offset": float(offset),
            "worst_old_class_margin": float(worst_old_class_margin),
            "old_class_intrusion_count": int(intrusion_count),
            "per_old_class": per_class,
        }
        all_log_scale.append(parent.log_scale[scenario_index].copy())
        all_weights.append(
            np.concatenate(
                [
                    parent.weights[scenario_index],
                    np.asarray(
                        new_weights.detach().cpu().tolist(), dtype=np.float32
                    ),
                ],
                axis=0,
            )
        )
        all_bias.append(
            np.concatenate(
                [
                    parent.bias[scenario_index],
                    np.zeros(len(new_classes), dtype=np.float32),
                ]
            )
        )
        all_offsets.append(offset)
        macs_per_sample = feature_dim + len(classes) * feature_dim
        estimated_adaptation_macs += int(
            ADAPTATION_EPOCHS * len(support) * macs_per_sample * 3
        )
        optimizer_steps += int(
            ADAPTATION_EPOCHS
            * math.ceil(len(support) / min(BATCH_SIZE, len(support)))
        )
    log_scale_np = np.stack(all_log_scale).astype(np.float32)
    weights_np = np.stack(all_weights).astype(np.float32)
    bias_np = np.stack(all_bias).astype(np.float32)
    offset_np = np.asarray(all_offsets, dtype=np.float32)
    if (
        not np.array_equal(log_scale_np, parent.log_scale)
        or not np.array_equal(
            weights_np[:, :old_class_count], parent.weights
        )
        or not np.array_equal(bias_np[:, :old_class_count], parent.bias)
    ):
        raise DiagCosineExplorationError("D3 parent old head mutation detected")
    parameter_state_bytes = int(
        (
            log_scale_np.size
            + weights_np.size
            + bias_np.size
            + offset_np.size
        )
        * np.dtype(np.float32).itemsize
        + np.dtype(np.int64).itemsize
    )
    registry_state_bytes = len(_canonical_json_bytes(classes.tolist()))
    resource = _d3_common_resource(
        class_count=len(classes),
        old_class_count=old_class_count,
        feature_dim=feature_dim,
        support_rows=sum(len(value[0]) for value in support_by_scenario.values()),
        trainable_parameters=(
            len(FORMAL_LEO_WEAK_SCENARIOS) * len(new_classes) * feature_dim
        ),
        parameter_state_bytes=parameter_state_bytes,
        registry_state_bytes=registry_state_bytes,
        estimated_adaptation_macs=estimated_adaptation_macs,
        optimizer_steps=optimizer_steps,
        adaptation_latency_sec=time.perf_counter() - started,
        peak_cuda_memory_bytes=(
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        trace=trace,
        phase="target_support_only_scenario_new_weight_fit",
        audits=audits,
    )
    return ScenarioDiagCosineState(
        candidate=CANDIDATE_D3_SCENARIO_OLDLOCK_NEWFIT,
        scenarios=np.asarray(FORMAL_LEO_WEAK_SCENARIOS),
        classes=classes,
        log_scale=log_scale_np,
        weights=weights_np,
        bias=bias_np,
        new_offset=offset_np,
        old_class_count=old_class_count,
        trace=tuple(trace),
        resource=resource,
    )


def _descriptor(manifest: Mapping[str, Any], kind: str) -> dict[str, Any]:
    matches = [dict(item) for item in manifest["members"] if item["kind"] == kind]
    if len(matches) != 1:
        raise DiagCosineExplorationError(f"package descriptor drift: {kind}")
    return matches[0]


def _validate_matched_packages(
    enrollment_manifest: Mapping[str, Any],
    apply_manifest: Mapping[str, Any],
) -> None:
    if enrollment_manifest.get("profile") != ENROLLMENT_ONLY:
        raise DiagCosineExplorationError("enrollment package profile drift")
    if apply_manifest.get("profile") != APPLY_ONLY:
        raise DiagCosineExplorationError("apply package profile drift")
    for field in (
        "stage",
        "registration_state",
        "receiver",
        "seed",
        "k_shot",
        "phase1_checkpoint_sha256",
        "feature_runtime_sha256",
        "method_lock_sha256",
        "registered_classes",
    ):
        if enrollment_manifest.get(field) != apply_manifest.get(field):
            raise DiagCosineExplorationError(f"enrollment/apply package mismatch: {field}")
    if (
        enrollment_manifest.get("row_handle") is not None
        or enrollment_manifest.get("row_manifest_sha256") is not None
        or not isinstance(apply_manifest.get("row_handle"), str)
        or not isinstance(apply_manifest.get("row_manifest_sha256"), str)
    ):
        raise DiagCosineExplorationError(
            "enrollment/apply row-binding profile semantics drift"
        )
    if enrollment_manifest["stage"] not in {"stage2b", "stage2c"}:
        raise DiagCosineExplorationError("package stage is not Stage2-B/C")
    if enrollment_manifest["registration_state"] not in {"before", "after"}:
        raise DiagCosineExplorationError("package registration state drift")
    k_shot = int(enrollment_manifest["k_shot"])
    if k_shot < 1 or k_shot > 20:
        raise DiagCosineExplorationError("package K-shot is outside the supported range")


def _device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if not value.startswith("cuda:") or not torch.cuda.is_available():
        raise DiagCosineExplorationError(f"requested device is unavailable: {value}")
    device = torch.device(value)
    if int(value.split(":", 1)[1]) >= torch.cuda.device_count():
        raise DiagCosineExplorationError(f"requested device is unavailable: {value}")
    return device


def _output_root(path: str | Path) -> Path:
    raw = Path(path)
    resolved = raw.resolve(strict=True)
    if raw.is_symlink() or not resolved.is_dir():
        raise DiagCosineExplorationError("output root must be a regular directory")
    if any(resolved.iterdir()):
        raise DiagCosineExplorationError("output root must be empty")
    return resolved


def _write_json_new(path: Path, payload: Mapping[str, Any] | list[Any]) -> str:
    raw = _canonical_json_bytes(payload) + b"\n"
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, stat.S_IREAD)
    return hashlib.sha256(raw).hexdigest()


def _write_npz_new(path: Path, **arrays: np.ndarray) -> str:
    with path.open("xb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, stat.S_IREAD)
    return _sha256_file(path)


def run_diag_cosine_exploration(
    *,
    enrollment_package_root: str | Path,
    enrollment_seal_path: str | Path,
    enrollment_seal_sha256: str,
    apply_package_root: str | Path,
    apply_seal_path: str | Path,
    apply_seal_sha256: str,
    output_root: str | Path,
    device: str,
    candidate: str = CANDIDATE_D1,
    parent_diag_root: str | Path | None = None,
    expected_parent_commit_sha256: str | None = None,
) -> dict[str, Any]:
    """Run one Stage2-B or Stage2-C state without ever opening truth."""

    enrollment_payloads, enrollment_manifest, enrollment_audit = (
        load_verified_somph_predictor_bundle(
            enrollment_package_root,
            detached_seal_path=enrollment_seal_path,
            expected_seal_sha256=str(enrollment_seal_sha256).lower(),
        )
    )
    query_payloads, apply_manifest, apply_audit = load_verified_somph_predictor_bundle(
        apply_package_root,
        detached_seal_path=apply_seal_path,
        expected_seal_sha256=str(apply_seal_sha256).lower(),
    )
    _validate_matched_packages(enrollment_manifest, apply_manifest)
    registration_state = str(enrollment_manifest["registration_state"])
    if candidate == CANDIDATE_D3_SCENARIO_OLDLOCK_NEWFIT:
        if registration_state == "before" and (
            parent_diag_root is not None
            or expected_parent_commit_sha256 is not None
        ):
            raise DiagCosineExplorationError(
                "D3 before state must not receive a parent diag closure"
            )
        if registration_state == "after" and (
            parent_diag_root is None
            or expected_parent_commit_sha256 is None
        ):
            raise DiagCosineExplorationError(
                "D3 after state requires a parent diag COMMIT closure"
            )
    elif (
        parent_diag_root is not None
        or expected_parent_commit_sha256 is not None
    ):
        raise DiagCosineExplorationError(
            "parent diag closure is reserved for D3 old-lock"
        )
    runtime_device = _device(device)
    model = load_torchscript_backbone_same_fd(
        enrollment_package_root,
        _descriptor(enrollment_manifest, "feature_runtime"),
        device=runtime_device,
    )

    class_handles = np.asarray(
        [item["class_handle"] for item in enrollment_manifest["registered_classes"]]
    )
    k_shot = int(enrollment_manifest["k_shot"])
    support_features: list[np.ndarray] = []
    support_labels: list[np.ndarray] = []
    support_by_scenario: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    support_forward_count = 0
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        payload = enrollment_payloads[scenario]
        ranks = np.asarray(payload["support_rank_within_class"], dtype=np.int64)
        class_indices = np.asarray(payload["support_class_indices"], dtype=np.int64)
        mask = ranks < k_shot
        if (
            class_indices.ndim != 1
            or ranks.shape != class_indices.shape
            or not np.any(mask)
            or int(class_indices[mask].min()) < 0
            or int(class_indices[mask].max()) >= len(class_handles)
        ):
            raise DiagCosineExplorationError("registered support assignment drift")
        iq = np.asarray(payload["support_leo_weak_iq"], dtype=np.float32)[mask]
        zid = forward_zid160(model, iq, device=runtime_device, batch_size=64)
        scenario_features = registered_feature(iq, zid)
        scenario_labels = class_handles[class_indices[mask]]
        support_features.append(scenario_features)
        support_labels.append(scenario_labels)
        support_by_scenario[scenario] = (
            scenario_features,
            scenario_labels,
        )
        support_forward_count += int(len(iq))
    parent_closure: dict[str, str] = {}
    if candidate == CANDIDATE_D3_SCENARIO_OLDLOCK_NEWFIT:
        if registration_state == "before":
            state: DiagCosineState | ScenarioDiagCosineState = _fit_d3_before(
                support_by_scenario,
                seed=int(enrollment_manifest["seed"]),
                device=runtime_device,
            )
        else:
            parent, parent_closure = _load_parent_scenario_state(
                parent_diag_root=parent_diag_root,
                expected_parent_commit_sha256=str(
                    expected_parent_commit_sha256
                ),
                enrollment_manifest=enrollment_manifest,
                apply_manifest=apply_manifest,
            )
            state = _fit_d3_after(
                parent,
                support_by_scenario,
                seed=int(enrollment_manifest["seed"]),
                device=runtime_device,
            )
    else:
        fit_x = np.concatenate(support_features, axis=0)
        fit_y = np.concatenate(support_labels, axis=0)
        state = fit_diag_cosine_state(
            fit_x,
            fit_y,
            seed=int(enrollment_manifest["seed"]),
            device=runtime_device,
            candidate=candidate,
        )

    query_tokens: list[np.ndarray] = []
    scenarios: list[np.ndarray] = []
    predicted_handles: list[np.ndarray] = []
    query_forward_count = 0
    scoring_started = time.perf_counter()
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        payload = query_payloads[scenario]
        iq = np.asarray(payload["query_leo_weak_iq"], dtype=np.float32)
        zid = forward_zid160(model, iq, device=runtime_device, batch_size=1)
        features = registered_feature(iq, zid)
        predicted = (
            _scenario_predict(state, scenario, features)
            if isinstance(state, ScenarioDiagCosineState)
            else predict_diag_cosine(state, features)
        )
        query_tokens.append(np.asarray(payload["query_tokens"]).astype(str))
        scenarios.append(np.asarray([scenario] * len(iq)))
        predicted_handles.append(predicted.astype(str))
        query_forward_count += int(len(iq))
    scoring_elapsed = time.perf_counter() - scoring_started

    output = _output_root(output_root)
    if isinstance(state, ScenarioDiagCosineState):
        state_sha256 = _write_npz_new(
            output / "diag_cosine_state.npz",
            scenarios=state.scenarios.astype(str),
            classes=state.classes.astype(str),
            log_scale=state.log_scale.astype(np.float32),
            weights=state.weights.astype(np.float32),
            bias=state.bias.astype(np.float32),
            new_offset=state.new_offset.astype(np.float32),
            old_class_count=np.asarray(
                [state.old_class_count], dtype=np.int64
            ),
        )
    else:
        state_sha256 = _write_npz_new(
            output / "diag_cosine_state.npz",
            classes=state.classes.astype(str),
            log_scale=state.log_scale.astype(np.float32),
            weights=state.weights.astype(np.float32),
            bias=state.bias.astype(np.float32),
        )
    serialized_state_bytes = (output / "diag_cosine_state.npz").stat().st_size
    if serialized_state_bytes > MAX_PERSISTENT_STATE_BYTES:
        raise DiagCosineExplorationError(
            "serialized diag-cosine state cap exceeded"
        )
    trace_sha256 = _write_json_new(output / "loss_trace.json", list(state.trace))
    prediction_sha256 = _write_npz_new(
        output / "prediction_artifact.npz",
        query_tokens=np.concatenate(query_tokens).astype(str),
        scenarios=np.concatenate(scenarios).astype(str),
        predicted_class_handles=np.concatenate(predicted_handles).astype(str),
    )
    resource = {
        **state.resource,
        "serialized_persistent_state_bytes": int(serialized_state_bytes),
        "support_backbone_forward_count": support_forward_count,
        "query_backbone_forward_count": query_forward_count,
        "query_backbone_forwards_per_sample": 1,
        "fft_extractions_per_query": 1,
        "score_matrix_latency_sec": float(scoring_elapsed),
        "onboard_scoring_latency_per_query_ms": float(
            scoring_elapsed * 1000.0 / max(1, query_forward_count)
        ),
    }
    receipt = {
        "schema": "cvs.phase2.diag_cosine_exploration_receipt.v1",
        "status": "DEVELOPMENT_EXPLORATION_COMPLETE",
        "formal_launch_authority": False,
        "formal_metric_claim_allowed": False,
        "stage": enrollment_manifest["stage"],
        "registration_state": enrollment_manifest["registration_state"],
        "receiver": enrollment_manifest["receiver"],
        "seed": enrollment_manifest["seed"],
        "k_shot": k_shot,
        "registered_class_count": len(class_handles),
        "row_handle": apply_manifest["row_handle"],
        "row_manifest_sha256": apply_manifest["row_manifest_sha256"],
        "phase1_checkpoint_sha256": enrollment_manifest[
            "phase1_checkpoint_sha256"
        ],
        "feature_runtime_sha256": enrollment_manifest["feature_runtime_sha256"],
        "method_lock_sha256": enrollment_manifest["method_lock_sha256"],
        "enrollment_package_root_sha256": enrollment_manifest[
            "package_root_sha256"
        ],
        "enrollment_package_seal_sha256": str(enrollment_seal_sha256).lower(),
        "apply_package_root_sha256": apply_manifest["package_root_sha256"],
        "apply_package_seal_sha256": str(apply_seal_sha256).lower(),
        "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "phase2_clean_dataset_reachable": False,
        "phase2_clean_cache_reachable": False,
        "phase2_clean_control_flow_reachable": False,
        "phase2_source_sample_access": False,
        "phase2_source_cache_access": False,
        "phase2_source_label_access": False,
        "phase2_source_derived_signal_access": False,
        "phase2_source_replay": False,
        "phase2_external_source_adapter_access": False,
        "phase2_pretrained_artifact_policy": "sealed_phase1_checkpoint_only",
        "phase2_query_decision_policy": "per_sample_all_registered_classes",
        "phase2_query_role_oracle_access": False,
        "phase2_query_true_batch_class_count_access": False,
        "phase2_query_class_quota_access": False,
        "phase2_query_batch_global_assignment": False,
        "query_truth_present_in_predictor": False,
        "parent_diag_closure": parent_closure,
        "support_scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "query_scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "candidate": {
            "name": candidate,
            "auxiliary_feature": "same_row_fft96_plus_gain_normalized_rf32",
            "auxiliary_weight": AUXILIARY_WEIGHT,
            "adaptation_epochs": ADAPTATION_EPOCHS,
            "query_view_count": 1,
            "classifier_state_policy": state.resource[
                "classifier_state_policy"
            ],
            "class_bias_enabled": state.resource["class_bias_enabled"],
            "log_scale_bounds": state.resource["log_scale_bounds"],
            "old_head_bitwise_locked": (
                candidate == CANDIDATE_D3_SCENARIO_OLDLOCK_NEWFIT
                and registration_state == "after"
            ),
            "new_class_bias_policy": state.resource.get(
                "new_class_bias_policy"
            ),
        },
        "resource": resource,
        "preopen_audit": {
            "enrollment": enrollment_audit,
            "apply": apply_audit,
        },
        "artifacts": {
            "diag_cosine_state.npz": state_sha256,
            "loss_trace.json": trace_sha256,
            "prediction_artifact.npz": prediction_sha256,
        },
    }
    receipt_sha256 = _write_json_new(output / "execution_receipt.json", receipt)
    members = [
        {
            "relative_path": path.name,
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(output.iterdir(), key=lambda value: value.name)
    ]
    commit = {
        "schema": "cvs.phase2.diag_cosine_exploration_commit.v1",
        "members": members,
        "artifact_root_sha256": hashlib.sha256(
            _canonical_json_bytes(members)
        ).hexdigest(),
        "execution_receipt_sha256": receipt_sha256,
        "prediction_artifact_sha256": prediction_sha256,
        **parent_closure,
    }
    commit_sha256 = _write_json_new(output / "COMMIT.json", commit)
    return {
        "schema": "cvs.phase2.diag_cosine_exploration_stdout.v1",
        "stage": enrollment_manifest["stage"],
        "registration_state": enrollment_manifest["registration_state"],
        "prediction_artifact_sha256": prediction_sha256,
        "execution_receipt_sha256": receipt_sha256,
        "commit_sha256": commit_sha256,
        "trainable_parameters": state.resource["trainable_parameters"],
        "persistent_state_bytes": state.resource["persistent_state_bytes"],
        "formal_launch_authority": False,
    }


__all__ = [
    "ADAPTATION_EPOCHS",
    "AUXILIARY_WEIGHT",
    "CANDIDATE_D1",
    "CANDIDATE_D1_B0_CAP",
    "CANDIDATE_D2",
    "CANDIDATE_D3_SCENARIO_OLDLOCK_NEWFIT",
    "CANDIDATES",
    "DiagCosineExplorationError",
    "DiagCosineState",
    "ScenarioDiagCosineState",
    "fit_diag_cosine_state",
    "forward_zid160",
    "log_scale_bounds",
    "predict_diag_cosine",
    "registered_feature",
    "rf_statistics",
    "run_diag_cosine_exploration",
    "spectral_logmag_sketch",
]
