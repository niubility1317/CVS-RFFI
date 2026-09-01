"""Canonical support-only row features for MARC-OT Phase1 and Phase2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .stage2_m23_rfguard import extract_rf_lite_quality


MARC_OT_SUPPORT_ROW_SCHEMA = "marc_ot.support.row.v1"
MARC_OT_SUPPORT_ROW_DIM = 685
MARC_OT_SUPPORTED_K = (1, 2, 5, 10, 20)
_MIN_DC_REMOVED_COMPLEX_RMS = 1.0e-8

MARC_OT_SUPPORT_LAYOUT: Mapping[str, slice] = MappingProxyType(
    {
        "z_id": slice(0, 160),
        "t_emb": slice(160, 320),
        "f_emb": slice(320, 480),
        "normalized_time_minus_frequency": slice(480, 640),
        "embedding_norms": slice(640, 643),
        "time_frequency_relation": slice(643, 645),
        "view_stability": slice(645, 651),
        "phase_clock_proxies": slice(651, 655),
        "normalized_log_psd_16": slice(655, 671),
        "rf_lite_10": slice(671, 681),
        "quality": slice(681, 682),
        "k_and_mask": slice(682, 685),
    }
)

MARC_OT_SUPPORT_FEATURE_CONFIG: Mapping[str, Any] = MappingProxyType(
    {
        "embedding_dim": 160,
        "deterministic_view": "dc_removed_complex_unit_rms_v1",
        "deterministic_view_min_complex_rms": _MIN_DC_REMOVED_COMPLEX_RMS,
        "embedding_normalization": "l2_eps_1e-8",
        "view_relative_drift": "l2_delta_over_reference_l2_eps_1e-8",
        "cfo_proxy": "wrapped_phase_increment_mean_std_PROXY_ONLY",
        "sfo_proxy": "phase_increment_half_drift_slope_PROXY_ONLY",
        "psd": "full_fft_adaptive_16_log_standardized_v1",
        "psd_bins": 16,
        "rf_lite": "stage2_m23_rfguard.scale_free.v1",
        "rf_lite_dim": 10,
        "quality": "stage2_m23_rfguard.quality.v1",
        "supported_k": MARC_OT_SUPPORTED_K,
    }
)


@dataclass(frozen=True)
class MARCOTSupportFeatureBatch:
    """Validated row-aligned ABI output without query or token semantics."""

    rows: Tensor
    labels: Tensor
    physical_tokens: tuple[object, ...]
    effective_mask: Tensor
    feature_schema: str
    feature_dim: int
    feature_config: Mapping[str, Any]
    audit: Mapping[str, Any]


def _validated_tokens(
    physical_tokens: Sequence[object], *, expected_rows: int
) -> tuple[object, ...]:
    try:
        tokens = tuple(physical_tokens)
        unique_count = len(set(tokens))
    except TypeError as error:
        raise ValueError("physical support tokens must be hashable and unique") from error
    if len(tokens) != expected_rows:
        raise ValueError("physical support token count does not match support rows")
    if unique_count != len(tokens):
        raise ValueError("physical support tokens must be unique")
    return tokens


def _validated_inputs(
    support_iq: Tensor,
    labels: Tensor,
    physical_tokens: Sequence[object],
    *,
    nominal_k: int,
    effective_mask: Tensor | Sequence[int | float] | None,
    validated_unpadded: bool,
    scope: str,
    fit_scope: str,
) -> tuple[Tensor, Tensor, tuple[object, ...], Tensor, dict[int, int]]:
    if not isinstance(support_iq, Tensor) or not support_iq.is_floating_point():
        raise ValueError("support IQ must be a floating torch.Tensor")
    if (
        support_iq.ndim != 3
        or support_iq.shape[0] == 0
        or support_iq.shape[1] != 2
        or support_iq.shape[2] < 16
    ):
        raise ValueError("support IQ geometry must be finite nonempty [N,2,L] with L>=16")
    if not bool(torch.isfinite(support_iq).all().item()):
        raise ValueError("support IQ must be finite")
    if not isinstance(labels, Tensor) or labels.shape != (support_iq.shape[0],):
        raise ValueError("support labels must be a row-aligned torch.Tensor")
    if labels.is_floating_point() or labels.dtype == torch.bool:
        raise ValueError("support labels must be integer indices")
    if labels.device != support_iq.device:
        raise ValueError("support IQ and labels must share a device")
    if (
        isinstance(nominal_k, bool)
        or not isinstance(nominal_k, int)
        or nominal_k not in MARC_OT_SUPPORTED_K
    ):
        raise ValueError("nominal K must be one of 1/2/5/10/20")
    legal_scope_pairs = {
        ("phase1_source", "full_episode"),
        ("phase2_support", "crossfit"),
        ("phase2_support", "full_support"),
    }
    if (scope, fit_scope) not in legal_scope_pairs:
        raise ValueError("MARC-OT support input scope/fit scope is invalid")
    tokens = _validated_tokens(physical_tokens, expected_rows=int(support_iq.shape[0]))

    unique_labels, label_counts = torch.unique(labels, sorted=True, return_counts=True)
    if (
        unique_labels.numel() == 0
        or bool((label_counts != label_counts[0]).any().item())
        or int(label_counts[0].item()) > nominal_k
    ):
        raise ValueError("support class K mismatch")
    allow_subset = scope == "phase2_support" and fit_scope == "crossfit"
    if int(label_counts[0].item()) < nominal_k and not allow_subset:
        raise ValueError(f"{fit_scope} class K mismatch")
    if effective_mask is None:
        if not bool(validated_unpadded):
            raise ValueError(
                "effective_mask may default to all ones only for a validated unpadded package"
            )
        if int(label_counts[0].item()) != nominal_k:
            raise ValueError("validated unpadded support class K mismatch")
        mask = torch.ones(len(labels), device=support_iq.device, dtype=support_iq.dtype)
    else:
        mask = torch.as_tensor(
            effective_mask, device=support_iq.device, dtype=support_iq.dtype
        )
        if mask.shape != (len(labels),) or not bool(torch.isfinite(mask).all().item()):
            raise ValueError("effective_mask must be finite and row aligned")
        if not bool(((mask == 0.0) | (mask == 1.0)).all().item()):
            raise ValueError("effective_mask must contain only zero or one")

    effective_by_class: dict[int, int] = {}
    for class_id in unique_labels:
        class_mask = labels == class_id
        effective_k = int(mask[class_mask].sum().item())
        if not 1 <= effective_k <= nominal_k:
            raise ValueError("every support class must retain at least one effective row")
        effective_by_class[int(class_id.item())] = effective_k
    if len(set(effective_by_class.values())) != 1:
        raise ValueError("support class effective K mismatch")
    return support_iq, labels, tokens, mask, effective_by_class


def _model_aux(model: nn.Module, values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    output = model(values, return_aux=True)
    if not isinstance(output, Mapping):
        raise ValueError("MARC-OT model aux output must be a mapping")
    aux_id = output.get("aux_id")
    if not isinstance(aux_id, Mapping):
        raise ValueError("MARC-OT model aux output is missing identity aux_id")
    members = (output.get("z_id"), aux_id.get("t_emb"), aux_id.get("f_emb"))
    for name, value in zip(("z_id", "t_emb", "f_emb"), members, strict=True):
        if (
            not isinstance(value, Tensor)
            or not value.is_floating_point()
            or value.shape != (values.shape[0], 160)
        ):
            raise ValueError(f"MARC-OT {name} geometry must be [N,160]")
        if value.device != values.device or not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"MARC-OT {name} must be finite and share the IQ device")
    return members  # type: ignore[return-value]


def _fixed_mathematical_view(values: Tensor) -> Tensor:
    centered = values - values.mean(dim=-1, keepdim=True)
    complex_rms = centered.square().sum(dim=1).mean(dim=-1, keepdim=True).sqrt()
    if bool((complex_rms <= _MIN_DC_REMOVED_COMPLEX_RMS).any().item()):
        raise ValueError("DC-removed complex RMS is below the fixed minimum")
    return centered / complex_rms.unsqueeze(1)


def _cosine_and_relative_drift(reference: Tensor, view: Tensor) -> Tensor:
    cosine = functional.cosine_similarity(reference, view, dim=1, eps=1.0e-8)
    drift = (view - reference).norm(dim=1) / reference.norm(dim=1).clamp_min(1.0e-8)
    return torch.stack((cosine, drift), dim=1)


def _phase_clock_proxies(values: Tensor) -> Tensor:
    complex_rows = torch.complex(values[:, 0].float(), values[:, 1].float())
    increments = torch.angle(complex_rows[:, 1:] * complex_rows[:, :-1].conj())
    mean = increments.mean(dim=1)
    std = increments.std(dim=1, unbiased=False)
    split = max(1, increments.shape[1] // 2)
    first = increments[:, :split].mean(dim=1)
    second = increments[:, split:].mean(dim=1) if split < increments.shape[1] else first
    half_drift = second - first
    position = torch.linspace(
        -1.0, 1.0, increments.shape[1], device=increments.device, dtype=increments.dtype
    )
    slope = (
        (increments - increments.mean(dim=1, keepdim=True)) * position.view(1, -1)
    ).sum(dim=1) / position.square().sum().clamp_min(1.0e-8)
    return torch.stack((mean, std, half_drift, slope), dim=1).to(values.dtype)


def _normalized_log_psd(values: Tensor) -> Tensor:
    complex_rows = torch.complex(values[:, 0].float(), values[:, 1].float())
    power = torch.fft.fft(complex_rows, dim=1).abs().square()
    pooled = functional.adaptive_avg_pool1d(power.unsqueeze(1), 16).squeeze(1)
    log_psd = torch.log(pooled.clamp_min(1.0e-12))
    centered = log_psd - log_psd.mean(dim=1, keepdim=True)
    scale = centered.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1.0e-8)
    return (centered / scale).to(values.dtype)


def _rf_lite(values: Tensor) -> tuple[Tensor, Tensor]:
    # The reviewed public implementation is NumPy-based and intentionally
    # does not carry gradients. The selected model features above retain their
    # autograd path; RF-lite remains a fixed observation of legal support IQ.
    iq = np.asarray(values.detach().cpu().tolist(), dtype=np.float32)
    lite, quality = extract_rf_lite_quality(iq)
    lite_tensor = torch.tensor(lite.tolist(), device=values.device, dtype=values.dtype)
    quality_tensor = torch.tensor(
        quality.tolist(), device=values.device, dtype=values.dtype
    ).view(-1, 1)
    return lite_tensor, quality_tensor


def build_marc_ot_support_features(
    model: nn.Module,
    support_iq: Tensor,
    labels: Tensor,
    physical_tokens: Sequence[object],
    *,
    nominal_k: int,
    effective_mask: Tensor | Sequence[int | float] | None = None,
    validated_unpadded: bool = False,
    scope: str,
    fit_scope: str,
) -> MARCOTSupportFeatureBatch:
    """Build the single production 685D support row ABI from legal IQ only."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    values, label_rows, tokens, mask, effective_by_class = _validated_inputs(
        support_iq,
        labels,
        physical_tokens,
        nominal_k=nominal_k,
        effective_mask=effective_mask,
        validated_unpadded=validated_unpadded,
        scope=scope,
        fit_scope=fit_scope,
    )
    module_modes = tuple((module, bool(module.training)) for module in model.modules())
    try:
        model.eval()
        z_id, t_emb, f_emb = _model_aux(model, values)
        view_z, view_t, view_f = _model_aux(model, _fixed_mathematical_view(values))
    finally:
        for module, training in module_modes:
            module.training = training

    embedding_norms = torch.stack(
        (z_id.norm(dim=1), t_emb.norm(dim=1), f_emb.norm(dim=1)), dim=1
    )
    relation = torch.stack(
        (
            functional.cosine_similarity(t_emb, f_emb, dim=1, eps=1.0e-8),
            torch.log(t_emb.norm(dim=1).clamp_min(1.0e-8))
            - torch.log(f_emb.norm(dim=1).clamp_min(1.0e-8)),
        ),
        dim=1,
    )
    view_stability = torch.cat(
        (
            _cosine_and_relative_drift(z_id, view_z),
            _cosine_and_relative_drift(t_emb, view_t),
            _cosine_and_relative_drift(f_emb, view_f),
        ),
        dim=1,
    )
    rf_lite, quality = _rf_lite(values)
    effective_k = torch.tensor(
        [float(effective_by_class[int(label.item())]) for label in label_rows],
        device=values.device,
        dtype=values.dtype,
    )
    k_and_mask = torch.stack(
        (
            torch.full_like(mask, float(nominal_k)),
            effective_k,
            mask,
        ),
        dim=1,
    )
    rows = torch.cat(
        (
            z_id,
            t_emb,
            f_emb,
            functional.normalize(t_emb, dim=1, eps=1.0e-8)
            - functional.normalize(f_emb, dim=1, eps=1.0e-8),
            embedding_norms,
            relation,
            view_stability,
            _phase_clock_proxies(values),
            _normalized_log_psd(values),
            rf_lite,
            quality,
            k_and_mask,
        ),
        dim=1,
    )
    if rows.shape != (values.shape[0], MARC_OT_SUPPORT_ROW_DIM):
        raise RuntimeError("MARC-OT support feature ABI width drift")
    if not bool(torch.isfinite(rows).all().item()):
        raise RuntimeError("MARC-OT support feature builder produced non-finite rows")
    audit = {
        "feature_schema": MARC_OT_SUPPORT_ROW_SCHEMA,
        "feature_dim": MARC_OT_SUPPORT_ROW_DIM,
        "feature_config": dict(MARC_OT_SUPPORT_FEATURE_CONFIG),
        "support_rows": int(len(rows)),
        "physical_token_count": int(len(tokens)),
        "input_scope": scope,
        "fit_scope": fit_scope,
        "nominal_k": int(nominal_k),
        "effective_k_by_class": dict(effective_by_class),
        "query_rows_used": 0,
        "deterministic_view": {
            "method": "DC_REMOVED_COMPLEX_UNIT_RMS",
            "adds_physical_rows": False,
            "adds_effective_k": False,
        },
        "cfo": {
            "status": "PROXY_ONLY",
            "coordinates": ("wrapped_phase_increment_mean", "wrapped_phase_increment_std"),
            "absolute_physical_units_available": False,
        },
        "sfo": {
            "status": "PROXY_ONLY",
            "coordinates": ("phase_increment_half_drift", "phase_increment_slope"),
            "physical_sfo_available": False,
        },
    }
    return MARCOTSupportFeatureBatch(
        rows=rows,
        labels=label_rows,
        physical_tokens=tokens,
        effective_mask=mask,
        feature_schema=MARC_OT_SUPPORT_ROW_SCHEMA,
        feature_dim=MARC_OT_SUPPORT_ROW_DIM,
        feature_config=MARC_OT_SUPPORT_FEATURE_CONFIG,
        audit=audit,
    )


__all__ = [
    "MARCOTSupportFeatureBatch",
    "MARC_OT_SUPPORTED_K",
    "MARC_OT_SUPPORT_FEATURE_CONFIG",
    "MARC_OT_SUPPORT_LAYOUT",
    "MARC_OT_SUPPORT_ROW_DIM",
    "MARC_OT_SUPPORT_ROW_SCHEMA",
    "build_marc_ot_support_features",
]
