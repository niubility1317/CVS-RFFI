"""Quantized Phase1 source-distribution summaries for WISER-RF.

The formal Phase2 surface accepts only the immutable aggregate members listed
below.  It never exposes source samples or a persistent dequantized source bank.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import torch
import torch.nn.functional as F


SUMMARY_SCHEMA: Final = "int8_domain_class_center_lowrank_residual_radius_v2"
ALLOWED_MEMBERS: Final = frozenset(
    {
        "schema",
        "feature_schema",
        "residual_rank",
        "center_domain_handle",
        "domain_registry",
        "residual_domain_registry",
        "class_registry",
        "core_q",
        "core_scale",
        "residual_basis_q",
        "residual_basis_scale",
        "residual_coeff_q",
        "residual_coeff_scale",
        "radius_q",
        "radius_scale",
    }
)
DENSE_DOMAIN_MEMBERS: Final = frozenset(
    {
        "domain_class_q",
        "domain_class_scale",
        "domain_class_mask",
        "domain_registry",
        "class_registry",
        "feature_schema",
    }
)


def _scalar_string(value: np.ndarray, label: str) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{label} must be a scalar string")
    result = str(array.item())
    if not result:
        raise ValueError(f"{label} must be nonempty")
    return result


def _string_tuple(value: np.ndarray, label: str) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.ndim != 1 or array.size < 1:
        raise ValueError(f"{label} must be a nonempty string registry")
    result = tuple(str(item) for item in array.tolist())
    if any(not item for item in result) or len(set(result)) != len(result):
        raise ValueError(f"{label} must contain unique nonempty strings")
    return result


def _finite_positive(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.isfinite(array).all() or not (array > 0).all():
        raise ValueError(f"{label} must be finite and positive")
    return array


@dataclass(frozen=True)
class QuantizedSourceSummary:
    """Immutable dequantized tensors required to construct source sigma points."""

    feature_schema: str
    class_registry: tuple[str, ...]
    centers: torch.Tensor
    basis: torch.Tensor
    coefficients: torch.Tensor
    radii: torch.Tensor
    direct_points: torch.Tensor | None = None

    def virtual_source_points(self) -> torch.Tensor:
        """Return normalized center and symmetric low-rank sigma points."""

        if self.direct_points is not None:
            return F.normalize(self.direct_points.detach().float(), dim=-1).detach()
        centers = F.normalize(self.centers.detach().float(), dim=-1)
        basis = F.normalize(self.basis.detach().float(), dim=-1)
        coefficients = self.coefficients.detach().float()
        radii = self.radii.detach().float()
        rank = int(basis.shape[1])
        coeff_rms = coefficients.square().mean(dim=0).sqrt()
        radius_share = radii.median(dim=0).values[:, None] / max(rank, 1) ** 0.5
        tau = (coeff_rms.square() + radius_share.square()).sqrt()
        offsets = tau[..., None] * basis
        positive = F.normalize(centers[:, None, :] + offsets, dim=-1)
        negative = F.normalize(centers[:, None, :] - offsets, dim=-1)
        points = torch.cat((centers[:, None, :], positive, negative), dim=1)
        return points.detach()


def load_quantized_source_summary(path: str | Path) -> QuantizedSourceSummary:
    """Load the exhaustive quantized aggregate allowlist from one NPZ file."""

    summary_path = Path(path)
    if not summary_path.is_file():
        raise ValueError(f"quantized source summary is missing: {summary_path}")
    with np.load(summary_path, allow_pickle=False) as arrays:
        members = set(arrays.files)
        if members == DENSE_DOMAIN_MEMBERS:
            feature_schema = _scalar_string(arrays["feature_schema"], "feature_schema")
            classes = _string_tuple(arrays["class_registry"], "class_registry")
            domains = _string_tuple(arrays["domain_registry"], "domain_registry")
            quantized = np.asarray(arrays["domain_class_q"])
            scales = _finite_positive(arrays["domain_class_scale"], "domain_class_scale")
            mask = np.asarray(arrays["domain_class_mask"])
            expected_prefix = (len(domains), len(classes))
            if (
                quantized.dtype != np.int8
                or quantized.ndim != 3
                or quantized.shape[:2] != expected_prefix
                or scales.shape != expected_prefix
                or mask.shape != expected_prefix
                or mask.dtype != np.uint8
                or not np.isin(mask, (0, 1)).all()
            ):
                raise ValueError("dense domain-class quantized summary geometry drift")
            counts = mask.sum(axis=0)
            if counts.min() < 1 or not np.all(counts == counts[0]):
                raise ValueError("every class needs the same nonzero valid domain count")
            dense = quantized.astype(np.float32) * scales.astype(np.float32)[..., None]
            by_class = np.stack(
                [dense[mask[:, class_index].astype(bool), class_index] for class_index in range(len(classes))],
                axis=0,
            )
            if not np.isfinite(by_class).all():
                raise ValueError("dense domain-class summary contains nonfinite values")
            points = F.normalize(torch.from_numpy(np.array(by_class, copy=True)), dim=-1)
            centers = F.normalize(points.mean(dim=1), dim=-1)
            empty = torch.empty(0, dtype=torch.float32)
            return QuantizedSourceSummary(
                feature_schema=feature_schema,
                class_registry=classes,
                centers=centers.detach(),
                basis=empty,
                coefficients=empty,
                radii=empty,
                direct_points=points.detach(),
            )
        if members != ALLOWED_MEMBERS:
            raise ValueError(
                "quantized source summary contains forbidden or unexpected members: "
                f"missing={sorted(ALLOWED_MEMBERS - members)}, "
                f"extra={sorted(members - ALLOWED_MEMBERS)}"
            )
        schema = _scalar_string(arrays["schema"], "schema")
        if schema != SUMMARY_SCHEMA:
            raise ValueError("quantized source summary schema mismatch")
        feature_schema = _scalar_string(arrays["feature_schema"], "feature_schema")
        classes = _string_tuple(arrays["class_registry"], "class_registry")
        domains = _string_tuple(arrays["domain_registry"], "domain_registry")
        residual_domains = _string_tuple(
            arrays["residual_domain_registry"], "residual_domain_registry"
        )
        center_domain = _scalar_string(
            arrays["center_domain_handle"], "center_domain_handle"
        )
        if center_domain not in domains or tuple(
            item for item in domains if item != center_domain
        ) != residual_domains:
            raise ValueError("center/residual domain registry mismatch")

        core_q = np.asarray(arrays["core_q"])
        core_scale = _finite_positive(arrays["core_scale"], "core_scale")
        basis_q = np.asarray(arrays["residual_basis_q"])
        basis_scale = _finite_positive(
            arrays["residual_basis_scale"], "residual_basis_scale"
        )
        coeff_q = np.asarray(arrays["residual_coeff_q"])
        coeff_scale = _finite_positive(
            arrays["residual_coeff_scale"], "residual_coeff_scale"
        )
        radius_q = np.asarray(arrays["radius_q"])
        radius_scale = _finite_positive(arrays["radius_scale"], "radius_scale")
        rank = int(np.asarray(arrays["residual_rank"]).item())

        class_count = len(classes)
        if core_q.dtype != np.int8 or core_q.ndim != 2 or core_q.shape[0] != class_count:
            raise ValueError("core_q must be int8 [class,feature]")
        feature_dim = int(core_q.shape[1])
        if core_scale.shape != (class_count,):
            raise ValueError("core_scale must align with classes")
        if (
            basis_q.dtype != np.int8
            or basis_q.shape != (class_count, rank, feature_dim)
            or basis_scale.shape != (class_count, rank)
        ):
            raise ValueError("residual basis shape/dtype mismatch")
        if (
            coeff_q.dtype != np.int8
            or coeff_q.shape != (len(residual_domains), class_count, rank)
            or coeff_scale.shape != (len(residual_domains), class_count)
        ):
            raise ValueError("residual coefficient shape/dtype mismatch")
        if (
            radius_q.dtype != np.int8
            or radius_q.shape != (len(domains), class_count)
            or radius_scale.shape != (class_count,)
        ):
            raise ValueError("radius shape/dtype mismatch")

        centers_np = core_q.astype(np.float32) * core_scale.astype(np.float32)[:, None]
        basis_np = basis_q.astype(np.float32) * basis_scale.astype(np.float32)[..., None]
        coeff_np = coeff_q.astype(np.float32) * coeff_scale.astype(np.float32)[..., None]
        radii_np = radius_q.astype(np.float32) * radius_scale.astype(np.float32)[None, :]

    tensors = [centers_np, basis_np, coeff_np, radii_np]
    if not all(np.isfinite(value).all() for value in tensors):
        raise ValueError("dequantized source summary contains nonfinite values")
    return QuantizedSourceSummary(
        feature_schema=feature_schema,
        class_registry=classes,
        centers=F.normalize(
            torch.from_numpy(np.array(centers_np, copy=True)), dim=-1
        ).requires_grad_(False),
        basis=torch.from_numpy(np.array(basis_np, copy=True)).requires_grad_(False),
        coefficients=torch.from_numpy(np.array(coeff_np, copy=True)).requires_grad_(False),
        radii=torch.from_numpy(np.array(radii_np, copy=True)).requires_grad_(False),
    )


def classwise_sliced_wasserstein(
    target_features: torch.Tensor,
    target_labels: torch.Tensor,
    virtual_source_points: torch.Tensor,
    *,
    num_projections: int = 32,
    seed: int = 0,
) -> torch.Tensor:
    """Differentiable class-wise sliced W2 against immutable source points."""

    if target_features.ndim != 2 or target_features.shape[0] != target_labels.numel():
        raise ValueError("target features/labels must align")
    if virtual_source_points.ndim != 3:
        raise ValueError("virtual source points must be [class,point,feature]")
    if target_features.shape[1] != virtual_source_points.shape[2]:
        raise ValueError("target and source feature dimensions must match")
    if int(num_projections) < 1:
        raise ValueError("num_projections must be positive")

    with torch.autocast(device_type=target_features.device.type, enabled=False):
        target = F.normalize(target_features.float(), dim=-1)
        source = F.normalize(
            virtual_source_points.detach().to(target.device, torch.float32), dim=-1
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        directions = torch.randn(
            (target.shape[1], int(num_projections)),
            generator=generator,
            dtype=torch.float32,
        ).to(target.device)
        directions = F.normalize(directions, dim=0)
        losses = []
        for class_index in range(int(source.shape[0])):
            selected = target[target_labels.view(-1).long() == class_index]
            if selected.shape[0] < 1:
                raise ValueError(f"target support is missing class {class_index}")
            target_proj = selected @ directions
            source_proj = source[class_index] @ directions
            quantile_count = max(int(target_proj.shape[0]), int(source_proj.shape[0]))
            quantiles = torch.linspace(
                0.0, 1.0, quantile_count, device=target.device, dtype=torch.float32
            )
            target_q = torch.quantile(target_proj, quantiles, dim=0)
            source_q = torch.quantile(source_proj, quantiles, dim=0)
            losses.append((target_q - source_q).square().mean())
        return torch.stack(losses).mean()


__all__ = [
    "ALLOWED_MEMBERS",
    "DENSE_DOMAIN_MEMBERS",
    "QuantizedSourceSummary",
    "SUMMARY_SCHEMA",
    "classwise_sliced_wasserstein",
    "load_quantized_source_summary",
]
