"""Finite-safe gradient surgery utilities for the BiCAD-XR tail modules."""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


DEFAULT_LOCAL_PROJECTION_ALLOWLIST = (
    "identity_last_block",
    "fusion",
    "projection",
)


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        resolved = operator.index(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if resolved <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(resolved)


def _finite_tensor(value: Tensor, name: str) -> Tensor:
    if not torch.is_tensor(value):
        raise ValueError(f"{name} must be a tensor")
    if not value.is_floating_point():
        raise ValueError(f"{name} must use a floating-point dtype")
    if value.numel() == 0:
        raise ValueError(f"{name} must be non-empty")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value


def safe_svd(
    matrix: Tensor,
    rank: int | None = None,
    *,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run a finite-checked reduced SVD and optionally keep its leading rank."""

    _finite_tensor(matrix, "matrix")
    if matrix.ndim != 2:
        raise ValueError("matrix must have shape [row, column]")
    try:
        resolved_eps = float(eps)
    except (TypeError, ValueError) as exc:
        raise ValueError("eps must be a finite positive number") from exc
    if not math.isfinite(resolved_eps) or resolved_eps <= 0.0:
        raise ValueError("eps must be a finite positive number")
    if rank is not None:
        rank = _positive_integer("rank", rank)

    try:
        u, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    except RuntimeError as exc:
        raise ValueError("SVD failed for the finite matrix") from exc
    if not (
        torch.isfinite(u).all()
        and torch.isfinite(singular_values).all()
        and torch.isfinite(vh).all()
    ):
        raise ValueError("SVD returned non-finite values")
    if rank is None:
        return u, singular_values, vh
    keep = min(rank, singular_values.numel(), vh.size(0))
    return u[:, :keep], singular_values[:keep], vh[:keep, :]


def _validate_gradient(value: Tensor, name: str) -> Tensor:
    return _finite_tensor(value, name)


def _project_one(gradient: Tensor, reference: Tensor, eps: float) -> Tensor:
    _validate_gradient(gradient, "gradient")
    _validate_gradient(reference, "reference")
    if gradient.device != reference.device:
        raise ValueError("gradient and reference must use the same device")
    if gradient.numel() != reference.numel():
        raise ValueError("gradient and reference must have the same number of elements")
    current = gradient
    dot = torch.sum(current.reshape(-1) * reference.reshape(-1))
    if float(dot.detach().item()) < 0.0:
        denominator = torch.sum(reference.reshape(-1).square()).clamp_min(eps)
        current = current - (dot / denominator) * reference
    if not torch.isfinite(current).all():
        raise ValueError("projected gradient became non-finite")
    return current


def project_conflicting_gradient(
    gradient: Tensor | Sequence[Tensor],
    references: Tensor | Sequence[Tensor] | None = None,
    *,
    eps: float = 1e-12,
    parameter_name: str | None = None,
    allowlist: Sequence[str] | None = None,
) -> Tensor | list[Tensor]:
    """Project a gradient away from conflicting reference gradients.

    A tensor input returns one tensor.  A sequence input applies pairwise
    projection to every task gradient and returns a same-order list.  All
    arithmetic is finite checked and no input tensor is modified in place.
    """

    try:
        resolved_eps = float(eps)
    except (TypeError, ValueError) as exc:
        raise ValueError("eps must be a finite positive number") from exc
    if not math.isfinite(resolved_eps) or resolved_eps <= 0.0:
        raise ValueError("eps must be a finite positive number")

    if parameter_name is not None or allowlist is not None:
        if not isinstance(parameter_name, str) or not parameter_name.strip():
            raise ValueError("parameter_name is required when allowlist is used")
        resolved_allowlist = _validate_projection_allowlist(
            DEFAULT_LOCAL_PROJECTION_ALLOWLIST if allowlist is None else allowlist
        )
        if not _projection_name_allowed(parameter_name, resolved_allowlist):
            if not torch.is_tensor(gradient):
                raise ValueError("allowlist is supported only for tensor gradients")
            _validate_gradient(gradient, "gradient")
            return gradient.clone()

    if torch.is_tensor(gradient):
        _validate_gradient(gradient, "gradient")
        if references is None:
            reference_list: list[Tensor] = []
        elif torch.is_tensor(references):
            reference_list = [references]
        elif isinstance(references, Sequence) and not isinstance(references, (str, bytes)):
            reference_list = list(references)
        else:
            raise ValueError("references must be a tensor, sequence, or None")
        projected = gradient.clone()
        for reference in reference_list:
            projected = _project_one(projected, reference, resolved_eps)
        return projected

    if not isinstance(gradient, Sequence) or isinstance(gradient, (str, bytes)):
        raise ValueError("gradient must be a tensor or sequence of tensors")
    gradients = list(gradient)
    if not gradients:
        raise ValueError("gradient sequence must be non-empty")
    for index, value in enumerate(gradients):
        _validate_gradient(value, f"gradient[{index}]")
    if references is not None:
        raise ValueError("references is not accepted for sequence gradient input")

    originals = [value.clone() for value in gradients]
    projected = [value.clone() for value in gradients]
    for index, current in enumerate(projected):
        for reference_index, reference in enumerate(originals):
            if index == reference_index:
                continue
            current = _project_one(current, reference, resolved_eps)
        projected[index] = current
    return projected


def _validate_projection_allowlist(
    allowlist: Sequence[str],
) -> tuple[str, ...]:
    if isinstance(allowlist, (str, bytes)) or not isinstance(allowlist, Sequence):
        raise ValueError("allowlist must contain non-empty parameter prefixes")
    resolved = tuple(allowlist)
    if not resolved or any(not isinstance(name, str) or not name.strip() for name in resolved):
        raise ValueError("allowlist must contain non-empty parameter prefixes")
    if len(set(resolved)) != len(resolved):
        raise ValueError("allowlist must not contain duplicates")
    return resolved


def _projection_name_allowed(name: str, allowlist: Sequence[str]) -> bool:
    return any(
        name == prefix
        or name.startswith(prefix + ".")
        or name.startswith(prefix + "/")
        for prefix in allowlist
    )


def project_local_conflicting_gradients(
    gradients: Mapping[str, Tensor],
    references: Mapping[str, Tensor],
    *,
    allowlist: Sequence[str] = DEFAULT_LOCAL_PROJECTION_ALLOWLIST,
    eps: float = 1e-12,
) -> dict[str, Tensor]:
    """Project only explicitly allowlisted named gradients.

    Non-allowlisted gradients are cloned unchanged.  The two mappings are
    keyed by parameter name so the caller cannot accidentally project an
    unrelated shared stem or discriminator parameter.
    """

    if not isinstance(gradients, Mapping) or not isinstance(references, Mapping):
        raise ValueError("gradients and references must be mappings")
    if not gradients:
        raise ValueError("gradients must be non-empty")
    resolved_allowlist = _validate_projection_allowlist(allowlist)
    projected: dict[str, Tensor] = {}
    for name, gradient in gradients.items():
        if not isinstance(name, str) or not name:
            raise ValueError("gradient names must be non-empty strings")
        _validate_gradient(gradient, f"gradients[{name!r}]")
        if not _projection_name_allowed(name, resolved_allowlist):
            projected[name] = gradient.clone()
            continue
        if name not in references:
            raise ValueError(f"missing reference gradient for {name}")
        projected_value = project_conflicting_gradient(
            gradient,
            references[name],
            eps=eps,
        )
        assert torch.is_tensor(projected_value)
        projected[name] = projected_value
    return projected


project_local_conflicting_gradient = project_local_conflicting_gradients
project_task_protected_gradients = project_local_conflicting_gradients


def _gradient_norm(value: Tensor, name: str, eps: float) -> Tensor:
    _validate_gradient(value, name)
    norm = torch.linalg.vector_norm(value.reshape(-1))
    if not torch.isfinite(norm):
        raise ValueError(f"{name} norm must be finite")
    return norm.clamp_min(eps).detach()


def _flatten_gradient_values(
    value: Tensor | Sequence[Tensor], name: str
) -> Tensor:
    if torch.is_tensor(value):
        _validate_gradient(value, name)
        return value.reshape(-1)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a tensor or tensor sequence")
    values = list(value)
    if not values:
        raise ValueError(f"{name} must be non-empty")
    flattened = [
        _flatten_gradient_values(item, f"{name}[{index}]")
        for index, item in enumerate(values)
    ]
    device = flattened[0].device
    if any(item.device != device for item in flattened):
        raise ValueError(f"{name} tensors must use the same device")
    return torch.cat(flattened)


@dataclass(frozen=True)
class GradientRatioAudit:
    """Measured weighted-gradient ratio and the multiplier used to cap it."""

    initial_weight: float
    max_ratio: float
    reference_norm: float
    controlled_norm: float
    raw_ratio: float
    effective_ratio: float
    scale: float
    effective_weight: float


def measure_bounded_gradient_ratio(
    reference_gradient: Tensor | Sequence[Tensor],
    controlled_gradient: Tensor | Sequence[Tensor],
    *,
    initial_weight: float,
    max_ratio: float = 0.05,
    eps: float = 1e-12,
) -> GradientRatioAudit:
    """Measure and cap a controlled loss relative to a task gradient.

    ``scale`` is a multiplier on ``initial_weight``.  The raw ratio includes
    that initial weight, while the effective ratio is the ratio represented
    by the weight returned in ``effective_weight``.  All returned values are
    detached Python floats so they are safe to persist in runtime audits.
    """

    try:
        resolved_weight = float(initial_weight)
        resolved_max_ratio = float(max_ratio)
        resolved_eps = float(eps)
    except (TypeError, ValueError) as exc:
        raise ValueError("gradient ratio parameters must be finite numbers") from exc
    if not math.isfinite(resolved_weight) or resolved_weight < 0.0:
        raise ValueError("initial_weight must be finite and non-negative")
    if not math.isfinite(resolved_max_ratio) or resolved_max_ratio < 0.0:
        raise ValueError("max_ratio must be finite and non-negative")
    if not math.isfinite(resolved_eps) or resolved_eps <= 0.0:
        raise ValueError("eps must be finite and positive")

    reference = _flatten_gradient_values(reference_gradient, "reference_gradient")
    controlled = _flatten_gradient_values(controlled_gradient, "controlled_gradient")
    if reference.device != controlled.device:
        raise ValueError("reference and controlled gradients must use the same device")
    reference_norm = float(torch.linalg.vector_norm(reference).detach().item())
    controlled_norm = float(torch.linalg.vector_norm(controlled).detach().item())
    if not math.isfinite(reference_norm) or not math.isfinite(controlled_norm):
        raise ValueError("gradient norms must be finite")
    raw_ratio = resolved_weight * controlled_norm / max(reference_norm, resolved_eps)
    if not math.isfinite(raw_ratio):
        raise ValueError("raw gradient ratio must be finite")
    scale = 1.0 if raw_ratio <= resolved_max_ratio else resolved_max_ratio / raw_ratio
    if not math.isfinite(scale) or scale < 0.0 or scale > 1.0:
        raise ValueError("gradient ratio scale must be finite and in [0,1]")
    effective_ratio = raw_ratio * scale
    effective_weight = resolved_weight * scale
    if not math.isfinite(effective_ratio) or not math.isfinite(effective_weight):
        raise ValueError("effective gradient ratio must be finite")
    return GradientRatioAudit(
        initial_weight=resolved_weight,
        max_ratio=resolved_max_ratio,
        reference_norm=reference_norm,
        controlled_norm=controlled_norm,
        raw_ratio=raw_ratio,
        effective_ratio=effective_ratio,
        scale=scale,
        effective_weight=effective_weight,
    )


def scale_explicit_gradients(parameters: Sequence[Tensor], scale: float) -> int:
    """Scale gradients only for the explicitly supplied parameter sequence."""

    if not isinstance(parameters, Sequence) or isinstance(parameters, (str, bytes)):
        raise ValueError("parameters must be an explicit parameter sequence")
    try:
        resolved_scale = float(scale)
    except (TypeError, ValueError) as exc:
        raise ValueError("scale must be a finite non-negative number") from exc
    if not math.isfinite(resolved_scale) or resolved_scale < 0.0:
        raise ValueError("scale must be a finite non-negative number")

    gradients: list[Tensor] = []
    for index, parameter in enumerate(parameters):
        if not torch.is_tensor(parameter):
            raise ValueError(f"parameters[{index}] must be a tensor")
        if parameter.grad is None:
            continue
        _validate_gradient(parameter.grad, f"parameters[{index}].grad")
        gradients.append(parameter.grad)
    originals = [gradient.detach().clone() for gradient in gradients]
    with torch.no_grad():
        for gradient in gradients:
            gradient.mul_(resolved_scale)
        if any(not torch.isfinite(gradient).all() for gradient in gradients):
            for gradient, original in zip(gradients, originals):
                gradient.copy_(original)
            raise ValueError("scaled gradients must contain only finite values")
    return len(gradients)


class GradientRatioController:
    """Detach-smoothed ratio controller for one explicitly named task pair."""

    def __init__(
        self,
        target_ratio: float = 1.0,
        ema_decay: float = 0.9,
        min_scale: float = 0.0,
        max_scale: float = 10.0,
        eps: float = 1e-8,
    ) -> None:
        values = {
            "target_ratio": target_ratio,
            "ema_decay": ema_decay,
            "min_scale": min_scale,
            "max_scale": max_scale,
            "eps": eps,
        }
        try:
            resolved = {name: float(value) for name, value in values.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("controller parameters must be finite numbers") from exc
        if not math.isfinite(resolved["target_ratio"]) or resolved["target_ratio"] < 0.0:
            raise ValueError("target_ratio must be finite and non-negative")
        if not math.isfinite(resolved["ema_decay"]) or not 0.0 <= resolved["ema_decay"] < 1.0:
            raise ValueError("ema_decay must be finite and in [0,1)")
        if not math.isfinite(resolved["min_scale"]) or resolved["min_scale"] < 0.0:
            raise ValueError("min_scale must be finite and non-negative")
        if not math.isfinite(resolved["max_scale"]) or resolved["max_scale"] < resolved["min_scale"]:
            raise ValueError("max_scale must be finite and >= min_scale")
        if not math.isfinite(resolved["eps"]) or resolved["eps"] <= 0.0:
            raise ValueError("eps must be finite and positive")
        self.target_ratio = resolved["target_ratio"]
        self.ema_decay = resolved["ema_decay"]
        self.min_scale = resolved["min_scale"]
        self.max_scale = resolved["max_scale"]
        self.eps = resolved["eps"]
        self._ema_ratio: Tensor | None = None
        self._last_scale: float | None = None

    @property
    def ema_ratio(self) -> Tensor | None:
        return None if self._ema_ratio is None else self._ema_ratio.detach().clone()

    @property
    def ratio(self) -> float | None:
        return None if self._ema_ratio is None else float(self._ema_ratio.item())

    @property
    def last_scale(self) -> float | None:
        return self._last_scale

    def update(self, reference_gradient: Tensor, controlled_gradient: Tensor) -> float:
        """Return the bounded detached scale for the controlled task."""

        reference_norm = _gradient_norm(reference_gradient, "reference_gradient", self.eps)
        controlled_norm = _gradient_norm(controlled_gradient, "controlled_gradient", self.eps)
        raw_ratio = (
            self.target_ratio * reference_norm / controlled_norm
        ).detach()
        if not torch.isfinite(raw_ratio).all():
            raise ValueError("raw gradient ratio must be finite")
        if self._ema_ratio is None:
            next_ratio = raw_ratio.clone().detach()
        else:
            if self._ema_ratio.device != raw_ratio.device:
                raise ValueError("gradient ratio device must remain constant")
            previous_component = self.ema_decay * self._ema_ratio
            current_component = (1.0 - self.ema_decay) * raw_ratio
            if not (
                torch.isfinite(previous_component).all()
                and torch.isfinite(current_component).all()
            ):
                raise ValueError("gradient ratio EMA components must be finite")
            next_ratio = (previous_component + current_component).detach()
        if not torch.isfinite(next_ratio).all():
            raise ValueError("gradient ratio EMA must be finite")
        bounded = next_ratio.clamp(self.min_scale, self.max_scale)
        if not torch.isfinite(bounded).all():
            raise ValueError("bounded gradient ratio scale must be finite")
        scale = float(bounded.item())
        if not math.isfinite(scale):
            raise ValueError("gradient ratio scale must be finite")
        self._ema_ratio = next_ratio
        self._last_scale = scale
        return scale

    compute_scale = update

    def scale_parameters(
        self,
        parameters: Sequence[Tensor],
        scale: float | None = None,
    ) -> int:
        """Scale only the explicit list, using the latest controller scale."""

        if scale is None:
            if self._last_scale is None:
                raise ValueError("update must be called before an implicit scale")
            scale = self._last_scale
        return scale_explicit_gradients(parameters, scale)

    apply = scale_parameters


scale_parameter_gradients = scale_explicit_gradients


__all__ = [
    "DEFAULT_LOCAL_PROJECTION_ALLOWLIST",
    "GradientRatioAudit",
    "GradientRatioController",
    "measure_bounded_gradient_ratio",
    "project_local_conflicting_gradient",
    "project_local_conflicting_gradients",
    "project_task_protected_gradients",
    "project_conflicting_gradient",
    "safe_svd",
    "scale_explicit_gradients",
    "scale_parameter_gradients",
]
