"""Small, explicit building blocks for the CV2 adversarial game.

The game keeps the two optimizer phases separate while sharing the feature
tensor produced by the caller's single backbone forward.  This module does
not own a model forward or a training loop; it only validates the phase
contract and constructs disjoint optimizers/controllers.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple

import torch
from torch import Tensor, nn

from .gradients import GradientRatioController


_DEFAULT_CONDITIONAL_BOUNDS = (0.10, 0.20)
_DEFAULT_ZDOM_TX_BOUNDS = (0.03, 0.08)
_DEFAULT_IDENTITY_DOSE_BOUNDS = (0.05, 0.40)
_DEFAULT_ZDOM_DOSE_BOUNDS = (0.03, 0.30)
_DEFAULT_PROJECTION_ALLOWLIST = (
    "identity_last_block",
    "fusion",
    "projection",
)


def _finite_positive(value: object, name: str, *, allow_zero: bool = False) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a finite {qualifier} number") from exc
    if not math.isfinite(resolved) or (resolved < 0.0 if allow_zero else resolved <= 0.0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a finite {qualifier} number")
    return resolved


def _ratio_bounds(value: object, name: str) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two finite bounds")
    lower = _finite_positive(value[0], f"{name}[0]", allow_zero=True)
    upper = _finite_positive(value[1], f"{name}[1]", allow_zero=True)
    if not lower < upper:
        raise ValueError(f"{name} must be strictly increasing")
    return lower, upper


def _unit_interval(value: object, name: str) -> float:
    resolved = _finite_positive(value, name, allow_zero=True)
    if resolved > 1.0:
        raise ValueError(f"{name} must lie in [0,1]")
    return resolved


def _feature_tensor(value: Tensor, name: str) -> Tensor:
    if not torch.is_tensor(value):
        raise ValueError(f"{name} must be a tensor")
    if not value.is_floating_point():
        raise ValueError(f"{name} must use a floating-point dtype")
    if value.numel() == 0:
        raise ValueError(f"{name} must be non-empty")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value


def _parameter_list(value: Iterable[Tensor] | nn.Module, name: str) -> list[Tensor]:
    if isinstance(value, nn.Module):
        parameters = list(value.parameters())
    else:
        if isinstance(value, (str, bytes)):
            raise ValueError(f"{name} must be a module or parameter iterable")
        try:
            parameters = list(value)
        except TypeError as exc:
            raise ValueError(f"{name} must be a module or parameter iterable") from exc
    if not parameters:
        raise ValueError(f"{name} must be non-empty")

    seen: set[int] = set()
    for index, parameter in enumerate(parameters):
        if not torch.is_tensor(parameter):
            raise ValueError(f"{name}[{index}] must be a tensor parameter")
        if not parameter.is_leaf:
            raise ValueError(f"{name}[{index}] must be a leaf tensor parameter")
        if id(parameter) in seen:
            raise ValueError(f"{name} must not contain duplicate parameters")
        seen.add(id(parameter))
    return parameters


@dataclass(frozen=True)
class AdversarialGamePlan:
    """Describe the two phases that consume one shared backbone output."""

    one_backbone_forward: bool = True
    phase_order: tuple[str, str] = ("discriminator", "encoder")
    discriminator_lr_ratio: float = 1.5
    conditional_ratio_bounds: tuple[float, float] = _DEFAULT_CONDITIONAL_BOUNDS
    zdom_tx_ratio_bounds: tuple[float, float] = _DEFAULT_ZDOM_TX_BOUNDS
    projection_allowlist: tuple[str, ...] = _DEFAULT_PROJECTION_ALLOWLIST

    def __post_init__(self) -> None:
        if self.one_backbone_forward is not True:
            raise ValueError("CV2 adversarial game requires one_backbone_forward=True")
        if self.phase_order != ("discriminator", "encoder"):
            raise ValueError("phase_order must be discriminator then encoder")
        _finite_positive(self.discriminator_lr_ratio, "discriminator_lr_ratio")
        _ratio_bounds(self.conditional_ratio_bounds, "conditional_ratio_bounds")
        _ratio_bounds(self.zdom_tx_ratio_bounds, "zdom_tx_ratio_bounds")
        if (
            not isinstance(self.projection_allowlist, tuple)
            or not self.projection_allowlist
            or any(
                not isinstance(name, str) or not name.strip()
                for name in self.projection_allowlist
            )
        ):
            raise ValueError("projection_allowlist must contain non-empty names")
        if len(set(self.projection_allowlist)) != len(self.projection_allowlist):
            raise ValueError("projection_allowlist must not contain duplicates")

    @property
    def phases(self) -> tuple[str, str]:
        return self.phase_order

    @property
    def backbone_forward_count(self) -> int:
        return 1

    def discriminator_features(self, features: Tensor) -> Tensor:
        """Return the shared output detached for discriminator-only updates."""

        return _feature_tensor(features, "features").detach()

    def encoder_features(self, features: Tensor) -> Tensor:
        """Return the same shared output for the encoder update phase."""

        return _feature_tensor(features, "features")

    # These aliases make the phase boundary explicit at call sites without
    # introducing any second forward or copy of the encoder feature graph.
    discriminator_phase = discriminator_features
    encoder_phase = encoder_features
    for_discriminator = discriminator_features
    for_encoder = encoder_features

    def phase_inputs(self, features: Tensor) -> dict[str, Tensor]:
        """Materialize both phase inputs from one already-computed feature tensor."""

        return {
            "discriminator": self.discriminator_features(features),
            "encoder": self.encoder_features(features),
        }

    @contextmanager
    def freeze_discriminator(self, discriminator: nn.Module):
        """Freeze discriminator parameters while preserving feature gradients."""

        if not isinstance(discriminator, nn.Module):
            raise ValueError("discriminator must be a torch module")
        states = [parameter.requires_grad for parameter in discriminator.parameters()]
        try:
            for parameter in discriminator.parameters():
                parameter.requires_grad_(False)
            yield discriminator
        finally:
            for parameter, requires_grad in zip(discriminator.parameters(), states):
                parameter.requires_grad_(requires_grad)


class AdversarialOptimizers(NamedTuple):
    """Named, unpackable pair of encoder and discriminator optimizers."""

    encoder: torch.optim.Optimizer
    discriminator: torch.optim.Optimizer


def build_adversarial_optimizers(
    encoder_parameters: Iterable[Tensor] | nn.Module,
    discriminator_parameters: Iterable[Tensor] | nn.Module,
    *,
    encoder_lr: float | None = None,
    lr: float | None = None,
    discriminator_lr_ratio: float = 1.5,
    discriminator_lr: float | None = None,
    optimizer_cls: Callable[..., torch.optim.Optimizer] = torch.optim.Adam,
    optimizer_kwargs: Mapping[str, Any] | None = None,
) -> AdversarialOptimizers:
    """Build two optimizers with disjoint parameters and a fixed LR ratio."""

    if encoder_lr is not None and lr is not None and float(encoder_lr) != float(lr):
        raise ValueError("encoder_lr and lr must agree when both are supplied")
    resolved_lr = _finite_positive(
        1e-3
        if encoder_lr is None and lr is None
        else (encoder_lr if encoder_lr is not None else lr),
        "encoder_lr",
    )
    resolved_ratio = _finite_positive(
        discriminator_lr_ratio, "discriminator_lr_ratio"
    )
    encoder = _parameter_list(encoder_parameters, "encoder_parameters")
    discriminator = _parameter_list(
        discriminator_parameters, "discriminator_parameters"
    )
    encoder_ids = {id(parameter) for parameter in encoder}
    discriminator_ids = {id(parameter) for parameter in discriminator}
    if not encoder_ids.isdisjoint(discriminator_ids):
        raise ValueError("encoder and discriminator parameters must be disjoint")

    expected_discriminator_lr = resolved_lr * resolved_ratio
    if discriminator_lr is not None:
        resolved_discriminator_lr = _finite_positive(
            discriminator_lr, "discriminator_lr"
        )
        if not math.isclose(
            resolved_discriminator_lr,
            expected_discriminator_lr,
            rel_tol=1e-6,
            abs_tol=1e-12,
        ):
            raise ValueError("discriminator_lr must preserve the configured LR ratio")
    else:
        resolved_discriminator_lr = expected_discriminator_lr

    if not callable(optimizer_cls):
        raise ValueError("optimizer_cls must be callable")
    kwargs = {} if optimizer_kwargs is None else dict(optimizer_kwargs)
    if "params" in kwargs or "lr" in kwargs:
        raise ValueError("optimizer_kwargs must not override params or lr")
    encoder_optimizer = optimizer_cls(encoder, lr=resolved_lr, **kwargs)
    discriminator_optimizer = optimizer_cls(
        discriminator, lr=resolved_discriminator_lr, **kwargs
    )
    return AdversarialOptimizers(encoder_optimizer, discriminator_optimizer)


def _flatten_gradient(value: Tensor | Iterable[Tensor], name: str) -> Tensor:
    if torch.is_tensor(value):
        if value.numel() == 0 or not value.is_floating_point():
            raise ValueError(f"{name} must be a non-empty floating-point tensor")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")
        return value.reshape(-1)
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a tensor or tensor iterable")
    try:
        values = list(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a tensor or tensor iterable") from exc
    if not values:
        raise ValueError(f"{name} must be non-empty")
    flattened = [
        _flatten_gradient(item, f"{name}[{index}]")
        for index, item in enumerate(values)
    ]
    device = flattened[0].device
    if any(item.device != device for item in flattened):
        raise ValueError(f"{name} tensors must use the same device")
    return torch.cat(flattened)


class DualRatioController:
    """Maintain separate detached gradient-dose controllers for two adversaries."""

    def __init__(
        self,
        *,
        conditional_target_ratio: float = 0.15,
        zdom_tx_target_ratio: float = 0.055,
        conditional_ratio_bounds: tuple[float, float] = _DEFAULT_CONDITIONAL_BOUNDS,
        zdom_tx_ratio_bounds: tuple[float, float] = _DEFAULT_ZDOM_TX_BOUNDS,
        ema_decay: float = 0.9,
        min_scale: float = 0.0,
        max_scale: float = 10.0,
        eps: float = 1e-8,
    ) -> None:
        self.conditional_ratio_bounds = _ratio_bounds(
            conditional_ratio_bounds, "conditional_ratio_bounds"
        )
        self.zdom_tx_ratio_bounds = _ratio_bounds(
            zdom_tx_ratio_bounds, "zdom_tx_ratio_bounds"
        )
        conditional_target = _finite_positive(
            conditional_target_ratio, "conditional_target_ratio", allow_zero=True
        )
        zdom_target = _finite_positive(
            zdom_tx_target_ratio, "zdom_tx_target_ratio", allow_zero=True
        )
        if not self.conditional_ratio_bounds[0] <= conditional_target <= self.conditional_ratio_bounds[1]:
            raise ValueError("conditional_target_ratio must lie within its ratio bounds")
        if not self.zdom_tx_ratio_bounds[0] <= zdom_target <= self.zdom_tx_ratio_bounds[1]:
            raise ValueError("zdom_tx_target_ratio must lie within its ratio bounds")
        self.conditional = GradientRatioController(
            target_ratio=conditional_target,
            ema_decay=ema_decay,
            min_scale=min_scale,
            max_scale=max_scale,
            eps=eps,
        )
        self.zdom_tx = GradientRatioController(
            target_ratio=zdom_target,
            ema_decay=ema_decay,
            min_scale=min_scale,
            max_scale=max_scale,
            eps=eps,
        )
        self.conditional_controller = self.conditional
        self.zdom_tx_controller = self.zdom_tx

    @property
    def conditional_ratio(self) -> float | None:
        return self.conditional.ratio

    @property
    def zdom_tx_ratio(self) -> float | None:
        return self.zdom_tx.ratio

    @property
    def ratios(self) -> dict[str, float | None]:
        return {
            "conditional": self.conditional_ratio,
            "zdom_tx": self.zdom_tx_ratio,
        }

    def update(
        self,
        reference_gradient: Tensor | Iterable[Tensor],
        conditional_gradient: Tensor | Iterable[Tensor] | None = None,
        zdom_tx_gradient: Tensor | Iterable[Tensor] | None = None,
    ) -> dict[str, float]:
        """Update either or both independent dose controllers.

        Missing controlled gradients leave that controller unchanged, which
        lets callers log or update the two adversarial ratios independently.
        """

        if conditional_gradient is None and zdom_tx_gradient is None:
            raise ValueError("at least one controlled gradient is required")
        reference = _flatten_gradient(reference_gradient, "reference_gradient")
        result: dict[str, float] = {}
        if conditional_gradient is not None:
            conditional = _flatten_gradient(
                conditional_gradient, "conditional_gradient"
            )
            if conditional.device != reference.device:
                raise ValueError("all gradient tensors must use the same device")
            result["conditional"] = self.conditional.update(reference, conditional)
        elif self.conditional.last_scale is not None:
            result["conditional"] = self.conditional.last_scale

        if zdom_tx_gradient is not None:
            zdom_tx = _flatten_gradient(zdom_tx_gradient, "zdom_tx_gradient")
            if zdom_tx.device != reference.device:
                raise ValueError("all gradient tensors must use the same device")
            result["zdom_tx"] = self.zdom_tx.update(reference, zdom_tx)
        elif self.zdom_tx.last_scale is not None:
            result["zdom_tx"] = self.zdom_tx.last_scale
        return result


class DynamicGRLDoseController:
    """Generate independent bounded GRL doses from current training signals.

    The controller is deliberately state-only: it consumes detached scalar
    feedback and returns Python floats, so the dose policy cannot become part
    of the encoder gradient graph.  ``conditional_ratio_bounds`` and
    ``zdom_tx_ratio_bounds`` preserve the existing adversarial gradient-dose
    targets while the returned GRL doses remain independently bounded.
    """

    def __init__(
        self,
        *,
        initial_identity_dose: float = 0.20,
        initial_zdom_dose: float = 0.20,
        identity_dose_bounds: tuple[float, float] = _DEFAULT_IDENTITY_DOSE_BOUNDS,
        zdom_dose_bounds: tuple[float, float] = _DEFAULT_ZDOM_DOSE_BOUNDS,
        conditional_target_ratio: float = 0.15,
        zdom_tx_target_ratio: float = 0.055,
        conditional_ratio_bounds: tuple[float, float] = _DEFAULT_CONDITIONAL_BOUNDS,
        zdom_tx_ratio_bounds: tuple[float, float] = _DEFAULT_ZDOM_TX_BOUNDS,
        ema_decay: float = 0.5,
    ) -> None:
        self.identity_dose_bounds = _ratio_bounds(
            identity_dose_bounds, "identity_dose_bounds"
        )
        self.zdom_dose_bounds = _ratio_bounds(
            zdom_dose_bounds, "zdom_dose_bounds"
        )
        self.conditional_ratio_bounds = _ratio_bounds(
            conditional_ratio_bounds, "conditional_ratio_bounds"
        )
        self.zdom_tx_ratio_bounds = _ratio_bounds(
            zdom_tx_ratio_bounds, "zdom_tx_ratio_bounds"
        )
        self.conditional_target_ratio = _finite_positive(
            conditional_target_ratio,
            "conditional_target_ratio",
            allow_zero=True,
        )
        self.zdom_tx_target_ratio = _finite_positive(
            zdom_tx_target_ratio,
            "zdom_tx_target_ratio",
            allow_zero=True,
        )
        if not (
            self.conditional_ratio_bounds[0]
            <= self.conditional_target_ratio
            <= self.conditional_ratio_bounds[1]
        ):
            raise ValueError("conditional_target_ratio must lie within its ratio bounds")
        if not (
            self.zdom_tx_ratio_bounds[0]
            <= self.zdom_tx_target_ratio
            <= self.zdom_tx_ratio_bounds[1]
        ):
            raise ValueError("zdom_tx_target_ratio must lie within its ratio bounds")
        self.initial_identity_dose = _finite_positive(
            initial_identity_dose,
            "initial_identity_dose",
            allow_zero=True,
        )
        self.initial_zdom_dose = _finite_positive(
            initial_zdom_dose,
            "initial_zdom_dose",
            allow_zero=True,
        )
        if not self.identity_dose_bounds[0] <= self.initial_identity_dose <= self.identity_dose_bounds[1]:
            raise ValueError("initial_identity_dose must lie within its dose bounds")
        if not self.zdom_dose_bounds[0] <= self.initial_zdom_dose <= self.zdom_dose_bounds[1]:
            raise ValueError("initial_zdom_dose must lie within its dose bounds")
        self.ema_decay = _finite_positive(ema_decay, "ema_decay", allow_zero=True)
        if self.ema_decay >= 1.0:
            raise ValueError("ema_decay must be in [0,1)")
        self._doses = {
            "identity": self.initial_identity_dose,
            "zdom": self.initial_zdom_dose,
        }
        self._feedback: dict[str, float] | None = None
        self._last_update: dict[str, Any] | None = None

    @property
    def doses(self) -> dict[str, float]:
        return dict(self._doses)

    @property
    def last_feedback(self) -> dict[str, float] | None:
        return None if self._feedback is None else dict(self._feedback)

    @property
    def last_update(self) -> dict[str, Any] | None:
        return None if self._last_update is None else dict(self._last_update)

    @staticmethod
    def _drive(value: float) -> float:
        return max(-1.0, min(1.0, value))

    def update(
        self,
        *,
        discriminator_accuracy: float,
        tx_margin: float,
        adversarial_gradient_ratio: float,
        conflict_signal: float,
    ) -> dict[str, float]:
        """Consume four detached feedback signals and return two GRL doses."""

        raw_feedback = {
            "discriminator_accuracy": _unit_interval(
                discriminator_accuracy, "discriminator_accuracy"
            ),
            "tx_margin": _finite_positive(tx_margin, "tx_margin", allow_zero=True)
            if float(tx_margin) >= 0.0
            else -_finite_positive(-float(tx_margin), "tx_margin", allow_zero=True),
            "adversarial_gradient_ratio": _finite_positive(
                adversarial_gradient_ratio,
                "adversarial_gradient_ratio",
                allow_zero=True,
            ),
            "conflict_signal": _unit_interval(conflict_signal, "conflict_signal"),
        }
        if self._feedback is None:
            feedback = dict(raw_feedback)
        else:
            feedback = {
                name: self.ema_decay * self._feedback[name]
                + (1.0 - self.ema_decay) * value
                for name, value in raw_feedback.items()
            }
        if not all(math.isfinite(value) for value in feedback.values()):
            raise ValueError("dynamic GRL feedback must remain finite")

        accuracy_drive = self._drive(2.0 * feedback["discriminator_accuracy"] - 1.0)
        margin_drive = self._drive(math.tanh(feedback["tx_margin"]))
        identity_ratio_drive = self._drive(
            1.0
            - feedback["adversarial_gradient_ratio"]
            / max(self.conditional_target_ratio, 1.0e-12)
        )
        zdom_ratio_drive = self._drive(
            1.0
            - feedback["adversarial_gradient_ratio"]
            / max(self.zdom_tx_target_ratio, 1.0e-12)
        )
        conflict_drive = self._drive(1.0 - 2.0 * feedback["conflict_signal"])
        identity_drive = self._drive(
            0.30 * accuracy_drive
            + 0.25 * margin_drive
            + 0.25 * identity_ratio_drive
            + 0.20 * conflict_drive
        )
        zdom_drive = self._drive(
            0.40 * accuracy_drive
            + 0.10 * margin_drive
            + 0.30 * zdom_ratio_drive
            + 0.20 * conflict_drive
        )
        identity = min(
            self.identity_dose_bounds[1],
            max(
                self.identity_dose_bounds[0],
                self.initial_identity_dose * (1.0 + 0.35 * identity_drive),
            ),
        )
        zdom = min(
            self.zdom_dose_bounds[1],
            max(
                self.zdom_dose_bounds[0],
                self.initial_zdom_dose * (1.0 + 0.35 * zdom_drive),
            ),
        )
        if not math.isfinite(identity) or not math.isfinite(zdom):
            raise ValueError("dynamic GRL doses must remain finite")
        self._feedback = {name: float(value) for name, value in feedback.items()}
        self._doses = {"identity": float(identity), "zdom": float(zdom)}
        self._last_update = {
            "feedback": dict(self._feedback),
            "doses": self.doses,
            "drives": {
                "identity": float(identity_drive),
                "zdom": float(zdom_drive),
            },
        }
        return self.doses


DynamicGRLController = DynamicGRLDoseController


__all__ = [
    "AdversarialGamePlan",
    "AdversarialOptimizers",
    "DynamicGRLController",
    "DynamicGRLDoseController",
    "DualRatioController",
    "build_adversarial_optimizers",
]
