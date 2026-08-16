"""Frozen, causally adjacent training-arm configuration for Phase1 MIRAGE."""

from __future__ import annotations

from dataclasses import dataclass, field

from .model import MIRAGEConfig


_ARM_ORDER = ("B0", "A", "B", "C")
_BASE_MECHANISMS = frozenset({"supervised_ce", "ema_pseudo", "weak_strong_consistency"})
_ARM_MECHANISMS: dict[str, frozenset[str]] = {
    "B0": _BASE_MECHANISMS,
    "A": _BASE_MECHANISMS
    | frozenset({"masked_latent", "cross_receiver", "prototype_pseudo"}),
    "B": _BASE_MECHANISMS
    | frozenset(
        {
            "masked_latent",
            "cross_receiver",
            "prototype_pseudo",
            "proxy_open_loss",
            "radius_energy",
            "boundary_mixup",
        }
    ),
    "C": _BASE_MECHANISMS
    | frozenset(
        {
            "masked_latent",
            "cross_receiver",
            "prototype_pseudo",
            "proxy_open_loss",
            "radius_energy",
            "boundary_mixup",
            "group_cvar",
        }
    ),
}

_SHARED_EPOCHS = 200
_SHARED_OPTIMIZER = "adamw"
_SHARED_LEARNING_RATE = 3e-4
_SHARED_WEIGHT_DECAY = 1e-4
_SHARED_BATCH_SIZE = 128
_SHARED_STEPS_PER_EPOCH = 1_000


@dataclass(frozen=True)
class ArmConfig:
    """Immutable budget and declared mechanisms for one causal MIRAGE arm.

    Every field that could alter capacity or optimization budget is shared by
    all arms.  The only arm-specific field is the cumulative mechanism set.
    """

    arm_id: str
    encoder: MIRAGEConfig = field(default_factory=MIRAGEConfig)
    epochs: int = _SHARED_EPOCHS
    optimizer: str = _SHARED_OPTIMIZER
    learning_rate: float = _SHARED_LEARNING_RATE
    weight_decay: float = _SHARED_WEIGHT_DECAY
    batch_size: int = _SHARED_BATCH_SIZE
    steps_per_epoch: int = _SHARED_STEPS_PER_EPOCH
    mechanisms: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.arm_id not in _ARM_MECHANISMS:
            raise ValueError(f"arm_id must be one of: {', '.join(_ARM_ORDER)}")
        if not isinstance(self.encoder, MIRAGEConfig):
            raise TypeError("encoder must be a MIRAGEConfig")
        if self.encoder != MIRAGEConfig():
            raise ValueError("encoder must equal the frozen default MIRAGEConfig()")
        if (
            self.epochs != _SHARED_EPOCHS
            or self.optimizer != _SHARED_OPTIMIZER
            or self.learning_rate != _SHARED_LEARNING_RATE
            or self.weight_decay != _SHARED_WEIGHT_DECAY
            or self.batch_size != _SHARED_BATCH_SIZE
            or self.steps_per_epoch != _SHARED_STEPS_PER_EPOCH
        ):
            raise ValueError("all MIRAGE arms must use the frozen shared budget")
        if self.mechanisms != _ARM_MECHANISMS[self.arm_id]:
            raise ValueError("mechanisms must equal the frozen causal arm definition")


def arm_config(arm_id: str) -> ArmConfig:
    """Return one frozen causal arm without inheriting hidden historical state."""

    if not isinstance(arm_id, str):
        raise TypeError("arm_id must be a string")
    if arm_id not in _ARM_MECHANISMS:
        raise ValueError(f"arm_id must be one of: {', '.join(_ARM_ORDER)}")
    return ArmConfig(arm_id=arm_id, mechanisms=_ARM_MECHANISMS[arm_id])


def arm_diff(arm_id: str, previous_arm_id: str) -> frozenset[str]:
    """Return only an adjacent arm's newly enabled mechanisms.

    Restricting this comparison to adjacent arms prevents a report from
    casually attributing the full B0-to-C stack to one mechanism transition.
    """

    current = arm_config(arm_id)
    previous = arm_config(previous_arm_id)
    if _ARM_ORDER.index(current.arm_id) != _ARM_ORDER.index(previous.arm_id) + 1:
        raise ValueError("arm_diff requires adjacent causal arms in forward order")
    return current.mechanisms - previous.mechanisms


__all__ = ["ArmConfig", "arm_config", "arm_diff"]
