from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class FCRLambdaConfig:
    self_reconstruction: float = 1.0
    swap: float = 1.0
    shared: float = 1.0
    latent_cycle: float = 1.0
    eta: float = 1.0
    factor: float = 1.0
    transplant_necessity: float = 1.0
    physical_features: float = 1.0

    def __post_init__(self) -> None:
        for name, value in self.as_scales().items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"FCR lambda {name} must be finite and >= 0")

    def as_scales(self) -> dict[str, float]:
        return {
            "self": float(self.self_reconstruction),
            "swap": float(self.swap),
            "shared": float(self.shared),
            "latent_cycle": float(self.latent_cycle),
            "eta": float(self.eta),
            "factor": float(self.factor),
            "need": float(self.transplant_necessity),
            "phys": float(self.physical_features),
        }


@dataclass(frozen=True)
class FCRStageState:
    name: str
    active: frozenset[str]
    scales: dict[str, float]
    freeze_decoder_for_necessity: bool

    @property
    def reconstruction_scale(self) -> float:
        return float(self.scales.get("self", 0.0))


@dataclass(frozen=True)
class FCRRolePermission:
    role: str
    allowed: frozenset[str]
    optimizer_step: bool


def permission_for_role(role: str) -> FCRRolePermission:
    normalized = str(role or "").strip().lower()
    if normalized in {"l_s", "labeled", "source_labeled"}:
        return FCRRolePermission(
            role="L_s",
            allowed=frozenset(
                {"id", "self", "swap", "shared", "latent_cycle", "eta", "factor", "transplant", "phys"}
            ),
            optimizer_step=True,
        )
    if normalized in {"u_s", "unlabeled", "source_unlabeled"}:
        return FCRRolePermission(
            role="U_s",
            allowed=frozenset({"self", "swap", "shared", "latent_cycle", "eta", "phys"}),
            optimizer_step=True,
        )
    if normalized in {"v", "validation", "source_validation"}:
        return FCRRolePermission(role="V", allowed=frozenset(), optimizer_step=False)
    if normalized == "query":
        raise ValueError("Phase2 query is unreachable from the Phase1 FCR path")
    raise ValueError(f"unknown FCR Phase1 role: {role!r}")


def _zero_scales() -> dict[str, float]:
    return {name: 0.0 for name in ("self", "swap", "shared", "latent_cycle", "eta", "factor", "need", "phys")}


def stage_for_epoch(
    epoch: int,
    *,
    optimizer_step: int = 0,
    configured: FCRLambdaConfig | None = None,
    total_epochs: int = 200,
) -> FCRStageState:
    """Return the frozen four-stage ADV3B02-FCR schedule.

    Epochs are one-based. E91-E150 alternates a full real-combination update
    on even optimizer steps and a decoder-frozen necessity-only update on odd
    optimizer steps.
    """

    if int(total_epochs) != 200:
        raise ValueError("ADV3B02-FCR schedule is defined for exactly 200 epochs")
    epoch = int(epoch)
    if epoch < 1 or epoch > 200:
        raise ValueError("ADV3B02-FCR epoch must be in 1..200")
    optimizer_step = int(optimizer_step)
    if optimizer_step < 0:
        raise ValueError("optimizer_step must be >= 0")
    weights = (configured or FCRLambdaConfig()).as_scales()
    scales = _zero_scales()

    if epoch <= 40:
        scales["self"] = weights["self"]
        scales["eta"] = weights["eta"]
        return FCRStageState(
            name="E1_40_reconstruction",
            active=frozenset({"id", "self", "eta"}),
            scales=scales,
            freeze_decoder_for_necessity=False,
        )

    if epoch <= 90:
        ramp = float(epoch - 41) / float(90 - 41)
        scales["self"] = weights["self"]
        scales["eta"] = weights["eta"]
        scales["swap"] = weights["swap"] * ramp
        scales["shared"] = weights["shared"] * ramp
        scales["latent_cycle"] = weights["latent_cycle"] * ramp
        return FCRStageState(
            name="E41_90_cross_ramp",
            active=frozenset({"id", "self", "eta", "swap", "shared", "latent_cycle"}),
            scales=scales,
            freeze_decoder_for_necessity=False,
        )

    if epoch <= 150 and optimizer_step % 2 == 1:
        scales["need"] = weights["need"]
        return FCRStageState(
            name="E91_150_intervention",
            active=frozenset({"transplant", "necessity"}),
            scales=scales,
            freeze_decoder_for_necessity=True,
        )

    if epoch <= 150:
        scales.update(weights)
        return FCRStageState(
            name="E91_150_intervention",
            active=frozenset(
                {"id", "self", "eta", "swap", "shared", "latent_cycle", "factor", "phys", "intervention", "transplant"}
            ),
            scales=scales,
            freeze_decoder_for_necessity=False,
        )

    scales.update(weights)
    scales["self"] *= 0.25
    scales["swap"] *= 0.25
    return FCRStageState(
        name="E151_200_identity_refine",
        active=frozenset(
            {"id", "self", "eta", "swap", "shared", "latent_cycle", "factor", "phys", "intervention", "transplant"}
        ),
        scales=scales,
        freeze_decoder_for_necessity=False,
    )
