from __future__ import annotations

from dataclasses import dataclass, field
import math

from .phase1_fcr_types import FCRV2CapabilityState


BASE_LOSS_WEIGHTS: dict[str, float] = {
    "identity_ce": 1.00,
    "prototype": 0.10,
    "tail": 0.075,
    "self": 0.10,
    "shared_f": 0.20,
    "shared_s": 0.05,
    "response": 0.05,
    "eta": 0.10,
    "swap": 0.05,
    "cycle": 0.05,
    "need": 0.05,
    "transplant": 0.05,
    "physical": 0.05,
    "factor": 0.05,
}
UNLABELED_FCR_WEIGHT = 0.35
_MECHANISM_NAMES = (
    "self",
    "shared_f",
    "shared_s",
    "response",
    "eta",
    "swap",
    "cycle",
    "need",
    "transplant",
    "physical",
    "factor",
)
_ROW_LOSS_REGISTRY: dict[str, frozenset[str]] = {
    "C1": frozenset(),
    "C2": frozenset(),
    "C3": frozenset({"self"}),
    # S0 is the structural shared-branch control.  It exercises the same V2
    # identity route as the shared rows while keeping every auxiliary loss off.
    "S0": frozenset(),
    "S1": frozenset({"self", "shared_f"}),
    "S2": frozenset({"self", "shared_s"}),
    "S3": frozenset({"self", "shared_f", "shared_s"}),
    "S4": frozenset({"self", "shared_f", "shared_s", "swap"}),
    "M1": frozenset({"self", "shared_f", "shared_s", "response", "eta"}),
    "M2": frozenset({"self", "shared_f", "shared_s", "response", "eta", "cycle"}),
    "M3": frozenset({"self", "shared_f", "shared_s", "response", "eta", "cycle", "need"}),
    "M4": frozenset({"self", "shared_f", "shared_s", "response", "eta", "cycle", "need", "transplant"}),
    "M5": frozenset(
        {"self", "shared_f", "shared_s", "response", "eta", "cycle", "need", "transplant", "physical"}
    ),
    "M6": frozenset(
        {"self", "shared_f", "shared_s", "response", "eta", "cycle", "need", "transplant", "physical", "factor"}
    ),
}


@dataclass(frozen=True)
class FCRStageState:
    name: str
    active_losses: frozenset[str]
    scales: dict[str, float]
    blocked: dict[str, str] = field(default_factory=dict)


def _mechanism_not_activated(reason: str | None) -> str:
    detail = str(reason or "unspecified")
    return f"MECHANISM_NOT_ACTIVATED:{detail}"


def _stage_scales(epoch: int) -> tuple[str, dict[str, float]]:
    epoch = int(epoch)
    if epoch < 1 or epoch > 200:
        raise ValueError("FCR-V2 epoch must be in 1..200")

    scales = {name: 0.0 for name in _MECHANISM_NAMES}
    if epoch <= 20:
        scales["self"] = 1.0
        scales["eta"] = 1.0
        return "E1_20_head_warmup", scales
    if epoch <= 60:
        scales["self"] = 1.0
        scales["shared_f"] = 1.0
        scales["shared_s"] = 1.0
        scales["eta"] = 1.0
        return "E21_60_shared_learning", scales
    if epoch <= 100:
        scales["self"] = 1.0
        scales["shared_f"] = 1.0
        scales["shared_s"] = 1.0
        scales["response"] = 1.0
        scales["eta"] = 1.0
        return "E61_100_nuisance_learning", scales
    if epoch <= 130:
        ramp = float(epoch - 101) / float(130 - 101)
        scales["self"] = 1.0
        scales["shared_f"] = 1.0
        scales["shared_s"] = 1.0
        scales["response"] = 1.0
        scales["eta"] = 1.0
        scales["swap"] = max(0.0, min(1.0, ramp))
        return "E101_130_true_swap", scales
    if epoch <= 160:
        scales["self"] = 1.0
        scales["shared_f"] = 1.0
        scales["shared_s"] = 1.0
        scales["response"] = 1.0
        scales["eta"] = 1.0
        scales["swap"] = 1.0
        scales["cycle"] = 1.0
        scales["need"] = 1.0
        scales["transplant"] = 1.0
        scales["physical"] = 1.0
        scales["factor"] = 1.0
        return "E131_160_cycle_need", scales

    scales["self"] = 0.25
    scales["shared_f"] = 0.25
    scales["shared_s"] = 0.25
    return "E161_200_identity_refinement", scales


def _loss_ready(loss_name: str, capabilities: FCRV2CapabilityState) -> tuple[bool, str | None]:
    if loss_name == "eta":
        return bool(capabilities.eta_ready), capabilities.reason_for("eta")
    if loss_name == "shared_f":
        return bool(capabilities.fingerprint_ready), capabilities.reason_for("fingerprint")
    if loss_name == "response":
        return bool(capabilities.decoder_ready), capabilities.reason_for("decoder")
    if loss_name == "swap":
        return bool(capabilities.swap_ready), capabilities.reason_for("swap")
    if loss_name == "cycle":
        ready = bool(capabilities.decoder_ready and capabilities.swap_ready)
        reason = capabilities.reason_for("decoder") or capabilities.reason_for("swap")
        return ready, reason
    if loss_name == "need":
        return bool(capabilities.fingerprint_ready), capabilities.reason_for("fingerprint")
    if loss_name == "transplant":
        ready = bool(capabilities.decoder_ready and capabilities.fingerprint_ready)
        reason = capabilities.reason_for("decoder") or capabilities.reason_for("fingerprint")
        return ready, reason
    if loss_name == "physical":
        return bool(capabilities.decoder_ready), capabilities.reason_for("decoder")
    if loss_name == "factor":
        return bool(capabilities.fingerprint_ready), capabilities.reason_for("fingerprint")
    return True, None


class FCRV2Schedule:
    def __init__(
        self,
        *,
        base_weights: dict[str, float] | None = None,
        unlabeled_fcr_weight: float = UNLABELED_FCR_WEIGHT,
    ) -> None:
        self.base_weights = dict(BASE_LOSS_WEIGHTS if base_weights is None else base_weights)
        self.unlabeled_fcr_weight = float(unlabeled_fcr_weight)
        for name, value in self.base_weights.items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"FCR-V2 weight {name} must be finite and >= 0")
        if not math.isfinite(self.unlabeled_fcr_weight) or self.unlabeled_fcr_weight < 0.0:
            raise ValueError("unlabeled_fcr_weight must be finite and >= 0")

    def row_losses(self, row: str) -> frozenset[str]:
        key = str(row).strip().upper()
        try:
            return _ROW_LOSS_REGISTRY[key]
        except KeyError as exc:
            raise ValueError(f"unknown FCR-V2 row: {row!r}") from exc

    def state(self, epoch: int, row: str, capabilities: FCRV2CapabilityState) -> FCRStageState:
        name, stage_scales = _stage_scales(epoch)
        row_losses = self.row_losses(row)
        active: set[str] = set()
        blocked: dict[str, str] = {}
        scales = {loss_name: 0.0 for loss_name in self.base_weights}

        for loss_name in row_losses:
            stage_scale = float(stage_scales.get(loss_name, 0.0))
            if stage_scale <= 0.0:
                continue
            ready, reason = _loss_ready(loss_name, capabilities)
            if not ready:
                blocked[loss_name] = _mechanism_not_activated(reason)
                continue
            scales[loss_name] = stage_scale
            active.add(loss_name)

        return FCRStageState(
            name=name,
            active_losses=frozenset(active),
            scales=scales,
            blocked=blocked,
        )
