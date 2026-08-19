"""Stateless MUSE-SSDG training schedule primitives."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MUSEConfig:
    """Immutable defaults for the 200-epoch MUSE training schedule."""

    s2a_start: int = 17
    s2b_start: int = 41
    s3a_start: int = 69
    s3b_start: int = 161
    s3c_start: int = 181
    final_epoch: int = 200
    lambda_u_full: float = 0.60
    lambda_u_consolidate: float = 0.25
    p_sat_s2a_end: float = 0.25
    p_sat_full: float = 0.50
    grl_min: float = 0.02
    grl_max: float = 0.10


@dataclass(frozen=True)
class MUSEScheduleState:
    """The complete stateless schedule decision for one training epoch."""

    stage: str
    ema_decay: float
    lambda_u: float
    p_sat: float
    grl_lambda: float
    proto_momentum: float
    pseudo_enabled: bool
    candidate_enabled: bool
    freeze_statistics: bool


def _linear_ramp(start: float, end: float, epoch: int, start_epoch: int, end_epoch: int) -> float:
    """Return an endpoint-inclusive linear ramp clipped to its interval."""

    if end_epoch <= start_epoch:
        return float(end)
    if epoch <= start_epoch:
        return float(start)
    if epoch >= end_epoch:
        return float(end)
    progress = (epoch - start_epoch) / float(end_epoch - start_epoch)
    return float(start + (end - start) * progress)


def _probability(value: float) -> float:
    """Keep schedule probabilities inside the closed unit interval."""

    return float(max(0.0, min(1.0, float(value))))


def muse_schedule_for_epoch(epoch: int, config: MUSEConfig) -> MUSEScheduleState:
    """Build the immutable MUSE schedule state for ``epoch``.

    The schedule has six named stages: S1 (1-16), S2A (17-40), S2B
    (41-68), S3A (69-160), S3B (161-180), and final consolidation S3C
    (181-200). S2A and S2B ramps include both segment endpoints.
    """

    if not isinstance(epoch, int) or isinstance(epoch, bool):
        raise TypeError("epoch must be an int")
    if epoch < 1 or epoch > config.final_epoch:
        raise ValueError(f"epoch must be in [1, {config.final_epoch}], got {epoch}")

    # Keep the documented decimal segment endpoints stable despite binary
    # floating-point representation (for example, 0.60 / 3 == 0.199999...).
    lambda_u_s2a_end = round(float(config.lambda_u_full) / 3.0, 12)
    lambda_u_s2b_end = round(float(config.lambda_u_full) * 5.0 / 6.0, 12)

    if epoch < config.s2a_start:
        stage = "S1"
        ema_decay = 0.99
        lambda_u = 0.0
        p_sat = 0.0
        proto_momentum = 0.95
    elif epoch < config.s2b_start:
        stage = "S2A"
        ema_decay = 0.995
        lambda_u = _linear_ramp(
            0.0,
            lambda_u_s2a_end,
            epoch,
            config.s2a_start,
            config.s2b_start - 1,
        )
        p_sat = _linear_ramp(
            0.0,
            float(config.p_sat_s2a_end),
            epoch,
            config.s2a_start,
            config.s2b_start - 1,
        )
        proto_momentum = 0.95
    elif epoch < config.s3a_start:
        stage = "S2B"
        ema_decay = 0.995
        lambda_u = _linear_ramp(
            lambda_u_s2a_end,
            lambda_u_s2b_end,
            epoch,
            config.s2b_start,
            config.s3a_start - 1,
        )
        p_sat = float(config.p_sat_full)
        proto_momentum = 0.95
    elif epoch < config.s3b_start:
        stage = "S3A"
        ema_decay = 0.999
        lambda_u = float(config.lambda_u_full)
        p_sat = float(config.p_sat_full)
        proto_momentum = 0.95
    elif epoch < config.s3c_start:
        stage = "S3B"
        ema_decay = 0.999
        lambda_u = float(config.lambda_u_full)
        p_sat = float(config.p_sat_full)
        proto_momentum = 0.99
    else:
        stage = "S3C"
        ema_decay = 0.999
        lambda_u = float(config.lambda_u_consolidate)
        p_sat = float(config.p_sat_full)
        proto_momentum = 0.99

    grl_lambda = _linear_ramp(
        float(config.grl_min),
        float(config.grl_max),
        epoch,
        1,
        config.final_epoch,
    )
    return MUSEScheduleState(
        stage=stage,
        ema_decay=float(ema_decay),
        lambda_u=float(lambda_u),
        p_sat=_probability(p_sat),
        grl_lambda=_probability(grl_lambda),
        proto_momentum=float(proto_momentum),
        pseudo_enabled=stage != "S1",
        candidate_enabled=stage in {"S2B", "S3A", "S3B", "S3C"},
        freeze_statistics=stage == "S3C",
    )


__all__ = ["MUSEConfig", "MUSEScheduleState", "muse_schedule_for_epoch"]
