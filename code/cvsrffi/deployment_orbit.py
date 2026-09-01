from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence

import torch


DAOT_NUISANCE_TANGENT_NAMES = (
    "doppler",
    "doppler_rate",
    "sfo",
    "sto",
    "multipath",
    "snr",
    "phase_noise",
    "agc",
)


@dataclass(frozen=True)
class TeacherViewSpec:
    name: str
    scenario: str
    severity: float


@dataclass(frozen=True)
class DeploymentOrbitConfig:
    """Physics-informed Phase1 deployment proxy.

    ``empirical_weight`` is deliberately unavailable unless a separately
    verified real-LEO parameter asset exists. This prevents a simulated
    source-only run from silently acquiring a real-deployment claim.
    """

    physics_weight: float = 0.75
    empirical_weight: float = 0.0
    tail_weight: float = 0.25
    has_real_leo_statistics: bool = False
    importance_clip: tuple[float, float] = (0.25, 4.0)
    severity_prior: tuple[float, ...] = (0.15, 0.25, 0.35, 0.25)

    def __post_init__(self) -> None:
        weights = (self.physics_weight, self.empirical_weight, self.tail_weight)
        if any(float(value) < 0.0 for value in weights):
            raise ValueError("deployment mixture weights must be non-negative")
        if float(sum(weights)) <= 0.0:
            raise ValueError("deployment mixture weights must have positive mass")
        if float(self.empirical_weight) > 0.0 and not bool(self.has_real_leo_statistics):
            raise ValueError("empirical deployment weight requires verified real LEO statistics")
        lo, hi = (float(value) for value in self.importance_clip)
        if not (0.0 < lo <= hi):
            raise ValueError("importance_clip must satisfy 0 < low <= high")
        if not self.severity_prior or any(float(value) < 0.0 for value in self.severity_prior):
            raise ValueError("severity_prior must contain non-negative mass")
        if float(sum(self.severity_prior)) <= 0.0:
            raise ValueError("severity_prior must contain positive mass")

    @property
    def claim_label(self) -> str:
        return "deployment-matched" if self.has_real_leo_statistics else "deployment-proxy matched"


def default_teacher_view_specs() -> tuple[TeacherViewSpec, ...]:
    """Performance-first teacher orbit requested for DAOT-STN-V1."""

    return (
        TeacherViewSpec("clean", "clean", 0.0),
        TeacherViewSpec("medium", "leo_clear_weak", 0.5),
        TeacherViewSpec("hard", "leo_low_elev_weak", 1.0),
    )


def daot_ablation_overrides(ablation_id: str) -> dict[str, Any]:
    """Return the report's A0-A8 matrix as executable CLI overrides."""

    key = str(ablation_id or "").upper().strip()
    if key not in {f"A{index}" for index in range(9)}:
        raise ValueError("DAOT ablation must be one of A0 through A8")
    config: dict[str, Any] = {
        "daot_ablation": key,
        "use_adv3b02_daot_stn": key != "A0",
        "daot_teacher_mode": "three_view",
        "daot_teacher_view_count": 3,
        "daot_aggregation": "robust_deployment",
        "daot_tangent_mode": "off",
        "daot_lambda_tangent": 0.0,
        "daot_lambda_nuisance": 0.0,
        "daot_lambda_fingerprint": 0.0,
    }
    if key == "A0":
        config["daot_teacher_view_count"] = 0
    elif key == "A1":
        config.update(daot_teacher_view_count=2, daot_aggregation="mean")
    elif key == "A2":
        config.update(daot_aggregation="mean")
    elif key == "A3":
        pass
    elif key == "A4":
        config.update(daot_tangent_mode="single_parameter", daot_lambda_tangent=0.05)
    elif key == "A5":
        config.update(daot_tangent_mode="covariance", daot_lambda_tangent=0.05)
    elif key == "A6":
        config.update(
            daot_tangent_mode="branch_selective",
            daot_lambda_tangent=0.05,
            daot_lambda_nuisance=0.10,
        )
    elif key == "A7":
        config.update(
            daot_tangent_mode="branch_selective",
            daot_lambda_tangent=0.05,
            daot_lambda_nuisance=0.10,
            daot_lambda_fingerprint=0.10,
        )
    elif key == "A8":
        config.update(
            daot_teacher_mode="temporal_memory",
            daot_teacher_view_count=2,
            daot_tangent_mode="branch_selective",
            daot_lambda_tangent=0.05,
            daot_lambda_nuisance=0.10,
            daot_lambda_fingerprint=0.10,
        )
    return config


def daot_loss_ablation_overrides(ablation_id: str) -> dict[str, Any]:
    """Independent loss-axis ablations requested after the A0-A8 mechanism matrix."""

    key = str(ablation_id or "none").lower().strip()
    overrides = {
        "none": {},
        "no_z": {"daot_lambda_orbit_z": 0.0},
        "no_logit": {"daot_lambda_orbit_logit": 0.0},
        "no_proto": {"daot_lambda_orbit_proto": 0.0},
        "relation_on": {"daot_enable_relation": True},
    }
    if key not in overrides:
        raise ValueError("DAOT loss ablation must be none, no_z, no_logit, no_proto, or relation_on")
    return dict(overrides[key])


def clipped_importance_ratio(
    deployment_probability: torch.Tensor,
    training_probability: torch.Tensor,
    *,
    clip: Sequence[float] = (0.25, 4.0),
) -> torch.Tensor:
    lo, hi = float(clip[0]), float(clip[1])
    if not (0.0 < lo <= hi):
        raise ValueError("importance clip must satisfy 0 < low <= high")
    ratio = deployment_probability.float() / training_probability.float().clamp_min(1e-12)
    return ratio.clamp(min=lo, max=hi)


def _meta_column(meta: Mapping[str, Any], name: str, batch_size: int, device: torch.device) -> torch.Tensor:
    value = meta.get(name)
    try:
        column = torch.as_tensor(value, device=device, dtype=torch.float32).reshape(-1)
    except (TypeError, ValueError, RuntimeError):
        return torch.zeros(int(batch_size), device=device, dtype=torch.float32)
    if column.numel() == 1 and int(batch_size) > 1:
        column = column.expand(int(batch_size))
    if column.numel() != int(batch_size):
        return torch.zeros(int(batch_size), device=device, dtype=torch.float32)
    return torch.nan_to_num(column, nan=0.0, posinf=0.0, neginf=0.0)


def physical_reliability_from_meta(
    meta: Mapping[str, Any],
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Recoverability score derived only from physical channel metadata."""

    snr = torch.sigmoid((_meta_column(meta, "snr_db", batch_size, device) - 14.0) / 5.0)
    elevation = (_meta_column(meta, "theta_deg", batch_size, device) / 90.0).clamp(0.0, 1.0)
    rician = torch.sigmoid((_meta_column(meta, "K_db", batch_size, device) - 6.0) / 5.0)
    reliability = (snr * elevation.clamp_min(0.05) * rician).pow(1.0 / 3.0)
    return reliability.clamp(0.0, 1.0)


def teacher_importance_matrix(*, batch_size: int, device: torch.device) -> torch.Tensor:
    """Clipped p_dep/q_train ratios for clean, medium, and hard proxy views."""

    deployment = torch.tensor([0.15, 0.35, 0.25], device=device, dtype=torch.float32)
    deployment = deployment / deployment.sum()
    training = torch.full_like(deployment, 1.0 / 3.0)
    ratio = clipped_importance_ratio(deployment, training)
    return ratio.unsqueeze(0).expand(int(batch_size), -1)


def stable_orbit_key_tensor(values: Sequence[Any], *, device: torch.device) -> torch.Tensor:
    keys = []
    for value in values:
        digest = hashlib.sha256(str(value).encode("utf-8")).digest()
        keys.append(int.from_bytes(digest[:8], byteorder="big", signed=False) & ((1 << 63) - 1))
    return torch.tensor(keys, device=device, dtype=torch.long)


def _fractional_delay_iq(x: torch.Tensor, delay_samples: torch.Tensor) -> torch.Tensor:
    batch, channels, length = x.shape
    positions = torch.arange(length, device=x.device, dtype=x.dtype).unsqueeze(0) - delay_samples.unsqueeze(1)
    lower = positions.floor()
    upper = lower + 1.0
    fraction = (positions - lower).unsqueeze(1)
    lower_index = lower.clamp(0, length - 1).long().unsqueeze(1).expand(batch, channels, length)
    upper_index = upper.clamp(0, length - 1).long().unsqueeze(1).expand(batch, channels, length)
    lower_value = torch.gather(x, 2, lower_index)
    upper_value = torch.gather(x, 2, upper_index)
    valid = ((positions >= 0.0) & (positions <= float(length - 1))).unsqueeze(1)
    return torch.where(valid, lower_value * (1.0 - fraction) + upper_value * fraction, torch.zeros_like(x))


def _sample_iq_at_positions(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    batch, channels, length = x.shape
    if positions.shape != (batch, length):
        raise ValueError("positions must have shape [batch,time]")
    lower = positions.floor()
    upper = lower + 1.0
    fraction = (positions - lower).unsqueeze(1)
    lower_index = lower.clamp(0, length - 1).long().unsqueeze(1).expand(batch, channels, length)
    upper_index = upper.clamp(0, length - 1).long().unsqueeze(1).expand(batch, channels, length)
    lower_value = torch.gather(x, 2, lower_index)
    upper_value = torch.gather(x, 2, upper_index)
    valid = ((positions >= 0.0) & (positions <= float(length - 1))).unsqueeze(1)
    return torch.where(valid, lower_value * (1.0 - fraction) + upper_value * fraction, torch.zeros_like(x))


def apply_named_local_nuisance_tangent(
    received_iq: torch.Tensor,
    *,
    name: str,
    direction: torch.Tensor,
    delta: float,
    sample_rate_hz: float,
) -> torch.Tensor:
    """Deterministic one-parameter audit view around a fixed received-IQ point.

    No random fading or noise is sampled. The SNR coordinate uses a fixed,
    signal-derived residual so the base and perturbed views share common
    randomness while still exposing an amplitude-to-residual sensitivity.
    """

    if received_iq.ndim != 3 or int(received_iq.shape[1]) != 2:
        raise ValueError("received_iq must have shape [batch,2,time]")
    key = str(name).lower().strip()
    if key not in DAOT_NUISANCE_TANGENT_NAMES:
        raise ValueError(f"unknown nuisance tangent: {name}")
    delta = float(delta)
    if delta <= 0.0:
        raise ValueError("delta must be positive")
    x = received_iq.float()
    batch, _, length = x.shape
    direction = torch.as_tensor(direction, device=x.device, dtype=x.dtype).reshape(-1)
    if direction.numel() == 1 and batch > 1:
        direction = direction.expand(batch)
    if direction.numel() != batch:
        raise ValueError("direction must provide one value per sample")
    sample_rate_hz = float(sample_rate_hz)
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive")
    sample_index = torch.arange(length, device=x.device, dtype=x.dtype)
    time = sample_index / sample_rate_hz

    if key == "sto":
        return _fractional_delay_iq(x, delta * 0.50 * direction).to(dtype=received_iq.dtype)
    if key == "sfo":
        centered = sample_index - 0.5 * float(length - 1)
        positions = sample_index.unsqueeze(0) - delta * 1e-3 * direction.unsqueeze(1) * centered.unsqueeze(0)
        return _sample_iq_at_positions(x, positions).to(dtype=received_iq.dtype)

    complex_x = torch.complex(x[:, 0], x[:, 1])
    if key == "doppler":
        phase = 2.0 * torch.pi * (delta * 50_000.0 * direction).unsqueeze(1) * time.unsqueeze(0)
        result = complex_x * torch.exp(1j * phase)
    elif key == "doppler_rate":
        phase = torch.pi * (delta * 5e9 * direction).unsqueeze(1) * time.square().unsqueeze(0)
        result = complex_x * torch.exp(1j * phase)
    elif key == "multipath":
        echo = torch.roll(complex_x, shifts=3, dims=1)
        echo[:, :3] = 0.0
        result = complex_x + (delta * 0.35 * direction).unsqueeze(1) * echo
    elif key == "snr":
        residual = torch.roll(complex_x, shifts=7, dims=1) - complex_x
        residual_rms = residual.abs().square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
        signal_rms = complex_x.abs().square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
        fixed_residual = residual * (signal_rms / residual_rms)
        result = complex_x + (delta * 0.15 * direction).unsqueeze(1) * fixed_residual
    elif key == "phase_noise":
        profile = torch.sin(2.0 * torch.pi * sample_index / max(2.0, float(length - 1)))
        phase = (delta * 0.20 * direction).unsqueeze(1) * profile.unsqueeze(0)
        result = complex_x * torch.exp(1j * phase)
    elif key == "agc":
        result = complex_x * torch.exp(delta * 0.20 * direction).unsqueeze(1)
    else:  # pragma: no cover - exhaustive guard above
        raise AssertionError(key)
    return torch.stack([result.real, result.imag], dim=1).to(dtype=received_iq.dtype)


def apply_local_nuisance_tangent(
    received_iq: torch.Tensor,
    directions: torch.Tensor,
    *,
    delta: float,
    sample_rate_hz: float,
) -> torch.Tensor:
    """Apply a deterministic local nuisance step to one received-IQ base point.

    Coordinates are residual CFO, fractional delay, RX-AGC and RX-filter. No
    new noise or fading is sampled, which supplies the required common-random
    comparison for the finite difference.
    """

    if received_iq.ndim != 3 or int(received_iq.shape[1]) != 2:
        raise ValueError("received_iq must have shape [batch,2,time]")
    if directions.shape != (received_iq.shape[0], 4):
        raise ValueError("directions must have shape [batch,4]")
    delta = float(delta)
    if delta <= 0.0:
        raise ValueError("delta must be positive")
    x = received_iq.float()
    d = directions.to(device=x.device, dtype=x.dtype)
    delayed = _fractional_delay_iq(x, delta * 0.50 * d[:, 1])
    complex_x = torch.complex(delayed[:, 0], delayed[:, 1])
    time = torch.arange(x.shape[-1], device=x.device, dtype=x.dtype) / float(sample_rate_hz)
    residual_cfo_hz = delta * 50_000.0 * d[:, 0]
    complex_x = complex_x * torch.exp(1j * (2.0 * torch.pi * residual_cfo_hz.unsqueeze(1) * time.unsqueeze(0)))
    complex_x = complex_x * torch.exp(delta * 0.20 * d[:, 2]).unsqueeze(1)
    filtered = torch.stack([complex_x.real, complex_x.imag], dim=1)
    alpha = (delta * 0.50 * d[:, 3].abs()).clamp(0.0, 0.25).view(-1, 1, 1)
    previous = torch.roll(filtered, shifts=1, dims=2)
    previous[:, :, 0] = filtered[:, :, 0]
    return (1.0 - alpha) * filtered + alpha * previous


def apply_fingerprint_intervention(
    clean_iq: torch.Tensor,
    *,
    strength: float,
    sample_rate_hz: float,
) -> torch.Tensor:
    """Mild deterministic PA/IQ/clock intervention for fingerprint retention."""

    if clean_iq.ndim != 3 or int(clean_iq.shape[1]) != 2:
        raise ValueError("clean_iq must have shape [batch,2,time]")
    strength = float(strength)
    if strength <= 0.0:
        raise ValueError("fingerprint intervention strength must be positive")
    x = torch.complex(clean_iq[:, 0].float(), clean_iq[:, 1].float())
    power = x.abs().square()
    pa = x * (1.0 + strength * power.clamp_max(4.0))
    iq = pa + (0.25 * strength) * torch.conj(pa)
    time = torch.arange(x.shape[-1], device=x.device, dtype=clean_iq.dtype) / float(sample_rate_hz)
    clock_offset_hz = 10_000.0 * strength
    shifted = iq * torch.exp(1j * (2.0 * torch.pi * clock_offset_hz * time.unsqueeze(0)))
    return torch.stack([shifted.real, shifted.imag], dim=1).to(dtype=clean_iq.dtype)


def sample_sparse_joint_direction(
    covariance: torch.Tensor,
    *,
    seed: int,
    max_active: int = 3,
) -> torch.Tensor:
    """Sample one standardized covariance direction with bounded support.

    The Cholesky transform preserves correlations before the largest physical
    modes are retained. The result is dimensionless and unit norm.
    """

    covariance = torch.as_tensor(covariance, dtype=torch.float32)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be a square matrix")
    dim = int(covariance.shape[0])
    if dim < 1 or int(max_active) < 1:
        raise ValueError("covariance and max_active must be non-empty")
    generator = torch.Generator(device=covariance.device)
    generator.manual_seed(int(seed))
    eps = torch.randn(dim, generator=generator, device=covariance.device, dtype=covariance.dtype)
    eye = torch.eye(dim, device=covariance.device, dtype=covariance.dtype)
    transform = torch.linalg.cholesky(covariance + 1e-6 * eye)
    direction = transform @ eps
    active = min(int(max_active), dim)
    indices = direction.abs().topk(active).indices
    sparse = torch.zeros_like(direction)
    sparse[indices] = direction[indices]
    norm = sparse.norm().clamp_min(1e-12)
    return sparse / norm
