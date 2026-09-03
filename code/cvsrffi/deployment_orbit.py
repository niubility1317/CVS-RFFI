from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterator, Mapping, Sequence

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

DAOT_RX_V2_EVAL_SCENARIOS = (
    "clean",
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)


@dataclass(frozen=True)
class TeacherViewSpec:
    name: str
    scenario: str
    severity: float


@dataclass(frozen=True)
class TangentDirectionSpec:
    name: str
    kind: str
    unit: str
    delta: float
    budget: float
    supports_tangent: bool = True
    null_identity: bool = True


class TangentDirectionRegistry(Mapping[str, TangentDirectionSpec]):
    """Single source of truth for nuisance, mixed, TX, and secant directions."""

    _KINDS = {"pure_nuisance", "mixed", "tx_fingerprint", "secant_only"}

    def __init__(self, specs: Sequence[TangentDirectionSpec]) -> None:
        self._specs = {spec.name: spec for spec in specs}
        if len(self._specs) != len(tuple(specs)):
            raise ValueError("tangent direction names must be unique")
        if any(spec.kind not in self._KINDS for spec in self._specs.values()):
            raise ValueError("unknown tangent direction kind")
        if any(spec.delta <= 0.0 or spec.budget < 0.0 for spec in self._specs.values()):
            raise ValueError("direction delta must be positive and budget non-negative")

    def __getitem__(self, key: str) -> TangentDirectionSpec:
        return self._specs[str(key)]

    def __iter__(self) -> Iterator[str]:
        return iter(self._specs)

    def __len__(self) -> int:
        return len(self._specs)

    @classmethod
    def default(cls) -> "TangentDirectionRegistry":
        return cls(
            (
                TangentDirectionSpec("doppler", "pure_nuisance", "Hz", 0.05, 0.0),
                TangentDirectionSpec("doppler_rate", "pure_nuisance", "Hz/s", 0.05, 0.0),
                TangentDirectionSpec("sto", "pure_nuisance", "sample", 0.05, 0.0),
                TangentDirectionSpec("rx_sfo", "pure_nuisance", "ppm", 0.05, 0.01),
                TangentDirectionSpec("rx_filter", "pure_nuisance", "standardized", 0.05, 0.0),
                TangentDirectionSpec("multipath", "pure_nuisance", "standardized", 0.05, 0.0),
                TangentDirectionSpec("snr", "pure_nuisance", "dB-standardized", 0.05, 0.01),
                TangentDirectionSpec("rx_phase_noise", "pure_nuisance", "rad-standardized", 0.05, 0.01),
                TangentDirectionSpec("agc", "pure_nuisance", "dB", 0.05, 0.0),
                TangentDirectionSpec("total_cfo", "mixed", "Hz", 0.05, 0.05),
                TangentDirectionSpec("total_iq_imbalance", "mixed", "standardized", 0.05, 0.05),
                TangentDirectionSpec("pa", "tx_fingerprint", "standardized", 0.03, 0.0, True, False),
                TangentDirectionSpec("iq_gain", "tx_fingerprint", "standardized", 0.03, 0.0, True, False),
                TangentDirectionSpec("iq_phase", "tx_fingerprint", "rad", 0.03, 0.0, True, False),
                TangentDirectionSpec("tx_cfo", "tx_fingerprint", "Hz", 0.03, 0.0, True, False),
                TangentDirectionSpec("clock_skew", "tx_fingerprint", "ppm", 0.03, 0.0, True, False),
                TangentDirectionSpec("dac_nonlinearity", "tx_fingerprint", "standardized", 0.03, 0.0, True, False),
                TangentDirectionSpec("clipping", "secant_only", "clip-ratio", 0.05, 0.0, False, False),
                TangentDirectionSpec("quantization", "secant_only", "bit-step", 0.05, 0.0, False, False),
            )
        )


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


def validate_daot_eval_scenarios(scenarios: Sequence[str]) -> tuple[str, ...]:
    """Enforce the RX-V2 clean plus LEO_WEAK evaluation boundary."""

    normalized = tuple(str(value).strip() for value in scenarios if str(value).strip())
    if normalized != DAOT_RX_V2_EVAL_SCENARIOS:
        raise ValueError(
            "ADV3B02-DAOT-STN-RX-V2 evaluation is limited to clean and the three LEO_WEAK scenarios"
        )
    return normalized


def daot_rx_v2_overrides() -> dict[str, Any]:
    """Deployment-default RX-V2 controls, separate from the A1-A8 upper-bound matrix."""

    return {
        "use_adv3b02_daot_stn": True,
        "daot_teacher_mode": "temporal_memory_rx",
        "daot_teacher_view_count": 2,
        "daot_lambda_orbit_z": 0.40,
        "daot_lambda_orbit_logit": 0.075,
        "daot_lambda_orbit_proto": 0.125,
        "daot_lambda_tangent": 0.035,
        "daot_lambda_nuisance": 0.0,
        "daot_lambda_fingerprint": 0.0,
        "daot_lambda_route": 0.05,
        "daot_lambda_rx": 0.075,
        "daot_lambda_tail": 0.10,
        "daot_lambda_clean_anchor": 0.025,
        "daot_lambda_subspace": 0.0,
        "daot_eval_scenarios": DAOT_RX_V2_EVAL_SCENARIOS,
        "use_tx_rx_balanced_sampler": True,
        "balanced_sampler_tx_per_batch": 6,
        "balanced_sampler_domain_per_batch": 5,
        "balanced_sampler_samples_per_cell": 3,
    }


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
    return_details: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """Recoverability score derived only from physical channel metadata."""

    names = (
        "snr_db", "theta_deg", "K_db", "deep_fade_ratio", "clip_ratio",
        "occupancy_error", "canonical_error", "phase_error", "spectral_error",
    )
    columns = {name: _meta_column(meta, name, batch_size, device) for name in names}
    present = {
        name: torch.full((int(batch_size),), name in meta, device=device, dtype=torch.bool)
        for name in names
    }
    factors = {
        "snr_db": torch.sigmoid((columns["snr_db"] - 14.0) / 5.0),
        "theta_deg": (columns["theta_deg"] / 90.0).clamp(0.05, 1.0),
        "K_db": torch.sigmoid((columns["K_db"] - 6.0) / 5.0),
        "deep_fade_ratio": torch.exp(-2.0 * columns["deep_fade_ratio"].clamp_min(0.0)),
        "clip_ratio": torch.exp(-3.0 * columns["clip_ratio"].clamp_min(0.0)),
        "occupancy_error": torch.exp(-columns["occupancy_error"].abs()),
        "canonical_error": torch.exp(-columns["canonical_error"].abs()),
        "phase_error": torch.exp(-columns["phase_error"].abs()),
        "spectral_error": torch.exp(-columns["spectral_error"].abs()),
    }
    log_sum = torch.zeros(int(batch_size), device=device)
    count = torch.zeros(int(batch_size), device=device)
    for name in names:
        mask = present[name].float()
        log_sum = log_sum + mask * factors[name].clamp_min(1e-6).log()
        count = count + mask
    reliability = torch.exp(log_sum / count.clamp_min(1.0)).clamp(0.0, 1.0)
    if not return_details:
        return reliability
    details = {
        "metadata_present": present,
        "factor_values": factors,
        "missing_fraction": 1.0 - count.mean() / float(len(names)),
    }
    return reliability, details


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
    key = {"rx_sfo": "sfo", "rx_phase_noise": "phase_noise"}.get(key, key)
    if key not in DAOT_NUISANCE_TANGENT_NAMES and key != "rx_filter":
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
    elif key == "rx_filter":
        previous = torch.roll(complex_x, shifts=1, dims=1)
        previous[:, 0] = complex_x[:, 0]
        alpha = (delta * 0.50 * direction.abs()).clamp(0.0, 0.25).unsqueeze(1)
        result = (1.0 - alpha) * complex_x + alpha * previous
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


def sample_single_tx_intervention(
    clean_iq: torch.Tensor,
    *,
    seed: int,
    strength: float,
    sample_rate_hz: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply exactly one randomized, replayable TX intervention per sample."""

    if clean_iq.ndim != 3 or int(clean_iq.shape[1]) != 2:
        raise ValueError("clean_iq must have shape [batch,2,time]")
    if float(strength) <= 0.0 or float(sample_rate_hz) <= 0.0:
        raise ValueError("strength and sample_rate_hz must be positive")
    batch, _, length = clean_iq.shape
    generator = torch.Generator(device=clean_iq.device)
    generator.manual_seed(int(seed))
    direction_ids = torch.randint(0, 6, (batch,), generator=generator, device=clean_iq.device)
    signs = torch.randint(0, 2, (batch,), generator=generator, device=clean_iq.device).float() * 2.0 - 1.0
    signed = signs * float(strength)
    x = torch.complex(clean_iq[:, 0].float(), clean_iq[:, 1].float())
    output = x.clone()
    time = torch.arange(length, device=clean_iq.device, dtype=torch.float32) / float(sample_rate_hz)

    for direction_id in range(6):
        selected = direction_ids.eq(direction_id)
        if not bool(selected.any()):
            continue
        value = x[selected]
        amount = signed[selected].unsqueeze(1)
        if direction_id == 0:  # PA
            changed = value * (1.0 + amount * value.abs().square().clamp_max(4.0))
        elif direction_id == 1:  # IQ gain
            changed = torch.complex(value.real * (1.0 + amount), value.imag * (1.0 - amount))
        elif direction_id == 2:  # IQ phase
            changed = value * torch.exp(1j * amount)
        elif direction_id == 3:  # TX CFO
            changed = value * torch.exp(1j * 2.0 * torch.pi * amount * 50_000.0 * time.unsqueeze(0))
        elif direction_id == 4:  # clock skew proxy
            changed = value * torch.exp(1j * 2.0 * torch.pi * amount * 12_500.0 * time.unsqueeze(0))
        else:  # DAC nonlinearity
            changed = value + amount * value * value.abs().square().clamp_max(4.0)
        output[selected] = changed
    return (
        torch.stack([output.real, output.imag], dim=1).to(dtype=clean_iq.dtype),
        direction_ids,
        signs,
    )


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
