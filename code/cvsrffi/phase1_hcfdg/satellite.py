"""Single-forward mixed-orbit views for the Phase1 HCF-DG trainer."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Callable

import torch


CHANNEL_FACTOR_NAMES = ("cfo", "phase_noise", "snr", "multipath", "elevation")

_FACTOR_ALIASES = {
    "cfo": ("cfo", "cfo_bin", "cfo_hz", "residual_cfo_hz"),
    "phase_noise": (
        "phase_noise",
        "phase_noise_bin",
        "phase_noise_sigma",
        "phase_noise_sigma_max",
        "phase_noise_std",
    ),
    "snr": ("snr", "snr_bin", "snr_db"),
    "multipath": (
        "multipath",
        "multipath_bin",
        "multipath_strength",
        "multipath_profile",
        "num_taps",
    ),
    "elevation": ("elevation", "elevation_bin", "elevation_deg", "elev", "theta_deg"),
}


@dataclass(frozen=True)
class ChannelFactors:
    """The five scalar physical factor bins emitted for a satellite view."""

    cfo: torch.Tensor
    phase_noise: torch.Tensor
    snr: torch.Tensor
    multipath: torch.Tensor
    elevation: torch.Tensor

    names: ClassVar[tuple[str, ...]] = CHANNEL_FACTOR_NAMES

    def as_tensor(self) -> torch.Tensor:
        """Return factors in the fixed CFO-to-elevation column order."""

        return torch.stack(
            (self.cfo, self.phase_noise, self.snr, self.multipath, self.elevation),
            dim=-1,
        )


@dataclass(frozen=True)
class SingleViewBatch:
    """One IQ view per clean sample plus its channel supervision."""

    iq: torch.Tensor
    satellite_mask: torch.Tensor
    channel_labels: torch.Tensor
    channel_factors: torch.Tensor


def _validate_probability(p_sat: float) -> float:
    try:
        probability = float(p_sat)
    except (TypeError, ValueError) as exc:
        raise ValueError("p_sat must be a finite value in [0, 1]") from exc
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("p_sat must be a finite value in [0, 1]")
    return probability


def _factor_column(
    value: Any,
    *,
    name: str,
    batch_size: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    try:
        column = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"channel factor {name!r} must be numeric") from exc

    if column.ndim == 0:
        return column.expand(batch_size)
    if column.ndim == 1 and column.shape[0] == batch_size:
        return column
    if column.ndim == 2 and column.shape == (batch_size, 1):
        return column[:, 0]
    raise ValueError(
        f"channel factor {name!r} must have one value per satellite row; "
        f"got shape {tuple(column.shape)} for batch size {batch_size}"
    )


def _normalise_mapping(mapping: Mapping[Any, Any]) -> dict[str, Any]:
    return {
        str(key).strip().lower().replace("-", "_").replace(" ", "_"): value
        for key, value in mapping.items()
    }


def _factor_matrix(
    raw_factors: Any,
    *,
    batch_size: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    if raw_factors is None:
        raise ValueError(
            "satellite channel factors are required for selected satellite rows"
        )

    if isinstance(raw_factors, ChannelFactors):
        raw_factors = raw_factors.as_tensor()

    if isinstance(raw_factors, torch.Tensor):
        matrix = raw_factors.to(device=reference.device, dtype=reference.dtype)
        if matrix.ndim == 2 and matrix.shape == (batch_size, len(CHANNEL_FACTOR_NAMES)):
            return matrix
        raise ValueError(
            "channel factors tensor must have shape "
            f"({batch_size}, {len(CHANNEL_FACTOR_NAMES)})"
        )

    if isinstance(raw_factors, Sequence) and not isinstance(raw_factors, (str, bytes)):
        if len(raw_factors) != len(CHANNEL_FACTOR_NAMES):
            raise ValueError(
                "channel factor sequences must contain "
                f"{len(CHANNEL_FACTOR_NAMES)} columns"
            )
        columns = tuple(
            _factor_column(
                value,
                name=name,
                batch_size=batch_size,
                reference=reference,
            )
            for name, value in zip(CHANNEL_FACTOR_NAMES, raw_factors)
        )
        return torch.stack(columns, dim=1)

    if not isinstance(raw_factors, Mapping):
        attributes = {
            name: getattr(raw_factors, name)
            for name in CHANNEL_FACTOR_NAMES
            if hasattr(raw_factors, name)
        }
        if attributes:
            raw_factors = attributes
        else:
            raise TypeError(
                "augmentor factors must be a tensor, mapping, sequence, or ChannelFactors"
            )

    values = _normalise_mapping(raw_factors)
    nested = values.get("channel_factors", values.get("factors"))
    if nested is not None and nested is not raw_factors:
        return _factor_matrix(nested, batch_size=batch_size, reference=reference)

    resolved_values = {}
    missing_names = []
    for name in CHANNEL_FACTOR_NAMES:
        value = next((values[key] for key in _FACTOR_ALIASES[name] if key in values), None)
        if value is None:
            missing_names.append(name)
        else:
            resolved_values[name] = value

    if missing_names:
        raise ValueError(
            "missing required satellite channel factors: "
            + ", ".join(missing_names)
        )

    columns = [
        _factor_column(
            resolved_values[name],
            name=name,
            batch_size=batch_size,
            reference=reference,
        )
        for name in CHANNEL_FACTOR_NAMES
    ]
    return torch.stack(columns, dim=1)


def _unpack_augmentor_result(result: Any) -> tuple[torch.Tensor, Any]:
    if isinstance(result, tuple):
        if len(result) < 2:
            return result[0], None
        return result[0], result[1]

    if isinstance(result, Mapping):
        values = _normalise_mapping(result)
        iq = values.get("iq", values.get("samples", values.get("x")))
        if iq is None:
            raise TypeError("augmentor mapping result must contain an IQ tensor")
        factors = values.get("channel_factors", values.get("factors", result))
        return iq, factors

    if hasattr(result, "iq"):
        factors = getattr(
            result,
            "channel_factors",
            getattr(result, "factors", getattr(result, "metadata", None)),
        )
        return result.iq, factors

    return result, None


def build_single_view_batch(
    x: torch.Tensor,
    augmentor: Callable[..., Any],
    generator: torch.Generator,
    p_sat: float = 0.30,
) -> SingleViewBatch:
    """Replace selected clean rows with one mixed-orbit view in place.

    The Bernoulli mask is sampled on CPU so a CPU ``torch.Generator`` gives
    reproducible row selection independently of the IQ tensor device.  The
    selected rows are sent through the augmentor once; no clean/satellite
    concatenation is created.
    """

    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.ndim == 0:
        raise ValueError("x must have a batch dimension")
    probability = _validate_probability(p_sat)
    batch_size = int(x.shape[0])

    satellite_mask = torch.rand(batch_size, generator=generator, device="cpu") < probability
    output = x.clone()
    channel_labels = satellite_mask.to(device=x.device, dtype=torch.long)
    channel_factors = torch.zeros(
        (batch_size, len(CHANNEL_FACTOR_NAMES)),
        device=x.device,
        dtype=x.dtype,
    )

    if bool(satellite_mask.any()):
        selected_mask = satellite_mask.to(device=x.device)
        selected_clean = x[selected_mask]
        augmented_result = augmentor(
            selected_clean,
            scenario="mixed_orbit",
            generator=generator,
        )
        selected_iq, raw_factors = _unpack_augmentor_result(augmented_result)
        if not isinstance(selected_iq, torch.Tensor):
            raise TypeError("augmentor must return an IQ tensor or an IQ-bearing result")
        if selected_iq.shape != selected_clean.shape:
            raise ValueError(
                "augmentor must preserve the selected IQ shape; "
                f"got {tuple(selected_iq.shape)} for {tuple(selected_clean.shape)}"
            )
        selected_factors = _factor_matrix(
            raw_factors,
            batch_size=int(satellite_mask.sum().item()),
            reference=x,
        )
        output[selected_mask] = selected_iq.to(device=x.device, dtype=x.dtype)
        channel_factors[selected_mask] = selected_factors

    return SingleViewBatch(
        iq=output,
        satellite_mask=satellite_mask,
        channel_labels=channel_labels,
        channel_factors=channel_factors,
    )


__all__ = [
    "CHANNEL_FACTOR_NAMES",
    "ChannelFactors",
    "SingleViewBatch",
    "build_single_view_batch",
]
