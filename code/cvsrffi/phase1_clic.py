"""Totalized Phase1 CLIC token operators over one received-IQ observation.

The operators in this module are deliberately pure: they derive local views
from the supplied ``received_i`` tensor only and never synthesize or read a
second observation.  Their zero-domain policy is mathematical rather than an
epsilon approximation: a token is defined only when its three amplitudes are
strictly positive and finite.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


CLIC_LAGS = (1, 2, 4, 8)
CLIC_INPUT_LENGTH = 256
CLIC_EMBED_DIM = 160
CLIC_INIT_SEED = 7281164
CLIC_EXTRA_PARAMETER_COUNT = 32529
FORMAL_LEO_WEAK_SCENARIOS = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
FROZEN_FOLDS = ("F1", "F2", "F3", "F4", "F5", "F6")

_RAW_PHASE_CONTROL = "raw_phase_control"
_COMPLEX_LOCAL_INVARIANT_CURVATURE = "complex_local_invariant_curvature"
_ALLOWED_OPERATOR_MODES = {
    _RAW_PHASE_CONTROL,
    _COMPLEX_LOCAL_INVARIANT_CURVATURE,
}
_MINIMUM_INPUT_LENGTH = 2 * max(CLIC_LAGS) + 1
_ALLOWED_INPUT_DTYPES = (torch.float32, torch.float64)


class CLICConfigError(ValueError):
    """Raised when a frozen CLIC operator contract is not satisfied."""


class CLICRuntimeError(RuntimeError):
    """Raised when finite CLIC arithmetic cannot be completed safely."""


@dataclass(frozen=True)
class CLICConfig:
    """Frozen public configuration surface for the Phase1 CLIC branch."""

    frozen_mode: bool
    operator_mode: str
    input_length: int = CLIC_INPUT_LENGTH
    embed_dim: int = CLIC_EMBED_DIM


@dataclass
class CLICTokenBatch:
    """Tokenized local views and their independently represented validity."""

    tokens: torch.Tensor
    valid_mask: torch.Tensor
    reliability: torch.Tensor
    valid_fraction: torch.Tensor
    reliability_mean: torch.Tensor


def _validate_received_i(received_i: torch.Tensor, *, operator_mode: str) -> None:
    if not isinstance(operator_mode, str) or operator_mode not in _ALLOWED_OPERATOR_MODES:
        raise CLICConfigError(
            "operator_mode must be 'raw_phase_control' or "
            "'complex_local_invariant_curvature'"
        )
    if not isinstance(received_i, torch.Tensor):
        raise CLICConfigError("received_i must be a torch.Tensor")
    if received_i.ndim != 3 or received_i.shape[1] != 2:
        raise CLICConfigError("received_i must have shape [B, 2, T]")
    if received_i.shape[2] < _MINIMUM_INPUT_LENGTH:
        raise CLICConfigError(
            f"received_i must have T >= {_MINIMUM_INPUT_LENGTH} for fixed CLIC lags"
        )
    if received_i.dtype not in _ALLOWED_INPUT_DTYPES:
        raise CLICConfigError("received_i dtype must be torch.float32 or torch.float64")
    _require_all_finite(received_i, name="received_i")


def _require_all_finite(*tensors: torch.Tensor, name: str = "CLIC intermediate") -> None:
    for tensor in tensors:
        if not bool(torch.isfinite(tensor).all().item()):
            raise CLICRuntimeError(f"non-finite {name}")


def _positive_phase(z: torch.Tensor, amplitude: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact unit phase and its strictly-positive amplitude mask."""

    _require_all_finite(amplitude, name="amplitude")
    positive = amplitude > 0
    phase = torch.zeros_like(z)
    if bool(positive.any().item()):
        phase[positive] = z[positive] / amplitude[positive]
    _require_all_finite(phase, name="normalized phase")
    return phase, positive


def _safe_ratio(
    a_left: torch.Tensor,
    a_center: torch.Tensor,
    a_right: torch.Tensor,
    valid_inner: torch.Tensor,
) -> torch.Tensor:
    """Compute min/max reliability only on the positive three-point domain."""

    ratio = torch.zeros_like(a_center)
    if not bool(valid_inner.any().item()):
        return ratio

    left_valid = a_left[valid_inner]
    center_valid = a_center[valid_inner]
    right_valid = a_right[valid_inner]
    numerator = torch.minimum(torch.minimum(left_valid, center_valid), right_valid)
    denominator = torch.maximum(torch.maximum(left_valid, center_valid), right_valid)
    _require_all_finite(numerator, denominator, name="reliability extrema")
    if bool((denominator <= 0).any().item()):
        raise CLICRuntimeError("non-positive reliability denominator")
    ratio_values = numerator / denominator
    _require_all_finite(ratio_values, name="reliability")
    ratio[valid_inner] = ratio_values
    return ratio


def _g_inner_channels(
    phase: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    a_left: torch.Tensor,
    a_center: torch.Tensor,
    a_right: torch.Tensor,
    valid_inner: torch.Tensor,
    ratio: torch.Tensor,
) -> torch.Tensor:
    """Build G channels without evaluating logarithms outside the definition domain."""

    batch, inner_length = valid_inner.shape
    channels = a_center.new_zeros((batch, 4, inner_length))
    if not bool(valid_inner.any().item()):
        return channels

    phase_left, phase_center, phase_right = phase
    u = (
        phase_right[valid_inner]
        * phase_left[valid_inner]
        * phase_center[valid_inner].conj().square()
    )
    _require_all_finite(u, name="complex local curvature")

    h = (
        torch.log(a_right[valid_inner])
        + torch.log(a_left[valid_inner])
        - 2 * torch.log(a_center[valid_inner])
    )
    _require_all_finite(h, name="log-amplitude curvature")
    h = h.clamp(-8, 8)
    _require_all_finite(h, name="clipped log-amplitude curvature")

    channels[:, 0, :][valid_inner] = u.real
    channels[:, 1, :][valid_inner] = u.imag
    channels[:, 2, :][valid_inner] = h
    channels[:, 3, :][valid_inner] = ratio[valid_inner]
    return channels


def _c_inner_channels(
    phase_left: torch.Tensor,
    phase_right: torch.Tensor,
    valid_inner: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the raw-phase control channels on the shared valid domain."""

    batch, inner_length = valid_inner.shape
    channels = torch.zeros((batch, 4, inner_length), dtype=dtype, device=phase_left.device)
    if not bool(valid_inner.any().item()):
        return channels

    channels[:, 0, :][valid_inner] = phase_right.real[valid_inner]
    channels[:, 1, :][valid_inner] = phase_right.imag[valid_inner]
    channels[:, 2, :][valid_inner] = phase_left.real[valid_inner]
    channels[:, 3, :][valid_inner] = phase_left.imag[valid_inner]
    return channels


def totalized_clic_tokens(
    received_i: torch.Tensor,
    *,
    operator_mode: str,
) -> CLICTokenBatch:
    """Return fixed-lag CLIC C or G tokens from one real-valued IQ tensor.

    Every output location outside the positive, finite three-point domain is
    exactly zero and separately marked invalid.  No epsilon is introduced:
    divisions and logarithms are performed only after the domain mask is
    established.
    """

    _validate_received_i(received_i, operator_mode=operator_mode)
    batch, _, length = received_i.shape

    in_phase = received_i[:, 0]
    quadrature = received_i[:, 1]
    z = torch.complex(in_phase, quadrature)
    amplitude = torch.hypot(in_phase, quadrature)
    phase, positive = _positive_phase(z, amplitude)

    token_blocks: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    reliabilities: list[torch.Tensor] = []
    for lag in CLIC_LAGS:
        inner_length = length - 2 * lag
        left = slice(0, inner_length)
        center = slice(lag, length - lag)
        right = slice(2 * lag, length)

        valid_inner = positive[:, left] & positive[:, center] & positive[:, right]
        mask = torch.zeros((batch, length), dtype=torch.bool, device=received_i.device)
        mask[:, center] = valid_inner

        a_left = amplitude[:, left]
        a_center = amplitude[:, center]
        a_right = amplitude[:, right]
        ratio_inner = _safe_ratio(a_left, a_center, a_right, valid_inner)

        reliability = received_i.new_zeros((batch, length))
        reliability[:, center] = ratio_inner
        _require_all_finite(reliability, name="reliability output")

        block = received_i.new_zeros((batch, 4, length))
        if operator_mode == _COMPLEX_LOCAL_INVARIANT_CURVATURE:
            inner_channels = _g_inner_channels(
                (phase[:, left], phase[:, center], phase[:, right]),
                a_left,
                a_center,
                a_right,
                valid_inner,
                ratio_inner,
            )
        else:
            inner_channels = _c_inner_channels(
                phase[:, left],
                phase[:, right],
                valid_inner,
                dtype=received_i.dtype,
            )
        block[:, :, center] = inner_channels
        _require_all_finite(block, name="token block")

        token_blocks.append(block)
        masks.append(mask)
        reliabilities.append(reliability)

    tokens = torch.cat(token_blocks, dim=1)
    valid_mask = torch.stack(masks, dim=1)
    reliability = torch.stack(reliabilities, dim=1)
    _require_all_finite(tokens, reliability, name="CLIC outputs")

    valid_count = valid_mask.sum(dim=(1, 2))
    reliability_mean = reliability.sum(dim=(1, 2)) / valid_count.clamp_min(1)
    reliability_mean = torch.where(
        valid_count > 0,
        reliability_mean,
        torch.zeros_like(reliability_mean),
    )
    valid_fraction = valid_mask.float().mean(dim=(1, 2))
    _require_all_finite(valid_fraction, reliability_mean, name="CLIC summary")
    return CLICTokenBatch(
        tokens=tokens,
        valid_mask=valid_mask,
        reliability=reliability,
        valid_fraction=valid_fraction,
        reliability_mean=reliability_mean,
    )


__all__ = [
    "CLICConfig",
    "CLICConfigError",
    "CLICRuntimeError",
    "CLICTokenBatch",
    "CLIC_EMBED_DIM",
    "CLIC_EXTRA_PARAMETER_COUNT",
    "CLIC_INIT_SEED",
    "CLIC_INPUT_LENGTH",
    "CLIC_LAGS",
    "FORMAL_LEO_WEAK_SCENARIOS",
    "FROZEN_FOLDS",
    "totalized_clic_tokens",
]
