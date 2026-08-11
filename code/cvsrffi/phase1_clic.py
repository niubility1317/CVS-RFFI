"""Totalized Phase1 CLIC token operators over one received-IQ observation.

The operators in this module are deliberately pure: they derive local views
from the supplied ``received_i`` tensor only and never synthesize or read a
second observation.  Their zero-domain policy is mathematical rather than an
epsilon approximation: a token is defined only when its three amplitudes are
strictly positive and finite.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


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


@dataclass
class CLICForwardResult:
    """One CLIC fusion result from exactly one received-IQ observation."""

    z_id: torch.Tensor
    q_clic: torch.Tensor
    token_batch: CLICTokenBatch


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


def _capture_rng_states() -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
    """Capture every caller-visible generator touched by frozen initialization."""

    cpu_state = torch.random.get_rng_state().clone()
    cuda_states: Optional[List[torch.Tensor]] = None
    if torch.cuda.is_available():
        cuda_states = [state.clone() for state in torch.cuda.get_rng_state_all()]
    return cpu_state, cuda_states


def _restore_rng_states(
    cpu_state: torch.Tensor,
    cuda_states: Optional[List[torch.Tensor]],
) -> None:
    """Restore CPU and all available CUDA RNG states byte-for-byte."""

    torch.random.set_rng_state(cpu_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)


def _require_module_finite(module: nn.Module) -> None:
    for name, parameter in module.named_parameters():
        _require_all_finite(parameter, name=f"parameter {name}")
    for name, buffer in module.named_buffers():
        _require_all_finite(buffer, name=f"buffer {name}")


def _validate_fusion_architecture(module: nn.Module) -> None:
    """Reject any drift from the frozen 32,529-parameter CLIC module."""

    required = (
        "depthwise",
        "gn1",
        "pointwise",
        "gn2",
        "embed",
        "correction",
        "gate_norm",
        "gate",
    )
    if any(not hasattr(module, name) for name in required):
        raise CLICConfigError("module does not expose the frozen CLIC architecture")

    if not isinstance(module.depthwise, nn.Conv1d) or (
        module.depthwise.in_channels,
        module.depthwise.out_channels,
        module.depthwise.kernel_size,
        module.depthwise.stride,
        module.depthwise.padding,
        module.depthwise.dilation,
        module.depthwise.groups,
        module.depthwise.padding_mode,
        module.depthwise.bias,
    ) != (16, 16, (5,), (1,), (2,), (1,), 16, "zeros", None):
        raise CLICConfigError("depthwise CLIC architecture drift")
    if not isinstance(module.gn1, nn.GroupNorm) or (
        module.gn1.num_groups,
        module.gn1.num_channels,
        module.gn1.eps,
        module.gn1.affine,
        module.gn1.weight is not None,
        module.gn1.bias is not None,
    ) != (4, 16, 1e-5, True, True, True):
        raise CLICConfigError("first GroupNorm CLIC architecture drift")
    if not isinstance(module.pointwise, nn.Conv1d) or (
        module.pointwise.in_channels,
        module.pointwise.out_channels,
        module.pointwise.kernel_size,
        module.pointwise.stride,
        module.pointwise.padding,
        module.pointwise.dilation,
        module.pointwise.groups,
        module.pointwise.padding_mode,
        module.pointwise.bias,
    ) != (16, 32, (1,), (1,), (0,), (1,), 1, "zeros", None):
        raise CLICConfigError("pointwise CLIC architecture drift")
    if not isinstance(module.gn2, nn.GroupNorm) or (
        module.gn2.num_groups,
        module.gn2.num_channels,
        module.gn2.eps,
        module.gn2.affine,
        module.gn2.weight is not None,
        module.gn2.bias is not None,
    ) != (8, 32, 1e-5, True, True, True):
        raise CLICConfigError("second GroupNorm CLIC architecture drift")
    if not isinstance(module.embed, nn.Linear) or (
        module.embed.in_features,
        module.embed.out_features,
        module.embed.bias is not None,
    ) != (32, CLIC_EMBED_DIM, True):
        raise CLICConfigError("token embedding CLIC architecture drift")
    if not isinstance(module.correction, nn.Linear) or (
        module.correction.in_features,
        module.correction.out_features,
        module.correction.bias,
    ) != (CLIC_EMBED_DIM, CLIC_EMBED_DIM, None):
        raise CLICConfigError("correction CLIC architecture drift")
    if not isinstance(module.gate_norm, nn.LayerNorm) or (
        tuple(module.gate_norm.normalized_shape),
        module.gate_norm.eps,
        module.gate_norm.elementwise_affine,
        module.gate_norm.weight is not None,
        module.gate_norm.bias is not None,
    ) != ((2 * CLIC_EMBED_DIM,), 1e-5, True, True, True):
        raise CLICConfigError("gate LayerNorm CLIC architecture drift")
    if not isinstance(module.gate, nn.Linear) or (
        module.gate.in_features,
        module.gate.out_features,
        module.gate.bias is not None,
    ) != (2 * CLIC_EMBED_DIM, 1, True):
        raise CLICConfigError("gate CLIC architecture drift")
    if sum(parameter.numel() for parameter in module.parameters()) != CLIC_EXTRA_PARAMETER_COUNT:
        raise CLICConfigError("CLIC parameter count drift")


def initialize_clic_module_(module: nn.Module, *, seed: int = CLIC_INIT_SEED) -> nn.Module:
    """Initialize a frozen CLIC module without changing the caller RNG state.

    The constructor separately snapshots state before default PyTorch layer
    initialization.  This public reinitializer owns its own snapshot as well,
    so direct callers receive the same no-side-effect guarantee.
    """

    if not isinstance(seed, int):
        raise CLICConfigError("CLIC initialization seed must be an integer")
    _validate_fusion_architecture(module)
    cpu_state, cuda_states = _capture_rng_states()
    try:
        torch.manual_seed(seed)
        if cuda_states is not None:
            torch.cuda.manual_seed_all(seed)

        with torch.no_grad():
            # Match PyTorch's Kaiming-uniform encoder convention, including
            # the deterministic embedding bias derived from its fan-in.
            nn.init.kaiming_uniform_(module.depthwise.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(module.pointwise.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(module.embed.weight, a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.embed.weight)
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(module.embed.bias, -bound, bound)

            module.gn1.weight.fill_(1.0)
            module.gn1.bias.zero_()
            module.gn2.weight.fill_(1.0)
            module.gn2.bias.zero_()
            module.gate_norm.weight.fill_(1.0)
            module.gate_norm.bias.zero_()

            nn.init.orthogonal_(module.correction.weight)
            module.correction.weight.mul_(0.01)
            module.gate.weight.zero_()
            module.gate.bias.fill_(math.log(0.1 / 0.9))

        _require_module_finite(module)
        return module
    finally:
        _restore_rng_states(cpu_state, cuda_states)


def clic_state_sha256(module: nn.Module) -> str:
    """Return a device-independent SHA-256 of the complete finite CLIC state."""

    _validate_fusion_architecture(module)
    _require_module_finite(module)
    digest = hashlib.sha256()
    digest.update(b"cvs.phase1.clic.state.v1\0")
    for name, tensor in module.state_dict().items():
        _require_all_finite(tensor, name=f"state {name}")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        raw = value.numpy().tobytes(order="C")
        digest.update(len(raw).to_bytes(8, byteorder="little", signed=False))
        digest.update(raw)
    return digest.hexdigest()


class CLICFusion(nn.Module):
    """Frozen parameter-matched fusion shared by the C and G token operators."""

    def __init__(
        self,
        *,
        embed_dim: int = CLIC_EMBED_DIM,
        input_length: int = CLIC_INPUT_LENGTH,
    ) -> None:
        if embed_dim != CLIC_EMBED_DIM:
            raise CLICConfigError(f"CLIC embed_dim must equal {CLIC_EMBED_DIM}")
        if input_length != CLIC_INPUT_LENGTH:
            raise CLICConfigError(f"CLIC input_length must equal {CLIC_INPUT_LENGTH}")

        # This snapshot must precede *every* nn layer creation: constructors
        # perform their own default random initialization before we overwrite it.
        caller_cpu_state, caller_cuda_states = _capture_rng_states()
        try:
            super().__init__()
            self.embed_dim = embed_dim
            self.input_length = input_length
            self.depthwise = nn.Conv1d(16, 16, 5, padding=2, groups=16, bias=False)
            self.gn1 = nn.GroupNorm(4, 16)
            self.pointwise = nn.Conv1d(16, 32, 1, bias=False)
            self.gn2 = nn.GroupNorm(8, 32)
            self.embed = nn.Linear(32, CLIC_EMBED_DIM, bias=True)
            self.correction = nn.Linear(CLIC_EMBED_DIM, CLIC_EMBED_DIM, bias=False)
            self.gate_norm = nn.LayerNorm(2 * CLIC_EMBED_DIM)
            self.gate = nn.Linear(2 * CLIC_EMBED_DIM, 1, bias=True)
            initialize_clic_module_(self, seed=CLIC_INIT_SEED)
            _validate_fusion_architecture(self)
        finally:
            _restore_rng_states(caller_cpu_state, caller_cuda_states)

    def _validate_forward_inputs(self, received_i: torch.Tensor, z_base: torch.Tensor) -> None:
        if not isinstance(received_i, torch.Tensor):
            raise CLICConfigError("received_i must be a torch.Tensor")
        if received_i.ndim != 3 or received_i.shape[1] != 2:
            raise CLICConfigError("received_i must have shape [B, 2, T]")
        if received_i.shape[2] != self.input_length:
            raise CLICConfigError(
                f"received_i must have frozen T={self.input_length} for CLIC fusion"
            )
        if not isinstance(z_base, torch.Tensor):
            raise CLICConfigError("z_base must be a torch.Tensor")
        if z_base.ndim != 2 or z_base.shape != (received_i.shape[0], self.embed_dim):
            raise CLICConfigError(
                f"z_base must have shape [B, {self.embed_dim}] matched to received_i"
            )

        parameter = self.depthwise.weight
        if received_i.device != parameter.device or z_base.device != parameter.device:
            raise CLICConfigError("received_i, z_base, and CLIC module must share one device")
        if received_i.dtype != parameter.dtype or z_base.dtype != parameter.dtype:
            raise CLICConfigError("received_i, z_base, and CLIC module must share one dtype")
        _require_all_finite(z_base, name="z_base")

    def forward(
        self,
        received_i: torch.Tensor,
        z_base: torch.Tensor,
        *,
        operator_mode: str,
    ) -> CLICForwardResult:
        """Fuse a C or G token batch into one identity embedding and quality row."""

        _require_module_finite(self)
        self._validate_forward_inputs(received_i, z_base)
        token_batch = totalized_clic_tokens(received_i, operator_mode=operator_mode)
        _require_all_finite(
            token_batch.tokens,
            token_batch.reliability,
            token_batch.valid_fraction,
            token_batch.reliability_mean,
            name="token batch",
        )

        channel_mask = token_batch.valid_mask.repeat_interleave(4, dim=1).to(
            dtype=token_batch.tokens.dtype
        )
        position_mask = token_batch.valid_mask.any(dim=1, keepdim=True).to(
            dtype=token_batch.tokens.dtype
        )
        masked_tokens = token_batch.tokens * channel_mask
        _require_all_finite(channel_mask, position_mask, masked_tokens, name="token masks")

        depthwise = self.depthwise(masked_tokens)
        _require_all_finite(depthwise, name="depthwise output")
        encoded = F.silu(self.gn1(depthwise))
        _require_all_finite(encoded, name="first encoder output")
        pointwise = self.pointwise(encoded)
        _require_all_finite(pointwise, name="pointwise output")
        encoded = F.silu(self.gn2(pointwise)) * position_mask
        _require_all_finite(encoded, name="second encoder output")

        denominator = position_mask.sum(dim=2).clamp_min(1)
        pooled = encoded.sum(dim=2) / denominator
        _require_all_finite(denominator, pooled, name="masked pooled embedding")
        token_embedding = self.embed(pooled)
        _require_all_finite(token_embedding, name="token embedding")
        corrected_embedding = self.correction(token_embedding)
        _require_all_finite(corrected_embedding, name="correction output")

        gate_input = torch.cat((z_base, token_embedding), dim=1)
        _require_all_finite(gate_input, name="gate input")
        normalized_gate_input = self.gate_norm(gate_input)
        _require_all_finite(normalized_gate_input, name="normalized gate input")
        gate_logit = self.gate(normalized_gate_input).squeeze(1)
        gate_probability = torch.sigmoid(gate_logit)
        _require_all_finite(gate_logit, gate_probability, name="gate output")

        full_fallback_bool = token_batch.valid_mask.sum(dim=(1, 2)) == 0
        full_fallback = full_fallback_bool.to(dtype=z_base.dtype)
        gamma = token_batch.reliability_mean * gate_probability
        gamma = gamma * (1.0 - full_fallback)
        residual = gamma[:, None] * corrected_embedding
        candidate_z_id = z_base + residual
        # `where` preserves the base branch byte-for-byte, including signed zero.
        z_id = torch.where(full_fallback_bool[:, None], z_base, candidate_z_id)
        q_clic = torch.stack(
            (
                gamma,
                token_batch.reliability_mean,
                token_batch.valid_fraction.to(dtype=z_base.dtype),
                full_fallback,
            ),
            dim=1,
        )
        _require_all_finite(gamma, residual, candidate_z_id, z_id, q_clic, name="CLIC fusion outputs")
        return CLICForwardResult(z_id=z_id, q_clic=q_clic, token_batch=token_batch)


__all__ = [
    "CLICConfig",
    "CLICConfigError",
    "CLICForwardResult",
    "CLICFusion",
    "CLICRuntimeError",
    "CLICTokenBatch",
    "CLIC_EMBED_DIM",
    "CLIC_EXTRA_PARAMETER_COUNT",
    "CLIC_INIT_SEED",
    "CLIC_INPUT_LENGTH",
    "CLIC_LAGS",
    "FORMAL_LEO_WEAK_SCENARIOS",
    "FROZEN_FOLDS",
    "clic_state_sha256",
    "initialize_clic_module_",
    "totalized_clic_tokens",
]
