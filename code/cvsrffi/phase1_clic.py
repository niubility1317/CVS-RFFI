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
import json
import math
import os
from pathlib import Path
from tempfile import mkstemp
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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


class CLICWarmStartError(CLICRuntimeError):
    """Raised when the exact baseline-to-CLIC state contract drifts."""


class CLICTerminalError(CLICRuntimeError):
    """Raised when a scalar CLIC training receipt cannot close strictly."""


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


def _is_lowercase_sha256(value: object) -> bool:
    """Return whether ``value`` is a canonical lower-case SHA-256 digest."""

    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _tensor_state_bytes(value: torch.Tensor) -> bytes:
    """Materialize the exact CPU C-order bytes of one finite tensor state item."""

    if not isinstance(value, torch.Tensor):
        raise CLICWarmStartError("CLIC state item is not a tensor")
    if not bool(torch.isfinite(value).all().item()):
        raise CLICWarmStartError("CLIC state contains a non-finite tensor")
    return value.detach().cpu().contiguous().numpy().tobytes(order="C")


def _canonical_tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash an exact finite tensor state with names, dtypes, shapes, and bytes."""

    if not isinstance(state, Mapping) or not state:
        raise CLICWarmStartError("CLIC existing state is absent")
    digest = hashlib.sha256()
    digest.update(b"cvs.phase1.clic.existing_state.v1\0")
    for key in sorted(state):
        if not isinstance(key, str) or not key:
            raise CLICWarmStartError("CLIC state key is invalid")
        value = state[key]
        if not isinstance(value, torch.Tensor):
            raise CLICWarmStartError(f"CLIC state item is not a tensor: {key}")
        raw = _tensor_state_bytes(value)
        stable = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stable.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(stable.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(len(raw).to_bytes(8, byteorder="little", signed=False))
        digest.update(raw)
    return digest.hexdigest()


def _clic_raw_model(model: nn.Module) -> nn.Module:
    """Resolve a compiled wrapper without accepting a non-module state owner."""

    raw_model = getattr(model, "_orig_mod", model)
    if not isinstance(raw_model, nn.Module):
        raise CLICWarmStartError("CLIC warm-start model is invalid")
    return raw_model


def _clic_module_from_model(model: nn.Module) -> nn.Module:
    """Resolve and validate the sole frozen CLIC module owned by id_backbone."""

    backbone = getattr(model, "id_backbone", None)
    module = getattr(backbone, "clic", None)
    if not isinstance(module, nn.Module):
        raise CLICWarmStartError("CLIC warm-start requires id_backbone.clic")
    _validate_fusion_architecture(module)
    return module


def strict_clic_warm_start(
    model: nn.Module,
    checkpoint_state: Mapping[str, torch.Tensor],
    *,
    checkpoint_sha256: str,
) -> Dict[str, Any]:
    """Load only exact old-model state while preserving deterministic CLIC init.

    The external checkpoint digest seals the checkpoint file supplied by the
    caller.  The helper independently validates the old tensor namespace and
    records a canonical post-load digest, so future mutation of the caller's
    mapping cannot alter the loaded model or its receipt.
    """

    if not _is_lowercase_sha256(checkpoint_sha256):
        raise CLICWarmStartError("CLIC checkpoint_sha256 must be lowercase SHA-256")
    if not isinstance(checkpoint_state, Mapping):
        raise CLICWarmStartError("CLIC checkpoint state must be a mapping")

    raw_model = _clic_raw_model(model)
    clic_module = _clic_module_from_model(raw_model)
    current = raw_model.state_dict()
    clic_keys = tuple(key for key in current if key.startswith("id_backbone.clic."))
    old_keys = tuple(key for key in current if key not in clic_keys)
    if not clic_keys or not old_keys:
        raise CLICWarmStartError("CLIC warm-start state namespace is incomplete")
    if any(key.startswith("id_backbone.clic.") for key in checkpoint_state):
        raise CLICWarmStartError("checkpoint must not contain id_backbone.clic state")
    if set(checkpoint_state) != set(old_keys):
        raise CLICWarmStartError("checkpoint keys differ from exact pre-CLIC state")

    isolated_old_state: Dict[str, torch.Tensor] = {}
    for key in old_keys:
        expected = current[key]
        supplied = checkpoint_state[key]
        if not isinstance(supplied, torch.Tensor):
            raise CLICWarmStartError(f"checkpoint state item is not a tensor: {key}")
        if supplied.shape != expected.shape or supplied.dtype != expected.dtype:
            raise CLICWarmStartError(f"checkpoint tensor contract drifted: {key}")
        _tensor_state_bytes(expected)
        _tensor_state_bytes(supplied)
        isolated_old_state[key] = supplied.detach().clone()

    clic_sha_before = clic_state_sha256(clic_module)
    clic_parameter_count = sum(parameter.numel() for parameter in clic_module.parameters())
    if clic_parameter_count != CLIC_EXTRA_PARAMETER_COUNT:
        raise CLICWarmStartError("CLIC parameter count drifted before warm-start")
    expected_old_sha = _canonical_tensor_state_sha256(isolated_old_state)
    merged_state: Dict[str, torch.Tensor] = {}
    for key, value in current.items():
        merged_state[key] = (
            value.detach().clone()
            if key.startswith("id_backbone.clic.")
            else isolated_old_state[key]
        )
    try:
        incompatible = raw_model.load_state_dict(merged_state, strict=True)
    except Exception as error:
        raise CLICWarmStartError("CLIC strict old-state load failed") from error
    missing = tuple(getattr(incompatible, "missing_keys", ()) or ())
    unexpected = tuple(getattr(incompatible, "unexpected_keys", ()) or ())
    if missing or unexpected:
        raise CLICWarmStartError("CLIC strict old-state load reported incompatible keys")

    loaded = raw_model.state_dict()
    loaded_old_state = {key: loaded[key] for key in old_keys}
    loaded_old_sha = _canonical_tensor_state_sha256(loaded_old_state)
    if loaded_old_sha != expected_old_sha:
        raise CLICWarmStartError("existing state bytes changed during strict CLIC load")
    for key in old_keys:
        if _tensor_state_bytes(loaded[key]) != _tensor_state_bytes(isolated_old_state[key]):
            raise CLICWarmStartError(f"existing state byte mismatch after strict load: {key}")
    if clic_state_sha256(clic_module) != clic_sha_before:
        raise CLICWarmStartError("CLIC initialization bytes changed during strict load")

    return {
        "checkpoint_sha256": checkpoint_sha256,
        "existing_state_sha256": loaded_old_sha,
        "clic_init_state_sha256": clic_sha_before,
        "clic_parameter_count": clic_parameter_count,
        "existing_state_unchanged": True,
        "optimizer_state_restored": False,
        "rng_state_restored": False,
        "warm_start_mode": "MODEL_WEIGHTS_ONLY_NEW_ADAMW_AMP",
    }


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


_CLIC_RECEIPT_SCHEMA = "cvs.phase1.clic_receipt.v1"
_CLIC_METHOD = "P1_CLIC"
_CLIC_TERMINAL_ENVELOPE_SCHEMA = "cvs.phase1.clic_terminal_envelope.v1"
_CLIC_TERMINAL_ENVELOPE_FIELDS = frozenset(
    {
        "schema",
        "method",
        "strict_core",
        "selected_checkpoint_path",
        "selected_checkpoint_sha256",
    }
)
_CLIC_OPERATOR_IDS = {
    "C": "C_RAW_PHASE_CONTROL",
    "G": "G_INVARIANT_CURVATURE",
}
_CLIC_ZERO_ONLY_FIELDS = {
    "use_target": False,
    "use_proxy": False,
    "use_held": False,
    "use_u": False,
    "use_v": False,
    "query_truth_access": False,
    "query_role_access": False,
    "new_view_count": 0,
    "second_forward_count": 0,
    "state_feedback_count": 0,
    "legacy_method_identity": False,
}
_CLIC_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "method",
        "arm",
        "operator_id",
        "batch_size",
        "input_length",
        "embed_dim",
        "local_class_count",
        "lags",
        "clip",
        "clic_active",
        "new_adamw",
        "checkpoint_sha256",
        "final_checkpoint_sha256",
        "existing_state_sha256",
        "clic_init_state_sha256",
        "clic_parameter_count",
        "existing_state_unchanged",
        "optimizer_state_restored",
        "rng_state_restored",
        "source_l_only",
        "source_split_count",
        "source_split_sha256",
        "class_order_count",
        "class_order_sha256",
        "physical_order_count",
        "physical_order_sha256",
        "common_batch_sequence_sha256",
        "common_batch_sequence_batches",
        "common_batch_sequence_rows",
        "common_scenario_batches",
        "common_binding_events",
        "scene_audits",
        "resource_observations",
        "amp_events",
        "amp_summary_events",
        "amp_attempts",
        "scaled_backward_count",
        "unscale_count",
        "optimizer_step_attempts",
        "effective_optimizer_steps",
        "raw_finite_overflow_skips",
        "scale_decrease_count",
        "optimizer_unchanged_count",
        "raw_nonfinite_count",
        "material_nonfinite_count",
        "consecutive_overflow_skips",
        "max_consecutive_overflow_skips",
        "persistent_overflow",
        "graph_release_count",
        "head_path",
        "completed",
        "terminal_contract",
        "terminal_contract_passed",
    }
) | frozenset(_CLIC_ZERO_ONLY_FIELDS)
_CLIC_VJP_SUMMARY_FIELDS = frozenset({"count", "norm", "finite", "nonzero"})
_CLIC_SCENE_AUDIT_FIELDS = frozenset(
    {
        "valid_token_coverage",
        "gate_or_correction_nonzero",
        "raw_unscaled",
        "diagnostic_only",
        "touches_amp_optimizer_rng",
        "completed",
        "token",
        "clic",
        "base",
        "head",
        "clic_groups",
    }
)
_CLIC_DETAILED_VJP_GROUPS = frozenset({"depthwise", "pointwise", "embed", "correction", "gate"})
_CLIC_COMMON_BINDING_FIELDS = frozenset(
    {
        "scene",
        "batch_index",
        "rows",
        "source_split_count",
        "source_split_sha256",
        "class_order_count",
        "class_order_sha256",
        "physical_order_count",
        "physical_order_sha256",
        "common_batch_sequence_sha256",
    }
)
_CLIC_RESOURCE_FIELDS = frozenset(
    {"scene", "batch_index", "peak_memory_bytes", "step_time_seconds", "selection_feedback"}
)
_CLIC_AMP_CORE_FIELDS = frozenset(
    {
        "amp_overflow_detected",
        "scaled_backward_count",
        "unscale_count",
        "optimizer_step_attempted",
        "effective_optimizer_step",
        "raw_finite",
        "scale_decreased",
        "optimizer_state_unchanged",
        "raw_nonfinite",
        "material_nonfinite",
    }
)
_CLIC_AMP_EVENT_FIELDS = _CLIC_AMP_CORE_FIELDS | frozenset({"scene", "batch_index"})


def _strict_nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CLICTerminalError(f"{field} must be a nonnegative integer")
    return int(value)


def _strict_positive_int(value: object, *, field: str) -> int:
    result = _strict_nonnegative_int(value, field=field)
    if result <= 0:
        raise CLICTerminalError(f"{field} must be a positive integer")
    return result


def _strict_finite_nonnegative_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CLICTerminalError(f"{field} must be a finite nonnegative scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise CLICTerminalError(f"{field} must be a finite nonnegative scalar")
    return result


def _require_receipt_sha256(receipt: Mapping[str, object], field: str) -> str:
    value = receipt.get(field)
    if not _is_lowercase_sha256(value):
        raise CLICTerminalError(f"{field} must be a lowercase SHA-256")
    return str(value)


def _validate_clic_receipt_field_name(key: str, value: object) -> None:
    """Reject data-bearing receipt names while allowing explicit zero proofs."""

    if key in _CLIC_ZERO_ONLY_FIELDS:
        expected = _CLIC_ZERO_ONLY_FIELDS[key]
        if value is not expected and value != expected:
            raise CLICTerminalError(f"forbidden nonzero receipt field: {key}")
        return
    if key in {"physical_order_count", "physical_order_sha256", "operator_id", "valid_token_coverage", "token"}:
        return
    lowered = key.lower()
    forbidden = (
        "raw_iq",
        "received_iq",
        "clean_iq",
        "feature",
        "token",
        "logit",
        "sample",
        "member",
        "physical",
        "receiver",
        "target",
        "label",
        "metric",
        "truth",
        "scorer",
        "query",
    )
    if any(fragment in lowered for fragment in forbidden):
        raise CLICTerminalError(f"forbidden receipt field: {key}")
    if (
        lowered in {"id", "ids"}
        or lowered.startswith(("id_", "ids_"))
        or lowered.endswith(("_id", "_ids"))
    ):
        raise CLICTerminalError(f"forbidden receipt identity field: {key}")
    if lowered == "role" or lowered.startswith("role_") or lowered.endswith("_role"):
        raise CLICTerminalError(f"forbidden receipt role field: {key}")
    if "legacy" in lowered:
        raise CLICTerminalError(f"forbidden receipt legacy field: {key}")


def _validate_clic_receipt_data_free(value: object, *, path: str = "receipt") -> None:
    """Require the receipt tree to contain only scalar/count/enum/SHA payloads."""

    if isinstance(value, torch.Tensor):
        raise CLICTerminalError(f"forbidden tensor payload at {path}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise CLICTerminalError(f"receipt key is invalid at {path}")
            _validate_clic_receipt_field_name(key, child)
            _validate_clic_receipt_data_free(child, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_clic_receipt_data_free(child, path=f"{path}[{index}]")
        return
    if isinstance(value, bool) or isinstance(value, str) or isinstance(value, int):
        return
    if isinstance(value, float):
        if math.isfinite(value):
            return
        raise CLICTerminalError(f"receipt scalar is non-finite at {path}")
    raise CLICTerminalError(f"receipt payload is not scalar/count/enum/SHA at {path}")


def _validate_clic_field_subset(
    value: object,
    *,
    allowed: frozenset[str],
    field: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CLICTerminalError(f"receipt {field} must be a mapping")
    unexpected = set(value) - set(allowed)
    if unexpected:
        raise CLICTerminalError(f"receipt {field} contains unsupported fields")
    return value


def _validate_clic_receipt_structure(receipt: Mapping[str, object]) -> None:
    """Forbid scalar/string side channels outside the frozen receipt grammar."""

    _validate_clic_field_subset(receipt, allowed=_CLIC_RECEIPT_FIELDS, field="root")
    scenarios = frozenset(FORMAL_LEO_WEAK_SCENARIOS)
    scenario_counts = receipt.get("common_scenario_batches")
    if scenario_counts is not None:
        _validate_clic_field_subset(
            scenario_counts,
            allowed=scenarios,
            field="common_scenario_batches",
        )
    audits = receipt.get("scene_audits")
    if audits is not None:
        audit_map = _validate_clic_field_subset(audits, allowed=scenarios, field="scene_audits")
        for scene, audit in audit_map.items():
            audit_map_value = _validate_clic_field_subset(
                audit,
                allowed=_CLIC_SCENE_AUDIT_FIELDS,
                field=f"scene_audits.{scene}",
            )
            for group in ("token", "clic", "base", "head"):
                if group in audit_map_value:
                    _validate_clic_field_subset(
                        audit_map_value[group],
                        allowed=_CLIC_VJP_SUMMARY_FIELDS,
                        field=f"scene_audits.{scene}.{group}",
                    )
            if "clic_groups" in audit_map_value:
                detailed = _validate_clic_field_subset(
                    audit_map_value["clic_groups"],
                    allowed=_CLIC_DETAILED_VJP_GROUPS,
                    field=f"scene_audits.{scene}.clic_groups",
                )
                for group, summary in detailed.items():
                    _validate_clic_field_subset(
                        summary,
                        allowed=_CLIC_VJP_SUMMARY_FIELDS,
                        field=f"scene_audits.{scene}.clic_groups.{group}",
                    )
    for field, allowed in (
        ("common_binding_events", _CLIC_COMMON_BINDING_FIELDS),
        ("resource_observations", _CLIC_RESOURCE_FIELDS),
        ("amp_events", _CLIC_AMP_EVENT_FIELDS),
        ("amp_summary_events", _CLIC_AMP_CORE_FIELDS),
    ):
        events = receipt.get(field)
        if events is None:
            continue
        if not isinstance(events, list):
            raise CLICTerminalError(f"receipt {field} must be a list")
        for index, event in enumerate(events):
            _validate_clic_field_subset(event, allowed=allowed, field=f"{field}[{index}]")


def _copy_clic_receipt(receipt: Mapping[str, object]) -> Dict[str, Any]:
    if not isinstance(receipt, Mapping):
        raise CLICTerminalError("CLIC receipt must be a mapping")
    _validate_clic_receipt_data_free(receipt)
    _validate_clic_receipt_structure(receipt)
    return dict(receipt)


def new_clic_receipt(*, arm: str) -> Dict[str, Any]:
    """Create the data-free source-L receipt for either active CLIC arm."""

    arm_name = str(arm)
    if arm_name not in _CLIC_OPERATOR_IDS:
        raise CLICTerminalError("CLIC arm must be C or G")
    receipt: Dict[str, Any] = {
        "schema": _CLIC_RECEIPT_SCHEMA,
        "method": _CLIC_METHOD,
        "arm": arm_name,
        "operator_id": _CLIC_OPERATOR_IDS[arm_name],
        "batch_size": 128,
        "input_length": CLIC_INPUT_LENGTH,
        "embed_dim": CLIC_EMBED_DIM,
        "local_class_count": 4,
        "lags": [1, 2, 4, 8],
        "clip": 8,
        "clic_active": True,
        "new_adamw": True,
        "checkpoint_sha256": "",
        "final_checkpoint_sha256": "",
        "existing_state_sha256": "",
        "clic_init_state_sha256": "",
        "clic_parameter_count": CLIC_EXTRA_PARAMETER_COUNT,
        "existing_state_unchanged": False,
        "optimizer_state_restored": False,
        "rng_state_restored": False,
        "source_l_only": True,
        "source_split_count": 0,
        "source_split_sha256": "",
        "class_order_count": 0,
        "class_order_sha256": "",
        "physical_order_count": 0,
        "physical_order_sha256": "",
        "common_batch_sequence_sha256": "",
        "common_batch_sequence_batches": 0,
        "common_batch_sequence_rows": 0,
        "common_scenario_batches": {scene: 0 for scene in FORMAL_LEO_WEAK_SCENARIOS},
        "common_binding_events": [],
        "scene_audits": {},
        "resource_observations": [],
        "amp_events": [],
        "amp_summary_events": [],
        "amp_attempts": 0,
        "scaled_backward_count": 0,
        "unscale_count": 0,
        "optimizer_step_attempts": 0,
        "effective_optimizer_steps": 0,
        "raw_finite_overflow_skips": 0,
        "scale_decrease_count": 0,
        "optimizer_unchanged_count": 0,
        "raw_nonfinite_count": 0,
        "material_nonfinite_count": 0,
        "consecutive_overflow_skips": 0,
        "max_consecutive_overflow_skips": 0,
        "persistent_overflow": False,
        "graph_release_count": 0,
        "head_path": "id_backbone.cls_head.head",
        "completed": False,
        **_CLIC_ZERO_ONLY_FIELDS,
    }
    return receipt


def _common_binding_sequence_sha256(events: Sequence[Mapping[str, object]]) -> str:
    """Return the deterministic rolling aggregate for scalar common-batch seals."""

    normalized = [dict(event) for event in events]
    encoded = json.dumps(
        normalized,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(b"cvs.phase1.clic.common_batch_sequence.v1\0" + encoded).hexdigest()


def _binding_scalar_or_sha(
    binding: Mapping[str, object],
    *,
    field: str,
    sha: bool,
) -> object:
    if field not in binding:
        raise CLICTerminalError(f"common binding lacks {field}")
    value = binding[field]
    if sha:
        if not _is_lowercase_sha256(value):
            raise CLICTerminalError(f"common binding {field} is not SHA-256")
        return str(value)
    return _strict_positive_int(value, field=f"common binding {field}")


def update_clic_common_binding_receipt(
    receipt: Mapping[str, object],
    *,
    binding: Mapping[str, object],
) -> Dict[str, Any]:
    """Append one scalar-only source-L common batch to the sealed sequence."""

    result = _copy_clic_receipt(receipt)
    if not isinstance(binding, Mapping):
        raise CLICTerminalError("common binding must be a mapping")
    allowed = {
        "scene",
        "batch_index",
        "rows",
        "source_split_count",
        "source_split_sha256",
        "class_order_count",
        "class_order_sha256",
        "physical_order_count",
        "physical_order_sha256",
        "common_batch_sequence_sha256",
    }
    if set(binding) != allowed:
        raise CLICTerminalError("common binding fields drift from frozen scalar contract")
    scene = str(binding["scene"])
    if scene not in FORMAL_LEO_WEAK_SCENARIOS:
        raise CLICTerminalError("common binding scene is not formal LEO weak")
    batch_index = _strict_positive_int(binding["batch_index"], field="common binding batch_index")
    rows = _strict_positive_int(binding["rows"], field="common binding rows")
    if rows != 128:
        raise CLICTerminalError("common binding rows must equal frozen batch size 128")

    event: Dict[str, object] = {"scene": scene, "batch_index": batch_index, "rows": rows}
    for count_field, sha_field in (
        ("source_split_count", "source_split_sha256"),
        ("class_order_count", "class_order_sha256"),
        ("physical_order_count", "physical_order_sha256"),
    ):
        count = _binding_scalar_or_sha(binding, field=count_field, sha=False)
        digest = _binding_scalar_or_sha(binding, field=sha_field, sha=True)
        present_count = result.get(count_field, 0)
        present_sha = result.get(sha_field, "")
        if present_count not in (0, count) or present_sha not in ("", digest):
            raise CLICTerminalError(f"common binding {count_field} or {sha_field} drifted")
        result[count_field] = count
        result[sha_field] = digest
        event[count_field] = count
        event[sha_field] = digest
    event["common_batch_sequence_sha256"] = _binding_scalar_or_sha(
        binding,
        field="common_batch_sequence_sha256",
        sha=True,
    )

    events = [dict(value) for value in result.get("common_binding_events", [])]
    if any(int(value.get("batch_index", -1)) == batch_index for value in events):
        raise CLICTerminalError("common binding batch_index is duplicated")
    events.append(event)
    result["common_binding_events"] = events
    result["common_batch_sequence_batches"] = len(events)
    result["common_batch_sequence_rows"] = sum(int(value["rows"]) for value in events)
    scenario_counts = {scene_name: 0 for scene_name in FORMAL_LEO_WEAK_SCENARIOS}
    for value in events:
        scenario_counts[str(value["scene"])] += 1
    result["common_scenario_batches"] = scenario_counts
    result["common_batch_sequence_sha256"] = _common_binding_sequence_sha256(events)
    return result


def update_clic_resource_receipt(
    receipt: Mapping[str, object],
    *,
    observation: Mapping[str, object],
) -> Dict[str, Any]:
    """Record exactly one scalar resource observation for one common batch."""

    result = _copy_clic_receipt(receipt)
    if not isinstance(observation, Mapping):
        raise CLICTerminalError("resource observation must be a mapping")
    allowed = {"scene", "batch_index", "peak_memory_bytes", "step_time_seconds", "selection_feedback"}
    if set(observation) != allowed:
        raise CLICTerminalError("resource observation fields drift from frozen contract")
    scene = str(observation["scene"])
    if scene not in FORMAL_LEO_WEAK_SCENARIOS:
        raise CLICTerminalError("resource observation scene is invalid")
    batch_index = _strict_positive_int(observation["batch_index"], field="resource batch_index")
    peak = _strict_nonnegative_int(observation["peak_memory_bytes"], field="resource peak_memory_bytes")
    step = _strict_finite_nonnegative_float(observation["step_time_seconds"], field="resource step_time_seconds")
    if observation["selection_feedback"] is not False:
        raise CLICTerminalError("resource selection_feedback must be false")
    observations = [dict(value) for value in result.get("resource_observations", [])]
    identity = (scene, batch_index)
    if any((str(value.get("scene", "")), int(value.get("batch_index", -1))) == identity for value in observations):
        raise CLICTerminalError("resource observation duplicates a common batch")
    observations.append(
        {
            "scene": scene,
            "batch_index": batch_index,
            "peak_memory_bytes": peak,
            "step_time_seconds": step,
            "selection_feedback": False,
        }
    )
    result["resource_observations"] = observations
    return result


def _validate_amp_event(
    event: Mapping[str, object],
    *,
    require_identity: bool = True,
) -> Dict[str, object]:
    core_required = {
        "amp_overflow_detected",
        "scaled_backward_count",
        "unscale_count",
        "optimizer_step_attempted",
        "effective_optimizer_step",
        "raw_finite",
        "scale_decreased",
        "optimizer_state_unchanged",
        "raw_nonfinite",
        "material_nonfinite",
    }
    required = core_required | ({"scene", "batch_index"} if require_identity else set())
    if set(event) != required:
        raise CLICTerminalError("AMP event fields drift from frozen contract")
    normalized: Dict[str, object] = {}
    if require_identity:
        scene = str(event["scene"])
        if scene not in FORMAL_LEO_WEAK_SCENARIOS:
            raise CLICTerminalError("AMP event scene is invalid")
        normalized["scene"] = scene
        normalized["batch_index"] = _strict_positive_int(event["batch_index"], field="AMP batch_index")
    for key in (
        "amp_overflow_detected",
        "optimizer_step_attempted",
        "effective_optimizer_step",
        "raw_finite",
        "scale_decreased",
        "optimizer_state_unchanged",
        "raw_nonfinite",
        "material_nonfinite",
    ):
        if not isinstance(event[key], bool):
            raise CLICTerminalError(f"AMP {key} must be boolean")
    scaled = _strict_nonnegative_int(event["scaled_backward_count"], field="AMP scaled_backward_count")
    unscale = _strict_nonnegative_int(event["unscale_count"], field="AMP unscale_count")
    if scaled != 1 or unscale != 1:
        raise CLICTerminalError("AMP requires exactly one scaled backward and one unscale")
    if event["raw_nonfinite"] or event["material_nonfinite"] or not event["raw_finite"]:
        raise CLICRuntimeError("CLIC raw/material non-finite AMP event is fatal")
    overflow = bool(event["amp_overflow_detected"])
    if overflow:
        if (
            event["effective_optimizer_step"]
            or not event["optimizer_step_attempted"]
            or not event["scale_decreased"]
            or not event["optimizer_state_unchanged"]
        ):
            raise CLICTerminalError("AMP overflow skip/backoff contract drifted")
    elif (
        not event["effective_optimizer_step"]
        or not event["optimizer_step_attempted"]
        or event["scale_decreased"]
        or event["optimizer_state_unchanged"]
    ):
        raise CLICTerminalError("finite AMP step contract drifted")
    normalized.update(
        {
        "amp_overflow_detected": overflow,
        "scaled_backward_count": scaled,
        "unscale_count": unscale,
        "optimizer_step_attempted": bool(event["optimizer_step_attempted"]),
        "effective_optimizer_step": bool(event["effective_optimizer_step"]),
        "raw_finite": True,
        "scale_decreased": bool(event["scale_decreased"]),
        "optimizer_state_unchanged": bool(event["optimizer_state_unchanged"]),
        "raw_nonfinite": False,
        "material_nonfinite": False,
        }
    )
    return normalized


def _set_amp_counters(result: Dict[str, Any], events: Sequence[Mapping[str, object]]) -> None:
    attempts = len(events)
    consecutive = 0
    maximum = 0
    for event in events:
        if bool(event["amp_overflow_detected"]) and not bool(event["effective_optimizer_step"]):
            consecutive += 1
            maximum = max(maximum, consecutive)
        else:
            consecutive = 0
    result["amp_attempts"] = attempts
    result["scaled_backward_count"] = sum(int(event["scaled_backward_count"]) for event in events)
    result["unscale_count"] = sum(int(event["unscale_count"]) for event in events)
    result["optimizer_step_attempts"] = sum(bool(event["optimizer_step_attempted"]) for event in events)
    result["effective_optimizer_steps"] = sum(bool(event["effective_optimizer_step"]) for event in events)
    result["raw_finite_overflow_skips"] = sum(bool(event["amp_overflow_detected"]) for event in events)
    result["scale_decrease_count"] = sum(bool(event["scale_decreased"]) for event in events)
    result["optimizer_unchanged_count"] = sum(bool(event["optimizer_state_unchanged"]) for event in events)
    result["raw_nonfinite_count"] = 0
    result["material_nonfinite_count"] = 0
    result["consecutive_overflow_skips"] = consecutive
    result["max_consecutive_overflow_skips"] = maximum
    result["persistent_overflow"] = maximum >= 2


def update_clic_amp_receipt(
    receipt: Mapping[str, object],
    *,
    event: Mapping[str, object],
) -> Dict[str, Any]:
    """Append one already-classified one-backward/one-unscale AMP event."""

    result = _copy_clic_receipt(receipt)
    if not isinstance(event, Mapping):
        raise CLICTerminalError("AMP event must be a mapping")
    events = [dict(value) for value in result.get("amp_events", [])]
    summaries = [dict(value) for value in result.get("amp_summary_events", [])]
    if "scene" in event or "batch_index" in event:
        normalized = _validate_amp_event(event, require_identity=True)
        identity = (str(normalized["scene"]), int(normalized["batch_index"]))
        if any((str(value.get("scene", "")), int(value.get("batch_index", -1))) == identity for value in events):
            raise CLICTerminalError("AMP event duplicates a common batch")
        events.append(normalized)
    else:
        normalized = _validate_amp_event(event, require_identity=False)
        summaries.append(normalized)
    result["amp_events"] = events
    result["amp_summary_events"] = summaries
    _set_amp_counters(result, [*events, *summaries])
    return result


def _validate_vjp_group(values: object, *, group: str) -> None:
    if not isinstance(values, Mapping):
        raise CLICTerminalError(f"scene VJP audit lacks {group}")
    count = _strict_positive_int(values.get("count"), field=f"{group} VJP count")
    norm = _strict_finite_nonnegative_float(values.get("norm"), field=f"{group} VJP norm")
    if norm <= 0.0 or values.get("finite") is not True or values.get("nonzero") is not True:
        raise CLICTerminalError(f"scene {group} VJP audit is zero or non-finite")
    if count <= 0:
        raise CLICTerminalError(f"scene {group} VJP audit is incomplete")


def _validate_scene_audit(scene: str, audit: object) -> None:
    if not isinstance(audit, Mapping):
        raise CLICTerminalError(f"scene audit is absent: {scene}")
    coverage = _strict_finite_nonnegative_float(
        audit.get("valid_token_coverage"),
        field=f"scene {scene} valid-token coverage",
    )
    if coverage <= 0.0 or audit.get("gate_or_correction_nonzero") is not True:
        raise CLICTerminalError(f"scene {scene} token/gate coverage is incomplete")
    if (
        audit.get("raw_unscaled") is not True
        or audit.get("diagnostic_only") is not True
        or audit.get("touches_amp_optimizer_rng") is not False
        or audit.get("completed") is not True
    ):
        raise CLICTerminalError(f"scene {scene} raw-unscaled VJP audit semantics drifted")
    for group in ("token", "clic", "base", "head"):
        _validate_vjp_group(audit.get(group), group=group)
    detailed_groups = audit.get("clic_groups")
    if not isinstance(detailed_groups, Mapping) or set(detailed_groups) != {
        "depthwise",
        "pointwise",
        "embed",
        "correction",
        "gate",
    }:
        raise CLICTerminalError(f"scene {scene} CLIC VJP group coverage drifted")
    for group in ("depthwise", "pointwise", "embed", "correction", "gate"):
        _validate_vjp_group(detailed_groups[group], group=f"{scene}.{group}")


def _validate_common_binding_terminal(receipt: Mapping[str, object]) -> int:
    batches = _strict_positive_int(
        receipt.get("common_batch_sequence_batches"),
        field="common batch sequence count",
    )
    rows = _strict_positive_int(
        receipt.get("common_batch_sequence_rows"),
        field="common batch sequence rows",
    )
    if rows != batches * 128:
        raise CLICTerminalError("common batch sequence rows do not close at B=128")
    events = receipt.get("common_binding_events")
    if not isinstance(events, list) or len(events) != batches:
        raise CLICTerminalError("common binding event count does not close")
    normalized_events = []
    expected_source = {
        "source_split_count": _strict_positive_int(receipt.get("source_split_count"), field="source binding count"),
        "source_split_sha256": _require_receipt_sha256(receipt, "source_split_sha256"),
        "class_order_count": _strict_positive_int(receipt.get("class_order_count"), field="class binding count"),
        "class_order_sha256": _require_receipt_sha256(receipt, "class_order_sha256"),
        "physical_order_count": _strict_positive_int(receipt.get("physical_order_count"), field="physical binding count"),
        "physical_order_sha256": _require_receipt_sha256(receipt, "physical_order_sha256"),
    }
    seen_indexes = set()
    scenario_counts = {scene: 0 for scene in FORMAL_LEO_WEAK_SCENARIOS}
    for raw_event in events:
        if not isinstance(raw_event, Mapping):
            raise CLICTerminalError("common binding event is malformed")
        required = {
            "scene",
            "batch_index",
            "rows",
            "source_split_count",
            "source_split_sha256",
            "class_order_count",
            "class_order_sha256",
            "physical_order_count",
            "physical_order_sha256",
            "common_batch_sequence_sha256",
        }
        if set(raw_event) != required:
            raise CLICTerminalError("common binding event fields drifted")
        scene = str(raw_event["scene"])
        if scene not in scenario_counts:
            raise CLICTerminalError("common binding scene drifted")
        index = _strict_positive_int(raw_event["batch_index"], field="common binding batch_index")
        if index in seen_indexes:
            raise CLICTerminalError("common binding batch_index duplicated")
        seen_indexes.add(index)
        if _strict_positive_int(raw_event["rows"], field="common binding rows") != 128:
            raise CLICTerminalError("common binding rows drifted")
        for field, expected in expected_source.items():
            if raw_event.get(field) != expected:
                raise CLICTerminalError(f"common binding {field} drifted")
        if not _is_lowercase_sha256(raw_event.get("common_batch_sequence_sha256")):
            raise CLICTerminalError("common binding sequence SHA is invalid")
        scenario_counts[scene] += 1
        normalized_events.append(dict(raw_event))
    receipt_counts = receipt.get("common_scenario_batches")
    if not isinstance(receipt_counts, Mapping) or set(receipt_counts) != set(FORMAL_LEO_WEAK_SCENARIOS):
        raise CLICTerminalError("common scenario batch counts are incomplete")
    if any(_strict_positive_int(receipt_counts[scene], field=f"common scenario {scene} count") != scenario_counts[scene] for scene in FORMAL_LEO_WEAK_SCENARIOS):
        raise CLICTerminalError("common scenario batch counts do not close")
    if sum(scenario_counts.values()) != batches:
        raise CLICTerminalError("common scenario count does not close")
    if receipt.get("common_batch_sequence_sha256") != _common_binding_sequence_sha256(normalized_events):
        raise CLICTerminalError("common batch sequence SHA aggregate drifted")
    return batches


def _validate_resource_terminal(receipt: Mapping[str, object], *, expected_count: int) -> None:
    observations = receipt.get("resource_observations")
    if not isinstance(observations, list) or len(observations) != expected_count:
        raise CLICTerminalError("resource observation count does not close")
    seen = set()
    for observation in observations:
        if not isinstance(observation, Mapping):
            raise CLICTerminalError("resource observation is malformed")
        if set(observation) != {"scene", "batch_index", "peak_memory_bytes", "step_time_seconds", "selection_feedback"}:
            raise CLICTerminalError("resource observation fields drifted")
        scene = str(observation["scene"])
        index = _strict_positive_int(observation["batch_index"], field="resource batch_index")
        if scene not in FORMAL_LEO_WEAK_SCENARIOS or (scene, index) in seen:
            raise CLICTerminalError("resource observation identity drifted")
        seen.add((scene, index))
        _strict_nonnegative_int(observation["peak_memory_bytes"], field="resource peak_memory_bytes")
        _strict_finite_nonnegative_float(observation["step_time_seconds"], field="resource step_time_seconds")
        if observation["selection_feedback"] is not False:
            raise CLICTerminalError("resource selection_feedback must be false")
    binding_events = receipt.get("common_binding_events", [])
    expected_identities = {
        (str(event.get("scene", "")), int(event.get("batch_index", -1)))
        for event in binding_events
        if isinstance(event, Mapping)
    }
    if seen != expected_identities:
        raise CLICTerminalError("resource observations do not match common batches")


def _validate_amp_terminal(receipt: Mapping[str, object], *, expected_count: int) -> None:
    if receipt.get("amp_summary_events", []) != []:
        raise CLICTerminalError("AMP terminal lacks per-batch identity closure")
    events = receipt.get("amp_events")
    if not isinstance(events, list) or len(events) != expected_count:
        raise CLICTerminalError("AMP attempt count does not close")
    normalized = [_validate_amp_event(event) for event in events]
    event_identities = {
        (str(event["scene"]), int(event["batch_index"]))
        for event in normalized
    }
    binding_events = receipt.get("common_binding_events", [])
    expected_identities = {
        (str(event.get("scene", "")), int(event.get("batch_index", -1)))
        for event in binding_events
        if isinstance(event, Mapping)
    }
    if event_identities != expected_identities:
        raise CLICTerminalError("AMP events do not match common batches")
    expected: Dict[str, object] = {}
    _set_amp_counters(expected, normalized)
    for field in (
        "amp_attempts",
        "scaled_backward_count",
        "unscale_count",
        "optimizer_step_attempts",
        "effective_optimizer_steps",
        "raw_finite_overflow_skips",
        "scale_decrease_count",
        "optimizer_unchanged_count",
        "raw_nonfinite_count",
        "material_nonfinite_count",
        "consecutive_overflow_skips",
        "max_consecutive_overflow_skips",
        "persistent_overflow",
    ):
        if receipt.get(field) != expected[field]:
            raise CLICTerminalError(f"AMP terminal {field} does not close")
    if int(expected["effective_optimizer_steps"]) <= 0:
        raise CLICTerminalError("effective optimizer steps are zero")
    if (
        expected["persistent_overflow"] is True
        or int(expected["consecutive_overflow_skips"]) > 1
        or int(expected["max_consecutive_overflow_skips"]) > 1
    ):
        raise CLICTerminalError("persistent consecutive AMP overflow is terminal")


def validate_clic_terminal_receipt(
    receipt: Mapping[str, object],
    *,
    arm: str,
) -> Dict[str, Any]:
    """Revalidate every scalar CLIC contract before accepting terminal closure."""

    result = _copy_clic_receipt(receipt)
    arm_name = str(arm)
    if arm_name not in _CLIC_OPERATOR_IDS or result.get("arm") != arm_name:
        raise CLICTerminalError("terminal arm identity drifted")
    if result.get("schema") != _CLIC_RECEIPT_SCHEMA or result.get("method") != _CLIC_METHOD:
        raise CLICTerminalError("terminal receipt schema or method drifted")
    if result.get("operator_id") != _CLIC_OPERATOR_IDS[arm_name]:
        raise CLICTerminalError("terminal C/G operator identity drifted")
    if (
        result.get("batch_size") != 128
        or result.get("input_length") != CLIC_INPUT_LENGTH
        or result.get("embed_dim") != CLIC_EMBED_DIM
        or result.get("local_class_count") != 4
        or result.get("lags") != [1, 2, 4, 8]
        or result.get("clip") != 8
    ):
        raise CLICTerminalError("terminal frozen CLIC configuration drifted")
    if result.get("clic_active") is not True or result.get("new_adamw") is not True:
        raise CLICTerminalError("terminal active CLIC or new AdamW contract drifted")
    if result.get("source_l_only") is not True:
        raise CLICTerminalError("terminal source-L-only contract drifted")
    for field, expected in _CLIC_ZERO_ONLY_FIELDS.items():
        if result.get(field) != expected:
            raise CLICTerminalError(f"terminal forbidden source/target feedback field: {field}")
    for field in (
        "checkpoint_sha256",
        "final_checkpoint_sha256",
        "existing_state_sha256",
        "clic_init_state_sha256",
    ):
        _require_receipt_sha256(result, field)
    if (
        result.get("clic_parameter_count") != CLIC_EXTRA_PARAMETER_COUNT
        or result.get("existing_state_unchanged") is not True
        or result.get("optimizer_state_restored") is not False
        or result.get("rng_state_restored") is not False
        or result.get("head_path") != "id_backbone.cls_head.head"
    ):
        raise CLICTerminalError("terminal warm-start or exact-head contract drifted")
    batches = _validate_common_binding_terminal(result)
    _validate_resource_terminal(result, expected_count=batches)
    _validate_amp_terminal(result, expected_count=batches)
    if _strict_nonnegative_int(result.get("graph_release_count"), field="graph release count") != batches:
        raise CLICTerminalError("graph release count does not close")
    audits = result.get("scene_audits")
    if not isinstance(audits, Mapping) or set(audits) != set(FORMAL_LEO_WEAK_SCENARIOS):
        raise CLICTerminalError("terminal scene audit coverage is incomplete")
    for scene in FORMAL_LEO_WEAK_SCENARIOS:
        _validate_scene_audit(scene, audits[scene])
    if result.get("completed") is not True:
        raise CLICTerminalError("terminal receipt is not completed")
    result["terminal_contract"] = "STRICT_CLIC_SOURCE_L_COMMON_C_G_RAW_UNSCALED_VJP_AMP_RESOURCE_GRAPH_RELEASE"
    result["terminal_contract_passed"] = True
    return result


def validate_clic_terminal_envelope(envelope: Mapping[str, object]) -> Dict[str, Any]:
    """Validate the external checkpoint binding without extending the strict core.

    The terminal core deliberately stays within the Task4 receipt grammar so a
    later postfreeze reader can revalidate it unchanged.  The selected file
    location and immutable file digest belong only to this small versioned
    envelope, whose digest must agree with the validated core final digest.
    """

    if not isinstance(envelope, Mapping):
        raise CLICTerminalError("CLIC terminal envelope must be a mapping")
    if set(envelope) != _CLIC_TERMINAL_ENVELOPE_FIELDS:
        raise CLICTerminalError("CLIC terminal envelope fields drifted")
    _validate_clic_receipt_data_free(envelope, path="terminal_envelope")
    if envelope.get("schema") != _CLIC_TERMINAL_ENVELOPE_SCHEMA:
        raise CLICTerminalError("CLIC terminal envelope schema drifted")
    if envelope.get("method") != _CLIC_METHOD:
        raise CLICTerminalError("CLIC terminal envelope method drifted")
    selected_checkpoint_path = envelope.get("selected_checkpoint_path")
    if not isinstance(selected_checkpoint_path, str) or not selected_checkpoint_path.strip():
        raise CLICTerminalError("CLIC terminal envelope checkpoint path is absent")
    selected_checkpoint_sha256 = envelope.get("selected_checkpoint_sha256")
    if not _is_lowercase_sha256(selected_checkpoint_sha256):
        raise CLICTerminalError("CLIC terminal envelope checkpoint SHA is invalid")
    strict_core = envelope.get("strict_core")
    if not isinstance(strict_core, Mapping):
        raise CLICTerminalError("CLIC terminal envelope strict core is absent")
    validated_core = validate_clic_terminal_receipt(
        strict_core,
        arm=str(strict_core.get("arm", "")),
    )
    if validated_core.get("final_checkpoint_sha256") != selected_checkpoint_sha256:
        raise CLICTerminalError("CLIC terminal envelope checkpoint SHA does not bind strict core")
    return {
        "schema": _CLIC_TERMINAL_ENVELOPE_SCHEMA,
        "method": _CLIC_METHOD,
        "strict_core": validated_core,
        "selected_checkpoint_path": selected_checkpoint_path,
        "selected_checkpoint_sha256": str(selected_checkpoint_sha256),
    }


def _clic_gradient_summary(gradients: Sequence[Optional[torch.Tensor]]) -> Dict[str, object]:
    """Reduce a gradient group to data-free finite/nonzero scalar evidence."""

    expected = len(gradients)
    present = [gradient for gradient in gradients if isinstance(gradient, torch.Tensor)]
    finite = len(present) == expected and all(
        bool(torch.isfinite(gradient.detach()).all().item()) for gradient in present
    )
    norm_squared = 0.0
    if finite:
        for gradient in present:
            norm_squared += float(gradient.detach().double().square().sum().item())
    norm = math.sqrt(norm_squared) if math.isfinite(norm_squared) and norm_squared >= 0.0 else float("nan")
    return {
        "count": len(present),
        "norm": norm,
        "finite": finite,
        "nonzero": bool(finite and math.isfinite(norm) and norm > 0.0),
    }


def _clic_named_gradient_groups(
    module: nn.Module,
) -> Tuple[Tuple[nn.Parameter, ...], Dict[str, Tuple[int, ...]]]:
    """Bind every CLIC parameter to the five frozen gradient evidence groups."""

    _validate_fusion_architecture(module)
    parameters: list[nn.Parameter] = []
    groups: Dict[str, list[int]] = {
        "depthwise": [],
        "pointwise": [],
        "embed": [],
        "correction": [],
        "gate": [],
    }
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            raise CLICRuntimeError(f"CLIC VJP parameter is not trainable: {name}")
        if name.startswith(("depthwise.", "gn1.")):
            group = "depthwise"
        elif name.startswith(("pointwise.", "gn2.")):
            group = "pointwise"
        elif name.startswith("embed."):
            group = "embed"
        elif name.startswith("correction."):
            group = "correction"
        elif name.startswith(("gate_norm.", "gate.")):
            group = "gate"
        else:
            raise CLICRuntimeError("CLIC VJP parameter group is unknown")
        groups[group].append(len(parameters))
        parameters.append(parameter)
    if not parameters or any(not groups[name] for name in groups):
        raise CLICRuntimeError("CLIC VJP parameter group coverage is incomplete")
    return tuple(parameters), {name: tuple(indexes) for name, indexes in groups.items()}


def _require_clic_nonzero_gradient_summary(summary: Mapping[str, object], *, group: str) -> None:
    try:
        count = int(summary["count"])
        norm = float(summary["norm"])
    except (KeyError, TypeError, ValueError) as error:
        raise CLICRuntimeError(f"CLIC VJP summary is malformed: {group}") from error
    if (
        count <= 0
        or not math.isfinite(norm)
        or norm <= 0.0
        or summary.get("finite") is not True
        or summary.get("nonzero") is not True
    ):
        raise CLICRuntimeError(f"CLIC VJP is zero or non-finite: {group}")


def clic_raw_unscaled_vjp_audit(
    loss: torch.Tensor,
    token_tensor: torch.Tensor,
    clic_module: nn.Module,
    base_parameters: Iterable[nn.Parameter],
    head_weight: nn.Parameter,
) -> Dict[str, Any]:
    """Audit one shared raw-unscaled graph without touching AMP or optimizer state."""

    if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
        raise CLICRuntimeError("CLIC raw-unscaled VJP requires one scalar loss")
    _require_all_finite(loss.detach(), name="raw-unscaled CLIC loss")
    if not isinstance(token_tensor, torch.Tensor) or not token_tensor.requires_grad:
        raise CLICRuntimeError("CLIC raw-unscaled VJP token tensor is not differentiable")
    clic_parameters, clic_group_indexes = _clic_named_gradient_groups(clic_module)
    base = tuple(base_parameters)
    if not base or any(not isinstance(parameter, nn.Parameter) or not parameter.requires_grad for parameter in base):
        raise CLICRuntimeError("CLIC raw-unscaled VJP base parameter binding is invalid")
    if not isinstance(head_weight, nn.Parameter) or not head_weight.requires_grad:
        raise CLICRuntimeError("CLIC raw-unscaled VJP exact head weight is invalid")
    targets = (token_tensor,) + clic_parameters + base + (head_weight,)
    if len({id(target) for target in targets}) != len(targets):
        raise CLICRuntimeError("CLIC raw-unscaled VJP target binding overlaps")
    try:
        gradients = torch.autograd.grad(
            loss,
            targets,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
    except RuntimeError as error:
        raise CLICRuntimeError("CLIC raw-unscaled VJP execution failed") from error
    token_gradients = gradients[:1]
    clic_gradients = gradients[1 : 1 + len(clic_parameters)]
    base_start = 1 + len(clic_parameters)
    base_gradients = gradients[base_start:-1]
    head_gradients = gradients[-1:]
    clic_group_summaries = {
        name: _clic_gradient_summary(tuple(clic_gradients[index] for index in indexes))
        for name, indexes in clic_group_indexes.items()
    }
    audit: Dict[str, Any] = {
        "raw_unscaled": True,
        "diagnostic_only": True,
        "touches_amp_optimizer_rng": False,
        "token": _clic_gradient_summary(token_gradients),
        "clic": _clic_gradient_summary(clic_gradients),
        "base": _clic_gradient_summary(base_gradients),
        "head": _clic_gradient_summary(head_gradients),
        "clic_groups": clic_group_summaries,
    }
    for group in ("token", "clic", "base", "head"):
        _require_clic_nonzero_gradient_summary(audit[group], group=group)
    for group, summary in clic_group_summaries.items():
        _require_clic_nonzero_gradient_summary(summary, group=f"clic.{group}")
    return audit


def _clic_trainable_parameter_binding(model: nn.Module) -> Tuple[nn.Parameter, ...]:
    if not isinstance(model, nn.Module):
        raise CLICRuntimeError("CLIC AMP model is invalid")
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    if not parameters:
        raise CLICRuntimeError("CLIC AMP model has no trainable parameters")
    return parameters


def _clic_require_finite_scalar_loss(loss: torch.Tensor) -> None:
    if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
        raise CLICRuntimeError("CLIC AMP loss must be one scalar tensor")
    if not bool(torch.isfinite(loss.detach()).all().item()):
        raise CLICRuntimeError("CLIC AMP loss is non-finite")


def clic_scaled_backward_and_classify(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    loss: torch.Tensor,
) -> Dict[str, Any]:
    """Run the one permitted scaled backward/unscale and classify overflow only.

    The caller owns the ordinary single scaler step/update afterwards.  A
    scaled overflow is recoverable only when its affected raw material
    gradients are finite on this same retained graph.
    """

    _clic_require_finite_scalar_loss(loss)
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise CLICRuntimeError("CLIC AMP optimizer is invalid")
    try:
        captured_scale = float(scaler.get_scale())
    except (AttributeError, TypeError, ValueError) as error:
        raise CLICRuntimeError("CLIC GradScaler scale is unavailable") from error
    if not math.isfinite(captured_scale) or captured_scale <= 0.0:
        raise CLICRuntimeError("CLIC GradScaler scale is invalid")
    parameters = _clic_trainable_parameter_binding(model)
    try:
        scaler.scale(loss).backward(retain_graph=True)
        scaler.unscale_(optimizer)
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise CLICRuntimeError("CLIC AMP backward or unscale failed") from error
    affected_indexes = tuple(
        index
        for index, parameter in enumerate(parameters)
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad.detach()).all().item())
    )
    if not affected_indexes:
        return {
            "amp_overflow_detected": False,
            "amp_overflow_recoverable": False,
            "captured_scale": captured_scale,
            "scaled_backward_count": 1,
            "optimizer_unscale_count": 1,
            "raw_finite": True,
            "raw_nonfinite": False,
            "material_nonfinite": False,
        }
    try:
        raw_gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
    except RuntimeError as error:
        raise CLICRuntimeError("CLIC raw material gradient audit failed") from error
    raw_finite = all(
        isinstance(raw_gradients[index], torch.Tensor)
        and bool(torch.isfinite(raw_gradients[index].detach()).all().item())
        for index in affected_indexes
    )
    if not raw_finite:
        raise CLICRuntimeError("CLIC raw/material non-finite gradient is fatal")
    return {
        "amp_overflow_detected": True,
        "amp_overflow_recoverable": True,
        "amp_overflow_kind": "COMBINED_SCALED_OVERFLOW_RAW_FINITE",
        "captured_scale": captured_scale,
        "scaled_backward_count": 1,
        "optimizer_unscale_count": 1,
        "raw_finite": True,
        "raw_nonfinite": False,
        "material_nonfinite": False,
        "scaled_nonfinite_parameter_count": len(affected_indexes),
    }


def release_clic_retained_graph_roots(roots: Dict[str, object]) -> None:
    """Release the caller-owned graph roots with one dedicated mapping clear."""

    if not isinstance(roots, dict):
        raise CLICRuntimeError("CLIC retained graph roots must be a dictionary")
    roots.clear()


def _clic_failure_receipt_projection(value: object) -> object:
    """Remove only already-proved zero access markers from a failure receipt."""

    if isinstance(value, Mapping):
        projected: Dict[str, object] = {}
        for key, child in value.items():
            lowered = str(key).lower()
            if key in _CLIC_ZERO_ONLY_FIELDS and ("target" in lowered or "query" in lowered):
                continue
            projected[str(key)] = _clic_failure_receipt_projection(child)
        return projected
    if isinstance(value, list):
        return [_clic_failure_receipt_projection(child) for child in value]
    if isinstance(value, tuple):
        return [_clic_failure_receipt_projection(child) for child in value]
    return value


def write_clic_failure_receipt(
    output_dir: str | Path,
    *,
    candidate_id: str,
    run_id: str,
    receipt: Mapping[str, object],
    error: BaseException,
    failure_stage: str,
) -> Path:
    """Atomically persist a data-free partial receipt for one terminal failure."""

    _validate_clic_receipt_data_free(receipt)
    _validate_clic_receipt_structure(receipt)
    safe_receipt = _clic_failure_receipt_projection(receipt)
    _validate_clic_receipt_data_free(safe_receipt)
    _validate_clic_receipt_structure(safe_receipt)
    target_dir = Path(output_dir)
    if not target_dir.is_dir():
        raise CLICRuntimeError("CLIC failure receipt output directory is absent")
    payload = {
        "schema": "cvs.phase1.clic_failure_receipt.v1",
        "candidate_id": str(candidate_id or ""),
        "run_id": str(run_id or ""),
        "failure_stage": str(failure_stage or ""),
        "exception_type": type(error).__name__,
        "message_digest": hashlib.sha256(str(error).encode("utf-8")).hexdigest(),
        "receipt": safe_receipt,
    }
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    target = target_dir / "phase1_clic_failure_receipt.json"
    descriptor, temporary_name = mkstemp(prefix=".clic_failure_receipt.", suffix=".tmp", dir=str(target_dir))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


__all__ = [
    "CLICConfig",
    "CLICConfigError",
    "CLICForwardResult",
    "CLICFusion",
    "CLICRuntimeError",
    "CLICTerminalError",
    "CLICTokenBatch",
    "CLICWarmStartError",
    "CLIC_EMBED_DIM",
    "CLIC_EXTRA_PARAMETER_COUNT",
    "CLIC_INIT_SEED",
    "CLIC_INPUT_LENGTH",
    "CLIC_LAGS",
    "FORMAL_LEO_WEAK_SCENARIOS",
    "FROZEN_FOLDS",
    "clic_raw_unscaled_vjp_audit",
    "clic_scaled_backward_and_classify",
    "clic_state_sha256",
    "initialize_clic_module_",
    "new_clic_receipt",
    "release_clic_retained_graph_roots",
    "strict_clic_warm_start",
    "totalized_clic_tokens",
    "update_clic_amp_receipt",
    "update_clic_common_binding_receipt",
    "update_clic_resource_receipt",
    "validate_clic_terminal_envelope",
    "validate_clic_terminal_receipt",
    "write_clic_failure_receipt",
]
