"""Strict real-checkpoint hooks and feature materialization for D127.

This module is deliberately narrower than a Stage2 predictor.  It binds the
three frozen D127 intervention sites to the eager ADV3B02 identity backbone,
derives the support-only DA state once, and materializes immutable base and
adapted ``z_id160`` caches.  It accepts neither query labels nor query truth,
and it never mutates checkpoint parameters or serializes a floating-point
asset sidecar.

The D127 pure-tensor DA rules live in :mod:`stage2_d127_da_candidates`.  The
only purpose here is to make those rules reach the exact checkpoint nodes
without approximating the downstream model by a second network.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping, TypeAlias

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from model_dual_cvsincnet import backbone_forward_compat

from cvsrffi import stage2_d127_da_candidates as da
from cvsrffi import stage2_d127_phase1_assets as phase1_assets
from cvsrffi.stage2_d127_torch_compat import numpy_to_torch_copy
from cvsrffi import stage2_zid_student_t_qknn as qknn


SCHEMA = "cvs.phase2.d127.checkpoint_hooks.v1"
INPUT_LEN = 256
IQ_CHANNELS = 2
Z_DIM = 160
HIDDEN_DIM = da.JOINT_PROJ_INPUT_DIM
SUPPORT_TEMPERATURE = 0.85
_UNIT_ATOL = 2.5e-5

D127Asset: TypeAlias = da.FSRGAsset | da.RDHAAsset
D127State: TypeAlias = da.FSRGState | da.RDHAState


class D127CheckpointHookError(ValueError):
    """Raised when the frozen real-model hook contract is not closed."""


def _frozen_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _readonly_zid(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.dtype != np.float32
        or array.ndim != 2
        or int(array.shape[0]) < 1
        or int(array.shape[1]) != Z_DIM
        or not np.isfinite(array).all()
    ):
        raise D127CheckpointHookError(
            f"{name} must be finite float32 [N,{Z_DIM}]"
        )
    result = np.array(array, dtype=np.float32, copy=True, order="C")
    norms = np.linalg.norm(result.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=_UNIT_ATOL):
        raise D127CheckpointHookError(f"{name} must contain unit z_id rows")
    result.setflags(write=False)
    return result


def _tensor_to_readonly_zid(value: Tensor, *, name: str) -> np.ndarray:
    if (
        not torch.is_tensor(value)
        or value.dtype != torch.float32
        or value.ndim != 2
        or int(value.shape[1]) != Z_DIM
        or not bool(torch.isfinite(value).all().item())
    ):
        raise D127CheckpointHookError(
            f"{name} must be finite float32 [N,{Z_DIM}]"
        )
    # ``tolist`` is intentionally used instead of ``Tensor.numpy``: historical
    # checkpoint tests exercise tensor subclasses that reject the NumPy bridge.
    array = np.asarray(value.detach().cpu().contiguous().tolist(), dtype=np.float32)
    return _readonly_zid(array, name=name)


def _array_receipt(value: np.ndarray) -> Mapping[str, Any]:
    array = np.ascontiguousarray(value)
    payload = (
        f"{array.dtype.str}|{tuple(int(item) for item in array.shape)}|".encode("utf-8")
        + array.tobytes(order="C")
    )
    return _frozen_mapping(
        {
            "dtype": array.dtype.str,
            "shape": tuple(int(item) for item in array.shape),
            "sha256": _sha256_bytes(payload),
            "readonly": not bool(array.flags.writeable),
        }
    )


def _unit_rows_from_pre_relu(pre_relu: Tensor) -> Tensor:
    """Build the frozen z160 rule with signed totalization only when needed."""

    if (
        not torch.is_tensor(pre_relu)
        or pre_relu.dtype != torch.float32
        or pre_relu.ndim != 2
        or int(pre_relu.shape[0]) < 1
        or int(pre_relu.shape[1]) != Z_DIM
        or not bool(torch.isfinite(pre_relu).all().item())
    ):
        raise D127CheckpointHookError(
            f"joint_proj.0 pre-ReLU must be finite float32 [N,{Z_DIM}]"
        )
    positive = torch.relu(pre_relu)
    positive_norm = torch.linalg.vector_norm(positive, dim=1, keepdim=True)
    signed_norm = torch.linalg.vector_norm(pre_relu, dim=1, keepdim=True)
    if bool(torch.any(signed_norm <= torch.finfo(pre_relu.dtype).eps).item()):
        raise D127CheckpointHookError(
            "zero joint_proj.0 pre-ReLU row cannot be signed-totalized"
        )
    z_positive = positive / positive_norm.clamp_min(torch.finfo(pre_relu.dtype).eps)
    z_signed = pre_relu / signed_norm
    z_id = torch.where(positive_norm > torch.finfo(pre_relu.dtype).eps, z_positive, z_signed)
    norms = torch.linalg.vector_norm(z_id, dim=1)
    if not bool(torch.allclose(norms, torch.ones_like(norms), atol=_UNIT_ATOL, rtol=0.0)):
        raise D127CheckpointHookError("z_id160 L2 normalization drift")
    return z_id


def _validate_received_iq(model: nn.Module, received_iq: Tensor) -> None:
    if (
        not isinstance(model, nn.Module)
        or model.training
        or not torch.is_tensor(received_iq)
        or received_iq.dtype != torch.float32
        or received_iq.ndim != 3
        or int(received_iq.shape[0]) < 1
        or int(received_iq.shape[1]) != IQ_CHANNELS
        or int(received_iq.shape[2]) != INPUT_LEN
        or not bool(torch.isfinite(received_iq).all().item())
    ):
        raise D127CheckpointHookError(
            f"D127 requires a frozen eval model and finite float32 [N,{IQ_CHANNELS},{INPUT_LEN}] IQ"
        )
    try:
        parameter_device = next(model.parameters()).device
    except StopIteration as exc:
        raise D127CheckpointHookError("D127 model must expose checkpoint parameters") from exc
    if received_iq.device != parameter_device:
        raise D127CheckpointHookError("received IQ must be on the checkpoint device")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise D127CheckpointHookError("D127 checkpoint parameters must be frozen")


@dataclass(frozen=True, slots=True)
class _HookBinding:
    candidate_id: str
    tap_name: str
    tap_module: nn.Module
    tap_kind: str
    linear: nn.Linear
    joint: nn.Sequential
    dimension: int


def _bind_candidate(model: nn.Module, candidate_id: str) -> _HookBinding:
    if candidate_id not in (da.CANDIDATE_A, da.CANDIDATE_B, da.CANDIDATE_C):
        raise D127CheckpointHookError("unknown D127 candidate ID")
    try:
        backbone = model.id_backbone
        joint = backbone.cls_head.joint_proj
        linear = joint[0]
    except (AttributeError, IndexError, TypeError) as exc:
        raise D127CheckpointHookError("ADV3B02 identity hook path is absent") from exc
    if (
        not isinstance(backbone, nn.Module)
        or backbone.training
        or not isinstance(joint, nn.Sequential)
        or len(joint) < 2
        or not isinstance(linear, nn.Linear)
        or not isinstance(joint[1], nn.ReLU)
        or tuple(linear.weight.shape) != (Z_DIM, HIDDEN_DIM)
        or linear.bias is None
        or linear.weight.dtype != torch.float32
        or linear.bias.dtype != torch.float32
    ):
        raise D127CheckpointHookError("joint_proj.0 [320,160] pre-ReLU contract drift")

    if candidate_id == da.CANDIDATE_A:
        try:
            time_fuse = backbone.time_fuse
            tap_module = time_fuse[1]
            activation = time_fuse[2]
        except (AttributeError, IndexError, TypeError) as exc:
            raise D127CheckpointHookError("time_fuse.1 hook path is absent") from exc
        if (
            not isinstance(time_fuse, nn.Sequential)
            or not isinstance(tap_module, nn.GroupNorm)
            or not isinstance(activation, nn.ReLU)
            or int(tap_module.num_channels) < da.RANK
        ):
            raise D127CheckpointHookError("time_fuse.1 pre-ReLU contract drift")
        return _HookBinding(
            candidate_id=candidate_id,
            tap_name=da.TAP_A,
            tap_module=tap_module,
            tap_kind="module_output_pre_relu",
            linear=linear,
            joint=joint,
            dimension=int(tap_module.num_channels),
        )

    if candidate_id == da.CANDIDATE_B:
        try:
            t2 = backbone.t2
            tap_module = t2.norm
            activation = t2.act
        except AttributeError as exc:
            raise D127CheckpointHookError("t2.norm hook path is absent") from exc
        if (
            not isinstance(tap_module, nn.GroupNorm)
            or not isinstance(activation, nn.ReLU)
            or int(tap_module.num_channels) < da.RANK
        ):
            raise D127CheckpointHookError("t2.norm pre-ReLU contract drift")
        return _HookBinding(
            candidate_id=candidate_id,
            tap_name=da.TAP_B,
            tap_module=tap_module,
            tap_kind="module_output_pre_relu",
            linear=linear,
            joint=joint,
            dimension=int(tap_module.num_channels),
        )

    return _HookBinding(
        candidate_id=candidate_id,
        tap_name=da.TAP_C,
        tap_module=linear,
        tap_kind="linear_input",
        linear=linear,
        joint=joint,
        dimension=HIDDEN_DIM,
    )


def freeze_d127_checkpoint_model(model: nn.Module) -> nn.Module:
    """Freeze the reconstructed checkpoint without changing any weights.

    The caller may construct a standard PyTorch module whose parameters still
    have ``requires_grad=True`` after strict ``load_state_dict``.  D127 learns
    only the temporary support state ``a``; this helper makes accidental model
    parameter updates impossible before hook materialization begins.
    """

    if not isinstance(model, nn.Module):
        raise D127CheckpointHookError("D127 checkpoint must be an nn.Module")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    # Bind every frozen path now.  This prevents a caller from using a model
    # that happens to contain joint_proj.0 but lacks either A or B's exact node.
    for candidate in (da.CANDIDATE_A, da.CANDIDATE_B, da.CANDIDATE_C):
        _bind_candidate(model, candidate)
    return model


def load_d127_frozen_checkpoint(
    checkpoint_path: str | Path, *, device: torch.device | str = "cpu"
) -> tuple[nn.Module, Mapping[str, Any]]:
    """Strictly rebuild the fixed 256-sample/14-domain ADV3B02 checkpoint.

    This is intentionally a bounded convenience for the real-checkpoint smoke
    and future frozen deployment entry.  It delegates exact-byte loading and
    complete state-dict closure to the already-audited D105 bridge.
    """

    from cvsrffi.phase2_candidate_capsule import BASE_CHECKPOINT_SHA256
    from cvsrffi.stage2_d105_phase1_bundle import (
        build_d105_exact_model_from_checkpoint,
        load_d105_exact_sha_bound_checkpoint,
    )

    path = Path(checkpoint_path)
    if path.is_symlink() or not path.is_file():
        raise D127CheckpointHookError("D127 checkpoint must be a regular file")
    runtime_device = torch.device(device)
    try:
        checkpoint, loader_receipt = load_d105_exact_sha_bound_checkpoint(
            path, BASE_CHECKPOINT_SHA256
        )
        model, model_receipt = build_d105_exact_model_from_checkpoint(
            checkpoint, input_len=INPUT_LEN, device=runtime_device
        )
    except Exception as exc:  # The D105 boundary supplies the precise cause.
        raise D127CheckpointHookError(
            "strict D127 checkpoint reconstruction failed"
        ) from exc
    if (
        model_receipt.get("checkpoint_load_strict") is not True
        or model_receipt.get("input_len") != INPUT_LEN
        or model_receipt.get("num_domains_from_state") != 14
        or int(getattr(model, "num_domains", -1)) != 14
    ):
        raise D127CheckpointHookError(
            "D127 requires strict input_len=256 and num_domains=14 reconstruction"
        )
    freeze_d127_checkpoint_model(model)
    return model, _frozen_mapping(
        {
            "schema": SCHEMA,
            "checkpoint_sha256": BASE_CHECKPOINT_SHA256,
            "checkpoint_loader": dict(loader_receipt),
            "model_reconstruction": dict(model_receipt),
            "input_len": INPUT_LEN,
            "num_domains": 14,
            "eval_mode": not model.training,
            "all_checkpoint_parameters_frozen": not any(
                parameter.requires_grad for parameter in model.parameters()
            ),
        }
    )


@dataclass(frozen=True, slots=True)
class _CapturedIDForward:
    z_id: Tensor
    pre_relu: Tensor
    hidden: Tensor
    tap: Tensor


def _same_tensor_bytes(left: Tensor, right: Tensor) -> bool:
    return (
        left.dtype == right.dtype
        and left.device == right.device
        and tuple(left.shape) == tuple(right.shape)
        and bool(torch.equal(left.detach(), right.detach()))
    )


def _capture_tensor(value: Tensor, *, name: str, require_grad: bool) -> Tensor:
    if not torch.is_tensor(value) or value.dtype != torch.float32:
        raise D127CheckpointHookError(f"{name} must be float32 tensor")
    copy = value.clone()
    if require_grad and not copy.requires_grad:
        # A/B's loss callback must be bound to the temporary rank-two state.
        # C intentionally calls this path only without a gradient requirement.
        raise D127CheckpointHookError(f"{name} unexpectedly detached from FSRG state")
    return copy


def _validate_tap(value: Tensor, binding: _HookBinding, *, name: str) -> None:
    expected_ndim = 2 if binding.candidate_id == da.CANDIDATE_C else 3
    if (
        not torch.is_tensor(value)
        or value.dtype != torch.float32
        or value.ndim != expected_ndim
        or int(value.shape[0]) < 1
        or int(value.shape[1]) != binding.dimension
        or (expected_ndim == 3 and int(value.shape[2]) < 1)
        or not bool(torch.isfinite(value).all().item())
    ):
        expected = "[N,320]" if expected_ndim == 2 else f"[N,{binding.dimension},T]"
        raise D127CheckpointHookError(f"{name} must be finite float32 {expected}")


def _validate_replacement(source: Tensor, replacement: Tensor, binding: _HookBinding) -> Tensor:
    _validate_tap(replacement, binding, name="adapted D127 tap")
    if (
        source.device != replacement.device
        or source.dtype != replacement.dtype
        or tuple(source.shape) != tuple(replacement.shape)
    ):
        raise D127CheckpointHookError("adapted tap must exactly preserve hook shape/device/dtype")
    # The model uses inplace ReLUs after A/B.  Returning a fresh clone keeps the
    # core's input tensor intact for autograd while preserving the exact same
    # downstream model path.
    return replacement.clone()


def _id_backbone_forward(
    model: nn.Module,
    received_iq: Tensor,
    binding: _HookBinding,
    *,
    intervention: Callable[[Tensor], Tensor] | None,
    require_grad: bool,
) -> _CapturedIDForward:
    """Run exactly one identity-backbone forward with optional local rewrite."""

    _validate_received_iq(model, received_iq)
    captured: dict[str, Tensor] = {}

    def capture_linear(
        _module: nn.Module, args: tuple[Tensor, ...], output: Tensor
    ) -> None:
        if len(args) != 1 or not torch.is_tensor(args[0]) or not torch.is_tensor(output):
            raise D127CheckpointHookError("joint_proj.0 hook arguments drift")
        captured["hidden"] = _capture_tensor(
            args[0], name="joint_proj.0 input", require_grad=require_grad and intervention is not None
        )
        captured["pre_relu"] = _capture_tensor(
            output, name="joint_proj.0 output", require_grad=require_grad and intervention is not None
        )

    def capture_joint(_module: nn.Module, _args: tuple[Tensor, ...], output: Tensor) -> None:
        if not torch.is_tensor(output):
            raise D127CheckpointHookError("joint_proj output drift")
        captured["joint"] = _capture_tensor(
            output, name="joint_proj output", require_grad=require_grad and intervention is not None
        )

    hooks: list[Any] = []
    if binding.tap_kind == "module_output_pre_relu":

        def rewrite_output(
            _module: nn.Module, _args: tuple[Tensor, ...], output: Tensor
        ) -> Tensor | None:
            _validate_tap(output, binding, name="captured D127 tap")
            captured["tap"] = _capture_tensor(
                output,
                name="captured D127 tap",
                # The native pre-hook value remains frozen.  Only the
                # replacement returned below must stay connected to ``a``.
                require_grad=False,
            )
            if intervention is None:
                return None
            return _validate_replacement(output, intervention(output), binding)

        hooks.append(binding.tap_module.register_forward_hook(rewrite_output))
    else:

        def rewrite_input(_module: nn.Module, args: tuple[Tensor, ...]) -> tuple[Tensor, ...] | None:
            if len(args) != 1 or not torch.is_tensor(args[0]):
                raise D127CheckpointHookError("joint_proj.0 input hook drift")
            source = args[0]
            _validate_tap(source, binding, name="captured D127 hidden")
            captured["tap"] = _capture_tensor(
                source,
                name="captured D127 hidden",
                require_grad=False,
            )
            if intervention is None:
                return None
            return (_validate_replacement(source, intervention(source), binding),)

        hooks.append(binding.tap_module.register_forward_pre_hook(rewrite_input))

    hooks.append(binding.linear.register_forward_hook(capture_linear))
    hooks.append(binding.joint.register_forward_hook(capture_joint))
    try:
        context = torch.enable_grad() if require_grad else torch.no_grad()
        with context:
            aux = backbone_forward_compat(
                model.id_backbone,
                received_iq,
                y=None,
                return_aux=True,
                domain_labels=None,
            )
    finally:
        for hook in reversed(hooks):
            hook.remove()

    z_raw = aux.get("feat_joint") if isinstance(aux, dict) else None
    if (
        not torch.is_tensor(z_raw)
        or set(captured) != {"tap", "hidden", "pre_relu", "joint"}
        or tuple(captured["hidden"].shape) != (len(received_iq), HIDDEN_DIM)
        or tuple(captured["pre_relu"].shape) != (len(received_iq), Z_DIM)
        or tuple(captured["joint"].shape) != (len(received_iq), Z_DIM)
        or not bool(torch.isfinite(z_raw).all().item())
    ):
        raise D127CheckpointHookError("D127 same-forward capture is incomplete")
    _validate_tap(captured["tap"], binding, name="captured D127 tap")

    # The capture clone protects pre-ReLU values from the inplace ReLU.  Verify
    # both native sequential execution and the aux mapping without changing the
    # returned autograd graph.
    with torch.no_grad():
        recomputed_pre = binding.linear(captured["hidden"].detach())
        recomputed_joint = torch.relu(captured["pre_relu"].detach())
    if (
        not _same_tensor_bytes(captured["pre_relu"], recomputed_pre)
        or not _same_tensor_bytes(captured["joint"], recomputed_joint)
        or not _same_tensor_bytes(z_raw, captured["joint"])
    ):
        raise D127CheckpointHookError(
            "D127 hook is not byte-bound to joint_proj.0, ReLU, and feat_joint"
        )
    z_id = _unit_rows_from_pre_relu(captured["pre_relu"])
    return _CapturedIDForward(
        z_id=z_id,
        pre_relu=captured["pre_relu"],
        hidden=captured["hidden"],
        tap=captured["tap"],
    )


def _labels_for_support(value: Tensor, *, rows: int, device: torch.device) -> Tensor:
    if (
        not torch.is_tensor(value)
        or value.dtype.is_floating_point
        or value.dtype == torch.bool
        or value.ndim != 1
        or int(value.shape[0]) != rows
        or value.device != device
    ):
        raise D127CheckpointHookError(
            "support_labels must be integral [N] on the support IQ device"
        )
    # The DA core verifies at least two classes and equal-K.  Recheck here so
    # the loss callback has an explicit stable class-index mapping.
    unique, inverse, counts = torch.unique(
        value, sorted=True, return_inverse=True, return_counts=True
    )
    if int(unique.numel()) < 2 or not bool(torch.all(counts == counts[0]).item()):
        raise D127CheckpointHookError(
            "support_labels must contain at least two equal-K registered classes"
        )
    return value


def _support_view_loss(pre_relu: Tensor, support_labels: Tensor) -> Tensor:
    """Per-support frozen two-view cosine CE for the FSRG state gradient."""

    z_a = _unit_rows_from_pre_relu(pre_relu)
    signed_norm = torch.linalg.vector_norm(pre_relu, dim=1, keepdim=True)
    z_b = pre_relu / signed_norm
    unique, inverse = torch.unique(support_labels, sorted=True, return_inverse=True)
    prototypes_a = torch.stack(
        [z_a.index_select(0, torch.nonzero(inverse == index, as_tuple=False).reshape(-1)).mean(dim=0)
         for index in range(int(unique.numel()))]
    )
    prototypes_b = torch.stack(
        [z_b.index_select(0, torch.nonzero(inverse == index, as_tuple=False).reshape(-1)).mean(dim=0)
         for index in range(int(unique.numel()))]
    )
    proto_a_norm = torch.linalg.vector_norm(prototypes_a, dim=1)
    proto_b_norm = torch.linalg.vector_norm(prototypes_b, dim=1)
    if bool(
        torch.any(proto_a_norm <= torch.finfo(pre_relu.dtype).eps).item()
        or torch.any(proto_b_norm <= torch.finfo(pre_relu.dtype).eps).item()
    ):
        raise D127CheckpointHookError(
            "support class-mean prototype cannot be normalized"
        )
    prototypes_a = prototypes_a.detach() / proto_a_norm.detach().reshape(-1, 1)
    prototypes_b = prototypes_b.detach() / proto_b_norm.detach().reshape(-1, 1)
    target = inverse.to(dtype=torch.long)
    loss_b_to_a = F.cross_entropy(
        (z_b @ prototypes_a.transpose(0, 1)) / SUPPORT_TEMPERATURE,
        target,
        reduction="none",
    )
    loss_a_to_b = F.cross_entropy(
        (z_a @ prototypes_b.transpose(0, 1)) / SUPPORT_TEMPERATURE,
        target,
        reduction="none",
    )
    losses = 0.5 * (loss_b_to_a + loss_a_to_b)
    if (
        losses.dtype != torch.float32
        or losses.ndim != 1
        or not bool(torch.isfinite(losses).all().item())
    ):
        raise D127CheckpointHookError("D127 support two-view CE is non-finite")
    return losses


@dataclass(frozen=True, slots=True)
class D127Phase1EpisodeIQ:
    """Raw Phase1 IQ rows bound to one immutable source episode ID.

    The bridge retains this record only in memory while a Phase1 asset is
    built.  It is deliberately separate from the feature episode: every
    callback must prove its episode ID, split, and row count before the real
    checkpoint is executed.
    """

    episode_id: str
    support_iq: Tensor
    query_iq: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise D127CheckpointHookError("Phase1 raw-IQ episode_id must be nonempty")
        for name, value in (("support_iq", self.support_iq), ("query_iq", self.query_iq)):
            if (
                not torch.is_tensor(value)
                or value.dtype != torch.float32
                or value.ndim != 3
                or int(value.shape[0]) < 1
                or int(value.shape[1]) != IQ_CHANNELS
                or int(value.shape[2]) != INPUT_LEN
                or not bool(torch.isfinite(value).all().item())
            ):
                raise D127CheckpointHookError(
                    f"{name} must be finite float32 [N,{IQ_CHANNELS},{INPUT_LEN}]"
                )


# A readable alias for callers which use the noun order "raw IQ episode".
D127Phase1RawIQEpisode = D127Phase1EpisodeIQ


@dataclass(frozen=True, slots=True)
class D127Phase1CheckpointForward:
    """One real, same-downstream Phase1 checkpoint forward.

    ``pre_relu`` and ``z_id`` intentionally retain the caller replacement's
    autograd graph.  This record is ephemeral and never becomes a Phase2
    feature cache or a persisted floating-point sidecar.
    """

    candidate_id: str
    episode_id: str
    split: Literal["support", "query"]
    tap: Tensor
    hidden: Tensor
    pre_relu: Tensor
    z_id: Tensor

    def __post_init__(self) -> None:
        if self.candidate_id not in (da.CANDIDATE_A, da.CANDIDATE_B, da.CANDIDATE_C):
            raise D127CheckpointHookError("Phase1 checkpoint forward candidate drift")
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise D127CheckpointHookError("Phase1 checkpoint forward episode ID drift")
        if self.split not in ("support", "query"):
            raise D127CheckpointHookError("Phase1 checkpoint forward split drift")
        rows = int(self.pre_relu.shape[0]) if torch.is_tensor(self.pre_relu) else -1
        if (
            not torch.is_tensor(self.pre_relu)
            or self.pre_relu.dtype != torch.float32
            or self.pre_relu.ndim != 2
            or rows < 1
            or int(self.pre_relu.shape[1]) != Z_DIM
            or not bool(torch.isfinite(self.pre_relu).all().item())
        ):
            raise D127CheckpointHookError("Phase1 forward pre_relu must be finite [N,160]")
        if (
            not torch.is_tensor(self.z_id)
            or self.z_id.dtype != torch.float32
            or tuple(self.z_id.shape) != (rows, Z_DIM)
            or self.z_id.device != self.pre_relu.device
            or not bool(torch.isfinite(self.z_id).all().item())
        ):
            raise D127CheckpointHookError("Phase1 forward z_id160 shape/device drift")
        if (
            not torch.is_tensor(self.hidden)
            or self.hidden.dtype != torch.float32
            or tuple(self.hidden.shape) != (rows, HIDDEN_DIM)
            or self.hidden.device != self.pre_relu.device
            or not bool(torch.isfinite(self.hidden).all().item())
        ):
            raise D127CheckpointHookError("Phase1 forward joint hidden shape/device drift")

    @property
    def z160(self) -> Tensor:
        """Alias retained for callers that name the deployed vector by width."""

        return self.z_id


def _phase1_qknn_locks(
    value: Mapping[int, qknn.Phase1ZIDStudentTLock],
) -> Mapping[int, qknn.Phase1ZIDStudentTLock]:
    if not isinstance(value, Mapping) or not value:
        raise D127CheckpointHookError("Phase1 qKNN locks must be a nonempty K mapping")
    result: dict[int, qknn.Phase1ZIDStudentTLock] = {}
    for raw_k, lock in value.items():
        if type(raw_k) is not int or type(lock) is not qknn.Phase1ZIDStudentTLock:
            raise D127CheckpointHookError("Phase1 qKNN locks must use exact int K keys and locks")
        if raw_k != lock.active_k or raw_k in result:
            raise D127CheckpointHookError("Phase1 qKNN lock key/active-K binding drift")
        result[raw_k] = lock
    return _frozen_mapping(result)


def _phase1_label_layout(
    episode: phase1_assets.FSRGEpisode | phase1_assets.RDHAEpisode,
) -> tuple[tuple[str, ...], tuple[str, ...], Tensor]:
    """Map source labels to a local, label-permutation-equivariant registry."""

    classes, support_inverse = torch.unique(
        episode.support_labels, sorted=True, return_inverse=True
    )
    if int(classes.numel()) < 2:
        raise D127CheckpointHookError("Phase1 qKNN requires at least two support classes")
    matches = episode.query_labels.reshape(-1, 1) == classes.reshape(1, -1)
    if not bool(torch.all(matches.any(dim=1)).item()):
        raise D127CheckpointHookError("Phase1 query labels drift from the support registry")
    query_target = torch.argmax(matches.to(dtype=torch.int64), dim=1).to(dtype=torch.long)
    registered = tuple(f"phase1_class_{index}" for index in range(int(classes.numel())))
    support_labels = tuple(
        registered[int(index)] for index in support_inverse.detach().cpu().tolist()
    )
    return registered, support_labels, query_target


def _torch_deployment_qknn_logits(
    bank: qknn.TypedINT8ZIDSupportBank, query_zid: Tensor
) -> Tensor:
    """Differentiable identity-qKNN scores over the exact decoded INT8 bank.

    The support side is already closed by ``build_typed_zid_support_bank`` and
    is copied from its deployed INT8/FP16 representation.  Query arithmetic is
    intentionally torch float64 so the real checkpoint path remains
    differentiable while following the NumPy deployment score term by term.
    """

    if type(bank) is not qknn.TypedINT8ZIDSupportBank:
        raise D127CheckpointHookError("Phase1 qKNN logits require an exact typed bank")
    if (
        not torch.is_tensor(query_zid)
        or query_zid.dtype != torch.float32
        or query_zid.ndim != 2
        or int(query_zid.shape[0]) < 1
        or int(query_zid.shape[1]) != Z_DIM
        or not bool(torch.isfinite(query_zid).all().item())
    ):
        raise D127CheckpointHookError("Phase1 qKNN query z_id must be finite float32 [N,160]")
    try:
        decoded = qknn.decode_zid_support_bank(bank)
    except qknn.ZIDStudentTQKNNError as exc:
        raise D127CheckpointHookError("Phase1 qKNN bank decode failed") from exc
    support = numpy_to_torch_copy(
        np.array(decoded, dtype=np.float32, copy=True),
        device=query_zid.device,
        dtype=torch.float64,
        name="D127 Phase1 qKNN decoded support",
    )
    # ``score_zid_student_t_logits`` normalizes a float32 query in float64 and
    # stores that normalization back into float32 before scoring.  Preserve the
    # same quantization boundary without severing the query autograd graph.
    query64 = query_zid.to(dtype=torch.float64)
    norms = torch.linalg.vector_norm(query64, dim=1, keepdim=True)
    if bool(torch.any(norms <= qknn.EPSILON).item()):
        raise D127CheckpointHookError("Phase1 qKNN query z_id contains a zero norm")
    query = (query64 / norms).to(dtype=torch.float32).to(dtype=torch.float64)
    cosine = torch.clamp(query @ support.transpose(0, 1), min=-1.0, max=1.0)
    distance = torch.clamp_min(2.0 * (1.0 - cosine), 0.0)
    columns: list[Tensor] = []
    for class_index, expected_count in enumerate(bank.support_counts):
        member_indices = np.flatnonzero(bank.class_indices_int16 == class_index)
        if int(member_indices.size) != int(expected_count):
            raise D127CheckpointHookError("Phase1 qKNN class support count drift")
        columns_index = numpy_to_torch_copy(
            member_indices,
            dtype=torch.long,
            device=query_zid.device,
            name="D127 Phase1 qKNN class indices",
        )
        local = distance.index_select(1, columns_index)
        h = float(bank.class_scales_fp16[class_index])
        lock = bank.config
        kernel = (
            -float(lock.kernel_volume_gamma)
            * float(lock.kernel_effective_dim)
            * math.log(h)
            - 0.5
            * (float(lock.student_nu) + float(lock.kernel_effective_dim))
            * torch.log1p(local / (float(lock.student_nu) * h * h))
        )
        maximum = torch.max(kernel, dim=1, keepdim=True).values
        column = (
            maximum[:, 0]
            + torch.log(torch.sum(torch.exp(kernel - maximum), dim=1))
            - math.log(int(expected_count))
        )
        columns.append(column)
    logits64 = torch.stack(columns, dim=1)
    if logits64.dtype != torch.float64 or not bool(torch.isfinite(logits64).all().item()):
        raise D127CheckpointHookError("Phase1 differentiable qKNN logits became non-finite")
    # The deployed NumPy scorer closes the Student-t columns as float32 before
    # its float64 temperature division.  This cast is therefore part of the
    # frozen deployment formula, not a lower-precision support proxy; PyTorch
    # preserves the query gradient through it.
    return logits64.to(dtype=torch.float32)


class D127Phase1CheckpointBridge:
    """Minimal real-checkpoint bridge for source-only D127 Phase1 training.

    A bridge is bound to exactly one frozen D127 candidate and a mapping from
    Phase1 episode IDs to their already-owned support/query raw-IQ tensors.
    It has no target, truth, role, quota, scorer, matrix, or runner surface.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        candidate_id: str,
        episode_iq_by_id: Mapping[str, D127Phase1EpisodeIQ],
    ) -> None:
        binding = _bind_candidate(model, candidate_id)
        if not isinstance(episode_iq_by_id, Mapping) or not episode_iq_by_id:
            raise D127CheckpointHookError("Phase1 bridge requires nonempty episode raw-IQ mapping")
        bound: dict[str, D127Phase1EpisodeIQ] = {}
        for raw_id, raw_iq in episode_iq_by_id.items():
            if (
                not isinstance(raw_id, str)
                or not raw_id
                or type(raw_iq) is not D127Phase1EpisodeIQ
                or raw_id != raw_iq.episode_id
                or raw_id in bound
            ):
                raise D127CheckpointHookError("Phase1 bridge raw-IQ episode-ID binding drift")
            _validate_received_iq(model, raw_iq.support_iq)
            _validate_received_iq(model, raw_iq.query_iq)
            bound[raw_id] = raw_iq
        self._model = model
        self._binding = binding
        self._episode_iq_by_id = _frozen_mapping(bound)

    @property
    def candidate_id(self) -> str:
        return self._binding.candidate_id

    @property
    def tap_name(self) -> str:
        return self._binding.tap_name

    def _raw_for(
        self, episode_id: str, split: Literal["support", "query"]
    ) -> Tensor:
        if not isinstance(episode_id, str) or not episode_id:
            raise D127CheckpointHookError("Phase1 bridge episode ID must be nonempty")
        if split not in ("support", "query"):
            raise D127CheckpointHookError("Phase1 bridge split must be support or query")
        raw = self._episode_iq_by_id.get(episode_id)
        if raw is None:
            raise D127CheckpointHookError("Phase1 bridge has no raw IQ for episode ID")
        return raw.support_iq if split == "support" else raw.query_iq

    def _episode_rows(
        self,
        episode: phase1_assets.FSRGEpisode | phase1_assets.RDHAEpisode,
        split: Literal["support", "query"],
    ) -> Tensor:
        if self.candidate_id in (da.CANDIDATE_A, da.CANDIDATE_B):
            if type(episode) is not phase1_assets.FSRGEpisode:
                raise D127CheckpointHookError("A/B bridge requires an FSRG episode")
            return episode.support_taps if split == "support" else episode.query_taps
        if type(episode) is not phase1_assets.RDHAEpisode:
            raise D127CheckpointHookError("C bridge requires an RDHA episode")
        return episode.support_hidden if split == "support" else episode.query_hidden

    def _validate_episode_binding(
        self,
        episode: phase1_assets.FSRGEpisode | phase1_assets.RDHAEpisode,
        split: Literal["support", "query"],
        captured_tap: Tensor,
    ) -> None:
        rows = self._episode_rows(episode, split)
        labels = episode.support_labels if split == "support" else episode.query_labels
        raw = self._raw_for(episode.episode_id, split)
        if (
            int(raw.shape[0]) != int(rows.shape[0])
            or int(labels.shape[0]) != int(rows.shape[0])
            or tuple(rows.shape) != tuple(captured_tap.shape)
            or rows.dtype != captured_tap.dtype
            or rows.device != captured_tap.device
            or not bool(torch.equal(rows.detach(), captured_tap.detach()))
        ):
            raise D127CheckpointHookError(
                "Phase1 bridge episode/raw-IQ split, row, or real-tap binding drift"
            )

    @staticmethod
    def _forward_record(
        captured: _CapturedIDForward,
        *,
        candidate_id: str,
        episode_id: str,
        split: Literal["support", "query"],
    ) -> D127Phase1CheckpointForward:
        return D127Phase1CheckpointForward(
            candidate_id=candidate_id,
            episode_id=episode_id,
            split=split,
            tap=captured.tap,
            hidden=captured.hidden,
            pre_relu=captured.pre_relu,
            z_id=captured.z_id,
        )

    def capture_raw(
        self, episode_id: str, *, split: Literal["support", "query"]
    ) -> D127Phase1CheckpointForward:
        """Capture the actual candidate tap and final pre-ReLU without rewrite."""

        raw = self._raw_for(episode_id, split)
        captured = _id_backbone_forward(
            self._model,
            raw,
            self._binding,
            intervention=None,
            require_grad=False,
        )
        return self._forward_record(
            captured,
            candidate_id=self.candidate_id,
            episode_id=episode_id,
            split=split,
        )

    def capture_episode(
        self,
        episode: phase1_assets.FSRGEpisode | phase1_assets.RDHAEpisode,
        *,
        split: Literal["support", "query"],
    ) -> D127Phase1CheckpointForward:
        """Capture and verify the real source tap against one frozen episode."""

        forward = self.capture_raw(episode.episode_id, split=split)
        self._validate_episode_binding(episode, split, forward.tap)
        return forward

    def forward_with_replacement(
        self,
        episode: phase1_assets.FSRGEpisode | phase1_assets.RDHAEpisode,
        *,
        split: Literal["support", "query"],
        replacement: Tensor,
    ) -> D127Phase1CheckpointForward:
        """Inject one non-detached tap replacement through the real downstream."""

        if not torch.is_tensor(replacement) or not replacement.requires_grad:
            raise D127CheckpointHookError(
                "Phase1 bridge replacement must retain a differentiable caller graph"
            )
        raw = self._raw_for(episode.episode_id, split)
        captured = _id_backbone_forward(
            self._model,
            raw,
            self._binding,
            intervention=lambda _source: replacement,
            require_grad=True,
        )
        self._validate_episode_binding(episode, split, captured.tap)
        forward = self._forward_record(
            captured,
            candidate_id=self.candidate_id,
            episode_id=episode.episode_id,
            split=split,
        )
        if not forward.pre_relu.requires_grad or not forward.z_id.requires_grad:
            raise D127CheckpointHookError("Phase1 replacement did not reach real final z_id160")
        return forward

    def build_deployment_qknn_bank(
        self,
        episode: phase1_assets.FSRGEpisode | phase1_assets.RDHAEpisode,
        *,
        support_zid: Tensor,
        qknn_locks: Mapping[int, qknn.Phase1ZIDStudentTLock],
    ) -> qknn.TypedINT8ZIDSupportBank:
        """Compile the exact support-only INT8/FP16 qKNN bank for one episode."""

        locks = _phase1_qknn_locks(qknn_locks)
        lock = locks.get(episode.k_shot)
        if lock is None:
            raise D127CheckpointHookError("Phase1 bridge is missing this episode's exact K lock")
        registered, support_labels, _query_target = _phase1_label_layout(episode)
        support = _tensor_to_readonly_zid(
            support_zid.detach(), name="Phase1 adapted support z_id"
        )
        if int(support.shape[0]) != len(support_labels):
            raise D127CheckpointHookError("Phase1 qKNN support z_id/label row drift")
        try:
            return qknn.build_typed_zid_support_bank(
                support,
                support_labels,
                registered,
                config=lock,
            )
        except qknn.ZIDStudentTQKNNError as exc:
            raise D127CheckpointHookError("Phase1 qKNN support-bank build failed") from exc

    def deployment_qknn_logits(
        self,
        episode: phase1_assets.FSRGEpisode | phase1_assets.RDHAEpisode,
        *,
        support_zid: Tensor,
        query_zid: Tensor,
        qknn_locks: Mapping[int, qknn.Phase1ZIDStudentTLock],
    ) -> Tensor:
        """Return differentiable deployed float32 qKNN logits for source query rows."""

        bank = self.build_deployment_qknn_bank(
            episode, support_zid=support_zid, qknn_locks=qknn_locks
        )
        logits = _torch_deployment_qknn_logits(bank, query_zid)
        if int(logits.shape[0]) != int(episode.query_labels.shape[0]):
            raise D127CheckpointHookError("Phase1 qKNN query z_id/episode row drift")
        return logits

    def fsrg_loss_callbacks(
        self, *, qknn_locks: Mapping[int, qknn.Phase1ZIDStudentTLock]
    ) -> phase1_assets.FSRGLossCallbacks:
        """Create episode-aware real-checkpoint A/B support and outer callbacks."""

        if self.candidate_id not in (da.CANDIDATE_A, da.CANDIDATE_B):
            raise D127CheckpointHookError("FSRG callbacks require an A or B bridge")
        locks = _phase1_qknn_locks(qknn_locks)

        def support_per_sample(
            episode: phase1_assets.FSRGEpisode, adapted_tap: Tensor
        ) -> Tensor:
            forward = self.forward_with_replacement(
                episode, split="support", replacement=adapted_tap
            )
            return _support_view_loss(forward.pre_relu, episode.support_labels)

        def outer_query_per_sample(
            episode: phase1_assets.FSRGEpisode,
            adapted_support: Tensor,
            adapted_query: Tensor,
        ) -> Tensor:
            support_forward = self.forward_with_replacement(
                episode, split="support", replacement=adapted_support
            )
            query_forward = self.forward_with_replacement(
                episode, split="query", replacement=adapted_query
            )
            logits = self.deployment_qknn_logits(
                episode,
                support_zid=support_forward.z_id,
                query_zid=query_forward.z_id,
                qknn_locks=locks,
            )
            _registered, _support_labels, target = _phase1_label_layout(episode)
            losses = F.cross_entropy(
                logits.to(dtype=torch.float64)
                / float(locks[episode.k_shot].temperature),
                target.to(device=logits.device),
                reduction="none",
            ).to(dtype=torch.float32)
            if not bool(torch.isfinite(losses).all().item()):
                raise D127CheckpointHookError("Phase1 FSRG qKNN outer CE is non-finite")
            return losses

        return phase1_assets.FSRGLossCallbacks(
            support_per_sample=support_per_sample,
            outer_query_per_sample=outer_query_per_sample,
        )

    def rdha_outer_callback(
        self, *, qknn_locks: Mapping[int, qknn.Phase1ZIDStudentTLock]
    ) -> phase1_assets.RDHALossCallback:
        """Create the episode-aware real-checkpoint C qKNN outer callback."""

        if self.candidate_id != da.CANDIDATE_C:
            raise D127CheckpointHookError("RDHA outer callback requires a C bridge")
        locks = _phase1_qknn_locks(qknn_locks)

        def outer_query_per_sample(
            episode: phase1_assets.RDHAEpisode,
            adapted_support: Tensor,
            adapted_query: Tensor,
        ) -> Tensor:
            support_forward = self.forward_with_replacement(
                episode, split="support", replacement=adapted_support
            )
            query_forward = self.forward_with_replacement(
                episode, split="query", replacement=adapted_query
            )
            logits = self.deployment_qknn_logits(
                episode,
                support_zid=support_forward.z_id,
                query_zid=query_forward.z_id,
                qknn_locks=locks,
            )
            _registered, _support_labels, target = _phase1_label_layout(episode)
            losses = F.cross_entropy(
                logits.to(dtype=torch.float64)
                / float(locks[episode.k_shot].temperature),
                target.to(device=logits.device),
                reduction="none",
            ).to(dtype=torch.float32)
            if not bool(torch.isfinite(losses).all().item()):
                raise D127CheckpointHookError("Phase1 RDHA qKNN outer CE is non-finite")
            return losses

        return outer_query_per_sample


def _require_asset(asset: D127Asset) -> str:
    if isinstance(asset, da.FSRGAsset):
        if asset.candidate_id not in (da.CANDIDATE_A, da.CANDIDATE_B):
            raise D127CheckpointHookError("FSRG asset candidate binding drift")
        return asset.candidate_id
    if isinstance(asset, da.RDHAAsset):
        if asset.candidate_id != da.CANDIDATE_C:
            raise D127CheckpointHookError("RDHA asset candidate binding drift")
        return da.CANDIDATE_C
    raise D127CheckpointHookError("D127 hook requires decoded FSRGAsset or RDHAAsset")


def _require_state(asset: D127Asset, state: D127State) -> None:
    if isinstance(asset, da.FSRGAsset):
        if not isinstance(state, da.FSRGState):
            raise D127CheckpointHookError("FSRG asset requires FSRG state")
        if state.candidate_id != asset.candidate_id or state.tap_name != asset.tap_name:
            raise D127CheckpointHookError("FSRG state/asset binding drift")
        return
    if not isinstance(state, da.RDHAState):
        raise D127CheckpointHookError("RDHA asset requires RDHA state")
    if state.candidate_id != da.CANDIDATE_C or state.tap_name != da.TAP_C:
        raise D127CheckpointHookError("RDHA state/asset binding drift")


def _fit_support_state(
    model: nn.Module,
    support_iq: Tensor,
    support_labels: Tensor,
    binding: _HookBinding,
    asset: D127Asset,
    base: _CapturedIDForward,
) -> tuple[D127State, int]:
    if isinstance(asset, da.FSRGAsset):
        if asset.dimension != binding.dimension or asset.tap_name != binding.tap_name:
            raise D127CheckpointHookError("FSRG asset does not match real checkpoint tap")
        callback_forwards = 0

        def per_sample_loss(adapted_tap: Tensor) -> Tensor:
            nonlocal callback_forwards
            forward = _id_backbone_forward(
                model,
                support_iq,
                binding,
                intervention=lambda _source: adapted_tap,
                require_grad=True,
            )
            callback_forwards += 1
            return _support_view_loss(forward.pre_relu, support_labels)

        state = da.fit_fsrg_support_state(
            base.tap,
            support_labels,
            asset,
            per_sample_loss,
        )
        if callback_forwards != 1:
            raise D127CheckpointHookError("FSRG support loss must use exactly one hooked forward")
        return state, callback_forwards

    if asset.candidate_id != da.CANDIDATE_C or binding.dimension != HIDDEN_DIM:
        raise D127CheckpointHookError("RDHA asset does not match joint_proj.0 input")
    return da.fit_rdah_support_state(base.tap, support_labels, asset), 0


def _forward_adapted(
    model: nn.Module,
    received_iq: Tensor,
    binding: _HookBinding,
    asset: D127Asset,
    state: D127State,
    *,
    require_grad: bool = False,
) -> _CapturedIDForward:
    _require_state(asset, state)
    if isinstance(asset, da.FSRGAsset):
        if not isinstance(state, da.FSRGState):  # narrowed for static analyzers.
            raise D127CheckpointHookError("FSRG state type drift")
        return _id_backbone_forward(
            model,
            received_iq,
            binding,
            intervention=lambda source: da.adapt_fsrg_query(source, asset, state),
            require_grad=require_grad,
        )
    if not isinstance(state, da.RDHAState):
        raise D127CheckpointHookError("RDHA state type drift")
    return _id_backbone_forward(
        model,
        received_iq,
        binding,
        intervention=lambda source: da.adapt_rdah_query(source, asset, state),
        require_grad=require_grad,
    )


@dataclass(frozen=True, slots=True)
class D127FeatureCache:
    """Immutable formal z160 support/query cache for one representation."""

    representation: str
    support_zid: np.ndarray
    query_zid: np.ndarray
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.representation not in ("base_zid160", "adapted_zid160"):
            raise D127CheckpointHookError("unknown D127 cache representation")
        support = _readonly_zid(self.support_zid, name="support_zid")
        query = _readonly_zid(self.query_zid, name="query_zid")
        payload = {
            "schema": SCHEMA,
            "representation": self.representation,
            "support_zid": dict(_array_receipt(support)),
            "query_zid": dict(_array_receipt(query)),
        }
        supplied = dict(self.receipt)
        if supplied and supplied != payload:
            raise D127CheckpointHookError("D127 cache receipt drift")
        object.__setattr__(self, "support_zid", support)
        object.__setattr__(self, "query_zid", query)
        object.__setattr__(self, "receipt", _frozen_mapping(payload))


@dataclass(frozen=True, slots=True)
class D127HookReceipt:
    """Hook, forward, and protocol counters for one support/query materialization."""

    candidate_id: str
    tap_name: str
    tap_shape: tuple[int, ...]
    base_support_id_backbone_forwards: int
    support_state_loss_id_backbone_forwards: int
    adapted_support_id_backbone_forwards: int
    base_query_id_backbone_forwards: int
    adapted_query_id_backbone_forwards: int
    state_fit_calls: int
    query_rows_used_for_fit: int
    query_state_updates: int
    query_selection_count: int
    query_gradient_calls: int
    phase2_optimizer_steps: int
    checkpoint_parameters_frozen: bool
    same_model_downstream: bool
    same_forward_tap_and_final_pre_relu: bool

    def __post_init__(self) -> None:
        if self.candidate_id not in (da.CANDIDATE_A, da.CANDIDATE_B, da.CANDIDATE_C):
            raise D127CheckpointHookError("hook receipt candidate drift")
        if (
            not self.tap_shape
            or int(self.tap_shape[0]) < 1
            or any(int(item) < 1 for item in self.tap_shape)
            or self.base_support_id_backbone_forwards != 1
            or self.adapted_support_id_backbone_forwards != 1
            or self.base_query_id_backbone_forwards != 1
            or self.adapted_query_id_backbone_forwards != 1
            or self.state_fit_calls != 1
            or self.query_rows_used_for_fit != 0
            or self.query_state_updates != 0
            or self.query_selection_count != 0
            or self.query_gradient_calls != 0
            or self.phase2_optimizer_steps != 0
            or not self.checkpoint_parameters_frozen
            or not self.same_model_downstream
            or not self.same_forward_tap_and_final_pre_relu
        ):
            raise D127CheckpointHookError("D127 hook/forward receipt contract drift")
        expected_loss_forwards = 1 if self.candidate_id in (da.CANDIDATE_A, da.CANDIDATE_B) else 0
        if self.support_state_loss_id_backbone_forwards != expected_loss_forwards:
            raise D127CheckpointHookError("D127 support-loss forward count drift")

    @property
    def total_id_backbone_forwards(self) -> int:
        return (
            self.base_support_id_backbone_forwards
            + self.support_state_loss_id_backbone_forwards
            + self.adapted_support_id_backbone_forwards
            + self.base_query_id_backbone_forwards
            + self.adapted_query_id_backbone_forwards
        )

    def as_dict(self) -> Mapping[str, Any]:
        return _frozen_mapping(
            {
                "schema": SCHEMA,
                "candidate_id": self.candidate_id,
                "tap_name": self.tap_name,
                "tap_shape": tuple(int(item) for item in self.tap_shape),
                "base_support_id_backbone_forwards": self.base_support_id_backbone_forwards,
                "support_state_loss_id_backbone_forwards": self.support_state_loss_id_backbone_forwards,
                "adapted_support_id_backbone_forwards": self.adapted_support_id_backbone_forwards,
                "base_query_id_backbone_forwards": self.base_query_id_backbone_forwards,
                "adapted_query_id_backbone_forwards": self.adapted_query_id_backbone_forwards,
                "total_id_backbone_forwards": self.total_id_backbone_forwards,
                "state_fit_calls": self.state_fit_calls,
                "query_rows_used_for_fit": self.query_rows_used_for_fit,
                "query_state_updates": self.query_state_updates,
                "query_selection_count": self.query_selection_count,
                "query_gradient_calls": self.query_gradient_calls,
                "phase2_optimizer_steps": self.phase2_optimizer_steps,
                "checkpoint_parameters_frozen": self.checkpoint_parameters_frozen,
                "same_model_downstream": self.same_model_downstream,
                "same_forward_tap_and_final_pre_relu": self.same_forward_tap_and_final_pre_relu,
            }
        )


@dataclass(frozen=True, slots=True)
class D127CheckpointMaterialization:
    """One state and the exact base/adapted caches consumed by the four-arm core."""

    candidate_id: str
    state: D127State
    base_cache: D127FeatureCache
    adapted_cache: D127FeatureCache
    hook_receipt: D127HookReceipt

    def __post_init__(self) -> None:
        if self.candidate_id != _require_asset_from_state(self.state):
            raise D127CheckpointHookError("materialization candidate/state drift")
        if (
            self.base_cache.representation != "base_zid160"
            or self.adapted_cache.representation != "adapted_zid160"
            or self.base_cache.support_zid.shape != self.adapted_cache.support_zid.shape
            or self.base_cache.query_zid.shape != self.adapted_cache.query_zid.shape
            or self.hook_receipt.candidate_id != self.candidate_id
        ):
            raise D127CheckpointHookError("materialization cache/receipt drift")


def _require_asset_from_state(state: D127State) -> str:
    if isinstance(state, da.FSRGState):
        if state.candidate_id not in (da.CANDIDATE_A, da.CANDIDATE_B):
            raise D127CheckpointHookError("FSRG state candidate drift")
        return state.candidate_id
    if isinstance(state, da.RDHAState):
        return da.CANDIDATE_C
    raise D127CheckpointHookError("D127 state type drift")


@torch.no_grad()
def materialize_d127_query(
    model: nn.Module,
    query_iq: Tensor,
    *,
    asset: D127Asset,
    state: D127State,
) -> np.ndarray:
    """Return adapted query z160 with no labels, truth, fit, or state update.

    The deliberately small signature is an executable query-isolation boundary:
    support labels belong only to :func:`materialize_d127_candidate`; this API
    cannot receive labels, roles, truth, quotas, or an optimizer handle.
    """

    candidate_id = _require_asset(asset)
    _require_state(asset, state)
    _validate_received_iq(model, query_iq)
    binding = _bind_candidate(model, candidate_id)
    forward = _forward_adapted(model, query_iq, binding, asset, state)
    return _tensor_to_readonly_zid(forward.z_id, name="adapted_query_zid")


def materialize_d127_candidate(
    model: nn.Module,
    support_iq: Tensor,
    support_labels: Tensor,
    query_iq: Tensor,
    *,
    asset: D127Asset,
) -> D127CheckpointMaterialization:
    """Fit one support-only D127 state and materialize its four z160 caches.

    The exact operation order is immutable:

    ``base support -> support state -> adapted support -> base query -> adapted query``.

    In particular, query IQ is not visible to the state fit.  The returned
    adapted cache is reused unchanged by both ``M_DA`` and ``M_JOINT``.
    """

    candidate_id = _require_asset(asset)
    _validate_received_iq(model, support_iq)
    _validate_received_iq(model, query_iq)
    if support_iq.device != query_iq.device:
        raise D127CheckpointHookError("support/query IQ must share the checkpoint device")
    binding = _bind_candidate(model, candidate_id)
    support_labels = _labels_for_support(
        support_labels, rows=len(support_iq), device=support_iq.device
    )

    base_support = _id_backbone_forward(
        model, support_iq, binding, intervention=None, require_grad=False
    )
    state, loss_forwards = _fit_support_state(
        model, support_iq, support_labels, binding, asset, base_support
    )
    _require_state(asset, state)
    adapted_support = _forward_adapted(model, support_iq, binding, asset, state)
    base_query = _id_backbone_forward(
        model, query_iq, binding, intervention=None, require_grad=False
    )
    adapted_query = _forward_adapted(model, query_iq, binding, asset, state)

    base_cache = D127FeatureCache(
        representation="base_zid160",
        support_zid=_tensor_to_readonly_zid(base_support.z_id, name="base_support_zid"),
        query_zid=_tensor_to_readonly_zid(base_query.z_id, name="base_query_zid"),
        receipt={},
    )
    adapted_cache = D127FeatureCache(
        representation="adapted_zid160",
        support_zid=_tensor_to_readonly_zid(
            adapted_support.z_id, name="adapted_support_zid"
        ),
        query_zid=_tensor_to_readonly_zid(
            adapted_query.z_id, name="adapted_query_zid"
        ),
        receipt={},
    )
    hook_receipt = D127HookReceipt(
        candidate_id=candidate_id,
        tap_name=binding.tap_name,
        tap_shape=tuple(int(item) for item in base_support.tap.shape),
        base_support_id_backbone_forwards=1,
        support_state_loss_id_backbone_forwards=loss_forwards,
        adapted_support_id_backbone_forwards=1,
        base_query_id_backbone_forwards=1,
        adapted_query_id_backbone_forwards=1,
        state_fit_calls=1,
        query_rows_used_for_fit=0,
        query_state_updates=0,
        query_selection_count=0,
        query_gradient_calls=0,
        phase2_optimizer_steps=0,
        checkpoint_parameters_frozen=not any(
            parameter.requires_grad for parameter in model.parameters()
        ),
        same_model_downstream=True,
        same_forward_tap_and_final_pre_relu=True,
    )
    if state.receipt.protocol_closed is not True:
        raise D127CheckpointHookError("DA core state receipt did not close protocol counters")
    return D127CheckpointMaterialization(
        candidate_id=candidate_id,
        state=state,
        base_cache=base_cache,
        adapted_cache=adapted_cache,
        hook_receipt=hook_receipt,
    )


def query_api_forbidden_parameter_names() -> tuple[str, ...]:
    """Return an auditable query API denial list without adding a query surface."""

    signature = inspect.signature(materialize_d127_query)
    forbidden = tuple(
        name
        for name in signature.parameters
        if any(token in name.lower() for token in ("label", "truth", "role", "quota", "update", "optimizer"))
    )
    return forbidden


__all__ = [
    "D127Asset",
    "D127CheckpointHookError",
    "D127CheckpointMaterialization",
    "D127Phase1CheckpointBridge",
    "D127Phase1CheckpointForward",
    "D127Phase1EpisodeIQ",
    "D127Phase1RawIQEpisode",
    "D127FeatureCache",
    "D127HookReceipt",
    "D127State",
    "HIDDEN_DIM",
    "INPUT_LEN",
    "IQ_CHANNELS",
    "SCHEMA",
    "SUPPORT_TEMPERATURE",
    "Z_DIM",
    "freeze_d127_checkpoint_model",
    "load_d127_frozen_checkpoint",
    "materialize_d127_candidate",
    "materialize_d127_query",
    "query_api_forbidden_parameter_names",
]
