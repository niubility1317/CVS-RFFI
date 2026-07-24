"""Formal closed loop for ``GRB-JP4-ADV-DRQKNN-BCRR/r1-sealed``.

The formal entry consumes only an already verified production deployment
bundle.  It owns support encoding, the four-coefficient closed-form fit,
strict INT8 replay, in-place merge, matched S_B/S_C banks and five-arm query
prediction.  Algebra-only helpers remain explicitly development-only.
"""
from __future__ import annotations

import hashlib
import gc
import io
import json
import struct
import time
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from cvsrffi.stage2_adv3b02_ts_drqknn_bcrr import (
    ADV3B02StateError,
    DualQKNNState,
    _directional_dual_loo,
    _raw_directional_loo,
    append_stage2_c as _parent_append_stage2_c,
    bcrr_fused_logits,
    build_stage2_b_state as _parent_build_stage2_b_state,
    dual_qknn_logits,
    fit_bcrr_branch,
    qknn_logits,
    repair_finite_exact_zero_singleton_class_medoid as _repair_support,
    sha256_bytes,
    typed_tokens,
    verify_zid_repair_receipt as _verify_repair_receipt,
)
from cvsrffi.phase1_adv3b02_deployment_bundle import (
    COMPONENT_PROFILE_GRB_JP4_Q4,
    FORMAL_CONTEXT_SCHEMA,
    VerifiedADV3B02DeploymentBundle,
    reverify_formal_adv3b02_deployment_bundle,
)
from cvsrffi.phase1_grb_jp4_bundle import (
    GRBJP4CompactComponent,
    NPZ_NAME as GRB_COMPONENT_NPZ_NAME,
)
from cvsrffi.stage2_predictor_bundle import (
    canonical_json_bytes as _bundle_canonical_json_bytes,
)
from model_dual_cvsincnet import backbone_forward_compat


CANDIDATE = "GRB-JP4-ADV-DRQKNN-BCRR/r1-sealed"
SCHEMA = "cvs.stage2.grb_jp4_adv_drqknn_bcrr.r1_sealed"
RANK = 4
Z_DIM = 160
HIDDEN_DIM = 320
OLD_CLASS_COUNT = 6
K_VALUES = (1, 5, 10)
THETA_WIRE_BYTES = 6
GROUND_NUMERIC_PAYLOAD_BYTES = 2914
JP4_WIRE_LIMIT = 4096
MAX_STATE_BYTES = 256 * 1024
PARENT_R6_MAX_STATE_BYTES = 159_691
COMBINED_MAX_STATE_BYTES = 163_787
HEAD_MAC_C26_K10 = 42_466
FIT_STATE_WIRE_SCHEMA = "cvs.stage2.grb_jp4.fit_state_wire.v1"
FORMAL_ORCHESTRATOR_SCHEMA = "cvs.stage2.grb_jp4.formal_orchestrator.v1"
FIVE_ARM_NAMES = ("M0", "M_DA_NG", "M_DA", "M_OTHER", "M_JOINT")


class GRBJP4SpikeError(ValueError):
    """A JP4 shape, binding, or lifecycle invariant was violated."""


def _canon(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _bundle_semantic_sha(value: Mapping[str, Any]) -> str:
    return _sha(_bundle_canonical_json_bytes(dict(value)))


def _require_sha(value: Any, name: str) -> str:
    if (type(value) is not str or len(value) != 64 or value.lower() != value
            or any(c not in "0123456789abcdef" for c in value)):
        raise GRBJP4SpikeError(f"{name} must be a lower-case SHA256")
    return value


def _formal_bundle_receipt_from_reverified(
    bundle: VerifiedADV3B02DeploymentBundle,
) -> tuple[Mapping[str, Any], str]:
    """Check loader facts on a fresh external-chain re-materialization only."""

    if (
        type(bundle) is not VerifiedADV3B02DeploymentBundle
        or bundle._issued_runtime_identity != id(bundle.runtime)
    ):
        raise GRBJP4SpikeError(
            "formal GRB entry requires a fresh loader-issued bundle"
        )
    context = dict(bundle.formal_phase2_context)
    receipt = dict(bundle.verification_receipt)
    audit = dict(bundle.audit)
    if (
        context.get("schema") != FORMAL_CONTEXT_SCHEMA
        or context.get("component_profile") != COMPONENT_PROFILE_GRB_JP4_Q4
        or context.get("component_inner_filename") != GRB_COMPONENT_NPZ_NAME
        or context.get("formal_phase2_eligible") is not True
        or context.get("standalone_component_formal_phase2_eligible") is not False
        or any(
            context.get(field) is not True
            for field in (
                "outer_signature_verified",
                "detached_seal_verified",
                "runtime_checkpoint_parity_verified",
            )
        )
    ):
        raise GRBJP4SpikeError("formal GRB profile or verified lifecycle drift")
    if (
        receipt.get("schema")
        != "cvs.phase1.adv3b02_verified_bundle_receipt.v1"
        or receipt.get("component_profile") != COMPONENT_PROFILE_GRB_JP4_Q4
        or audit.get("status") != "PASS"
        or audit.get("exact_root_member_allowlist") is not True
        or audit.get("raw_training_checkpoint_present") is not False
    ):
        raise GRBJP4SpikeError("formal GRB verification receipt/audit drift")
    receipt_sha = _bundle_semantic_sha(receipt)
    if (
        context.get("verified_bundle_receipt_sha256") != receipt_sha
        or audit.get("verified_bundle_receipt_sha256") != receipt_sha
        or receipt.get("outer_content_root_sha256")
        != context.get("outer_content_root_sha256")
        or audit.get("outer_content_root_sha256")
        != context.get("outer_content_root_sha256")
        or receipt.get("component_outer_slot_sha256")
        != context.get("component_outer_slot_sha256")
        or receipt.get("checkpoint_lineage_sha256")
        != context.get("checkpoint_lineage_sha256")
        or receipt.get("checkpoint_lineage_sha256")
        != context.get("checkpoint_sha256")
        or receipt.get("runtime_sha256") != context.get("runtime_sha256")
        or receipt.get("parity_receipt_sha256")
        != context.get("parity_receipt_sha256")
        or receipt.get("method_lock_sha256") != context.get("method_lock_sha256")
    ):
        raise GRBJP4SpikeError(
            "formal GRB outer-root/component/checkpoint member binding drift"
        )
    for field in (
        "verified_bundle_receipt_sha256",
        "outer_content_root_sha256",
        "component_outer_slot_sha256",
        "checkpoint_sha256",
        "runtime_sha256",
        "parity_receipt_sha256",
        "method_lock_sha256",
    ):
        _require_sha(context.get(field), field)

    if type(bundle.component) is not GRBJP4CompactComponent:
        raise GRBJP4SpikeError("formal GRB component type drift")
    component_manifest = dict(bundle.component.manifest)
    if (
        _bundle_semantic_sha(component_manifest)
        != receipt.get("component_manifest_semantic_sha256")
        or
        component_manifest.get("component_profile")
        != COMPONENT_PROFILE_GRB_JP4_Q4
        or component_manifest.get("formal_phase2_eligible") is not False
        or component_manifest.get("component_state")
        != "PENDING_OUTER_JOINT_SEAL"
        or component_manifest.get("checkpoint_sha256")
        != context["checkpoint_sha256"]
        or component_manifest.get("component_npz_sha256")
        != context["component_outer_slot_sha256"]
    ):
        raise GRBJP4SpikeError("formal GRB component signed-slot binding drift")

    parity = dict(bundle.parity_receipt)
    if (
        _bundle_semantic_sha(parity)
        != receipt.get("parity_receipt_semantic_sha256")
        or parity.get("schema")
        != "cvs.phase1.runtime_checkpoint_parity_receipt.v1"
        or parity.get("parity_status") != "PASS"
        or parity.get("checkpoint_lineage_sha256") != context["checkpoint_sha256"]
        or parity.get("runtime_sha256") != context["runtime_sha256"]
        or parity.get("runtime_structure_sha256")
        != receipt.get("runtime_structure_sha256")
    ):
        raise GRBJP4SpikeError("formal checkpoint/runtime parity binding drift")

    method = dict(bundle.method_lock)
    if (
        _bundle_semantic_sha(method)
        != receipt.get("method_lock_semantic_sha256")
        or method.get("schema") != "cvs.phase1.adv3b02_method_lock.v1"
        or method.get("method_id") != "ADV3B02-GRB-JP4"
        or method.get("checkpoint_lineage_sha256") != context["checkpoint_sha256"]
        or method.get("runtime_sha256") != context["runtime_sha256"]
        or method.get("component_pre_sign_content_root_sha256")
        != context["component_pre_sign_content_root_sha256"]
        or method.get("class_handle_binding_sha256")
        != context["class_handle_binding_sha256"]
        or method.get("parity_receipt_sha256")
        != context["parity_receipt_sha256"]
        or method.get("generation_lock_sha256")
        != context["generation_lock_sha256"]
    ):
        raise GRBJP4SpikeError("formal GRB method-lock binding drift")

    binding = dict(bundle.class_binding)
    rows = binding.get("class_id_to_handle")
    if (
        _bundle_semantic_sha(binding)
        != receipt.get("class_binding_semantic_sha256")
        or binding.get("checkpoint_lineage_sha256") != context["checkpoint_sha256"]
        or binding.get("class_handle_binding_sha256")
        != context["class_handle_binding_sha256"]
        or type(rows) is not list
        or tuple(row.get("class_handle") for row in rows)
        != tuple(bundle.component.class_registry)
    ):
        raise GRBJP4SpikeError("formal GRB ordered class binding drift")

    generation = dict(bundle.generation_lock)
    if (
        _bundle_semantic_sha(generation)
        != receipt.get("generation_lock_semantic_sha256")
        or generation.get("schema")
        != "cvs.phase1.prototype_generation_lock.v1"
        or generation.get("checkpoint_lineage_sha256")
        != context["checkpoint_sha256"]
        or generation.get("component_pre_sign_content_root_sha256")
        != context["component_pre_sign_content_root_sha256"]
        or generation.get("class_handle_binding_sha256")
        != context["class_handle_binding_sha256"]
        or generation.get("generation_config_sha256")
        != context["generation_config_sha256"]
        or generation.get("generation_code_sha256")
        != context["generation_code_sha256"]
    ):
        raise GRBJP4SpikeError("formal GRB generation-lock binding drift")

    runtime = bundle.runtime
    method_names = ()
    try:
        method_names = tuple(runtime._c._method_names())
    except (AttributeError, RuntimeError):
        pass
    if "grb_jp4_forward" not in method_names:
        raise GRBJP4SpikeError(
            "sealed TorchScript runtime lacks exported grb_jp4_forward"
        )
    return receipt, receipt_sha


def _reverified_formal_bundle_receipt(
    bundle: VerifiedADV3B02DeploymentBundle,
) -> tuple[VerifiedADV3B02DeploymentBundle, Mapping[str, Any], str]:
    """Re-run the outer seal/member/signature chain before formal consumption."""

    try:
        fresh = reverify_formal_adv3b02_deployment_bundle(bundle)
    except Exception as exc:
        raise GRBJP4SpikeError(
            "formal GRB bundle external re-verification failed"
        ) from exc
    receipt, receipt_sha = _formal_bundle_receipt_from_reverified(fresh)
    return fresh, receipt, receipt_sha


def _formal_bundle_receipt(
    bundle: VerifiedADV3B02DeploymentBundle,
) -> tuple[Mapping[str, Any], str]:
    """Compatibility helper that never trusts the supplied materialization."""

    _fresh, receipt, receipt_sha = _reverified_formal_bundle_receipt(bundle)
    return receipt, receipt_sha


def _joint_proj_linear(model: Any) -> Any:
    try:
        linear = model.id_backbone.cls_head.joint_proj[0]
    except (AttributeError, IndexError, TypeError, NotImplementedError):
        try:
            linear = dict(model.named_modules())[
                "id_backbone.cls_head.joint_proj.0"
            ]
        except (AttributeError, KeyError) as exc:
            raise GRBJP4SpikeError("ADV3B02 joint_proj.0 path is absent") from exc
    try:
        weight = linear.weight
    except AttributeError as exc:
        raise GRBJP4SpikeError("ADV3B02 joint_proj.0 weight is absent") from exc
    if (
        not torch.is_tensor(weight)
        or tuple(weight.shape) != (Z_DIM, HIDDEN_DIM)
        or weight.dtype != torch.float32
    ):
        raise GRBJP4SpikeError("ADV3B02 joint_proj.0 contract drift")
    return linear


def _rows(value: Any, width: int, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype != np.float32 or result.ndim != 2 or result.shape[1] != width or not len(result):
        raise GRBJP4SpikeError(f"{name} must be nonempty float32 [N,{width}]")
    return np.ascontiguousarray(result)


def _unit(rows: np.ndarray) -> np.ndarray:
    value = np.asarray(rows, np.float64)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise GRBJP4SpikeError("z_id rows must be finite rank-two values")
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    if np.any(norms <= 1.0e-12) or not np.isfinite(norms).all():
        raise GRBJP4SpikeError("z_id rows must be nonzero")
    return np.ascontiguousarray(np.asarray(value / norms, np.float32))


def _quantize_rows(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(rows, np.float32)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise GRBJP4SpikeError("INT8 rows must be finite rank-two values")
    scale = np.maximum(np.max(np.abs(value), axis=1), 1.0e-8) / 127.0
    codes = np.clip(np.rint(value / scale[:, None]), -127, 127).astype(np.int8)
    return np.ascontiguousarray(codes), np.ascontiguousarray(scale.astype("<f2"))


def _decode_rows(codes: np.ndarray, scales: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    if (codes.dtype != np.int8 or codes.shape != shape or scales.dtype != np.dtype("<f2")
            or scales.shape != (shape[0],) or np.any(codes == -128)
            or not np.isfinite(scales).all() or np.any(scales <= 0.0)):
        raise GRBJP4SpikeError(f"{name} INT8/FP16 layout drift")
    result = codes.astype(np.float32) * scales.astype(np.float32)[:, None]
    if not np.isfinite(result).all():
        raise GRBJP4SpikeError(f"{name} decoded nonfinite")
    return np.ascontiguousarray(result)


def _weight_bytes(weight: torch.Tensor | np.ndarray) -> bytes:
    if torch.is_tensor(weight):
        array = weight.detach().cpu().contiguous().numpy()
    else:
        array = np.asarray(weight)
    if array.shape != (Z_DIM, HIDDEN_DIM) or array.dtype != np.float32 or not np.isfinite(array).all():
        raise GRBJP4SpikeError("joint_proj.0 weight must be finite float32 [160,320]")
    return np.ascontiguousarray(array).tobytes()


def _theta_codec(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(theta, np.float32)
    if value.shape != (RANK,) or not np.isfinite(value).all():
        raise GRBJP4SpikeError("theta must be finite float32 [4]")
    # A zero theta uses scale=1 so the mandatory FP16 scale remains legal and
    # decoded zero is exact.  No FP32 theta is persisted in a state.
    peak = float(np.max(np.abs(value)))
    scale = np.float16(1.0 if peak == 0.0 else max(peak / 127.0, float(np.finfo(np.float16).smallest_subnormal)))
    codes = np.clip(np.rint(value / np.float32(scale)), -127, 127).astype(np.int8)
    return np.ascontiguousarray(codes), np.asarray(scale, dtype="<f2")


def _decode_theta(codes: np.ndarray, scale: np.ndarray) -> np.ndarray:
    if (codes.dtype != np.int8 or codes.shape != (RANK,) or np.any(codes == -128)
            or scale.dtype != np.dtype("<f2") or scale.shape != ()
            or not np.isfinite(scale).all() or float(scale) <= 0.0):
        raise GRBJP4SpikeError("theta INT8/FP16 state drift")
    return np.ascontiguousarray(codes.astype(np.float32) * np.float32(scale))


@dataclass(frozen=True)
class GroundReceiverBasis:
    """The compact, checkpoint-bound Phase1 ground component.

    It contains only aggregate INT8 factors and class anchors; it cannot carry
    source rows, source paths, member IDs, or a replaceable FP32 sidecar.
    """

    prototype_codes: np.ndarray
    prototype_scales: np.ndarray
    left_codes: np.ndarray
    left_scales: np.ndarray
    right_codes: np.ndarray
    right_scales: np.ndarray
    kappa_ground: float
    old_class_order: tuple[str, ...]
    checkpoint_sha256: str
    joint_weight_sha256: str
    method_lock_sha256: str
    generation_digest: str

    def __post_init__(self) -> None:
        _decode_rows(self.prototype_codes, self.prototype_scales, (OLD_CLASS_COUNT, Z_DIM), "P_g")
        _decode_rows(self.left_codes, self.left_scales, (RANK, Z_DIM), "L_g")
        _decode_rows(self.right_codes, self.right_scales, (RANK, HIDDEN_DIM), "R")
        if not np.isfinite(self.kappa_ground) or self.kappa_ground <= 0.0:
            raise GRBJP4SpikeError("kappa_G must be finite positive")
        if (len(self.old_class_order) != OLD_CLASS_COUNT or len(set(self.old_class_order)) != OLD_CLASS_COUNT
                or any(type(v) is not str or not v for v in self.old_class_order)):
            raise GRBJP4SpikeError("ground old-class binding must contain six unique typed tokens")
        for name in ("checkpoint_sha256", "joint_weight_sha256", "method_lock_sha256", "generation_digest"):
            _require_sha(getattr(self, name), name)

    @classmethod
    def from_decoded(cls, *, prototypes: np.ndarray, left: np.ndarray, right: np.ndarray,
                     kappa_ground: float, old_class_order: Sequence[str], checkpoint_sha256: str,
                     joint_weight_sha256: str, method_lock_sha256: str,
                     generation_digest: str) -> "GroundReceiverBasis":
        p = _unit(_rows(prototypes, Z_DIM, "P_g"))
        l = _rows(left, Z_DIM, "L_g")
        r = _rows(right, HIDDEN_DIM, "R")
        pc, ps = _quantize_rows(p); lc, ls = _quantize_rows(l); rc, rs = _quantize_rows(r)
        return cls(pc, ps, lc, ls, rc, rs, float(kappa_ground), tuple(old_class_order),
                   checkpoint_sha256, joint_weight_sha256, method_lock_sha256, generation_digest)

    @classmethod
    def from_verified_joint_component(
        cls, component: Any, *, formal_phase2_context: Mapping[str, Any],
        checkpoint_weight: torch.Tensor | np.ndarray, method_lock_sha256: str,
    ) -> "GroundReceiverBasis":
        """Development adapter for an already separated component/context.

        Formal orchestration never calls this separable surface; it consumes a
        single ``VerifiedADV3B02DeploymentBundle`` through
        :meth:`from_verified_bundle`.
        """
        context = dict(formal_phase2_context)
        required = ("formal_phase2_eligible", "outer_signature_verified",
                    "detached_seal_verified", "runtime_checkpoint_parity_verified")
        if any(context.get(name) is not True for name in required):
            raise GRBJP4SpikeError("GRB component lacks a verified production outer joint seal")
        manifest = dict(getattr(component, "manifest", {}))
        if (manifest.get("formal_phase2_eligible") is not False
                or manifest.get("component_state") != "PENDING_OUTER_JOINT_SEAL"):
            raise GRBJP4SpikeError("GRB component manifest lifecycle drift")
        checkpoint_sha = _require_sha(manifest.get("checkpoint_sha256"), "component checkpoint_sha256")
        if context.get("checkpoint_sha256") != checkpoint_sha:
            raise GRBJP4SpikeError("outer joint-seal/component checkpoint binding drift")
        raw_weight = np.frombuffer(_weight_bytes(checkpoint_weight), dtype=np.float32).reshape(Z_DIM, HIDDEN_DIM)
        arrays = (
            (getattr(component, "p_g_q", None), getattr(component, "p_g_scale", None), (OLD_CLASS_COUNT, Z_DIM), "P_g"),
            (getattr(component, "l_g_q", None), getattr(component, "l_g_scale", None), (RANK, Z_DIM), "L_g"),
            (getattr(component, "r_q", None), getattr(component, "r_scale", None), (RANK, HIDDEN_DIM), "R"),
        )
        decoded: list[tuple[np.ndarray, np.ndarray]] = []
        for codes, scales, shape, name in arrays:
            c_raw, s_raw = np.asarray(codes), np.asarray(scales)
            if c_raw.dtype != np.int8 or s_raw.dtype != np.dtype("<f2"):
                raise GRBJP4SpikeError(f"{name} verified component dtype drift")
            c = np.ascontiguousarray(c_raw); s = np.ascontiguousarray(s_raw)
            _decode_rows(c, s, shape, name); decoded.append((c, s))
        registry = tuple(getattr(component, "class_registry", ()))
        generation = _require_sha(manifest.get("source_aggregate_generation_digest_sha256"), "component generation digest")
        return cls(decoded[0][0], decoded[0][1], decoded[1][0], decoded[1][1], decoded[2][0], decoded[2][1],
                   float(getattr(component, "kappa_g", np.nan)), registry, checkpoint_sha, _sha(raw_weight.tobytes()),
                   _require_sha(method_lock_sha256, "method_lock_sha256"), generation)

    @classmethod
    def from_verified_bundle(
        cls, bundle: VerifiedADV3B02DeploymentBundle
    ) -> "GroundReceiverBasis":
        """Build the only formal ground basis directly from the verified bundle."""

        fresh, _receipt, _receipt_sha = _reverified_formal_bundle_receipt(bundle)
        return cls._from_reverified_bundle(fresh)

    @classmethod
    def _from_reverified_bundle(
        cls, bundle: VerifiedADV3B02DeploymentBundle
    ) -> "GroundReceiverBasis":
        """Build from the fresh materialization held by formal orchestration."""

        linear = _joint_proj_linear(bundle.runtime)
        context = dict(bundle.formal_phase2_context)
        return cls.from_verified_joint_component(
            bundle.component,
            formal_phase2_context=context,
            checkpoint_weight=linear.weight,
            method_lock_sha256=context["method_lock_sha256"],
        )

    def prototypes(self) -> np.ndarray:
        return _unit(_decode_rows(self.prototype_codes, self.prototype_scales, (OLD_CLASS_COUNT, Z_DIM), "P_g"))

    def left(self) -> np.ndarray:
        return _decode_rows(self.left_codes, self.left_scales, (RANK, Z_DIM), "L_g")

    def right(self) -> np.ndarray:
        return _decode_rows(self.right_codes, self.right_scales, (RANK, HIDDEN_DIM), "R")

    def wire_bytes(self) -> bytes:
        header = _canon({"schema": SCHEMA, "kind": "ground_receiver_basis", "kappa_ground": float(self.kappa_ground),
                         "old_class_order": list(self.old_class_order), "checkpoint_sha256": self.checkpoint_sha256,
                         "joint_weight_sha256": self.joint_weight_sha256, "method_lock_sha256": self.method_lock_sha256,
                         "generation_digest": self.generation_digest})
        return b"".join((header, self.prototype_codes.tobytes(), self.prototype_scales.tobytes(),
                          self.left_codes.tobytes(), self.left_scales.tobytes(),
                          self.right_codes.tobytes(), self.right_scales.tobytes()))

    @property
    def digest(self) -> str:
        return _sha(self.wire_bytes())


@dataclass(frozen=True)
class JP4FitState:
    """A sealed Stage2-B theta replay with no FP32 theta or DeltaW field."""

    theta_codes: np.ndarray
    theta_scale: np.ndarray
    k_shot: int
    ground_digest: str
    checkpoint_sha256: str
    joint_weight_sha256_before: str
    joint_weight_semantic_sha256: str
    fit_receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        theta = _decode_theta(self.theta_codes, self.theta_scale)
        if type(self.k_shot) is not int or self.k_shot not in K_VALUES:
            raise GRBJP4SpikeError("JP4 state K must be one of 1/5/10")
        for name in ("ground_digest", "checkpoint_sha256", "joint_weight_sha256_before", "joint_weight_semantic_sha256"):
            _require_sha(getattr(self, name), name)
        receipt = dict(self.fit_receipt)
        if (receipt.get("schema") != SCHEMA or receipt.get("query_rows_used_for_fit") != 0
                or receipt.get("optimizer_steps") != 0 or receipt.get("k_shot") != self.k_shot
                or receipt.get("ground_digest") != self.ground_digest
                or receipt.get("checkpoint_sha256") != self.checkpoint_sha256
                or receipt.get("joint_weight_sha256_before") != self.joint_weight_sha256_before
                or receipt.get("joint_weight_semantic_sha256") != self.joint_weight_semantic_sha256):
            raise GRBJP4SpikeError("JP4 support-only receipt drift")
        if self.k_shot == 1 and (
            not np.array_equal(theta, np.zeros((RANK,), np.float32))
            or self.joint_weight_semantic_sha256 != self.joint_weight_sha256_before
            or receipt.get("fallback") != "K1_identity"
        ):
            raise GRBJP4SpikeError("K1 JP4 state must be exact identity")
        formal = receipt.get("formal_bundle_binding")
        if formal is not None:
            if (
                type(formal) is not dict
                or formal.get("candidate") != CANDIDATE
                or formal.get("component_profile") != COMPONENT_PROFILE_GRB_JP4_Q4
                or formal.get("checkpoint_sha256") != self.checkpoint_sha256
            ):
                raise GRBJP4SpikeError("JP4 formal bundle receipt drift")
            for field in (
                "verified_bundle_receipt_sha256",
                "outer_content_root_sha256",
                "component_outer_slot_sha256",
                "method_lock_sha256",
                "runtime_sha256",
            ):
                _require_sha(formal.get(field), f"formal_bundle_binding.{field}")

    def theta(self) -> np.ndarray:
        return _decode_theta(self.theta_codes, self.theta_scale)

    def wire_bytes(self) -> bytes:
        header = _canon({"schema": SCHEMA, "kind": "jp4_fit", "k_shot": self.k_shot,
                         "ground_digest": self.ground_digest, "checkpoint_sha256": self.checkpoint_sha256,
                         "joint_weight_sha256_before": self.joint_weight_sha256_before,
                         "joint_weight_semantic_sha256": self.joint_weight_semantic_sha256,
                         "fit_receipt_sha256": _sha(_canon(dict(self.fit_receipt)))})
        return header + self.theta_codes.tobytes() + self.theta_scale.tobytes()

    @property
    def digest(self) -> str:
        return _sha(self.wire_bytes())


def serialize_jp4_fit_state(state: JP4FitState) -> bytes:
    """Serialize a JP4 fit with canonical metadata and an authenticated trailer."""

    if type(state) is not JP4FitState:
        raise GRBJP4SpikeError("JP4 serializer requires an exact JP4FitState")
    payload = {
        "schema": FIT_STATE_WIRE_SCHEMA,
        "state_schema": SCHEMA,
        "theta_codes_hex": state.theta_codes.tobytes().hex(),
        "theta_scale_hex": state.theta_scale.tobytes().hex(),
        "k_shot": state.k_shot,
        "ground_digest": state.ground_digest,
        "checkpoint_sha256": state.checkpoint_sha256,
        "joint_weight_sha256_before": state.joint_weight_sha256_before,
        "joint_weight_semantic_sha256": state.joint_weight_semantic_sha256,
        "fit_receipt": dict(state.fit_receipt),
        "state_digest": state.digest,
    }
    raw = _canon(payload)
    return b"GRBJP4S1" + struct.pack(">I", len(raw)) + raw + hashlib.sha256(raw).digest()


def deserialize_jp4_fit_state(
    wire: bytes,
    *,
    expected_ground_digest: str,
    expected_checkpoint_sha256: str,
    expected_joint_weight_sha256_before: str,
    expected_formal_bundle_receipt_sha256: str | None,
) -> JP4FitState:
    """Fail closed on every byte and all external replay bindings."""

    if type(wire) is not bytes or len(wire) < 8 + 4 + 32 or len(wire) > 64 * 1024:
        raise GRBJP4SpikeError("JP4 fit-state wire size/type drift")
    if wire[:8] != b"GRBJP4S1":
        raise GRBJP4SpikeError("JP4 fit-state wire magic drift")
    size = struct.unpack(">I", wire[8:12])[0]
    if size != len(wire) - 12 - 32:
        raise GRBJP4SpikeError("JP4 fit-state wire length drift")
    raw, trailer = wire[12:12 + size], wire[12 + size:]
    if hashlib.sha256(raw).digest() != trailer:
        raise GRBJP4SpikeError("JP4 fit-state wire digest drift")
    try:
        payload = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GRBJP4SpikeError("JP4 fit-state JSON is invalid") from exc
    expected_keys = {
        "schema", "state_schema", "theta_codes_hex", "theta_scale_hex",
        "k_shot", "ground_digest", "checkpoint_sha256",
        "joint_weight_sha256_before", "joint_weight_semantic_sha256",
        "fit_receipt", "state_digest",
    }
    if (
        type(payload) is not dict
        or set(payload) != expected_keys
        or _canon(payload) != raw
        or payload.get("schema") != FIT_STATE_WIRE_SCHEMA
        or payload.get("state_schema") != SCHEMA
        or type(payload.get("fit_receipt")) is not dict
        or type(payload.get("theta_codes_hex")) is not str
        or type(payload.get("theta_scale_hex")) is not str
        or len(payload["theta_codes_hex"]) != RANK * 2
        or len(payload["theta_scale_hex"]) != 4
    ):
        raise GRBJP4SpikeError("JP4 fit-state canonical schema drift")
    try:
        codes = np.frombuffer(bytes.fromhex(payload["theta_codes_hex"]), dtype=np.int8).copy()
        scale = np.frombuffer(bytes.fromhex(payload["theta_scale_hex"]), dtype="<f2").copy().reshape(())
    except (ValueError, TypeError) as exc:
        raise GRBJP4SpikeError("JP4 fit-state numeric codec drift") from exc
    state = JP4FitState(
        np.ascontiguousarray(codes),
        np.asarray(scale, dtype="<f2"),
        payload["k_shot"],
        payload["ground_digest"],
        payload["checkpoint_sha256"],
        payload["joint_weight_sha256_before"],
        payload["joint_weight_semantic_sha256"],
        payload["fit_receipt"],
    )
    if (
        state.digest != payload["state_digest"]
        or state.ground_digest
        != _require_sha(expected_ground_digest, "expected_ground_digest")
        or state.checkpoint_sha256
        != _require_sha(expected_checkpoint_sha256, "expected_checkpoint_sha256")
        or state.joint_weight_sha256_before
        != _require_sha(
            expected_joint_weight_sha256_before,
            "expected_joint_weight_sha256_before",
        )
    ):
        raise GRBJP4SpikeError("JP4 fit-state external replay binding drift")
    formal = dict(state.fit_receipt).get("formal_bundle_binding")
    if expected_formal_bundle_receipt_sha256 is None:
        if formal is not None:
            raise GRBJP4SpikeError("formal JP4 state requires an expected bundle receipt")
    elif (
        type(formal) is not dict
        or formal.get("verified_bundle_receipt_sha256")
        != _require_sha(
            expected_formal_bundle_receipt_sha256,
            "expected_formal_bundle_receipt_sha256",
        )
    ):
        raise GRBJP4SpikeError("JP4 fit-state formal bundle replay drift")
    return state


@dataclass(frozen=True)
class GRBStage2BState:
    """Frozen JP4 theta plus the unmodified r6 Stage2-B old state."""

    jp4: JP4FitState
    parent_state: DualQKNNState
    old_bank_digest: str

    def __post_init__(self) -> None:
        if self.parent_state.domain.stage != "S_B" or self.parent_state.domain.k_shot != self.jp4.k_shot:
            raise GRBJP4SpikeError("S_B JP4/r6 lifecycle or K binding drift")
        if self.old_bank_digest != self.parent_state.id_bank.digest:
            raise GRBJP4SpikeError("S_B old bank digest drift")

    @property
    def digest(self) -> str:
        return _sha(self.jp4.wire_bytes() + self.parent_state.wire_bytes())


@dataclass(frozen=True)
class GRBStage2CState:
    """Append-only Stage2-C state; the JP4 replay is byte-identical to S_B."""

    jp4: JP4FitState
    parent_state: DualQKNNState
    frozen_stage_b_digest: str
    append_receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            self.parent_state.domain.stage != "S_C"
            or self.parent_state.domain.k_shot != self.jp4.k_shot
        ):
            raise GRBJP4SpikeError("S_C requires an appended r6 state at matched K")
        _require_sha(self.frozen_stage_b_digest, "frozen_stage_b_digest")
        receipt = dict(self.append_receipt)
        if (receipt.get("stage") != "S_C" or receipt.get("query_rows_used_for_fit") != 0
                or type(receipt.get("old_state_sha256")) is not str
                or len(receipt["old_state_sha256"]) != 64
                or self.parent_state.domain.frozen_old_digest is None
                or receipt.get("k_shot", self.jp4.k_shot) != self.jp4.k_shot):
            raise GRBJP4SpikeError("S_C parent append receipt drift")

    @property
    def digest(self) -> str:
        return _sha(self.jp4.wire_bytes() + self.parent_state.wire_bytes())


def observed_ground_left_factors(domain_shift: Any, domain_counts: Any) -> tuple[np.ndarray, dict[str, float | int]]:
    """Diagnostic-only canonical four ground directions from aggregate shifts."""
    shifts = np.asarray(domain_shift, np.float32); counts = np.asarray(domain_counts)
    if shifts.ndim != 2 or shifts.shape[1] != Z_DIM or counts.shape != (len(shifts),):
        raise GRBJP4SpikeError("diagnostic domain-shift layout drift")
    observed = counts > 0
    if int(observed.sum()) < RANK:
        raise GRBJP4SpikeError("fewer than four observed diagnostic domains")
    centered = shifts[observed].astype(np.float64); centered -= centered.mean(axis=0, keepdims=True)
    _u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    if len(singular) < RANK or singular[RANK - 1] <= 0.0:
        raise GRBJP4SpikeError("diagnostic ground SVD cannot provide rank four")
    factors = np.asarray(vt[:RANK], np.float32)
    for row in factors:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0.0:
            row *= -1.0
    return factors, {"observed_domain_count": int(observed.sum()), "rank": RANK,
                     "energy_q4": float(np.sum(singular[:RANK] ** 2) / np.sum(singular ** 2)),
                     "condition_q4": float(singular[0] / singular[RANK - 1])}


def checkpoint_right_factors(weight: torch.Tensor) -> np.ndarray:
    raw = np.asarray(weight.detach().cpu().tolist(), np.float32)
    if raw.shape != (Z_DIM, HIDDEN_DIM):
        raise GRBJP4SpikeError("real joint_proj.0 weight must be [160,320]")
    _u, singular, vt = np.linalg.svd(raw.astype(np.float64), full_matrices=False)
    if len(singular) < RANK or singular[RANK - 1] <= 0.0:
        raise GRBJP4SpikeError("real joint weight cannot provide rank four")
    factors = np.asarray(vt[:RANK], np.float32)
    # The SVD is sign-indeterminate.  Persisting a deterministic orientation is
    # required because the ground component is checkpoint-bound and signed.
    for row in factors:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0.0:
            row *= -1.0
    return np.ascontiguousarray(factors)


def directions(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a, b = np.asarray(left, np.float32), np.asarray(right, np.float32)
    if a.shape != (RANK, Z_DIM) or b.shape != (RANK, HIDDEN_DIM):
        raise GRBJP4SpikeError("rank-four factor shape drift")
    return np.ascontiguousarray(np.einsum("ri,rj->rij", a, b).astype(np.float32))


@dataclass(frozen=True)
class StrictForward:
    z_id: np.ndarray
    hidden: np.ndarray
    pre_relu: np.ndarray
    hook_exact_bytes: bool
    z_dom: np.ndarray | None = None
    execution_path: str = "eager_forward_hook"


def strict_zid_with_hook(model: Any, iq: torch.Tensor) -> StrictForward:
    """Execute either the eager hook or sealed TorchScript functional tap."""

    if (
        bool(getattr(model, "training", True))
        or iq.dtype != torch.float32
        or iq.ndim != 3
        or iq.shape[1] != 2
        or not torch.isfinite(iq).all()
    ):
        raise GRBJP4SpikeError("strict z_id forward requires eval float32 [N,2,T]")

    method_names = ()
    try:
        method_names = tuple(model._c._method_names())
    except (AttributeError, RuntimeError):
        pass
    if "grb_jp4_forward" in method_names:
        linear = _joint_proj_linear(model)
        with torch.no_grad():
            output = model.grb_jp4_forward(iq)
        if (
            type(output) not in (tuple, list)
            or len(output) != 4
            or any(not torch.is_tensor(item) for item in output)
        ):
            raise GRBJP4SpikeError(
                "TorchScript grb_jp4_forward must return z_id/z_dom/hidden/pre_relu"
            )
        z_id, z_dom, hidden, pre_relu = output
        if (
            z_id.dtype != torch.float32
            or z_dom.dtype != torch.float32
            or hidden.dtype != torch.float32
            or pre_relu.dtype != torch.float32
            or tuple(z_id.shape) != (len(iq), Z_DIM)
            or z_dom.ndim != 2
            or len(z_dom) != len(iq)
            or tuple(hidden.shape) != (len(iq), HIDDEN_DIM)
            or tuple(pre_relu.shape) != (len(iq), Z_DIM)
            or any(
                not torch.isfinite(item).all()
                for item in (z_id, z_dom, hidden, pre_relu)
            )
        ):
            raise GRBJP4SpikeError("TorchScript GRB functional tap layout drift")
        with torch.no_grad():
            recomputed_pre = linear(hidden)
            recomputed_zid = torch.relu(recomputed_pre)
        if not torch.equal(pre_relu, recomputed_pre) or not torch.equal(
            z_id, recomputed_zid
        ):
            raise GRBJP4SpikeError(
                "TorchScript GRB functional tap is not byte-bound to joint_proj.0/ReLU"
            )
        return StrictForward(
            np.ascontiguousarray(z_id.detach().cpu().numpy(), dtype=np.float32),
            np.ascontiguousarray(hidden.detach().cpu().numpy(), dtype=np.float32),
            np.ascontiguousarray(pre_relu.detach().cpu().numpy(), dtype=np.float32),
            True,
            np.ascontiguousarray(z_dom.detach().cpu().numpy(), dtype=np.float32),
            "torchscript_exported_functional_tap",
        )

    try:
        linear = model.id_backbone.cls_head.joint_proj[0]; joint = model.id_backbone.cls_head.joint_proj
    except (AttributeError, IndexError, TypeError) as exc:
        raise GRBJP4SpikeError("real ADV3B02 joint_proj hook path is absent") from exc
    if not isinstance(linear, torch.nn.Linear) or tuple(linear.weight.shape) != (Z_DIM, HIDDEN_DIM):
        raise GRBJP4SpikeError("real ADV3B02 joint_proj.0 contract drift")
    caught: dict[str, torch.Tensor] = {}

    def capture_linear(_module: torch.nn.Module, args: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        # Forward hooks may replace the module output when they return a value.
        # This collector must return ``None`` so the sealed checkpoint forward
        # remains byte-for-byte its native execution.
        caught["hidden"] = args[0].detach().clone()
        caught["pre_relu"] = out.detach().clone()

    def capture_joint(_module: torch.nn.Module, _args: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        caught["joint"] = out.detach().clone()

    one = linear.register_forward_hook(capture_linear)
    two = joint.register_forward_hook(capture_joint)
    try:
        with torch.no_grad():
            aux = backbone_forward_compat(model.id_backbone, iq, y=None, return_aux=True, domain_labels=None)
    finally:
        one.remove(); two.remove()
    z = aux.get("feat_joint") if isinstance(aux, dict) else None
    if not torch.is_tensor(z) or not all(k in caught for k in ("hidden", "pre_relu", "joint")):
        raise GRBJP4SpikeError("formal feat_joint/hook capture is incomplete")
    if z is not caught["joint"] and not torch.equal(z, caught["joint"]):
        raise GRBJP4SpikeError("formal feat_joint differs from post-ReLU hook")
    exact_bytes = (
        np.ascontiguousarray(z.detach().cpu().numpy()).tobytes()
        == np.ascontiguousarray(caught["joint"].cpu().numpy()).tobytes()
    )
    if not exact_bytes:
        raise GRBJP4SpikeError("formal feat_joint hook byte binding drift")
    zdom = None
    if isinstance(aux, dict):
        for key in ("feat_dom", "feat_domain", "z_dom"):
            if torch.is_tensor(aux.get(key)):
                zdom = np.ascontiguousarray(
                    aux[key].detach().cpu().numpy(), dtype=np.float32
                )
                break
    return StrictForward(
        np.ascontiguousarray(z.detach().cpu().numpy(), dtype=np.float32),
        np.ascontiguousarray(caught["hidden"].cpu().numpy(), dtype=np.float32),
        np.ascontiguousarray(caught["pre_relu"].cpu().numpy(), dtype=np.float32),
        True,
        zdom,
        "eager_forward_hook",
    )


def analytic_jacobian(forward: StrictForward, dirs: np.ndarray) -> np.ndarray:
    if forward.hook_exact_bytes is not True:
        raise GRBJP4SpikeError("analytic Jacobian requires byte-bound formal hook")
    raw, h, pre = np.asarray(forward.z_id, np.float64), np.asarray(forward.hidden, np.float64), np.asarray(forward.pre_relu, np.float64)
    delta = np.einsum("rij,nj->nri", np.asarray(dirs, np.float64), h) * (pre[:, None, :] > 0.0)
    norm = np.linalg.norm(raw, axis=1, keepdims=True)
    if np.any(norm <= 1.0e-12) or not np.isfinite(norm).all():
        raise GRBJP4SpikeError("analytic Jacobian requires finite nonzero z_id")
    unit = raw / norm
    projected = delta - np.einsum("ni,nri->nr", unit, delta)[:, :, None] * unit[:, None, :]
    return np.ascontiguousarray(np.asarray(projected / norm[:, None, :], np.float32))


def autograd_jacobian(model: torch.nn.Module, iq: torch.Tensor, dirs: np.ndarray) -> np.ndarray:
    """Independent autograd reference used only by local correctness tests."""
    linear = model.id_backbone.cls_head.joint_proj[0]; weight = linear.weight
    name = next((key for key, value in model.id_backbone.named_parameters() if value is weight), None)
    if name is None: raise GRBJP4SpikeError("cannot bind real joint weight for autograd")
    from torch.func import functional_call, jacrev
    delta = torch.tensor(dirs, device=weight.device, dtype=weight.dtype)
    def forward(theta: torch.Tensor) -> torch.Tensor:
        overrides = {name: weight.detach() + torch.tensordot(theta, delta, dims=1)}
        try: aux = functional_call(model.id_backbone, overrides, (iq,), {"y": None, "return_aux": True, "domain_labels": None})
        except TypeError: aux = functional_call(model.id_backbone, overrides, (iq,), {"y": None, "return_aux": True})
        if not isinstance(aux, dict) or not torch.is_tensor(aux.get("feat_joint")):
            raise GRBJP4SpikeError("autograd strict backbone does not expose feat_joint")
        return F.normalize(aux["feat_joint"], dim=1)
    value = jacrev(forward)(torch.zeros((RANK,), device=weight.device, dtype=weight.dtype))
    return np.ascontiguousarray(np.asarray(value.detach().cpu().tolist(), np.float32).transpose(0, 2, 1))


def _balanced_k(labels: tuple[str, ...], classes: tuple[str, ...]) -> int:
    counts = tuple(labels.count(item) for item in classes)
    if len(set(counts)) != 1 or counts[0] not in K_VALUES:
        raise GRBJP4SpikeError("support must be exact balanced K in {1,5,10}")
    return counts[0]


def _fit_theta_stream(unit_z: np.ndarray, jacobian: np.ndarray, labels: tuple[str, ...],
                      classes: tuple[str, ...], prototypes: np.ndarray, *, kappa_ground: float,
                      checkpoint_weight: np.ndarray, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    k = _balanced_k(labels, classes)
    if k == 1:
        return np.zeros(RANK, np.float32), {"k_shot": 1, "lambda": 0.0, "rank_diagnostic": 0,
            "condition": 1.0, "g_k": 0.0, "s_kappa": 1.0, "s_weight": 1.0,
            "fallback": "K1_identity", "query_rows_used_for_fit": 0, "optimizer_steps": 0}
    if not np.isfinite(jacobian).all():
        return np.zeros(RANK, np.float32), {"k_shot": k, "lambda": 0.0, "rank_diagnostic": 0,
            "condition": 0.0, "g_k": float(k / (k + RANK)), "s_kappa": 0.0, "s_weight": 1.0,
            "fallback": "nonfinite_jacobian_identity", "query_rows_used_for_fit": 0, "optimizer_steps": 0}
    indices = np.asarray([classes.index(x) for x in labels], np.intp)
    residual = unit_z.astype(np.float64) - prototypes[indices].astype(np.float64)
    t = np.zeros((RANK, RANK), np.float64); u = np.zeros((RANK,), np.float64)
    # Each physical support is accumulated once.  1/K is the exact class
    # equalisation weight; no [6K*160,4] A matrix is materialized.
    for row in range(len(unit_z)):
        ji = jacobian[row].astype(np.float64)
        bi = residual[row]
        t += (ji @ ji.T) / float(k)
        u += (ji @ bi) / float(k)
    trace = float(np.trace(t)); threshold = 64.0 * np.finfo(np.float64).eps
    if not np.isfinite(t).all() or not np.isfinite(u).all() or not np.isfinite(trace) or trace <= threshold:
        return np.zeros(RANK, np.float32), {"k_shot": k, "lambda": 0.0, "rank_diagnostic": 0,
            "condition": 0.0, "g_k": float(k / (k + RANK)), "s_kappa": 0.0, "s_weight": 1.0,
            "fallback": "trace_or_numeric_identity", "query_rows_used_for_fit": 0, "optimizer_steps": 0}
    lam = 0.01 * trace / RANK; system = t + lam * np.eye(RANK, dtype=np.float64)
    try:
        theta_ridge = -np.linalg.solve(system, u); condition = float(np.linalg.cond(system))
    except np.linalg.LinAlgError:
        theta_ridge = np.zeros(RANK, np.float64); condition = float("inf")
    if not np.isfinite(theta_ridge).all() or not np.isfinite(condition) or condition <= 0.0:
        return np.zeros(RANK, np.float32), {"k_shot": k, "lambda": float(lam), "rank_diagnostic": int(np.linalg.matrix_rank(t)),
            "condition": 0.0, "g_k": float(k / (k + RANK)), "s_kappa": 0.0, "s_weight": 1.0,
            "fallback": "solve_nonfinite_identity", "query_rows_used_for_fit": 0, "optimizer_steps": 0}
    delta = np.einsum("r,ri,rj->ij", theta_ridge, left.astype(np.float64), right.astype(np.float64))
    delta_norm = float(np.linalg.norm(delta, ord="fro")); tau = float(np.linalg.norm(checkpoint_weight.astype(np.float64), ord="fro") / np.sqrt(Z_DIM))
    gk = float(k / (k + RANK)); sk = float(min(1.0, np.sqrt(kappa_ground / condition)))
    sw = float(min(1.0, tau / delta_norm)) if delta_norm > 0.0 and np.isfinite(delta_norm) else 1.0
    theta = np.asarray(gk * sk * sw * theta_ridge, np.float32)
    if not np.isfinite(theta).all(): theta = np.zeros(RANK, np.float32)
    return theta, {"k_shot": k, "lambda": float(lam), "rank_diagnostic": int(np.linalg.matrix_rank(t)),
        "condition": condition, "g_k": gk, "s_kappa": sk, "s_weight": sw, "fallback": "none",
        "query_rows_used_for_fit": 0, "optimizer_steps": 0}


def prepare_support_for_jp4_fit(*, support_zid: np.ndarray, support_jacobian: np.ndarray,
                                support_labels: Sequence[Any] | np.ndarray,
                                support_physical_tokens: Sequence[Any] | np.ndarray,
                                registered_old_classes: Sequence[Any] | np.ndarray,
                                support_repair_receipt: Mapping[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
    """Apply the existing support-only zero-row rule before JP4 algebra.

    For K5/K10, a single exact-zero row is replaced by the existing
    same-class medoid and receives that donor's analytic Jacobian.  K1 has no
    JP4 fit at all, so its finite zero is accepted unchanged and leads to the
    exact identity theta.  This helper never opens a query.
    """
    labels = typed_tokens(support_labels, name="JP4 repair labels")
    tokens = typed_tokens(support_physical_tokens, name="JP4 repair tokens", unique=True)
    classes = typed_tokens(registered_old_classes, name="JP4 repair class registry", unique=True)
    raw = _rows(support_zid, Z_DIM, "JP4 repair z_id")
    jac = np.asarray(support_jacobian, np.float32)
    if len(raw) != len(labels) or len(raw) != len(tokens) or jac.shape != (len(raw), RANK, Z_DIM):
        raise GRBJP4SpikeError("JP4 repair support/Jacobian layout drift")
    k = _balanced_k(labels, classes)
    if k == 1:
        return raw, np.ascontiguousarray(jac), {"schema": SCHEMA, "rule": "K1_identity_no_jp4_fit",
            "repaired_row_count": 0, "query_rows_used_for_fit": 0}
    try:
        if support_repair_receipt is None:
            repaired, receipt = _repair_support(raw, labels, classes, tokens)
        else:
            receipt = dict(_verify_repair_receipt(support_repair_receipt))
            if receipt.get("output_support_sha256") != _sha(raw.tobytes()):
                raise GRBJP4SpikeError("provided JP4 repair receipt does not bind supplied support")
            repaired = raw
    except ADV3B02StateError as exc:
        raise GRBJP4SpikeError("JP4 support repair closure failed") from exc
    adjusted = np.array(jac, dtype=np.float32, copy=True, order="C")
    zeros = np.flatnonzero(np.all(raw == np.float32(0.0), axis=1))
    for index in zeros:
        candidates = [pos for pos, label in enumerate(labels)
                      if label == labels[index] and pos != index and np.array_equal(raw[pos], repaired[index])]
        if len(candidates) != 1:
            raise GRBJP4SpikeError("JP4 repaired zero row has no unique same-class Jacobian donor")
        adjusted[index] = jac[candidates[0]]
    return np.ascontiguousarray(repaired), np.ascontiguousarray(adjusted), dict(receipt)


def solve_theta(unit_z: np.ndarray, jacobian: np.ndarray, class_index: np.ndarray, prototypes: np.ndarray) -> tuple[np.ndarray, dict[str, float | int]]:
    """Compatibility helper for the old spike; formal callers use support-IQ fit."""
    z = _unit(_rows(unit_z, Z_DIM, "support z_id")); j = np.asarray(jacobian, np.float32); ci = np.asarray(class_index, np.int64)
    p = _unit(_rows(prototypes, Z_DIM, "prototype"))
    if j.shape != (len(z), RANK, Z_DIM) or ci.shape != (len(z),) or np.any(ci < 0) or np.any(ci >= len(p)):
        raise GRBJP4SpikeError("support Jacobian/class layout drift")
    labels = tuple(str(x) for x in ci.tolist()); classes = tuple(str(x) for x in range(len(p)))
    # This legacy path retains its small synthetic two-class contract while
    # applying the same streaming ridge core (identity ground/weight shrink).
    left = np.zeros((RANK, Z_DIM), np.float32); left[np.arange(RANK), np.arange(RANK)] = 1.0
    right = np.zeros((RANK, HIDDEN_DIM), np.float32); right[np.arange(RANK), np.arange(RANK)] = 1.0
    weight = np.zeros((Z_DIM, HIDDEN_DIM), np.float32); weight[0, 0] = 1.0
    theta, receipt = _fit_theta_stream(z, j, labels, classes, p, kappa_ground=1.0,
                                        checkpoint_weight=weight, left=left, right=right)
    return theta, {"lambda": float(receipt["lambda"]), "rank": int(receipt["rank_diagnostic"]), "condition": float(receipt["condition"])}


def _semantic_weight_sha(weight: np.ndarray, ground: GroundReceiverBasis, theta: np.ndarray) -> str:
    left, right = ground.left(), ground.right(); digest = hashlib.sha256()
    for row in range(Z_DIM):
        delta_row = np.sum((theta * left[:, row])[:, None] * right, axis=0, dtype=np.float32)
        digest.update(np.ascontiguousarray(weight[row] + delta_row).astype(np.float32).tobytes())
    return digest.hexdigest()


def _fit_stage2_b_from_precomputed_jacobian_development_only(
    *, support_zid: np.ndarray, support_jacobian: np.ndarray,
    support_labels: Sequence[Any] | np.ndarray, support_physical_tokens: Sequence[Any] | np.ndarray,
    ground: GroundReceiverBasis, checkpoint_weight: torch.Tensor | np.ndarray,
    checkpoint_sha256: str, support_repair_receipt: Mapping[str, Any] | None = None,
    _teacher_theta_out: list[np.ndarray] | None = None,
) -> JP4FitState:
    """Development-only algebra entry; formal code must use support-IQ entry.

    A bare Jacobian is useful for deterministic unit tests, but it carries no
    proof that it was generated from the sealed checkpoint, ``L_g/R`` and the
    same physical support.  It is deliberately private and excluded from the
    public API so it cannot be a formal release path.
    """
    labels = typed_tokens(support_labels, name="JP4 S_B labels")
    tokens = typed_tokens(support_physical_tokens, name="JP4 S_B physical tokens", unique=True)
    if len(labels) != len(tokens): raise GRBJP4SpikeError("JP4 support token layout drift")
    if tuple(sorted(set(labels))) != tuple(sorted(ground.old_class_order)):
        raise GRBJP4SpikeError("JP4 support classes must exactly bind the six ground old classes")
    k = _balanced_k(labels, ground.old_class_order)
    repaired_z, jac, repair = prepare_support_for_jp4_fit(
        support_zid=support_zid, support_jacobian=support_jacobian, support_labels=labels,
        support_physical_tokens=tokens, registered_old_classes=ground.old_class_order,
        support_repair_receipt=support_repair_receipt)
    # K1 never uses a support feature/Jacobian to fit JP4; this preserves the
    # frozen identity even when a real support feature is exactly zero.
    z = np.zeros((len(repaired_z), Z_DIM), np.float32) if k == 1 else _unit(repaired_z)
    raw_weight = np.frombuffer(_weight_bytes(checkpoint_weight), dtype=np.float32).reshape(Z_DIM, HIDDEN_DIM)
    before = _sha(raw_weight.tobytes())
    if _require_sha(checkpoint_sha256, "checkpoint_sha256") != ground.checkpoint_sha256 or before != ground.joint_weight_sha256:
        raise GRBJP4SpikeError("JP4 checkpoint/layer binding drift")
    theta, receipt = _fit_theta_stream(z, jac, labels, ground.old_class_order, ground.prototypes(),
                                        kappa_ground=ground.kappa_ground, checkpoint_weight=raw_weight,
                                        left=ground.left(), right=ground.right())
    if k == 1: theta = np.zeros(RANK, np.float32)
    if _teacher_theta_out is not None:
        if type(_teacher_theta_out) is not list or _teacher_theta_out:
            raise GRBJP4SpikeError("FP32 teacher theta transient output drift")
        _teacher_theta_out.append(np.array(theta, dtype=np.float32, copy=True))
    codes, scale = _theta_codec(theta); deployed = _decode_theta(codes, scale)
    semantic = _semantic_weight_sha(raw_weight, ground, deployed)
    full_receipt = {"schema": SCHEMA, **receipt, "theta_codec": "symmetric_int8_single_fp16_scale_v1",
                    "theta_wire_bytes": THETA_WIRE_BYTES, "ground_digest": ground.digest,
                    "checkpoint_sha256": checkpoint_sha256, "joint_weight_sha256_before": before,
                    "joint_weight_semantic_sha256": semantic, "support_token_root_sha256": _sha(_canon(sorted(tokens))),
                    "support_repair_receipt_sha256": _sha(_canon(dict(repair)))}
    return JP4FitState(codes, scale, k, ground.digest, checkpoint_sha256, before, semantic, full_receipt)


def _subset_strict_forward(forward: StrictForward, positions: np.ndarray) -> StrictForward:
    """Keep hook provenance while selecting nonzero support rows for Jacobians."""
    index = np.asarray(positions, np.intp)
    return StrictForward(
        np.ascontiguousarray(forward.z_id[index]),
        np.ascontiguousarray(forward.hidden[index]),
        np.ascontiguousarray(forward.pre_relu[index]),
        forward.hook_exact_bytes,
        None
        if forward.z_dom is None
        else np.ascontiguousarray(forward.z_dom[index]),
        forward.execution_path,
    )


def _fit_stage2_b_from_support_iq_development_only(
    *, model: Any, support_iq: torch.Tensor,
    support_labels: Sequence[Any] | np.ndarray, support_physical_tokens: Sequence[Any] | np.ndarray,
    ground: GroundReceiverBasis, checkpoint_sha256: str,
    _precomputed_strict_forward: StrictForward | None = None,
    _teacher_theta_out: list[np.ndarray] | None = None,
) -> JP4FitState:
    """Development-only support-IQ fit for local checkpoint feasibility.

    It intentionally accepts a decoded ground object and therefore cannot be
    a formal release entry.  The formal orchestrator below accepts only a
    ``VerifiedADV3B02DeploymentBundle`` and constructs its ground internally.
    """
    labels = typed_tokens(support_labels, name="JP4 formal S_B labels")
    tokens = typed_tokens(support_physical_tokens, name="JP4 formal S_B physical tokens", unique=True)
    if len(labels) != len(tokens) or tuple(sorted(set(labels))) != tuple(sorted(ground.old_class_order)):
        raise GRBJP4SpikeError("formal JP4 support class/token binding drift")
    k = _balanced_k(labels, ground.old_class_order)
    forward = (
        strict_zid_with_hook(model, support_iq)
        if _precomputed_strict_forward is None
        else _precomputed_strict_forward
    )
    if forward.hook_exact_bytes is not True or len(forward.z_id) != len(labels):
        raise GRBJP4SpikeError("formal JP4 hook/support row binding drift")
    linear = _joint_proj_linear(model)
    dirs = directions(ground.left(), ground.right())
    raw_zid = np.ascontiguousarray(forward.z_id)
    if k == 1:
        jacobian = np.zeros((len(raw_zid), RANK, Z_DIM), np.float32)
        repaired_zid, repair = raw_zid, {
            "schema": SCHEMA, "rule": "K1_identity_no_jp4_fit",
            "repaired_row_count": 0, "query_rows_used_for_fit": 0,
        }
    else:
        zero_mask = np.all(raw_zid == np.float32(0.0), axis=1)
        nonzero = np.flatnonzero(~zero_mask)
        if not len(nonzero):
            raise GRBJP4SpikeError("formal JP4 support contains no nonzero Jacobian source")
        jacobian = np.zeros((len(raw_zid), RANK, Z_DIM), np.float32)
        jacobian[nonzero] = analytic_jacobian(_subset_strict_forward(forward, nonzero), dirs)
        repaired_zid, jacobian, repair = prepare_support_for_jp4_fit(
            support_zid=raw_zid, support_jacobian=jacobian, support_labels=labels,
            support_physical_tokens=tokens, registered_old_classes=ground.old_class_order,
        )
    state = _fit_stage2_b_from_precomputed_jacobian_development_only(
        support_zid=repaired_zid, support_jacobian=jacobian, support_labels=labels,
        support_physical_tokens=tokens, ground=ground, checkpoint_weight=linear.weight,
        checkpoint_sha256=checkpoint_sha256, support_repair_receipt=repair,
        _teacher_theta_out=_teacher_theta_out,
    )
    receipt = dict(state.fit_receipt)
    receipt["development_support_iq_binding"] = {
        "checkpoint_sha256": checkpoint_sha256,
        "ground_digest": ground.digest,
        "support_token_root_sha256": _sha(_canon(sorted(tokens))),
        "hook_exact_bytes": True,
        "execution_path": forward.execution_path,
        "jacobian": "analytic_joint_proj0_relu_l2_from_sealed_lg_r",
        "query_rows_used_for_fit": 0,
    }
    return JP4FitState(state.theta_codes, state.theta_scale, state.k_shot, state.ground_digest,
                       state.checkpoint_sha256, state.joint_weight_sha256_before,
                       state.joint_weight_semantic_sha256, receipt)


def merge_into_joint_proj(linear: Any, *, state: JP4FitState, ground: GroundReceiverBasis,
                          checkpoint_sha256: str) -> None:
    """Apply only decoded INT8 theta in place, using one 320-wide row scratch."""
    state.__post_init__()
    ground.__post_init__()
    if (
        not hasattr(linear, "weight")
        or not torch.is_tensor(linear.weight)
        or tuple(linear.weight.shape) != (Z_DIM, HIDDEN_DIM)
        or linear.weight.dtype != torch.float32
    ):
        raise GRBJP4SpikeError("JP4 merge requires joint_proj.0 [160,320]")
    if checkpoint_sha256 != state.checkpoint_sha256 or state.ground_digest != ground.digest:
        raise GRBJP4SpikeError("JP4 merge state/bundle binding drift")
    before = _sha(_weight_bytes(linear.weight))
    if before != state.joint_weight_sha256_before or before != ground.joint_weight_sha256:
        raise GRBJP4SpikeError("JP4 merge attempted against a non-base layer")
    theta, left, right = state.theta(), ground.left(), ground.right()
    if state.k_shot == 1 and not np.array_equal(
        theta, np.zeros((RANK,), np.float32)
    ):
        raise GRBJP4SpikeError("K1 JP4 merge rejected nonidentity theta")
    with torch.no_grad():
        for row in range(Z_DIM):
            row_delta = np.sum((theta * left[:, row])[:, None] * right, axis=0, dtype=np.float32)
            linear.weight[row].add_(torch.as_tensor(row_delta, dtype=linear.weight.dtype, device=linear.weight.device))
    # Hash the current rows directly; the fit-time semantic hash was streamed
    # over the same row order and decoded INT8 theta.
    after = _sha(_weight_bytes(linear.weight))
    if after != state.joint_weight_semantic_sha256:
        raise GRBJP4SpikeError("JP4 decoded INT8 merge semantic hash drift")
    if state.k_shot == 1 and after != before:
        raise GRBJP4SpikeError("K1 JP4 merge changed checkpoint bytes")


def _merge_fp32_teacher_theta(
    linear: Any,
    *,
    theta: np.ndarray,
    ground: GroundReceiverBasis,
    checkpoint_sha256: str,
) -> None:
    """Transient support-only FP32 teacher merge used solely by the INT8 gate."""

    value = np.asarray(theta, dtype=np.float32)
    if (
        value.shape != (RANK,)
        or not np.isfinite(value).all()
        or checkpoint_sha256 != ground.checkpoint_sha256
        or _sha(_weight_bytes(linear.weight)) != ground.joint_weight_sha256
    ):
        raise GRBJP4SpikeError("FP32 teacher theta/checkpoint binding drift")
    left, right = ground.left(), ground.right()
    with torch.no_grad():
        for row in range(Z_DIM):
            row_delta = np.sum(
                (value * left[:, row])[:, None] * right,
                axis=0,
                dtype=np.float32,
            )
            linear.weight[row].add_(
                torch.as_tensor(
                    row_delta,
                    dtype=linear.weight.dtype,
                    device=linear.weight.device,
                )
            )


def _support_int8_theta_audit(
    *,
    teacher: StrictForward,
    deployed: StrictForward,
    labels: tuple[str, ...],
    tokens: tuple[str, ...],
    classes: tuple[str, ...],
) -> Mapping[str, Any]:
    """Fail closed on support-only FP32-teacher versus INT8-theta decisions."""

    if (
        teacher.z_dom is None
        or deployed.z_dom is None
        or teacher.z_id.shape != deployed.z_id.shape
        or len(teacher.z_id) != len(labels)
        or len(labels) != len(tokens)
    ):
        raise GRBJP4SpikeError("INT8 theta support audit layout drift")
    try:
        frozen_teacher_state = _parent_build_stage2_b_state(
            support_zid=teacher.z_id,
            support_zdom=teacher.z_dom,
            support_labels=labels,
            registered_classes=classes,
            support_physical_tokens=tokens,
        )
        teacher_logits = np.asarray(
            qknn_logits(frozen_teacher_state.id_bank, teacher.z_id), np.float64
        )
        deployed_logits = np.asarray(
            qknn_logits(frozen_teacher_state.id_bank, deployed.z_id), np.float64
        )
    except ADV3B02StateError as exc:
        raise GRBJP4SpikeError("INT8 theta frozen qKNN audit failed") from exc
    if (
        teacher_logits.shape != deployed_logits.shape
        or teacher_logits.shape != (len(labels), len(classes))
        or not np.isfinite(teacher_logits).all()
        or not np.isfinite(deployed_logits).all()
    ):
        raise GRBJP4SpikeError("INT8 theta support audit logit layout drift")
    rows = np.arange(len(teacher_logits))
    winner = np.argmax(teacher_logits, axis=1)
    runner_scores = np.array(teacher_logits, copy=True)
    runner_scores[rows, winner] = -np.inf
    runner = np.argmax(runner_scores, axis=1)
    teacher_margin = teacher_logits[rows, winner] - teacher_logits[rows, runner]
    max_error = np.max(np.abs(teacher_logits - deployed_logits), axis=1)
    any_flip = winner != np.argmax(deployed_logits, axis=1)
    large_flip = any_flip & (teacher_margin > 2.0 * max_error)
    receipt = {
        "schema": _INT8_THETA_AUDIT_SCHEMA,
        "surface": "support_only_fp32_teacher_vs_int8_theta_deployed",
        "frozen_decision_rule": "teacher_support_bank_student_t_qknn",
        "validation_row_count": len(labels),
        "top1_agreement": float(np.mean(~any_flip)),
        "any_margin_flip_count": int(np.count_nonzero(any_flip)),
        "large_margin_flip_count": int(np.count_nonzero(large_flip)),
        "teacher_margin_mean": float(np.mean(teacher_margin)),
        "logit_abs_error_max": float(
            np.max(np.abs(teacher_logits - deployed_logits))
        ),
        "teacher_support_zid_sha256": _array_digest(teacher.z_id),
        "deployed_support_zid_sha256": _array_digest(deployed.z_id),
        "frozen_qknn_state_sha256": frozen_teacher_state.id_bank.digest,
        "support_token_root_sha256": _ordered_root(tokens),
        "support_label_root_sha256": _ordered_root(labels),
        "query_rows_used_for_fit": 0,
        "state_updates_from_query": 0,
    }
    if (
        receipt["top1_agreement"] < 0.995
        or receipt["large_margin_flip_count"] != 0
    ):
        raise GRBJP4SpikeError("INT8 theta support decision audit gate failed")
    receipt["receipt_sha256"] = _sha(_canon(receipt))
    return receipt


def _build_stage2_b_state_development_only(*, jp4: JP4FitState, support_zid_after_merge: np.ndarray, support_zdom: np.ndarray,
                         support_labels: Sequence[Any] | np.ndarray, registered_classes: Sequence[Any] | np.ndarray,
                         support_physical_tokens: Sequence[Any] | np.ndarray,
                         support_repair_receipt: Mapping[str, Any] | None = None) -> GRBStage2BState:
    """Development-only precomputed-feature S_B helper."""
    parent = _parent_build_stage2_b_state(support_zid=support_zid_after_merge, support_zdom=support_zdom,
                                          support_labels=support_labels, registered_classes=registered_classes,
                                          support_physical_tokens=support_physical_tokens,
                                          support_repair_receipt=support_repair_receipt)
    return GRBStage2BState(jp4, parent, parent.id_bank.digest)


def _append_stage2_c_development_only(old_state: GRBStage2BState, *, new_support_zid_after_merge: np.ndarray,
                    new_support_zdom: np.ndarray, new_support_labels: Sequence[Any] | np.ndarray,
                    new_registered_classes: Sequence[Any] | np.ndarray, new_support_physical_tokens: Sequence[Any] | np.ndarray,
                    after_full_teacher_zid: np.ndarray, after_full_teacher_physical_tokens: Sequence[Any] | np.ndarray,
                    after_support_repair_receipt: Mapping[str, Any] | None = None) -> GRBStage2CState:
    """Development-only precomputed-feature S_C helper."""
    parent, receipt = _parent_append_stage2_c(old_state.parent_state, new_support_zid=new_support_zid_after_merge,
        new_support_zdom=new_support_zdom, new_support_labels=new_support_labels,
        new_registered_classes=new_registered_classes, new_support_physical_tokens=new_support_physical_tokens,
        after_full_teacher_zid=after_full_teacher_zid, after_full_teacher_physical_tokens=after_full_teacher_physical_tokens,
        after_support_repair_receipt=after_support_repair_receipt)
    return GRBStage2CState(old_state.jp4, parent, old_state.digest, receipt)


def _ordered_root(values: Sequence[str]) -> str:
    return _sha(_canon(list(values)))


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return _sha(
        _canon({"dtype": array.dtype.str, "shape": list(array.shape)})
        + array.tobytes()
    )


def _iq_digest(value: torch.Tensor) -> str:
    array = np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32)
    return _array_digest(array)


def _formal_iq(value: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    if (
        not torch.is_tensor(value)
        or value.dtype != torch.float32
        or value.ndim != 3
        or value.shape[1] != 2
        or not len(value)
        or not torch.isfinite(value).all()
    ):
        raise GRBJP4SpikeError("formal IQ must be finite float32 [N,2,T]")
    return value.detach().to(device=device, dtype=torch.float32).contiguous()


def _runtime_device(runtime: Any) -> torch.device:
    return _joint_proj_linear(runtime).weight.device


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_gpu_bytes(device: torch.device) -> int:
    return (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )


def _reload_verified_base_runtime(
    runtime_member_path: str,
    *,
    expected_runtime_sha256: str,
    expected_joint_weight_sha256: str,
    device: torch.device,
) -> Any:
    """Reload one signed base runtime after releasing the adapted instance."""

    path = Path(runtime_member_path)
    if (
        type(runtime_member_path) is not str
        or not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or path.stat().st_size <= 0
        or path.stat().st_size > 512 * 1024 * 1024
    ):
        raise GRBJP4SpikeError("verified runtime reload path drift")
    raw = path.read_bytes()
    if _sha(raw) != _require_sha(
        expected_runtime_sha256, "expected_runtime_sha256"
    ):
        raise GRBJP4SpikeError("verified runtime reload member SHA drift")
    try:
        runtime = torch.jit.load(io.BytesIO(raw), map_location=device)
    except (RuntimeError, ValueError) as exc:
        raise GRBJP4SpikeError("verified TorchScript runtime reload failed") from exc
    runtime.eval()
    method_names = tuple(runtime._c._method_names())
    if "grb_jp4_forward" not in method_names:
        raise GRBJP4SpikeError("reloaded runtime lost grb_jp4_forward")
    if (
        _sha(_weight_bytes(_joint_proj_linear(runtime).weight))
        != _require_sha(
            expected_joint_weight_sha256, "expected_joint_weight_sha256"
        )
    ):
        raise GRBJP4SpikeError("reloaded runtime/base checkpoint weight drift")
    return runtime


_RUNTIME_OWNERSHIP_SCHEMA = "cvs.phase2.grb_jp4.runtime_ownership.v2"
_INT8_THETA_AUDIT_SCHEMA = "cvs.phase2.grb_jp4.int8_theta_support_audit.v1"


def _live_runtime_semantic_sha256(runtime: Any) -> str:
    """Hash the current TorchScript method graphs and complete tensor state."""

    try:
        method_names = tuple(sorted(runtime._c._method_names()))
        graph_rows = [
            {
                "name": name,
                "graph_sha256": _sha(
                    str(runtime._c._get_method(name).graph).encode("utf-8")
                ),
            }
            for name in method_names
        ]
        state_rows = []
        for kind, values in (
            ("parameter", runtime.named_parameters()),
            ("buffer", runtime.named_buffers()),
        ):
            for name, tensor in values:
                value = tensor.detach().cpu().contiguous()
                try:
                    raw = value.numpy().tobytes(order="C")
                except TypeError:
                    raw = value.view(torch.uint8).reshape(-1).numpy().tobytes()
                state_rows.append(
                    {
                        "kind": kind,
                        "name": name,
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "sha256": _sha(raw),
                    }
                )
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise GRBJP4SpikeError(
            "formal live runtime graph/state fingerprint failed"
        ) from exc
    if "grb_jp4_forward" not in method_names:
        raise GRBJP4SpikeError("formal live runtime lost grb_jp4_forward")
    return _sha(
        _canon(
            {
                "method_graphs": graph_rows,
                "state": sorted(
                    state_rows, key=lambda item: (item["kind"], item["name"])
                ),
            }
        )
    )


@dataclass
class _ObservedRuntimeOwnership:
    """Ownership evidence derived from live TorchScript weak references only."""

    runtime_refs: list[Any]
    observed_live_instances: list[int]
    released_weakref_dead: list[bool] = field(default_factory=list)
    prior_observed_live_instances: tuple[int, ...] = ()
    prior_released_weakref_dead: tuple[bool, ...] = ()


def _runtime_ownership_from_bundle(
    bundle: VerifiedADV3B02DeploymentBundle,
) -> _ObservedRuntimeOwnership:
    source = dict(bundle._formal_runtime_lifecycle)
    if (
        source.get("schema") != "cvs.phase1.adv3b02_runtime_ownership.v2"
        or source.get("source_bundle_runtime_consumed") is not True
        or source.get("source_runtime_weakref_released_before_reload") is not True
        or source.get("formal_runtime_weakref_live_after_materialization") is not True
    ):
        raise GRBJP4SpikeError("formal runtime ownership transfer receipt drift")
    try:
        reference = weakref.ref(bundle.runtime)
    except TypeError as exc:
        raise GRBJP4SpikeError(
            "formal runtime does not support weak lifecycle observation"
        ) from exc
    if reference() is not bundle.runtime:
        raise GRBJP4SpikeError("formal runtime weakref materialization drift")
    return _ObservedRuntimeOwnership(
        runtime_refs=[reference], observed_live_instances=[1]
    )


def _runtime_ownership_release(ownership: _ObservedRuntimeOwnership) -> None:
    """Require actual collection before a caller reloads a new runtime."""

    gc.collect()
    released = ownership.runtime_refs[-1]() is None
    ownership.released_weakref_dead.append(released)
    if not released:
        raise GRBJP4SpikeError(
            "formal runtime weakref remained live after ownership release"
        )


def _runtime_ownership_materialized(
    ownership: _ObservedRuntimeOwnership, *, runtime: Any
) -> None:
    if ownership.runtime_refs[-1]() is not None:
        raise GRBJP4SpikeError("formal runtime materialized before prior release")
    try:
        reference = weakref.ref(runtime)
    except TypeError as exc:
        raise GRBJP4SpikeError(
            "reloaded formal runtime does not support weak lifecycle observation"
        ) from exc
    if reference() is not runtime:
        raise GRBJP4SpikeError("reloaded formal runtime weakref materialization drift")
    ownership.runtime_refs.append(reference)
    live_count = sum(reference() is not None for reference in ownership.runtime_refs)
    if live_count != 1:
        raise GRBJP4SpikeError("formal runtime weakref overlap detected")
    ownership.observed_live_instances.append(live_count)


def _runtime_ownership_receipt(ownership: _ObservedRuntimeOwnership) -> dict[str, Any]:
    materialization_observations = (
        list(ownership.prior_observed_live_instances)
        + list(ownership.observed_live_instances)
    )
    release_observations = (
        list(ownership.prior_released_weakref_dead)
        + list(ownership.released_weakref_dead)
    )
    receipt = {
        "schema": _RUNTIME_OWNERSHIP_SCHEMA,
        "source_bundle_runtime_consumed_before_reverification": True,
        "ownership_evidence_mode": "weakref_observed_torchscript_transfer_v1",
        "formal_materialization_count": len(materialization_observations),
        "verified_runtime_reload_count": len(materialization_observations) - 1,
        "release_count": len(release_observations),
        "live_runtime_instances_observed_at_materialization": materialization_observations,
        "release_weakref_dead_observed": release_observations,
        "live_runtime_count": sum(
            reference() is not None for reference in ownership.runtime_refs
        ),
        "live_runtime_instances_max": max(materialization_observations, default=0),
    }
    if (
        receipt.get("schema") != _RUNTIME_OWNERSHIP_SCHEMA
        or receipt.get("ownership_evidence_mode")
        != "weakref_observed_torchscript_transfer_v1"
        or receipt.get("source_bundle_runtime_consumed_before_reverification")
        is not True
        or receipt.get("live_runtime_count") != 1
        or receipt.get("live_runtime_instances_max") != 1
        or any(count != 1 for count in materialization_observations)
        or not all(release_observations)
        or int(receipt.get("formal_materialization_count", -1))
        != int(receipt.get("verified_runtime_reload_count", -1)) + 1
        or int(receipt.get("release_count", -1))
        != int(receipt.get("verified_runtime_reload_count", -1))
    ):
        raise GRBJP4SpikeError("formal runtime ownership lifecycle drift")
    receipt["receipt_sha256"] = _sha(_canon(receipt))
    return receipt


def _runtime_ownership_from_state(
    state: "FormalGRBJP4State",
) -> _ObservedRuntimeOwnership:
    receipt = dict(dict(state.resources).get("runtime_ownership_receipt", {}))
    recorded_sha = receipt.pop("receipt_sha256", None)
    if (
        recorded_sha != _sha(_canon(receipt))
        or receipt.get("schema") != _RUNTIME_OWNERSHIP_SCHEMA
        or receipt.get("ownership_evidence_mode")
        != "weakref_observed_torchscript_transfer_v1"
        or receipt.get("live_runtime_count") != 1
        or receipt.get("live_runtime_instances_max") != 1
        or not isinstance(
            receipt.get("live_runtime_instances_observed_at_materialization"), list
        )
        or not isinstance(receipt.get("release_weakref_dead_observed"), list)
        or any(
            count != 1
            for count in receipt["live_runtime_instances_observed_at_materialization"]
        )
        or not all(receipt["release_weakref_dead_observed"])
        or int(receipt.get("formal_materialization_count", -1))
        != int(receipt.get("verified_runtime_reload_count", -1)) + 1
        or int(receipt.get("release_count", -1))
        != int(receipt.get("verified_runtime_reload_count", -1))
    ):
        raise GRBJP4SpikeError("formal state runtime ownership receipt drift")
    try:
        reference = weakref.ref(state.runtime)
    except TypeError as exc:
        raise GRBJP4SpikeError(
            "formal state runtime does not support weak lifecycle observation"
        ) from exc
    if reference() is not state.runtime:
        raise GRBJP4SpikeError("formal state runtime weakref materialization drift")
    materializations = receipt["live_runtime_instances_observed_at_materialization"]
    return _ObservedRuntimeOwnership(
        runtime_refs=[reference],
        observed_live_instances=[1],
        prior_observed_live_instances=tuple(materializations[:-1]),
        prior_released_weakref_dead=tuple(
            receipt["release_weakref_dead_observed"]
        ),
    )


def _runtime_terminal_ownership_receipt(
    ownership: _ObservedRuntimeOwnership,
) -> dict[str, Any]:
    materialization_observations = (
        list(ownership.prior_observed_live_instances)
        + list(ownership.observed_live_instances)
    )
    release_observations = (
        list(ownership.prior_released_weakref_dead)
        + list(ownership.released_weakref_dead)
    )
    receipt = {
        "schema": _RUNTIME_OWNERSHIP_SCHEMA,
        "ownership_evidence_mode": "weakref_observed_torchscript_transfer_v1",
        "formal_materialization_count": len(materialization_observations),
        "verified_runtime_reload_count": len(materialization_observations) - 1,
        "release_count": len(release_observations),
        "live_runtime_instances_observed_at_materialization": materialization_observations,
        "release_weakref_dead_observed": release_observations,
        "live_runtime_count": sum(
            reference() is not None for reference in ownership.runtime_refs
        ),
        "live_runtime_instances_max": max(materialization_observations, default=0),
    }
    if (
        receipt.get("schema") != _RUNTIME_OWNERSHIP_SCHEMA
        or receipt.get("live_runtime_count") != 0
        or receipt.get("live_runtime_instances_max") != 1
        or any(count != 1 for count in materialization_observations)
        or not all(release_observations)
        or int(receipt.get("formal_materialization_count", -1))
        != int(receipt.get("verified_runtime_reload_count", -1)) + 1
        or int(receipt.get("release_count", -1))
        != int(receipt.get("verified_runtime_reload_count", -1)) + 1
    ):
        raise GRBJP4SpikeError("formal terminal runtime ownership drift")
    receipt["receipt_sha256"] = _sha(_canon(receipt))
    return receipt


@dataclass(frozen=True, init=False)
class FormalGRBJP4State:
    """One independently reviewable formal S_B or S_C five-arm lifecycle."""

    runtime: Any
    runtime_member_path: str
    runtime_sha256: str
    runtime_phase: str
    ground: GroundReceiverBasis
    jp4: JP4FitState
    no_ground_state: DualQKNNState
    adapted_state: GRBStage2BState | GRBStage2CState
    stage: str
    old_support_labels: tuple[str, ...]
    old_support_tokens: tuple[str, ...]
    old_support_iq_sha256: str
    all_support_tokens: tuple[str, ...]
    registered_classes: tuple[str, ...]
    bundle_receipt_sha256: str
    lifecycle_receipt: Mapping[str, Any]
    resources: Mapping[str, Any]
    _issued_runtime_identity: int = field(init=False, repr=False, compare=False)
    _runtime_semantic_sha256: str = field(init=False, repr=False, compare=False)
    _issued_coordinator_sha256: str = field(
        init=False, repr=False, compare=False
    )

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise GRBJP4SpikeError(
            "FormalGRBJP4State is production-factory issued only"
        )

    def _validate_contents(self) -> None:
        self.jp4.__post_init__()
        self.ground.__post_init__()
        if (
            self.runtime_phase != "BASE_READY"
            or self.runtime is None
            or not Path(self.runtime_member_path).is_absolute()
            or _sha(_weight_bytes(_joint_proj_linear(self.runtime).weight))
            != self.jp4.joint_weight_sha256_before
        ):
            raise GRBJP4SpikeError("formal runtime ownership/base phase drift")
        _require_sha(self.runtime_sha256, "runtime_sha256")
        if self.stage not in ("S_B", "S_C"):
            raise GRBJP4SpikeError("formal lifecycle stage drift")
        adapted = self.adapted_state.parent_state
        if (
            adapted.domain.stage != self.stage
            or self.no_ground_state.domain.stage != self.stage
            or adapted.domain.k_shot != self.jp4.k_shot
            or self.no_ground_state.domain.k_shot != self.jp4.k_shot
            or tuple(adapted.id_bank.classes) != self.registered_classes
            or tuple(self.no_ground_state.id_bank.classes)
            != self.registered_classes
            or tuple(adapted.id_bank.support_tokens) != self.all_support_tokens
            or tuple(self.no_ground_state.id_bank.support_tokens)
            != self.all_support_tokens
        ):
            raise GRBJP4SpikeError("formal five-arm state/class/token lifecycle drift")
        _require_sha(self.old_support_iq_sha256, "old_support_iq_sha256")
        _require_sha(self.bundle_receipt_sha256, "bundle_receipt_sha256")
        formal = dict(self.jp4.fit_receipt).get("formal_bundle_binding")
        if (
            type(formal) is not dict
            or formal.get("verified_bundle_receipt_sha256")
            != self.bundle_receipt_sha256
        ):
            raise GRBJP4SpikeError("formal lifecycle/bundle receipt drift")
        receipt = dict(self.lifecycle_receipt)
        if (
            receipt.get("schema") != FORMAL_ORCHESTRATOR_SCHEMA
            or receipt.get("stage") != self.stage
            or receipt.get("k_shot") != self.jp4.k_shot
            or receipt.get("support_row_count") != len(self.all_support_tokens)
            or receipt.get("class_count") != len(self.registered_classes)
            or receipt.get("support_token_root_sha256")
            != _ordered_root(self.all_support_tokens)
            or receipt.get("class_order_sha256")
            != _ordered_root(self.registered_classes)
            or receipt.get("jp4_serialized_sha256")
            != _sha(serialize_jp4_fit_state(self.jp4))
            or receipt.get("query_rows_used_for_fit") != 0
        ):
            raise GRBJP4SpikeError("formal lifecycle closure receipt drift")
        if (
            dict(self.resources).get("query_rows_used_for_fit") != 0
            or dict(self.resources).get("live_model_weight_instances_max") != 1
            or int(dict(self.resources).get("total_candidate_state_bytes", -1))
            > MAX_STATE_BYTES
        ):
            raise GRBJP4SpikeError("formal lifecycle resource closure drift")
        int8_audit = dict(
            dict(self.resources).get("int8_theta_support_audit", {})
        )
        recorded_audit = dict(self.jp4.fit_receipt).get(
            "int8_theta_support_audit"
        )
        audit_sha = int8_audit.pop("receipt_sha256", None)
        if (
            type(recorded_audit) is not dict
            or recorded_audit != dict(
                dict(self.resources).get("int8_theta_support_audit", {})
            )
            or int8_audit.get("schema") != _INT8_THETA_AUDIT_SCHEMA
            or audit_sha != _sha(_canon(int8_audit))
            or float(int8_audit.get("top1_agreement", -1.0)) < 0.995
            or int(int8_audit.get("large_margin_flip_count", -1)) != 0
            or int8_audit.get("query_rows_used_for_fit") != 0
            or int8_audit.get("state_updates_from_query") != 0
        ):
            raise GRBJP4SpikeError("formal INT8 theta support audit drift")
        _runtime_ownership_from_state(self)

    def _coordinator_sha256(self) -> str:
        return _sha(
            _canon(
                {
                    "runtime_member_path": self.runtime_member_path,
                    "runtime_sha256": self.runtime_sha256,
                    "runtime_phase": self.runtime_phase,
                    "runtime_semantic_sha256": self._runtime_semantic_sha256,
                    "ground_sha256": self.ground.digest,
                    "jp4_sha256": self.jp4.digest,
                    "no_ground_state_sha256": self.no_ground_state.digest,
                    "adapted_state_sha256": self.adapted_state.digest,
                    "stage": self.stage,
                    "old_support_labels": list(self.old_support_labels),
                    "old_support_tokens": list(self.old_support_tokens),
                    "old_support_iq_sha256": self.old_support_iq_sha256,
                    "all_support_tokens": list(self.all_support_tokens),
                    "registered_classes": list(self.registered_classes),
                    "bundle_receipt_sha256": self.bundle_receipt_sha256,
                    "lifecycle_receipt": dict(self.lifecycle_receipt),
                    "resources": dict(self.resources),
                }
            )
        )

    @property
    def digest(self) -> str:
        return _sha(
            self.jp4.wire_bytes()
            + self.no_ground_state.wire_bytes()
            + self.adapted_state.parent_state.wire_bytes()
            + _canon(dict(self.lifecycle_receipt))
        )


def _validate_formal_state(
    state: FormalGRBJP4State, *, expected_stage: str
) -> None:
    if (
        not _was_formal_state_issued(state)
        or state.stage != expected_stage
        or state.runtime_phase != "BASE_READY"
        or state.runtime is None
        or getattr(state, "_issued_runtime_identity", None) != id(state.runtime)
    ):
        raise GRBJP4SpikeError("formal state issuance/runtime ownership drift")
    if (
        _live_runtime_semantic_sha256(state.runtime)
        != state._runtime_semantic_sha256
        or state._coordinator_sha256() != state._issued_coordinator_sha256
    ):
        raise GRBJP4SpikeError("formal state current runtime graph/state binding drift")
    state._validate_contents()


def _fit_stage2_b_from_support_iq_impl(
    *,
    bundle: VerifiedADV3B02DeploymentBundle,
    support_iq: torch.Tensor,
    support_labels: Sequence[Any] | np.ndarray,
    support_physical_tokens: Sequence[Any] | np.ndarray,
    _issue_state: Any,
) -> FormalGRBJP4State:
    """Formal production entry: verified bundle to merged, matched S_B state."""

    bundle, verified_receipt, verified_receipt_sha = _reverified_formal_bundle_receipt(
        bundle
    )
    runtime_ownership = _runtime_ownership_from_bundle(bundle)
    runtime = bundle.runtime
    runtime_member_path = bundle.runtime_member_path
    runtime_sha256 = verified_receipt["runtime_sha256"]
    runtime.eval()
    device = _runtime_device(runtime)
    iq = _formal_iq(support_iq, device=device)
    labels = typed_tokens(support_labels, name="formal S_B labels")
    tokens = typed_tokens(
        support_physical_tokens, name="formal S_B physical tokens", unique=True
    )
    ground = GroundReceiverBasis._from_reverified_bundle(bundle)
    if (
        len(labels) != len(tokens)
        or tuple(sorted(set(labels))) != tuple(sorted(ground.old_class_order))
    ):
        raise GRBJP4SpikeError("formal S_B old support class/token binding drift")
    _balanced_k(labels, ground.old_class_order)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    fit_started = time.perf_counter()
    base_forward = strict_zid_with_hook(runtime, iq)
    if base_forward.z_dom is None:
        raise GRBJP4SpikeError("formal TorchScript tap did not expose z_dom")
    teacher_theta_out: list[np.ndarray] = []
    jp4 = _fit_stage2_b_from_support_iq_development_only(
        model=runtime,
        support_iq=iq,
        support_labels=labels,
        support_physical_tokens=tokens,
        ground=ground,
        checkpoint_sha256=ground.checkpoint_sha256,
        _precomputed_strict_forward=base_forward,
        _teacher_theta_out=teacher_theta_out,
    )
    receipt = dict(jp4.fit_receipt)
    receipt.pop("development_support_iq_binding", None)
    receipt["formal_bundle_binding"] = {
        "candidate": CANDIDATE,
        "component_profile": COMPONENT_PROFILE_GRB_JP4_Q4,
        "verified_bundle_receipt_sha256": verified_receipt_sha,
        "outer_content_root_sha256": verified_receipt[
            "outer_content_root_sha256"
        ],
        "component_outer_slot_sha256": verified_receipt[
            "component_outer_slot_sha256"
        ],
        "method_lock_sha256": verified_receipt["method_lock_sha256"],
        "runtime_sha256": verified_receipt["runtime_sha256"],
        "checkpoint_sha256": ground.checkpoint_sha256,
    }
    receipt["formal_support_iq_binding"] = {
        "support_iq_sha256": _iq_digest(iq),
        "support_token_root_sha256": _ordered_root(tokens),
        "support_label_root_sha256": _ordered_root(labels),
        "support_row_count": len(tokens),
        "execution_path": base_forward.execution_path,
        "hook_exact_bytes": True,
        "query_rows_used_for_fit": 0,
    }
    _synchronize(device)
    merge_started = time.perf_counter()
    if len(teacher_theta_out) != 1:
        raise GRBJP4SpikeError("formal FP32 teacher theta capture drift")
    _merge_fp32_teacher_theta(
        _joint_proj_linear(runtime),
        theta=teacher_theta_out.pop(),
        ground=ground,
        checkpoint_sha256=ground.checkpoint_sha256,
    )
    teacher_forward = strict_zid_with_hook(runtime, iq)
    if teacher_forward.z_dom is None:
        raise GRBJP4SpikeError("formal FP32 teacher tap did not expose z_dom")
    # Release the FP32-teacher instance before reloading signed base bytes for
    # the deployed INT8 replay.  The ownership counter records the real event.
    object.__setattr__(bundle, "runtime", None)
    runtime = None
    _runtime_ownership_release(runtime_ownership)
    runtime = _reload_verified_base_runtime(
        runtime_member_path,
        expected_runtime_sha256=runtime_sha256,
        expected_joint_weight_sha256=jp4.joint_weight_sha256_before,
        device=device,
    )
    _runtime_ownership_materialized(runtime_ownership, runtime=runtime)
    merge_into_joint_proj(
        _joint_proj_linear(runtime),
        state=jp4,
        ground=ground,
        checkpoint_sha256=ground.checkpoint_sha256,
    )
    _synchronize(device)
    merge_ms = (time.perf_counter() - merge_started) * 1000.0
    adapted_forward = strict_zid_with_hook(runtime, iq)
    if adapted_forward.z_dom is None:
        raise GRBJP4SpikeError("adapted TorchScript tap did not expose z_dom")
    int8_theta_audit = _support_int8_theta_audit(
        teacher=teacher_forward,
        deployed=adapted_forward,
        labels=labels,
        tokens=tokens,
        classes=ground.old_class_order,
    )
    receipt["int8_theta_support_audit"] = dict(int8_theta_audit)
    jp4 = JP4FitState(
        jp4.theta_codes,
        jp4.theta_scale,
        jp4.k_shot,
        jp4.ground_digest,
        jp4.checkpoint_sha256,
        jp4.joint_weight_sha256_before,
        jp4.joint_weight_semantic_sha256,
        receipt,
    )
    wire = serialize_jp4_fit_state(jp4)
    jp4 = deserialize_jp4_fit_state(
        wire,
        expected_ground_digest=ground.digest,
        expected_checkpoint_sha256=ground.checkpoint_sha256,
        expected_joint_weight_sha256_before=ground.joint_weight_sha256,
        expected_formal_bundle_receipt_sha256=verified_receipt_sha,
    )
    try:
        no_ground = _parent_build_stage2_b_state(
            support_zid=base_forward.z_id,
            support_zdom=base_forward.z_dom,
            support_labels=labels,
            registered_classes=ground.old_class_order,
            support_physical_tokens=tokens,
        )
        adapted_parent = _parent_build_stage2_b_state(
            support_zid=adapted_forward.z_id,
            support_zdom=adapted_forward.z_dom,
            support_labels=labels,
            registered_classes=ground.old_class_order,
            support_physical_tokens=tokens,
        )
    except ADV3B02StateError as exc:
        raise GRBJP4SpikeError("formal S_B r6 state closure failed") from exc
    adapted = GRBStage2BState(jp4, adapted_parent, adapted_parent.id_bank.digest)
    _synchronize(device)
    peak_gpu = _peak_gpu_bytes(device)
    # Formal ownership is single-use.  Release the adapted instance before
    # loading the same signed member back at exact base bytes; the returned
    # lifecycle therefore owns one live model-weight set, never a base clone.
    runtime = None
    _runtime_ownership_release(runtime_ownership)
    runtime = _reload_verified_base_runtime(
        runtime_member_path,
        expected_runtime_sha256=runtime_sha256,
        expected_joint_weight_sha256=jp4.joint_weight_sha256_before,
        device=device,
    )
    _runtime_ownership_materialized(runtime_ownership, runtime=runtime)
    ownership_receipt = _runtime_ownership_receipt(runtime_ownership)
    fit_ms = (time.perf_counter() - fit_started) * 1000.0
    classes = tuple(adapted_parent.id_bank.classes)
    ordered_tokens = tuple(adapted_parent.id_bank.support_tokens)
    lifecycle = {
        "schema": FORMAL_ORCHESTRATOR_SCHEMA,
        "stage": "S_B",
        "k_shot": jp4.k_shot,
        "support_row_count": len(ordered_tokens),
        "class_count": len(classes),
        "support_token_root_sha256": _ordered_root(ordered_tokens),
        "class_order_sha256": _ordered_root(classes),
        "old_support_iq_sha256": _iq_digest(iq),
        "base_support_zid_sha256": _array_digest(base_forward.z_id),
        "adapted_support_zid_sha256": _array_digest(adapted_forward.z_id),
        "no_ground_state_sha256": no_ground.digest,
        "adapted_state_sha256": adapted_parent.digest,
        "jp4_serialized_sha256": _sha(wire),
        "same_iq_two_model_states": True,
        "verified_runtime_reloaded_to_base": True,
        "formal_bundle_runtime_ownership_consumed": True,
        "runtime_ownership_receipt_sha256": ownership_receipt["receipt_sha256"],
        "int8_theta_support_audit_sha256": int8_theta_audit["receipt_sha256"],
        "second_full_model_weight_copy": False,
        "query_rows_used_for_fit": 0,
    }
    resources = resource_receipt(
        jp4,
        ground,
        parent_state_bytes=max(
            len(no_ground.wire_bytes()), len(adapted_parent.wire_bytes())
        ),
        support_rows=len(tokens),
        support_forward_calls=2,
        analytic_jacobian_rows=0 if jp4.k_shot == 1 else len(tokens),
        fit_time_ms=fit_ms,
        merge_time_ms=merge_ms,
        peak_gpu_memory_bytes=max(peak_gpu, _peak_gpu_bytes(device)),
    )
    resources.update(
        {
            "verified_runtime_member_bytes": Path(runtime_member_path).stat().st_size,
            "verified_runtime_member_sha256": runtime_sha256,
            "verified_runtime_reload_count": ownership_receipt[
                "verified_runtime_reload_count"
            ],
            "live_model_weight_instances_max": ownership_receipt[
                "live_runtime_instances_max"
            ],
            "runtime_ownership_receipt": ownership_receipt,
            "int8_theta_support_audit": dict(int8_theta_audit),
        }
    )
    return _issue_state(
        runtime=runtime,
        runtime_member_path=runtime_member_path,
        runtime_sha256=runtime_sha256,
        runtime_phase="BASE_READY",
        ground=ground,
        jp4=jp4,
        no_ground_state=no_ground,
        adapted_state=adapted,
        stage="S_B",
        old_support_labels=labels,
        old_support_tokens=tokens,
        old_support_iq_sha256=_iq_digest(iq),
        all_support_tokens=ordered_tokens,
        registered_classes=classes,
        bundle_receipt_sha256=verified_receipt_sha,
        lifecycle_receipt=lifecycle,
        resources=resources,
    )


def _append_formal_stage2_c_impl(
    old: FormalGRBJP4State,
    *,
    old_support_iq: torch.Tensor,
    old_support_labels: Sequence[Any] | np.ndarray,
    old_support_physical_tokens: Sequence[Any] | np.ndarray,
    new_support_iq: torch.Tensor,
    new_support_labels: Sequence[Any] | np.ndarray,
    new_registered_classes: Sequence[Any] | np.ndarray,
    new_support_physical_tokens: Sequence[Any] | np.ndarray,
    _issue_state: Any,
) -> FormalGRBJP4State:
    """Re-encode matched old+new support internally and append without refit."""

    _validate_formal_state(old, expected_stage="S_B")
    runtime_ownership = _runtime_ownership_from_state(old)
    if (
        old.runtime_phase != "BASE_READY"
        or _sha(_weight_bytes(_joint_proj_linear(old.runtime).weight))
        != old.jp4.joint_weight_sha256_before
    ):
        raise GRBJP4SpikeError("formal S_C runtime is not at verified base bytes")
    runtime = old.runtime
    device = _runtime_device(runtime)
    old_iq = _formal_iq(old_support_iq, device=device)
    new_iq = _formal_iq(new_support_iq, device=device)
    old_labels = typed_tokens(old_support_labels, name="formal S_C old labels")
    old_tokens = typed_tokens(
        old_support_physical_tokens,
        name="formal S_C old physical tokens",
        unique=True,
    )
    new_labels = typed_tokens(new_support_labels, name="formal S_C new labels")
    new_classes = typed_tokens(
        new_registered_classes, name="formal S_C new registry", unique=True
    )
    new_tokens = typed_tokens(
        new_support_physical_tokens,
        name="formal S_C new physical tokens",
        unique=True,
    )
    if (
        old_labels != old.old_support_labels
        or old_tokens != old.old_support_tokens
        or _iq_digest(old_iq) != old.old_support_iq_sha256
        or len(new_labels) != len(new_tokens)
        or set(old_tokens).intersection(new_tokens)
        or set(new_classes).intersection(old.registered_classes)
        or tuple(sorted(set(new_labels))) != tuple(sorted(new_classes))
        or _balanced_k(new_labels, new_classes) != old.jp4.k_shot
    ):
        raise GRBJP4SpikeError("formal S_C old/new IQ/class/token binding drift")
    if old.jp4.k_shot == 1 and (
        not np.array_equal(old.jp4.theta(), np.zeros((RANK,), np.float32))
        or old.jp4.joint_weight_semantic_sha256
        != old.jp4.joint_weight_sha256_before
    ):
        raise GRBJP4SpikeError("formal S_C rejected a nonidentity K1 state")
    full_iq = torch.cat((old_iq, new_iq), dim=0)
    append_started = time.perf_counter()
    base_full = strict_zid_with_hook(runtime, full_iq)
    if base_full.z_dom is None:
        raise GRBJP4SpikeError("formal S_C TorchScript tap lacks z_dom")
    split = len(old_tokens)
    full_tokens = old_tokens + new_tokens
    try:
        base_after, base_receipt = _parent_append_stage2_c(
            old.no_ground_state,
            new_support_zid=base_full.z_id[split:],
            new_support_zdom=base_full.z_dom[split:],
            new_support_labels=new_labels,
            new_registered_classes=new_classes,
            new_support_physical_tokens=new_tokens,
            after_full_teacher_zid=base_full.z_id,
            after_full_teacher_physical_tokens=full_tokens,
        )
    except ADV3B02StateError as exc:
        raise GRBJP4SpikeError("formal S_C base r6 append closure failed") from exc
    _synchronize(device)
    merge_started = time.perf_counter()
    merge_into_joint_proj(
        _joint_proj_linear(runtime),
        state=old.jp4,
        ground=old.ground,
        checkpoint_sha256=old.jp4.checkpoint_sha256,
    )
    _synchronize(device)
    append_merge_ms = (time.perf_counter() - merge_started) * 1000.0
    adapted_full = strict_zid_with_hook(runtime, full_iq)
    if adapted_full.z_dom is None:
        raise GRBJP4SpikeError("formal S_C adapted TorchScript tap lacks z_dom")
    try:
        adapted_after_parent, adapted_receipt = _parent_append_stage2_c(
            old.adapted_state.parent_state,
            new_support_zid=adapted_full.z_id[split:],
            new_support_zdom=adapted_full.z_dom[split:],
            new_support_labels=new_labels,
            new_registered_classes=new_classes,
            new_support_physical_tokens=new_tokens,
            after_full_teacher_zid=adapted_full.z_id,
            after_full_teacher_physical_tokens=full_tokens,
        )
    except ADV3B02StateError as exc:
        raise GRBJP4SpikeError("formal S_C adapted r6 append closure failed") from exc
    adapted_after = GRBStage2CState(
        old.jp4,
        adapted_after_parent,
        old.adapted_state.digest,
        {**adapted_receipt, "k_shot": old.jp4.k_shot},
    )
    classes = tuple(adapted_after_parent.id_bank.classes)
    ordered_tokens = tuple(adapted_after_parent.id_bank.support_tokens)
    _synchronize(device)
    append_ms = (time.perf_counter() - append_started) * 1000.0
    peak_gpu = _peak_gpu_bytes(device)
    object.__setattr__(old, "runtime", None)
    object.__setattr__(old, "runtime_phase", "CONSUMED_BY_S_C")
    runtime = None
    _runtime_ownership_release(runtime_ownership)
    runtime = _reload_verified_base_runtime(
        old.runtime_member_path,
        expected_runtime_sha256=old.runtime_sha256,
        expected_joint_weight_sha256=old.jp4.joint_weight_sha256_before,
        device=device,
    )
    _runtime_ownership_materialized(runtime_ownership, runtime=runtime)
    ownership_receipt = _runtime_ownership_receipt(runtime_ownership)
    lifecycle = {
        "schema": FORMAL_ORCHESTRATOR_SCHEMA,
        "stage": "S_C",
        "k_shot": old.jp4.k_shot,
        "support_row_count": len(ordered_tokens),
        "class_count": len(classes),
        "support_token_root_sha256": _ordered_root(ordered_tokens),
        "class_order_sha256": _ordered_root(classes),
        "old_support_iq_sha256": old.old_support_iq_sha256,
        "new_support_iq_sha256": _iq_digest(new_iq),
        "base_full_teacher_sha256": _array_digest(base_full.z_id),
        "adapted_full_teacher_sha256": _array_digest(adapted_full.z_id),
        "no_ground_state_sha256": base_after.digest,
        "adapted_state_sha256": adapted_after_parent.digest,
        "base_append_receipt_sha256": base_receipt["receipt_sha256"],
        "adapted_append_receipt_sha256": adapted_receipt["receipt_sha256"],
        "jp4_serialized_sha256": _sha(serialize_jp4_fit_state(old.jp4)),
        "jp4_state_byte_identical_to_s_b": (
            old.jp4.wire_bytes() == adapted_after.jp4.wire_bytes()
        ),
        "same_iq_two_model_states": True,
        "verified_runtime_reloaded_to_base": True,
        "consumed_stage_b_runtime_phase": "CONSUMED_BY_S_C",
        "runtime_ownership_receipt_sha256": ownership_receipt["receipt_sha256"],
        "append_time_ms": append_ms,
        "query_rows_used_for_fit": 0,
    }
    resources = resource_receipt(
        old.jp4,
        old.ground,
        parent_state_bytes=max(
            len(base_after.wire_bytes()), len(adapted_after_parent.wire_bytes())
        ),
        support_rows=len(full_tokens),
        support_forward_calls=int(old.resources["support_forward_calls"]) + 2,
        analytic_jacobian_rows=int(old.resources["analytic_jacobian_rows"]),
        fit_time_ms=float(old.resources["fit_time_ms"]),
        merge_time_ms=(
            float(old.resources["merge_time_ms"]) + append_merge_ms
        ),
        peak_gpu_memory_bytes=max(
            int(old.resources["peak_gpu_memory_bytes"]),
            peak_gpu,
            _peak_gpu_bytes(device),
        ),
    )
    resources.update(
        {
            "verified_runtime_member_bytes": int(
                old.resources["verified_runtime_member_bytes"]
            ),
            "verified_runtime_member_sha256": old.runtime_sha256,
            "verified_runtime_reload_count": ownership_receipt[
                "verified_runtime_reload_count"
            ],
            "live_model_weight_instances_max": ownership_receipt[
                "live_runtime_instances_max"
            ],
            "runtime_ownership_receipt": ownership_receipt,
            "int8_theta_support_audit": dict(
                old.resources["int8_theta_support_audit"]
            ),
        }
    )
    return _issue_state(
        runtime=runtime,
        runtime_member_path=old.runtime_member_path,
        runtime_sha256=old.runtime_sha256,
        runtime_phase="BASE_READY",
        ground=old.ground,
        jp4=old.jp4,
        no_ground_state=base_after,
        adapted_state=adapted_after,
        stage="S_C",
        old_support_labels=old_labels,
        old_support_tokens=old_tokens,
        old_support_iq_sha256=old.old_support_iq_sha256,
        all_support_tokens=ordered_tokens,
        registered_classes=classes,
        bundle_receipt_sha256=old.bundle_receipt_sha256,
        lifecycle_receipt=lifecycle,
        resources=resources,
    )


def _formal_state_orchestrator_factory() -> tuple[Any, Any, Any]:
    """Bind state materialization inside the two public formal orchestrators.

    No constructor, issuer token, or issuer function is installed in the
    module namespace.  Calling this factory again creates a different weak
    registry whose states fail the module's official consumer validation.
    """

    issued: dict[int, weakref.ReferenceType[FormalGRBJP4State]] = {}
    required = {
        "runtime",
        "runtime_member_path",
        "runtime_sha256",
        "runtime_phase",
        "ground",
        "jp4",
        "no_ground_state",
        "adapted_state",
        "stage",
        "old_support_labels",
        "old_support_tokens",
        "old_support_iq_sha256",
        "all_support_tokens",
        "registered_classes",
        "bundle_receipt_sha256",
        "lifecycle_receipt",
        "resources",
    }

    def issue(**values: Any) -> FormalGRBJP4State:
        if set(values) != required:
            raise GRBJP4SpikeError("formal state production field set drift")
        state = object.__new__(FormalGRBJP4State)
        for name in sorted(required):
            object.__setattr__(state, name, values[name])
        object.__setattr__(state, "_issued_runtime_identity", id(state.runtime))
        object.__setattr__(
            state,
            "_runtime_semantic_sha256",
            _live_runtime_semantic_sha256(state.runtime),
        )
        state._validate_contents()
        object.__setattr__(
            state,
            "_issued_coordinator_sha256",
            state._coordinator_sha256(),
        )
        issued[id(state)] = weakref.ref(state)
        return state

    def was_issued(state: Any) -> bool:
        reference = issued.get(id(state))
        return (
            type(state) is FormalGRBJP4State
            and reference is not None
            and reference() is state
        )

    def fit(
        *,
        bundle: VerifiedADV3B02DeploymentBundle,
        support_iq: torch.Tensor,
        support_labels: Sequence[Any] | np.ndarray,
        support_physical_tokens: Sequence[Any] | np.ndarray,
    ) -> FormalGRBJP4State:
        return _fit_stage2_b_from_support_iq_impl(
            bundle=bundle,
            support_iq=support_iq,
            support_labels=support_labels,
            support_physical_tokens=support_physical_tokens,
            _issue_state=issue,
        )

    def append(
        old: FormalGRBJP4State,
        *,
        old_support_iq: torch.Tensor,
        old_support_labels: Sequence[Any] | np.ndarray,
        old_support_physical_tokens: Sequence[Any] | np.ndarray,
        new_support_iq: torch.Tensor,
        new_support_labels: Sequence[Any] | np.ndarray,
        new_registered_classes: Sequence[Any] | np.ndarray,
        new_support_physical_tokens: Sequence[Any] | np.ndarray,
    ) -> FormalGRBJP4State:
        return _append_formal_stage2_c_impl(
            old,
            old_support_iq=old_support_iq,
            old_support_labels=old_support_labels,
            old_support_physical_tokens=old_support_physical_tokens,
            new_support_iq=new_support_iq,
            new_support_labels=new_support_labels,
            new_registered_classes=new_registered_classes,
            new_support_physical_tokens=new_support_physical_tokens,
            _issue_state=issue,
        )

    fit.__name__ = "fit_stage2_b_from_support_iq"
    fit.__qualname__ = "fit_stage2_b_from_support_iq"
    append.__name__ = "append_formal_stage2_c"
    append.__qualname__ = "append_formal_stage2_c"
    return fit, append, was_issued


(
    fit_stage2_b_from_support_iq,
    append_formal_stage2_c,
    _was_formal_state_issued,
) = _formal_state_orchestrator_factory()


def build_five_arm_state_view(*, no_ground_state: DualQKNNState,
                              adapted_state: GRBStage2BState | GRBStage2CState) -> Mapping[str, Any]:
    """Compose the frozen five-arm view without changing parent formulas.

    ``no_ground_state`` is the matched base r6 state (S_B or S_C) and supplies
    M0/M_DA_NG/M_OTHER; ``adapted_state`` supplies M_DA/M_JOINT at the same
    lifecycle stage.  The returned objects preserve exact object sharing
    inside each arm pair and have no prediction or query parameters.
    """
    if not isinstance(adapted_state, (GRBStage2BState, GRBStage2CState)):
        raise GRBJP4SpikeError("five-arm closure requires a GRB S_B or S_C state")
    adapted = adapted_state.parent_state
    stage = adapted.domain.stage
    if stage not in ("S_B", "S_C") or no_ground_state.domain.stage != stage:
        raise GRBJP4SpikeError("five-arm base/adapted lifecycle stage drift")
    if no_ground_state.domain.k_shot != adapted.domain.k_shot:
        raise GRBJP4SpikeError("five-arm base/adapted K drift")
    base_raw = no_ground_state.id_bank
    base_qscore, base_bscore, base_exact_masked_degenerate = (
        _raw_directional_loo(base_raw)
    )
    base_other = fit_bcrr_branch(id_bank=base_raw, directional_qscore=base_qscore,
                                 directional_bscore=base_bscore)
    adapted_raw = adapted.id_bank
    (
        adapted_qscore,
        adapted_bscore,
        adapted_exact_masked_degenerate,
    ) = _raw_directional_loo(adapted_raw)
    adapted_dual_qscore = _directional_dual_loo(
        adapted,
        adapted_qscore,
        exact_masked_degenerate=adapted_exact_masked_degenerate,
    )
    adapted_joint = fit_bcrr_branch(id_bank=adapted_raw, directional_qscore=adapted_dual_qscore,
                                    directional_bscore=adapted_bscore)
    state = {"M0": base_raw, "M_DA_NG": no_ground_state, "M_DA": adapted,
              "M_OTHER": (base_raw, base_other), "M_JOINT": (adapted, adapted_joint),
              "base_r8_exact_masked_degenerate": base_exact_masked_degenerate,
              "adapted_r8_exact_masked_degenerate": adapted_exact_masked_degenerate,
              "query_rows_used_for_fit": 0}
    if state["M_OTHER"][0] is not state["M0"] or state["M_JOINT"][0] is not state["M_DA"]:
        raise GRBJP4SpikeError("five-arm required bank/domain object sharing drift")
    if adapted_state.jp4.k_shot == 1:
        if no_ground_state.wire_bytes() != adapted.wire_bytes():
            raise GRBJP4SpikeError("K1 JP4 identity must preserve r6 state bytes")
    return state


def five_arm_state_receipt(states: Mapping[str, Any], *, jp4: JP4FitState) -> Mapping[str, Any]:
    """Report five-arm pairing and K1 identity evidence for the runner receipt."""
    required = {"M0", "M_DA_NG", "M_DA", "M_OTHER", "M_JOINT",
                "base_r8_exact_masked_degenerate",
                "adapted_r8_exact_masked_degenerate", "query_rows_used_for_fit"}
    if set(states) != required or states.get("query_rows_used_for_fit") != 0:
        raise GRBJP4SpikeError("five-arm state schema drift")
    raw = states["M0"]; ng = states["M_DA_NG"]; adapted = states["M_DA"]
    other_bank, _other = states["M_OTHER"]; joint_bank, _joint = states["M_JOINT"]
    if (not isinstance(ng, DualQKNNState) or not isinstance(adapted, DualQKNNState)
            or raw is not ng.id_bank or other_bank is not raw or joint_bank is not adapted):
        raise GRBJP4SpikeError("five-arm state type/pairing drift")
    base_degenerate = states["base_r8_exact_masked_degenerate"]
    adapted_degenerate = states["adapted_r8_exact_masked_degenerate"]
    if type(base_degenerate) is not bool or type(adapted_degenerate) is not bool:
        raise GRBJP4SpikeError("five-arm r8 masked-degenerate receipt drift")
    return {"schema": SCHEMA, "query_rows_used_for_fit": 0, "k_shot": jp4.k_shot,
            "m0_id_bank_sha256": raw.digest, "m_da_ng_state_sha256": ng.digest,
            "m_da_state_sha256": adapted.digest, "m_other_reuses_m0_bank": True,
            "m_joint_reuses_m_da_state": True,
            "k1_identity_r6_bytes": bool(jp4.k_shot != 1 or ng.wire_bytes() == adapted.wire_bytes()),
            "base_r8_exact_masked_degenerate": base_degenerate,
            "adapted_r8_exact_masked_degenerate": adapted_degenerate,
            "jp4_theta_digest": jp4.digest}


def predict_five_arms(
    formal: FormalGRBJP4State,
    *,
    query_iq: torch.Tensor,
    query_physical_tokens: Sequence[Any] | np.ndarray,
) -> tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray], Mapping[str, Any]]:
    """Encode one matched query surface and close all five frozen arms."""

    if type(formal) is not FormalGRBJP4State:
        raise GRBJP4SpikeError("five-arm prediction requires a formal lifecycle")
    _validate_formal_state(formal, expected_stage=formal.stage)
    runtime_ownership = _runtime_ownership_from_state(formal)
    if (
        formal.runtime_phase != "BASE_READY"
        or formal.runtime is None
        or _sha(_weight_bytes(_joint_proj_linear(formal.runtime).weight))
        != formal.jp4.joint_weight_sha256_before
    ):
        raise GRBJP4SpikeError("five-arm prediction requires a formal lifecycle")
    tokens = typed_tokens(
        query_physical_tokens, name="formal query physical tokens", unique=True
    )
    device = _runtime_device(formal.runtime)
    iq = _formal_iq(query_iq, device=device)
    if len(tokens) != len(iq) or set(tokens).intersection(formal.all_support_tokens):
        raise GRBJP4SpikeError("formal query row/token or support disjointness drift")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    started = time.perf_counter()
    base = strict_zid_with_hook(formal.runtime, iq)
    merge_started = time.perf_counter()
    merge_into_joint_proj(
        _joint_proj_linear(formal.runtime),
        state=formal.jp4,
        ground=formal.ground,
        checkpoint_sha256=formal.jp4.checkpoint_sha256,
    )
    _synchronize(device)
    query_merge_ms = (time.perf_counter() - merge_started) * 1000.0
    object.__setattr__(formal, "runtime_phase", "ADAPTED_QUERY_ACTIVE")
    adapted = strict_zid_with_hook(formal.runtime, iq)
    if base.z_dom is None or adapted.z_dom is None:
        raise GRBJP4SpikeError("formal query TorchScript tap lacks z_dom")
    states = build_five_arm_state_view(
        no_ground_state=formal.no_ground_state,
        adapted_state=formal.adapted_state,
    )
    try:
        m0 = qknn_logits(states["M0"], base.z_id)
        m_da_ng = dual_qknn_logits(
            states["M_DA_NG"], base.z_id, base.z_dom
        )
        m_da = dual_qknn_logits(
            states["M_DA"], adapted.z_id, adapted.z_dom
        )
        m_other = bcrr_fused_logits(
            m0,
            base.z_id,
            states["M_OTHER"][1],
            bank=states["M_OTHER"][0],
        )
        m_joint = bcrr_fused_logits(
            m_da,
            adapted.z_id,
            states["M_JOINT"][1],
            bank=states["M_JOINT"][0].id_bank,
        )
    except ADV3B02StateError as exc:
        raise GRBJP4SpikeError("formal five-arm scoring closure failed") from exc
    logits = {
        "M0": np.ascontiguousarray(m0, dtype=np.float32),
        "M_DA_NG": np.ascontiguousarray(m_da_ng, dtype=np.float32),
        "M_DA": np.ascontiguousarray(m_da, dtype=np.float32),
        "M_OTHER": np.ascontiguousarray(m_other, dtype=np.float32),
        "M_JOINT": np.ascontiguousarray(m_joint, dtype=np.float32),
    }
    expected_shape = (len(tokens), len(formal.registered_classes))
    if tuple(logits) != FIVE_ARM_NAMES or any(
        value.shape != expected_shape or not np.isfinite(value).all()
        for value in logits.values()
    ):
        raise GRBJP4SpikeError("formal five-arm row/class logits closure drift")
    if formal.jp4.k_shot == 1 and (
        not np.array_equal(logits["M0"], logits["M_DA_NG"])
        or not np.array_equal(logits["M0"], logits["M_DA"])
        or not np.array_equal(logits["M_OTHER"], logits["M_JOINT"])
    ):
        raise GRBJP4SpikeError("formal K1 five-arm prediction identity drift")
    predictions = {
        arm: np.asarray(
            [
                formal.registered_classes[int(index)]
                for index in np.argmax(value, axis=1)
            ],
            dtype=object,
        )
        for arm, value in logits.items()
    }
    _synchronize(device)
    query_ms = (time.perf_counter() - started) * 1000.0
    token_root = _ordered_root(tokens)
    class_root = _ordered_root(formal.registered_classes)
    rows = {
        arm: {
            "row_count": len(tokens),
            "class_count": len(formal.registered_classes),
            "query_token_root_sha256": token_root,
            "class_order_sha256": class_root,
            "lifecycle_stage": formal.stage,
            "logits_sha256": _array_digest(value),
            "prediction_sha256": _sha(
                _canon([str(item) for item in predictions[arm].tolist()])
            ),
        }
        for arm, value in logits.items()
    }
    if len({_sha(_canon({
        "row_count": row["row_count"],
        "class_count": row["class_count"],
        "query_token_root_sha256": row["query_token_root_sha256"],
        "class_order_sha256": row["class_order_sha256"],
        "lifecycle_stage": row["lifecycle_stage"],
    })) for row in rows.values()}) != 1:
        raise GRBJP4SpikeError("formal five-arm matched closure receipt drift")
    object.__setattr__(formal, "runtime_phase", "ADAPTED_QUERY_CONSUMED")
    object.__setattr__(formal, "runtime", None)
    _runtime_ownership_release(runtime_ownership)
    terminal_ownership = _runtime_terminal_ownership_receipt(
        runtime_ownership
    )
    receipt = {
        "schema": FORMAL_ORCHESTRATOR_SCHEMA,
        "stage": formal.stage,
        "k_shot": formal.jp4.k_shot,
        "arm_order": list(FIVE_ARM_NAMES),
        "rows": rows,
        "row_class_token_lifecycle_matched": True,
        "query_iq_sha256": _iq_digest(iq),
        "query_token_root_sha256": token_root,
        "class_order_sha256": class_root,
        "support_token_root_sha256": _ordered_root(formal.all_support_tokens),
        "support_query_physical_tokens_disjoint": True,
        "query_decision_mode": "independent_all_registered_classes",
        "query_batch_consumes_runtime_once": True,
        "query_rows_used_for_fit": 0,
        "state_receipt": dict(five_arm_state_receipt(states, jp4=formal.jp4)),
        "resource_receipt": {
            **dict(formal.resources),
            "query_rows": len(tokens),
            "query_forward_calls": 2,
            "query_time_ms": query_ms,
            "query_one_way_merge_time_ms": query_merge_ms,
            "query_one_way_merge_mac": int(
                dict(formal.resources)["weight_merge_mac"]
            ),
            "query_peak_gpu_memory_bytes": _peak_gpu_bytes(device),
            "adapter_mac_per_query": 0,
            "five_arm_evaluation_one_way_merge": True,
            "runtime_terminal_phase": formal.runtime_phase,
            "runtime_terminal_ownership_receipt": terminal_ownership,
            "second_full_model_weight_copy": False,
        },
    }
    return logits, predictions, receipt


def resource_receipt(
    state: JP4FitState,
    ground: GroundReceiverBasis,
    *,
    parent_state_bytes: int = 0,
    support_rows: int = 0,
    support_forward_calls: int = 0,
    analytic_jacobian_rows: int = 0,
    fit_time_ms: float = 0.0,
    merge_time_ms: float = 0.0,
    peak_gpu_memory_bytes: int = 0,
) -> dict[str, Any]:
    """Frozen payload plus complete formal fit/merge resource evidence."""
    if state.ground_digest != ground.digest:
        raise GRBJP4SpikeError("resource receipt state/ground binding drift")
    numeric = (ground.prototype_codes.nbytes + ground.prototype_scales.nbytes
               + ground.left_codes.nbytes + ground.left_scales.nbytes
               + ground.right_codes.nbytes + ground.right_scales.nbytes
               + state.theta_codes.nbytes + state.theta_scale.nbytes)
    if numeric != GROUND_NUMERIC_PAYLOAD_BYTES:
        raise GRBJP4SpikeError("frozen GRB numeric payload contract drift")
    wire_bytes = len(ground.wire_bytes()) + len(state.wire_bytes())
    if wire_bytes > JP4_WIRE_LIMIT:
        raise GRBJP4SpikeError("JP4 metadata/payload exceeds frozen 4096-byte wire limit")
    values = (
        parent_state_bytes,
        support_rows,
        support_forward_calls,
        analytic_jacobian_rows,
        peak_gpu_memory_bytes,
    )
    if any(type(value) is not int or value < 0 for value in values):
        raise GRBJP4SpikeError("JP4 resource counter drift")
    if (
        not np.isfinite(fit_time_ms)
        or float(fit_time_ms) < 0.0
        or not np.isfinite(merge_time_ms)
        or float(merge_time_ms) < 0.0
    ):
        raise GRBJP4SpikeError("JP4 resource timing drift")
    if parent_state_bytes > PARENT_R6_MAX_STATE_BYTES:
        raise GRBJP4SpikeError("parent r6 state exceeds frozen maximum")
    total_state = parent_state_bytes + wire_bytes
    if total_state > COMBINED_MAX_STATE_BYTES or total_state > MAX_STATE_BYTES:
        raise GRBJP4SpikeError("combined GRB+r6 state exceeds frozen maximum")
    analytic_mac = (
        analytic_jacobian_rows
        * RANK
        * (Z_DIM * HIDDEN_DIM + 3 * Z_DIM)
    )
    solve_mac = 0 if state.k_shot == 1 else RANK ** 3
    merge_mac = Z_DIM * RANK * (2 * HIDDEN_DIM - 1)
    return {"schema": SCHEMA, "rank": RANK, "optimizer_steps": 0,
            "ground_numeric_payload_bytes": numeric - THETA_WIRE_BYTES,
            "theta_numeric_payload_bytes": THETA_WIRE_BYTES,
            "numeric_payload_bytes": numeric, "jp4_metadata_and_payload_bytes": wire_bytes,
            "jp4_wire_limit_bytes": JP4_WIRE_LIMIT, "adapter_mac_per_query": 0,
            "merge_scratch_fp16_max_bytes": Z_DIM * HIDDEN_DIM * 2,
            "support_rows": support_rows,
            "support_forward_calls": support_forward_calls,
            "analytic_jacobian_rows": analytic_jacobian_rows,
            "analytic_jacobian_mac": analytic_mac,
            "closed_form_solve_shape": [RANK, RANK],
            "closed_form_solve_mac": solve_mac,
            "weight_merge_mac": merge_mac,
            "fit_time_ms": float(fit_time_ms),
            "merge_time_ms": float(merge_time_ms),
            "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
            "parent_r6_state_bytes": parent_state_bytes,
            "parent_r6_state_limit_bytes": PARENT_R6_MAX_STATE_BYTES,
            "total_candidate_state_bytes": total_state,
            "combined_candidate_state_limit_bytes": COMBINED_MAX_STATE_BYTES,
            "absolute_state_limit_bytes": MAX_STATE_BYTES,
            "r6_head_mac_c26_k10": HEAD_MAC_C26_K10,
            "second_full_model_weight_copy": False,
            "query_rows_used_for_fit": 0}


def factor_int8_replay(left: np.ndarray, right: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Diagnostic compact replay retained for the feasibility test."""
    lc, ls = _quantize_rows(np.asarray(left, np.float32)); rc, rs = _quantize_rows(np.asarray(right, np.float32)); tc, ts = _theta_codec(np.asarray(theta, np.float32))
    replay = np.einsum("r,ri,rj->ij", _decode_theta(tc, ts), _decode_rows(lc, ls, (RANK, Z_DIM), "left"), _decode_rows(rc, rs, (RANK, HIDDEN_DIM), "right")).astype(np.float32)
    return replay, {"left_codes_shape": list(lc.shape), "right_codes_shape": list(rc.shape), "theta_codes": tc.tolist(),
                    "state_bytes": int(lc.nbytes + ls.nbytes + rc.nbytes + rs.nbytes + tc.nbytes + ts.nbytes)}


def geometry_change(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    b, a = np.asarray(before["neighbor_class"]), np.asarray(after["neighbor_class"])
    bm, am = np.asarray(before["margin"], np.float64), np.asarray(after["margin"], np.float64)
    return {"neighbor_changed_count": int(np.count_nonzero(b != a)), "margin_changed_count": int(np.count_nonzero(np.abs(bm - am) > 0.0)),
            "large_margin_flip_count": int(np.count_nonzero((bm > 0.05) & (b != a)))}


__all__ = [
    "CANDIDATE", "SCHEMA", "RANK", "Z_DIM", "HIDDEN_DIM",
    "OLD_CLASS_COUNT", "THETA_WIRE_BYTES", "GROUND_NUMERIC_PAYLOAD_BYTES",
    "GRBJP4SpikeError", "GroundReceiverBasis", "JP4FitState",
    "GRBStage2BState", "GRBStage2CState", "FormalGRBJP4State",
    "StrictForward", "analytic_jacobian", "append_formal_stage2_c",
    "autograd_jacobian", "build_five_arm_state_view",
    "checkpoint_right_factors", "deserialize_jp4_fit_state", "directions",
    "factor_int8_replay", "fit_stage2_b_from_support_iq",
    "geometry_change", "merge_into_joint_proj", "five_arm_state_receipt",
    "observed_ground_left_factors", "predict_five_arms",
    "prepare_support_for_jp4_fit", "resource_receipt",
    "serialize_jp4_fit_state", "solve_theta", "strict_zid_with_hook",
]
