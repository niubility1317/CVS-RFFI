"""DSSC-ZDOM-JG-qKNN-R4-BCRR/design-r1f support-only primitives.

This module deliberately has no capsule builder, scorer, or query-label API.
It turns a sealed Phase-1 dual-feature archive into a compact ground bundle,
fits the four shared adapter coefficients from support only, merges the delta
into the real identity head, and exposes a small all-class INT8 qKNN/BCRR
runtime.  Tokens are deliberately typed: conflicting values are rejected
rather than normalised through ``str``.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import parametrize
from cvsrffi.stage2_svrn_bcr import (build_branch_state as _build_svrn_branch,
    qknn_neighbor_receipt as _svrn_neighbors, score_branch_logits as _svrn_scores,
    serialize_branch_state as _serialize_svrn_branch)
from cvsrffi.stage2_zid_student_t_qknn import (
    Phase1ZIDStudentTLock,
    deserialize_typed_zid_runtime_state,
    score_zid_student_t_logits,
)

CANDIDATE = "DSSC_ZDOM_JG_QKNN_R4_BCRR/design-r1f"
SCHEMA = "cvs.stage2.dssc_zdom_jg_qknn_r4_bcrr.v1"
BUNDLE_SCHEMA = "cvs.phase1.dssc_zdom_jg_bundle.v1"
ARMS = ("M0", "M_DA_NG", "M_DA", "M_OTHER", "M_JOINT")
SCENES = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
RANK = 4
Z_DIM = 160
MAX_WIRE_BYTES = 256 * 1024
GEOFF_R8_COVERAGE_SHA256 = "c6e25ebeaed32b577e3321e78cd569acff934a7c804d0cb621b26e68f26d0c17"
SOMPH_PACKAGE_LOCK_SHA256 = "0496594db4a82efbbf17ec3d67ebc3fb1f0c7ced41b542a5a0bde3482e704523"
SEALED_RUNTIME_SHA256 = "f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a"
PHASE1_CHECKPOINT_SHA256 = "2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98"
PHASE1_ARCHIVE_SHA256 = "dd2a2b0c8ab1a1d8edbeed81e78ffb79c253240998a9ac2404b75699f4ca68d0"
PHASE1_ARCHIVE_MANIFEST_SHA256 = "34213331d20594dceface61680ab0fea8ffc40ee72d7e13c844763c70fef26d4"
PHASE1_PARITY_RECEIPT_SHA256 = "b93219c40b79be8ecdf8c0a51d77710d8119f8899331ae7e2518b77adfeac60b"
SGD_LR = 0.02
SGD_WEIGHT_DECAY = 1.0e-4
BCRR_MAX_OMEGA = 0.5
ADAPTER_SCALE_GROUPS = ((0, 1), (2, 3))
MIN_ADAPTER_FP16_SCALE = float(
    np.nextafter(np.float16(0.0), np.float16(1.0))
)
ZID_ZERO_NORM_TOTALIZATION_REVISION = "r1f-techfix4"
_ZID_ZERO_NORM_EPSILON = 1.0e-12


class DSSCStateError(ValueError):
    """Raised when a frozen r1f type, state, or protocol invariant drifts."""


@dataclass(frozen=True)
class ZIDZeroNormTotalizationReceipt:
    """Immutable audit record for the row-local adapted-z_id identity fallback."""

    revision: str
    branch: str
    scope: str
    state: str
    scene: str
    row_count: int
    replaced_count: int
    rate: float
    raw_valid: bool
    query_truth_used: bool = False
    state_updated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "branch": self.branch,
            "scope": self.scope,
            "state": self.state,
            "scene": self.scene,
            "row_count": self.row_count,
            "replaced_count": self.replaced_count,
            "rate": self.rate,
            "raw_valid": self.raw_valid,
            "query_truth_used": self.query_truth_used,
            "state_updated": self.state_updated,
        }


def totalize_adapted_zid(
    adapted_zid: np.ndarray,
    raw_zid: np.ndarray,
    *,
    branch: str,
    scope: str,
    state: str,
    scene: str,
    revision: str = ZID_ZERO_NORM_TOTALIZATION_REVISION,
) -> tuple[np.ndarray, ZIDZeroNormTotalizationReceipt]:
    """Replace only zero-norm adapted rows with same-IQ raw ``z_id`` rows.

    This is a strict row-local totalization, not a fit or a gate: it sees no
    labels, roles, truth, other queries, or mutable runtime state.  Raw M0
    features are validated here but never modified by this helper.
    """
    if revision != ZID_ZERO_NORM_TOTALIZATION_REVISION:
        raise DSSCStateError("z_id totalization revision drift")
    if branch not in ("no_ground", "ground"):
        raise DSSCStateError("z_id totalization branch must be no_ground/ground")
    if scope not in ("support", "query"):
        raise DSSCStateError("z_id totalization scope must be support/query")
    if state not in ("before", "after") or scene not in SCENES:
        raise DSSCStateError("z_id totalization state/scene drift")
    adapted = _rows(adapted_zid, name="adapted z_id for totalization")
    raw = _rows(raw_zid, name="raw z_id for totalization")
    if adapted.shape != raw.shape:
        raise DSSCStateError("adapted/raw z_id totalization shape drift")
    raw_norm = np.linalg.norm(raw.astype(np.float64), axis=1)
    if np.any(raw_norm <= _ZID_ZERO_NORM_EPSILON):
        raise DSSCStateError("raw z_id totalization requires every row norm > 1e-12")
    adapted_norm = np.linalg.norm(adapted.astype(np.float64), axis=1)
    replace = adapted_norm <= _ZID_ZERO_NORM_EPSILON
    # Copy first so that all nonzero rows retain their exact float32 bytes.
    totalized = adapted.copy()
    totalized[replace] = raw[replace]
    count = int(np.count_nonzero(replace))
    receipt = ZIDZeroNormTotalizationReceipt(
        revision=revision,
        branch=branch,
        scope=scope,
        state=state,
        scene=scene,
        row_count=int(len(totalized)),
        replaced_count=count,
        rate=float(count / len(totalized)) if len(totalized) else 0.0,
        raw_valid=True,
    )
    return totalized, receipt


def _canon(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


_FROZEN_AUTHORITY_BINDING = {
    "phase1_checkpoint_sha256": PHASE1_CHECKPOINT_SHA256,
    "phase1_archive_sha256": PHASE1_ARCHIVE_SHA256,
    "phase1_archive_manifest_sha256": PHASE1_ARCHIVE_MANIFEST_SHA256,
    "phase1_parity_receipt_sha256": PHASE1_PARITY_RECEIPT_SHA256,
    "somph_package_lock_sha256": SOMPH_PACKAGE_LOCK_SHA256,
    "sealed_runtime_sha256": SEALED_RUNTIME_SHA256,
    "geoff_r8_coverage_sha256": GEOFF_R8_COVERAGE_SHA256,
}
_QKNN_LODO_RECEIPT_SHA256 = sha256_bytes(
    _canon(
        {
            "schema": "cvs.stage2.dssc.phase1_lodo_authority_binding.v1",
            "purpose": "phase1_lodo_parameter_authority",
            "authority": _FROZEN_AUTHORITY_BINDING,
        }
    )
)
_QKNN_QUANTIZATION_RECEIPT_SHA256 = sha256_bytes(
    _canon(
        {
            "schema": "cvs.stage2.dssc.quantization_authority_binding.v1",
            "purpose": "int8_margin_audit_authority",
            "authority": _FROZEN_AUTHORITY_BINDING,
        }
    )
)
_CANONICAL_METHOD_LOCK = {
    "schema": "cvs.stage2.dssc_zdom_jg_qknn_r4_bcrr.lock.v1",
    "candidate": CANDIDATE,
    "protocol_schema": "p2_min_v1",
    "ground_old_multiprototype_enabled": True,
    "enable": True,
    "arms": list(ARMS),
    "rank": RANK,
    "dual_view": "raw_received_iq+fixed_rms_view",
    "loss": "symmetric_class_balanced_cross_view_prototype_ce+ground_center_ridge",
    "k1_steps": [2, 3],
    "k5_k10_steps": [25, 25],
    "optimizer": {
        "name": "SGD",
        "lr": SGD_LR,
        "weight_decay": SGD_WEIGHT_DECAY,
        "momentum": 0.0,
    },
    "adapter_quantization": {
        "code_dtype": "int8_symmetric_no_minus128",
        "scale_dtype": "float16",
        "scale_groups": [list(group) for group in ADAPTER_SCALE_GROUPS],
        "scale_floor": "minimum_positive_float16_subnormal",
        "persistent_fp32_sidecar_allowed": False,
    },
    "bcrr": {
        "formula": "F=N(Q)+omega[N(B)-N(Q)]",
        "omega_max": BCRR_MAX_OMEGA,
        "k1_omega": 0.0,
    },
    **_FROZEN_AUTHORITY_BINDING,
    "qknn": {
        "student_nu": 3.0,
        "kernel_effective_dim": Z_DIM,
        "kernel_volume_gamma": 1.0,
        "shared_h0": 0.2,
        "scale_prior_strength": 2.0,
        "scale_min_ratio": 0.5,
        "scale_max_ratio": 2.0,
        "temperature": 1.0,
        "phase1_lodo_receipt_sha256": _QKNN_LODO_RECEIPT_SHA256,
        "quantization_margin_audit_sha256": _QKNN_QUANTIZATION_RECEIPT_SHA256,
    },
    "resource_profile": {
        "tool": "runtime_F.conv1d_F.linear_interception_plus_torch_profiler_mm",
        "tool_version": "torch_environment_bound",
        "checkpoint_sha256": PHASE1_CHECKPOINT_SHA256,
        "input_shape": [1, 2, 256],
        "full_dual_return_aux_mac_per_sample": 38890840,
        "id_backbone_feat_joint_mac_per_sample": 9927476,
        "scope": "conv1d+linear+mm_batch1_frozen_checkpoint;_head_MAC_runtime_exact",
    },
    "query_fit_access": False,
    "query_policy": "per_sample_all_registered_classes",
    "full125": {
        "jobs": 125,
        "prediction_slices": 375,
        "score_rows": 1875,
        "gpu_dynamic_queue": 8,
    },
    "d01_d18_contract": {
        "D01": {
            "arms": list(ARMS),
            "arm_count": 5,
            "sixth_arm_allowed": False,
        },
        "D02": {
            "matched_fields": [
                "rank4_adapter",
                "dual_view",
                "optimizer",
                "steps",
                "S_B_S_C",
                "quantization",
            ],
            "only_difference_M_DA_NG_vs_M_DA": "ground_prior_mask",
            "both_require_nonzero_merged_delta": True,
        },
        "D03": {
            "adapter_targets": ["id_gate.0", "joint_proj.0"],
            "shared_coefficient_count": 4,
            "weight_delta_rank": RANK,
            "query_path_after_merge": "identity_backbone_only_feat_joint_no_domain_backbone",
            "query_adapter_extra_mac": 0,
        },
        "D04": {
            "S_B_fit": "old_support_only",
            "S_C_fit": "continue_S_B_all_registered_class_balanced",
            "query_fit_access": False,
            "query_early_stop_access": False,
            "query_fallback_access": False,
            "query_candidate_selection_access": False,
        },
        "D05": {
            "state_scope": "per_scene_independent_physical_ids_adapter_BCRR",
            "historical_repeated_three_LEO_support_interface": False,
        },
        "D06": {
            "views": ["fixed_received_IQ_raw", "fixed_received_IQ_RMS"],
            "mathematical_view_adds_K": False,
            "qknn_bank_view": "raw_view_adapted_feature_only",
        },
        "D07": {
            "loss": "symmetric_class_balanced_cross_view_prototype_CE",
            "ground_only_addition": "ground_center_ridge",
            "additional_losses_allowed": False,
        },
        "D08": {
            "K1_steps": {"S_B": 2, "S_C": 3},
            "K5_K10_steps": {"S_B": 25, "S_C": 25},
            "optimizer": {
                "name": "SGD",
                "lr": SGD_LR,
                "weight_decay": SGD_WEIGHT_DECAY,
                "momentum": 0.0,
            },
        },
        "D09": {
            "ground_prototypes_per_old_class": [1, 3],
            "min_distinct_phase1_physical_records_per_prototype": 2,
            "body_dtype": "int8_with_fp16_row_scale",
            "member_physical_ids_persisted": False,
            "persistent_fp32_sidecar_allowed": False,
        },
        "D10": {
            "domain_basis": "class_shared_independently_sealed",
            "max_rank": RANK,
            "dtype": "int8_with_fp16_row_scale",
            "legacy_ground_component_reuse_or_rename_allowed": False,
            "ground_old_multiprototype_enabled": True,
        },
        "D11": {
            "ground_roles": ["adapter_initialization", "adapter_constraint"],
            "ground_direct_vote_allowed": False,
            "ground_old_logit_boost_allowed": False,
            "ground_new_unknown_state_allowed": False,
            "method_change_triggers_phase2_data_revalidation": False,
        },
        "D12": {
            "qknn_competition": "per_query_all_registered_classes",
            "M0_M_OTHER_share_raw_qknn": True,
            "M_DA_M_JOINT_share_adapter_and_adapted_qknn": True,
        },
        "D13": {
            "formula": "F=N(Q)+omega[N(B)-N(Q)]",
            "omega_range": [0.0, BCRR_MAX_OMEGA],
            "fit": "branch_local_support_only",
            "K1_omega": 0.0,
            "global_synergy_gate_allowed": False,
        },
        "D14": {
            "wire_limit_bytes": MAX_WIRE_BYTES,
            "required_state_components": [
                "ground",
                "adapter",
                "qknn",
                "BCRR",
            ],
            "required_resource_evidence": [
                "int8_consistency",
                "build_time",
                "predict_mean",
                "predict_p95",
                "VRAM",
                "forward_counts",
                "MAC",
            ],
            "id_backbone_mac_per_sample": 9927476,
            "full_dual_return_aux_mac_per_sample": 38890840,
            "adapter_quantization": "two_contiguous_rank_groups_int8_with_fp16_scale",
        },
        "D15": {
            "required_held_changes": [
                "feature",
                "neighbor_order",
                "argmax",
                "wrong_to_correct",
                "correct_to_wrong",
            ],
            "numeric_only_change_is_DA_success": False,
        },
        "D16": {
            "scientific_file_scope": [
                "stage2_dssc_zdom_jg_qknn_r4_bcrr.py",
                "build_phase1_dssc_zdom_jg_bundle.py",
                "run_dssc_zdom_jg_qknn_r4_bcrr_125.py",
                "test_stage2_dssc_zdom_jg_qknn_r4_bcrr.py",
            ],
            "existing_model_qknn_bcrr_data_builder_scorer_modification_allowed": False,
        },
        "D17": {
            "jobs": 125,
            "scenes_per_job": 3,
            "arms_per_scene": 5,
            "prediction_slices": 375,
            "score_rows": 1875,
            "default_gpu_dynamic_queue_workers": 8,
            "narrow_performance_run_allowed": False,
        },
        "D18": {
            "missing_prediction_status": "TECHNICAL_FAILURE_NO_PERFORMANCE_RESULT",
            "completed_gate_failure_status": "COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE",
            "full125_is_for_frozen_candidate_validation_only": True,
        },
    },
}


def canonical_method_lock() -> dict[str, Any]:
    """Return a detached copy of the one frozen DSSC component lock."""
    return json.loads(_canon(_CANONICAL_METHOD_LOCK).decode("ascii"))


def validate_method_lock(value: Mapping[str, Any] | str) -> tuple[dict[str, Any], str]:
    """Require exact schema, keys and values for every D01--D18 field."""
    if type(value) is str:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise DSSCStateError("method lock JSON is invalid") from exc
    elif type(value) is dict:
        parsed = dict(value)
    else:
        raise DSSCStateError("method lock must be a JSON object")
    if parsed != _CANONICAL_METHOD_LOCK:
        raise DSSCStateError("DSSC method lock exact schema/value drift")
    return canonical_method_lock(), sha256_bytes(_canon(parsed))


def qknn_lock_from_method_lock(method_lock: Mapping[str, Any] | str, *, k_shot: int) -> Phase1ZIDStudentTLock:
    """Construct the only permitted qKNN lock from the sealed r1f lock."""
    lock, _digest = validate_method_lock(method_lock)
    q = lock["qknn"]
    if q["phase1_lodo_receipt_sha256"] == q["quantization_margin_audit_sha256"]:
        raise DSSCStateError("qKNN authority receipts must be purpose-distinct")
    if type(k_shot) is not int or k_shot not in (1, 5, 10):
        raise DSSCStateError("qKNN active K is outside the frozen set")
    try:
        return Phase1ZIDStudentTLock(
            k_shot,
            q["student_nu"],
            q["kernel_effective_dim"],
            q["kernel_volume_gamma"],
            q["shared_h0"],
            q["scale_prior_strength"],
            q["scale_min_ratio"],
            q["scale_max_ratio"],
            q["temperature"],
            _require_sha(q["phase1_lodo_receipt_sha256"], "qKNN phase1 receipt"),
            _require_sha(
                q["quantization_margin_audit_sha256"],
                "qKNN quantization receipt",
            ),
        )
    except (TypeError, ValueError) as exc:
        raise DSSCStateError("DSSC exact qKNN lock values are invalid") from exc


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha(value: Any, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise DSSCStateError(f"{name} must be an exact lower-case SHA256")
    return value


def typed_tokens(values: Sequence[Any] | np.ndarray, *, name: str, unique: bool = False) -> tuple[str, ...]:
    """Accept only actual unicode tokens; never silently stringify identifiers."""
    if isinstance(values, np.ndarray):
        if values.ndim != 1 or values.dtype.kind != "U":
            raise DSSCStateError(f"{name} must be a one-dimensional unicode ndarray")
        result = tuple(values.tolist())
    elif isinstance(values, (tuple, list)):
        result = tuple(values)
    else:
        raise DSSCStateError(f"{name} must be a typed token sequence")
    if not result or any(type(value) is not str or not value for value in result):
        raise DSSCStateError(f"{name} contains a non-string or empty token")
    if unique and len(set(result)) != len(result):
        raise DSSCStateError(f"{name} must be unique")
    return result


def _rows(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.float32 or array.ndim != 2 or array.shape[1] != Z_DIM or not np.isfinite(array).all():
        raise DSSCStateError(f"{name} must be finite float32 [N,{Z_DIM}]")
    return np.ascontiguousarray(array)


def _unit(value: np.ndarray) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64)
    return np.asarray(rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1.0e-12), np.float32)


def rms_view(value: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(value) or value.ndim < 2 or not torch.isfinite(value).all():
        raise DSSCStateError("dual-view input must be finite received-IQ tensor")
    rms = torch.sqrt(torch.mean(value.square(), dim=tuple(range(1, value.ndim)), keepdim=True) + 1.0e-8)
    return value / rms


def _quantize_rows(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(value, np.float32)
    scale = np.maximum(np.max(np.abs(rows), axis=1), 1.0e-8) / 127.0
    codes = np.clip(np.rint(rows / scale[:, None]), -127, 127).astype(np.int8)
    return codes, scale.astype(np.float16)


def _dequantize_rows(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    if codes.dtype != np.int8 or scales.dtype != np.float16 or codes.ndim != 2 or scales.shape != (codes.shape[0],):
        raise DSSCStateError("INT8 row state shape/dtype drift")
    return np.asarray(codes.astype(np.float32) * scales.astype(np.float32)[:, None], np.float32)


@dataclass(frozen=True)
class GroundBundle:
    classes: tuple[str, ...]
    prototype_class_indices: np.ndarray
    prototype_physical_counts: np.ndarray
    z_id_codes: np.ndarray
    z_id_scales: np.ndarray
    z_dom_codes: np.ndarray
    z_dom_scales: np.ndarray
    u_id_codes: np.ndarray
    u_id_scales: np.ndarray
    v_dom_codes: np.ndarray
    v_dom_scales: np.ndarray
    singular_values_fp16: np.ndarray
    z_dom_center_fp16: np.ndarray
    manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.classes or len(set(self.classes)) != len(self.classes):
            raise DSSCStateError("ground class registry must be nonempty and unique")
        p = len(self.prototype_class_indices)
        if (self.prototype_class_indices.dtype != np.int16 or self.prototype_physical_counts.dtype != np.uint16
                or self.prototype_class_indices.shape != (p,) or self.prototype_physical_counts.shape != (p,)):
            raise DSSCStateError("ground prototype metadata drift")
        if np.any(self.prototype_physical_counts < 2) or np.any(self.prototype_class_indices < 0) or np.any(self.prototype_class_indices >= len(self.classes)):
            raise DSSCStateError("ground prototype physical-count/class invariant drift")
        for codes, scales in ((self.z_id_codes, self.z_id_scales), (self.z_dom_codes, self.z_dom_scales)):
            if codes.shape != (p, Z_DIM):
                raise DSSCStateError("ground prototype dimension drift")
            _dequantize_rows(codes, scales)
        if self.u_id_codes.dtype != np.int8 or self.v_dom_codes.dtype != np.int8 or self.u_id_codes.ndim != 2 or self.v_dom_codes.ndim != 2 or self.u_id_codes.shape != self.v_dom_codes.shape or self.u_id_codes.shape[1] != Z_DIM:
            raise DSSCStateError("ground cross-covariance basis INT8 shape drift")
        if not 1 <= self.u_id_codes.shape[0] <= RANK:
            raise DSSCStateError("ground domain basis rank drift")
        _dequantize_rows(self.u_id_codes, self.u_id_scales); _dequantize_rows(self.v_dom_codes, self.v_dom_scales)
        if self.singular_values_fp16.dtype != np.float16 or self.singular_values_fp16.shape != (self.u_id_codes.shape[0],) or self.z_dom_center_fp16.dtype != np.float16 or self.z_dom_center_fp16.shape != (Z_DIM,):
            raise DSSCStateError("ground cross-covariance singular/center state drift")
        if type(self.manifest) is not dict or self.manifest.get("schema") != BUNDLE_SCHEMA:
            raise DSSCStateError("ground bundle manifest schema drift")
        if self.manifest.get("ground_old_multiprototype_enabled") is not True or self.manifest.get("enable") is not True:
            raise DSSCStateError("ground bundle is not explicitly enabled")
        lock, lock_sha = validate_method_lock(self.manifest.get("method_lock"))
        if (
            self.manifest.get("method_lock") != lock
            or self.manifest.get("method_lock_sha256") != lock_sha
            or self.manifest.get("checkpoint_sha256") != PHASE1_CHECKPOINT_SHA256
            or self.manifest.get("archive_sha256") != PHASE1_ARCHIVE_SHA256
            or self.manifest.get("archive_manifest_sha256") != PHASE1_ARCHIVE_MANIFEST_SHA256
        ):
            raise DSSCStateError("ground bundle canonical archive/lock/checkpoint binding drift")

    @property
    def z_id(self) -> np.ndarray:
        return _unit(_dequantize_rows(self.z_id_codes, self.z_id_scales))

    @property
    def z_dom(self) -> np.ndarray:
        return _unit(_dequantize_rows(self.z_dom_codes, self.z_dom_scales))

    @property
    def u_id(self) -> np.ndarray:
        return _unit(_dequantize_rows(self.u_id_codes, self.u_id_scales))

    @property
    def v_dom(self) -> np.ndarray:
        return _unit(_dequantize_rows(self.v_dom_codes, self.v_dom_scales))


def _lock_document(value: Mapping[str, Any] | str) -> tuple[dict[str, Any], str]:
    return validate_method_lock(value)


def build_ground_bundle_arrays(*, z_id: np.ndarray, z_dom: np.ndarray, labels: Sequence[Any] | np.ndarray,
                               physical_ids: Sequence[Any] | np.ndarray, archive_sha256: str,
                               archive_manifest_sha256: str, checkpoint_sha256: str,
                               method_lock: Mapping[str, Any] | str) -> dict[str, np.ndarray]:
    """Build the target-independent Phase-1-only INT8 ground bundle arrays."""
    zid, zdom = _rows(z_id, name="archive z_id"), _rows(z_dom, name="archive z_dom")
    labs = typed_tokens(labels, name="archive labels")
    ids = typed_tokens(physical_ids, name="archive physical IDs", unique=True)
    if len(zid) != len(zdom) or len(labs) != len(zid) or len(ids) != len(zid):
        raise DSSCStateError("archive dual-feature/label/physical layout drift")
    archive_sha256 = _require_sha(archive_sha256, "archive_sha256")
    archive_manifest_sha256 = _require_sha(archive_manifest_sha256, "archive_manifest_sha256")
    checkpoint_sha256 = _require_sha(checkpoint_sha256, "checkpoint_sha256")
    lock, lock_sha = _lock_document(method_lock)
    if checkpoint_sha256 != PHASE1_CHECKPOINT_SHA256:
        raise DSSCStateError("ground bundle checkpoint differs from frozen DSSC lock")
    if archive_sha256 != PHASE1_ARCHIVE_SHA256 or archive_manifest_sha256 != PHASE1_ARCHIVE_MANIFEST_SHA256:
        raise DSSCStateError("ground bundle source differs from frozen GEOFF/r8 archive")
    # The archive order is the only order available at Phase-1 sealing time.
    # Do not lexicographically re-sort opaque Stage2 handles later: row code
    # verifies this exact registry against the sealed target-old registry.
    classes = tuple(dict.fromkeys(labs))
    p_zid: list[np.ndarray] = []; p_zdom: list[np.ndarray] = []; p_cls: list[int] = []; p_count: list[int] = []
    lab_array = np.asarray(labs)
    id_array = np.asarray(ids)
    for ci, label in enumerate(classes):
        positions = np.flatnonzero(lab_array == label)
        # 1--3 deterministic groups, all independently aggregated physical records.
        group_count = min(3, max(1, len(positions) // 2))
        if group_count < 1:
            raise DSSCStateError("each old class requires at least two Phase-1 physical records")
        ordered = positions[np.argsort(id_array[positions], kind="stable")]
        groups = [part for part in np.array_split(ordered, group_count) if len(part)]
        if any(len(part) < 2 for part in groups):
            raise DSSCStateError("ground prototype would contain fewer than two physical records")
        for part in groups:
            p_zid.append(np.mean(zid[part], axis=0)); p_zdom.append(np.mean(zdom[part], axis=0))
            p_cls.append(ci); p_count.append(int(len(part)))
    proto_zid = _unit(np.stack(p_zid)); proto_zdom = _unit(np.stack(p_zdom))
    # Domain nuisance is a within-class z_dom -> z_id cross-covariance, not
    # a six-class identity-mean basis.  It is sealed before target access.
    id_residual = np.empty_like(zid); dom_residual = np.empty_like(zdom)
    for label in classes:
        part = lab_array == label
        id_residual[part] = zid[part] - np.mean(zid[part], axis=0, keepdims=True)
        dom_residual[part] = zdom[part] - np.mean(zdom[part], axis=0, keepdims=True)
    cross = id_residual.astype(np.float64).T @ dom_residual.astype(np.float64) / float(len(zid))
    u, singular, vt = np.linalg.svd(cross, full_matrices=False)
    effective = int(np.sum(singular > max(float(singular[0]) if len(singular) else 0.0, 1.0) * 1.0e-10))
    rank = min(RANK, max(1, effective))
    u_id, v_dom = _unit(np.asarray(u[:, :rank].T, np.float32)), _unit(np.asarray(vt[:rank], np.float32))
    zid_codes, zid_scales = _quantize_rows(proto_zid); zdom_codes, zdom_scales = _quantize_rows(proto_zdom)
    u_codes, u_scales = _quantize_rows(u_id); v_codes, v_scales = _quantize_rows(v_dom)
    arrays = {
        "classes": np.asarray(classes), "prototype_class_indices": np.asarray(p_cls, np.int16),
        "prototype_physical_counts": np.asarray(p_count, np.uint16), "z_id_codes": zid_codes,
        "z_id_scales": zid_scales, "z_dom_codes": zdom_codes, "z_dom_scales": zdom_scales,
        "u_id_codes": u_codes, "u_id_scales": u_scales, "v_dom_codes": v_codes, "v_dom_scales": v_scales,
        "singular_values_fp16": np.asarray(singular[:rank], np.float16), "z_dom_center_fp16": np.asarray(np.mean(zdom,axis=0),np.float16),
    }
    array_sha={name:sha256_bytes(np.ascontiguousarray(value).tobytes()) for name,value in arrays.items()}
    manifest = {
        "schema": BUNDLE_SCHEMA, "candidate": CANDIDATE, "enable": True,
        "ground_old_multiprototype_enabled": True, "archive_sha256": archive_sha256,
        "archive_manifest_sha256": archive_manifest_sha256, "checkpoint_sha256": checkpoint_sha256,
        "method_lock_sha256": lock_sha, "method_lock": lock, "feature_dim": Z_DIM,
        "prototype_count": len(p_cls), "prototype_max_per_class": 3,
        "prototype_min_distinct_physical_count": 2, "physical_member_ids_persisted": False,
        "domain_basis_rank": rank, "domain_basis_kind": "within_class_zdom_to_zid_cross_covariance",
        "component_dtype": "int8_with_fp16_row_scale",
        "target_access_before_seal": False, "array_sha256": array_sha,
        "bundle_id": sha256_bytes(_canon({"candidate":CANDIDATE,"checkpoint_sha256":checkpoint_sha256,"method_lock_sha256":lock_sha,"array_sha256":array_sha})),
    }
    return {**arrays,"manifest_json":np.asarray(json.dumps(manifest,sort_keys=True,separators=(",",":")))}


def load_ground_bundle(path: str | Path, *, checkpoint_sha256: str | None = None) -> GroundBundle:
    with np.load(Path(path), allow_pickle=False) as archive:
        required = {"classes", "prototype_class_indices", "prototype_physical_counts", "z_id_codes", "z_id_scales",
                    "z_dom_codes", "z_dom_scales", "u_id_codes", "u_id_scales", "v_dom_codes", "v_dom_scales", "singular_values_fp16", "z_dom_center_fp16", "manifest_json"}
        if set(archive.files) != required:
            raise DSSCStateError("ground bundle member allowlist drift")
        manifest_raw = archive["manifest_json"]
        if manifest_raw.shape != () or manifest_raw.dtype.kind != "U":
            raise DSSCStateError("ground manifest must be a scalar unicode JSON value")
        manifest = json.loads(manifest_raw.item())
        declared=manifest.get("array_sha256")
        if type(declared) is not dict or set(declared)!=(required-{ "manifest_json"}): raise DSSCStateError("ground bundle array-SHA allowlist drift")
        for name in declared:
            if declared[name] != sha256_bytes(np.ascontiguousarray(archive[name]).tobytes()): raise DSSCStateError("ground bundle array SHA drift")
        expected_id=sha256_bytes(_canon({"candidate":CANDIDATE,"checkpoint_sha256":manifest.get("checkpoint_sha256"),"method_lock_sha256":manifest.get("method_lock_sha256"),"array_sha256":declared}))
        if manifest.get("bundle_id")!=expected_id: raise DSSCStateError("ground bundle identity/lock binding drift")
        bundle = GroundBundle(typed_tokens(archive["classes"], name="ground classes", unique=True), archive["prototype_class_indices"],
                              archive["prototype_physical_counts"], archive["z_id_codes"], archive["z_id_scales"],
                              archive["z_dom_codes"], archive["z_dom_scales"], archive["u_id_codes"], archive["u_id_scales"], archive["v_dom_codes"], archive["v_dom_scales"], archive["singular_values_fp16"], archive["z_dom_center_fp16"], manifest)
    if checkpoint_sha256 is not None and bundle.manifest.get("checkpoint_sha256") != _require_sha(checkpoint_sha256, "checkpoint_sha256"):
        raise DSSCStateError("ground bundle/checkpoint binding drift")
    return bundle


def bundle_wire_bytes(bundle: GroundBundle) -> int:
    payload = {"schema": BUNDLE_SCHEMA, "manifest": dict(bundle.manifest), "classes": list(bundle.classes),
               "prototype_class_indices": bundle.prototype_class_indices.tolist(),
               "prototype_physical_counts": bundle.prototype_physical_counts.tolist(),
               "z_id_codes": bundle.z_id_codes.tolist(), "z_id_scales": bundle.z_id_scales.tolist(),
               "z_dom_codes": bundle.z_dom_codes.tolist(), "z_dom_scales": bundle.z_dom_scales.tolist(), "u_id_codes": bundle.u_id_codes.tolist(), "u_id_scales": bundle.u_id_scales.tolist(), "v_dom_codes": bundle.v_dom_codes.tolist(), "v_dom_scales": bundle.v_dom_scales.tolist(), "singular_values_fp16": bundle.singular_values_fp16.tolist(), "z_dom_center_fp16": bundle.z_dom_center_fp16.tolist()}
    return len(_canon(payload))


class _AddRank4(torch.nn.Module):
    def __init__(self, coefficients: torch.nn.Parameter, directions: torch.Tensor) -> None:
        super().__init__(); self.coefficients = coefficients; self.register_buffer("directions", directions)
    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return weight + torch.tensordot(self.coefficients, self.directions.to(dtype=weight.dtype), dims=1)


@dataclass
class Rank4Adapter:
    coefficients: torch.nn.Parameter
    target_modules: tuple[torch.nn.Module, ...]
    target_names: tuple[str, ...]
    parametrizations: tuple[_AddRank4, ...]
    ground_enabled: bool
    merged: bool = False
    coefficient_codes: np.ndarray | None = None
    coefficient_scale_fp16: np.ndarray | None = None

    def delta_norm(self) -> float:
        return float(sum(torch.linalg.vector_norm(p.directions * self.coefficients[:, None, None] if p.directions.ndim == 3 else p.directions * self.coefficients.reshape((-1,) + (1,) * (p.directions.ndim - 1))).detach().cpu().item() for p in self.parametrizations))

    def quantize_in_place(self) -> None:
        if self.merged:
            raise DSSCStateError("merged adapter cannot be requantized")
        if not torch.isfinite(self.coefficients).all() or float(torch.linalg.vector_norm(self.coefficients).detach()) == 0.0:
            raise DSSCStateError("adapter must have a non-zero finite four-coefficient delta")
        raw = np.asarray(self.coefficients.detach().cpu().tolist(), np.float32)
        group_maxima = [
            float(np.max(np.abs(raw[list(group)])))
            for group in ADAPTER_SCALE_GROUPS
        ]
        scale = np.asarray(
            [
                max(value / 127.0, MIN_ADAPTER_FP16_SCALE)
                for value in group_maxima
            ],
            np.float16,
        )
        codes = np.zeros((RANK,), np.int8)
        deployed_raw = np.zeros((RANK,), np.float32)
        for group_index, group in enumerate(ADAPTER_SCALE_GROUPS):
            indices = np.asarray(group, np.int64)
            codes[indices] = np.clip(
                np.rint(raw[indices] / float(scale[group_index])), -127, 127
            ).astype(np.int8)
            deployed_raw[indices] = codes[indices].astype(np.float32) * float(
                scale[group_index]
            )
        deployed = torch.tensor(deployed_raw, dtype=self.coefficients.dtype, device=self.coefficients.device)
        with torch.no_grad(): self.coefficients.copy_(deployed)
        self.coefficient_codes, self.coefficient_scale_fp16 = codes, scale

    def load_quantized(self, codes: np.ndarray, scale: np.ndarray) -> None:
        if self.merged:
            raise DSSCStateError("merged adapter cannot load a quantized state")
        q = np.asarray(codes)
        s = np.asarray(scale)
        if (
            q.dtype != np.int8
            or q.shape != (RANK,)
            or np.any(q == np.int8(-128))
            or s.dtype != np.float16
            or s.shape != (len(ADAPTER_SCALE_GROUPS),)
            or not np.isfinite(s).all()
            or np.any(s <= 0.0)
        ):
            raise DSSCStateError("adapter INT8 code/FP16 scale schema drift")
        deployed_raw = np.zeros((RANK,), np.float32)
        for group_index, group in enumerate(ADAPTER_SCALE_GROUPS):
            indices = np.asarray(group, np.int64)
            deployed_raw[indices] = q[indices].astype(np.float32) * float(
                s[group_index]
            )
        deployed = torch.tensor(deployed_raw, dtype=self.coefficients.dtype, device=self.coefficients.device)
        with torch.no_grad():
            self.coefficients.copy_(deployed)
        self.coefficient_codes = q.copy()
        self.coefficient_scale_fp16 = s.copy()

    def merge(self) -> None:
        if self.merged:
            raise DSSCStateError("adapter was already merged")
        if self.coefficient_codes is None or self.coefficient_scale_fp16 is None:
            self.quantize_in_place()
        for module in self.target_modules:
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
        self.merged = True


def _resolve_identity_targets(model: torch.nn.Module) -> tuple[tuple[str, torch.nn.Module], ...]:
    candidates: list[tuple[str, torch.nn.Module]] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear): continue
        if name.endswith("id_gate.0") or name.endswith("joint_proj.0"):
            if name.startswith("id_backbone.") or not any(x.startswith("id_backbone.") for x, _ in model.named_modules()):
                candidates.append((name, module))
    desired = [x for x in candidates if x[0].endswith("id_gate.0")] + [x for x in candidates if x[0].endswith("joint_proj.0")]
    if len(desired) != 2 or len({x[0] for x in desired}) != 2:
        raise DSSCStateError("exact identity id_gate.0/joint_proj.0 targets are required")
    return tuple(desired)


def _directions_for(weight: torch.Tensor, *, index: int) -> torch.Tensor:
    if weight.ndim != 2: raise DSSCStateError("rank-4 adapter only permits linear id-head weights")
    # Frozen shared rank directions are derived from the actual checkpoint
    # target weights.  This is deliberately independent of target support and
    # ground knowledge, so M_DA_NG and M_DA have identical delta geometry.
    left, _singular, right_t = torch.linalg.svd(weight.detach().to(torch.float32), full_matrices=False)
    available = min(RANK, int(left.shape[1]), int(right_t.shape[0]))
    if available < RANK:
        raise DSSCStateError("identity target weight cannot supply four checkpoint SVD directions")
    directions = [torch.outer(left[:, r], right_t[r, :]) for r in range(RANK)]
    return torch.stack(directions).to(device=weight.device, dtype=weight.dtype)


def attach_rank4_adapter(model: torch.nn.Module, bundle: GroundBundle | None, *, ground_enabled: bool,
                         support_zid_hint: np.ndarray | None = None, support_zdom_hint: np.ndarray | None = None) -> Rank4Adapter:
    if ground_enabled and bundle is None: raise DSSCStateError("M_DA requires a sealed ground bundle")
    # Only the shared four coefficients are trainable.  Freezing the checkpoint
    # parameters also prevents unused base-model gradients from inflating the
    # support-fit VRAM receipt while preserving the exact forward graph.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    targets = _resolve_identity_targets(model)
    device = targets[0][1].weight.device; dtype = targets[0][1].weight.dtype
    coeff = torch.nn.Parameter(torch.full((RANK,), 1.0e-4, dtype=dtype, device=device))
    if ground_enabled and support_zid_hint is not None and support_zdom_hint is not None:
        assert bundle is not None
        id_hint = _rows(support_zid_hint, name="support z_id hint").mean(axis=0)
        dom_hint = _rows(support_zdom_hint, name="support z_dom hint").mean(axis=0)
        # Explicitly use the sealed within-class U_id/V_dom nuisance map.
        # U_id scores the identity residual and V_dom scores the centred domain
        # residual in matching rank coordinates; only this prior differs from
        # the no-ground arm and subsequent support loss is identical.
        rank = len(bundle.singular_values_fp16)
        id_coordinate = bundle.u_id @ (id_hint - bundle.z_id.mean(axis=0))
        dom_coordinate = bundle.v_dom @ (dom_hint - bundle.z_dom_center_fp16.astype(np.float32))
        mapped = (id_coordinate + dom_coordinate) * bundle.singular_values_fp16.astype(np.float32)
        initial = np.zeros(RANK,np.float32); initial[:rank] = mapped
        with torch.no_grad(): coeff.copy_(torch.tensor(initial + 1.0e-4, dtype=dtype, device=device))
    parametrizations: list[_AddRank4] = []
    for index, (_, module) in enumerate(targets):
        addition = _AddRank4(coeff, _directions_for(module.weight.detach(), index=index))
        parametrize.register_parametrization(module, "weight", addition, unsafe=False); parametrizations.append(addition)
    adapter = Rank4Adapter(coeff, tuple(x[1] for x in targets), tuple(x[0] for x in targets), tuple(parametrizations), ground_enabled)
    if sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) != RANK:
        raise DSSCStateError("rank-4 adapter must be the model's only trainable state")
    return adapter


def _dual_features(model: torch.nn.Module, raw: torch.Tensor, *, need_dom: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    output = model(raw, return_aux=True) if "return_aux" in model.forward.__code__.co_varnames else model(raw)
    if not isinstance(output, Mapping) or not torch.is_tensor(output.get("z_id")):
        raise DSSCStateError("dual model must expose z_id in auxiliary support forward")
    zdom = output.get("z_dom")
    if need_dom and not torch.is_tensor(zdom): raise DSSCStateError("dual model must expose z_dom for ground initialisation")
    return F.normalize(output["z_id"], dim=1), None if zdom is None else F.normalize(zdom, dim=1)


def _class_balanced_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    terms = []
    for label in torch.unique(labels, sorted=True):
        mask = labels == label; terms.append(F.cross_entropy(logits[mask], labels[mask]))
    return torch.stack(terms).mean()


def _cross_view_loss(a: torch.Tensor, b: torch.Tensor, labels: torch.Tensor, physical_ids: tuple[str, ...], *, temperature: float) -> torch.Tensor:
    class_count = int(labels.max().detach()) + 1
    if len(physical_ids) != len(labels) or len(set(physical_ids)) != len(physical_ids): raise DSSCStateError("support physical IDs must be unique")
    # K>=2 removes the same physical row from the opposite-view prototype;
    # K1 deliberately uses that one paired RMS mathematical view.
    losses=[]
    for source,destination in ((a,b),(b,a)):
        logits=[]
        for i in range(len(labels)):
            columns=[]
            for c in range(class_count):
                same=(labels==c); keep=same.clone()
                if int(same.sum())>=2: keep[i]=False
                columns.append(F.normalize(destination[keep].mean(0,keepdim=True),dim=1)[0])
            logits.append(source[i] @ torch.stack(columns).T / temperature)
        losses.append(_class_balanced_ce(torch.stack(logits),labels))
    return 0.5*(losses[0]+losses[1])


def _labels_to_indices(labels: Sequence[Any] | np.ndarray, registry: Sequence[Any] | np.ndarray) -> torch.Tensor:
    values = typed_tokens(labels, name="support labels"); classes = typed_tokens(registry, name="registered classes", unique=True)
    if any(value not in classes for value in values):
        raise DSSCStateError("support labels/registered class registry drift")
    return torch.tensor([classes.index(value) for value in values], dtype=torch.long)


def adaptation_steps(k_shot: int, stage: str) -> int:
    if type(k_shot) is not int or k_shot not in (1, 5, 10) or stage not in ("S_B", "S_C"):
        raise DSSCStateError("r1f only permits K={1,5,10} and S_B/S_C")
    return (2 if stage == "S_B" else 3) if k_shot == 1 else 25


def adapt_support_only(model: torch.nn.Module, support_iq: torch.Tensor, support_labels: Sequence[Any] | np.ndarray,
                       registered_classes: Sequence[Any] | np.ndarray, *, k_shot: int, stage: str,
                       bundle: GroundBundle | None, ground_enabled: bool, support_physical_ids: Sequence[Any] | np.ndarray | None = None,
                       ground_old_registry: Sequence[Any] | np.ndarray | None = None,
                       continue_adapter: Rank4Adapter | None = None, merge: bool = True) -> tuple[Rank4Adapter, dict[str, Any]]:
    """Fit only four coefficients on legal support; this function has no query input."""
    labels = _labels_to_indices(support_labels, registered_classes).to(support_iq.device)
    classes = typed_tokens(registered_classes, name="registered classes", unique=True)
    if len(support_iq) != len(labels) or tuple(int((labels == c).sum()) for c in range(len(classes))) != (k_shot,) * len(classes):
        raise DSSCStateError("support must be exact class-balanced K-shot")
    if ground_enabled:
        if bundle is None or ground_old_registry is None:
            raise DSSCStateError("ground adaptation requires an explicit sealed old-registry binding")
        ground_registry = typed_tokens(ground_old_registry, name="sealed ground old registry", unique=True)
        if len(ground_registry) != len(bundle.classes):
            raise DSSCStateError("ground prototype/old-registry cardinality drift")
        if any(name not in classes for name in ground_registry):
            raise DSSCStateError("ground old registry is not contained in the current row registry")
    else:
        ground_registry = ()
    physical=typed_tokens([f"local_support_{i}" for i in range(len(labels))] if support_physical_ids is None else support_physical_ids,name="support physical IDs",unique=True)
    model.eval()
    with torch.no_grad(): zid_hint, zdom = _dual_features(model, support_iq, need_dom=True)
    if continue_adapter is not None:
        if continue_adapter.merged or continue_adapter.ground_enabled != ground_enabled: raise DSSCStateError("S_C must continue the unmerged matching S_B adapter state")
        adapter=continue_adapter
    else:
        adapter = attach_rank4_adapter(model, bundle, ground_enabled=ground_enabled,
                                       support_zid_hint=np.asarray(zid_hint.detach().cpu().tolist(), np.float32), support_zdom_hint=np.asarray(zdom.detach().cpu().tolist(), np.float32))
    optimizer = torch.optim.SGD([adapter.coefficients], lr=SGD_LR, weight_decay=SGD_WEIGHT_DECAY, momentum=0.0)
    steps = adaptation_steps(k_shot, stage); losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        raw, _ = _dual_features(model, support_iq, need_dom=False)
        rms, _ = _dual_features(model, rms_view(support_iq), need_dom=False)
        loss = _cross_view_loss(raw, rms, labels, physical, temperature=0.10)
        if ground_enabled:
            assert bundle is not None
            # Prototype slots are Phase-1 class slots; only this sealed row
            # binding gives them opaque Stage2 handles.  No archive label is
            # consulted by the target runtime.
            old_centers = {name: bundle.z_id[bundle.prototype_class_indices == i].mean(0) for i, name in enumerate(ground_registry)}
            ridge = []
            for ci, name in enumerate(classes):
                if name in old_centers:
                    target = torch.tensor(old_centers[name], device=raw.device, dtype=raw.dtype)
                    ridge.append((raw[labels == ci].mean(0) - target).square().mean())
            if ridge: loss = loss + 0.05 * torch.stack(ridge).mean()
        loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
    # Every deployable state, including the unmerged S_B continuation state,
    # is quantized before it can score or continue into S_C.
    with torch.no_grad():
        teacher, _ = _dual_features(model, support_iq, need_dom=False)
        centers = torch.stack(
            [
                F.normalize(
                    teacher[labels == ci].mean(0, keepdim=True), dim=1
                )[0]
                for ci in range(len(classes))
            ]
        )
        teacher_logits = teacher @ centers.T
    adapter.quantize_in_place()
    with torch.no_grad():
        deployed, _ = _dual_features(model, support_iq, need_dom=False)
        # Deployment rebuilds the support bank from the quantized model.  The
        # audit must therefore compare teacher geometry with deployed geometry,
        # rather than scoring deployed rows against stale teacher centers.
        deployed_centers = torch.stack(
            [
                F.normalize(
                    deployed[labels == ci].mean(0, keepdim=True), dim=1
                )[0]
                for ci in range(len(classes))
            ]
        )
        deployed_logits = deployed @ deployed_centers.T
    teacher_order = torch.topk(teacher_logits, k=2, dim=1).values
    row_error = torch.max(torch.abs(teacher_logits - deployed_logits), dim=1).values
    teacher_margin = teacher_order[:, 0] - teacher_order[:, 1]
    teacher_top = torch.argmax(teacher_logits, dim=1)
    deployed_top = torch.argmax(deployed_logits, dim=1)
    agree = teacher_top == deployed_top
    large = teacher_margin > 2.0 * row_error
    quant_audit = {
        "scope": "support_only_teacher_geometry_vs_int8_deployed_geometry",
        "top1_agreement": float(torch.mean(agree.float()).cpu()),
        "large_margin_row_count": int(torch.sum(large).cpu()),
        "large_margin_flip_count": int(torch.sum(large & ~agree).cpu()),
        "max_abs_logit_error": float(torch.max(row_error).cpu()),
        "teacher_margin_mean": float(torch.mean(teacher_margin).cpu()),
        "query_rows_used_for_fit": 0,
    }
    if (
        quant_audit["top1_agreement"] < 0.995
        or quant_audit["large_margin_flip_count"] != 0
    ):
        raise DSSCStateError(
            "adapter INT8 teacher/deployed support gate failed: "
            + json.dumps(quant_audit, sort_keys=True, separators=(",", ":"))
        )
    if merge:
        adapter.merge()
    return adapter, {"stage": stage, "K": k_shot, "steps": steps, "optimizer": "SGD", "lr": SGD_LR,
                     "weight_decay": SGD_WEIGHT_DECAY, "momentum": 0.0, "loss_first": losses[0], "loss_last": losses[-1],
                     "query_rows_used_for_fit": 0, "dual_view": "raw_received_iq+fixed_rms_view", "delta_norm": adapter.delta_norm(),
                     "targets": list(adapter.target_names), "merged": adapter.merged, "same_physical_loo": k_shot == 1,
                     "ground_old_registry_sha256": None if not ground_enabled else sha256_bytes(_canon(list(ground_registry))),
                     "adapter_int8_teacher_deployed": quant_audit,
                     "adapter_coefficients_qint8": adapter.coefficient_codes.tolist(),
                     "adapter_coefficient_scale_fp16": adapter.coefficient_scale_fp16.tolist()}


@dataclass(frozen=True)
class QKNNState:
    classes: tuple[str, ...]
    support_tokens: tuple[str, ...]
    branch_state: Any
    lock: Phase1ZIDStudentTLock


def _canonical_registered_axis(classes: Sequence[str]) -> tuple[str, ...]:
    """Return the legacy runtime's deterministic class axis without changing handles."""
    return tuple(sorted(classes))


def _restore_registered_axis(classes: Sequence[str], scores: np.ndarray) -> np.ndarray:
    """Project legacy canonical score columns back to the sealed registry order."""
    values = np.asarray(scores)
    sealed = tuple(classes)
    canonical = _canonical_registered_axis(sealed)
    if values.ndim != 2 or values.shape[1] != len(canonical):
        raise DSSCStateError("qKNN score/class-axis drift")
    inverse = np.asarray([canonical.index(name) for name in sealed], np.intp)
    return values[:, inverse]


def _canonicalize_registered_axis(classes: Sequence[str], scores: np.ndarray) -> np.ndarray:
    """Map sealed-order score columns to the legacy runtime's canonical axis."""
    values = np.asarray(scores)
    sealed = tuple(classes)
    if values.ndim != 2 or values.shape[1] != len(sealed):
        raise DSSCStateError("qKNN score/class-axis drift")
    source = np.asarray([sealed.index(name) for name in _canonical_registered_axis(sealed)], np.intp)
    return values[:, source]


def build_qknn_state(features: np.ndarray, labels: Sequence[Any] | np.ndarray, registered_classes: Sequence[Any] | np.ndarray,
                     physical_tokens: Sequence[Any] | np.ndarray, *, k_shot: int | None = None,
                     qknn_lock: Phase1ZIDStudentTLock | None = None) -> QKNNState:
    rows = _unit(_rows(features, name="adapted raw-view support z_id")); classes = typed_tokens(registered_classes, name="qKNN classes", unique=True)
    values = typed_tokens(labels, name="qKNN labels"); tokens = typed_tokens(physical_tokens, name="support physical tokens", unique=True)
    if len(values) != len(rows) or len(tokens) != len(rows) or any(x not in classes for x in values):
        raise DSSCStateError("qKNN support layout drift")
    inferred=tuple(values.count(c) for c in classes)
    if len(set(inferred)) != 1 or inferred[0] not in (1,5,10): raise DSSCStateError("qKNN requires exact balanced K={1,5,10}")
    active=inferred[0] if k_shot is None else k_shot
    if active != inferred[0]: raise DSSCStateError("qKNN K lock/support drift")
    if type(qknn_lock) is not Phase1ZIDStudentTLock or qknn_lock.active_k != active:
        raise DSSCStateError("formal qKNN requires the exact active-K lock from DSSC method lock")
    lock=qknn_lock
    canonical_classes = _canonical_registered_axis(classes)
    try: branch=_build_svrn_branch(rows,list(values),list(canonical_classes),list(tokens),qknn_config=lock,branch="raw")
    except Exception as exc: raise DSSCStateError(f"typed real qKNN/BCRR build failed: {exc}") from exc
    return QKNNState(classes,tokens,branch,lock)


def qknn_logits(state: QKNNState, features: np.ndarray) -> np.ndarray:
    rows=_rows(features,name="qKNN query features")
    state.branch_state.__post_init__()
    if state.branch_state.branch != "raw" or float(state.branch_state.eta) != 0.0:
        raise DSSCStateError("DSSC qKNN only permits the identity raw branch")
    bank, metric = deserialize_typed_zid_runtime_state(state.branch_state.qknn_wire)
    return _restore_registered_axis(state.classes, score_zid_student_t_logits(bank, rows, metric=metric))


def qknn_neighbor_receipt(state: QKNNState, features: np.ndarray) -> Mapping[str, Any]:
    return _svrn_neighbors(state.branch_state,_rows(features,name="neighbor query features"))


def normalize_scores(value: np.ndarray) -> np.ndarray:
    scores = np.asarray(value, np.float64)
    if scores.ndim != 2 or scores.shape[1] < 2 or not np.isfinite(scores).all(): raise DSSCStateError("score geometry drift")
    centered = scores - scores.mean(1, keepdims=True); norm = np.linalg.norm(centered, axis=1, keepdims=True)
    if np.any(norm <= 1.0e-12): raise DSSCStateError("score normalization degeneracy")
    return np.asarray(math.sqrt(scores.shape[1]) * centered / norm, np.float32)


@dataclass(frozen=True)
class BCRRState:
    branch_state: Any
    omega: float
    receipt: Mapping[str, Any]
    classes: tuple[str, ...] | None = None


def _bcrr_registered_axis(state: BCRRState) -> tuple[str, ...]:
    if state.classes is not None:
        return typed_tokens(state.classes, name="BCRR classes", unique=True)
    try:
        bank, _metric = deserialize_typed_zid_runtime_state(state.branch_state.qknn_wire)
        classes = typed_tokens(bank.classes, name="legacy BCRR bank classes", unique=True)
    except Exception as exc:
        raise DSSCStateError("legacy BCRR class-axis recovery failed") from exc
    if classes != _canonical_registered_axis(classes):
        raise DSSCStateError("legacy BCRR bank class-axis is not canonical")
    return classes


def fit_bcrr_support_only(state: QKNNState, features: np.ndarray, *, k_shot: int) -> BCRRState:
    if k_shot != state.lock.active_k: raise DSSCStateError("BCRR lock K drift")
    _rows(features,name="BCRR support z_id")
    receipt=state.branch_state.bcrr_receipt
    return BCRRState(state.branch_state,float(receipt["omega_q"]),receipt,classes=state.classes)


def bcrr_fused_logits(qknn: np.ndarray, query_features: np.ndarray, state: BCRRState) -> np.ndarray:
    q=np.asarray(qknn,np.float32); raw,fused=_svrn_scores(state.branch_state,_rows(query_features,name="BCRR query z_id"))
    classes = _bcrr_registered_axis(state)
    q_canonical = _canonicalize_registered_axis(classes, q)
    if q_canonical.shape!=raw.shape or not np.allclose(q_canonical,raw,rtol=0.0,atol=1.0e-6): raise DSSCStateError("BCRR must fuse the same real qKNN branch")
    return _restore_registered_axis(classes, fused)


def build_five_arm_states(*, raw_support_features: np.ndarray, ng_support_features: np.ndarray,
                          ground_support_features: np.ndarray, support_labels: Sequence[Any] | np.ndarray,
                          registered_classes: Sequence[Any] | np.ndarray, support_physical_ids: Sequence[Any] | np.ndarray,
                          k_shot: int, qknn_lock: Phase1ZIDStudentTLock | None = None) -> Mapping[str, Any]:
    """Build the frozen five r1f arm states solely from one row's support."""
    labels=typed_tokens(support_labels,name="row support labels"); classes=typed_tokens(registered_classes,name="row registry",unique=True); ids=typed_tokens(support_physical_ids,name="row support physical IDs",unique=True)
    if len(labels)!=len(ids): raise DSSCStateError("row support label/physical-ID drift")
    raw=build_qknn_state(raw_support_features,labels,classes,ids,k_shot=k_shot,qknn_lock=qknn_lock)
    ng=build_qknn_state(ng_support_features,labels,classes,ids,k_shot=k_shot,qknn_lock=qknn_lock)
    ground=build_qknn_state(ground_support_features,labels,classes,ids,k_shot=k_shot,qknn_lock=qknn_lock)
    other=fit_bcrr_support_only(raw,raw_support_features,k_shot=k_shot); joint=fit_bcrr_support_only(ground,ground_support_features,k_shot=k_shot)
    return {"M0":raw,"M_DA_NG":ng,"M_DA":ground,"M_OTHER":(raw,other),"M_JOINT":(ground,joint),"query_rows_used_for_fit":0}


def predict_five_arms(states: Mapping[str, Any], *, raw_query_features: np.ndarray, ng_query_features: np.ndarray,
                      ground_query_features: np.ndarray) -> Mapping[str, np.ndarray]:
    """Per-sample all-class prediction logits; query labels and roles are absent."""
    if set(states) != set(ARMS) | {"query_rows_used_for_fit"} or states["query_rows_used_for_fit"] != 0: raise DSSCStateError("five-arm state schema/query-fit drift")
    raw=states["M0"]; ng=states["M_DA_NG"]; ground=states["M_DA"]; other_q,other_b=states["M_OTHER"]; joint_q,joint_b=states["M_JOINT"]
    if other_q is not raw or joint_q is not ground or other_b.branch_state is not raw.branch_state or joint_b.branch_state is not ground.branch_state:
        raise DSSCStateError("OTHER/JOINT must reuse their exact matched qKNN branch")
    raw_rows = _rows(raw_query_features, name="raw five-arm query features")
    ground_rows = _rows(ground_query_features, name="ground five-arm query features")
    q0, other = _svrn_scores(raw.branch_state, raw_rows)
    qng = qknn_logits(ng, ng_query_features)
    qg, joint = _svrn_scores(ground.branch_state, ground_rows)
    return {"M0":_restore_registered_axis(raw.classes,q0),"M_DA_NG":qng,
            "M_DA":_restore_registered_axis(ground.classes,qg),
            "M_OTHER":_restore_registered_axis(raw.classes,other),
            "M_JOINT":_restore_registered_axis(ground.classes,joint)}


def resource_receipt(*, bundle: GroundBundle, qknn: QKNNState, bcrr: BCRRState, adapter: Rank4Adapter | None,
                     train_receipt: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    branch_wire=_serialize_svrn_branch(qknn.branch_state)
    ground_bytes = bundle_wire_bytes(bundle)
    qknn_bytes = len(qknn.branch_state.qknn_wire)
    bcrr_bytes = int(
        qknn.branch_state.bcr_weight_codes_qint8.nbytes
        + qknn.branch_state.bcr_weight_scales_fp16.nbytes
    )
    adapter_bytes = (
        0
        if adapter is None or adapter.coefficient_codes is None
        else int(
            adapter.coefficient_codes.nbytes
            + adapter.coefficient_scale_fp16.nbytes
        )
    )
    wire = ground_bytes + len(branch_wire) + adapter_bytes
    if wire > MAX_WIRE_BYTES: raise DSSCStateError("r1f persistent wire state exceeds 256KiB")
    body = {"schema": SCHEMA + ".resource.v1", "wire_bytes": wire, "wire_limit_bytes": MAX_WIRE_BYTES,
            "state_bytes": {"ground_bundle": ground_bytes, "adapter": adapter_bytes,
                            "qknn": qknn_bytes, "bcrr": bcrr_bytes},
            "adapter_rank": RANK, "adapter_coefficients": 4, "trainable_parameter_count": RANK, "adapter_merged": adapter is None or adapter.merged,
            "query_adapter_extra_mac": 0, "query_rows_used_for_fit": 0, "ground_int8": True, "qknn_int8": True,
            "bcrr_int8": True, "adapter_int8": adapter is None or adapter.coefficient_codes is not None, "optimizer": None if train_receipt is None else train_receipt.get("optimizer"),
            "optimizer_steps": None if train_receipt is None else train_receipt.get("steps"),
            "qknn_bcrr_build_resource": dict(qknn.branch_state.resource),
            "qknn_quantization_audit": dict(qknn.branch_state.quantization_audit["qknn"]),
            "bcrr_quantization_audit": dict(qknn.branch_state.quantization_audit["bcr"]),
            "bcrr_fit_receipt": dict(bcrr.receipt)}
    return {**body, "receipt_sha256": sha256_bytes(_canon(body))}


__all__ = ["ADAPTER_SCALE_GROUPS", "ARMS", "BCRR_MAX_OMEGA", "BCRRState", "BUNDLE_SCHEMA", "CANDIDATE", "DSSCStateError", "GroundBundle", "MIN_ADAPTER_FP16_SCALE",
            "GEOFF_R8_COVERAGE_SHA256", "MAX_WIRE_BYTES", "PHASE1_ARCHIVE_MANIFEST_SHA256", "PHASE1_ARCHIVE_SHA256",
            "PHASE1_CHECKPOINT_SHA256", "PHASE1_PARITY_RECEIPT_SHA256", "QKNNState", "RANK", "SCENES",
            "SEALED_RUNTIME_SHA256", "SGD_LR", "SGD_WEIGHT_DECAY", "SOMPH_PACKAGE_LOCK_SHA256", "Z_DIM", "adapt_support_only",
            "adaptation_steps", "attach_rank4_adapter", "bcrr_fused_logits", "build_five_arm_states", "build_ground_bundle_arrays", "build_qknn_state",
            "bundle_wire_bytes", "canonical_method_lock", "fit_bcrr_support_only", "load_ground_bundle", "qknn_logits", "qknn_neighbor_receipt",
            "predict_five_arms", "qknn_lock_from_method_lock", "resource_receipt", "rms_view", "sha256_file", "typed_tokens",
            "totalize_adapted_zid", "validate_method_lock", "ZIDZeroNormTotalizationReceipt",
            "ZID_ZERO_NORM_TOTALIZATION_REVISION"]
