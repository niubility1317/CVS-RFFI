#!/usr/bin/env python3
"""D25 development support-only screen over sealed LEO_weak enrollment rows.

This runner deliberately reuses the already-audited D19 sealed-support helpers
without changing the historical D19/D20 runner.  Its CLI has no query, truth,
score-table, or scorer input.  FFT96 and RF32 are each extracted exactly once
per physical received-IQ row and shared by the B3 diagnostic and D25 routes.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_d19_support_only_ciaf as legacy  # noqa: E402
from cvsrffi.stage2_diag_cosine_exploration import (  # noqa: E402
    rf_statistics,
    spectral_logmag_sketch,
)
from cvsrffi.stage2_multimodal_concat_fusion import (  # noqa: E402
    SCORE_COSINE,
    SCORE_RADIUS,
    MultimodalConcatConfig,
    append_new_classes_concat,
    build_concat288,
    fit_old_concat,
    predict_one as predict_one_concat,
    score_one as score_one_concat,
)
from cvsrffi.stage2_multimodal_diag_floor_adapter import (  # noqa: E402
    D25C3Config,
    D25C3LossWeights,
    D25C3State,
    append_stage2c_new_suffix,
    fit_stage2b_diag_floor,
    predict_one as predict_one_c3,
    score_one as score_one_c3,
)
from cvsrffi.stage2_multimodal_compact_diag import (  # noqa: E402
    D26CompactDiagConfig,
    D26CompactDiagState,
    append_stage2c_new_suffix as append_stage2c_d26,
    fit_stage2b_compact_diag,
    predict_all_registered as predict_all_d26,
    score_all_registered as score_all_d26,
)
from cvsrffi.stage2_support_evidence_gate import (  # noqa: E402
    SupportEvidenceGateConfig,
    apply_support_evidence_gate,
    fit_support_evidence_gate,
    predict_with_support_evidence_gate,
)
from cvsrffi.stage2_classwise_safe_release import (  # noqa: E402
    ClasswiseSafeReleaseConfig,
    apply_classwise_safe_release,
    fit_classwise_safe_release,
    predict_with_classwise_safe_release,
)
from cvsrffi.stage2_dali import (  # noqa: E402
    DaliConfig,
    _component_maximin_medoid,
    _transient_domain_anchors,
    fit_old_dali,
    register_new_dali,
    rerank_old_scores_dali,
)
from cvsrffi.stage2_max_envelope_calibration import (  # noqa: E402
    MaxEnvelopeCalibrationConfig,
    apply_max_envelope_calibration,
    audit_envelope_confusions,
    fit_max_envelope_calibration,
)
from cvsrffi.stage2_all_registered_new_suffix import (  # noqa: E402
    D31Stage2CConfig,
    D31_NEW_CVAR_FLOOR,
    D31_OLD_MARGIN_PROTECTION,
    D31_PLAIN_BALANCED_CE,
    append_stage2c_all_registered_new_suffix,
    predict_all_registered as predict_all_d31,
    score_all_registered as score_all_d31,
)
from cvsrffi.stage2_inloop_safe_cap_suffix import (  # noqa: E402
    D32Stage2CConfig,
    D32_BIAS_RECOVERY_CAP,
    D32_GROUP_BALANCED_CAP,
    D32_NEW_CVAR_CAP,
    append_stage2c_inloop_safe_cap_suffix,
    score_all_registered as score_all_d32,
)
from cvsrffi.stage2_d33_spherical_registration import (  # noqa: E402
    D33SphericalRegistrationConfig,
    fit_d33_spherical_registration,
    score_d33_spherical_registration,
)
from cvsrffi.stage2_b3_fisher_closed_form import (  # noqa: E402
    B3FisherClosedFormConfig,
    fit_b3_fisher_closed_form,
    score_b3_fisher_closed_form,
)
from cvsrffi.stage2_d34_collision_local_registration import (  # noqa: E402
    D34CollisionLocalConfig,
    fit_d34_collision_local_registration,
    score_d34_collision_local_registration,
)
from cvsrffi.stage2_d35_dense_safe_registration import (  # noqa: E402
    D35DenseSafeConfig,
    fit_d35_dense_safe_registration,
    score_d35_dense_safe_registration,
)
from cvsrffi.stage2_d36_compiled_joint_int8 import (  # noqa: E402
    D36CompiledJointConfig,
    _fit_irls as _d36_fit_irls,
    fit_d36_compiled_joint_int8,
    margin_features_d36_compiled_joint_int8,
    score_d36_compiled_joint_int8,
    with_oof_calibration_d36_compiled_joint_int8,
)
from cvsrffi.stage2_d37_b3_preserving_int8 import (  # noqa: E402
    D37B3PreservingInt8Config,
    D37B3PreservingInt8Error,
    OOF_SOURCE as D37_OOF_SOURCE,
    base_score_d37_b3_preserving_int8,
    fit_d37_b3_preserving_int8,
    fit_oof_feasible_offset_d37,
    old_prefix_bitwise_unchanged_d37,
    score_d37_b3_preserving_int8,
)
from cvsrffi.stage2_d38_strong_b3_quantized import (  # noqa: E402
    D38StrongB3Config,
    fit_d38_strong_b3_quantized,
    old_prefix_bitwise_unchanged_d38,
    pairwise_support_diagnostics_d38,
    score_d38_strong_b3,
)
from cvsrffi.stage2_d39_angular_radius import (  # noqa: E402
    R0_FLOOR as D39_R0_FLOOR,
    RADIUS_EPSILON as D39_RADIUS_EPSILON,
    RADIUS_NU as D39_RADIUS_NU,
    D39AngularRadiusConfig,
    fit_d39_angular_radius,
    old_prefix_bitwise_unchanged_d39,
    pairwise_support_diagnostics_d39,
    score_d39_angular_radius,
)
from cvsrffi.stage2_d40_hnbr import (  # noqa: E402
    D40HNBRConfig,
    fit_d40_hnbr,
    old_prefix_bitwise_unchanged_d40,
    pairwise_support_diagnostics_d40,
    score_d40_hnbr,
)


MODE = legacy.MODE
SUPPORT_QUERY_DISJOINTNESS_STATUS = legacy.SUPPORT_QUERY_DISJOINTNESS_STATUS
HELD_RANKS = legacy.HELD_RANKS
IDENTITY_CANDIDATE = legacy.IDENTITY_CANDIDATE
DIAG_CANDIDATE = legacy.DIAG_CANDIDATE
D25_C0 = "D25-C0-DIM-CONCAT"
D25_C1 = "D25-C1-UF-GROUNDZ"
D25_C2 = "D25-C2-BLOCK-RADIUS"
D25_CANDIDATES = (D25_C0, D25_C1, D25_C2)
CANDIDATE_SET_D25_V4 = "d25_v4"
CANDIDATE_SET_C3_V1 = "c3_v1"
CANDIDATE_SET_D26_V1 = "d26_v1"
CANDIDATE_SET_D26_V2 = "d26_v2_strictbias"
CANDIDATE_SET_D27_V1 = "d27_v1_perclassbias"
CANDIDATE_SET_D28_V1 = "d28_v1_evidence_gate"
CANDIDATE_SET_D29_V1 = "d29_v1_pcsr"
CANDIDATE_SET_D30_V1 = "d30_v1"
CANDIDATE_SET_D31_V1 = "d31_v1"
CANDIDATE_SET_D32_V1 = "d32_v1"
CANDIDATE_SET_D33_V1 = "d33_v1"
CANDIDATE_SET_D34_V1 = "d34_v1"
CANDIDATE_SET_D35_V1 = "d35_v1"
CANDIDATE_SET_D36_V1 = "d36_v1"
CANDIDATE_SET_D37_V1 = "d37_v1"
CANDIDATE_SET_D38_V1 = "d38_v1"
CANDIDATE_SET_D39_V1 = "d39_v1"
CANDIDATE_SET_D40_V1 = "d40_v1"
D38_DEVELOPMENT_RECEIVER = "20-1"
D38_DEVELOPMENT_SEED = 713101
D38_DEVELOPMENT_NEW_CLASS_COUNT = 5
D39_DEVELOPMENT_RECEIVER = "20-1"
D39_DEVELOPMENT_SEED = 713101
D39_DEVELOPMENT_NEW_CLASS_COUNT = 5
D40_DEVELOPMENT_RECEIVER = "20-1"
D40_DEVELOPMENT_SEED = 713101
D40_DEVELOPMENT_NEW_CLASS_COUNT = 5
C3_A = "D25-C3A-DIAG-CE-CLOSEDREG"
C3_B = "D25-C3B-DIAG-CE-NEWFIT"
C3_C = "D25-C3C-DIAG-STRONGFLOOR-NEWFIT"
C3_CANDIDATES = (C3_A, C3_B, C3_C)
D26_A = "D26-A-COMPACT-DIAG-CLOSEDREG"
D26_B = "D26-B-COMPACT-DIAG-NEWFIT10"
D26_C = "D26-C-COMPACT-DIAG-NEWFIT15"
D26_CANDIDATES = (D26_A, D26_B, D26_C)
D27_A = "D27-A-PERCLASS-BIAS-CLOSEDREG"
D27_B = "D27-B-PERCLASS-BIAS-NEWFIT10"
D27_C = "D27-C-PERCLASS-BIAS-NEWFIT15"
D27_CANDIDATES = (D27_A, D27_B, D27_C)
D28_A = "D28-A-D27B-NOGATE"
D28_B = "D28-B-E5-RIDGE-D1"
D28_C = "D28-C-E5-RIDGE-D2"
D28_CANDIDATES = (D28_A, D28_B, D28_C)
D29_A = "D29-A-PCSR-RHO25-OVERALL"
D29_B = "D29-B-PCSR-RHO50-BALANCE"
D29_C = "D29-C-PCSR-RHO100-FLOOR"
D29_CANDIDATES = (D29_A, D29_B, D29_C)
D30_A = "D30-A-B3-DALI025-ENVELOPE-OVERALL"
D30_B = "D30-B-B3-DALI050-ENVELOPE-BALANCE"
D30_C = "D30-C-B3-DALI100-ENVELOPE-FLOOR"
D30_CANDIDATES = (D30_A, D30_B, D30_C)
D31_A = D31_PLAIN_BALANCED_CE
D31_B = D31_NEW_CVAR_FLOOR
D31_C = D31_OLD_MARGIN_PROTECTION
D31_CANDIDATES = (D31_A, D31_B, D31_C)
D32_A = D32_GROUP_BALANCED_CAP
D32_B = D32_NEW_CVAR_CAP
D32_C = D32_BIAS_RECOVERY_CAP
D32_CANDIDATES = (D32_A, D32_B, D32_C)
D33_A = "D33-A-ADAM15-SPHERICAL-OVERALL"
D33_B = "D33-B-ADAM15-SPHERICAL-BALANCED"
D33_C = "D33-C-ADAM15-SPHERICAL-FLOOR"
D33_B3_FAST = "D33-B3-FAST-FISHER-SPHERICAL-BALANCED"
D33_CANDIDATES = (D33_A, D33_B, D33_C, D33_B3_FAST)
D34_A = "D34-A-COLLISION-LOCAL-TOP1"
D34_B = "D34-B-COLLISION-LOCAL-TOP2-MEDOID"
D34_C = "D34-C-COLLISION-LOCAL-ADAPTIVE-FLOOR"
D34_CANDIDATES = (D34_A, D34_B, D34_C)
D35_A = "D35-A-DENSE-SAFE-MEAN"
D35_B = "D35-B-DENSE-SAFE-DUAL"
D35_C = "D35-C-DENSE-SAFE-DUAL-FLOOR"
D35_CANDIDATES = (D35_A, D35_B, D35_C)
D36_A = "D36-A-COMPILED-JOINT-INT8"
D36_B = "D36-B-COMPILED-JOINT-INT8-GROUND"
D36_C = "D36-C-COMPILED-JOINT-INT8-GROUND-FLOOR"
D36_CANDIDATES = (D36_A, D36_B, D36_C)
D37_A = "D37-A-B3-PRESERVING-INT8-M0"
D37_B = "D37-B-B3-PRESERVING-INT8-M005"
D37_C = "D37-C-B3-PRESERVING-INT8-M010"
D37_CANDIDATES = (D37_A, D37_B, D37_C)
D38_PROTONET_CDA = "D38-PROTOnet-CDA-ZID160"
D38_A_INT8 = "D38-A-STRONG-B3-RESIDUAL-INT8"
D38_B_INT8 = "D38-B-STRONG-B3-RESIDUAL-INT8"
D38_B_FP32 = "D38-B-STRONG-B3-FP32-MATCHED"
D38_METHOD_CANDIDATES = (D38_A_INT8, D38_B_INT8, D38_B_FP32)
D38_CANDIDATES = (
    IDENTITY_CANDIDATE,
    D38_PROTONET_CDA,
    DIAG_CANDIDATE,
) + D38_METHOD_CANDIDATES
D39_PROTONET_CDA = "D39-PROTOnet-CDA-ZID160"
D39_D38_B_INT8 = "D39-D38-B-RESIDUAL-INT8-NEGATIVE"
D39_INT8 = "D39-ANGULAR-RADIUS-INT8"
D39_FP32 = "D39-ANGULAR-RADIUS-FP32-MATCHED"
D39_METHOD_CANDIDATES = (D39_D38_B_INT8, D39_INT8, D39_FP32)
D39_CANDIDATES = (
    IDENTITY_CANDIDATE,
    D39_PROTONET_CDA,
    DIAG_CANDIDATE,
) + D39_METHOD_CANDIDATES
D40_PROTONET_CDA = "D40-PROTOnet-CDA-ZID160"
D40_D38_B_INT8 = "D40-D38-B-RESIDUAL-INT8-NEGATIVE"
D40_INT8 = "D40-HNBR-INT8"
D40_FP32 = "D40-HNBR-FP32-MATCHED"
D40_METHOD_CANDIDATES = (D40_D38_B_INT8, D40_INT8, D40_FP32)
D40_CANDIDATES = (
    IDENTITY_CANDIDATE,
    D40_PROTONET_CDA,
    DIAG_CANDIDATE,
) + D40_METHOD_CANDIDATES
D40_HNBR_TEMPERATURE = 18.0
D40_NEW_NEW_CONFUSION_CAP = 32
CORE_COMMIT = "f349850dbd94841ae2ef8105ac76bd7a9912c128"
D26_CORE_GIT_COMMIT = "67b9d2275782339e0ac07800652b997adbcca534"


def _positive_route_candidates(candidate_set: str) -> tuple[str, ...]:
    """Single source of truth for a candidate set's promotable route IDs."""

    if candidate_set == CANDIDATE_SET_C3_V1:
        return C3_CANDIDATES
    if candidate_set == CANDIDATE_SET_D27_V1:
        return D27_CANDIDATES
    if candidate_set == CANDIDATE_SET_D28_V1:
        return D28_CANDIDATES
    if candidate_set == CANDIDATE_SET_D29_V1:
        return D29_CANDIDATES
    if candidate_set == CANDIDATE_SET_D30_V1:
        return D30_CANDIDATES
    if candidate_set == CANDIDATE_SET_D31_V1:
        return D31_CANDIDATES
    if candidate_set == CANDIDATE_SET_D32_V1:
        return D32_CANDIDATES
    if candidate_set == CANDIDATE_SET_D33_V1:
        return D33_CANDIDATES
    if candidate_set == CANDIDATE_SET_D34_V1:
        return D34_CANDIDATES
    if candidate_set == CANDIDATE_SET_D35_V1:
        return D35_CANDIDATES
    if candidate_set == CANDIDATE_SET_D36_V1:
        return D36_CANDIDATES
    if candidate_set == CANDIDATE_SET_D37_V1:
        return D37_CANDIDATES
    if candidate_set == CANDIDATE_SET_D38_V1:
        return (D38_B_INT8,)
    if candidate_set == CANDIDATE_SET_D39_V1:
        return (D39_INT8,)
    if candidate_set == CANDIDATE_SET_D40_V1:
        return (D40_INT8,)
    if candidate_set in (CANDIDATE_SET_D26_V1, CANDIDATE_SET_D26_V2):
        return D26_CANDIDATES
    return D25_CANDIDATES


def _artifact_schema(candidate_set: str, artifact: str) -> str:
    """Return the closed artifact schema namespace for the active screen."""

    if candidate_set == CANDIDATE_SET_D40_V1:
        return f"cvs.phase2.d40.{artifact}.v1"
    if candidate_set == CANDIDATE_SET_D39_V1:
        return f"cvs.phase2.d39.{artifact}.v1"
    if candidate_set == CANDIDATE_SET_D38_V1:
        return f"cvs.phase2.d38.{artifact}.v1"
    return f"cvs.phase2.d25.{artifact}.v1"


def _full_state_refit_required(
    candidate_set: str, candidate_id: str, selected_id: str
) -> bool:
    """D38-D40 refit only the globally selected route after outer selection."""

    if candidate_set in (
        CANDIDATE_SET_D38_V1,
        CANDIDATE_SET_D39_V1,
        CANDIDATE_SET_D40_V1,
    ):
        return candidate_id == selected_id
    return True


class D25RunnerError(ValueError):
    """Raised when the D25 support-only screen must fail closed."""


@dataclass(frozen=True)
class D28CandidateConfig:
    """Method-locked D27-B head plus one optional D28 row-local gate."""

    base: D26CompactDiagConfig
    gate: SupportEvidenceGateConfig

    def __post_init__(self) -> None:
        self.base.validate()
        self.gate.validate()
        if (
            int(self.base.stage2b_steps) != 15
            or int(self.base.stage2c_steps) != 10
            or self.base.bias_guard_mode
            != "per_new_class_pre_registration_old_only"
        ):
            raise D25RunnerError("D28 must remain attached to locked D27-B")


@dataclass(frozen=True)
class D29CandidateConfig:
    """Method-locked D27-B head plus D29 per-class safe release."""

    base: D26CompactDiagConfig
    release: ClasswiseSafeReleaseConfig

    def __post_init__(self) -> None:
        self.base.validate()
        self.release.validate()
        if (
            int(self.base.stage2b_steps) != 15
            or int(self.base.stage2c_steps) != 10
            or self.base.bias_guard_mode
            != "per_new_class_pre_registration_old_only"
        ):
            raise D25RunnerError("D29 must remain attached to locked D27-B")


@dataclass(frozen=True)
class D30CandidateConfig:
    """B3 auxiliary-dominant geometry plus bounded DALI/envelope reranking."""

    base: D26CompactDiagConfig
    dali: DaliConfig
    envelope_objective: str

    def __post_init__(self) -> None:
        self.base.validate()
        self.dali.validate()
        if (
            int(self.base.stage2b_steps) != 15
            or int(self.base.stage2c_steps) != 10
            or self.base.bias_guard_mode
            != "per_new_class_pre_registration_old_only"
            or float(self.dali.direct_weight) != 0.0
            or self.envelope_objective
            not in ("overall_first", "balance_first", "floor_first")
        ):
            raise D25RunnerError("D30 method lock drift")


@dataclass(frozen=True)
class D31CandidateConfig:
    """B3 geometry, frozen 15-step Stage2-B, and D31 all-support suffix."""

    base: D26CompactDiagConfig
    stage2c: D31Stage2CConfig
    dali: DaliConfig = DaliConfig(ground_weight=0.05, direct_weight=0.0)

    def __post_init__(self) -> None:
        self.base.validate()
        self.dali.validate()
        if (
            int(self.base.stage2b_steps) != 15
            or int(self.base.stage2c_steps) != 0
            or float(self.dali.direct_weight) != 0.0
            or self.stage2c.method_id not in D31_CANDIDATES
        ):
            raise D25RunnerError("D31 method lock drift")


@dataclass(frozen=True)
class D32CandidateConfig:
    """B3 geometry, frozen 15-step Stage2-B, and in-loop safe-cap suffix."""

    base: D26CompactDiagConfig
    stage2c: D32Stage2CConfig
    dali: DaliConfig = DaliConfig(ground_weight=0.05, direct_weight=0.0)

    def __post_init__(self) -> None:
        self.base.validate()
        self.dali.validate()
        total_steps = int(self.base.stage2b_steps) + int(self.stage2c.optimizer_steps)
        if (
            int(self.base.stage2b_steps) != 15
            or int(self.base.stage2c_steps) != 0
            or total_steps > 30
            or float(self.dali.direct_weight) != 0.0
            or self.stage2c.method_id not in D32_CANDIDATES
        ):
            raise D25RunnerError("D32 method lock drift")


@dataclass(frozen=True)
class D33CandidateConfig:
    """Locked old-domain solver plus symmetric spherical registration."""

    old_solver: str
    registration: D33SphericalRegistrationConfig
    base: D26CompactDiagConfig | None = None
    fisher: B3FisherClosedFormConfig | None = None

    def __post_init__(self) -> None:
        if self.old_solver == "adam15_compact_diag":
            if self.base is None or self.fisher is not None:
                raise D25RunnerError("D33 Adam15 solver lock drift")
            self.base.validate()
            if int(self.base.stage2b_steps) != 15 or int(self.base.stage2c_steps) != 0:
                raise D25RunnerError("D33 Adam15 step lock drift")
        elif self.old_solver == "b3_fisher_closed_form":
            if self.fisher is None or self.base is not None:
                raise D25RunnerError("D33 Fisher solver lock drift")
        else:
            raise D25RunnerError("unknown D33 old solver lock")


@dataclass(frozen=True)
class D34CandidateConfig:
    """FAST Fisher old head plus frozen collision-local registration."""

    fisher: B3FisherClosedFormConfig
    registration: D34CollisionLocalConfig

    def __post_init__(self) -> None:
        if str(self.registration.arm) not in ("A", "B", "C"):
            raise D25RunnerError("D34 arm lock drift")


@dataclass(frozen=True)
class D35CandidateConfig:
    """FAST Fisher old head plus globally visible dense-safe registration."""

    fisher: B3FisherClosedFormConfig
    registration: D35DenseSafeConfig

    def __post_init__(self) -> None:
        if str(self.registration.arm) not in ("A", "B", "C"):
            raise D25RunnerError("D35 arm lock drift")


@dataclass(frozen=True)
class D36CandidateConfig:
    """FAST initialization plus compiled all-int8 joint old/new head."""

    fisher: B3FisherClosedFormConfig
    compiled: D36CompiledJointConfig

    def __post_init__(self) -> None:
        if str(self.compiled.arm) not in ("A", "B", "C"):
            raise D25RunnerError("D36 arm lock drift")


@dataclass(frozen=True)
class D37CandidateConfig:
    """FAST Fisher B3 head plus append-only residual-int8 registration."""

    fisher: B3FisherClosedFormConfig
    compiled: D37B3PreservingInt8Config

    def __post_init__(self) -> None:
        if str(self.compiled.arm) not in ("A", "B", "C"):
            raise D25RunnerError("D37 arm lock drift")


@dataclass(frozen=True)
class D38ProtoNetCDAConfig:
    """Independent matched ProtoNet-CDA row in ADV3B02 identity geometry."""

    feature_geometry: str = "adv3b02_z_id160_support_mean_nearest_prototype"


@dataclass(frozen=True)
class D38CandidateConfig:
    """Method-locked D38 arm and final deployment precision."""

    core: D38StrongB3Config
    deploy_precision: str = "int8"

    def __post_init__(self) -> None:
        if self.deploy_precision not in ("int8", "fp32"):
            raise D25RunnerError("D38 deployment precision drift")
        if self.deploy_precision == "fp32" and self.core.arm != "B":
            raise D25RunnerError("D38 FP32 ablation must use arm B")


@dataclass(frozen=True)
class D39CandidateConfig:
    """Locked zero-step angular-radius wrapper and deployment precision."""

    core: D39AngularRadiusConfig
    deploy_precision: str = "int8"

    def __post_init__(self) -> None:
        if self.deploy_precision not in ("int8", "fp32"):
            raise D25RunnerError("D39 deployment precision drift")


@dataclass(frozen=True)
class D40CandidateConfig:
    """Locked zero-step synchronous HNBR wrapper and deployment precision."""

    core: D40HNBRConfig
    deploy_precision: str = "int8"

    def __post_init__(self) -> None:
        if self.deploy_precision not in ("int8", "fp32"):
            raise D25RunnerError("D40 deployment precision drift")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _row_hashes(value: np.ndarray) -> list[str]:
    rows = np.asarray(value, dtype=np.float32)
    return [
        hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
        for row in rows
    ]


def preregistered_candidates(
    candidate_set: str = CANDIDATE_SET_D25_V4,
) -> dict[str, object]:
    """Return the candidate set fixed before any support materialization."""

    controls = legacy.preregistered_candidates()
    historical = {
        IDENTITY_CANDIDATE: controls[IDENTITY_CANDIDATE],
        DIAG_CANDIDATE: controls[DIAG_CANDIDATE],
        D25_C0: MultimodalConcatConfig(
            score_mode=SCORE_COSINE,
            use_ground_identity_fusion=False,
        ),
        D25_C1: MultimodalConcatConfig(
            score_mode=SCORE_COSINE,
            use_ground_identity_fusion=True,
        ),
        D25_C2: MultimodalConcatConfig(
            score_mode=SCORE_RADIUS,
            use_ground_identity_fusion=True,
        ),
    }
    if candidate_set == CANDIDATE_SET_D25_V4:
        return historical
    if candidate_set == CANDIDATE_SET_D33_V1:
        adam15 = D26CompactDiagConfig(
            stage2b_steps=15,
            stage2c_steps=0,
            bias_guard_mode="per_new_class_pre_registration_old_only",
        )
        return {
            IDENTITY_CANDIDATE: controls[IDENTITY_CANDIDATE],
            DIAG_CANDIDATE: controls[DIAG_CANDIDATE],
            D25_C0: historical[D25_C0],
            D33_A: D33CandidateConfig(
                old_solver="adam15_compact_diag",
                base=adam15,
                registration=D33SphericalRegistrationConfig(
                    selection_policy="A_overall_first"
                ),
            ),
            D33_B: D33CandidateConfig(
                old_solver="adam15_compact_diag",
                base=adam15,
                registration=D33SphericalRegistrationConfig(
                    selection_policy="B_balanced"
                ),
            ),
            D33_C: D33CandidateConfig(
                old_solver="adam15_compact_diag",
                base=adam15,
                registration=D33SphericalRegistrationConfig(
                    selection_policy="C_floor_first"
                ),
            ),
            D33_B3_FAST: D33CandidateConfig(
                old_solver="b3_fisher_closed_form",
                fisher=B3FisherClosedFormConfig(),
                registration=D33SphericalRegistrationConfig(
                    selection_policy="B_balanced"
                ),
            ),
        }
    if candidate_set == CANDIDATE_SET_D36_V1:
        fisher = B3FisherClosedFormConfig()
        return {
            IDENTITY_CANDIDATE: controls[IDENTITY_CANDIDATE],
            D25_C0: historical[D25_C0],
            DIAG_CANDIDATE: controls[DIAG_CANDIDATE],
            D33_B3_FAST: D33CandidateConfig(
                old_solver="b3_fisher_closed_form",
                fisher=fisher,
                registration=D33SphericalRegistrationConfig(
                    selection_policy="B_balanced"
                ),
            ),
            D36_A: D36CandidateConfig(
                fisher=fisher,
                compiled=D36CompiledJointConfig(arm="A"),
            ),
            D36_B: D36CandidateConfig(
                fisher=fisher,
                compiled=D36CompiledJointConfig(arm="B"),
            ),
            D36_C: D36CandidateConfig(
                fisher=fisher,
                compiled=D36CompiledJointConfig(arm="C"),
            ),
        }
    if candidate_set == CANDIDATE_SET_D38_V1:
        return {
            IDENTITY_CANDIDATE: controls[IDENTITY_CANDIDATE],
            D38_PROTONET_CDA: D38ProtoNetCDAConfig(),
            DIAG_CANDIDATE: controls[DIAG_CANDIDATE],
            D38_A_INT8: D38CandidateConfig(
                core=D38StrongB3Config(arm="A"), deploy_precision="int8"
            ),
            D38_B_INT8: D38CandidateConfig(
                core=D38StrongB3Config(arm="B"), deploy_precision="int8"
            ),
            D38_B_FP32: D38CandidateConfig(
                core=D38StrongB3Config(arm="B"), deploy_precision="fp32"
            ),
        }
    if candidate_set == CANDIDATE_SET_D39_V1:
        return {
            IDENTITY_CANDIDATE: controls[IDENTITY_CANDIDATE],
            D39_PROTONET_CDA: D38ProtoNetCDAConfig(),
            DIAG_CANDIDATE: controls[DIAG_CANDIDATE],
            D39_D38_B_INT8: D38CandidateConfig(
                core=D38StrongB3Config(arm="B"), deploy_precision="int8"
            ),
            D39_INT8: D39CandidateConfig(
                core=D39AngularRadiusConfig(), deploy_precision="int8"
            ),
            D39_FP32: D39CandidateConfig(
                core=D39AngularRadiusConfig(), deploy_precision="fp32"
            ),
        }
    if candidate_set == CANDIDATE_SET_D40_V1:
        return {
            IDENTITY_CANDIDATE: controls[IDENTITY_CANDIDATE],
            D40_PROTONET_CDA: D38ProtoNetCDAConfig(),
            DIAG_CANDIDATE: controls[DIAG_CANDIDATE],
            D40_D38_B_INT8: D38CandidateConfig(
                core=D38StrongB3Config(arm="B"), deploy_precision="int8"
            ),
            D40_INT8: D40CandidateConfig(
                core=D40HNBRConfig(), deploy_precision="int8"
            ),
            D40_FP32: D40CandidateConfig(
                core=D40HNBRConfig(), deploy_precision="fp32"
            ),
        }
    if candidate_set == CANDIDATE_SET_D37_V1:
        fisher = B3FisherClosedFormConfig()
        return {
            IDENTITY_CANDIDATE: controls[IDENTITY_CANDIDATE],
            D25_C0: historical[D25_C0],
            DIAG_CANDIDATE: controls[DIAG_CANDIDATE],
            D33_B3_FAST: D33CandidateConfig(
                old_solver="b3_fisher_closed_form",
                fisher=fisher,
                registration=D33SphericalRegistrationConfig(
                    selection_policy="B_balanced"
                ),
            ),
            D37_A: D37CandidateConfig(
                fisher=fisher,
                compiled=D37B3PreservingInt8Config(arm="A"),
            ),
            D37_B: D37CandidateConfig(
                fisher=fisher,
                compiled=D37B3PreservingInt8Config(arm="B"),
            ),
            D37_C: D37CandidateConfig(
                fisher=fisher,
                compiled=D37B3PreservingInt8Config(arm="C"),
            ),
        }
    if candidate_set == CANDIDATE_SET_D34_V1:
        fisher = B3FisherClosedFormConfig()
        return {
            IDENTITY_CANDIDATE: controls[IDENTITY_CANDIDATE],
            D25_C0: historical[D25_C0],
            DIAG_CANDIDATE: controls[DIAG_CANDIDATE],
            D33_B3_FAST: D33CandidateConfig(
                old_solver="b3_fisher_closed_form",
                fisher=fisher,
                registration=D33SphericalRegistrationConfig(
                    selection_policy="B_balanced"
                ),
            ),
            D34_A: D34CandidateConfig(
                fisher=fisher,
                registration=D34CollisionLocalConfig(arm="A"),
            ),
            D34_B: D34CandidateConfig(
                fisher=fisher,
                registration=D34CollisionLocalConfig(arm="B"),
            ),
            D34_C: D34CandidateConfig(
                fisher=fisher,
                registration=D34CollisionLocalConfig(arm="C"),
            ),
        }
    if candidate_set == CANDIDATE_SET_D35_V1:
        fisher = B3FisherClosedFormConfig()
        return {
            IDENTITY_CANDIDATE: controls[IDENTITY_CANDIDATE],
            D25_C0: historical[D25_C0],
            DIAG_CANDIDATE: controls[DIAG_CANDIDATE],
            D33_B3_FAST: D33CandidateConfig(
                old_solver="b3_fisher_closed_form",
                fisher=fisher,
                registration=D33SphericalRegistrationConfig(
                    selection_policy="B_balanced"
                ),
            ),
            D35_A: D35CandidateConfig(
                fisher=fisher,
                registration=D35DenseSafeConfig(arm="A"),
            ),
            D35_B: D35CandidateConfig(
                fisher=fisher,
                registration=D35DenseSafeConfig(arm="B"),
            ),
            D35_C: D35CandidateConfig(
                fisher=fisher,
                registration=D35DenseSafeConfig(arm="C"),
            ),
        }
    if candidate_set == CANDIDATE_SET_D27_V1:
        return {
            IDENTITY_CANDIDATE: controls[IDENTITY_CANDIDATE],
            DIAG_CANDIDATE: controls[DIAG_CANDIDATE],
            D25_C0: historical[D25_C0],
            D27_A: D26CompactDiagConfig(
                stage2b_steps=15,
                stage2c_steps=0,
                bias_guard_mode="per_new_class_pre_registration_old_only",
            ),
            D27_B: D26CompactDiagConfig(
                stage2b_steps=15,
                stage2c_steps=10,
                bias_guard_mode="per_new_class_pre_registration_old_only",
            ),
            D27_C: D26CompactDiagConfig(
                stage2b_steps=15,
                stage2c_steps=15,
                bias_guard_mode="per_new_class_pre_registration_old_only",
            ),
        }
    if candidate_set == CANDIDATE_SET_D32_V1:
        base = D26CompactDiagConfig(
            stage2b_steps=15,
            stage2c_steps=0,
            bias_guard_mode="per_new_class_pre_registration_old_only",
        )
        return {
            IDENTITY_CANDIDATE: controls[IDENTITY_CANDIDATE],
            DIAG_CANDIDATE: controls[DIAG_CANDIDATE],
            D25_C0: historical[D25_C0],
            D32_A: D32CandidateConfig(
                base=base, stage2c=D32Stage2CConfig(method_id=D32_A)
            ),
            D32_B: D32CandidateConfig(
                base=base, stage2c=D32Stage2CConfig(method_id=D32_B)
            ),
            D32_C: D32CandidateConfig(
                base=base, stage2c=D32Stage2CConfig(method_id=D32_C)
            ),
        }
    if candidate_set == CANDIDATE_SET_D28_V1:
        d27b = D26CompactDiagConfig(
            stage2b_steps=15,
            stage2c_steps=10,
            bias_guard_mode="per_new_class_pre_registration_old_only",
        )
        return {
            IDENTITY_CANDIDATE: controls[IDENTITY_CANDIDATE],
            DIAG_CANDIDATE: controls[DIAG_CANDIDATE],
            D25_C0: historical[D25_C0],
            D28_A: d27b,
            D28_B: D28CandidateConfig(
                base=d27b,
                gate=SupportEvidenceGateConfig(alpha=1.0, delta=1.0),
            ),
            D28_C: D28CandidateConfig(
                base=d27b,
                gate=SupportEvidenceGateConfig(alpha=1.0, delta=2.0),
            ),
        }
    if candidate_set == CANDIDATE_SET_D29_V1:
        d27b = D26CompactDiagConfig(
            stage2b_steps=15,
            stage2c_steps=10,
            bias_guard_mode="per_new_class_pre_registration_old_only",
        )
        return {
            IDENTITY_CANDIDATE: controls[IDENTITY_CANDIDATE],
            DIAG_CANDIDATE: controls[DIAG_CANDIDATE],
            D25_C0: historical[D25_C0],
            D29_A: D29CandidateConfig(
                base=d27b,
                release=ClasswiseSafeReleaseConfig(
                    safety_budget=0.25,
                    objective="overall_first",
                ),
            ),
            D29_B: D29CandidateConfig(
                base=d27b,
                release=ClasswiseSafeReleaseConfig(
                    safety_budget=0.50,
                    objective="balance_first",
                ),
            ),
            D29_C: D29CandidateConfig(
                base=d27b,
                release=ClasswiseSafeReleaseConfig(
                    safety_budget=1.00,
                    objective="floor_first",
                ),
            ),
        }
    if candidate_set == CANDIDATE_SET_D30_V1:
        d27b = D26CompactDiagConfig(
            stage2b_steps=15,
            stage2c_steps=10,
            bias_guard_mode="per_new_class_pre_registration_old_only",
        )
        return {
            IDENTITY_CANDIDATE: controls[IDENTITY_CANDIDATE],
            DIAG_CANDIDATE: controls[DIAG_CANDIDATE],
            D25_C0: historical[D25_C0],
            D30_A: D30CandidateConfig(
                base=d27b,
                dali=DaliConfig(ground_weight=0.025, direct_weight=0.0),
                envelope_objective="overall_first",
            ),
            D30_B: D30CandidateConfig(
                base=d27b,
                dali=DaliConfig(ground_weight=0.05, direct_weight=0.0),
                envelope_objective="balance_first",
            ),
            D30_C: D30CandidateConfig(
                base=d27b,
                dali=DaliConfig(ground_weight=0.10, direct_weight=0.0),
                envelope_objective="floor_first",
            ),
        }
    if candidate_set == CANDIDATE_SET_D31_V1:
        base = D26CompactDiagConfig(
            stage2b_steps=15,
            stage2c_steps=0,
            bias_guard_mode="per_new_class_pre_registration_old_only",
        )
        return {
            IDENTITY_CANDIDATE: controls[IDENTITY_CANDIDATE],
            DIAG_CANDIDATE: controls[DIAG_CANDIDATE],
            D25_C0: historical[D25_C0],
            D31_A: D31CandidateConfig(
                base=base, stage2c=D31Stage2CConfig(method_id=D31_A)
            ),
            D31_B: D31CandidateConfig(
                base=base, stage2c=D31Stage2CConfig(method_id=D31_B)
            ),
            D31_C: D31CandidateConfig(
                base=base, stage2c=D31Stage2CConfig(method_id=D31_C)
            ),
        }
    if candidate_set in (CANDIDATE_SET_D26_V1, CANDIDATE_SET_D26_V2):
        strict_bias = candidate_set == CANDIDATE_SET_D26_V2
        bias_grid = (
            (-12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0.0)
            if strict_bias
            else (-2.0, -1.0, -0.5, 0.0, 0.5)
        )
        guard_mode = (
            "pre_registration_old_only" if strict_bias else "joint_bias0"
        )
        return {
            IDENTITY_CANDIDATE: controls[IDENTITY_CANDIDATE],
            DIAG_CANDIDATE: controls[DIAG_CANDIDATE],
            D25_C0: historical[D25_C0],
            D26_A: D26CompactDiagConfig(
                stage2b_steps=15,
                stage2c_steps=0,
                bias_guard_mode=guard_mode,
                new_group_bias_grid=bias_grid,
            ),
            D26_B: D26CompactDiagConfig(
                stage2b_steps=15,
                stage2c_steps=10,
                bias_guard_mode=guard_mode,
                new_group_bias_grid=bias_grid,
            ),
            D26_C: D26CompactDiagConfig(
                stage2b_steps=15,
                stage2c_steps=15,
                bias_guard_mode=guard_mode,
                new_group_bias_grid=bias_grid,
            ),
        }
    if candidate_set != CANDIDATE_SET_C3_V1:
        raise D25RunnerError("unknown D25 candidate set")
    ce_weights = D25C3LossWeights(
        equal_class_ce=1.0,
        tail_cvar=0.0,
        hard_negative_margin=0.0,
        proximity=0.01,
    )
    strong_weights = D25C3LossWeights(
        equal_class_ce=1.0,
        tail_cvar=0.20,
        hard_negative_margin=0.10,
        proximity=0.01,
    )
    return {
        IDENTITY_CANDIDATE: controls[IDENTITY_CANDIDATE],
        DIAG_CANDIDATE: controls[DIAG_CANDIDATE],
        D25_C0: historical[D25_C0],
        C3_A: D25C3Config(
            loss_weights=ce_weights,
            stage2b_steps=20,
            stage2c_steps=0,
        ),
        C3_B: D25C3Config(
            loss_weights=ce_weights,
            stage2b_steps=20,
            stage2c_steps=10,
        ),
        C3_C: D25C3Config(
            loss_weights=strong_weights,
            stage2b_steps=15,
            stage2c_steps=15,
        ),
    }


def _candidate_lock(
    candidates: Mapping[str, object],
    candidate_set: str = CANDIDATE_SET_D25_V4,
) -> dict[str, Any]:
    source_closure = {
        "d25_core_sha256": _sha256_file(
            CODE_ROOT / "cvsrffi" / "stage2_multimodal_concat_fusion.py"
        ),
        "d24_uncertainty_fusion_sha256": _sha256_file(
            CODE_ROOT / "cvsrffi" / "stage2_uncertainty_proto_fusion.py"
        ),
        "ciaf_sha256": _sha256_file(CODE_ROOT / "cvsrffi" / "stage2_ciaf.py"),
        "d19_control_helper_sha256": _sha256_file(
            SCRIPT_DIR / "run_d19_support_only_ciaf.py"
        ),
        "diag_cosine_feature_operator_sha256": _sha256_file(
            CODE_ROOT / "cvsrffi" / "stage2_diag_cosine_exploration.py"
        ),
        "d25_runner_sha256": _sha256_file(Path(__file__).resolve()),
    }
    if any(isinstance(value, D25C3Config) for value in candidates.values()):
        source_closure["d25_c3_core_sha256"] = _sha256_file(
            CODE_ROOT / "cvsrffi" / "stage2_multimodal_diag_floor_adapter.py"
        )
    if any(isinstance(value, D26CompactDiagConfig) for value in candidates.values()):
        source_closure["d26_compact_diag_core_sha256"] = _sha256_file(
            CODE_ROOT / "cvsrffi" / "stage2_multimodal_compact_diag.py"
        )
    if any(isinstance(value, D28CandidateConfig) for value in candidates.values()):
        source_closure["d26_compact_diag_core_sha256"] = _sha256_file(
            CODE_ROOT / "cvsrffi" / "stage2_multimodal_compact_diag.py"
        )
        source_closure["d28_support_evidence_gate_core_sha256"] = _sha256_file(
            CODE_ROOT / "cvsrffi" / "stage2_support_evidence_gate.py"
        )
    if any(isinstance(value, D29CandidateConfig) for value in candidates.values()):
        source_closure["d26_compact_diag_core_sha256"] = _sha256_file(
            CODE_ROOT / "cvsrffi" / "stage2_multimodal_compact_diag.py"
        )
        source_closure["d29_classwise_safe_release_core_sha256"] = _sha256_file(
            CODE_ROOT / "cvsrffi" / "stage2_classwise_safe_release.py"
        )
    if any(isinstance(value, D30CandidateConfig) for value in candidates.values()):
        source_closure["d26_compact_diag_core_sha256"] = _sha256_file(
            CODE_ROOT / "cvsrffi" / "stage2_multimodal_compact_diag.py"
        )
        source_closure["d20_dali_core_sha256"] = _sha256_file(
            CODE_ROOT / "cvsrffi" / "stage2_dali.py"
        )
        source_closure["d30_max_envelope_core_sha256"] = _sha256_file(
            CODE_ROOT / "cvsrffi" / "stage2_max_envelope_calibration.py"
        )
    if any(isinstance(value, D31CandidateConfig) for value in candidates.values()):
        source_closure["d26_compact_diag_core_sha256"] = _sha256_file(
            CODE_ROOT / "cvsrffi" / "stage2_multimodal_compact_diag.py"
        )
        source_closure["d20_dali_core_sha256"] = _sha256_file(
            CODE_ROOT / "cvsrffi" / "stage2_dali.py"
        )
        source_closure["d31_all_registered_suffix_core_sha256"] = _sha256_file(
            CODE_ROOT / "cvsrffi" / "stage2_all_registered_new_suffix.py"
        )
    if any(isinstance(value, D32CandidateConfig) for value in candidates.values()):
        source_closure["d26_compact_diag_core_sha256"] = _sha256_file(
            CODE_ROOT / "cvsrffi" / "stage2_multimodal_compact_diag.py"
        )
        source_closure["d20_dali_core_sha256"] = _sha256_file(
            CODE_ROOT / "cvsrffi" / "stage2_dali.py"
        )
        source_closure["d32_inloop_safe_cap_suffix_core_sha256"] = _sha256_file(
            CODE_ROOT / "cvsrffi" / "stage2_inloop_safe_cap_suffix.py"
        )
    if any(isinstance(value, D33CandidateConfig) for value in candidates.values()):
        source_closure["d26_compact_diag_core_sha256"] = _sha256_file(
            CODE_ROOT / "cvsrffi" / "stage2_multimodal_compact_diag.py"
        )
        source_closure["d33_spherical_registration_core_sha256"] = _sha256_file(
            CODE_ROOT / "cvsrffi" / "stage2_d33_spherical_registration.py"
        )
        source_closure["d33_b3_fisher_closed_form_core_sha256"] = _sha256_file(
            CODE_ROOT / "cvsrffi" / "stage2_b3_fisher_closed_form.py"
        )
    if any(isinstance(value, D34CandidateConfig) for value in candidates.values()):
        source_closure["d34_collision_local_registration_core_sha256"] = (
            _sha256_file(
                CODE_ROOT
                / "cvsrffi"
                / "stage2_d34_collision_local_registration.py"
            )
        )
        source_closure["d34_b3_fisher_closed_form_core_sha256"] = _sha256_file(
            CODE_ROOT / "cvsrffi" / "stage2_b3_fisher_closed_form.py"
        )
    if any(isinstance(value, D35CandidateConfig) for value in candidates.values()):
        source_closure["d35_dense_safe_registration_core_sha256"] = _sha256_file(
            CODE_ROOT / "cvsrffi" / "stage2_d35_dense_safe_registration.py"
        )
        source_closure["d35_b3_fisher_closed_form_core_sha256"] = _sha256_file(
            CODE_ROOT / "cvsrffi" / "stage2_b3_fisher_closed_form.py"
        )
    if any(isinstance(value, D36CandidateConfig) for value in candidates.values()):
        source_closure["d36_compiled_joint_int8_core_sha256"] = _sha256_file(
            CODE_ROOT / "cvsrffi" / "stage2_d36_compiled_joint_int8.py"
        )
        source_closure["d36_b3_fisher_closed_form_core_sha256"] = _sha256_file(
            CODE_ROOT / "cvsrffi" / "stage2_b3_fisher_closed_form.py"
        )
    if any(isinstance(value, D37CandidateConfig) for value in candidates.values()):
        source_closure["d37_b3_preserving_int8_core_sha256"] = _sha256_file(
            CODE_ROOT / "cvsrffi" / "stage2_d37_b3_preserving_int8.py"
        )
        source_closure["d37_b3_fisher_closed_form_core_sha256"] = _sha256_file(
            CODE_ROOT / "cvsrffi" / "stage2_b3_fisher_closed_form.py"
        )
    if any(isinstance(value, D38CandidateConfig) for value in candidates.values()):
        source_closure["d38_strong_b3_quantized_core_sha256"] = _sha256_file(
            CODE_ROOT / "cvsrffi" / "stage2_d38_strong_b3_quantized.py"
        )
    if any(isinstance(value, D39CandidateConfig) for value in candidates.values()):
        source_closure["d39_angular_radius_core_sha256"] = _sha256_file(
            CODE_ROOT / "cvsrffi" / "stage2_d39_angular_radius.py"
        )
    if any(isinstance(value, D40CandidateConfig) for value in candidates.values()):
        source_closure["d40_hnbr_core_sha256"] = _sha256_file(
            CODE_ROOT / "cvsrffi" / "stage2_d40_hnbr.py"
        )
    rows: list[dict[str, Any]] = []
    for candidate_id, config in candidates.items():
        if isinstance(config, D40CandidateConfig):
            config_row = {
                "core": asdict(config.core),
                "deploy_precision": config.deploy_precision,
                "base_route": "D38_arm_A_stage2B_trajectory",
                "feature_geometry": "normalized_z160_plus_4x_joint_normalized_fft96_rf32",
                "hnbr_temperature": D40_HNBR_TEMPERATURE,
                "hnbr_formula": "normalize(base-max(0,dot(base,negative_centroid))*negative_centroid)",
                "old_negative_set": "other_old_base_directions_synchronous",
                "new_negative_set": "frozen_old_final_plus_other_new_base_synchronous",
                "stage2c_optimizer_steps": 0,
                "label_permutation_equivariant": True,
                "old_prefix_policy": "stage2b_hnbr_prefix_bitwise_unchanged_after_append",
                "pairwise_support_diagnostics": True,
            }
            family = "d40_hnbr"
        elif isinstance(config, D39CandidateConfig):
            config_row = {
                "core": asdict(config.core),
                "deploy_precision": config.deploy_precision,
                "base_route": "exact_D38_B_training_trajectory",
                "feature_geometry": "normalized_z160_plus_4x_joint_normalized_fft96_rf32",
                "radius_formula": "(nu*r0^2+(K-1)*m2)/(nu+K-1)",
                "radius_nu": D39_RADIUS_NU,
                "radius_epsilon": D39_RADIUS_EPSILON,
                "radius_r0_floor": D39_R0_FLOOR,
                "k1_radius_rule": "K_minus_1_zero_radius_equals_r0_bitwise",
                "label_permutation_equivariant": True,
                "radius_storage_dtype": "float16",
                "r0_storage_dtype": "float16",
                "old_radius_source": "before_state_int8_old_support_only",
                "new_radius_source": "final_state_int8_new_support_only",
                "old_prefix_policy": "base_radius_r0_bitwise_unchanged",
                "pairwise_support_diagnostics": True,
            }
            family = "d39_angular_radius"
        elif isinstance(config, D38CandidateConfig):
            config_row = {
                "core": asdict(config.core),
                "deploy_precision": config.deploy_precision,
                "feature_geometry": "normalized_z160_plus_4x_joint_normalized_fft96_rf32",
                "stage2b": "fullbatch_adamw20_no_bias",
                "stage2c": (
                    "centroid_only"
                    if config.core.arm == "A"
                    else "all_support_new_weight_only_sgd10"
                ),
                "old_prefix_policy": "compiled_before_stage2c_bitwise_unchanged",
                "pairwise_support_diagnostics": True,
            }
            family = "d38_strong_b3_quantized"
        elif isinstance(config, D38ProtoNetCDAConfig):
            config_row = {
                "feature_geometry": config.feature_geometry,
                "support_mean": True,
                "nearest_prototype": True,
                "independent_candidate_row": True,
                "equivalence_audit_required": True,
            }
            family = "d38_protonet_cda"
        elif isinstance(config, D37CandidateConfig):
            config_row = {
                "old_solver": "b3_fisher_closed_form",
                "fisher": {
                    "shrinkage_strengths": list(config.fisher.shrinkage_strengths),
                    "variance_ridge": float(config.fisher.variance_ridge),
                    "fisher_shrinkage": float(config.fisher.fisher_shrinkage),
                },
                "compiled": asdict(config.compiled),
                "feature_geometry": "b3_auxiliary_dominant_z160_fft96_rf32_v1",
                "target_old_new_prototype_storage": (
                    "two_level_residual_int8_plus_fp16_block_scales"
                ),
                "old_prefix_policy": "append_only_bitwise_b3_int8_prefix",
                "inner_crossfit": "rank_pairs_base_scores_and_physical_labels",
                "oof_calibration": "shared_new_offset_feasible_interval",
            }
            family = "d37_b3_preserving_residual_int8"
        elif isinstance(config, D36CandidateConfig):
            config_row = {
                "old_solver": "b3_fisher_closed_form",
                "fisher": {
                    "shrinkage_strengths": list(config.fisher.shrinkage_strengths),
                    "variance_ridge": float(config.fisher.variance_ridge),
                    "fisher_shrinkage": float(config.fisher.fisher_shrinkage),
                },
                "compiled": asdict(config.compiled),
                "feature_geometry": "b3_auxiliary_dominant_z160_fft96_rf32_v1",
                "target_old_new_prototype_storage": (
                    "symmetric_int8_plus_fp16_scale_inverse_norm_radius"
                ),
                "inner_crossfit": "four_rank_pairs_within_outer_train",
                "phase1_anchor": (
                    "disabled" if candidate_id == D36_A else "fixed_maximin_medoid_read_only"
                ),
            }
            family = "d36_compiled_joint_int8"
        elif isinstance(config, D35CandidateConfig):
            config_row = {
                "old_solver": "b3_fisher_closed_form",
                "fisher": {
                    "shrinkage_strengths": list(config.fisher.shrinkage_strengths),
                    "variance_ridge": float(config.fisher.variance_ridge),
                    "fisher_shrinkage": float(config.fisher.fisher_shrinkage),
                },
                "registration": asdict(config.registration),
                "feature_geometry": "b3_auxiliary_dominant_z160_fft96_rf32_v1",
                "old_score_prefix_policy": "bitwise_frozen_fast_fisher",
                "new_prototype_storage": "symmetric_int8_plus_fp32_scale_inverse_norm",
                "all_new_classes_globally_visible": True,
                "support_only_dense_safe_thresholds": True,
            }
            family = "d35_dense_safe_registration"
        elif isinstance(config, D34CandidateConfig):
            config_row = {
                "old_solver": "b3_fisher_closed_form",
                "fisher": {
                    "shrinkage_strengths": list(config.fisher.shrinkage_strengths),
                    "variance_ridge": float(config.fisher.variance_ridge),
                    "fisher_shrinkage": float(config.fisher.fisher_shrinkage),
                },
                "registration": asdict(config.registration),
                "feature_geometry": "b3_auxiliary_dominant_z160_fft96_rf32_v1",
                "old_score_prefix_policy": "bitwise_frozen_fast_fisher",
                "new_prototype_storage": "symmetric_int8_plus_fp32_scale_inverse_norm",
                "support_only_sparse_collision_graph": True,
            }
            family = "d34_collision_local_registration"
        elif isinstance(config, D33CandidateConfig):
            config_row = {
                "old_solver": config.old_solver,
                "base": None
                if config.base is None
                else {
                    "stage2b_steps": int(config.base.stage2b_steps),
                    "stage2c_steps": int(config.base.stage2c_steps),
                    "bias_guard_mode": str(config.base.bias_guard_mode),
                },
                "fisher": None
                if config.fisher is None
                else {
                    "shrinkage_strengths": list(config.fisher.shrinkage_strengths),
                    "variance_ridge": float(config.fisher.variance_ridge),
                    "fisher_shrinkage": float(config.fisher.fisher_shrinkage),
                },
                "registration": {
                    "selection_policy": config.registration.selection_policy,
                    "radius_quantiles": list(config.registration.radius_quantiles),
                    "radius_shrinkages": list(
                        config.registration.radius_shrinkages
                    ),
                    "radius_ratio_caps": list(config.registration.radius_ratio_caps),
                },
                "feature_geometry": "b3_auxiliary_dominant_z160_fft96_rf32_v1",
                "int8_predictor_dependency": False,
                "training_deployment_score_surface_identical": True,
            }
            family = "d33_spherical_registration"
        elif isinstance(config, D32CandidateConfig):
            config_row = {
                "base": {
                    "stage2b_steps": int(config.base.stage2b_steps),
                    "stage2c_steps": int(config.base.stage2c_steps),
                    "bias_guard_mode": str(config.base.bias_guard_mode),
                },
                "feature_geometry": "b3_auxiliary_dominant_z160_fft96_rf32_v1",
                "stage2c": config.stage2c.audit(),
                "dali": {
                    "ground_weight": float(config.dali.ground_weight),
                    "direct_weight": float(config.dali.direct_weight),
                    "fixed_medoid": True,
                    "support_old_classwise_atomic_gate": True,
                    "max_old_preserved": True,
                },
                "training_deployment_safe_cap_surface_identical": True,
            }
            family = "d32_inloop_safe_cap_suffix_with_dali"
        elif isinstance(config, D31CandidateConfig):
            config_row = {
                "base": {
                    "stage2b_steps": int(config.base.stage2b_steps),
                    "stage2c_steps": int(config.base.stage2c_steps),
                    "bias_guard_mode": str(config.base.bias_guard_mode),
                },
                "feature_geometry": "b3_auxiliary_dominant_z160_fft96_rf32_v1",
                "stage2c": config.stage2c.audit(),
                "dali": {
                    "ground_weight": float(config.dali.ground_weight),
                    "direct_weight": float(config.dali.direct_weight),
                    "fixed_medoid": True,
                    "support_old_classwise_atomic_gate": True,
                    "max_old_preserved": True,
                },
            }
            family = "d31_all_registered_suffix_with_dali"
        elif isinstance(config, D30CandidateConfig):
            config_row = {
                "base": {
                    "stage2b_steps": int(config.base.stage2b_steps),
                    "stage2c_steps": int(config.base.stage2c_steps),
                    "learning_rate": float(config.base.learning_rate),
                    "weight_decay": float(config.base.weight_decay),
                    "prototype_anchor_weight": float(
                        config.base.prototype_anchor_weight
                    ),
                    "diagonal_proximity_weight": float(
                        config.base.diagonal_proximity_weight
                    ),
                    "bias_guard_mode": str(config.base.bias_guard_mode),
                    "new_group_bias_grid": list(config.base.new_group_bias_grid),
                    "new_class_bias_offsets": list(
                        config.base.new_class_bias_offsets
                    ),
                },
                "feature_geometry": "b3_auxiliary_dominant_z160_fft96_rf32_v1",
                "dali": {
                    "ground_weight": float(config.dali.ground_weight),
                    "direct_weight": float(config.dali.direct_weight),
                    "fixed_medoid": True,
                    "support_old_classwise_atomic_gate": True,
                    "max_old_preserved": True,
                },
                "max_new_envelope": {
                    "objective": config.envelope_objective,
                    "support_only": True,
                    "row_local_inference": True,
                    "max_new_preserved": True,
                    "k1_exact_passthrough": True,
                },
            }
            family = "d30_b3_dali_dual_envelope"
        elif isinstance(config, D29CandidateConfig):
            config_row = {
                "base": {
                    "stage2b_steps": int(config.base.stage2b_steps),
                    "stage2c_steps": int(config.base.stage2c_steps),
                    "learning_rate": float(config.base.learning_rate),
                    "weight_decay": float(config.base.weight_decay),
                    "prototype_anchor_weight": float(
                        config.base.prototype_anchor_weight
                    ),
                    "diagonal_proximity_weight": float(
                        config.base.diagonal_proximity_weight
                    ),
                    "bias_guard_mode": str(config.base.bias_guard_mode),
                    "new_group_bias_grid": list(
                        config.base.new_group_bias_grid
                    ),
                    "new_class_bias_offsets": list(
                        config.base.new_class_bias_offsets
                    ),
                },
                "release": {
                    "safety_budget": float(config.release.safety_budget),
                    "objective": str(config.release.objective),
                    "support_only": True,
                    "row_local_inference": True,
                    "coordinate_passes": int(config.release.coordinate_passes),
                },
            }
            family = "d29_per_class_safe_release"
        elif isinstance(config, D28CandidateConfig):
            config_row = {
                "base": {
                    "stage2b_steps": int(config.base.stage2b_steps),
                    "stage2c_steps": int(config.base.stage2c_steps),
                    "learning_rate": float(config.base.learning_rate),
                    "weight_decay": float(config.base.weight_decay),
                    "prototype_anchor_weight": float(
                        config.base.prototype_anchor_weight
                    ),
                    "diagonal_proximity_weight": float(
                        config.base.diagonal_proximity_weight
                    ),
                    "bias_guard_mode": str(config.base.bias_guard_mode),
                    "new_class_bias_offsets": list(
                        config.base.new_class_bias_offsets
                    ),
                },
                "gate": {
                    "ridge_lambdas": list(config.gate.ridge_lambdas),
                    "alpha": float(config.gate.alpha),
                    "delta": float(config.gate.delta),
                    "oof_fold_count": int(config.gate.oof_fold_count),
                    "support_only": True,
                    "row_local_inference": True,
                },
            }
            family = "d28_support_evidence_gate"
        elif isinstance(config, D25C3Config):
            config_row = config.lock_payload()
            family = "d25_c3"
        elif isinstance(config, D26CompactDiagConfig):
            config_row = {
                "stage2b_steps": int(config.stage2b_steps),
                "stage2c_steps": int(config.stage2c_steps),
                "learning_rate": float(config.learning_rate),
                "weight_decay": float(config.weight_decay),
                "prototype_anchor_weight": float(config.prototype_anchor_weight),
                "diagonal_proximity_weight": float(
                    config.diagonal_proximity_weight
                ),
                "new_group_bias_grid": list(config.new_group_bias_grid),
                "bias_guard_mode": str(config.bias_guard_mode),
                "new_class_bias_offsets": list(
                    getattr(config, "new_class_bias_offsets", ())
                ),
            }
            family = (
                "d27_per_new_class_bias"
                if candidate_set in (CANDIDATE_SET_D27_V1, CANDIDATE_SET_D28_V1)
                else "d26_compact_diag"
            )
        elif isinstance(config, MultimodalConcatConfig):
            config_row: dict[str, Any] = {
                "block_energy": list(config.block_energy),
                "r0_by_block": list(config.r0_by_block),
                "r_min": config.r_min,
                "separation_margin": config.separation_margin,
                "score_mode": config.score_mode,
                "use_ground_identity_fusion": config.use_ground_identity_fusion,
            }
            family = "d25"
        else:
            config_row = {
                "ground_weight": float(config.ground_weight),
                "direct_weight": float(config.direct_weight),
            }
            family = "control"
        rows.append(
            {
                "candidate_id": candidate_id,
                "family": family,
                "config": config_row,
                "eligible_positive_route": candidate_id
                in _positive_route_candidates(candidate_set),
            }
        )
    lock = {
        "schema": (
            "cvs.phase2.d25.candidate_lock.v2"
            if candidate_set == CANDIDATE_SET_C3_V1
            else "cvs.phase2.d25.candidate_lock.v3"
            if candidate_set == CANDIDATE_SET_D26_V1
            else "cvs.phase2.d25.candidate_lock.v4"
            if candidate_set == CANDIDATE_SET_D26_V2
            else "cvs.phase2.d25.candidate_lock.v5"
            if candidate_set == CANDIDATE_SET_D27_V1
            else "cvs.phase2.d25.candidate_lock.v6"
            if candidate_set == CANDIDATE_SET_D28_V1
            else "cvs.phase2.d25.candidate_lock.v7"
            if candidate_set == CANDIDATE_SET_D29_V1
            else "cvs.phase2.d25.candidate_lock.v8"
            if candidate_set == CANDIDATE_SET_D30_V1
            else "cvs.phase2.d25.candidate_lock.v9"
            if candidate_set == CANDIDATE_SET_D31_V1
            else "cvs.phase2.d25.candidate_lock.v10"
            if candidate_set == CANDIDATE_SET_D32_V1
            else "cvs.phase2.d25.candidate_lock.v11"
            if candidate_set == CANDIDATE_SET_D33_V1
            else "cvs.phase2.d25.candidate_lock.v12"
            if candidate_set == CANDIDATE_SET_D34_V1
            else "cvs.phase2.d25.candidate_lock.v13"
            if candidate_set == CANDIDATE_SET_D35_V1
            else "cvs.phase2.d25.candidate_lock.v14"
            if candidate_set == CANDIDATE_SET_D36_V1
            else "cvs.phase2.d25.candidate_lock.v15"
            if candidate_set == CANDIDATE_SET_D37_V1
            else "cvs.phase2.d25.candidate_lock.v16"
            if candidate_set == CANDIDATE_SET_D38_V1
            else "cvs.phase2.d25.candidate_lock.v17"
            if candidate_set == CANDIDATE_SET_D39_V1
            else "cvs.phase2.d25.candidate_lock.v18"
            if candidate_set == CANDIDATE_SET_D40_V1
            else "cvs.phase2.d25.candidate_lock.v1"
        ),
        "core_commit": CORE_COMMIT,
        "held_ranks": [list(value) for value in HELD_RANKS],
        "candidates": rows,
        "selection_baseline": (
            D25_C0
            if candidate_set
            in (
                CANDIDATE_SET_C3_V1,
                CANDIDATE_SET_D26_V1,
                CANDIDATE_SET_D26_V2,
                CANDIDATE_SET_D27_V1,
                CANDIDATE_SET_D28_V1,
                CANDIDATE_SET_D29_V1,
                CANDIDATE_SET_D30_V1,
                CANDIDATE_SET_D31_V1,
                CANDIDATE_SET_D32_V1,
                CANDIDATE_SET_D33_V1,
                CANDIDATE_SET_D34_V1,
                CANDIDATE_SET_D35_V1,
                CANDIDATE_SET_D36_V1,
                CANDIDATE_SET_D37_V1,
            )
            else IDENTITY_CANDIDATE
        ),
        "diagnostic_comparator": DIAG_CANDIDATE,
        "source_closure": source_closure,
    }
    if candidate_set in (
        CANDIDATE_SET_C3_V1,
        CANDIDATE_SET_D26_V1,
        CANDIDATE_SET_D26_V2,
        CANDIDATE_SET_D27_V1,
        CANDIDATE_SET_D28_V1,
        CANDIDATE_SET_D29_V1,
        CANDIDATE_SET_D30_V1,
        CANDIDATE_SET_D31_V1,
        CANDIDATE_SET_D32_V1,
        CANDIDATE_SET_D33_V1,
        CANDIDATE_SET_D34_V1,
        CANDIDATE_SET_D35_V1,
        CANDIDATE_SET_D36_V1,
        CANDIDATE_SET_D37_V1,
        CANDIDATE_SET_D38_V1,
        CANDIDATE_SET_D39_V1,
        CANDIDATE_SET_D40_V1,
    ):
        lock["candidate_set"] = candidate_set
    if candidate_set in (
        CANDIDATE_SET_D26_V1,
        CANDIDATE_SET_D26_V2,
        CANDIDATE_SET_D27_V1,
        CANDIDATE_SET_D28_V1,
        CANDIDATE_SET_D29_V1,
        CANDIDATE_SET_D30_V1,
        CANDIDATE_SET_D31_V1,
        CANDIDATE_SET_D32_V1,
        CANDIDATE_SET_D33_V1,
        CANDIDATE_SET_D34_V1,
        CANDIDATE_SET_D35_V1,
        CANDIDATE_SET_D36_V1,
        CANDIDATE_SET_D37_V1,
        CANDIDATE_SET_D38_V1,
        CANDIDATE_SET_D39_V1,
        CANDIDATE_SET_D40_V1,
    ):
        # CORE_COMMIT above identifies the sealed Phase1 model lineage.  Keep
        # the D26 implementation commit separate so the receipt cannot imply
        # that the new adapter was already present in that older model commit.
        lock["d26_core_git_commit"] = D26_CORE_GIT_COMMIT
    if candidate_set in (
        CANDIDATE_SET_D29_V1,
        CANDIDATE_SET_D30_V1,
        CANDIDATE_SET_D31_V1,
        CANDIDATE_SET_D32_V1,
        CANDIDATE_SET_D33_V1,
        CANDIDATE_SET_D34_V1,
        CANDIDATE_SET_D35_V1,
        CANDIDATE_SET_D36_V1,
        CANDIDATE_SET_D37_V1,
        CANDIDATE_SET_D38_V1,
        CANDIDATE_SET_D39_V1,
        CANDIDATE_SET_D40_V1,
    ):
        lock["protocol_contract"] = {
            "screen_authority": "PRE_FORMAL_SUPPORT_ONLY_INT8_SCREEN",
            "phase2_query_decision_policy": "per_sample_all_registered_classes",
            "phase2_query_role_oracle_access": False,
            "phase2_query_true_batch_class_count_access": False,
            "phase2_query_class_quota_access": False,
            "phase2_query_batch_global_assignment": False,
            "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
            "clean_sample_access": False,
            "clean_derived_signal_access": False,
            "phase2_clean_dataset_reachable": False,
            "phase2_clean_cache_reachable": False,
            "phase2_clean_control_flow_reachable": False,
            "phase2_source_sample_access": False,
            "phase2_source_cache_access": False,
            "phase2_source_label_access": False,
            "phase2_unapproved_source_derived_signal_access": False,
            "phase2_source_replay": False,
            "phase2_external_source_adapter_access": False,
            "phase2_pretrained_artifact_policy": (
                "sealed_phase1_deployment_bundle_with_optional_int8_"
                "domain_class_prototypes_v1"
            ),
            "query_opened": False,
            "formal_launch_authority": False,
            "formal_metric_claim_allowed": False,
            "performance_claim_allowed": False,
        }
    if candidate_set == CANDIDATE_SET_D39_V1:
        lock["d39_formula_lock"] = {
            "radius_formula": "(nu*r0^2+(K-1)*m2)/(nu+K-1)",
            "nu": D39_RADIUS_NU,
            "epsilon": D39_RADIUS_EPSILON,
            "r0_floor": D39_R0_FLOOR,
            "k1_policy": "all_radius_equals_frozen_r0",
            "score": "-0.5*(theta/(radius+epsilon))^2-log(radius+epsilon)",
            "training_trajectory": "exact_D38_B_20_plus_10",
        }
    if candidate_set == CANDIDATE_SET_D40_V1:
        lock["d40_formula_lock"] = {
            "temperature": D40_HNBR_TEMPERATURE,
            "negative_centroid": "stable_softmax_over_current_competitor_base_directions",
            "projection": "rho=max(0,dot(base,negative_centroid))",
            "residual": "normalize(base-rho*negative_centroid)",
            "stage2b": "D38_arm_A_stage2B_fullbatch_adamw20_then_synchronous_old_hnbr",
            "stage2c": "zero_step_synchronous_new_hnbr_append",
            "new_new_confusion_cap_exclusive": D40_NEW_NEW_CONFUSION_CAP,
        }
    return {**lock, "sha256": hashlib.sha256(_canonical_bytes(lock)).hexdigest()}


def _d1_feature_from_blocks(
    z_id160: np.ndarray, fft96: np.ndarray, rf32: np.ndarray
) -> np.ndarray:
    """Rebuild the historical B3 288-D feature without recomputing FFT/RF."""

    z_rows = legacy._normalize_matrix(np.asarray(z_id160, dtype=np.float32))
    auxiliary = np.concatenate(
        [np.asarray(fft96, dtype=np.float32), np.asarray(rf32, dtype=np.float32)],
        axis=1,
    )
    auxiliary = legacy._normalize_matrix(auxiliary)
    return legacy._normalize_matrix(
        np.concatenate([z_rows, np.float32(4.0) * auxiliary], axis=1)
    )


def _d30_observed_block_energy(features: np.ndarray) -> dict[str, float]:
    value = np.asarray(features, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 288:
        raise D25RunnerError("D30 feature-energy audit shape drift")
    squared = np.square(value, dtype=np.float32)
    return {
        "z160": float(np.mean(np.sum(squared[:, :160], axis=1))),
        "fft96": float(np.mean(np.sum(squared[:, 160:256], axis=1))),
        "rf32": float(np.mean(np.sum(squared[:, 256:], axis=1))),
        "fft96_rf32_aux_total": float(
            np.mean(np.sum(squared[:, 160:], axis=1))
        ),
    }


def _operator_lineage(rows: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    tokens = np.asarray(rows["tokens"]).astype(str)
    hashes = np.asarray(rows["hashes"]).astype(str)
    if (
        len(tokens) != len(hashes)
        or len(set(tokens.tolist())) != len(tokens)
        or any(len(value) != 64 for value in hashes)
    ):
        raise D25RunnerError("D25 feature-operator lineage parent drift")
    operators = (
        "adv3b02_zid160_base_v1",
        "same_received_iq_fft96_v1",
        "same_received_iq_rf32_v1",
    )
    return [
        {
            "physical_sample_id": token,
            "parent_received_iq_sha256": parent,
            "feature_operator_ids": list(operators),
            "support_row_multiplicity": 1,
            "derived_support_rows": 0,
            "additional_physical_sample_count": 0,
            "additional_leo_overlay_count": 0,
        }
        for token, parent in zip(tokens.tolist(), hashes.tolist())
    ]


def _geometry_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    pairs = list(value["pairs"])
    worst = min(pairs, key=lambda row: float(row["gap"])) if pairs else None
    return {
        "schema": "cvs.phase2.d25.geometry_summary.v1",
        "pair_count": int(value["pair_count"]),
        "collision_count": int(value["collision_count"]),
        "pass": bool(value["pass"]),
        "worst_pair": worst,
        "query_rows_used": 0,
    }


def _evaluate_d25_fold(
    component: object,
    rows: Mapping[str, np.ndarray],
    z_id160: np.ndarray,
    fft96: np.ndarray,
    rf32: np.ndarray,
    *,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    held_ranks: tuple[int, int],
    candidate_id: str,
    config: MultimodalConcatConfig,
) -> dict[str, Any]:
    labels = np.asarray(rows["labels"]).astype(str)
    ranks = np.asarray(rows["ranks"], dtype=np.int64)
    held = np.isin(ranks, np.asarray(held_ranks, dtype=np.int64))
    train = ~held
    old = np.isin(labels, np.asarray(old_classes))
    new = np.isin(labels, np.asarray(new_classes))
    if (
        int(np.sum(train & old)) != 8 * len(old_classes)
        or int(np.sum(train & new)) != 8 * len(new_classes)
        or int(np.sum(held & old)) != 2 * len(old_classes)
        or int(np.sum(held & new)) != 2 * len(new_classes)
    ):
        raise D25RunnerError("D25 leave-two-out class symmetry drift")

    ground_component = component if config.use_ground_identity_fusion else None
    before = fit_old_concat(
        ground_component,
        z_id160[train & old],
        fft96[train & old],
        rf32[train & old],
        labels[train & old],
        registered_classes=old_classes,
        config=config,
    )
    after = append_new_classes_concat(
        before,
        z_id160[train & new],
        fft96[train & new],
        rf32[train & new],
        labels[train & new],
        registered_classes=new_classes,
    )
    if before.classes != old_classes or after.classes != old_classes + new_classes:
        raise D25RunnerError("D25 registered class order drift")
    if before.old_prefix_sha256 != after.old_prefix_sha256:
        raise D25RunnerError("D25 old prefix changed after registration")

    held_old_feature = build_concat288(
        z_id160[held & old],
        fft96[held & old],
        rf32[held & old],
        block_energy=config.block_energy,
    )
    held_new_feature = build_concat288(
        z_id160[held & new],
        fft96[held & new],
        rf32[held & new],
        block_energy=config.block_energy,
    )
    before_predictions = [
        predict_one_concat(before, row)[0] for row in held_old_feature
    ]
    after_old_predictions = [
        predict_one_concat(after, row)[0] for row in held_old_feature
    ]
    after_new_predictions = [
        predict_one_concat(after, row)[0] for row in held_new_feature
    ]
    old_scores_unchanged = all(
        np.array_equal(
            score_one_concat(before, row),
            score_one_concat(after, row)[: len(old_classes)],
        )
        for row in held_old_feature
    )
    if not old_scores_unchanged:
        raise D25RunnerError("D25 old score columns changed after registration")

    before_old = legacy._metric_block(
        labels[held & old], before_predictions, old_classes
    )
    after_old = legacy._metric_block(
        labels[held & old], after_old_predictions, old_classes
    )
    after_new = legacy._metric_block(
        labels[held & new], after_new_predictions, new_classes
    )
    h_value = legacy._harmonic(
        float(after_old["overall_accuracy"]),
        float(after_new["overall_accuracy"]),
    )
    forgetting = float(
        before_old["overall_accuracy"] - after_old["overall_accuracy"]
    )
    geometry = after.geometry_audit()
    resource = dict(after.resource_audit())
    resource.update(
        {
            "int8_component_used_for_prediction": config.use_ground_identity_fusion,
            "old_score_columns_bitwise_unchanged_after_registration": True,
            "query_role_oracle_access": False,
            "query_true_batch_class_count_access": False,
            "query_class_quota_access": False,
            "query_batch_global_assignment": False,
            "query_features_used_for_fit": False,
            "query_labels_used_for_fit": False,
            "source_sample_access": False,
            "source_cache_access": False,
            "source_derived_signal_access": False,
            "clean_sample_access": False,
            "clean_derived_signal_access": False,
            "complete_loss_trace": [],
        }
    )
    return {
        "candidate_id": candidate_id,
        "held_ranks": list(held_ranks),
        "fit_k_shot": 8,
        "before_old": before_old,
        "after_old": after_old,
        "after_new": after_new,
        "H_old_new": h_value,
        "forgetting": forgetting,
        "joint_floor": float(
            min(
                float(after_old["class_floor_accuracy"]),
                float(after_new["class_floor_accuracy"]),
            )
        ),
        "old_score_columns_bitwise_unchanged": True,
        "old_prefix_sha256_before": before.old_prefix_sha256,
        "old_prefix_sha256_after": after.old_prefix_sha256,
        "geometry_summary": _geometry_summary(geometry),
        "resource": resource,
    }


def _c3_geometry(state: D25C3State) -> dict[str, Any]:
    prototypes = np.asarray(state.prototypes, dtype=np.float32)
    pairs: list[dict[str, Any]] = []
    for left in range(len(state.classes)):
        for right in range(left + 1, len(state.classes)):
            distance = float(
                1.0
                - np.dot(
                    prototypes[left].astype(np.float32),
                    prototypes[right].astype(np.float32),
                )
            )
            role = (
                "old_old"
                if right < state.old_class_count
                else ("new_new" if left >= state.old_class_count else "old_new")
            )
            pairs.append(
                {
                    "left": state.classes[left],
                    "right": state.classes[right],
                    "role": role,
                    "cosine_distance": distance,
                    "collision_below_0p05": distance < 0.05,
                }
            )
    distances = [float(row["cosine_distance"]) for row in pairs]
    return {
        "schema": "cvs.phase2.d25_c3.prototype_geometry.v1",
        "class_count": len(state.classes),
        "old_class_count": state.old_class_count,
        "pair_count": len(pairs),
        "minimum_cosine_distance": min(distances) if distances else None,
        "collision_count_below_0p05": sum(
            int(bool(row["collision_below_0p05"])) for row in pairs
        ),
        "pairs": pairs,
    }


def _d26_geometry(state: D26CompactDiagState) -> dict[str, Any]:
    weights = np.asarray(state.weights, dtype=np.float32)
    pairs: list[dict[str, Any]] = []
    for left in range(len(state.classes)):
        for right in range(left + 1, len(state.classes)):
            distance = float(1.0 - np.dot(weights[left], weights[right]))
            role = (
                "old_old"
                if right < state.old_class_count
                else ("new_new" if left >= state.old_class_count else "old_new")
            )
            pairs.append(
                {
                    "left": state.classes[left],
                    "right": state.classes[right],
                    "role": role,
                    "cosine_distance": distance,
                    "collision_below_0p05": distance < 0.05,
                }
            )
    distances = [float(row["cosine_distance"]) for row in pairs]
    bias_audit = json.loads(state.bias_audit_json)
    return {
        "schema": "cvs.phase2.d26_compact_diag_geometry.v1",
        "class_count": len(state.classes),
        "old_class_count": state.old_class_count,
        "pair_count": len(pairs),
        "minimum_cosine_distance": min(distances) if distances else None,
        "collision_count_below_0p05": sum(
            int(bool(row["collision_below_0p05"])) for row in pairs
        ),
        "new_group_bias": float(state.new_group_bias),
        "bias_applied_to_new_suffix_only": True,
        "bias_support_only_audit": bias_audit,
        "pairs": pairs,
    }


def _d26_new_class_biases(state: D26CompactDiagState) -> list[float]:
    values = np.asarray(
        getattr(state, "new_class_biases", np.empty(0, dtype=np.float32)),
        dtype=np.float32,
    )
    if values.ndim != 1:
        raise D25RunnerError("D26/D27 new-class bias vector drift")
    return [float(value) for value in values.tolist()]


def _evaluate_c3_fold(
    rows: Mapping[str, np.ndarray],
    z_id160: np.ndarray,
    fft96: np.ndarray,
    rf32: np.ndarray,
    *,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    held_ranks: tuple[int, int],
    candidate_id: str,
    config: D25C3Config,
) -> dict[str, Any]:
    labels = np.asarray(rows["labels"]).astype(str)
    ranks = np.asarray(rows["ranks"], dtype=np.int64)
    held = np.isin(ranks, np.asarray(held_ranks, dtype=np.int64))
    train = ~held
    old = np.isin(labels, np.asarray(old_classes))
    new = np.isin(labels, np.asarray(new_classes))
    if (
        int(np.sum(train & old)) != 8 * len(old_classes)
        or int(np.sum(train & new)) != 8 * len(new_classes)
        or int(np.sum(held & old)) != 2 * len(old_classes)
        or int(np.sum(held & new)) != 2 * len(new_classes)
    ):
        raise D25RunnerError("C3 leave-two-out class symmetry drift")
    features = build_concat288(z_id160, fft96, rf32)
    before_fit = fit_stage2b_diag_floor(
        features[train & old],
        labels[train & old],
        old_classes,
        config=config,
    )
    before = before_fit.state
    after_fit = append_stage2c_new_suffix(
        before,
        features[train & new],
        labels[train & new],
        new_classes,
    )
    after = after_fit.state
    if before.classes != old_classes or after.classes != old_classes + new_classes:
        raise D25RunnerError("C3 registered class order drift")
    if (
        before.old_prefix_sha256 != after.old_prefix_sha256
        or before.shared_sha256 != after.shared_sha256
    ):
        raise D25RunnerError("C3 shared or old prefix changed after registration")

    held_old = features[held & old]
    held_new = features[held & new]
    before_predictions = [predict_one_c3(before, row)[0] for row in held_old]
    after_old_predictions = [predict_one_c3(after, row)[0] for row in held_old]
    after_new_predictions = [predict_one_c3(after, row)[0] for row in held_new]
    old_scores_unchanged = all(
        np.array_equal(
            score_one_c3(before, row),
            score_one_c3(after, row)[: len(old_classes)],
        )
        for row in held_old
    )
    if not old_scores_unchanged:
        raise D25RunnerError("C3 old raw score prefix changed after registration")

    fit_old_features = features[train & old]
    fit_old_labels = labels[train & old]
    fit_before_predictions = [
        predict_one_c3(before, row)[0] for row in fit_old_features
    ]
    fit_after_predictions = [
        predict_one_c3(after, row)[0] for row in fit_old_features
    ]
    fit_before = legacy._metric_block(
        fit_old_labels, fit_before_predictions, old_classes
    )
    fit_after = legacy._metric_block(
        fit_old_labels, fit_after_predictions, old_classes
    )
    tolerance = 1.0e-12
    fit_classwise_non_degradation = all(
        float(fit_after["per_class_accuracy"][label]) + tolerance
        >= float(fit_before["per_class_accuracy"][label])
        for label in old_classes
    )
    fit_floor_non_degradation = (
        float(fit_after["class_floor_accuracy"]) + tolerance
        >= float(fit_before["class_floor_accuracy"])
    )
    old_support_non_degradation = bool(
        fit_classwise_non_degradation and fit_floor_non_degradation
    )

    before_old = legacy._metric_block(
        labels[held & old], before_predictions, old_classes
    )
    after_old = legacy._metric_block(
        labels[held & old], after_old_predictions, old_classes
    )
    after_new = legacy._metric_block(
        labels[held & new], after_new_predictions, new_classes
    )
    resource = dict(after.resource_audit())
    resource.update(
        {
            "old_support_non_degradation_pass": old_support_non_degradation,
            "old_score_columns_bitwise_unchanged_after_registration": True,
            "complete_loss_trace": list(before_fit.training_trace)
            + list(after_fit.training_trace),
            "query_features_used_for_fit": False,
            "query_labels_used_for_fit": False,
            "source_sample_access": False,
            "clean_sample_access": False,
        }
    )
    return {
        "candidate_id": candidate_id,
        "held_ranks": list(held_ranks),
        "fit_k_shot": 8,
        "before_old": before_old,
        "after_old": after_old,
        "after_new": after_new,
        "H_old_new": legacy._harmonic(
            float(after_old["overall_accuracy"]),
            float(after_new["overall_accuracy"]),
        ),
        "forgetting": float(
            before_old["overall_accuracy"] - after_old["overall_accuracy"]
        ),
        "joint_floor": float(
            min(
                float(after_old["class_floor_accuracy"]),
                float(after_new["class_floor_accuracy"]),
            )
        ),
        "old_score_columns_bitwise_unchanged": True,
        "old_prefix_sha256_before": before.old_prefix_sha256,
        "old_prefix_sha256_after": after.old_prefix_sha256,
        "shared_sha256_before": before.shared_sha256,
        "shared_sha256_after": after.shared_sha256,
        "fit_old_before_registration": fit_before,
        "fit_old_after_registration": fit_after,
        "old_support_classwise_non_degradation": fit_classwise_non_degradation,
        "old_support_floor_non_degradation": fit_floor_non_degradation,
        "old_support_non_degradation_pass": old_support_non_degradation,
        "training_trace": list(before_fit.training_trace)
        + list(after_fit.training_trace),
        "geometry_summary": _c3_geometry(after),
        "resource": resource,
    }


def _evaluate_d26_fold(
    rows: Mapping[str, np.ndarray],
    z_id160: np.ndarray,
    fft96: np.ndarray,
    rf32: np.ndarray,
    *,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    held_ranks: tuple[int, int],
    candidate_id: str,
    config: D26CompactDiagConfig,
) -> dict[str, Any]:
    labels = np.asarray(rows["labels"]).astype(str)
    ranks = np.asarray(rows["ranks"], dtype=np.int64)
    held = np.isin(ranks, np.asarray(held_ranks, dtype=np.int64))
    train = ~held
    old = np.isin(labels, np.asarray(old_classes))
    new = np.isin(labels, np.asarray(new_classes))
    if (
        int(np.sum(train & old)) != 8 * len(old_classes)
        or int(np.sum(train & new)) != 8 * len(new_classes)
        or int(np.sum(held & old)) != 2 * len(old_classes)
        or int(np.sum(held & new)) != 2 * len(new_classes)
    ):
        raise D25RunnerError("D26 leave-two-out class symmetry drift")
    features = build_concat288(z_id160, fft96, rf32)
    fit_old_features = features[train & old]
    fit_old_labels = labels[train & old]
    before_fit = fit_stage2b_compact_diag(
        fit_old_features,
        fit_old_labels,
        old_classes,
        config=config,
    )
    before = before_fit.state
    after_fit = append_stage2c_d26(
        before,
        features[train & new],
        labels[train & new],
        new_classes,
        fit_old_features,
        fit_old_labels,
    )
    after = after_fit.state
    if before.classes != old_classes or after.classes != old_classes + new_classes:
        raise D25RunnerError("D26 registered class order drift")
    if (
        before.old_lock_sha256 != after.old_lock_sha256
        or before.log_diag.tobytes() != after.log_diag.tobytes()
        or before.weights.tobytes()
        != after.weights[: len(old_classes)].tobytes()
    ):
        raise D25RunnerError("D26 shared diagonal or old weight prefix changed")

    held_old = features[held & old]
    held_new = features[held & new]
    before_old_scores = score_all_d26(before, held_old)
    after_old_scores = score_all_d26(after, held_old)
    if not np.array_equal(
        before_old_scores, after_old_scores[:, : len(old_classes)]
    ):
        raise D25RunnerError("D26 old raw score prefix changed after registration")
    before_predictions = predict_all_d26(before, held_old).astype(str).tolist()
    after_old_predictions = predict_all_d26(after, held_old).astype(str).tolist()
    after_new_predictions = predict_all_d26(after, held_new).astype(str).tolist()

    fit_before_predictions = (
        predict_all_d26(before, fit_old_features).astype(str).tolist()
    )
    fit_after_predictions = (
        predict_all_d26(after, fit_old_features).astype(str).tolist()
    )
    fit_before = legacy._metric_block(
        fit_old_labels, fit_before_predictions, old_classes
    )
    fit_after = legacy._metric_block(
        fit_old_labels, fit_after_predictions, old_classes
    )
    tolerance = 1.0e-12
    fit_classwise_non_degradation = all(
        float(fit_after["per_class_accuracy"][label]) + tolerance
        >= float(fit_before["per_class_accuracy"][label])
        for label in old_classes
    )
    fit_floor_non_degradation = (
        float(fit_after["class_floor_accuracy"]) + tolerance
        >= float(fit_before["class_floor_accuracy"])
    )
    old_support_non_degradation = bool(
        fit_classwise_non_degradation and fit_floor_non_degradation
    )
    before_old = legacy._metric_block(
        labels[held & old], before_predictions, old_classes
    )
    after_old = legacy._metric_block(
        labels[held & old], after_old_predictions, old_classes
    )
    after_new = legacy._metric_block(
        labels[held & new], after_new_predictions, new_classes
    )
    training_trace = list(before_fit.loss_trace) + list(after_fit.loss_trace)
    bias_audit = json.loads(after.bias_audit_json)
    resource = dict(after.resource_audit())
    resource.update(
        {
            "old_support_non_degradation_pass": old_support_non_degradation,
            "old_score_columns_bitwise_unchanged_after_registration": True,
            "complete_loss_trace": training_trace,
            "new_group_bias": float(after.new_group_bias),
            "new_class_biases": _d26_new_class_biases(after),
            "new_group_bias_support_only_audit": bias_audit,
            "query_features_used_for_fit": False,
            "query_labels_used_for_fit": False,
            "source_sample_access": False,
            "clean_sample_access": False,
        }
    )
    return {
        "candidate_id": candidate_id,
        "held_ranks": list(held_ranks),
        "fit_k_shot": 8,
        "before_old": before_old,
        "after_old": after_old,
        "after_new": after_new,
        "H_old_new": legacy._harmonic(
            float(after_old["overall_accuracy"]),
            float(after_new["overall_accuracy"]),
        ),
        "forgetting": float(
            before_old["overall_accuracy"] - after_old["overall_accuracy"]
        ),
        "joint_floor": float(
            min(
                float(after_old["class_floor_accuracy"]),
                float(after_new["class_floor_accuracy"]),
            )
        ),
        "old_score_columns_bitwise_unchanged": True,
        "old_prefix_sha256_before": before.old_lock_sha256,
        "old_prefix_sha256_after": after.old_lock_sha256,
        "fit_old_before_registration": fit_before,
        "fit_old_after_registration": fit_after,
        "old_support_classwise_non_degradation": fit_classwise_non_degradation,
        "old_support_floor_non_degradation": fit_floor_non_degradation,
        "old_support_non_degradation_pass": old_support_non_degradation,
        "new_group_bias": float(after.new_group_bias),
        "new_class_biases": _d26_new_class_biases(after),
        "new_group_bias_support_only_audit": bias_audit,
        "training_trace": training_trace,
        "geometry_summary": _d26_geometry(after),
        "resource": resource,
    }


def _dense_fold_shot_ranks(
    labels: np.ndarray, ranks: np.ndarray, classes: Sequence[str]
) -> np.ndarray:
    """Map held-rank support back to a class-symmetric dense 0..K-1 rank."""

    result = np.empty(len(labels), dtype=np.int64)
    expected: tuple[int, ...] | None = None
    for class_name in classes:
        selected = labels == class_name
        observed = tuple(sorted(int(value) for value in ranks[selected].tolist()))
        if expected is None:
            expected = observed
        elif observed != expected:
            raise D25RunnerError("D28 support shot-rank symmetry drift")
        mapping = {value: index for index, value in enumerate(observed)}
        result[selected] = np.asarray(
            [mapping[int(value)] for value in ranks[selected].tolist()],
            dtype=np.int64,
        )
    return result


def _require_d38_development_cell(
    before_manifest: Mapping[str, Any], after_manifest: Mapping[str, Any]
) -> None:
    """Fail closed before opening support outside the preregistered D38 cell."""

    old_classes = legacy._registered_handles(before_manifest)
    all_classes = legacy._registered_handles(after_manifest)
    if (
        str(before_manifest.get("receiver")) != D38_DEVELOPMENT_RECEIVER
        or int(before_manifest.get("seed", -1)) != D38_DEVELOPMENT_SEED
        or int(before_manifest.get("k_shot", -1)) != 10
        or all_classes[: len(old_classes)] != old_classes
        or len(all_classes) - len(old_classes) != D38_DEVELOPMENT_NEW_CLASS_COUNT
    ):
        raise D25RunnerError(
            "D38 preregistered development cell must be receiver 20-1, "
            "seed 713101, K10, new5"
        )


def _require_d39_development_cell(
    before_manifest: Mapping[str, Any], after_manifest: Mapping[str, Any]
) -> None:
    """Fail closed before support opening outside the locked D39 cell."""

    old_classes = legacy._registered_handles(before_manifest)
    all_classes = legacy._registered_handles(after_manifest)
    if (
        str(before_manifest.get("receiver")) != D39_DEVELOPMENT_RECEIVER
        or int(before_manifest.get("seed", -1)) != D39_DEVELOPMENT_SEED
        or int(before_manifest.get("k_shot", -1)) != 10
        or all_classes[: len(old_classes)] != old_classes
        or len(all_classes) - len(old_classes) != D39_DEVELOPMENT_NEW_CLASS_COUNT
    ):
        raise D25RunnerError(
            "D39 preregistered development cell must be receiver 20-1, "
            "seed 713101, K10, new5"
        )


def _require_d40_development_cell(
    before_manifest: Mapping[str, Any], after_manifest: Mapping[str, Any]
) -> None:
    """Fail closed before support opening outside the locked D40 cell."""

    old_classes = legacy._registered_handles(before_manifest)
    all_classes = legacy._registered_handles(after_manifest)
    if (
        str(before_manifest.get("receiver")) != D40_DEVELOPMENT_RECEIVER
        or int(before_manifest.get("seed", -1)) != D40_DEVELOPMENT_SEED
        or int(before_manifest.get("k_shot", -1)) != 10
        or all_classes[: len(old_classes)] != old_classes
        or len(all_classes) - len(old_classes) != D40_DEVELOPMENT_NEW_CLASS_COUNT
    ):
        raise D25RunnerError(
            "D40 preregistered development cell must be receiver 20-1, "
            "seed 713101, K10, new5"
        )


def _evaluate_d28_fold(
    rows: Mapping[str, np.ndarray],
    z_id160: np.ndarray,
    fft96: np.ndarray,
    rf32: np.ndarray,
    *,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    held_ranks: tuple[int, int],
    candidate_id: str,
    config: D28CandidateConfig,
) -> dict[str, Any]:
    labels = np.asarray(rows["labels"]).astype(str)
    ranks = np.asarray(rows["ranks"], dtype=np.int64)
    held = np.isin(ranks, np.asarray(held_ranks, dtype=np.int64))
    train = ~held
    old = np.isin(labels, np.asarray(old_classes))
    new = np.isin(labels, np.asarray(new_classes))
    all_classes = old_classes + new_classes
    if (
        int(np.sum(train & old)) != 8 * len(old_classes)
        or int(np.sum(train & new)) != 8 * len(new_classes)
        or int(np.sum(held & old)) != 2 * len(old_classes)
        or int(np.sum(held & new)) != 2 * len(new_classes)
    ):
        raise D25RunnerError("D28 leave-two-out class symmetry drift")
    features = build_concat288(z_id160, fft96, rf32)
    fit_old_features = features[train & old]
    fit_old_labels = labels[train & old]
    before_fit = fit_stage2b_compact_diag(
        fit_old_features, fit_old_labels, old_classes, config=config.base
    )
    before = before_fit.state
    after_fit = append_stage2c_d26(
        before,
        features[train & new],
        labels[train & new],
        new_classes,
        fit_old_features,
        fit_old_labels,
    )
    after = after_fit.state
    if before.classes != old_classes or after.classes != all_classes:
        raise D25RunnerError("D28 registered class order drift")
    if (
        before.old_lock_sha256 != after.old_lock_sha256
        or before.log_diag.tobytes() != after.log_diag.tobytes()
        or before.weights.tobytes()
        != after.weights[: len(old_classes)].tobytes()
    ):
        raise D25RunnerError("D28 base mutated D27 frozen old state")

    train_labels = labels[train]
    train_ranks = _dense_fold_shot_ranks(
        train_labels, ranks[train], all_classes
    )
    train_scores = score_all_d26(after, features[train])
    gate = fit_support_evidence_gate(
        train_scores,
        train_labels,
        train_ranks,
        all_classes,
        len(old_classes),
        config=config.gate,
    )

    held_old = features[held & old]
    held_new = features[held & new]
    before_predictions = predict_all_d26(before, held_old).astype(str).tolist()
    held_old_raw = score_all_d26(after, held_old)
    held_new_raw = score_all_d26(after, held_new)
    held_old_adjusted = apply_support_evidence_gate(gate, held_old_raw)
    held_new_adjusted = apply_support_evidence_gate(gate, held_new_raw)
    if not np.array_equal(
        held_old_adjusted[:, : len(old_classes)],
        held_old_raw[:, : len(old_classes)],
    ):
        raise D25RunnerError("D28 gate changed held old score columns")
    after_old_predictions = predict_with_support_evidence_gate(
        gate, held_old_raw
    ).astype(str).tolist()
    after_new_predictions = predict_with_support_evidence_gate(
        gate, held_new_raw
    ).astype(str).tolist()

    fit_before_predictions = (
        predict_all_d26(before, fit_old_features).astype(str).tolist()
    )
    fit_old_raw = score_all_d26(after, fit_old_features)
    fit_old_adjusted = apply_support_evidence_gate(gate, fit_old_raw)
    if not np.array_equal(
        fit_old_adjusted[:, : len(old_classes)],
        fit_old_raw[:, : len(old_classes)],
    ):
        raise D25RunnerError("D28 gate changed fit-old score columns")
    fit_after_predictions = predict_with_support_evidence_gate(
        gate, fit_old_raw
    ).astype(str).tolist()
    fit_before = legacy._metric_block(
        fit_old_labels, fit_before_predictions, old_classes
    )
    fit_after = legacy._metric_block(
        fit_old_labels, fit_after_predictions, old_classes
    )
    tolerance = 1.0e-12
    fit_classwise_non_degradation = all(
        float(fit_after["per_class_accuracy"][label]) + tolerance
        >= float(fit_before["per_class_accuracy"][label])
        for label in old_classes
    )
    fit_floor_non_degradation = (
        float(fit_after["class_floor_accuracy"]) + tolerance
        >= float(fit_before["class_floor_accuracy"])
    )
    old_support_non_degradation = bool(
        fit_classwise_non_degradation and fit_floor_non_degradation
    )
    before_old = legacy._metric_block(
        labels[held & old], before_predictions, old_classes
    )
    after_old = legacy._metric_block(
        labels[held & old], after_old_predictions, old_classes
    )
    after_new = legacy._metric_block(
        labels[held & new], after_new_predictions, new_classes
    )
    training_trace = list(before_fit.loss_trace) + list(after_fit.loss_trace)
    gate_audit = json.loads(gate.audit_json)
    gate_resource = dict(gate.resource_audit())
    resource = dict(after.resource_audit())
    base_state_bytes = int(resource["persistent_state_bytes"])
    base_query_macs = int(resource["estimated_macs_per_query"])
    gate_state_bytes = int(gate_resource["deployable_predictor_state_bytes"])
    gate_query_macs = int(gate_resource["estimated_gate_macs_per_query"])
    resource.update(
        {
            "schema": "cvs.phase2.d28_combined_resource.v1",
            "base_d27_resource": dict(after.resource_audit()),
            "gate_resource": gate_resource,
            "gate_enabled": bool(gate.enabled),
            "gate_fit_audit": gate_audit,
            "gate_fitted_parameter_count": int(
                gate_resource["fitted_parameter_count"]
            ),
            "active_adaptation_parameter_count": int(
                resource["peak_trainable_parameters"]
                + gate_resource["fitted_parameter_count"]
            ),
            "persistent_state_bytes": base_state_bytes + gate_state_bytes,
            "external_gate_evidence_audit_bytes": int(
                gate_resource["external_evidence_audit_bytes"]
            ),
            "persistent_state_cap_pass": (
                base_state_bytes + gate_state_bytes <= 256 * 1024
            ),
            "estimated_macs_per_query": base_query_macs + gate_query_macs,
            "old_support_non_degradation_pass": old_support_non_degradation,
            "old_score_columns_bitwise_unchanged_after_registration": True,
            "complete_loss_trace": training_trace,
            "new_group_bias": float(after.new_group_bias),
            "new_class_biases": _d26_new_class_biases(after),
            "new_group_bias_support_only_audit": json.loads(after.bias_audit_json),
            "query_features_used_for_fit": False,
            "query_labels_used_for_fit": False,
            "query_role_oracle_access": False,
            "query_true_batch_class_count_access": False,
            "query_class_quota_access": False,
            "query_batch_global_assignment": False,
            "source_sample_access": False,
            "clean_sample_access": False,
        }
    )
    geometry = _d26_geometry(after)
    geometry["schema"] = "cvs.phase2.d28_evidence_gate_geometry.v1"
    geometry["gate_enabled"] = bool(gate.enabled)
    geometry["gate_fit_audit"] = gate_audit
    return {
        "candidate_id": candidate_id,
        "held_ranks": list(held_ranks),
        "fit_k_shot": 8,
        "before_old": before_old,
        "after_old": after_old,
        "after_new": after_new,
        "H_old_new": legacy._harmonic(
            float(after_old["overall_accuracy"]),
            float(after_new["overall_accuracy"]),
        ),
        "forgetting": float(
            before_old["overall_accuracy"] - after_old["overall_accuracy"]
        ),
        "joint_floor": float(
            min(
                float(after_old["class_floor_accuracy"]),
                float(after_new["class_floor_accuracy"]),
            )
        ),
        "old_score_columns_bitwise_unchanged": True,
        "old_prefix_sha256_before": before.old_lock_sha256,
        "old_prefix_sha256_after": after.old_lock_sha256,
        "fit_old_before_registration": fit_before,
        "fit_old_after_registration": fit_after,
        "old_support_classwise_non_degradation": fit_classwise_non_degradation,
        "old_support_floor_non_degradation": fit_floor_non_degradation,
        "old_support_non_degradation_pass": old_support_non_degradation,
        "new_group_bias": float(after.new_group_bias),
        "new_class_biases": _d26_new_class_biases(after),
        "new_group_bias_support_only_audit": json.loads(after.bias_audit_json),
        "gate_enabled": bool(gate.enabled),
        "gate_fit_audit": gate_audit,
        "training_trace": training_trace,
        "geometry_summary": geometry,
        "resource": resource,
    }


def _evaluate_d29_fold(
    rows: Mapping[str, np.ndarray],
    z_id160: np.ndarray,
    fft96: np.ndarray,
    rf32: np.ndarray,
    *,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    held_ranks: tuple[int, int],
    candidate_id: str,
    config: D29CandidateConfig,
) -> dict[str, Any]:
    """Evaluate D29 using only the outer-fold training support rows."""

    labels = np.asarray(rows["labels"]).astype(str)
    ranks = np.asarray(rows["ranks"], dtype=np.int64)
    held = np.isin(ranks, np.asarray(held_ranks, dtype=np.int64))
    train = ~held
    old = np.isin(labels, np.asarray(old_classes))
    new = np.isin(labels, np.asarray(new_classes))
    all_classes = old_classes + new_classes
    if (
        int(np.sum(train & old)) != 8 * len(old_classes)
        or int(np.sum(train & new)) != 8 * len(new_classes)
        or int(np.sum(held & old)) != 2 * len(old_classes)
        or int(np.sum(held & new)) != 2 * len(new_classes)
    ):
        raise D25RunnerError("D29 leave-two-out class symmetry drift")

    features = build_concat288(z_id160, fft96, rf32)
    fit_old_features = features[train & old]
    fit_old_labels = labels[train & old]
    before_fit = fit_stage2b_compact_diag(
        fit_old_features, fit_old_labels, old_classes, config=config.base
    )
    before = before_fit.state
    after_fit = append_stage2c_d26(
        before,
        features[train & new],
        labels[train & new],
        new_classes,
        fit_old_features,
        fit_old_labels,
    )
    after = after_fit.state
    if before.classes != old_classes or after.classes != all_classes:
        raise D25RunnerError("D29 registered class order drift")
    if (
        before.old_lock_sha256 != after.old_lock_sha256
        or before.log_diag.tobytes() != after.log_diag.tobytes()
        or before.weights.tobytes()
        != after.weights[: len(old_classes)].tobytes()
    ):
        raise D25RunnerError("D29 base mutated D27 frozen old state")

    train_labels = labels[train]
    train_ranks = _dense_fold_shot_ranks(
        train_labels, ranks[train], all_classes
    )
    train_scores = score_all_d26(after, features[train])
    release = fit_classwise_safe_release(
        train_scores,
        train_labels,
        train_ranks,
        all_classes,
        len(old_classes),
        config=config.release,
    )

    held_old = features[held & old]
    held_new = features[held & new]
    before_predictions = predict_all_d26(before, held_old).astype(str).tolist()
    held_old_raw = score_all_d26(after, held_old)
    held_new_raw = score_all_d26(after, held_new)
    held_old_adjusted = apply_classwise_safe_release(release, held_old_raw)
    held_new_adjusted = apply_classwise_safe_release(release, held_new_raw)
    if not np.array_equal(
        held_old_adjusted[:, : len(old_classes)],
        held_old_raw[:, : len(old_classes)],
    ):
        raise D25RunnerError("D29 release changed held old score columns")
    after_old_predictions = predict_with_classwise_safe_release(
        release, held_old_raw
    ).astype(str).tolist()
    after_new_predictions = predict_with_classwise_safe_release(
        release, held_new_raw
    ).astype(str).tolist()

    fit_before_predictions = (
        predict_all_d26(before, fit_old_features).astype(str).tolist()
    )
    fit_old_raw = score_all_d26(after, fit_old_features)
    fit_old_adjusted = apply_classwise_safe_release(release, fit_old_raw)
    if not np.array_equal(
        fit_old_adjusted[:, : len(old_classes)],
        fit_old_raw[:, : len(old_classes)],
    ):
        raise D25RunnerError("D29 release changed fit-old score columns")
    fit_after_predictions = predict_with_classwise_safe_release(
        release, fit_old_raw
    ).astype(str).tolist()
    fit_before = legacy._metric_block(
        fit_old_labels, fit_before_predictions, old_classes
    )
    fit_after = legacy._metric_block(
        fit_old_labels, fit_after_predictions, old_classes
    )
    tolerance = 1.0e-12
    fit_classwise_non_degradation = all(
        float(fit_after["per_class_accuracy"][label]) + tolerance
        >= float(fit_before["per_class_accuracy"][label])
        for label in old_classes
    )
    fit_floor_non_degradation = (
        float(fit_after["class_floor_accuracy"]) + tolerance
        >= float(fit_before["class_floor_accuracy"])
    )
    old_support_non_degradation = bool(
        fit_classwise_non_degradation and fit_floor_non_degradation
    )
    before_old = legacy._metric_block(
        labels[held & old], before_predictions, old_classes
    )
    after_old = legacy._metric_block(
        labels[held & old], after_old_predictions, old_classes
    )
    after_new = legacy._metric_block(
        labels[held & new], after_new_predictions, new_classes
    )
    training_trace = list(before_fit.loss_trace) + list(after_fit.loss_trace)
    release_audit = json.loads(release.audit_json)
    release_resource = dict(release.resource_audit())
    base_resource = dict(after.resource_audit())
    base_state_bytes = int(base_resource["persistent_state_bytes"])
    base_query_macs = int(base_resource["estimated_macs_per_query"])
    release_state_bytes = int(
        release_resource["deployable_predictor_state_bytes"]
    )
    release_ops = int(
        release_resource["estimated_release_scalar_ops_per_query"]
    )
    resource = {
        **base_resource,
        "schema": "cvs.phase2.d29_combined_resource.v1",
        "base_d27_resource": base_resource,
        "release_resource": release_resource,
        "release_enabled": bool(release.enabled),
        "release_fit_audit": release_audit,
        "release_fitted_parameter_count": int(
            release_resource["fitted_parameter_count"]
        ),
        "active_adaptation_parameter_count": int(
            base_resource["peak_trainable_parameters"]
            + release_resource["fitted_parameter_count"]
        ),
        "persistent_state_bytes": base_state_bytes + release_state_bytes,
        "external_release_evidence_audit_bytes": int(
            release_resource["external_evidence_audit_bytes"]
        ),
        "persistent_state_cap_pass": (
            base_state_bytes + release_state_bytes <= 256 * 1024
        ),
        "estimated_macs_per_query": base_query_macs,
        "estimated_row_local_scalar_ops_per_query": release_ops,
        "old_support_non_degradation_pass": old_support_non_degradation,
        "old_score_columns_bitwise_unchanged_after_registration": True,
        "complete_loss_trace": training_trace,
        "new_group_bias": float(after.new_group_bias),
        "new_class_biases": _d26_new_class_biases(after),
        "new_group_bias_support_only_audit": json.loads(after.bias_audit_json),
        "query_features_used_for_fit": False,
        "query_labels_used_for_fit": False,
        "query_role_oracle_access": False,
        "query_true_batch_class_count_access": False,
        "query_class_quota_access": False,
        "query_batch_global_assignment": False,
        "source_sample_access": False,
        "clean_sample_access": False,
    }
    geometry = _d26_geometry(after)
    geometry["schema"] = "cvs.phase2.d29_pcsr_geometry.v1"
    geometry["release_enabled"] = bool(release.enabled)
    geometry["release_fit_audit"] = release_audit
    return {
        "candidate_id": candidate_id,
        "held_ranks": list(held_ranks),
        "fit_k_shot": 8,
        "before_old": before_old,
        "after_old": after_old,
        "after_new": after_new,
        "H_old_new": legacy._harmonic(
            float(after_old["overall_accuracy"]),
            float(after_new["overall_accuracy"]),
        ),
        "forgetting": float(
            before_old["overall_accuracy"] - after_old["overall_accuracy"]
        ),
        "joint_floor": float(
            min(
                float(after_old["class_floor_accuracy"]),
                float(after_new["class_floor_accuracy"]),
            )
        ),
        "old_score_columns_bitwise_unchanged": True,
        "old_prefix_sha256_before": before.old_lock_sha256,
        "old_prefix_sha256_after": after.old_lock_sha256,
        "fit_old_before_registration": fit_before,
        "fit_old_after_registration": fit_after,
        "old_support_classwise_non_degradation": fit_classwise_non_degradation,
        "old_support_floor_non_degradation": fit_floor_non_degradation,
        "old_support_non_degradation_pass": old_support_non_degradation,
        "new_group_bias": float(after.new_group_bias),
        "new_class_biases": _d26_new_class_biases(after),
        "new_group_bias_support_only_audit": json.loads(after.bias_audit_json),
        "release_enabled": bool(release.enabled),
        "release_fit_audit": release_audit,
        "training_trace": training_trace,
        "geometry_summary": geometry,
        "resource": resource,
    }


def _d30_rerank_matrix(
    dali_state: Any,
    base_scores: np.ndarray,
    z_id160: np.ndarray,
    direct_logits: np.ndarray | None,
) -> np.ndarray:
    scores = np.asarray(base_scores, dtype=np.float32)
    z_rows = np.asarray(z_id160, dtype=np.float32)
    direct = (
        None
        if direct_logits is None
        else np.asarray(direct_logits, dtype=np.float32)
    )
    if len(scores) != len(z_rows) or (direct is not None and len(direct) != len(scores)):
        raise D25RunnerError("D30 DALI row alignment drift")
    return np.stack(
        [
            rerank_old_scores_dali(
                dali_state,
                scores[index],
                z_rows[index],
                None if direct is None else direct[index],
            )
            for index in range(len(scores))
        ],
        axis=0,
    ).astype(np.float32)


def _d30_old_support_gate(
    raw_scores: np.ndarray,
    reranked_scores: np.ndarray,
    labels: np.ndarray,
    classes: tuple[str, ...],
) -> tuple[bool, dict[str, Any]]:
    raw_predictions = np.asarray(classes)[np.argmax(raw_scores, axis=1)]
    reranked_predictions = np.asarray(classes)[np.argmax(reranked_scores, axis=1)]
    raw_metric = legacy._metric_block(labels, raw_predictions, classes)
    reranked_metric = legacy._metric_block(labels, reranked_predictions, classes)
    tolerance = 1.0e-12
    overall_pass = (
        float(reranked_metric["overall_accuracy"]) + tolerance
        >= float(raw_metric["overall_accuracy"])
    )
    floor_pass = (
        float(reranked_metric["class_floor_accuracy"]) + tolerance
        >= float(raw_metric["class_floor_accuracy"])
    )
    classwise_pass = all(
        float(reranked_metric["per_class_accuracy"][name]) + tolerance
        >= float(raw_metric["per_class_accuracy"][name])
        for name in classes
    )
    return bool(overall_pass and floor_pass and classwise_pass), {
        "schema": "cvs.phase2.d30_dali_old_support_gate.v1",
        "selection_rows": "outer_train_old_support_only",
        "raw": raw_metric,
        "reranked": reranked_metric,
        "overall_non_degradation": overall_pass,
        "floor_non_degradation": floor_pass,
        "per_class_non_degradation": classwise_pass,
        "enabled": bool(overall_pass and floor_pass and classwise_pass),
        "atomic_passthrough_on_failure": True,
        "held_rows_used": 0,
        "query_rows_used": 0,
    }


def _d30_enable_dali(k_shot: int, support_gate_pass: bool) -> bool:
    """K1 is an exact base-head passthrough; other K use the support gate."""

    if (
        isinstance(k_shot, (bool, np.bool_))
        or not isinstance(k_shot, (int, np.integer))
        or int(k_shot) < 1
    ):
        raise D25RunnerError("D30 DALI K-shot drift")
    return bool(int(k_shot) > 1 and support_gate_pass)


def _d31_confusion_audit(
    scores: np.ndarray,
    labels: np.ndarray,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
) -> dict[str, Any]:
    classes = old_classes + new_classes
    values = np.asarray(scores, dtype=np.float32)
    truth = np.asarray(labels).astype(str)
    predictions = np.asarray(classes)[np.argmax(values, axis=1)]
    old_mask = np.isin(truth, np.asarray(old_classes))
    new_mask = ~old_mask
    predicted_old = np.isin(predictions, np.asarray(old_classes))
    correct = predictions == truth
    return {
        "schema": "cvs.phase2.d31_group_confusion.v1",
        "sample_count": int(len(truth)),
        "old_to_new": int(np.sum(old_mask & ~predicted_old)),
        "old_to_wrong_old": int(np.sum(old_mask & predicted_old & ~correct)),
        "new_to_old": int(np.sum(new_mask & predicted_old)),
        "new_to_wrong_new": int(np.sum(new_mask & ~predicted_old & ~correct)),
        "new_correct": int(np.sum(new_mask & correct)),
        "per_new_class": {
            name: {
                "sample_count": int(np.sum(truth == name)),
                "new_to_old": int(np.sum((truth == name) & predicted_old)),
                "new_to_wrong_new": int(
                    np.sum((truth == name) & ~predicted_old & ~correct)
                ),
                "correct": int(np.sum((truth == name) & correct)),
            }
            for name in new_classes
        },
    }


def _d31_dali_state_accounting(dali_state: Any) -> dict[str, Any]:
    resource = dict(dali_state.resource_audit())
    old_count = int(dali_state.old_class_count)
    component = dali_state.component
    medoid_q_bytes = int(old_count * 160 * np.dtype(np.int8).itemsize)
    medoid_scale_bytes = int(old_count * np.dtype(np.float32).itemsize)
    medoid_radius_bytes = int(old_count * np.dtype(np.float32).itemsize)
    medoid_column_index_bytes = int(old_count * np.dtype(np.uint16).itemsize)
    medoid_class_digest_bytes = int(old_count * 32)
    medoid_header_hash_bytes = 128
    medoid_view_bytes = int(
        medoid_q_bytes
        + medoid_scale_bytes
        + medoid_radius_bytes
        + medoid_column_index_bytes
        + medoid_class_digest_bytes
        + medoid_header_hash_bytes
    )
    projected_runtime_bytes = int(
        medoid_view_bytes
        + dali_state.ground_weight_by_old_class.nbytes
        + dali_state.support_margin_q25_by_old_class.nbytes
        + 4 * np.dtype(np.float32).itemsize
        + 32
    )
    return {
        "dali_resource": resource,
        "authorized_full_bundle_state_bytes": int(component.state_bytes),
        "actual_current_dali_state_bytes": int(resource["persistent_state_bytes"]),
        "selected_medoid_int8_view_bytes": medoid_view_bytes,
        "selected_medoid_int8_anchor_bytes": medoid_q_bytes,
        "selected_medoid_fp32_scale_bytes": medoid_scale_bytes,
        "selected_medoid_fp32_radius_bytes": medoid_radius_bytes,
        "selected_medoid_u16_column_index_bytes": medoid_column_index_bytes,
        "selected_medoid_class_digest_bytes": medoid_class_digest_bytes,
        "selected_medoid_header_hash_bytes": medoid_header_hash_bytes,
        "selected_medoid_payload_schema": (
            "fixed_phase1_int8_anchor_scale_radius_column_digest_v1"
        ),
        "projected_slim_dali_runtime_bytes": projected_runtime_bytes,
        "slim_runtime_projection_only": True,
        "current_core_materializes_full_authorized_component": True,
        "full_authorized_bundle_must_remain_resident_or_sealed_accessible": True,
        "fixed_medoid_domain_index": int(dali_state.fixed_medoid_domain_index),
    }


def _fit_d31_route(
    component: Any,
    features: np.ndarray,
    z_id160: np.ndarray,
    direct_logits: np.ndarray,
    labels: np.ndarray,
    old_mask: np.ndarray,
    new_mask: np.ndarray,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    config: D31CandidateConfig,
) -> dict[str, Any]:
    before_fit = fit_stage2b_compact_diag(
        features[old_mask], labels[old_mask], old_classes, config=config.base
    )
    before = before_fit.state
    stage2c_fit = append_stage2c_all_registered_new_suffix(
        before,
        features[new_mask],
        labels[new_mask],
        new_classes,
        features[old_mask],
        labels[old_mask],
        config=config.stage2c,
    )
    after = stage2c_fit.state
    if before.classes != old_classes or after.classes != old_classes + new_classes:
        raise D25RunnerError("D31 registered class order drift")
    if (
        before.log_diag.tobytes() != after.log_diag.tobytes()
        or before.weights.tobytes()
        != after.weights[: len(old_classes)].tobytes()
        or before.old_lock_sha256 != after.base_old_lock_sha256
    ):
        raise D25RunnerError("D31 mutated frozen Stage2-B state")
    dali_old = fit_old_dali(
        component,
        z_id160[old_mask],
        labels[old_mask],
        direct_logits[old_mask],
        config=config.dali,
    )
    dali_state = register_new_dali(
        dali_old,
        z_id160[new_mask],
        labels[new_mask],
        registered_classes=new_classes,
    )
    raw_scores = score_all_d31(after, features)
    raw_old = raw_scores[old_mask]
    reranked_old = _d30_rerank_matrix(
        dali_state, raw_old, z_id160[old_mask], direct_logits[old_mask]
    )
    dali_gate_pass, dali_gate = _d30_old_support_gate(
        raw_old, reranked_old, labels[old_mask], old_classes
    )
    dali_enabled = bool(dali_gate_pass)
    dali_gate.update(
        {
            "schema": "cvs.phase2.d31_dali_old_support_gate.v1",
            "enabled": dali_enabled,
            "selection_rows": "fit_old_support_only",
        }
    )
    adjusted_scores = (
        _d30_rerank_matrix(dali_state, raw_scores, z_id160, direct_logits)
        if dali_enabled
        else raw_scores.copy()
    )
    return {
        "before_fit": before_fit,
        "before": before,
        "stage2c_fit": stage2c_fit,
        "after": after,
        "dali_state": dali_state,
        "dali_enabled": dali_enabled,
        "dali_gate": dali_gate,
        "raw_scores": raw_scores,
        "adjusted_scores": adjusted_scores,
    }


def _fit_d32_route(
    component: Any,
    features: np.ndarray,
    z_id160: np.ndarray,
    direct_logits: np.ndarray,
    labels: np.ndarray,
    old_mask: np.ndarray,
    new_mask: np.ndarray,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    config: D32CandidateConfig,
) -> dict[str, Any]:
    """Fit D32 from registered support only and atomically gate DALI."""

    before_fit = fit_stage2b_compact_diag(
        features[old_mask], labels[old_mask], old_classes, config=config.base
    )
    before = before_fit.state
    stage2c_fit = append_stage2c_inloop_safe_cap_suffix(
        before,
        features[new_mask],
        labels[new_mask],
        new_classes,
        features[old_mask],
        labels[old_mask],
        config=config.stage2c,
    )
    after = stage2c_fit.state
    if before.classes != old_classes or after.classes != old_classes + new_classes:
        raise D25RunnerError("D32 registered class order drift")
    if (
        before.log_diag.tobytes() != after.log_diag.tobytes()
        or before.weights.tobytes()
        != after.weights[: len(old_classes)].tobytes()
        or before.old_lock_sha256 != after.base_old_lock_sha256
    ):
        raise D25RunnerError("D32 mutated frozen Stage2-B state")
    dali_old = fit_old_dali(
        component,
        z_id160[old_mask],
        labels[old_mask],
        direct_logits[old_mask],
        config=config.dali,
    )
    dali_state = register_new_dali(
        dali_old,
        z_id160[new_mask],
        labels[new_mask],
        registered_classes=new_classes,
    )
    raw_scores = score_all_d32(after, features)
    raw_old = raw_scores[old_mask]
    reranked_old = _d30_rerank_matrix(
        dali_state, raw_old, z_id160[old_mask], direct_logits[old_mask]
    )
    dali_gate_pass, dali_gate = _d30_old_support_gate(
        raw_old, reranked_old, labels[old_mask], old_classes
    )
    dali_enabled = _d30_enable_dali(dali_state.k_shot, dali_gate_pass)
    dali_gate.update(
        {
            "schema": "cvs.phase2.d32_dali_old_support_gate.v1",
            "enabled": bool(dali_enabled),
            "selection_rows": "fit_old_support_only",
            "k_shot": int(dali_state.k_shot),
            "k1_exact_base_head_passthrough": bool(int(dali_state.k_shot) == 1),
        }
    )
    adjusted_scores = (
        _d30_rerank_matrix(dali_state, raw_scores, z_id160, direct_logits)
        if dali_enabled
        else raw_scores.copy()
    )
    return {
        "before_fit": before_fit,
        "before": before,
        "stage2c_fit": stage2c_fit,
        "after": after,
        "dali_state": dali_state,
        "dali_enabled": bool(dali_enabled),
        "dali_gate": dali_gate,
        "raw_scores": raw_scores,
        "adjusted_scores": adjusted_scores,
    }


def _fit_d33_route(
    features: np.ndarray,
    labels: np.ndarray,
    old_mask: np.ndarray,
    new_mask: np.ndarray,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    config: D33CandidateConfig,
) -> dict[str, Any]:
    """Fit the locked D33 old solver and symmetric support-only registry."""

    if config.old_solver == "adam15_compact_diag":
        if config.base is None:
            raise D25RunnerError("D33 Adam15 config missing")
        old_fit = fit_stage2b_compact_diag(
            features[old_mask], labels[old_mask], old_classes, config=config.base
        )
        old_state = old_fit.state
        old_log_diag = old_state.log_diag
        old_trace = list(old_fit.loss_trace)
        old_resource = dict(old_state.resource_audit())
    else:
        if config.fisher is None:
            raise D25RunnerError("D33 Fisher config missing")
        old_fit = fit_b3_fisher_closed_form(
            features[old_mask],
            labels[old_mask],
            old_classes,
            config=config.fisher,
        )
        old_state = old_fit.state
        old_log_diag = old_state.log_diag
        old_trace = list(old_fit.solver_trace)
        old_resource = dict(old_fit.resource_audit)
    registration_fit = fit_d33_spherical_registration(
        features[old_mask],
        labels[old_mask],
        old_classes,
        features[new_mask],
        labels[new_mask],
        new_classes,
        old_log_diag,
        config=config.registration,
    )
    after = registration_fit.state
    if after.classes != old_classes + new_classes:
        raise D25RunnerError("D33 registered class order drift")
    if not np.array_equal(np.asarray(old_log_diag), after.log_diag):
        raise D25RunnerError("D33 mutated the locked old diagonal")
    return {
        "old_fit": old_fit,
        "old_state": old_state,
        "old_trace": old_trace,
        "old_resource": old_resource,
        "registration_fit": registration_fit,
        "after": after,
        "registration_trace": list(registration_fit.selection_trace),
        "registration_resource": dict(registration_fit.resource_audit),
    }


def _score_d33_old_stage(fit: Mapping[str, Any], features: np.ndarray) -> np.ndarray:
    old_state = fit["old_state"]
    if isinstance(old_state, D26CompactDiagState):
        return score_all_d26(old_state, features)
    return score_b3_fisher_closed_form(old_state, features)


def _d33_resource(
    fit: Mapping[str, Any], registered_count: int
) -> dict[str, Any]:
    old_resource = dict(fit["old_resource"])
    registration_resource = dict(fit["registration_resource"])
    old_steps = int(old_resource.get("total_optimizer_steps", old_resource.get("optimizer_steps", 0)))
    old_macs = int(old_resource["estimated_adaptation_macs"])
    registration_macs = int(registration_resource["estimated_adaptation_macs"])
    state_bytes = int(registration_resource["persistent_state_bytes"])
    trainable = int(
        old_resource.get(
            "peak_trainable_parameters", old_resource.get("trainable_parameters", 0)
        )
    )
    query_macs = int(registration_resource["estimated_macs_per_query"])
    identity_macs = int(registered_count * 10 * 160)
    complete_trace = list(fit["old_trace"]) + list(fit["registration_trace"])
    return {
        "schema": "cvs.phase2.d33_combined_resource.v1",
        "old_solver": (
            "adam15_compact_diag"
            if isinstance(fit["old_state"], D26CompactDiagState)
            else "b3_fisher_closed_form"
        ),
        "old_solver_resource": old_resource,
        "spherical_registration_resource": registration_resource,
        "peak_trainable_parameters": trainable,
        "trainable_parameter_cap": 80_000,
        "trainable_parameter_cap_pass": trainable <= 80_000,
        "total_optimizer_steps": old_steps,
        "total_adaptation_epochs": old_steps,
        "optimizer_step_cap": 30,
        "optimizer_step_cap_pass": old_steps <= 30,
        "stage2b_adaptation_macs": old_macs,
        "stage2c_adaptation_macs": registration_macs,
        "total_adaptation_macs": old_macs + registration_macs,
        "estimated_adaptation_macs": old_macs + registration_macs,
        "base_head_macs_per_query": query_macs,
        "argmax_scalar_comparisons_per_query": max(0, registered_count - 1),
        "total_post_backbone_macs_per_query": query_macs,
        "estimated_macs_per_query": query_macs,
        "identity_single_qknn_macs_same_registered_count": identity_macs,
        "estimated_score_mac_ratio_vs_identity_single_qknn": float(
            query_macs / identity_macs
        ),
        "persistent_state_bytes": state_bytes,
        "persistent_state_cap_bytes": 256 * 1024,
        "persistent_state_cap_pass": state_bytes <= 256 * 1024,
        "active_predictor_state_bytes": state_bytes,
        "authorized_int8_component_available_in_sealed_bundle": True,
        "actual_int8_component_used_for_prediction": False,
        "int8_predictor_dependency": False,
        "dense_query_graph_bytes": 0,
        "complete_loss_trace": complete_trace,
        "feature_geometry": "b3_auxiliary_dominant_z160_fft96_rf32_v1",
        "query_rows_used_for_fit": 0,
        "query_labels_used_for_fit": False,
        "query_features_used_for_fit": False,
        "query_role_oracle_access": False,
        "query_true_batch_class_count_access": False,
        "query_class_quota_access": False,
        "query_batch_global_assignment": False,
        "clean_sample_access": False,
        "source_sample_access": False,
    }


def _fit_d34_route(
    features: np.ndarray,
    labels: np.ndarray,
    old_mask: np.ndarray,
    new_mask: np.ndarray,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    config: D34CandidateConfig,
) -> dict[str, Any]:
    """Fit FAST once, then build a support-only sparse registration state."""

    old_fit = fit_b3_fisher_closed_form(
        features[old_mask],
        labels[old_mask],
        old_classes,
        config=config.fisher,
    )
    old_state = old_fit.state
    adapted, all_old_prefix = _d34_fast_unit_and_prefix(old_state, features)
    reference_prefix = score_b3_fisher_closed_form(old_state, features)
    if not np.array_equal(all_old_prefix, reference_prefix):
        raise D25RunnerError("D34 FAST prefix implementation is not bitwise exact")
    old_prefix = all_old_prefix[old_mask]
    new_old_prefix = all_old_prefix[new_mask]
    registration_fit = fit_d34_collision_local_registration(
        adapted[old_mask],
        labels[old_mask],
        old_classes,
        old_prefix,
        adapted[new_mask],
        labels[new_mask],
        new_classes,
        new_old_prefix,
        config=config.registration,
    )
    if tuple(registration_fit.state.classes) != old_classes + new_classes:
        raise D25RunnerError("D34 registered class order drift")
    geometry = dict(registration_fit.geometry_audit)
    trace: list[dict[str, Any]] = list(old_fit.solver_trace)
    for key in (
        "old_loso_trace",
        "new_loso_trace",
        "collision_edges",
        "old_anchor_offset_trace",
    ):
        value = geometry.get(key, ())
        if isinstance(value, (list, tuple)):
            trace.extend(dict(row) for row in value if isinstance(row, Mapping))
    return {
        "old_fit": old_fit,
        "old_state": old_state,
        "registration_fit": registration_fit,
        "after": registration_fit.state,
        "geometry": geometry,
        "old_resource": dict(old_fit.resource_audit),
        "registration_resource": dict(registration_fit.resource_audit),
        "complete_trace": trace,
    }


def _d34_fast_unit_and_prefix(
    old_state: Any, features: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(features, dtype=np.float32)
    scaled = rows * np.exp(old_state.log_diag, dtype=np.float32)[None, :]
    norms = np.linalg.norm(scaled, axis=1, keepdims=True).astype(np.float32)
    if bool(np.any(norms <= np.float32(1.0e-12))):
        raise D25RunnerError("D34 zero-norm FAST-adapted feature")
    adapted = np.asarray(scaled / norms, dtype=np.float32)
    old_prefix = np.float32(18.0) * (adapted @ old_state.weights.T)
    return adapted, np.asarray(old_prefix, dtype=np.float32)


def _d36_fixed_ground_anchor(component: Any) -> tuple[np.ndarray, int, str]:
    """Resolve the sealed component-only maximin medoid before support fit."""

    medoid = int(_component_maximin_medoid(component))
    anchors = np.asarray(_transient_domain_anchors(component, medoid), dtype=np.float32)
    if anchors.shape != (len(component.class_registry), 160):
        raise D25RunnerError("D36 fixed ground anchor shape drift")
    digest = hashlib.sha256(np.ascontiguousarray(anchors).tobytes()).hexdigest()
    anchors.setflags(write=False)
    return anchors, medoid, digest


def _score_d34(fit: Mapping[str, Any], features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    adapted, old_prefix = _d34_fast_unit_and_prefix(fit["old_state"], features)
    all_scores = score_d34_collision_local_registration(
        fit["after"], adapted, old_prefix
    )
    return old_prefix, all_scores


def _d34_resource(fit: Mapping[str, Any], registered_count: int) -> dict[str, Any]:
    old_resource = dict(fit["old_resource"])
    registration = dict(fit["registration_resource"])
    old_state_bytes = int(old_resource["persistent_state_bytes"])
    registration_state_bytes = int(registration["persistent_state_bytes"])
    old_query_macs = int(old_resource["estimated_macs_per_query"])
    registration_query_macs_average = float(
        registration["estimated_macs_per_query_average_degree"]
    )
    registration_query_macs = int(
        registration["estimated_macs_per_query_worst_degree"]
    )
    total_query_macs_average = old_query_macs + registration_query_macs_average
    total_query_macs = old_query_macs + registration_query_macs
    old_adaptation_macs = int(old_resource["estimated_adaptation_macs"])
    registration_adaptation_macs = int(registration["estimated_adaptation_macs"])
    state_bytes = old_state_bytes + registration_state_bytes
    identity_macs = int(registered_count * 10 * 160)
    geometry = dict(fit["geometry"])
    edge_count = int(
        registration.get(
            "collision_edge_count",
            geometry.get("collision_edge_count", 0),
        )
    )
    unreachable_count = int(
        registration.get(
            "unreachable_edge_count",
            geometry.get("unreachable_edge_count", 0),
        )
    )
    old_loso_pass = bool(
        geometry.get(
            "old_loso_zero_intrusion_pass",
            registration.get("old_loso_zero_intrusion_pass", False),
        )
    )
    return {
        "schema": "cvs.phase2.d34_combined_resource.v1",
        "old_solver": "b3_fisher_closed_form",
        "old_solver_resource": old_resource,
        "collision_local_registration_resource": registration,
        "peak_trainable_parameters": 0,
        "active_closed_form_scalars": int(old_resource["active_scalars"]),
        "trainable_parameter_cap": 50_000,
        "trainable_parameter_cap_pass": True,
        "total_optimizer_steps": 0,
        "total_adaptation_epochs": 0,
        "optimizer_step_cap": 20,
        "optimizer_step_cap_pass": True,
        "stage2b_adaptation_macs": old_adaptation_macs,
        "stage2c_adaptation_macs": registration_adaptation_macs,
        "total_adaptation_macs": old_adaptation_macs + registration_adaptation_macs,
        "estimated_adaptation_macs": old_adaptation_macs
        + registration_adaptation_macs,
        "old_prefix_macs_per_query": old_query_macs,
        "collision_local_extra_macs_per_query": registration_query_macs,
        "collision_local_extra_macs_per_query_average_degree": (
            registration_query_macs_average
        ),
        "collision_local_extra_macs_per_query_worst_degree": registration_query_macs,
        "total_post_backbone_macs_per_query_average_degree": total_query_macs_average,
        "total_post_backbone_macs_per_query_worst_degree": total_query_macs,
        "total_post_backbone_macs_per_query": total_query_macs,
        "estimated_macs_per_query": total_query_macs,
        "argmax_scalar_comparisons_per_query": max(0, registered_count - 1),
        "identity_single_qknn_macs_same_registered_count": identity_macs,
        "estimated_score_mac_ratio_vs_identity_single_qknn": float(
            total_query_macs / identity_macs
        ),
        "old_solver_state_bytes": old_state_bytes,
        "collision_local_state_bytes": registration_state_bytes,
        "persistent_state_bytes": state_bytes,
        "persistent_state_cap_bytes": 256 * 1024,
        "persistent_state_cap_pass": state_bytes <= 256 * 1024,
        "active_predictor_state_bytes": state_bytes,
        "collision_edge_count": edge_count,
        "unreachable_edge_count": unreachable_count,
        "old_loso_zero_intrusion_pass": old_loso_pass,
        "actual_int8_component_used_for_prediction": False,
        "target_new_int8_prototypes_used_for_prediction": True,
        "dense_query_graph_bytes": 0,
        "complete_loss_trace": list(fit["complete_trace"]),
        "feature_geometry": "b3_auxiliary_dominant_z160_fft96_rf32_v1",
        "query_rows_used_for_fit": 0,
        "query_labels_used_for_fit": False,
        "query_features_used_for_fit": False,
        "query_role_oracle_access": False,
        "query_true_batch_class_count_access": False,
        "query_class_quota_access": False,
        "query_batch_global_assignment": False,
        "clean_sample_access": False,
        "source_sample_access": False,
    }


def _fit_d35_route(
    features: np.ndarray,
    labels: np.ndarray,
    old_mask: np.ndarray,
    new_mask: np.ndarray,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    config: D35CandidateConfig,
) -> dict[str, Any]:
    """Fit FAST once, then append the globally visible dense-safe head."""

    old_fit = fit_b3_fisher_closed_form(
        features[old_mask], labels[old_mask], old_classes, config=config.fisher
    )
    adapted, old_prefix_all = _d34_fast_unit_and_prefix(old_fit.state, features)
    reference = score_b3_fisher_closed_form(old_fit.state, features)
    if not np.array_equal(old_prefix_all, reference):
        raise D25RunnerError("D35 FAST prefix implementation is not bitwise exact")
    registration_fit = fit_d35_dense_safe_registration(
        adapted[old_mask],
        labels[old_mask],
        old_classes,
        old_prefix_all[old_mask],
        adapted[new_mask],
        labels[new_mask],
        new_classes,
        old_prefix_all[new_mask],
        config=config.registration,
    )
    if tuple(registration_fit.state.classes) != old_classes + new_classes:
        raise D25RunnerError("D35 registered class order drift")
    geometry = dict(registration_fit.geometry_audit)
    trace: list[dict[str, Any]] = list(old_fit.solver_trace)
    for key in (
        "uncertainty_trace",
        "prototype_selector_trace",
        "threshold_trace",
        "old_leave_one_out",
        "new_physical_leave_one_out",
    ):
        value = geometry.get(key, ())
        if isinstance(value, (list, tuple)):
            trace.extend(dict(row) for row in value if isinstance(row, Mapping))
    return {
        "old_fit": old_fit,
        "old_state": old_fit.state,
        "registration_fit": registration_fit,
        "after": registration_fit.state,
        "geometry": geometry,
        "old_resource": dict(old_fit.resource_audit),
        "registration_resource": dict(registration_fit.resource_audit),
        "complete_trace": trace,
    }


def _score_d35(
    fit: Mapping[str, Any], features: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    adapted, old_prefix = _d34_fast_unit_and_prefix(fit["old_state"], features)
    scores = score_d35_dense_safe_registration(fit["after"], adapted, old_prefix)
    return old_prefix, scores


def _d35_new_reachability(geometry: Mapping[str, Any]) -> dict[str, bool]:
    rows = list(geometry.get("new_physical_leave_one_out", ()))
    by_class: dict[str, list[bool]] = {}
    for row in rows:
        name = str(row.get("new_class", ""))
        if not name:
            continue
        by_class.setdefault(name, []).append(
            bool(row.get("correct", False)) and float(row.get("margin", -np.inf)) > 0.0
        )
    return {name: bool(values) and all(values) for name, values in by_class.items()}


def _d35_resource(fit: Mapping[str, Any], registered_count: int) -> dict[str, Any]:
    old_resource = dict(fit["old_resource"])
    registration = dict(fit["registration_resource"])
    geometry = dict(fit["geometry"])
    old_state_bytes = int(old_resource["persistent_state_bytes"])
    registration_state_bytes = int(registration["persistent_state_bytes"])
    old_query_macs = int(old_resource["estimated_macs_per_query"])
    registration_query_macs = int(
        registration["estimated_registration_macs_per_unit_query"]
    )
    total_query_macs = old_query_macs + registration_query_macs
    old_adaptation_macs = int(old_resource["estimated_adaptation_macs"])
    registration_adaptation_macs = int(registration["estimated_deploy_refit_macs"])
    state_bytes = old_state_bytes + registration_state_bytes
    identity_macs = int(registered_count * 10 * 160)
    reachability = _d35_new_reachability(geometry)
    return {
        "schema": "cvs.phase2.d35_combined_resource.v1",
        "old_solver": "b3_fisher_closed_form",
        "old_solver_resource": old_resource,
        "dense_safe_registration_resource": registration,
        "peak_trainable_parameters": 0,
        "active_closed_form_scalars": int(old_resource["active_scalars"])
        + int(registration["active_parameters"]),
        "trainable_parameter_cap": 50_000,
        "trainable_parameter_cap_pass": True,
        "total_optimizer_steps": 0,
        "total_adaptation_epochs": 0,
        "optimizer_step_cap": 20,
        "optimizer_step_cap_pass": True,
        "stage2b_adaptation_macs": old_adaptation_macs,
        "stage2c_adaptation_macs": registration_adaptation_macs,
        "total_adaptation_macs": old_adaptation_macs + registration_adaptation_macs,
        "estimated_adaptation_macs": old_adaptation_macs
        + registration_adaptation_macs,
        "development_old_loso_macs": int(
            registration["estimated_development_old_loso_macs"]
        ),
        "development_new_loso_macs": int(
            registration["estimated_development_new_loso_macs"]
        ),
        "development_total_loso_macs": int(
            registration["estimated_development_total_loso_macs"]
        ),
        "old_prefix_macs_per_query": old_query_macs,
        "dense_safe_extra_macs_per_query": registration_query_macs,
        "dense_safe_scalar_ops_per_query": int(
            registration["estimated_scalar_ops_per_query"]
        ),
        "query_prototype_dot_macs": int(registration["query_prototype_dot_macs"]),
        "query_inverse_temperature_scalar_ops": int(
            registration["query_inverse_temperature_scalar_ops"]
        ),
        "query_threshold_subtraction_scalar_ops": int(
            registration["query_threshold_subtraction_scalar_ops"]
        ),
        "query_prototype_max_comparisons": int(
            registration["query_prototype_max_comparisons"]
        ),
        "query_old_winner_argmax_comparisons": int(
            registration["query_old_winner_argmax_comparisons"]
        ),
        "total_post_backbone_macs_per_query": total_query_macs,
        "estimated_macs_per_query": total_query_macs,
        "argmax_scalar_comparisons_per_query": max(0, registered_count - 1),
        "identity_single_qknn_macs_same_registered_count": identity_macs,
        "estimated_score_mac_ratio_vs_identity_single_qknn": float(
            total_query_macs / identity_macs
        ),
        "old_solver_state_bytes": old_state_bytes,
        "dense_safe_state_bytes": registration_state_bytes,
        "persistent_state_bytes": state_bytes,
        "persistent_state_cap_bytes": 256 * 1024,
        "persistent_state_cap_pass": state_bytes <= 256 * 1024,
        "active_predictor_state_bytes": state_bytes,
        "all_new_classes_globally_visible": bool(
            geometry.get("all_new_classes_global_visible", False)
        ),
        "new_class_reachability": reachability,
        "unreachable_new_class_count": int(
            sum(not value for value in reachability.values())
        ),
        "old_loso_intrusion_count": int(
            geometry.get("old_loso_intrusion_count", 0)
        ),
        "actual_int8_component_used_for_prediction": False,
        "target_new_int8_prototypes_used_for_prediction": True,
        "dense_query_graph_bytes": 0,
        "complete_loss_trace": list(fit["complete_trace"]),
        "feature_geometry": "b3_auxiliary_dominant_z160_fft96_rf32_v1",
        "query_rows_used_for_fit": 0,
        "query_labels_used_for_fit": False,
        "query_features_used_for_fit": False,
        "query_role_oracle_access": False,
        "query_true_batch_class_count_access": False,
        "query_class_quota_access": False,
        "query_batch_global_assignment": False,
        "clean_sample_access": False,
        "source_sample_access": False,
    }


def _fit_d36_route(
    features: np.ndarray,
    labels: np.ndarray,
    ranks: np.ndarray,
    old_mask: np.ndarray,
    new_mask: np.ndarray,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    config: D36CandidateConfig,
    ground_anchor: np.ndarray | None,
) -> dict[str, Any]:
    """Fit four inner rank-pair models, then OOF-calibrate one final D36 state."""

    unique_ranks = tuple(sorted(int(v) for v in np.unique(ranks).tolist()))
    if len(unique_ranks) not in (8, 10):
        raise D25RunnerError("D36 crossfit requires eight outer-train or ten full-K ranks")
    inner_pairs = tuple(
        (unique_ranks[index], unique_ranks[index + 1])
        for index in range(0, len(unique_ranks), 2)
    )
    oof_features: list[np.ndarray] = []
    oof_roles: list[np.ndarray] = []
    development_trace: list[dict[str, Any]] = []
    development_macs = 0
    for inner_index, held_pair in enumerate(inner_pairs):
        inner_held = np.isin(ranks, np.asarray(held_pair, dtype=np.int64))
        inner_train = ~inner_held
        fisher_fit = fit_b3_fisher_closed_form(
            features[inner_train & old_mask],
            labels[inner_train & old_mask],
            old_classes,
            config=config.fisher,
        )
        inner_result = fit_d36_compiled_joint_int8(
            features[inner_train & old_mask],
            labels[inner_train & old_mask],
            old_classes,
            features[inner_train & new_mask],
            labels[inner_train & new_mask],
            new_classes,
            fisher_fit.state.log_diag,
            config=config.compiled,
            ground_anchor_z=(ground_anchor if config.compiled.arm in ("B", "C") else None),
        )
        oof_features.append(
            margin_features_d36_compiled_joint_int8(
                inner_result.state, features[inner_held]
            )
        )
        oof_roles.append(np.asarray(new_mask[inner_held], dtype=np.int64))
        development_macs += int(
            fisher_fit.resource_audit["estimated_adaptation_macs"]
        ) + int(inner_result.resource_audit["estimated_total_adaptation_macs"])
        for trace_row in inner_result.training_trace:
            development_trace.append(
                {
                    **dict(trace_row),
                    "scope": "inner_crossfit",
                    "inner_fold": inner_index,
                    "held_ranks": list(held_pair),
                    "query_rows_used": 0,
                }
            )
    margin_features = np.concatenate(oof_features, axis=0).astype(np.float32)
    roles = np.concatenate(oof_roles, axis=0).astype(np.int64)
    if len(margin_features) != len(features) or set(roles.tolist()) != {0, 1}:
        raise D25RunnerError("D36 OOF margin row closure drift")

    fisher_fit = fit_b3_fisher_closed_form(
        features[old_mask], labels[old_mask], old_classes, config=config.fisher
    )
    final_result = fit_d36_compiled_joint_int8(
        features[old_mask],
        labels[old_mask],
        old_classes,
        features[new_mask],
        labels[new_mask],
        new_classes,
        fisher_fit.state.log_diag,
        config=config.compiled,
        ground_anchor_z=(ground_anchor if config.compiled.arm in ("B", "C") else None),
    )
    calibrated_state = with_oof_calibration_d36_compiled_joint_int8(
        final_result.state, margin_features, roles
    )
    oof_calibration_trace: list[dict[str, Any]] = []
    if config.compiled.arm == "C":
        oof_weights, oof_calibration_trace = _d36_fit_irls(margin_features, roles)
        if not np.array_equal(
            oof_weights.astype(np.float16), calibrated_state.calibration_fp16
        ):
            raise D25RunnerError("D36 OOF IRLS state/trace drift")
    elif config.compiled.arm == "B":
        oof_calibration_trace.append(
            {
                "calibration": "constant_oof_margin",
                "offset": float(calibrated_state.calibration_fp16[0]),
            }
        )
    else:
        oof_calibration_trace.append({"calibration": "none"})
    final_trace = [
        {**dict(row), "scope": "deploy_refit", "query_rows_used": 0}
        for row in final_result.training_trace
    ]
    final_trace.extend(
        {
            **dict(row),
            "scope": "oof_calibration",
            "oof_row_count": len(margin_features),
            "query_rows_used": 0,
        }
        for row in oof_calibration_trace
    )
    return {
        "fisher_fit": fisher_fit,
        "before_state": final_result.before_state,
        "state": calibrated_state,
        "core_result": final_result,
        "inner_pairs": inner_pairs,
        "oof_margin_features": margin_features,
        "oof_roles": roles,
        "development_trace": development_trace,
        "complete_trace": development_trace + final_trace,
        "development_oof_macs": development_macs,
        "ground_anchor_used": config.compiled.arm in ("B", "C"),
    }


def _score_d36(
    fit: Mapping[str, Any], features: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    before = score_d36_compiled_joint_int8(fit["before_state"], features)
    after = score_d36_compiled_joint_int8(fit["state"], features)
    return before, after


def _d36_resource(fit: Mapping[str, Any], registered_count: int) -> dict[str, Any]:
    core = dict(fit["core_result"].resource_audit)
    state = fit["state"]
    before_state = fit["before_state"]
    fisher = dict(fit["fisher_fit"].resource_audit)
    query_macs = int(core["query_dot_macs"])
    identity_macs = int(registered_count * 10 * 160)
    deploy_macs = int(fisher["estimated_adaptation_macs"]) + int(
        core["estimated_total_adaptation_macs"]
    ) + int(core["estimated_compile_macs"])
    return {
        "schema": "cvs.phase2.d36_compiled_joint_resource.v1",
        "active_adapter_parameters": int(core["active_adapter_parameters"]),
        "peak_trainable_parameters": int(core["active_adapter_parameters"]),
        "trainable_parameter_cap": 50_000,
        "trainable_parameter_cap_pass": bool(core["active_parameters_under_50k"]),
        "total_optimizer_steps": int(core["optimizer_steps"]),
        "total_adaptation_epochs": int(core["adaptation_epochs"]),
        "optimizer_step_cap": 20,
        "optimizer_step_cap_pass": int(core["optimizer_steps"]) <= 20,
        "total_adaptation_macs": deploy_macs,
        "estimated_adaptation_macs": deploy_macs,
        "development_inner_crossfit_macs": int(fit["development_oof_macs"]),
        "total_post_backbone_macs_per_query": query_macs,
        "estimated_macs_per_query": query_macs,
        "query_scalar_ops": int(core["query_scalar_ops"]),
        "identity_single_qknn_macs_same_registered_count": identity_macs,
        "estimated_score_mac_ratio_vs_identity_single_qknn": float(
            query_macs / identity_macs
        ),
        "pre_registration_state_bytes": int(before_state.persistent_state_bytes),
        "persistent_state_bytes": int(state.persistent_state_bytes),
        "active_predictor_state_bytes": int(state.persistent_state_bytes),
        "persistent_state_cap_bytes": 256 * 1024,
        "persistent_state_cap_pass": int(state.persistent_state_bytes) <= 256 * 1024,
        "resident_fp32_target_prototype_count": int(
            core["resident_fp32_target_prototype_count"]
        ),
        "target_old_int8_prototypes_used_for_prediction": True,
        "target_new_int8_prototypes_used_for_prediction": True,
        "actual_int8_component_used_for_prediction": bool(fit["ground_anchor_used"]),
        "phase1_anchor_compiled_once": bool(fit["ground_anchor_used"]),
        "oof_calibration_row_count": int(len(fit["oof_roles"])),
        "oof_calibration_old_row_count": int(np.sum(fit["oof_roles"] == 0)),
        "oof_calibration_new_row_count": int(np.sum(fit["oof_roles"] == 1)),
        "inner_crossfit_fold_count": len(fit["inner_pairs"]),
        "complete_loss_trace": list(fit["complete_trace"]),
        "dense_query_graph_bytes": 0,
        "query_rows_used_for_fit": 0,
        "query_labels_used_for_fit": False,
        "query_features_used_for_fit": False,
        "query_role_oracle_access": False,
        "query_true_batch_class_count_access": False,
        "query_class_quota_access": False,
        "query_batch_global_assignment": False,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "source_sample_access": False,
        "source_derived_signal_access": False,
        "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
        "phase2_query_decision_policy": "per_sample_all_registered_classes",
    }


def _fit_d37_route(
    features: np.ndarray,
    labels: np.ndarray,
    ranks: np.ndarray,
    physical_ids: np.ndarray,
    old_mask: np.ndarray,
    new_mask: np.ndarray,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    config: D37CandidateConfig,
) -> dict[str, Any]:
    """Cross-fit physical support base scores and fail closed per candidate."""

    unique_ranks = tuple(sorted(int(value) for value in np.unique(ranks).tolist()))
    physical_ids = np.asarray(physical_ids).astype(str)
    if (
        len(physical_ids) != len(features)
        or len(set(physical_ids.tolist())) != len(physical_ids)
        or any(not value for value in physical_ids.tolist())
    ):
        raise D25RunnerError("D37 physical support ID closure drift")
    if len(unique_ranks) not in (8, 10):
        raise D25RunnerError("D37 crossfit requires eight outer-train or ten full-K ranks")
    inner_pairs = tuple(
        (unique_ranks[index], unique_ranks[index + 1])
        for index in range(0, len(unique_ranks), 2)
    )
    oof_scores: list[np.ndarray] = []
    oof_labels: list[np.ndarray] = []
    oof_fold_ids: list[np.ndarray] = []
    oof_physical_ids: list[np.ndarray] = []
    development_trace: list[dict[str, Any]] = []
    development_macs = 0
    for inner_index, held_pair in enumerate(inner_pairs):
        inner_held = np.isin(ranks, np.asarray(held_pair, dtype=np.int64))
        inner_train = ~inner_held
        fisher_fit = fit_b3_fisher_closed_form(
            features[inner_train & old_mask],
            labels[inner_train & old_mask],
            old_classes,
            config=config.fisher,
        )
        inner_result = fit_d37_b3_preserving_int8(
            features[inner_train & old_mask],
            labels[inner_train & old_mask],
            old_classes,
            features[inner_train & new_mask],
            labels[inner_train & new_mask],
            new_classes,
            fisher_fit.state,
            config=config.compiled,
        )
        oof_scores.append(
            base_score_d37_b3_preserving_int8(
                inner_result.state_no_offset, features[inner_held]
            )
        )
        oof_labels.append(np.asarray(labels[inner_held]).astype(str))
        oof_fold_ids.append(
            np.full(int(np.sum(inner_held)), inner_index, dtype=np.int64)
        )
        oof_physical_ids.append(np.asarray(physical_ids[inner_held]).astype(str))
        development_macs += int(
            fisher_fit.resource_audit["estimated_adaptation_macs"]
        ) + int(inner_result.resource_audit["estimated_registration_macs"])
        development_trace.extend(
            {
                **dict(row),
                "scope": "inner_crossfit",
                "inner_fold": inner_index,
                "held_ranks": list(held_pair),
                "query_rows_used": 0,
            }
            for row in inner_result.training_trace
        )
    base_scores = np.ascontiguousarray(np.concatenate(oof_scores, axis=0), dtype=np.float32)
    physical_labels = np.concatenate(oof_labels, axis=0).astype(str)
    fold_ids = np.concatenate(oof_fold_ids, axis=0).astype(np.int64)
    ordered_physical_ids = np.concatenate(oof_physical_ids, axis=0).astype(str)
    if (
        len(base_scores) != len(features)
        or base_scores.shape[1] != len(old_classes) + len(new_classes)
        or set(physical_labels.tolist()) != set(old_classes + new_classes)
        or len(set(ordered_physical_ids.tolist())) != len(ordered_physical_ids)
        or len(set(fold_ids.tolist())) != len(inner_pairs)
    ):
        raise D25RunnerError("D37 OOF base-score/label closure drift")

    fisher_fit = fit_b3_fisher_closed_form(
        features[old_mask], labels[old_mask], old_classes, config=config.fisher
    )
    final_result = fit_d37_b3_preserving_int8(
        features[old_mask],
        labels[old_mask],
        old_classes,
        features[new_mask],
        labels[new_mask],
        new_classes,
        fisher_fit.state,
        config=config.compiled,
    )
    feasible = True
    failure_reason: str | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    offset: float | None = None
    try:
        calibrated = fit_oof_feasible_offset_d37(
            final_result.state_no_offset,
            base_scores,
            physical_labels,
            oof_fold_ids=fold_ids,
            oof_physical_ids=ordered_physical_ids,
            source=D37_OOF_SOURCE,
        )
        state = calibrated.state
        lower_bound = float(calibrated.lower_bound)
        upper_bound = float(calibrated.upper_bound)
        offset = float(calibrated.offset)
        old_oof_count = int(calibrated.old_oof_count)
        new_oof_count = int(calibrated.new_oof_count)
        oof_fold_count = int(calibrated.fold_count)
        oof_physical_id_sha256 = str(calibrated.physical_id_sha256)
        oof_source = str(calibrated.source)
    except D37B3PreservingInt8Error as exc:
        if not any(
            marker in str(exc)
            for marker in (
                "empty OOF feasible interval",
                "OOF feasible interval contains no deployable FP16 offset",
            )
        ):
            raise
        feasible = False
        failure_reason = str(exc)
        state = final_result.state_no_offset
        old_oof_count = int(np.sum(np.isin(physical_labels, np.asarray(old_classes))))
        new_oof_count = int(len(physical_labels) - old_oof_count)
        oof_fold_count = len(set(fold_ids.tolist()))
        oof_physical_id_sha256 = hashlib.sha256(
            "\n".join(sorted(ordered_physical_ids.tolist())).encode("utf-8")
        ).hexdigest()
        oof_source = D37_OOF_SOURCE
    calibration_trace = {
        "solver": "support_oof_shared_new_offset_feasible_interval",
        "scope": "oof_calibration",
        "arm": str(config.compiled.arm),
        "margin": float(config.compiled.margin),
        "feasible": feasible,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "offset": offset,
        "failure_reason": failure_reason,
        "oof_row_count": len(physical_labels),
        "query_rows_used": 0,
    }
    final_trace = [
        {**dict(row), "scope": "deploy_refit", "query_rows_used": 0}
        for row in final_result.training_trace
    ] + [calibration_trace]
    return {
        "fisher_fit": fisher_fit,
        "before_state": final_result.before_state,
        "state_no_offset": final_result.state_no_offset,
        "state": state,
        "core_result": final_result,
        "inner_pairs": inner_pairs,
        "oof_base_scores": base_scores,
        "oof_labels": physical_labels,
        "oof_fold_ids": fold_ids,
        "oof_physical_ids": ordered_physical_ids,
        "oof_old_row_count": old_oof_count,
        "oof_new_row_count": new_oof_count,
        "oof_fold_count": oof_fold_count,
        "oof_physical_id_sha256": oof_physical_id_sha256,
        "oof_source": oof_source,
        "oof_feasible_interval_pass": feasible,
        "oof_feasible_interval_lower_bound": lower_bound,
        "oof_feasible_interval_upper_bound": upper_bound,
        "oof_offset": offset,
        "oof_failure_reason": failure_reason,
        "development_trace": development_trace,
        "complete_trace": development_trace + final_trace,
        "development_oof_macs": development_macs,
    }


def _score_d37(
    fit: Mapping[str, Any], features: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    before = score_d37_b3_preserving_int8(fit["before_state"], features)
    after = (
        score_d37_b3_preserving_int8(fit["state"], features)
        if bool(fit["oof_feasible_interval_pass"])
        else base_score_d37_b3_preserving_int8(fit["state_no_offset"], features)
    )
    return before, after


def _d37_resource(fit: Mapping[str, Any], registered_count: int) -> dict[str, Any]:
    core = dict(fit["core_result"].resource_audit)
    fisher = dict(fit["fisher_fit"].resource_audit)
    state = fit["state"]
    before_state = fit["before_state"]
    query_macs = int(core["estimated_macs_per_query"])
    identity_macs = int(registered_count * 10 * 160)
    deploy_macs = int(fisher["estimated_adaptation_macs"]) + int(
        core["estimated_registration_macs"]
    )
    return {
        "schema": "cvs.phase2.d37_b3_preserving_int8_resource.v1",
        "active_adapter_parameters": 0,
        "peak_trainable_parameters": 0,
        "trainable_parameter_cap": 80_000,
        "trainable_parameter_cap_pass": True,
        "total_optimizer_steps": 0,
        "total_adaptation_epochs": 0,
        "optimizer_step_cap": 30,
        "optimizer_step_cap_pass": True,
        "total_adaptation_macs": deploy_macs,
        "estimated_adaptation_macs": deploy_macs,
        "development_inner_crossfit_macs": int(fit["development_oof_macs"]),
        "total_post_backbone_macs_per_query": query_macs,
        "estimated_macs_per_query": query_macs,
        "identity_single_qknn_macs_same_registered_count": identity_macs,
        "estimated_score_mac_ratio_vs_identity_single_qknn": float(
            query_macs / identity_macs
        ),
        "pre_registration_state_bytes": int(before_state.persistent_state_bytes),
        "persistent_state_bytes": int(state.persistent_state_bytes),
        "active_predictor_state_bytes": int(state.persistent_state_bytes),
        "persistent_state_cap_bytes": 256 * 1024,
        "persistent_state_cap_pass": int(state.persistent_state_bytes) <= 256 * 1024,
        "resident_fp32_target_prototype_count": 0,
        "target_old_int8_prototypes_used_for_prediction": True,
        "target_new_int8_prototypes_used_for_prediction": True,
        "actual_int8_component_used_for_prediction": True,
        "old_prefix_bitwise_unchanged": bool(
            old_prefix_bitwise_unchanged_d37(before_state, fit["state_no_offset"])
        ),
        "oof_calibration_row_count": int(len(fit["oof_labels"])),
        "oof_calibration_old_row_count": int(fit["oof_old_row_count"]),
        "oof_calibration_new_row_count": int(fit["oof_new_row_count"]),
        "oof_crossfit_fold_count": int(fit["oof_fold_count"]),
        "oof_physical_id_sha256": str(fit["oof_physical_id_sha256"]),
        "oof_source": str(fit["oof_source"]),
        "oof_feasible_interval_pass": bool(fit["oof_feasible_interval_pass"]),
        "deployable_calibrated_state": bool(fit["oof_feasible_interval_pass"]),
        "infeasible_state_scored_only_by_base_score_for_diagnostics": bool(
            not fit["oof_feasible_interval_pass"]
        ),
        "oof_feasible_interval_lower_bound": fit["oof_feasible_interval_lower_bound"],
        "oof_feasible_interval_upper_bound": fit["oof_feasible_interval_upper_bound"],
        "oof_offset": fit["oof_offset"],
        "oof_failure_reason": fit["oof_failure_reason"],
        "inner_crossfit_fold_count": len(fit["inner_pairs"]),
        "complete_loss_trace": list(fit["complete_trace"]),
        "dense_query_graph_bytes": 0,
        "query_rows_used_for_fit": 0,
        "query_labels_used_for_fit": False,
        "query_features_used_for_fit": False,
        "query_role_oracle_access": False,
        "query_true_batch_class_count_access": False,
        "query_class_quota_access": False,
        "query_batch_global_assignment": False,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "source_sample_access": False,
        "source_derived_signal_access": False,
        "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
        "phase2_query_decision_policy": "per_sample_all_registered_classes",
    }


def _evaluate_d30_fold(
    component: Any,
    rows: Mapping[str, np.ndarray],
    z_id160: np.ndarray,
    direct_logits: np.ndarray,
    fft96: np.ndarray,
    rf32: np.ndarray,
    *,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    held_ranks: tuple[int, int],
    candidate_id: str,
    config: D30CandidateConfig,
) -> dict[str, Any]:
    """Evaluate D30 without letting held support select either calibrator."""

    labels = np.asarray(rows["labels"]).astype(str)
    ranks = np.asarray(rows["ranks"], dtype=np.int64)
    held = np.isin(ranks, np.asarray(held_ranks, dtype=np.int64))
    train = ~held
    old = np.isin(labels, np.asarray(old_classes))
    new = np.isin(labels, np.asarray(new_classes))
    all_classes = old_classes + new_classes
    if (
        int(np.sum(train & old)) != 8 * len(old_classes)
        or int(np.sum(train & new)) != 8 * len(new_classes)
        or int(np.sum(held & old)) != 2 * len(old_classes)
        or int(np.sum(held & new)) != 2 * len(new_classes)
    ):
        raise D25RunnerError("D30 leave-two-out class symmetry drift")

    features = _d1_feature_from_blocks(z_id160, fft96, rf32)
    fit_old_features = features[train & old]
    fit_old_labels = labels[train & old]
    before_fit = fit_stage2b_compact_diag(
        fit_old_features, fit_old_labels, old_classes, config=config.base
    )
    before = before_fit.state
    after_fit = append_stage2c_d26(
        before,
        features[train & new],
        labels[train & new],
        new_classes,
        fit_old_features,
        fit_old_labels,
    )
    after = after_fit.state
    if before.classes != old_classes or after.classes != all_classes:
        raise D25RunnerError("D30 registered class order drift")
    if (
        before.old_lock_sha256 != after.old_lock_sha256
        or before.log_diag.tobytes() != after.log_diag.tobytes()
        or before.weights.tobytes()
        != after.weights[: len(old_classes)].tobytes()
    ):
        raise D25RunnerError("D30 base mutated D27 frozen old state")

    dali_old = fit_old_dali(
        component,
        z_id160[train & old],
        labels[train & old],
        direct_logits[train & old],
        config=config.dali,
    )
    dali_state = register_new_dali(
        dali_old,
        z_id160[train & new],
        labels[train & new],
        registered_classes=new_classes,
    )
    train_old_raw = score_all_d26(after, features[train & old])
    train_old_dali = _d30_rerank_matrix(
        dali_state,
        train_old_raw,
        z_id160[train & old],
        direct_logits[train & old],
    )
    dali_gate_pass, dali_gate_audit = _d30_old_support_gate(
        train_old_raw,
        train_old_dali,
        labels[train & old],
        old_classes,
    )
    dali_enabled = _d30_enable_dali(dali_state.k_shot, dali_gate_pass)
    dali_gate_audit["k_shot"] = int(dali_state.k_shot)
    dali_gate_audit["k1_exact_base_head_passthrough"] = bool(
        int(dali_state.k_shot) == 1
    )
    dali_gate_audit["enabled"] = dali_enabled

    train_raw = score_all_d26(after, features[train])
    train_dali = (
        _d30_rerank_matrix(
            dali_state,
            train_raw,
            z_id160[train],
            direct_logits[train],
        )
        if dali_enabled
        else train_raw.copy()
    )
    envelope = fit_max_envelope_calibration(
        train_dali,
        labels[train],
        _dense_fold_shot_ranks(labels[train], ranks[train], all_classes),
        all_classes,
        len(old_classes),
        config=MaxEnvelopeCalibrationConfig(
            objective=config.envelope_objective,
            coordinate_passes=2,
        ),
    )

    held_old_features = features[held & old]
    held_new_features = features[held & new]
    before_predictions = (
        predict_all_d26(before, held_old_features).astype(str).tolist()
    )
    held_old_raw = score_all_d26(after, held_old_features)
    held_new_raw = score_all_d26(after, held_new_features)
    if dali_enabled:
        held_old_dali = _d30_rerank_matrix(
            dali_state,
            held_old_raw,
            z_id160[held & old],
            direct_logits[held & old],
        )
        held_new_dali = _d30_rerank_matrix(
            dali_state,
            held_new_raw,
            z_id160[held & new],
            direct_logits[held & new],
        )
    else:
        held_old_dali = held_old_raw.copy()
        held_new_dali = held_new_raw.copy()
    held_old_adjusted = apply_max_envelope_calibration(envelope, held_old_dali)
    held_new_adjusted = apply_max_envelope_calibration(envelope, held_new_dali)
    if not np.array_equal(
        np.max(held_old_adjusted[:, len(old_classes):], axis=1),
        np.max(held_old_dali[:, len(old_classes):], axis=1),
    ) or not np.array_equal(
        np.max(held_new_adjusted[:, len(old_classes):], axis=1),
        np.max(held_new_dali[:, len(old_classes):], axis=1),
    ):
        raise D25RunnerError("D30 max-new envelope drift")
    after_old_predictions = np.asarray(all_classes)[
        np.argmax(held_old_adjusted, axis=1)
    ].tolist()
    after_new_predictions = np.asarray(all_classes)[
        np.argmax(held_new_adjusted, axis=1)
    ].tolist()

    fit_before_predictions = (
        predict_all_d26(before, fit_old_features).astype(str).tolist()
    )
    fit_after_scores = apply_max_envelope_calibration(
        envelope,
        train_dali[old[train]],
    )
    fit_after_predictions = np.asarray(all_classes)[
        np.argmax(fit_after_scores, axis=1)
    ].tolist()
    fit_before = legacy._metric_block(
        fit_old_labels, fit_before_predictions, old_classes
    )
    fit_after = legacy._metric_block(
        fit_old_labels, fit_after_predictions, old_classes
    )
    tolerance = 1.0e-12
    fit_classwise_non_degradation = all(
        float(fit_after["per_class_accuracy"][name]) + tolerance
        >= float(fit_before["per_class_accuracy"][name])
        for name in old_classes
    )
    fit_floor_non_degradation = (
        float(fit_after["class_floor_accuracy"]) + tolerance
        >= float(fit_before["class_floor_accuracy"])
    )
    old_support_non_degradation = bool(
        fit_classwise_non_degradation and fit_floor_non_degradation
    )
    before_old = legacy._metric_block(
        labels[held & old], before_predictions, old_classes
    )
    after_old = legacy._metric_block(
        labels[held & old], after_old_predictions, old_classes
    )
    after_new = legacy._metric_block(
        labels[held & new], after_new_predictions, new_classes
    )
    held_labels = np.concatenate(
        [labels[held & old], labels[held & new]], axis=0
    )
    held_dali_scores = np.concatenate([held_old_dali, held_new_dali], axis=0)
    held_adjusted_scores = np.concatenate(
        [held_old_adjusted, held_new_adjusted], axis=0
    )
    held_confusion_before_envelope = audit_envelope_confusions(
        held_dali_scores,
        held_labels,
        all_classes,
        len(old_classes),
    )
    held_confusion_after_envelope = audit_envelope_confusions(
        held_adjusted_scores,
        held_labels,
        all_classes,
        len(old_classes),
    )
    if (
        held_confusion_before_envelope["new_aggregate"]["old_win"]
        != held_confusion_after_envelope["new_aggregate"]["old_win"]
    ):
        raise D25RunnerError("D30 envelope changed old/new confusion count")
    training_trace = list(before_fit.loss_trace) + list(after_fit.loss_trace)
    base_resource = dict(after.resource_audit())
    dali_resource = dict(dali_state.resource_audit())
    envelope_resource = dict(envelope.resource_audit())
    dali_extra_macs = int(dali_resource["fixed_medoid_ground_macs_per_query"])
    dali_scalar_ops = 12 * len(old_classes) if dali_enabled else 0
    combined_state = (
        int(base_resource["persistent_state_bytes"])
        + int(dali_resource["persistent_state_bytes"])
        + int(envelope_resource["deployable_predictor_state_bytes"])
    )
    resource = {
        **base_resource,
        "schema": "cvs.phase2.d30_combined_resource.v1",
        "base_d27_b3_geometry_resource": base_resource,
        "dali_resource": dali_resource,
        "max_envelope_resource": envelope_resource,
        "dali_enabled_by_old_support_gate": dali_enabled,
        "dali_old_support_gate": dali_gate_audit,
        "max_envelope_enabled": bool(envelope.enabled),
        "actual_int8_component_used_for_prediction": dali_enabled,
        "int8_component_loaded_and_audited": True,
        "int8_component_state_bytes": int(
            dali_resource["int8_component_state_bytes"]
        ),
        "active_adaptation_parameter_count": int(
            base_resource["peak_trainable_parameters"]
            + envelope_resource["fitted_parameter_count"]
        ),
        "persistent_state_bytes": combined_state,
        "persistent_state_cap_pass": combined_state <= 256 * 1024,
        "estimated_macs_per_query": int(
            base_resource["estimated_macs_per_query"]
            + (dali_extra_macs if dali_enabled else 0)
        ),
        "estimated_row_local_scalar_ops_per_query": int(
            dali_scalar_ops + envelope_resource["estimated_scalar_ops_per_query"]
        ),
        "old_support_non_degradation_pass": old_support_non_degradation,
        "total_optimizer_steps": int(base_resource["total_optimizer_steps"]),
        "total_adaptation_epochs": int(base_resource["total_adaptation_epochs"]),
        "complete_loss_trace": training_trace,
        "feature_geometry": "b3_auxiliary_dominant_z160_fft96_rf32_v1",
        "feature_block_energy_target": {
            "z160": 1.0 / 17.0,
            "fft96_rf32_aux_total": 16.0 / 17.0,
        },
        "query_rows_used_for_fit": 0,
        "query_features_used_for_fit": False,
        "query_labels_used_for_fit": False,
        "query_role_oracle_access": False,
        "query_true_batch_class_count_access": False,
        "query_class_quota_access": False,
        "query_batch_global_assignment": False,
        "dense_query_graph_bytes": 0,
        "source_sample_access": False,
        "clean_sample_access": False,
    }
    geometry = _d26_geometry(after)
    geometry.update(
        {
            "schema": "cvs.phase2.d30_dual_envelope_geometry.v1",
            "feature_geometry": "b3_auxiliary_dominant_z160_fft96_rf32_v1",
            "observed_feature_block_energy": _d30_observed_block_energy(features),
            "dali_enabled": dali_enabled,
            "dali_old_support_gate": dali_gate_audit,
            "max_envelope_enabled": bool(envelope.enabled),
            "max_envelope_biases": [float(value) for value in envelope.biases],
            "max_envelope_fit_audit": json.loads(envelope.audit_json),
            "held_confusion_before_envelope": held_confusion_before_envelope,
            "held_confusion_after_envelope": held_confusion_after_envelope,
        }
    )
    return {
        "candidate_id": candidate_id,
        "held_ranks": list(held_ranks),
        "fit_k_shot": 8,
        "before_old": before_old,
        "after_old": after_old,
        "after_new": after_new,
        "H_old_new": legacy._harmonic(
            float(after_old["overall_accuracy"]),
            float(after_new["overall_accuracy"]),
        ),
        "forgetting": float(
            before_old["overall_accuracy"] - after_old["overall_accuracy"]
        ),
        "joint_floor": float(
            min(
                float(after_old["class_floor_accuracy"]),
                float(after_new["class_floor_accuracy"]),
            )
        ),
        "old_score_columns_bitwise_unchanged": True,
        "base_old_prefix_sha256_before": before.old_lock_sha256,
        "base_old_prefix_sha256_after": after.old_lock_sha256,
        "fit_old_before_registration": fit_before,
        "fit_old_after_registration": fit_after,
        "old_support_classwise_non_degradation": fit_classwise_non_degradation,
        "old_support_floor_non_degradation": fit_floor_non_degradation,
        "old_support_non_degradation_pass": old_support_non_degradation,
        "dali_enabled": dali_enabled,
        "dali_old_support_gate": dali_gate_audit,
        "max_envelope_enabled": bool(envelope.enabled),
        "max_envelope_fit_audit": json.loads(envelope.audit_json),
        "new_confusion_before_envelope": held_confusion_before_envelope,
        "new_confusion_after_envelope": held_confusion_after_envelope,
        "training_trace": training_trace,
        "geometry_summary": geometry,
        "resource": resource,
    }


def _evaluate_d31_fold(
    component: Any,
    rows: Mapping[str, np.ndarray],
    z_id160: np.ndarray,
    direct_logits: np.ndarray,
    fft96: np.ndarray,
    rf32: np.ndarray,
    *,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    held_ranks: tuple[int, int],
    candidate_id: str,
    config: D31CandidateConfig,
) -> dict[str, Any]:
    labels = np.asarray(rows["labels"]).astype(str)
    ranks = np.asarray(rows["ranks"], dtype=np.int64)
    held = np.isin(ranks, np.asarray(held_ranks, dtype=np.int64))
    train = ~held
    old = np.isin(labels, np.asarray(old_classes))
    new = np.isin(labels, np.asarray(new_classes))
    if (
        int(np.sum(train & old)) != 8 * len(old_classes)
        or int(np.sum(train & new)) != 8 * len(new_classes)
        or int(np.sum(held & old)) != 2 * len(old_classes)
        or int(np.sum(held & new)) != 2 * len(new_classes)
    ):
        raise D25RunnerError("D31 leave-two-out class symmetry drift")
    features = _d1_feature_from_blocks(z_id160, fft96, rf32)
    fit = _fit_d31_route(
        component,
        features[train],
        z_id160[train],
        direct_logits[train],
        labels[train],
        old[train],
        new[train],
        old_classes,
        new_classes,
        config,
    )
    before = fit["before"]
    after = fit["after"]
    before_old_predictions = predict_all_d26(
        before, features[held & old]
    ).astype(str).tolist()
    held_raw = score_all_d31(after, features[held])
    held_adjusted = (
        _d30_rerank_matrix(
            fit["dali_state"],
            held_raw,
            z_id160[held],
            direct_logits[held],
        )
        if fit["dali_enabled"]
        else held_raw.copy()
    )
    held_labels = labels[held]
    held_predictions = np.asarray(old_classes + new_classes)[
        np.argmax(held_adjusted, axis=1)
    ].tolist()
    held_old_mask = np.isin(held_labels, np.asarray(old_classes))
    held_new_mask = ~held_old_mask
    after_old_predictions = np.asarray(held_predictions)[held_old_mask].tolist()
    after_new_predictions = np.asarray(held_predictions)[held_new_mask].tolist()
    before_old = legacy._metric_block(
        labels[held & old], before_old_predictions, old_classes
    )
    after_old = legacy._metric_block(
        held_labels[held_old_mask], after_old_predictions, old_classes
    )
    after_new = legacy._metric_block(
        held_labels[held_new_mask], after_new_predictions, new_classes
    )
    fit_before_predictions = predict_all_d26(
        before, features[train & old]
    ).astype(str).tolist()
    fit_adjusted = fit["adjusted_scores"]
    fit_predictions = np.asarray(old_classes + new_classes)[
        np.argmax(fit_adjusted, axis=1)
    ]
    fit_old_mask = old[train]
    fit_before = legacy._metric_block(
        labels[train & old], fit_before_predictions, old_classes
    )
    fit_after = legacy._metric_block(
        labels[train][fit_old_mask], fit_predictions[fit_old_mask], old_classes
    )
    classwise_pass = all(
        float(fit_after["per_class_accuracy"][name]) + 1.0e-12
        >= float(fit_before["per_class_accuracy"][name])
        for name in old_classes
    )
    floor_pass = (
        float(fit_after["class_floor_accuracy"]) + 1.0e-12
        >= float(fit_before["class_floor_accuracy"])
    )
    training_trace = list(fit["before_fit"].loss_trace) + list(
        fit["stage2c_fit"].loss_trace
    )
    base_resource = dict(after.resource_audit())
    dali_accounting = _d31_dali_state_accounting(fit["dali_state"])
    dali_resource = dali_accounting["dali_resource"]
    combined_resident_state = int(base_resource["persistent_state_bytes"]) + int(
        dali_accounting["actual_current_dali_state_bytes"]
    )
    projected_active_state = int(base_resource["persistent_state_bytes"]) + int(
        dali_accounting["projected_slim_dali_runtime_bytes"]
    )
    resource = {
        **base_resource,
        "schema": "cvs.phase2.d31_combined_resource.v1",
        "d31_suffix_resource": base_resource,
        **dali_accounting,
        "dali_enabled_by_old_support_gate": bool(fit["dali_enabled"]),
        "dali_old_support_gate": fit["dali_gate"],
        "actual_int8_component_used_for_prediction": bool(fit["dali_enabled"]),
        "full_bundle_resident_combined_state_bytes": combined_resident_state,
        "projected_slim_active_predictor_state_bytes": projected_active_state,
        "deployment_resource_primary_state_view": (
            "projected_slim_fixed_medoid_predictor_with_full_bundle_residency_disclosed"
        ),
        "deployable_predictor_state_bytes_projected_slim_medoid": projected_active_state,
        "persistent_state_bytes": combined_resident_state,
        "persistent_state_cap_pass": combined_resident_state <= 256 * 1024,
        "estimated_macs_per_query": int(base_resource["estimated_macs_per_query"])
        + (
            int(dali_resource["fixed_medoid_ground_macs_per_query"])
            if fit["dali_enabled"]
            else 0
        ),
        "total_optimizer_steps": int(after.stage2b_optimizer_steps)
        + int(after.stage2c_optimizer_steps),
        "total_adaptation_epochs": int(after.stage2b_optimizer_steps)
        + int(after.stage2c_optimizer_steps),
        "complete_loss_trace": training_trace,
        "feature_geometry": "b3_auxiliary_dominant_z160_fft96_rf32_v1",
        "old_support_non_degradation_pass": bool(classwise_pass and floor_pass),
        "query_rows_used_for_fit": 0,
        "query_role_oracle_access": False,
        "query_true_batch_class_count_access": False,
        "query_class_quota_access": False,
        "query_batch_global_assignment": False,
        "dense_query_graph_bytes": 0,
        "clean_sample_access": False,
        "source_sample_access": False,
    }
    confusion_raw = _d31_confusion_audit(
        held_raw, held_labels, old_classes, new_classes
    )
    confusion_final = _d31_confusion_audit(
        held_adjusted, held_labels, old_classes, new_classes
    )
    geometry = {
        "schema": "cvs.phase2.d31_all_registered_geometry.v1",
        "feature_geometry": "b3_auxiliary_dominant_z160_fft96_rf32_v1",
        "observed_feature_block_energy": _d30_observed_block_energy(features),
        "old_prefix_sha256": after.old_prefix_sha256,
        "dali_enabled": bool(fit["dali_enabled"]),
        "dali_old_support_gate": fit["dali_gate"],
        "raw_confusion": confusion_raw,
        "final_confusion": confusion_final,
        "support_gate": json.loads(after.support_gate_json),
    }
    return {
        "candidate_id": candidate_id,
        "held_ranks": list(held_ranks),
        "fit_k_shot": 8,
        "before_old": before_old,
        "after_old": after_old,
        "after_new": after_new,
        "H_old_new": legacy._harmonic(
            float(after_old["overall_accuracy"]),
            float(after_new["overall_accuracy"]),
        ),
        "forgetting": float(
            before_old["overall_accuracy"] - after_old["overall_accuracy"]
        ),
        "joint_floor": float(
            min(after_old["class_floor_accuracy"], after_new["class_floor_accuracy"])
        ),
        "old_score_columns_bitwise_unchanged": True,
        "old_prefix_sha256_before": before.old_lock_sha256,
        "old_prefix_sha256_after": after.base_old_lock_sha256,
        "fit_old_before_registration": fit_before,
        "fit_old_after_registration": fit_after,
        "old_support_classwise_non_degradation": classwise_pass,
        "old_support_floor_non_degradation": floor_pass,
        "old_support_non_degradation_pass": bool(classwise_pass and floor_pass),
        "dali_enabled": bool(fit["dali_enabled"]),
        "dali_old_support_gate": fit["dali_gate"],
        "raw_confusion": confusion_raw,
        "final_confusion": confusion_final,
        "training_trace": training_trace,
        "geometry_summary": geometry,
        "resource": resource,
    }


def _evaluate_d33_fold(
    rows: Mapping[str, np.ndarray],
    z_id160: np.ndarray,
    fft96: np.ndarray,
    rf32: np.ndarray,
    *,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    held_ranks: tuple[int, int],
    candidate_id: str,
    config: D33CandidateConfig,
) -> dict[str, Any]:
    """Leave-two-out D33 evaluation with no held-row fitting or selection."""

    labels = np.asarray(rows["labels"]).astype(str)
    ranks = np.asarray(rows["ranks"], dtype=np.int64)
    held = np.isin(ranks, np.asarray(held_ranks, dtype=np.int64))
    train = ~held
    old = np.isin(labels, np.asarray(old_classes))
    new = np.isin(labels, np.asarray(new_classes))
    if (
        int(np.sum(train & old)) != 8 * len(old_classes)
        or int(np.sum(train & new)) != 8 * len(new_classes)
        or int(np.sum(held & old)) != 2 * len(old_classes)
        or int(np.sum(held & new)) != 2 * len(new_classes)
    ):
        raise D25RunnerError("D33 leave-two-out class symmetry drift")
    features = _d1_feature_from_blocks(z_id160, fft96, rf32)
    fit = _fit_d33_route(
        features[train],
        labels[train],
        old[train],
        new[train],
        old_classes,
        new_classes,
        config,
    )
    after = fit["after"]
    all_classes = old_classes + new_classes
    before_held_scores = _score_d33_old_stage(fit, features[held & old])
    before_predictions = np.asarray(old_classes)[
        np.argmax(before_held_scores, axis=1)
    ]
    held_scores = score_d33_spherical_registration(after, features[held])
    held_labels = labels[held]
    held_predictions = np.asarray(all_classes)[np.argmax(held_scores, axis=1)]
    held_old = np.isin(held_labels, np.asarray(old_classes))
    held_new = ~held_old
    before_old = legacy._metric_block(
        labels[held & old], before_predictions.astype(str).tolist(), old_classes
    )
    after_old = legacy._metric_block(
        held_labels[held_old], held_predictions[held_old].astype(str).tolist(), old_classes
    )
    after_new = legacy._metric_block(
        held_labels[held_new], held_predictions[held_new].astype(str).tolist(), new_classes
    )
    fit_before_scores = _score_d33_old_stage(fit, features[train & old])
    fit_before_predictions = np.asarray(old_classes)[
        np.argmax(fit_before_scores, axis=1)
    ]
    fit_after_scores = score_d33_spherical_registration(after, features[train])
    fit_after_predictions = np.asarray(all_classes)[np.argmax(fit_after_scores, axis=1)]
    fit_old = old[train]
    fit_before_metric = legacy._metric_block(
        labels[train & old], fit_before_predictions.astype(str).tolist(), old_classes
    )
    fit_after_metric = legacy._metric_block(
        labels[train][fit_old],
        fit_after_predictions[fit_old].astype(str).tolist(),
        old_classes,
    )
    classwise_pass = all(
        float(fit_after_metric["per_class_accuracy"][name]) + 1.0e-12
        >= float(fit_before_metric["per_class_accuracy"][name])
        for name in old_classes
    )
    floor_pass = (
        float(fit_after_metric["class_floor_accuracy"]) + 1.0e-12
        >= float(fit_before_metric["class_floor_accuracy"])
    )
    resource = _d33_resource(fit, len(all_classes))
    resource["old_support_non_degradation_pass"] = bool(
        classwise_pass and floor_pass
    )
    before_all_old_scores = _score_d33_old_stage(fit, features[held])
    raw_old_unchanged = bool(
        np.array_equal(before_all_old_scores, held_scores[:, : len(old_classes)])
    )
    confusion = _d31_confusion_audit(
        held_scores, held_labels, old_classes, new_classes
    )
    geometry = {
        "schema": "cvs.phase2.d33_spherical_geometry.v1",
        "feature_geometry": "b3_auxiliary_dominant_z160_fft96_rf32_v1",
        "observed_feature_block_energy": _d30_observed_block_energy(features),
        "old_solver": config.old_solver,
        "selection_policy": config.registration.selection_policy,
        "raw_confusion": confusion,
        "final_confusion": confusion,
        "base_old_parameter_prefix_bitwise_unchanged": True,
        "raw_old_score_columns_bitwise_unchanged_after_registration": raw_old_unchanged,
        "final_old_score_columns_bitwise_unchanged_after_registration": raw_old_unchanged,
        "final_old_score_transform_policy": "none_after_spherical_score",
    }
    return {
        "candidate_id": candidate_id,
        "held_ranks": list(held_ranks),
        "fit_k_shot": 8,
        "before_old": before_old,
        "after_old": after_old,
        "after_new": after_new,
        "H_old_new": legacy._harmonic(
            float(after_old["overall_accuracy"]),
            float(after_new["overall_accuracy"]),
        ),
        "forgetting": float(
            before_old["overall_accuracy"] - after_old["overall_accuracy"]
        ),
        "joint_floor": float(
            min(after_old["class_floor_accuracy"], after_new["class_floor_accuracy"])
        ),
        # Compatibility alias required by legacy._aggregate_candidate.  It is
        # the measured raw pre/post-registration equality, never a forced True.
        "old_score_columns_bitwise_unchanged": raw_old_unchanged,
        "old_score_columns_bitwise_unchanged_semantics": (
            "raw_old_scores_before_vs_after_spherical_registration"
        ),
        "base_old_parameter_prefix_bitwise_unchanged": True,
        "raw_old_score_columns_bitwise_unchanged_after_registration": raw_old_unchanged,
        "final_old_score_columns_bitwise_unchanged_after_registration": raw_old_unchanged,
        "final_old_score_transform_policy": "none_after_spherical_score",
        "fit_old_before_registration": fit_before_metric,
        "fit_old_after_registration": fit_after_metric,
        "old_support_classwise_non_degradation": classwise_pass,
        "old_support_floor_non_degradation": floor_pass,
        "old_support_non_degradation_pass": bool(classwise_pass and floor_pass),
        "raw_confusion": confusion,
        "final_confusion": confusion,
        "training_trace": list(resource["complete_loss_trace"]),
        "geometry_summary": geometry,
        "resource": resource,
    }


def _evaluate_d34_fold(
    rows: Mapping[str, np.ndarray],
    z_id160: np.ndarray,
    fft96: np.ndarray,
    rf32: np.ndarray,
    *,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    held_ranks: tuple[int, int],
    candidate_id: str,
    config: D34CandidateConfig,
) -> dict[str, Any]:
    """D34 leave-two-out evidence with frozen FAST old-score columns."""

    labels = np.asarray(rows["labels"]).astype(str)
    ranks = np.asarray(rows["ranks"], dtype=np.int64)
    held = np.isin(ranks, np.asarray(held_ranks, dtype=np.int64))
    train = ~held
    old = np.isin(labels, np.asarray(old_classes))
    new = np.isin(labels, np.asarray(new_classes))
    if (
        int(np.sum(train & old)) != 8 * len(old_classes)
        or int(np.sum(train & new)) != 8 * len(new_classes)
        or int(np.sum(held & old)) != 2 * len(old_classes)
        or int(np.sum(held & new)) != 2 * len(new_classes)
    ):
        raise D25RunnerError("D34 leave-two-out class symmetry drift")
    features = _d1_feature_from_blocks(z_id160, fft96, rf32)
    fit = _fit_d34_route(
        features[train],
        labels[train],
        old[train],
        new[train],
        old_classes,
        new_classes,
        config,
    )
    all_classes = old_classes + new_classes
    held_old_prefix, held_scores = _score_d34(fit, features[held])
    if not np.array_equal(
        held_old_prefix, held_scores[:, : len(old_classes)]
    ):
        raise D25RunnerError("D34 changed held old score prefix")
    held_labels = labels[held]
    held_predictions = np.asarray(all_classes)[np.argmax(held_scores, axis=1)]
    held_old = np.isin(held_labels, np.asarray(old_classes))
    held_new = ~held_old
    before_old_predictions = np.asarray(old_classes)[
        np.argmax(held_old_prefix[held_old], axis=1)
    ]
    before_old = legacy._metric_block(
        held_labels[held_old], before_old_predictions.astype(str).tolist(), old_classes
    )
    after_old = legacy._metric_block(
        held_labels[held_old], held_predictions[held_old].astype(str).tolist(), old_classes
    )
    after_new = legacy._metric_block(
        held_labels[held_new], held_predictions[held_new].astype(str).tolist(), new_classes
    )
    before_old_correct = before_old_predictions.astype(str) == held_labels[held_old]
    old_loso_intrusion_count = int(
        np.sum(
            before_old_correct
            & (held_predictions[held_old].astype(str) != before_old_predictions.astype(str))
        )
    )

    fit_old_prefix, fit_scores = _score_d34(fit, features[train])
    if not np.array_equal(fit_old_prefix, fit_scores[:, : len(old_classes)]):
        raise D25RunnerError("D34 changed fit old score prefix")
    fit_predictions = np.asarray(all_classes)[np.argmax(fit_scores, axis=1)]
    fit_old = old[train]
    fit_before_predictions = np.asarray(old_classes)[
        np.argmax(fit_old_prefix[fit_old], axis=1)
    ]
    fit_before = legacy._metric_block(
        labels[train][fit_old], fit_before_predictions.astype(str).tolist(), old_classes
    )
    fit_after = legacy._metric_block(
        labels[train][fit_old], fit_predictions[fit_old].astype(str).tolist(), old_classes
    )
    classwise_pass = all(
        float(fit_after["per_class_accuracy"][name]) + 1.0e-12
        >= float(fit_before["per_class_accuracy"][name])
        for name in old_classes
    )
    floor_pass = (
        float(fit_after["class_floor_accuracy"]) + 1.0e-12
        >= float(fit_before["class_floor_accuracy"])
    )
    resource = _d34_resource(fit, len(all_classes))
    core_old_loso_flag = bool(resource["old_loso_zero_intrusion_pass"])
    old_loso_pass = old_loso_intrusion_count == 0
    old_support_pass = bool(classwise_pass and floor_pass)
    resource["old_support_non_degradation_pass"] = old_support_pass
    resource["core_internal_old_loso_zero_intrusion_flag"] = core_old_loso_flag
    resource["old_loso_intrusion_count"] = old_loso_intrusion_count
    resource["old_loso_zero_intrusion_pass"] = old_loso_pass
    resource["complete_loss_trace"].append(
        {
            "audit": "outer_leave_two_rank_old_intrusion",
            "held_ranks": list(held_ranks),
            "before_correct_old_count": int(np.sum(before_old_correct)),
            "intrusion_count": old_loso_intrusion_count,
            "zero_intrusion_pass": old_loso_pass,
            "query_rows_used": 0,
        }
    )
    geometry = {
        **dict(fit["geometry"]),
        "schema": "cvs.phase2.d34_collision_local_geometry.v1",
        "feature_geometry": "b3_auxiliary_dominant_z160_fft96_rf32_v1",
        "observed_feature_block_energy": _d30_observed_block_energy(features),
        "arm": str(config.registration.arm),
        "old_score_prefix_bitwise_unchanged": True,
        "core_internal_old_loso_zero_intrusion_flag": core_old_loso_flag,
        "old_loso_protocol": "outer_leave_two_shot_ranks_not_seen_by_fit",
        "old_loso_intrusion_count": old_loso_intrusion_count,
        "old_loso_zero_intrusion_pass": old_loso_pass,
    }
    return {
        "candidate_id": candidate_id,
        "held_ranks": list(held_ranks),
        "fit_k_shot": 8,
        "before_old": before_old,
        "after_old": after_old,
        "after_new": after_new,
        "H_old_new": legacy._harmonic(
            float(after_old["overall_accuracy"]),
            float(after_new["overall_accuracy"]),
        ),
        "forgetting": float(
            before_old["overall_accuracy"] - after_old["overall_accuracy"]
        ),
        "joint_floor": float(
            min(after_old["class_floor_accuracy"], after_new["class_floor_accuracy"])
        ),
        "old_score_columns_bitwise_unchanged": True,
        "old_score_prefix_bitwise_unchanged": True,
        "old_score_prefix_bitwise_unchanged_semantics": (
            "FAST_score_prefix_passed_unchanged_to_D34_registration"
        ),
        "fit_old_before_registration": fit_before,
        "fit_old_after_registration": fit_after,
        "old_support_classwise_non_degradation": classwise_pass,
        "old_support_floor_non_degradation": floor_pass,
        "old_support_non_degradation_pass": old_support_pass,
        "old_loso_zero_intrusion_pass": old_loso_pass,
        "old_loso_intrusion_count": old_loso_intrusion_count,
        "collision_edge_count": int(resource["collision_edge_count"]),
        "unreachable_edge_count": int(resource["unreachable_edge_count"]),
        "training_trace": list(resource["complete_loss_trace"]),
        "geometry_summary": geometry,
        "resource": resource,
    }


def _evaluate_d35_fold(
    rows: Mapping[str, np.ndarray],
    z_id160: np.ndarray,
    fft96: np.ndarray,
    rf32: np.ndarray,
    *,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    held_ranks: tuple[int, int],
    candidate_id: str,
    config: D35CandidateConfig,
) -> dict[str, Any]:
    """D35 outer leave-two-out evidence with a frozen FAST old prefix."""

    labels = np.asarray(rows["labels"]).astype(str)
    ranks = np.asarray(rows["ranks"], dtype=np.int64)
    held = np.isin(ranks, np.asarray(held_ranks, dtype=np.int64))
    train = ~held
    old = np.isin(labels, np.asarray(old_classes))
    new = np.isin(labels, np.asarray(new_classes))
    if (
        int(np.sum(train & old)) != 8 * len(old_classes)
        or int(np.sum(train & new)) != 8 * len(new_classes)
        or int(np.sum(held & old)) != 2 * len(old_classes)
        or int(np.sum(held & new)) != 2 * len(new_classes)
    ):
        raise D25RunnerError("D35 leave-two-out class symmetry drift")
    features = _d1_feature_from_blocks(z_id160, fft96, rf32)
    fit = _fit_d35_route(
        features[train],
        labels[train],
        old[train],
        new[train],
        old_classes,
        new_classes,
        config,
    )
    all_classes = old_classes + new_classes
    held_old_prefix, held_scores = _score_d35(fit, features[held])
    if not np.array_equal(held_old_prefix, held_scores[:, : len(old_classes)]):
        raise D25RunnerError("D35 changed held old score prefix")
    held_labels = labels[held]
    held_predictions = np.asarray(all_classes)[np.argmax(held_scores, axis=1)]
    held_old = np.isin(held_labels, np.asarray(old_classes))
    held_new = ~held_old
    before_old_predictions = np.asarray(old_classes)[
        np.argmax(held_old_prefix[held_old], axis=1)
    ]
    before_old = legacy._metric_block(
        held_labels[held_old], before_old_predictions.astype(str).tolist(), old_classes
    )
    after_old = legacy._metric_block(
        held_labels[held_old], held_predictions[held_old].astype(str).tolist(), old_classes
    )
    after_new = legacy._metric_block(
        held_labels[held_new], held_predictions[held_new].astype(str).tolist(), new_classes
    )
    before_old_correct = before_old_predictions.astype(str) == held_labels[held_old]
    outer_new_intrusion_count = int(
        np.sum(
            before_old_correct
            & np.isin(
                held_predictions[held_old].astype(str), np.asarray(new_classes)
            )
        )
    )

    fit_old_prefix, fit_scores = _score_d35(fit, features[train])
    if not np.array_equal(fit_old_prefix, fit_scores[:, : len(old_classes)]):
        raise D25RunnerError("D35 changed fit old score prefix")
    fit_predictions = np.asarray(all_classes)[np.argmax(fit_scores, axis=1)]
    fit_old = old[train]
    fit_before_predictions = np.asarray(old_classes)[
        np.argmax(fit_old_prefix[fit_old], axis=1)
    ]
    fit_before = legacy._metric_block(
        labels[train][fit_old], fit_before_predictions.astype(str).tolist(), old_classes
    )
    fit_after = legacy._metric_block(
        labels[train][fit_old], fit_predictions[fit_old].astype(str).tolist(), old_classes
    )
    classwise_pass = all(
        float(fit_after["per_class_accuracy"][name]) + 1.0e-12
        >= float(fit_before["per_class_accuracy"][name])
        for name in old_classes
    )
    floor_pass = (
        float(fit_after["class_floor_accuracy"]) + 1.0e-12
        >= float(fit_before["class_floor_accuracy"])
    )
    fit_old_pass = bool(classwise_pass and floor_pass)
    resource = _d35_resource(fit, len(all_classes))
    reachability = dict(resource["new_class_reachability"])
    all_reachable = bool(
        set(reachability) == set(new_classes) and all(reachability.values())
    )
    resource.update(
        {
            "fit_old_support_non_degradation_pass": fit_old_pass,
            "outer_held_new_intrusion_count": outer_new_intrusion_count,
            "outer_held_zero_new_intrusion_pass": outer_new_intrusion_count == 0,
            "new_physical_loso_all_reachable": all_reachable,
        }
    )
    resource["complete_loss_trace"].append(
        {
            "audit": "outer_leave_two_rank_new_intrusion",
            "held_ranks": list(held_ranks),
            "before_correct_old_count": int(np.sum(before_old_correct)),
            "new_intrusion_count": outer_new_intrusion_count,
            "zero_new_intrusion_pass": outer_new_intrusion_count == 0,
            "query_rows_used": 0,
        }
    )
    geometry = {
        **dict(fit["geometry"]),
        "schema": "cvs.phase2.d35_dense_safe_geometry.v1",
        "feature_geometry": "b3_auxiliary_dominant_z160_fft96_rf32_v1",
        "observed_feature_block_energy": _d30_observed_block_energy(features),
        "arm": str(config.registration.arm),
        "old_score_prefix_bitwise_unchanged": True,
        "outer_held_protocol": "outer_leave_two_shot_ranks_not_seen_by_fit",
        "outer_held_new_intrusion_count": outer_new_intrusion_count,
        "outer_held_zero_new_intrusion_pass": outer_new_intrusion_count == 0,
        "new_class_reachability": reachability,
        "new_physical_loso_all_reachable": all_reachable,
    }
    return {
        "candidate_id": candidate_id,
        "held_ranks": list(held_ranks),
        "fit_k_shot": 8,
        "before_old": before_old,
        "after_old": after_old,
        "after_new": after_new,
        "H_old_new": legacy._harmonic(
            float(after_old["overall_accuracy"]),
            float(after_new["overall_accuracy"]),
        ),
        "forgetting": float(
            before_old["overall_accuracy"] - after_old["overall_accuracy"]
        ),
        "joint_floor": float(
            min(after_old["class_floor_accuracy"], after_new["class_floor_accuracy"])
        ),
        "old_score_columns_bitwise_unchanged": True,
        "old_score_prefix_bitwise_unchanged": True,
        "fit_old_before_registration": fit_before,
        "fit_old_after_registration": fit_after,
        "fit_old_support_classwise_non_degradation": classwise_pass,
        "fit_old_support_floor_non_degradation": floor_pass,
        "fit_old_support_non_degradation_pass": fit_old_pass,
        "outer_held_new_intrusion_count": outer_new_intrusion_count,
        "outer_held_zero_new_intrusion_pass": outer_new_intrusion_count == 0,
        "new_class_reachability": reachability,
        "new_physical_loso_all_reachable": all_reachable,
        "unreachable_new_class_count": int(sum(not v for v in reachability.values())),
        "training_trace": list(resource["complete_loss_trace"]),
        "geometry_summary": geometry,
        "resource": resource,
    }


def _d37_old_to_new_intrusion_count(
    predictions: np.ndarray,
    old_mask: np.ndarray,
    new_classes: tuple[str, ...],
) -> int:
    """Count every true-old row assigned to a newly registered class."""

    values = np.asarray(predictions).astype(str)
    mask = np.asarray(old_mask, dtype=bool)
    if values.shape != mask.shape:
        raise D25RunnerError("D37 intrusion prediction/mask shape drift")
    return int(np.sum(np.isin(values[mask], np.asarray(new_classes))))


def _evaluate_d37_fold(
    rows: Mapping[str, np.ndarray],
    z_id160: np.ndarray,
    fft96: np.ndarray,
    rf32: np.ndarray,
    *,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    held_ranks: tuple[int, int],
    candidate_id: str,
    config: D37CandidateConfig,
) -> dict[str, Any]:
    """D37 outer fold; an empty OOF interval is an audited failed row."""

    labels = np.asarray(rows["labels"]).astype(str)
    ranks = np.asarray(rows["ranks"], dtype=np.int64)
    held = np.isin(ranks, np.asarray(held_ranks, dtype=np.int64))
    train = ~held
    old = np.isin(labels, np.asarray(old_classes))
    new = np.isin(labels, np.asarray(new_classes))
    if (
        int(np.sum(train & old)) != 8 * len(old_classes)
        or int(np.sum(train & new)) != 8 * len(new_classes)
        or int(np.sum(held & old)) != 2 * len(old_classes)
        or int(np.sum(held & new)) != 2 * len(new_classes)
    ):
        raise D25RunnerError("D37 leave-two-rank class symmetry drift")
    features = _d1_feature_from_blocks(z_id160, fft96, rf32)
    fit = _fit_d37_route(
        features[train],
        labels[train],
        ranks[train],
        np.asarray(rows["tokens"])[train],
        old[train],
        new[train],
        old_classes,
        new_classes,
        config,
    )
    all_classes = old_classes + new_classes
    before_scores, after_scores = _score_d37(fit, features[held])
    held_labels = labels[held]
    held_old = old[held]
    held_new = new[held]
    before_predictions = np.asarray(old_classes)[
        np.argmax(before_scores[held_old], axis=1)
    ]
    after_predictions = np.asarray(all_classes)[np.argmax(after_scores, axis=1)]
    before_old = legacy._metric_block(
        held_labels[held_old], before_predictions.astype(str).tolist(), old_classes
    )
    after_old = legacy._metric_block(
        held_labels[held_old], after_predictions[held_old].astype(str).tolist(), old_classes
    )
    after_new = legacy._metric_block(
        held_labels[held_new], after_predictions[held_new].astype(str).tolist(), new_classes
    )
    b3_scores = score_b3_fisher_closed_form(
        fit["fisher_fit"].state, features[held][held_old]
    )
    b3_predictions = np.asarray(old_classes)[np.argmax(b3_scores, axis=1)]
    b3_old = legacy._metric_block(
        held_labels[held_old], b3_predictions.astype(str).tolist(), old_classes
    )
    outer_intrusion = _d37_old_to_new_intrusion_count(
        after_predictions, held_old, new_classes
    )
    new_loso: list[dict[str, Any]] = []
    for local_index in np.flatnonzero(held_new).tolist():
        target = len(old_classes) + new_classes.index(str(held_labels[local_index]))
        scores = after_scores[local_index]
        competitor = float(np.max(np.delete(scores, target)))
        margin = float(scores[target] - competitor)
        new_loso.append(
            {
                "held_rank": int(ranks[held][local_index]),
                "new_class": str(held_labels[local_index]),
                "correct": bool(np.argmax(scores) == target),
                "margin": margin,
            }
        )
    new_reachability = {
        name: bool(values)
        and all(bool(row["correct"]) and float(row["margin"]) > 0.0 for row in values)
        for name in new_classes
        for values in [[row for row in new_loso if row["new_class"] == name]]
    }
    resource = _d37_resource(fit, len(all_classes))
    old_score_columns_bitwise_unchanged = bool(
        old_prefix_bitwise_unchanged_d37(
            fit["before_state"], fit["state_no_offset"]
        )
    )
    old_score_column_max_abs_diff = float(
        np.max(np.abs(before_scores - after_scores[:, : len(old_classes)]))
    )
    geometry = {
        **dict(fit["core_result"].geometry_audit),
        "schema": "cvs.phase2.d37_b3_preserving_int8_geometry.v1",
        "feature_geometry": "b3_auxiliary_dominant_z160_fft96_rf32_v1",
        "observed_feature_block_energy": _d30_observed_block_energy(features),
        "inner_crossfit_rank_pairs": [list(pair) for pair in fit["inner_pairs"]],
        "inner_crossfit_no_self_participation": True,
        "oof_base_score_row_count": len(fit["oof_labels"]),
        "oof_uses_physical_labels_only": True,
        "oof_feasible_interval_pass": bool(fit["oof_feasible_interval_pass"]),
        "oof_failure_reason": fit["oof_failure_reason"],
        "outer_held_protocol": "outer_leave_two_shot_ranks_not_seen_by_fit",
        "outer_held_new_intrusion_count": outer_intrusion,
        "outer_held_zero_new_intrusion_pass": outer_intrusion == 0,
        "new_physical_leave_one_out": new_loso,
        "new_class_reachability": new_reachability,
        "new_physical_loso_all_reachable": all(new_reachability.values()),
    }
    return {
        "candidate_id": candidate_id,
        "held_ranks": list(held_ranks),
        "fit_k_shot": 8,
        "before_old": before_old,
        "after_old": after_old,
        "after_new": after_new,
        "b3_reference_old": b3_old,
        "H_old_new": legacy._harmonic(
            float(after_old["overall_accuracy"]),
            float(after_new["overall_accuracy"]),
        ),
        "forgetting": float(
            before_old["overall_accuracy"] - after_old["overall_accuracy"]
        ),
        "joint_floor": float(
            min(after_old["class_floor_accuracy"], after_new["class_floor_accuracy"])
        ),
        "old_score_columns_bitwise_unchanged": old_score_columns_bitwise_unchanged,
        "old_score_columns_bitwise_unchanged_semantics": (
            "measured_append_only_target_old_int8_state_prefix"
        ),
        "old_score_column_max_abs_diff": old_score_column_max_abs_diff,
        "oof_feasible_interval_pass": bool(fit["oof_feasible_interval_pass"]),
        "oof_feasible_interval_lower_bound": fit["oof_feasible_interval_lower_bound"],
        "oof_feasible_interval_upper_bound": fit["oof_feasible_interval_upper_bound"],
        "oof_failure_reason": fit["oof_failure_reason"],
        "outer_held_new_intrusion_count": outer_intrusion,
        "outer_held_zero_new_intrusion_pass": outer_intrusion == 0,
        "new_class_reachability": new_reachability,
        "new_physical_loso_all_reachable": all(new_reachability.values()),
        "unreachable_new_class_count": int(
            sum(not value for value in new_reachability.values())
        ),
        "target_old_int8_prototypes_used_for_prediction": True,
        "target_new_int8_prototypes_used_for_prediction": True,
        "training_trace": list(resource["complete_loss_trace"]),
        "geometry_summary": geometry,
        "resource": resource,
    }


def _d38_direct_old_anchor(
    rows: Mapping[str, np.ndarray],
    direct_logits: np.ndarray,
    *,
    old_classes: tuple[str, ...],
    held_ranks: tuple[int, int] | None,
) -> dict[str, Any]:
    """Read-only ADV3B02 old-only anchor; never a candidate row."""

    labels = np.asarray(rows["labels"]).astype(str)
    ranks = np.asarray(rows["ranks"], dtype=np.int64)
    old = np.isin(labels, np.asarray(old_classes))
    held = (
        np.ones(len(labels), dtype=bool)
        if held_ranks is None
        else np.isin(ranks, np.asarray(held_ranks, dtype=np.int64))
    )
    logits = np.asarray(direct_logits, dtype=np.float32)
    if logits.shape != (len(labels), len(old_classes)):
        raise D25RunnerError("D38 direct ADV3B02 anchor column binding drift")
    mask = held & old
    predictions = np.asarray(old_classes)[np.argmax(logits[mask], axis=1)]
    return {
        "anchor_id": "DIRECT_ADV3B02_OLD_ONLY_ZERO_SUPPORT",
        "candidate_row": False,
        "support_rows_used": 0,
        "old_only": True,
        "held_ranks": None if held_ranks is None else list(held_ranks),
        "metrics": legacy._metric_block(
            labels[mask], predictions.astype(str).tolist(), old_classes
        ),
        "query_rows_used": 0,
    }


def _evaluate_d38_fold(
    rows: Mapping[str, np.ndarray],
    z_id160: np.ndarray,
    fft96: np.ndarray,
    rf32: np.ndarray,
    *,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    held_ranks: tuple[int, int],
    candidate_id: str,
    config: D38CandidateConfig,
    seed: int,
    device: torch.device | str = "cpu",
    scenario: str = "unit_test_scene",
    outer_fold: int | None = None,
) -> dict[str, Any]:
    """D38 outer 8-shot fit/2-shot held evaluation with matched FP32 audit."""

    labels = np.asarray(rows["labels"]).astype(str)
    ranks = np.asarray(rows["ranks"], dtype=np.int64)
    tokens = np.asarray(rows["tokens"]).astype(str)
    held = np.isin(ranks, np.asarray(held_ranks, dtype=np.int64))
    train = ~held
    old = np.isin(labels, np.asarray(old_classes))
    new = np.isin(labels, np.asarray(new_classes))
    if (
        int(np.sum(train & old)) != 8 * len(old_classes)
        or int(np.sum(train & new)) != 8 * len(new_classes)
        or int(np.sum(held & old)) != 2 * len(old_classes)
        or int(np.sum(held & new)) != 2 * len(new_classes)
    ):
        raise D25RunnerError("D38 leave-two-rank class symmetry drift")
    features = _d1_feature_from_blocks(z_id160, fft96, rf32)
    result = fit_d38_strong_b3_quantized(
        features[train & old],
        labels[train & old],
        old_classes,
        features[train & new],
        labels[train & new],
        new_classes,
        seed=int(seed),
        device=device,
        config=config.core,
    )
    deployed_state = (
        result.state if config.deploy_precision == "int8" else result.matched_fp32_state
    )
    held_features = features[held]
    held_labels = labels[held]
    held_old = old[held]
    held_new = new[held]
    all_classes = old_classes + new_classes
    before_scores = score_d38_strong_b3(result.before_state, held_features[held_old])
    after_scores = score_d38_strong_b3(deployed_state, held_features)
    int8_scores = score_d38_strong_b3(result.state, held_features)
    fp32_scores = score_d38_strong_b3(result.matched_fp32_state, held_features)
    before_predictions = np.asarray(old_classes)[np.argmax(before_scores, axis=1)]
    after_predictions = np.asarray(all_classes)[np.argmax(after_scores, axis=1)]
    before_old = legacy._metric_block(
        held_labels[held_old], before_predictions.astype(str).tolist(), old_classes
    )
    after_old = legacy._metric_block(
        held_labels[held_old], after_predictions[held_old].astype(str).tolist(), old_classes
    )
    after_new = legacy._metric_block(
        held_labels[held_new], after_predictions[held_new].astype(str).tolist(), new_classes
    )
    pairwise = pairwise_support_diagnostics_d38(
        deployed_state,
        held_features[held_new],
        held_labels[held_new],
        tokens[held][held_new],
        scenario=scenario,
        outer_fold=(
            int(HELD_RANKS.index(held_ranks))
            if outer_fold is None
            else int(outer_fold)
        ),
        physical_ranks=ranks[held][held_new],
    )
    margins = np.asarray([float(row["new_new_margin"]) for row in pairwise])
    intrusion = _d37_old_to_new_intrusion_count(after_predictions, held_old, new_classes)
    argmax_changes = int(
        np.sum(np.argmax(int8_scores, axis=1) != np.argmax(fp32_scores, axis=1))
    )
    prefix_unchanged = old_prefix_bitwise_unchanged_d38(
        result.before_state, result.state
    )
    resource = dict(result.resource_audit)
    deployed_bytes = int(deployed_state.persistent_state_bytes)
    resource.update(
        {
            "deployment_precision": config.deploy_precision,
            "peak_trainable_parameters": int(resource["trainable_parameters"]),
            "total_optimizer_steps": int(resource["optimizer_steps"]),
            "persistent_state_bytes": deployed_bytes,
            "persistent_state_cap_pass": deployed_bytes <= 256 * 1024,
            "target_old_int8_prototypes_used_for_prediction": (
                config.deploy_precision == "int8"
            ),
            "target_new_int8_prototypes_used_for_prediction": (
                config.deploy_precision == "int8"
            ),
            "resident_fp32_target_prototype_count": (
                0 if config.deploy_precision == "int8" else len(all_classes)
            ),
            "clean_sample_access": False,
            "source_sample_access": False,
            "old_prefix_bitwise_unchanged": prefix_unchanged,
            "matched_fp32_outer_argmax_change_count": argmax_changes,
            "pairwise_support_diagnostic_row_count": len(pairwise),
            "new_new_confusion_count": int(np.sum(margins <= 0.0)),
            "new_new_margin_min": float(np.min(margins)),
            "complete_loss_trace": [dict(row) for row in result.training_trace],
            "latency_includes_argmax": True,
        }
    )
    geometry = {
        **dict(result.geometry_audit),
        "schema": "cvs.phase2.d38.outer_geometry.v1",
        "observed_feature_block_energy": _d30_observed_block_energy(features),
        "outer_held_protocol": "outer_leave_two_shot_ranks_not_seen_by_fit",
        "pairwise_support_diagnostics": pairwise,
        "new_new_confusion_count": int(np.sum(margins <= 0.0)),
        "new_new_margin_min": float(np.min(margins)),
        "new_new_margin_mean": float(np.mean(margins)),
        "outer_held_new_intrusion_count": intrusion,
        "old_prefix_bitwise_unchanged": prefix_unchanged,
        "matched_fp32_outer_argmax_change_count": argmax_changes,
        "query_rows_used": 0,
    }
    return {
        "candidate_id": candidate_id,
        "held_ranks": list(held_ranks),
        "fit_k_shot": 8,
        "before_old": before_old,
        "after_old": after_old,
        "after_new": after_new,
        "H_old_new": legacy._harmonic(
            float(after_old["overall_accuracy"]),
            float(after_new["overall_accuracy"]),
        ),
        "forgetting": float(
            before_old["overall_accuracy"] - after_old["overall_accuracy"]
        ),
        "joint_floor": float(
            min(after_old["class_floor_accuracy"], after_new["class_floor_accuracy"])
        ),
        "old_score_columns_bitwise_unchanged": prefix_unchanged,
        "outer_held_new_intrusion_count": intrusion,
        "pairwise_support_diagnostics": pairwise,
        "new_new_confusion_count": int(np.sum(margins <= 0.0)),
        "new_new_margin_min": float(np.min(margins)),
        "new_new_margin_mean": float(np.mean(margins)),
        "matched_fp32_outer_argmax_change_count": argmax_changes,
        "target_old_int8_prototypes_used_for_prediction": (
            config.deploy_precision == "int8"
        ),
        "target_new_int8_prototypes_used_for_prediction": (
            config.deploy_precision == "int8"
        ),
        "deployment_precision": config.deploy_precision,
        "registration_before_prediction_sha256": hashlib.sha256(
            _canonical_bytes(before_predictions.astype(str).tolist())
        ).hexdigest(),
        "training_trace": [dict(row) for row in result.training_trace],
        "geometry_summary": geometry,
        "resource": resource,
    }


def _evaluate_d39_fold(
    rows: Mapping[str, np.ndarray],
    z_id160: np.ndarray,
    fft96: np.ndarray,
    rf32: np.ndarray,
    *,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    held_ranks: tuple[int, int],
    candidate_id: str,
    config: D39CandidateConfig,
    seed: int,
    device: torch.device | str = "cpu",
    scenario: str = "unit_test_scene",
    outer_fold: int | None = None,
) -> dict[str, Any]:
    """D39 outer 8-shot fit/2-shot held angular-radius evaluation."""

    labels = np.asarray(rows["labels"]).astype(str)
    ranks = np.asarray(rows["ranks"], dtype=np.int64)
    tokens = np.asarray(rows["tokens"]).astype(str)
    held = np.isin(ranks, np.asarray(held_ranks, dtype=np.int64))
    train = ~held
    old = np.isin(labels, np.asarray(old_classes))
    new = np.isin(labels, np.asarray(new_classes))
    if (
        int(np.sum(train & old)) != 8 * len(old_classes)
        or int(np.sum(train & new)) != 8 * len(new_classes)
        or int(np.sum(held & old)) != 2 * len(old_classes)
        or int(np.sum(held & new)) != 2 * len(new_classes)
    ):
        raise D25RunnerError("D39 leave-two-rank class symmetry drift")
    features = _d1_feature_from_blocks(z_id160, fft96, rf32)
    result = fit_d39_angular_radius(
        features[train & old],
        labels[train & old],
        old_classes,
        features[train & new],
        labels[train & new],
        new_classes,
        seed=int(seed),
        device=device,
        config=config.core,
    )
    deployed_state = (
        result.state if config.deploy_precision == "int8" else result.matched_fp32_state
    )
    held_features = features[held]
    held_labels = labels[held]
    held_old = old[held]
    held_new = new[held]
    all_classes = old_classes + new_classes
    before_scores = score_d39_angular_radius(
        result.before_state, held_features[held_old]
    )
    after_scores = score_d39_angular_radius(deployed_state, held_features)
    int8_scores = score_d39_angular_radius(result.state, held_features)
    fp32_scores = score_d39_angular_radius(
        result.matched_fp32_state, held_features
    )
    before_predictions = np.asarray(old_classes)[np.argmax(before_scores, axis=1)]
    after_predictions = np.asarray(all_classes)[np.argmax(after_scores, axis=1)]
    before_old = legacy._metric_block(
        held_labels[held_old], before_predictions.astype(str).tolist(), old_classes
    )
    after_old = legacy._metric_block(
        held_labels[held_old], after_predictions[held_old].astype(str).tolist(), old_classes
    )
    after_new = legacy._metric_block(
        held_labels[held_new], after_predictions[held_new].astype(str).tolist(), new_classes
    )
    pairwise = pairwise_support_diagnostics_d39(
        deployed_state,
        held_features[held_new],
        held_labels[held_new],
        tokens[held][held_new],
        scenario=scenario,
        outer_fold=(
            int(HELD_RANKS.index(held_ranks))
            if outer_fold is None
            else int(outer_fold)
        ),
        physical_ranks=ranks[held][held_new],
    )
    margins = np.asarray([float(row["new_new_margin"]) for row in pairwise])
    intrusion = _d37_old_to_new_intrusion_count(
        after_predictions, held_old, new_classes
    )
    argmax_changes = int(
        np.sum(np.argmax(int8_scores, axis=1) != np.argmax(fp32_scores, axis=1))
    )
    prefix_unchanged = old_prefix_bitwise_unchanged_d39(
        result.before_state, result.state
    )
    old_prototype_prefix_unchanged = old_prefix_bitwise_unchanged_d38(
        result.before_state.base_state, result.state.base_state
    )
    old_radius_prefix_unchanged = bool(
        np.array_equal(
            result.before_state.radius_fp16,
            result.state.radius_fp16[: len(old_classes)],
        )
    )
    r0_unchanged = bool(
        np.array_equal(result.before_state.r0_fp16, result.state.r0_fp16)
    )
    radius_positive_finite = bool(
        np.isfinite(result.state.radius_fp16).all()
        and np.all(result.state.radius_fp16 > 0)
        and np.isfinite(result.state.r0_fp16).all()
        and np.all(result.state.r0_fp16 > 0)
    )
    radius_shared = bool(
        np.array_equal(
            result.state.radius_fp16, result.matched_fp32_state.radius_fp16
        )
        and np.array_equal(result.state.r0_fp16, result.matched_fp32_state.r0_fp16)
    )
    outer_prediction_sha256 = hashlib.sha256(
        _canonical_bytes(after_predictions.astype(str).tolist())
    ).hexdigest()
    radius_fp16_sha256 = hashlib.sha256(
        np.ascontiguousarray(deployed_state.radius_fp16).tobytes()
    ).hexdigest()
    r0_fp16_sha256 = hashlib.sha256(
        np.ascontiguousarray(deployed_state.r0_fp16).tobytes()
    ).hexdigest()
    old_source_tokens = sorted(tokens[train & old].tolist())
    new_source_tokens = sorted(tokens[train & new].tolist())
    held_tokens = set(tokens[held].tolist())
    radius_source_audit = {
        "old_source_physical_token_sha256": hashlib.sha256(
            _canonical_bytes(old_source_tokens)
        ).hexdigest(),
        "new_source_physical_token_sha256": hashlib.sha256(
            _canonical_bytes(new_source_tokens)
        ).hexdigest(),
        "old_source_row_count": len(old_source_tokens),
        "new_source_row_count": len(new_source_tokens),
        "old_source_held_intersection_count": len(set(old_source_tokens) & held_tokens),
        "new_source_held_intersection_count": len(set(new_source_tokens) & held_tokens),
        "old_source_new_class_row_count": int(np.sum(train & old & new)),
        "new_source_old_class_row_count": int(np.sum(train & new & old)),
        "query_rows_used": 0,
    }
    resource = dict(result.resource_audit)
    deployed_bytes = int(deployed_state.persistent_state_bytes)
    resource.update(
        {
            "deployment_precision": config.deploy_precision,
            "peak_trainable_parameters": int(resource["trainable_parameters"]),
            "total_optimizer_steps": int(resource["optimizer_steps"]),
            "persistent_state_bytes": deployed_bytes,
            "persistent_state_cap_pass": deployed_bytes <= 256 * 1024,
            "target_old_int8_prototypes_used_for_prediction": (
                config.deploy_precision == "int8"
            ),
            "target_new_int8_prototypes_used_for_prediction": (
                config.deploy_precision == "int8"
            ),
            "resident_fp32_target_prototype_count": (
                0 if config.deploy_precision == "int8" else len(all_classes)
            ),
            "clean_sample_access": False,
            "source_sample_access": False,
            "old_prefix_bitwise_unchanged": prefix_unchanged,
            "matched_fp32_outer_argmax_change_count": argmax_changes,
            "pairwise_support_diagnostic_row_count": len(pairwise),
            "new_new_confusion_count": int(np.sum(margins <= 0.0)),
            "new_new_margin_min": float(np.min(margins)),
            "complete_loss_trace": [dict(row) for row in result.training_trace],
            "latency_includes_argmax": True,
        }
    )
    geometry = {
        **dict(result.geometry_audit),
        "schema": "cvs.phase2.d39.outer_geometry.v1",
        "observed_feature_block_energy": _d30_observed_block_energy(features),
        "outer_held_protocol": "outer_leave_two_shot_ranks_not_seen_by_fit_or_radius",
        "pairwise_support_diagnostics": pairwise,
        "new_new_confusion_count": int(np.sum(margins <= 0.0)),
        "new_new_margin_min": float(np.min(margins)),
        "new_new_margin_mean": float(np.mean(margins)),
        "outer_held_new_intrusion_count": intrusion,
        "old_prefix_bitwise_unchanged": prefix_unchanged,
        "old_prototype_prefix_bitwise_unchanged": old_prototype_prefix_unchanged,
        "old_radius_prefix_bitwise_unchanged": old_radius_prefix_unchanged,
        "r0_bitwise_unchanged": r0_unchanged,
        "radius_positive_finite": radius_positive_finite,
        "radius_fp16_shared_between_int8_fp32": radius_shared,
        "outer_prediction_sha256": outer_prediction_sha256,
        "radius_fp16_sha256": radius_fp16_sha256,
        "r0_fp16_sha256": r0_fp16_sha256,
        "radius_source_audit": radius_source_audit,
        "matched_fp32_outer_argmax_change_count": argmax_changes,
        "query_rows_used": 0,
    }
    return {
        "candidate_id": candidate_id,
        "held_ranks": list(held_ranks),
        "fit_k_shot": 8,
        "before_old": before_old,
        "after_old": after_old,
        "after_new": after_new,
        "H_old_new": legacy._harmonic(
            float(after_old["overall_accuracy"]),
            float(after_new["overall_accuracy"]),
        ),
        "forgetting": float(
            before_old["overall_accuracy"] - after_old["overall_accuracy"]
        ),
        "joint_floor": float(
            min(after_old["class_floor_accuracy"], after_new["class_floor_accuracy"])
        ),
        "old_score_columns_bitwise_unchanged": prefix_unchanged,
        "old_prototype_prefix_bitwise_unchanged": old_prototype_prefix_unchanged,
        "old_radius_prefix_bitwise_unchanged": old_radius_prefix_unchanged,
        "r0_bitwise_unchanged": r0_unchanged,
        "radius_positive_finite": radius_positive_finite,
        "radius_fp16_shared_between_int8_fp32": radius_shared,
        "outer_prediction_sha256": outer_prediction_sha256,
        "radius_fp16_sha256": radius_fp16_sha256,
        "r0_fp16_sha256": r0_fp16_sha256,
        "radius_source_audit": radius_source_audit,
        "outer_held_new_intrusion_count": intrusion,
        "pairwise_support_diagnostics": pairwise,
        "new_new_confusion_count": int(np.sum(margins <= 0.0)),
        "new_new_margin_min": float(np.min(margins)),
        "new_new_margin_mean": float(np.mean(margins)),
        "matched_fp32_outer_argmax_change_count": argmax_changes,
        "target_old_int8_prototypes_used_for_prediction": (
            config.deploy_precision == "int8"
        ),
        "target_new_int8_prototypes_used_for_prediction": (
            config.deploy_precision == "int8"
        ),
        "deployment_precision": config.deploy_precision,
        "registration_before_prediction_sha256": hashlib.sha256(
            _canonical_bytes(before_predictions.astype(str).tolist())
        ).hexdigest(),
        "training_trace": [dict(row) for row in result.training_trace],
        "geometry_summary": geometry,
        "resource": resource,
    }


def _evaluate_d40_fold(
    rows: Mapping[str, np.ndarray],
    z_id160: np.ndarray,
    fft96: np.ndarray,
    rf32: np.ndarray,
    *,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    held_ranks: tuple[int, int],
    candidate_id: str,
    config: D40CandidateConfig,
    seed: int,
    device: torch.device | str = "cpu",
    scenario: str = "unit_test_scene",
    outer_fold: int | None = None,
) -> dict[str, Any]:
    """D40 outer 8-shot fit/2-shot held synchronous-HNBR evaluation."""

    labels = np.asarray(rows["labels"]).astype(str)
    ranks = np.asarray(rows["ranks"], dtype=np.int64)
    tokens = np.asarray(rows["tokens"]).astype(str)
    held = np.isin(ranks, np.asarray(held_ranks, dtype=np.int64))
    train = ~held
    old = np.isin(labels, np.asarray(old_classes))
    new = np.isin(labels, np.asarray(new_classes))
    if (
        int(np.sum(train & old)) != 8 * len(old_classes)
        or int(np.sum(train & new)) != 8 * len(new_classes)
        or int(np.sum(held & old)) != 2 * len(old_classes)
        or int(np.sum(held & new)) != 2 * len(new_classes)
    ):
        raise D25RunnerError("D40 leave-two-rank class symmetry drift")
    features = _d1_feature_from_blocks(z_id160, fft96, rf32)
    result = fit_d40_hnbr(
        features[train & old],
        labels[train & old],
        old_classes,
        features[train & new],
        labels[train & new],
        new_classes,
        seed=int(seed),
        device=device,
        config=config.core,
    )
    deployed_before = (
        result.before_state
        if config.deploy_precision == "int8"
        else result.matched_fp32_before_state
    )
    deployed_state = (
        result.state if config.deploy_precision == "int8" else result.matched_fp32_state
    )
    held_features = features[held]
    held_labels = labels[held]
    held_old = old[held]
    held_new = new[held]
    all_classes = old_classes + new_classes
    before_scores = score_d40_hnbr(deployed_before, held_features[held_old])
    after_scores = score_d40_hnbr(deployed_state, held_features)
    int8_scores = score_d40_hnbr(result.state, held_features)
    fp32_scores = score_d40_hnbr(result.matched_fp32_state, held_features)
    before_predictions = np.asarray(old_classes)[np.argmax(before_scores, axis=1)]
    after_predictions = np.asarray(all_classes)[np.argmax(after_scores, axis=1)]
    before_old = legacy._metric_block(
        held_labels[held_old], before_predictions.astype(str).tolist(), old_classes
    )
    after_old = legacy._metric_block(
        held_labels[held_old], after_predictions[held_old].astype(str).tolist(), old_classes
    )
    after_new = legacy._metric_block(
        held_labels[held_new], after_predictions[held_new].astype(str).tolist(), new_classes
    )
    pairwise = pairwise_support_diagnostics_d40(
        deployed_state,
        held_features[held_new],
        held_labels[held_new],
        tokens[held][held_new],
        scenario=scenario,
        outer_fold=(
            int(HELD_RANKS.index(held_ranks))
            if outer_fold is None
            else int(outer_fold)
        ),
        physical_ranks=ranks[held][held_new],
    )
    new_new_margins = np.asarray(
        [float(row["new_new_margin"]) for row in pairwise], dtype=np.float64
    )
    new_old_margins = np.asarray(
        [float(row["new_old_margin"]) for row in pairwise], dtype=np.float64
    )
    intrusion = _d37_old_to_new_intrusion_count(after_predictions, held_old, new_classes)
    argmax_changes = int(
        np.sum(np.argmax(int8_scores, axis=1) != np.argmax(fp32_scores, axis=1))
    )
    prefix_unchanged = old_prefix_bitwise_unchanged_d40(
        result.before_state, result.state
    )
    old_base_prefix_unchanged = old_prefix_bitwise_unchanged_d38(
        result.before_state.base_state, result.state.base_state
    )
    old_source_tokens = sorted(tokens[train & old].tolist())
    new_source_tokens = sorted(tokens[train & new].tolist())
    held_tokens = set(tokens[held].tolist())
    direction_source_audit = {
        "old_source_physical_token_sha256": hashlib.sha256(
            _canonical_bytes(old_source_tokens)
        ).hexdigest(),
        "new_source_physical_token_sha256": hashlib.sha256(
            _canonical_bytes(new_source_tokens)
        ).hexdigest(),
        "old_source_row_count": len(old_source_tokens),
        "new_source_row_count": len(new_source_tokens),
        "old_source_held_intersection_count": len(set(old_source_tokens) & held_tokens),
        "new_source_held_intersection_count": len(set(new_source_tokens) & held_tokens),
        "old_source_new_class_row_count": int(np.sum(train & old & new)),
        "new_source_old_class_row_count": int(np.sum(train & new & old)),
        "held_direction_fit_row_count": 0,
        "query_rows_used": 0,
    }
    outer_prediction_sha256 = hashlib.sha256(
        _canonical_bytes(after_predictions.astype(str).tolist())
    ).hexdigest()
    resource = dict(result.resource_audit)
    deployed_bytes = int(deployed_state.persistent_state_bytes)
    resource.update(
        {
            "deployment_precision": config.deploy_precision,
            "peak_trainable_parameters": int(resource["trainable_parameters"]),
            "total_optimizer_steps": int(resource["optimizer_steps"]),
            "persistent_state_bytes": deployed_bytes,
            "persistent_state_cap_pass": deployed_bytes <= 256 * 1024,
            "target_old_int8_prototypes_used_for_prediction": bool(
                config.deploy_precision == "int8"
            ),
            "target_new_int8_prototypes_used_for_prediction": bool(
                config.deploy_precision == "int8"
            ),
            "resident_fp32_target_prototype_count": (
                0 if config.deploy_precision == "int8" else len(all_classes)
            ),
            "formal_state_int8_only": bool(
                config.deploy_precision == "int8" and result.state.is_int8
            ),
            "clean_sample_access": False,
            "source_sample_access": False,
            "old_prefix_bitwise_unchanged": prefix_unchanged,
            "old_base_prefix_bitwise_unchanged": old_base_prefix_unchanged,
            "matched_fp32_outer_argmax_change_count": argmax_changes,
            "pairwise_support_diagnostic_row_count": len(pairwise),
            "new_new_confusion_count": int(np.sum(new_new_margins <= 0.0)),
            "new_new_margin_min": float(np.min(new_new_margins)),
            "new_old_margin_min": float(np.min(new_old_margins)),
            "complete_loss_trace": [dict(row) for row in result.training_trace],
            "latency_includes_argmax": True,
        }
    )
    geometry = {
        **dict(result.geometry_audit),
        "schema": "cvs.phase2.d40.outer_geometry.v1",
        "observed_feature_block_energy": _d30_observed_block_energy(features),
        "outer_held_protocol": "outer_leave_two_shot_ranks_not_seen_by_fit_or_hnbr",
        "pairwise_support_diagnostics": pairwise,
        "new_new_confusion_count": int(np.sum(new_new_margins <= 0.0)),
        "new_new_margin_min": float(np.min(new_new_margins)),
        "new_new_margin_mean": float(np.mean(new_new_margins)),
        "new_old_margin_min": float(np.min(new_old_margins)),
        "outer_held_new_intrusion_count": intrusion,
        "old_prefix_bitwise_unchanged": prefix_unchanged,
        "old_base_prefix_bitwise_unchanged": old_base_prefix_unchanged,
        "outer_prediction_sha256": outer_prediction_sha256,
        "direction_source_audit": direction_source_audit,
        "matched_fp32_outer_argmax_change_count": argmax_changes,
        "query_rows_used": 0,
    }
    return {
        "candidate_id": candidate_id,
        "held_ranks": list(held_ranks),
        "fit_k_shot": 8,
        "before_old": before_old,
        "after_old": after_old,
        "after_new": after_new,
        "H_old_new": legacy._harmonic(
            float(after_old["overall_accuracy"]),
            float(after_new["overall_accuracy"]),
        ),
        "forgetting": float(
            before_old["overall_accuracy"] - after_old["overall_accuracy"]
        ),
        "joint_floor": float(
            min(after_old["class_floor_accuracy"], after_new["class_floor_accuracy"])
        ),
        "old_score_columns_bitwise_unchanged": prefix_unchanged,
        "old_base_prefix_bitwise_unchanged": old_base_prefix_unchanged,
        "outer_prediction_sha256": outer_prediction_sha256,
        "direction_source_audit": direction_source_audit,
        "outer_held_new_intrusion_count": intrusion,
        "pairwise_support_diagnostics": pairwise,
        "new_new_confusion_count": int(np.sum(new_new_margins <= 0.0)),
        "new_new_margin_min": float(np.min(new_new_margins)),
        "new_new_margin_mean": float(np.mean(new_new_margins)),
        "new_old_margin_min": float(np.min(new_old_margins)),
        "matched_fp32_outer_argmax_change_count": argmax_changes,
        "target_old_int8_prototypes_used_for_prediction": bool(
            config.deploy_precision == "int8"
        ),
        "target_new_int8_prototypes_used_for_prediction": bool(
            config.deploy_precision == "int8"
        ),
        "deployment_precision": config.deploy_precision,
        "registration_before_prediction_sha256": hashlib.sha256(
            _canonical_bytes(before_predictions.astype(str).tolist())
        ).hexdigest(),
        "training_trace": [dict(row) for row in result.training_trace],
        "geometry_summary": geometry,
        "resource": resource,
    }


def _enrich_d40_strong_b3_pairwise(
    row: dict[str, Any],
    rows: Mapping[str, np.ndarray],
    diag_features: np.ndarray,
    diag_state: Mapping[str, Any],
    *,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    held_ranks: tuple[int, int],
    scenario: str,
    outer_fold: int,
) -> None:
    """Attach exact strong-B3 held pairwise evidence to its matched row."""

    labels = np.asarray(rows["labels"]).astype(str)
    ranks = np.asarray(rows["ranks"], dtype=np.int64)
    tokens = np.asarray(rows["tokens"]).astype(str)
    held = np.isin(ranks, np.asarray(held_ranks, dtype=np.int64))
    old = np.isin(labels, np.asarray(old_classes))
    new = np.isin(labels, np.asarray(new_classes))
    held_new = held & new
    held_old = held & old
    scores_new = legacy._diag_scores(
        diag_state, np.asarray(diag_features)[held_new], include_new=True
    )
    scores_old = legacy._diag_scores(
        diag_state, np.asarray(diag_features)[held_old], include_new=True
    )
    all_classes = old_classes + new_classes
    pairwise: list[dict[str, Any]] = []
    for score, truth, physical_id, rank in zip(
        scores_new,
        labels[held_new].tolist(),
        tokens[held_new].tolist(),
        ranks[held_new].tolist(),
        strict=True,
    ):
        truth_index = all_classes.index(str(truth))
        competing_new = np.array(score[len(old_classes) :], copy=True)
        competing_new[truth_index - len(old_classes)] = -np.inf
        competitor_index = len(old_classes) + int(np.argmax(competing_new))
        top_old_index = int(np.argmax(score[: len(old_classes)]))
        pairwise.append(
            {
                "scenario": str(scenario),
                "outer_fold": int(outer_fold),
                "physical_rank": int(rank),
                "physical_sample_id": str(physical_id),
                "true_new_handle": str(truth),
                "top_competing_new_handle": all_classes[competitor_index],
                "true_new_score": float(score[truth_index]),
                "top_competing_new_score": float(score[competitor_index]),
                "new_new_margin": float(score[truth_index] - score[competitor_index]),
                "top_old_handle": all_classes[top_old_index],
                "top_old_score": float(score[top_old_index]),
                "new_old_margin": float(score[truth_index] - score[top_old_index]),
                "query_rows_used": 0,
            }
        )
    margins = np.asarray([float(item["new_new_margin"]) for item in pairwise])
    old_predictions = np.asarray(all_classes)[np.argmax(scores_old, axis=1)]
    intrusion = _d37_old_to_new_intrusion_count(
        old_predictions, np.ones(len(old_predictions), dtype=bool), new_classes
    )
    row.update(
        {
            "outer_held_new_intrusion_count": intrusion,
            "pairwise_support_diagnostics": pairwise,
            "new_new_confusion_count": int(np.sum(margins <= 0.0)),
            "new_new_margin_min": float(np.min(margins)),
            "new_new_margin_mean": float(np.mean(margins)),
            "new_old_margin_min": float(
                min(float(item["new_old_margin"]) for item in pairwise)
            ),
        }
    )
    row.setdefault("geometry_summary", {}).update(
        {
            "schema": "cvs.phase2.d40.strong_b3_pairwise_geometry.v1",
            "pairwise_support_diagnostics": pairwise,
            "outer_held_new_intrusion_count": intrusion,
            "new_new_confusion_count": int(np.sum(margins <= 0.0)),
            "new_new_margin_min": float(np.min(margins)),
            "new_old_margin_min": float(
                min(float(item["new_old_margin"]) for item in pairwise)
            ),
            "query_rows_used": 0,
        }
    )
    row.setdefault("resource", {}).update(
        {
            "pairwise_support_diagnostic_row_count": len(pairwise),
            "query_rows_used_for_pairwise_diagnostic": 0,
        }
    )


def _evaluate_d36_fold(
    rows: Mapping[str, np.ndarray],
    z_id160: np.ndarray,
    fft96: np.ndarray,
    rf32: np.ndarray,
    *,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    held_ranks: tuple[int, int],
    candidate_id: str,
    config: D36CandidateConfig,
    ground_anchor: np.ndarray | None,
    ground_medoid_index: int | None,
    ground_anchor_sha256: str | None,
) -> dict[str, Any]:
    """D36 outer leave-two-rank evaluation with four-fold inner OOF calibration."""

    labels = np.asarray(rows["labels"]).astype(str)
    ranks = np.asarray(rows["ranks"], dtype=np.int64)
    held = np.isin(ranks, np.asarray(held_ranks, dtype=np.int64))
    train = ~held
    old = np.isin(labels, np.asarray(old_classes))
    new = np.isin(labels, np.asarray(new_classes))
    if (
        int(np.sum(train & old)) != 8 * len(old_classes)
        or int(np.sum(train & new)) != 8 * len(new_classes)
        or int(np.sum(held & old)) != 2 * len(old_classes)
        or int(np.sum(held & new)) != 2 * len(new_classes)
    ):
        raise D25RunnerError("D36 leave-two-rank class symmetry drift")
    features = _d1_feature_from_blocks(z_id160, fft96, rf32)
    fit = _fit_d36_route(
        features[train],
        labels[train],
        ranks[train],
        old[train],
        new[train],
        old_classes,
        new_classes,
        config,
        ground_anchor,
    )
    all_classes = old_classes + new_classes
    before_scores, after_scores = _score_d36(fit, features[held])
    held_labels = labels[held]
    held_old = old[held]
    held_new = new[held]
    before_predictions = np.asarray(old_classes)[
        np.argmax(before_scores[held_old], axis=1)
    ]
    after_predictions = np.asarray(all_classes)[np.argmax(after_scores, axis=1)]
    before_old = legacy._metric_block(
        held_labels[held_old], before_predictions.astype(str).tolist(), old_classes
    )
    after_old = legacy._metric_block(
        held_labels[held_old], after_predictions[held_old].astype(str).tolist(), old_classes
    )
    after_new = legacy._metric_block(
        held_labels[held_new], after_predictions[held_new].astype(str).tolist(), new_classes
    )
    b3_scores = score_b3_fisher_closed_form(fit["fisher_fit"].state, features[held][held_old])
    b3_predictions = np.asarray(old_classes)[np.argmax(b3_scores, axis=1)]
    b3_old = legacy._metric_block(
        held_labels[held_old], b3_predictions.astype(str).tolist(), old_classes
    )
    before_correct = before_predictions.astype(str) == held_labels[held_old]
    outer_intrusion = int(
        np.sum(
            before_correct
            & np.isin(after_predictions[held_old].astype(str), np.asarray(new_classes))
        )
    )
    new_indices = np.flatnonzero(held_new)
    new_loso: list[dict[str, Any]] = []
    for local_index in new_indices.tolist():
        target = len(old_classes) + new_classes.index(str(held_labels[local_index]))
        scores = after_scores[local_index]
        competitors = np.delete(scores, target)
        margin = float(scores[target] - np.max(competitors))
        new_loso.append(
            {
                "held_rank": int(ranks[held][local_index]),
                "new_class": str(held_labels[local_index]),
                "correct": bool(np.argmax(scores) == target),
                "margin": margin,
            }
        )
    new_reachability = {
        name: bool(values)
        and all(bool(row["correct"]) and float(row["margin"]) > 0.0 for row in values)
        for name in new_classes
        for values in [[row for row in new_loso if row["new_class"] == name]]
    }
    resource = _d36_resource(fit, len(all_classes))
    old_score_columns_bitwise_unchanged = bool(
        np.array_equal(before_scores, after_scores[:, : len(old_classes)])
    )
    geometry = {
        **dict(fit["core_result"].geometry_audit),
        "schema": "cvs.phase2.d36_compiled_joint_int8_geometry.v1",
        "feature_geometry": "b3_auxiliary_dominant_z160_fft96_rf32_v1",
        "observed_feature_block_energy": _d30_observed_block_energy(features),
        "inner_crossfit_rank_pairs": [list(pair) for pair in fit["inner_pairs"]],
        "inner_crossfit_no_self_participation": True,
        "oof_calibration_row_count": len(fit["oof_roles"]),
        "outer_held_protocol": "outer_leave_two_shot_ranks_not_seen_by_fit",
        "outer_held_new_intrusion_count": outer_intrusion,
        "outer_held_zero_new_intrusion_pass": outer_intrusion == 0,
        "new_physical_leave_one_out": new_loso,
        "new_class_reachability": new_reachability,
        "new_physical_loso_all_reachable": all(new_reachability.values()),
        "ground_anchor_medoid_index": (
            ground_medoid_index if config.compiled.arm in ("B", "C") else None
        ),
        "ground_anchor_sha256": (
            ground_anchor_sha256 if config.compiled.arm in ("B", "C") else None
        ),
        "ground_anchor_read_only": bool(
            config.compiled.arm == "A"
            or (ground_anchor is not None and not ground_anchor.flags.writeable)
        ),
    }
    return {
        "candidate_id": candidate_id,
        "held_ranks": list(held_ranks),
        "fit_k_shot": 8,
        "before_old": before_old,
        "after_old": after_old,
        "after_new": after_new,
        "b3_reference_old": b3_old,
        "H_old_new": legacy._harmonic(
            float(after_old["overall_accuracy"]),
            float(after_new["overall_accuracy"]),
        ),
        "forgetting": float(
            before_old["overall_accuracy"] - after_old["overall_accuracy"]
        ),
        "joint_floor": float(
            min(after_old["class_floor_accuracy"], after_new["class_floor_accuracy"])
        ),
        # Compatibility field consumed by the shared candidate aggregator.
        # D36 recompiles the target-old head, so unlike frozen-prefix routes
        # this is measured from the actual held scores instead of assumed.
        "old_score_columns_bitwise_unchanged": (
            old_score_columns_bitwise_unchanged
        ),
        "old_score_columns_bitwise_unchanged_semantics": (
            "measured_before_vs_after_compiled_target_old_scores"
        ),
        "outer_held_new_intrusion_count": outer_intrusion,
        "outer_held_zero_new_intrusion_pass": outer_intrusion == 0,
        "new_class_reachability": new_reachability,
        "new_physical_loso_all_reachable": all(new_reachability.values()),
        "unreachable_new_class_count": int(sum(not value for value in new_reachability.values())),
        "target_old_int8_prototypes_used_for_prediction": True,
        "target_new_int8_prototypes_used_for_prediction": True,
        "training_trace": list(resource["complete_loss_trace"]),
        "geometry_summary": geometry,
        "resource": resource,
    }


def _evaluate_d32_fold(
    component: Any,
    rows: Mapping[str, np.ndarray],
    z_id160: np.ndarray,
    direct_logits: np.ndarray,
    fft96: np.ndarray,
    rf32: np.ndarray,
    *,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    held_ranks: tuple[int, int],
    candidate_id: str,
    config: D32CandidateConfig,
) -> dict[str, Any]:
    """Leave-two-out D32 evaluation; held rows never enter fitting or gates."""

    labels = np.asarray(rows["labels"]).astype(str)
    ranks = np.asarray(rows["ranks"], dtype=np.int64)
    held = np.isin(ranks, np.asarray(held_ranks, dtype=np.int64))
    train = ~held
    old = np.isin(labels, np.asarray(old_classes))
    new = np.isin(labels, np.asarray(new_classes))
    if (
        int(np.sum(train & old)) != 8 * len(old_classes)
        or int(np.sum(train & new)) != 8 * len(new_classes)
        or int(np.sum(held & old)) != 2 * len(old_classes)
        or int(np.sum(held & new)) != 2 * len(new_classes)
    ):
        raise D25RunnerError("D32 leave-two-out class symmetry drift")
    features = _d1_feature_from_blocks(z_id160, fft96, rf32)
    fit = _fit_d32_route(
        component,
        features[train],
        z_id160[train],
        direct_logits[train],
        labels[train],
        old[train],
        new[train],
        old_classes,
        new_classes,
        config,
    )
    before = fit["before"]
    after = fit["after"]
    all_classes = old_classes + new_classes
    before_old_predictions = predict_all_d26(
        before, features[held & old]
    ).astype(str).tolist()
    held_raw = score_all_d32(after, features[held])
    held_adjusted = (
        _d30_rerank_matrix(
            fit["dali_state"], held_raw, z_id160[held], direct_logits[held]
        )
        if fit["dali_enabled"]
        else held_raw.copy()
    )
    held_labels = labels[held]
    held_predictions = np.asarray(all_classes)[np.argmax(held_adjusted, axis=1)]
    held_old_mask = np.isin(held_labels, np.asarray(old_classes))
    held_new_mask = ~held_old_mask
    before_old = legacy._metric_block(
        labels[held & old], before_old_predictions, old_classes
    )
    after_old = legacy._metric_block(
        held_labels[held_old_mask], held_predictions[held_old_mask].tolist(), old_classes
    )
    after_new = legacy._metric_block(
        held_labels[held_new_mask], held_predictions[held_new_mask].tolist(), new_classes
    )
    fit_before_predictions = predict_all_d26(
        before, features[train & old]
    ).astype(str).tolist()
    fit_predictions = np.asarray(all_classes)[np.argmax(fit["adjusted_scores"], axis=1)]
    fit_old_mask = old[train]
    fit_before = legacy._metric_block(
        labels[train & old], fit_before_predictions, old_classes
    )
    fit_after = legacy._metric_block(
        labels[train][fit_old_mask], fit_predictions[fit_old_mask].tolist(), old_classes
    )
    classwise_pass = all(
        float(fit_after["per_class_accuracy"][name]) + 1.0e-12
        >= float(fit_before["per_class_accuracy"][name])
        for name in old_classes
    )
    floor_pass = (
        float(fit_after["class_floor_accuracy"]) + 1.0e-12
        >= float(fit_before["class_floor_accuracy"])
    )
    training_trace = list(fit["before_fit"].loss_trace) + list(
        fit["stage2c_fit"].loss_trace
    )
    base_resource = dict(after.resource_audit())
    dali_accounting = _d31_dali_state_accounting(fit["dali_state"])
    dali_resource = dali_accounting["dali_resource"]
    combined_resident_state = int(base_resource["persistent_state_bytes"]) + int(
        dali_accounting["actual_current_dali_state_bytes"]
    )
    projected_active_state = int(base_resource["persistent_state_bytes"]) + int(
        dali_accounting["projected_slim_dali_runtime_bytes"]
    )
    registered_count = len(all_classes)
    base_head_macs = int(base_resource["estimated_macs_per_query"])
    dali_macs = (
        int(dali_resource["fixed_medoid_ground_macs_per_query"])
        if fit["dali_enabled"]
        else 0
    )
    total_macs = base_head_macs + dali_macs
    identity_macs = registered_count * 10 * 160
    argmax_ops = max(0, registered_count - 1)
    dali_scalar_ops = 12 * len(old_classes) if fit["dali_enabled"] else 0
    resource = {
        **base_resource,
        "schema": "cvs.phase2.d32_combined_resource.v1",
        "d32_suffix_resource": base_resource,
        **dali_accounting,
        "dali_enabled_by_old_support_gate": bool(fit["dali_enabled"]),
        "dali_old_support_gate": fit["dali_gate"],
        "actual_int8_component_used_for_prediction": bool(fit["dali_enabled"]),
        "authorized_full_bundle_state_bytes": int(
            dali_accounting["authorized_full_bundle_state_bytes"]
        ),
        "full_bundle_resident_combined_state_bytes": combined_resident_state,
        "projected_slim_active_predictor_state_bytes": projected_active_state,
        "slim_runtime_projection_only": True,
        "deployment_resource_primary_state_view": (
            "projected_slim_fixed_medoid_predictor_with_full_bundle_residency_disclosed"
        ),
        "deployable_predictor_state_bytes_projected_slim_medoid": projected_active_state,
        "persistent_state_bytes": combined_resident_state,
        "persistent_state_cap_pass": combined_resident_state <= 256 * 1024,
        "stage2b_adaptation_macs": int(
            base_resource["estimated_stage2b_adaptation_macs"]
        ),
        "stage2c_adaptation_macs": int(
            base_resource["estimated_stage2c_adaptation_macs"]
        ),
        "total_adaptation_macs": int(base_resource["estimated_adaptation_macs"]),
        "base_head_macs_per_query": base_head_macs,
        "d32_extra_scalar_bias_adds_per_query": int(
            base_resource["estimated_scalar_bias_adds_per_query"]
        ),
        "dali_medoid_macs_per_query": dali_macs,
        "argmax_scalar_comparisons_per_query": argmax_ops,
        "total_post_backbone_macs_per_query": total_macs,
        "estimated_macs_per_query": total_macs,
        "estimated_row_local_scalar_ops_per_query": int(
            base_resource["estimated_scalar_bias_adds_per_query"]
            + dali_scalar_ops
            + argmax_ops
        ),
        "identity_single_qknn_macs_same_registered_count": identity_macs,
        "estimated_score_mac_ratio_vs_identity_single_qknn": float(
            total_macs / identity_macs
        ),
        "total_optimizer_steps": int(after.stage2b_optimizer_steps)
        + int(after.stage2c_optimizer_steps),
        "total_adaptation_epochs": int(after.stage2b_optimizer_steps)
        + int(after.stage2c_optimizer_steps),
        "complete_loss_trace": training_trace,
        "feature_geometry": "b3_auxiliary_dominant_z160_fft96_rf32_v1",
        "old_support_non_degradation_pass": bool(classwise_pass and floor_pass),
        "query_rows_used_for_fit": 0,
        "query_role_oracle_access": False,
        "query_true_batch_class_count_access": False,
        "query_class_quota_access": False,
        "query_batch_global_assignment": False,
        "dense_query_graph_bytes": 0,
        "clean_sample_access": False,
        "source_sample_access": False,
    }
    confusion_raw = _d31_confusion_audit(
        held_raw, held_labels, old_classes, new_classes
    )
    confusion_final = _d31_confusion_audit(
        held_adjusted, held_labels, old_classes, new_classes
    )
    geometry = {
        "schema": "cvs.phase2.d32_inloop_safe_cap_geometry.v1",
        "feature_geometry": "b3_auxiliary_dominant_z160_fft96_rf32_v1",
        "observed_feature_block_energy": _d30_observed_block_energy(features),
        "old_prefix_sha256": after.old_prefix_sha256,
        "dali_enabled": bool(fit["dali_enabled"]),
        "dali_old_support_gate": fit["dali_gate"],
        "raw_confusion": confusion_raw,
        "final_confusion": confusion_final,
        "support_gate": json.loads(after.support_gate_json),
    }
    final_old_scores_unchanged = np.array_equal(
        held_raw[:, : len(old_classes)],
        held_adjusted[:, : len(old_classes)],
    )
    return {
        "candidate_id": candidate_id,
        "held_ranks": list(held_ranks),
        "fit_k_shot": 8,
        "before_old": before_old,
        "after_old": after_old,
        "after_new": after_new,
        "H_old_new": legacy._harmonic(
            float(after_old["overall_accuracy"]),
            float(after_new["overall_accuracy"]),
        ),
        "forgetting": float(
            before_old["overall_accuracy"] - after_old["overall_accuracy"]
        ),
        "joint_floor": float(
            min(after_old["class_floor_accuracy"], after_new["class_floor_accuracy"])
        ),
        # Compatibility field consumed by the shared candidate aggregator.
        # It refers to the immutable raw old-score prefix before optional DALI;
        # the distinct final-DALI column equality is reported below.
        "old_score_columns_bitwise_unchanged": True,
        "base_old_parameter_prefix_bitwise_unchanged": True,
        "final_old_score_columns_bitwise_unchanged": bool(final_old_scores_unchanged),
        "dali_max_old_preserved": True,
        "old_prefix_sha256_before": before.old_lock_sha256,
        "old_prefix_sha256_after": after.base_old_lock_sha256,
        "fit_old_before_registration": fit_before,
        "fit_old_after_registration": fit_after,
        "old_support_classwise_non_degradation": classwise_pass,
        "old_support_floor_non_degradation": floor_pass,
        "old_support_non_degradation_pass": bool(classwise_pass and floor_pass),
        "dali_enabled": bool(fit["dali_enabled"]),
        "dali_old_support_gate": fit["dali_gate"],
        "raw_confusion": confusion_raw,
        "final_confusion": confusion_final,
        "training_trace": training_trace,
        "geometry_summary": geometry,
        "resource": resource,
    }


def _fold_guard(row: Mapping[str, Any], baseline: Mapping[str, Any]) -> bool:
    tolerance = 1.0e-12
    old_classwise = all(
        float(row["after_old"]["per_class_accuracy"][label]) + tolerance
        >= float(baseline["after_old"]["per_class_accuracy"][label])
        for label in row["after_old"]["per_class_accuracy"]
    )
    new_classwise = all(
        float(row["after_new"]["per_class_accuracy"][label]) + tolerance
        >= float(baseline["after_new"]["per_class_accuracy"][label])
        for label in row["after_new"]["per_class_accuracy"]
    )
    return bool(
        float(row["before_old"]["class_floor_accuracy"]) + tolerance
        >= float(baseline["before_old"]["class_floor_accuracy"])
        and float(row["after_old"]["class_floor_accuracy"]) + tolerance
        >= float(baseline["after_old"]["class_floor_accuracy"])
        and float(row["after_new"]["class_floor_accuracy"]) + tolerance
        >= float(baseline["after_new"]["class_floor_accuracy"])
        and float(row["H_old_new"]) + tolerance
        >= float(baseline["H_old_new"])
        and float(row["forgetting"])
        <= float(baseline["forgetting"]) + tolerance
        and old_classwise
        and new_classwise
    )


def _select_candidate(
    folds_by_candidate: Mapping[str, Sequence[Mapping[str, Any]]]
) -> tuple[str, list[dict[str, Any]]]:
    baseline = list(folds_by_candidate[IDENTITY_CANDIDATE])
    baseline_aggregate = legacy._aggregate_candidate(baseline)
    diagnostic = list(folds_by_candidate[DIAG_CANDIDATE])
    decisions: list[dict[str, Any]] = []
    eligible: list[tuple[str, float, float, float]] = []
    tolerance = 1.0e-12
    for candidate_id, raw_rows in folds_by_candidate.items():
        rows = list(raw_rows)
        aggregate = legacy._aggregate_candidate(rows)
        guards = [
            _fold_guard(row, zero) for row, zero in zip(rows, baseline)
        ]
        diagnostic_guards = [
            _fold_guard(row, diag) for row, diag in zip(rows, diagnostic)
        ]
        strict_old_floor = bool(
            float(aggregate["worst_after_old_floor"])
            > float(baseline_aggregate["worst_after_old_floor"]) + tolerance
        )
        strict_joint_floor = bool(
            float(aggregate["worst_joint_floor"])
            > float(baseline_aggregate["worst_joint_floor"]) + tolerance
        )
        is_d25 = candidate_id in D25_CANDIDATES
        eligible_positive = bool(
            is_d25 and all(guards) and (strict_old_floor or strict_joint_floor)
        )
        decision = {
            **aggregate,
            "candidate_id": candidate_id,
            "family": "d25" if is_d25 else "control",
            "atomic_noninferiority_vs_Z0": bool(all(guards)),
            "noninferior_fold_count": int(sum(guards)),
            "noninferior_vs_B3_fold_count": int(sum(diagnostic_guards)),
            "strict_worst_old_floor_improvement_vs_Z0": strict_old_floor,
            "strict_worst_joint_floor_improvement_vs_Z0": strict_joint_floor,
            "eligible_positive_route": eligible_positive,
            "fallback": candidate_id == IDENTITY_CANDIDATE,
            "diagnostic_only": candidate_id == DIAG_CANDIDATE,
        }
        decisions.append(decision)
        if eligible_positive:
            eligible.append(
                (
                    candidate_id,
                    float(aggregate["worst_joint_floor"]),
                    float(aggregate["worst_after_old_floor"]),
                    float(aggregate["mean_H_old_new"]),
                )
            )
    selected = (
        max(eligible, key=lambda value: (value[1], value[2], value[3], value[0]))[0]
        if eligible
        else IDENTITY_CANDIDATE
    )
    return selected, decisions


def _pooled_scenario_classwise(
    rows: Sequence[Mapping[str, Any]], metric_key: str
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for scenario in legacy.FORMAL_LEO_WEAK_SCENARIOS:
        selected = [row for row in rows if row.get("scenario") == scenario]
        if len(selected) != len(HELD_RANKS):
            raise D25RunnerError("C3 pooled scenario fold cardinality drift")
        labels = tuple(selected[0][metric_key]["per_class_accuracy"])
        result[scenario] = {
            label: float(
                np.mean(
                    [
                        float(row[metric_key]["per_class_accuracy"][label])
                        for row in selected
                    ]
                )
            )
            for label in labels
        }
    return result


def _select_c3_candidate(
    folds_by_candidate: Mapping[str, Sequence[Mapping[str, Any]]]
) -> tuple[str, list[dict[str, Any]]]:
    baseline_rows = list(folds_by_candidate[D25_C0])
    baseline_old = _pooled_scenario_classwise(baseline_rows, "after_old")
    baseline_new = _pooled_scenario_classwise(baseline_rows, "after_new")
    baseline_h = float(np.mean([float(row["H_old_new"]) for row in baseline_rows]))
    baseline_forgetting = float(
        np.mean([float(row["forgetting"]) for row in baseline_rows])
    )
    decisions: list[dict[str, Any]] = []
    eligible: list[tuple[str, float, float, float, float, int]] = []
    for candidate_id, raw_rows in folds_by_candidate.items():
        rows = list(raw_rows)
        aggregate = legacy._aggregate_candidate(rows)
        decision: dict[str, Any] = {
            **aggregate,
            "candidate_id": candidate_id,
            "family": (
                "d25_c3"
                if candidate_id in C3_CANDIDATES
                else ("d25" if candidate_id == D25_C0 else "control")
            ),
            "fallback": candidate_id == D25_C0,
            "diagnostic_only": candidate_id == DIAG_CANDIDATE,
            "eligible_positive_route": False,
        }
        if candidate_id not in C3_CANDIDATES:
            decisions.append(decision)
            continue
        candidate_old = _pooled_scenario_classwise(rows, "after_old")
        candidate_new = _pooled_scenario_classwise(rows, "after_new")
        safety = True
        old_floor_gain: list[float] = []
        new_floor_gain: list[float] = []
        for scenario in legacy.FORMAL_LEO_WEAK_SCENARIOS:
            for label, baseline_value in baseline_old[scenario].items():
                safety = safety and (
                    candidate_old[scenario][label] + 1.0e-12
                    >= baseline_value - 0.10
                )
            for label, baseline_value in baseline_new[scenario].items():
                safety = safety and (
                    candidate_new[scenario][label] + 1.0e-12
                    >= baseline_value - 0.10
                )
            old_floor_gain.append(
                min(candidate_old[scenario].values())
                - min(baseline_old[scenario].values())
            )
            new_floor_gain.append(
                min(candidate_new[scenario].values())
                - min(baseline_new[scenario].values())
            )
        old_support_pass = all(
            bool(row["old_support_non_degradation_pass"]) for row in rows
        )
        mean_h = float(np.mean([float(row["H_old_new"]) for row in rows]))
        mean_forgetting = float(
            np.mean([float(row["forgetting"]) for row in rows])
        )
        floor_pass = bool(
            all(value >= 0.10 - 1.0e-12 for value in old_floor_gain)
            and all(value >= 0.10 - 1.0e-12 for value in new_floor_gain)
        )
        balance_pass = bool(
            mean_h + 1.0e-12 >= baseline_h
            and mean_forgetting <= baseline_forgetting + 1.0e-12
        )
        eligible_positive = bool(
            safety and floor_pass and balance_pass and old_support_pass
        )
        decision.update(
            {
                "pooled_per_class_safety_vs_C0_pass": safety,
                "pooled_old_floor_gain_by_scenario": dict(
                    zip(legacy.FORMAL_LEO_WEAK_SCENARIOS, old_floor_gain)
                ),
                "pooled_new_floor_gain_by_scenario": dict(
                    zip(legacy.FORMAL_LEO_WEAK_SCENARIOS, new_floor_gain)
                ),
                "pooled_floor_gate_pass": floor_pass,
                "old_support_non_degradation_all_folds": old_support_pass,
                "mean_H_noninferior_vs_C0": mean_h + 1.0e-12 >= baseline_h,
                "mean_forgetting_noninferior_vs_C0": mean_forgetting
                <= baseline_forgetting + 1.0e-12,
                "eligible_positive_route": eligible_positive,
            }
        )
        decisions.append(decision)
        if eligible_positive:
            steps = int(rows[0]["resource"]["total_optimizer_steps"])
            eligible.append(
                (
                    candidate_id,
                    min(min(old_floor_gain), min(new_floor_gain)),
                    float(aggregate["worst_joint_floor"]),
                    mean_h,
                    -mean_forgetting,
                    -steps,
                )
            )
    selected = (
        max(eligible, key=lambda value: value[1:])[0] if eligible else D25_C0
    )
    return selected, decisions


def _select_d26_candidate(
    folds_by_candidate: Mapping[str, Sequence[Mapping[str, Any]]],
    eligible_candidate_ids: Sequence[str] = D26_CANDIDATES,
) -> tuple[str, list[dict[str, Any]]]:
    eligible_candidate_ids = tuple(str(value) for value in eligible_candidate_ids)
    baseline_rows = list(folds_by_candidate[D25_C0])
    diagnostic_rows = list(folds_by_candidate[DIAG_CANDIDATE])
    baseline_old = _pooled_scenario_classwise(baseline_rows, "after_old")
    baseline_new = _pooled_scenario_classwise(baseline_rows, "after_new")
    diagnostic_old = _pooled_scenario_classwise(diagnostic_rows, "after_old")
    diagnostic_new = _pooled_scenario_classwise(diagnostic_rows, "after_new")
    baseline_h = float(np.mean([float(row["H_old_new"]) for row in baseline_rows]))
    baseline_forgetting = float(
        np.mean([float(row["forgetting"]) for row in baseline_rows])
    )
    diagnostic_h = float(
        np.mean([float(row["H_old_new"]) for row in diagnostic_rows])
    )
    diagnostic_forgetting = float(
        np.mean([float(row["forgetting"]) for row in diagnostic_rows])
    )
    decisions: list[dict[str, Any]] = []
    eligible: list[tuple[str, float, float, float, float, int]] = []
    for candidate_id, raw_rows in folds_by_candidate.items():
        rows = list(raw_rows)
        aggregate = legacy._aggregate_candidate(rows)
        decision: dict[str, Any] = {
            **aggregate,
            "candidate_id": candidate_id,
            "family": (
                "d33_spherical_registration"
                if candidate_id in D33_CANDIDATES
                else "d32_inloop_safe_cap_suffix_with_dali"
                if candidate_id in D32_CANDIDATES
                else "d31_all_registered_suffix_with_dali"
                if candidate_id in D31_CANDIDATES
                else "d30_b3_dali_dual_envelope"
                if candidate_id in D30_CANDIDATES
                else "d29_per_class_safe_release"
                if candidate_id in D29_CANDIDATES
                else "d28_support_evidence_gate"
                if candidate_id in D28_CANDIDATES
                else "d27_per_new_class_bias"
                if candidate_id in D27_CANDIDATES
                else "d26_compact_diag"
                if candidate_id in eligible_candidate_ids
                else ("d25" if candidate_id == D25_C0 else "control")
            ),
            "fallback": candidate_id == D25_C0,
            "diagnostic_only": candidate_id == DIAG_CANDIDATE,
            "eligible_positive_route": False,
        }
        if candidate_id not in eligible_candidate_ids:
            decisions.append(decision)
            continue
        candidate_old = _pooled_scenario_classwise(rows, "after_old")
        candidate_new = _pooled_scenario_classwise(rows, "after_new")
        safety = True
        old_floor_gain: list[float] = []
        new_floor_gain: list[float] = []
        old_floor_delta_vs_b3: list[float] = []
        new_floor_delta_vs_b3: list[float] = []
        for scenario in legacy.FORMAL_LEO_WEAK_SCENARIOS:
            for label, baseline_value in baseline_old[scenario].items():
                safety = safety and (
                    candidate_old[scenario][label] + 1.0e-12
                    >= baseline_value - 0.10
                )
            for label, baseline_value in baseline_new[scenario].items():
                safety = safety and (
                    candidate_new[scenario][label] + 1.0e-12
                    >= baseline_value - 0.10
                )
            old_floor_gain.append(
                min(candidate_old[scenario].values())
                - min(baseline_old[scenario].values())
            )
            new_floor_gain.append(
                min(candidate_new[scenario].values())
                - min(baseline_new[scenario].values())
            )
            old_floor_delta_vs_b3.append(
                min(candidate_old[scenario].values())
                - min(diagnostic_old[scenario].values())
            )
            new_floor_delta_vs_b3.append(
                min(candidate_new[scenario].values())
                - min(diagnostic_new[scenario].values())
            )
        old_support_pass = all(
            bool(row["old_support_non_degradation_pass"]) for row in rows
        )
        mean_h = float(np.mean([float(row["H_old_new"]) for row in rows]))
        mean_forgetting = float(
            np.mean([float(row["forgetting"]) for row in rows])
        )
        floor_pass = bool(
            all(value >= 0.10 - 1.0e-12 for value in old_floor_gain)
            and all(value >= 0.10 - 1.0e-12 for value in new_floor_gain)
        )
        balance_pass = bool(
            mean_h + 1.0e-12 >= baseline_h
            and mean_forgetting <= baseline_forgetting + 1.0e-12
        )
        eligible_positive = bool(
            safety and floor_pass and balance_pass and old_support_pass
        )
        decision.update(
            {
                "pooled_per_class_safety_vs_C0_pass": safety,
                "pooled_old_floor_gain_by_scenario": dict(
                    zip(legacy.FORMAL_LEO_WEAK_SCENARIOS, old_floor_gain)
                ),
                "pooled_new_floor_gain_by_scenario": dict(
                    zip(legacy.FORMAL_LEO_WEAK_SCENARIOS, new_floor_gain)
                ),
                "pooled_floor_gate_pass": floor_pass,
                "old_support_non_degradation_all_folds": old_support_pass,
                "mean_H_noninferior_vs_C0": mean_h + 1.0e-12 >= baseline_h,
                "mean_forgetting_noninferior_vs_C0": mean_forgetting
                <= baseline_forgetting + 1.0e-12,
                "B3_performance_reference_only": True,
                "mean_H_delta_vs_B3": mean_h - diagnostic_h,
                "mean_forgetting_delta_vs_B3": (
                    mean_forgetting - diagnostic_forgetting
                ),
                "pooled_old_floor_delta_vs_B3_by_scenario": dict(
                    zip(legacy.FORMAL_LEO_WEAK_SCENARIOS, old_floor_delta_vs_b3)
                ),
                "pooled_new_floor_delta_vs_B3_by_scenario": dict(
                    zip(legacy.FORMAL_LEO_WEAK_SCENARIOS, new_floor_delta_vs_b3)
                ),
                "eligible_positive_route": eligible_positive,
            }
        )
        decisions.append(decision)
        if eligible_positive:
            steps = int(rows[0]["resource"]["total_optimizer_steps"])
            eligible.append(
                (
                    candidate_id,
                    min(min(old_floor_gain), min(new_floor_gain)),
                    float(aggregate["worst_joint_floor"]),
                    mean_h,
                    -mean_forgetting,
                    -steps,
                )
            )
    selected = (
        max(eligible, key=lambda value: value[1:])[0] if eligible else D25_C0
    )
    return selected, decisions


def _select_d34_candidate(
    folds_by_candidate: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[str, list[dict[str, Any]]]:
    """Apply D34's non-negotiable old-safety gates before joint ranking."""

    decisions: list[dict[str, Any]] = []
    eligible: list[tuple[str, float, float, float, int]] = []
    comparator_ids = (DIAG_CANDIDATE, D33_B3_FAST)
    comparator_rows = {
        candidate_id: list(folds_by_candidate[candidate_id])
        for candidate_id in comparator_ids
    }
    comparator_thresholds = {
        "mean_after_old": max(
            float(
                np.mean(
                    [
                        float(row["after_old"]["overall_accuracy"])
                        for row in rows
                    ]
                )
            )
            for rows in comparator_rows.values()
        ),
        "mean_after_new": max(
            float(
                np.mean(
                    [
                        float(row["after_new"]["overall_accuracy"])
                        for row in rows
                    ]
                )
            )
            for rows in comparator_rows.values()
        ),
        "mean_h": max(
            float(np.mean([float(row["H_old_new"]) for row in rows]))
            for rows in comparator_rows.values()
        ),
        "mean_forgetting": min(
            float(np.mean([float(row["forgetting"]) for row in rows]))
            for rows in comparator_rows.values()
        ),
        "worst_joint_floor": max(
            min(float(row["joint_floor"]) for row in rows)
            for rows in comparator_rows.values()
        ),
    }
    for candidate_id, raw_rows in folds_by_candidate.items():
        rows = list(raw_rows)
        aggregate = legacy._aggregate_candidate(rows)
        decision: dict[str, Any] = {
            **aggregate,
            "candidate_id": candidate_id,
            "family": (
                "d34_collision_local_registration"
                if candidate_id in D34_CANDIDATES
                else "d33_fast_negative_control"
                if candidate_id == D33_B3_FAST
                else "d25"
                if candidate_id == D25_C0
                else "control"
            ),
            "fallback": candidate_id == D25_C0,
            "diagnostic_only": candidate_id in (DIAG_CANDIDATE, D33_B3_FAST),
            "eligible_positive_route": False,
        }
        if candidate_id not in D34_CANDIDATES:
            decisions.append(decision)
            continue
        old_support_pass = all(
            bool(row["old_support_non_degradation_pass"]) for row in rows
        )
        old_loso_pass = all(bool(row["old_loso_zero_intrusion_pass"]) for row in rows)
        old_prefix_pass = all(
            bool(row["old_score_prefix_bitwise_unchanged"]) for row in rows
        )
        worst_joint_floor = min(float(row["joint_floor"]) for row in rows)
        worst_new_floor = min(
            float(row["after_new"]["class_floor_accuracy"]) for row in rows
        )
        mean_h = float(np.mean([float(row["H_old_new"]) for row in rows]))
        mean_after_old = float(
            np.mean([float(row["after_old"]["overall_accuracy"]) for row in rows])
        )
        mean_after_new = float(
            np.mean([float(row["after_new"]["overall_accuracy"]) for row in rows])
        )
        mean_forgetting = float(
            np.mean([float(row["forgetting"]) for row in rows])
        )
        total_edges = int(sum(int(row["collision_edge_count"]) for row in rows))
        unreachable_edges = int(
            sum(int(row["unreachable_edge_count"]) for row in rows)
        )
        all_new_classes_reachable = unreachable_edges == 0
        comparator_gate = bool(
            mean_after_old + 1.0e-12 >= comparator_thresholds["mean_after_old"]
            and mean_after_new + 1.0e-12
            >= comparator_thresholds["mean_after_new"]
            and mean_h > comparator_thresholds["mean_h"] + 1.0e-12
            and mean_forgetting
            <= comparator_thresholds["mean_forgetting"] + 1.0e-12
            and worst_joint_floor + 1.0e-12
            >= comparator_thresholds["worst_joint_floor"]
        )
        hard_gate = bool(
            old_support_pass
            and old_loso_pass
            and old_prefix_pass
            and all_new_classes_reachable
            and comparator_gate
        )
        decision.update(
            {
                "old_support_non_degradation_all_folds": old_support_pass,
                "old_loso_zero_intrusion_all_folds": old_loso_pass,
                "old_score_prefix_bitwise_unchanged_all_folds": old_prefix_pass,
                "d34_old_safety_hard_gate_pass": hard_gate,
                "all_new_classes_reachable_all_folds": all_new_classes_reachable,
                "d34_joint_comparator_gate_pass": comparator_gate,
                "joint_comparator_thresholds": comparator_thresholds,
                "rank_mean_after_old": mean_after_old,
                "rank_mean_after_new": mean_after_new,
                "rank_mean_forgetting": mean_forgetting,
                "rank_worst_joint_floor": worst_joint_floor,
                "rank_worst_new_floor": worst_new_floor,
                "rank_mean_H_old_new": mean_h,
                "rank_collision_edge_count": total_edges,
                "unreachable_edge_count": unreachable_edges,
                "eligible_positive_route": hard_gate,
            }
        )
        decisions.append(decision)
        if hard_gate:
            eligible.append(
                (
                    candidate_id,
                    worst_joint_floor,
                    worst_new_floor,
                    mean_h,
                    -total_edges,
                )
            )
    selected = max(eligible, key=lambda row: row[1:])[0] if eligible else D25_C0
    return selected, decisions


def _select_d35_candidate(
    folds_by_candidate: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[str, list[dict[str, Any]]]:
    """Require D35 safety/reachability and joint B3+D33 comparator closure."""

    comparator_ids = (DIAG_CANDIDATE, D33_B3_FAST)
    comparator_rows = {
        candidate_id: list(folds_by_candidate[candidate_id])
        for candidate_id in comparator_ids
    }
    thresholds = {
        "mean_after_old": max(
            float(np.mean([float(r["after_old"]["overall_accuracy"]) for r in rows]))
            for rows in comparator_rows.values()
        ),
        "mean_after_new": max(
            float(np.mean([float(r["after_new"]["overall_accuracy"]) for r in rows]))
            for rows in comparator_rows.values()
        ),
        "mean_h": max(
            float(np.mean([float(r["H_old_new"]) for r in rows]))
            for rows in comparator_rows.values()
        ),
        "mean_forgetting": min(
            float(np.mean([float(r["forgetting"]) for r in rows]))
            for rows in comparator_rows.values()
        ),
        "worst_joint_floor": max(
            min(float(r["joint_floor"]) for r in rows)
            for rows in comparator_rows.values()
        ),
    }
    comparator_old_names = tuple(
        comparator_rows[DIAG_CANDIDATE][0]["after_old"]["per_class_accuracy"]
    )
    comparator_new_names = tuple(
        comparator_rows[DIAG_CANDIDATE][0]["after_new"]["per_class_accuracy"]
    )
    classwise_thresholds = {
        "old": {
            name: max(
                float(
                    np.mean(
                        [
                            float(row["after_old"]["per_class_accuracy"][name])
                            for row in comparator_rows[candidate_id]
                        ]
                    )
                )
                for candidate_id in comparator_ids
            )
            for name in comparator_old_names
        },
        "new": {
            name: max(
                float(
                    np.mean(
                        [
                            float(row["after_new"]["per_class_accuracy"][name])
                            for row in comparator_rows[candidate_id]
                        ]
                    )
                )
                for candidate_id in comparator_ids
            )
            for name in comparator_new_names
        },
    }
    decisions: list[dict[str, Any]] = []
    eligible: list[tuple[str, float, float, float, int]] = []
    for candidate_id, raw_rows in folds_by_candidate.items():
        rows = list(raw_rows)
        aggregate = legacy._aggregate_candidate(rows)
        decision: dict[str, Any] = {
            **aggregate,
            "candidate_id": candidate_id,
            "family": (
                "d35_dense_safe_registration"
                if candidate_id in D35_CANDIDATES
                else "d33_fast_negative_control"
                if candidate_id == D33_B3_FAST
                else "d25"
                if candidate_id == D25_C0
                else "control"
            ),
            "fallback": candidate_id == D25_C0,
            "diagnostic_only": candidate_id in comparator_ids,
            "eligible_positive_route": False,
        }
        if candidate_id not in D35_CANDIDATES:
            decisions.append(decision)
            continue
        fit_old_pass = all(
            bool(r["fit_old_support_non_degradation_pass"]) for r in rows
        )
        held_intrusion_pass = all(
            bool(r["outer_held_zero_new_intrusion_pass"]) for r in rows
        )
        prefix_pass = all(
            bool(r["old_score_prefix_bitwise_unchanged"]) for r in rows
        )
        reachable_pass = all(
            bool(r["new_physical_loso_all_reachable"]) for r in rows
        )
        mean_after_old = float(
            np.mean([float(r["after_old"]["overall_accuracy"]) for r in rows])
        )
        mean_after_new = float(
            np.mean([float(r["after_new"]["overall_accuracy"]) for r in rows])
        )
        mean_h = float(np.mean([float(r["H_old_new"]) for r in rows]))
        mean_forgetting = float(
            np.mean([float(r["forgetting"]) for r in rows])
        )
        worst_joint_floor = min(float(r["joint_floor"]) for r in rows)
        worst_new_floor = min(
            float(r["after_new"]["class_floor_accuracy"]) for r in rows
        )
        new_names = tuple(rows[0]["after_new"]["per_class_accuracy"])
        old_names = tuple(rows[0]["after_old"]["per_class_accuracy"])
        mean_old_by_class = {
            name: float(
                np.mean(
                    [float(r["after_old"]["per_class_accuracy"][name]) for r in rows]
                )
            )
            for name in old_names
        }
        mean_new_by_class = {
            name: float(
                np.mean(
                    [float(r["after_new"]["per_class_accuracy"][name]) for r in rows]
                )
            )
            for name in new_names
        }
        classwise_comparator_gate = bool(
            set(mean_old_by_class) == set(classwise_thresholds["old"])
            and set(mean_new_by_class) == set(classwise_thresholds["new"])
            and all(
                mean_old_by_class[name] + 1.0e-12
                >= classwise_thresholds["old"][name]
                for name in mean_old_by_class
            )
            and all(
                mean_new_by_class[name] + 1.0e-12
                >= classwise_thresholds["new"][name]
                for name in mean_new_by_class
            )
        )
        comparator_gate = bool(
            mean_after_old + 1.0e-12 >= thresholds["mean_after_old"]
            and mean_after_new + 1.0e-12 >= thresholds["mean_after_new"]
            and mean_h > thresholds["mean_h"] + 1.0e-12
            and mean_forgetting <= thresholds["mean_forgetting"] + 1.0e-12
            and worst_joint_floor + 1.0e-12 >= thresholds["worst_joint_floor"]
            and classwise_comparator_gate
        )
        hard_gate = bool(
            fit_old_pass
            and held_intrusion_pass
            and prefix_pass
            and reachable_pass
            and comparator_gate
        )
        decision.update(
            {
                "fit_old_support_non_degradation_all_folds": fit_old_pass,
                "outer_held_zero_new_intrusion_all_folds": held_intrusion_pass,
                "old_score_prefix_bitwise_unchanged_all_folds": prefix_pass,
                "all_new_classes_reachable_all_folds": reachable_pass,
                "d35_joint_comparator_gate_pass": comparator_gate,
                "d35_classwise_comparator_gate_pass": classwise_comparator_gate,
                "d35_hard_gate_pass": hard_gate,
                "joint_comparator_thresholds": thresholds,
                "classwise_comparator_thresholds": classwise_thresholds,
                "rank_mean_after_old": mean_after_old,
                "rank_mean_after_new": mean_after_new,
                "rank_mean_forgetting": mean_forgetting,
                "rank_mean_H_old_new": mean_h,
                "rank_worst_joint_floor": worst_joint_floor,
                "rank_worst_new_floor": worst_new_floor,
                "outer_held_new_intrusion_count": int(
                    sum(int(r["outer_held_new_intrusion_count"]) for r in rows)
                ),
                "unreachable_new_class_count": int(
                    sum(int(r["unreachable_new_class_count"]) for r in rows)
                ),
                "mean_old_per_class_accuracy": mean_old_by_class,
                "mean_new_per_class_accuracy": mean_new_by_class,
                "eligible_positive_route": hard_gate,
            }
        )
        decisions.append(decision)
        if hard_gate:
            active = int(rows[0]["resource"]["active_closed_form_scalars"])
            eligible.append(
                (
                    candidate_id,
                    worst_joint_floor,
                    worst_new_floor,
                    mean_h,
                    -active,
                )
            )
    selected = max(eligible, key=lambda row: row[1:])[0] if eligible else D25_C0
    return selected, decisions


def _apply_full_k10_c3_old_support_gate(
    selected_id: str,
    candidate_decisions: list[dict[str, Any]],
    deployment_resources: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> tuple[str, str | None]:
    for decision in candidate_decisions:
        candidate_id = str(decision["candidate_id"])
        if candidate_id not in C3_CANDIDATES:
            continue
        by_scenario = {
            scenario: bool(
                deployment_resources[candidate_id][scenario][
                    "old_support_non_degradation_pass"
                ]
            )
            for scenario in legacy.FORMAL_LEO_WEAK_SCENARIOS
        }
        full_pass = all(by_scenario.values())
        decision["full_k10_old_support_non_degradation_by_scenario"] = by_scenario
        decision["full_k10_old_support_non_degradation_pass"] = full_pass
        decision["eligible_positive_route"] = bool(
            decision.get("eligible_positive_route", False) and full_pass
        )
    if selected_id not in C3_CANDIDATES:
        return selected_id, None
    selected_decision = next(
        row for row in candidate_decisions if row["candidate_id"] == selected_id
    )
    if bool(selected_decision["full_k10_old_support_non_degradation_pass"]):
        return selected_id, None
    return D25_C0, "FULL_K10_OLD_SUPPORT_NON_DEGRADATION_FAILED"


def _apply_full_k10_d26_old_support_gate(
    selected_id: str,
    candidate_decisions: list[dict[str, Any]],
    deployment_resources: Mapping[str, Mapping[str, Mapping[str, Any]]],
    candidate_ids: Sequence[str] = D26_CANDIDATES,
) -> tuple[str, str | None]:
    candidate_ids = tuple(str(value) for value in candidate_ids)
    for decision in candidate_decisions:
        candidate_id = str(decision["candidate_id"])
        if candidate_id not in candidate_ids:
            continue
        by_scenario = {
            scenario: bool(
                deployment_resources[candidate_id][scenario][
                    "old_support_non_degradation_pass"
                ]
            )
            for scenario in legacy.FORMAL_LEO_WEAK_SCENARIOS
        }
        resource_by_scenario = {
            scenario: bool(
                int(deployment_resources[candidate_id][scenario].get(
                    "peak_trainable_parameters", 80_001
                ))
                <= 80_000
                and int(deployment_resources[candidate_id][scenario].get(
                    "total_optimizer_steps", 31
                ))
                <= 30
                and bool(deployment_resources[candidate_id][scenario].get(
                    "persistent_state_cap_pass", False
                ))
                and int(deployment_resources[candidate_id][scenario].get(
                    "dense_query_graph_bytes", 1
                ))
                == 0
                and "total_post_backbone_macs_per_query"
                in deployment_resources[candidate_id][scenario]
                and bool(deployment_resources[candidate_id][scenario].get(
                    "latency_includes_argmax", False
                ))
            )
            for scenario in legacy.FORMAL_LEO_WEAK_SCENARIOS
        } if candidate_id in D32_CANDIDATES + D33_CANDIDATES else {
            scenario: True for scenario in legacy.FORMAL_LEO_WEAK_SCENARIOS
        }
        full_pass = all(by_scenario.values()) and all(resource_by_scenario.values())
        decision["full_k10_old_support_non_degradation_by_scenario"] = by_scenario
        decision["full_k10_old_support_non_degradation_pass"] = full_pass
        if candidate_id in D32_CANDIDATES + D33_CANDIDATES:
            decision["full_k10_resource_protocol_gate_by_scenario"] = (
                resource_by_scenario
            )
            decision["full_k10_resource_protocol_gate_pass"] = all(
                resource_by_scenario.values()
            )
        decision["eligible_positive_route"] = bool(
            decision.get("eligible_positive_route", False) and full_pass
        )
    if selected_id not in candidate_ids:
        return selected_id, None
    selected_decision = next(
        row for row in candidate_decisions if row["candidate_id"] == selected_id
    )
    if bool(selected_decision["full_k10_old_support_non_degradation_pass"]):
        return selected_id, None
    reason = (
        "FULL_K10_OLD_SUPPORT_OR_RESOURCE_PROTOCOL_GATE_FAILED"
        if selected_id in D32_CANDIDATES + D33_CANDIDATES
        else "FULL_K10_OLD_SUPPORT_NON_DEGRADATION_FAILED"
    )
    return D25_C0, reason


def _select_d36_candidate(
    folds_by_candidate: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[str, list[dict[str, Any]]]:
    """Select D36 only after OOF safety, reachability, floor, and comparator gates."""

    comparator_ids = (DIAG_CANDIDATE, D33_B3_FAST)
    comparators = {name: list(folds_by_candidate[name]) for name in comparator_ids}
    old_names = tuple(comparators[DIAG_CANDIDATE][0]["after_old"]["per_class_accuracy"])
    new_names = tuple(comparators[DIAG_CANDIDATE][0]["after_new"]["per_class_accuracy"])
    old_thresholds = {
        name: max(
            float(np.mean([float(row["after_old"]["per_class_accuracy"][name]) for row in comparators[cid]]))
            for cid in comparator_ids
        )
        for name in old_names
    }
    new_thresholds = {
        name: max(
            float(np.mean([float(row["after_new"]["per_class_accuracy"][name]) for row in comparators[cid]]))
            for cid in comparator_ids
        )
        for name in new_names
    }
    joint = {
        "old": max(float(np.mean([float(row["after_old"]["overall_accuracy"]) for row in comparators[cid]])) for cid in comparator_ids),
        "new": max(float(np.mean([float(row["after_new"]["overall_accuracy"]) for row in comparators[cid]])) for cid in comparator_ids),
        "h": max(float(np.mean([float(row["H_old_new"]) for row in comparators[cid]])) for cid in comparator_ids),
        "forgetting": min(float(np.mean([float(row["forgetting"]) for row in comparators[cid]])) for cid in comparator_ids),
        "joint_floor": max(min(float(row["joint_floor"]) for row in comparators[cid]) for cid in comparator_ids),
    }
    decisions: list[dict[str, Any]] = []
    eligible: list[tuple[str, float, float, float, int]] = []
    for candidate_id, raw_rows in folds_by_candidate.items():
        rows = list(raw_rows)
        aggregate = legacy._aggregate_candidate(rows)
        decision: dict[str, Any] = {
            **aggregate,
            "candidate_id": candidate_id,
            "family": (
                "d36_compiled_joint_int8"
                if candidate_id in D36_CANDIDATES
                else "d33_fast_negative_control"
                if candidate_id == D33_B3_FAST
                else "d25" if candidate_id == D25_C0 else "control"
            ),
            "fallback": candidate_id == D25_C0,
            "diagnostic_only": candidate_id in comparator_ids,
            "eligible_positive_route": False,
        }
        if candidate_id not in D36_CANDIDATES:
            decisions.append(decision)
            continue
        mean_before_by_class = {
            name: float(np.mean([float(row["before_old"]["per_class_accuracy"][name]) for row in rows]))
            for name in old_names
        }
        mean_b3_by_class = {
            name: float(np.mean([float(row["b3_reference_old"]["per_class_accuracy"][name]) for row in rows]))
            for name in old_names
        }
        mean_old = {
            name: float(np.mean([float(row["after_old"]["per_class_accuracy"][name]) for row in rows]))
            for name in old_names
        }
        mean_new = {
            name: float(np.mean([float(row["after_new"]["per_class_accuracy"][name]) for row in rows]))
            for name in new_names
        }
        quantized_old_b3_gate = all(
            mean_before_by_class[name] + 1.0e-12 >= mean_b3_by_class[name]
            for name in old_names
        )
        safety_gate = all(bool(row["outer_held_zero_new_intrusion_pass"]) for row in rows)
        reachability_gate = all(bool(row["new_physical_loso_all_reachable"]) for row in rows)
        int8_gate = all(
            bool(row["target_old_int8_prototypes_used_for_prediction"])
            and bool(row["target_new_int8_prototypes_used_for_prediction"])
            for row in rows
        )
        classwise_gate = all(mean_old[name] + 1.0e-12 >= old_thresholds[name] for name in old_names) and all(
            mean_new[name] + 1.0e-12 >= new_thresholds[name] for name in new_names
        )
        all_old_floor_gate = all(
            mean_old[name] + 1.0e-12 >= old_thresholds[name] for name in old_names
        )
        all_new_floor_gate = all(
            mean_new[name] + 1.0e-12 >= new_thresholds[name] for name in new_names
        )
        generic_floor_gate = bool(all_old_floor_gate and all_new_floor_gate)
        mean_after_old = float(np.mean([float(row["after_old"]["overall_accuracy"]) for row in rows]))
        mean_after_new = float(np.mean([float(row["after_new"]["overall_accuracy"]) for row in rows]))
        mean_h = float(np.mean([float(row["H_old_new"]) for row in rows]))
        mean_forgetting = float(np.mean([float(row["forgetting"]) for row in rows]))
        worst_joint_floor = min(float(row["joint_floor"]) for row in rows)
        joint_gate = bool(
            mean_after_old + 1.0e-12 >= joint["old"]
            and mean_after_new + 1.0e-12 >= joint["new"]
            and mean_h > joint["h"] + 1.0e-12
            and mean_forgetting <= joint["forgetting"] + 1.0e-12
            and worst_joint_floor + 1.0e-12 >= joint["joint_floor"]
        )
        hard_gate = bool(
            quantized_old_b3_gate and safety_gate and reachability_gate and int8_gate
        )
        positive = bool(
            hard_gate and classwise_gate and generic_floor_gate and joint_gate
        )
        decision.update(
            {
                "quantized_old_head_classwise_noninferior_to_b3": quantized_old_b3_gate,
                "outer_held_zero_new_intrusion_all_folds": safety_gate,
                "all_new_classes_reachable_all_folds": reachability_gate,
                "target_old_new_int8_gate_pass": int8_gate,
                "d36_classwise_comparator_gate_pass": classwise_gate,
                "d36_all_old_class_floor_gate_pass": all_old_floor_gate,
                "d36_all_new_class_floor_gate_pass": all_new_floor_gate,
                "d36_generic_floor_gate_pass": generic_floor_gate,
                "d36_joint_comparator_gate_pass": joint_gate,
                "d36_hard_gate_pass": hard_gate,
                "mean_old_per_class_accuracy": mean_old,
                "mean_new_per_class_accuracy": mean_new,
                "mean_quantized_before_old_per_class_accuracy": mean_before_by_class,
                "mean_b3_reference_old_per_class_accuracy": mean_b3_by_class,
                "classwise_comparator_thresholds": {"old": old_thresholds, "new": new_thresholds},
                "joint_comparator_thresholds": joint,
                "outer_held_new_intrusion_count": int(sum(int(row["outer_held_new_intrusion_count"]) for row in rows)),
                "unreachable_new_class_count": int(sum(int(row["unreachable_new_class_count"]) for row in rows)),
                "eligible_positive_route": positive,
            }
        )
        decisions.append(decision)
        if positive:
            eligible.append(
                (
                    candidate_id,
                    worst_joint_floor,
                    mean_h,
                    mean_after_new,
                    -int(decision["outer_held_new_intrusion_count"]),
                )
            )
    selected = max(eligible, key=lambda item: item[1:])[0] if eligible else D25_C0
    return selected, decisions


def _validate_d39_matrix_rows(
    folds_by_candidate: Mapping[str, Sequence[Mapping[str, Any]]],
) -> set[tuple[str, int]]:
    """Validate exact 6x3x5 D39 row identity and matched-key closure."""

    if tuple(folds_by_candidate) != D39_CANDIDATES:
        raise D25RunnerError("D39 exact six-candidate lock drift")
    keyed: dict[str, set[tuple[str, int]]] = {}
    for candidate_id, rows in folds_by_candidate.items():
        if len(rows) != 15 or any(
            str(row["candidate_id"]) != candidate_id for row in rows
        ):
            raise D25RunnerError("D39 candidate row identity/cardinality drift")
        if any(
            int(row["fold_index"]) not in range(len(HELD_RANKS))
            or tuple(int(value) for value in row["held_ranks"])
            != tuple(HELD_RANKS[int(row["fold_index"])])
            or int(row["held_physical_token_count"]) <= 0
            or len(str(row["held_physical_token_sha256"])) != 64
            for row in rows
        ):
            raise D25RunnerError("D39 held-rank/physical identity drift")
        keys = {(str(row["scenario"]), int(row["fold_index"])) for row in rows}
        if len(keys) != 15:
            raise D25RunnerError("D39 duplicate scene/fold key drift")
        keyed[candidate_id] = keys
    expected_keys = set(keyed[IDENTITY_CANDIDATE])
    locked_keys = {
        (scenario, fold_index)
        for scenario in legacy.FORMAL_LEO_WEAK_SCENARIOS
        for fold_index in range(len(HELD_RANKS))
    }
    if expected_keys != locked_keys or any(
        keys != expected_keys for keys in keyed.values()
    ):
        raise D25RunnerError("D39 matched 15-key scene/fold closure drift")
    reference_physical = {
        (str(row["scenario"]), int(row["fold_index"])): (
            int(row["held_physical_token_count"]),
            str(row["held_physical_token_sha256"]),
        )
        for row in folds_by_candidate[IDENTITY_CANDIDATE]
    }
    if any(
        (
            int(row["held_physical_token_count"]),
            str(row["held_physical_token_sha256"]),
        )
        != reference_physical[(str(row["scenario"]), int(row["fold_index"]))]
        for rows in folds_by_candidate.values()
        for row in rows
    ):
        raise D25RunnerError("D39 matched held physical-token closure drift")
    return expected_keys


def _select_d39_candidate(
    folds_by_candidate: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[str, list[dict[str, Any]]]:
    """Apply all 15 matched D39 gates; only formal int8 may promote."""

    expected_keys = _validate_d39_matrix_rows(folds_by_candidate)
    keyed = {
        candidate_id: {
            (str(row["scenario"]), int(row["fold_index"])): row for row in rows
        }
        for candidate_id, rows in folds_by_candidate.items()
    }
    first_key = next(iter(expected_keys))
    old_names = tuple(
        keyed[DIAG_CANDIDATE][first_key]["before_old"]["per_class_accuracy"]
    )
    new_names = tuple(
        keyed[D39_D38_B_INT8][first_key]["after_new"]["per_class_accuracy"]
    )
    d38_rows = list(folds_by_candidate[D39_D38_B_INT8])
    d39_rows = list(folds_by_candidate[D39_INT8])
    diag_rows = list(folds_by_candidate[DIAG_CANDIDATE])

    before_trajectory_gate = all(
        str(keyed[D39_INT8][key]["registration_before_prediction_sha256"])
        == str(keyed[D39_D38_B_INT8][key]["registration_before_prediction_sha256"])
        and _canonical_bytes(keyed[D39_INT8][key]["training_trace"])
        == _canonical_bytes(keyed[D39_D38_B_INT8][key]["training_trace"])
        for key in expected_keys
    )
    before_old_gate = all(
        float(keyed[D39_INT8][key]["before_old"]["per_class_accuracy"][name])
        + 1.0e-12
        >= float(keyed[DIAG_CANDIDATE][key]["before_old"]["per_class_accuracy"][name])
        for key in expected_keys
        for name in old_names
    )
    after_old_gate = all(
        float(keyed[D39_INT8][key]["after_old"]["overall_accuracy"]) + 1.0e-12
        >= float(keyed[DIAG_CANDIDATE][key]["after_old"]["overall_accuracy"])
        and float(keyed[D39_INT8][key]["forgetting"])
        <= float(keyed[DIAG_CANDIDATE][key]["forgetting"]) + 1.0e-12
        and all(
            float(keyed[D39_INT8][key]["after_old"]["per_class_accuracy"][name])
            + 1.0e-12
            >= float(
                keyed[DIAG_CANDIDATE][key]["after_old"]["per_class_accuracy"][name]
            )
            for name in old_names
        )
        for key in expected_keys
    )
    d39_intrusion = sum(int(row["outer_held_new_intrusion_count"]) for row in d39_rows)
    d38_intrusion = sum(int(row["outer_held_new_intrusion_count"]) for row in d38_rows)
    diag_intrusion = sum(int(row["outer_held_new_intrusion_count"]) for row in diag_rows)
    intrusion_gate = bool(
        d39_intrusion < d38_intrusion
        and d39_intrusion <= diag_intrusion
    )
    d39_seen_new = float(
        np.mean([float(row["after_new"]["overall_accuracy"]) for row in d39_rows])
    )
    d38_seen_new = float(
        np.mean([float(row["after_new"]["overall_accuracy"]) for row in d38_rows])
    )
    d39_confusions = sum(int(row["new_new_confusion_count"]) for row in d39_rows)
    d38_confusions = sum(int(row["new_new_confusion_count"]) for row in d38_rows)
    d39_new_mean = {
        name: float(
            np.mean(
                [float(row["after_new"]["per_class_accuracy"][name]) for row in d39_rows]
            )
        )
        for name in new_names
    }
    d38_new_mean = {
        name: float(
            np.mean(
                [float(row["after_new"]["per_class_accuracy"][name]) for row in d38_rows]
            )
        )
        for name in new_names
    }
    new_gate = bool(
        d39_seen_new + 1.0e-12 >= d38_seen_new
        and d39_confusions < d38_confusions
        and min(d39_new_mean.values()) > min(d38_new_mean.values()) + 1.0e-12
        and min(float(row["new_new_margin_min"]) for row in d39_rows)
        > min(float(row["new_new_margin_min"]) for row in d38_rows) + 1.0e-12
    )
    joint_per_key_gate = all(
        float(keyed[D39_INT8][key]["H_old_new"]) + 1.0e-12
        >= float(keyed[DIAG_CANDIDATE][key]["H_old_new"])
        and float(keyed[D39_INT8][key]["joint_floor"]) + 1.0e-12
        >= float(keyed[DIAG_CANDIDATE][key]["joint_floor"])
        for key in expected_keys
    )
    joint_aggregate_gate = bool(
        np.mean([float(row["H_old_new"]) for row in d39_rows])
        > np.mean([float(row["H_old_new"]) for row in diag_rows]) + 1.0e-12
        and np.mean([float(row["joint_floor"]) for row in d39_rows])
        > np.mean([float(row["joint_floor"]) for row in diag_rows]) + 1.0e-12
    )
    internal_precision_gate = all(
        int(row["matched_fp32_outer_argmax_change_count"]) == 0 for row in d39_rows
    )
    explicit_fp32_candidate_gate = all(
        str(keyed[D39_INT8][key]["deployment_precision"]) == "int8"
        and str(keyed[D39_FP32][key]["deployment_precision"]) == "fp32"
        and str(keyed[D39_INT8][key]["outer_prediction_sha256"])
        == str(keyed[D39_FP32][key]["outer_prediction_sha256"])
        and str(keyed[D39_INT8][key]["radius_fp16_sha256"])
        == str(keyed[D39_FP32][key]["radius_fp16_sha256"])
        and str(keyed[D39_INT8][key]["r0_fp16_sha256"])
        == str(keyed[D39_FP32][key]["r0_fp16_sha256"])
        and _canonical_bytes(keyed[D39_INT8][key]["training_trace"])
        == _canonical_bytes(keyed[D39_FP32][key]["training_trace"])
        for key in expected_keys
    )
    precision_gate = bool(internal_precision_gate and explicit_fp32_candidate_gate)
    radius_prefix_gate = all(
        bool(row["old_score_columns_bitwise_unchanged"])
        and bool(row["old_prototype_prefix_bitwise_unchanged"])
        and bool(row["old_radius_prefix_bitwise_unchanged"])
        and bool(row["r0_bitwise_unchanged"])
        and bool(row["radius_positive_finite"])
        and bool(row["radius_fp16_shared_between_int8_fp32"])
        for row in d39_rows
    )
    radius_source_gate = all(
        row["geometry_summary"]["old_radius_source_state"]
        == "registration_preceding_int8_before_state"
        and bool(
            row["geometry_summary"]["old_radius_materialized_before_stage2c"]
        )
        and int(
            row["geometry_summary"]["old_radius_materialization_hook_call_count"]
        )
        == 1
        and int(
            row["geometry_summary"][
                "old_radius_materialization_stage2b_trace_length"
            ]
        )
        == 20
        and row["geometry_summary"]["new_radius_source_state"]
        == "final_int8_append_state"
        and int(row["geometry_summary"]["old_radius_new_support_row_count"]) == 0
        and int(row["geometry_summary"]["held_radius_fit_row_count"]) == 0
        and int(row["geometry_summary"]["query_rows_used"]) == 0
        and int(row["radius_source_audit"]["old_source_held_intersection_count"])
        == 0
        and int(row["radius_source_audit"]["new_source_held_intersection_count"])
        == 0
        and int(row["radius_source_audit"]["old_source_new_class_row_count"])
        == 0
        and int(row["radius_source_audit"]["new_source_old_class_row_count"])
        == 0
        and int(row["radius_source_audit"]["query_rows_used"]) == 0
        and float(row["geometry_summary"]["radius_nu"]) == D39_RADIUS_NU
        and float(row["geometry_summary"]["radius_epsilon"])
        == D39_RADIUS_EPSILON
        and bool(row["geometry_summary"]["label_permutation_equivariant"])
        for row in d39_rows
    )
    resource_gate = all(
        int(row["resource"]["peak_trainable_parameters"]) <= 80_000
        and int(row["resource"]["adaptation_epochs"]) <= 30
        and int(row["resource"]["total_optimizer_steps"]) <= 50
        and bool(row["resource"]["persistent_state_cap_pass"])
        and str(row["resource"]["radius_storage_dtype"]) == "float16"
        and str(row["resource"]["r0_storage_dtype"]) == "float16"
        and np.isfinite(float(row["resource"]["r0_fp16"]))
        and float(row["resource"]["r0_fp16"]) > 0.0
        and int(row["resource"]["resident_fp32_target_prototype_count"]) == 0
        and int(row["resource"]["dense_query_graph_bytes"]) == 0
        and int(row["resource"]["query_rows_used_for_fit"]) == 0
        and not bool(row["resource"]["query_labels_used_for_fit"])
        and not bool(row["resource"]["query_role_oracle_access"])
        and not bool(row["resource"]["query_true_batch_class_count_access"])
        and not bool(row["resource"]["query_class_quota_access"])
        and not bool(row["resource"]["query_batch_global_assignment"])
        and not bool(row["resource"]["clean_sample_access"])
        and not bool(row["resource"]["source_sample_access"])
        and not bool(row["resource"]["class_id_specific_branch"])
        and bool(row["target_old_int8_prototypes_used_for_prediction"])
        and bool(row["target_new_int8_prototypes_used_for_prediction"])
        for row in d39_rows
    )
    positive = bool(
        before_trajectory_gate
        and before_old_gate
        and after_old_gate
        and intrusion_gate
        and new_gate
        and joint_per_key_gate
        and joint_aggregate_gate
        and precision_gate
        and radius_prefix_gate
        and radius_source_gate
        and resource_gate
    )
    decisions: list[dict[str, Any]] = []
    for candidate_id, rows in folds_by_candidate.items():
        decision: dict[str, Any] = {
            **legacy._aggregate_candidate(rows),
            "candidate_id": candidate_id,
            "family": (
                "d39_angular_radius"
                if candidate_id in (D39_INT8, D39_FP32)
                else "d38_structural_negative"
                if candidate_id == D39_D38_B_INT8
                else "d39_protonet_cda"
                if candidate_id == D39_PROTONET_CDA
                else "control"
            ),
            "fallback": candidate_id == IDENTITY_CANDIDATE,
            "diagnostic_only": candidate_id != D39_INT8,
            "eligible_positive_route": candidate_id == D39_INT8 and positive,
        }
        if candidate_id == D39_INT8:
            decision.update(
                {
                    "d39_registration_before_and_d38_trace_identity_gate_pass": before_trajectory_gate,
                    "d39_before_old_strong_b3_classwise_gate_pass": before_old_gate,
                    "d39_after_old_forgetting_classwise_gate_pass": after_old_gate,
                    "d39_intrusion_strong_b3_and_d38_gate_pass": intrusion_gate,
                    "d39_new_accuracy_confusion_floor_margin_gate_pass": new_gate,
                    "d39_joint_per_key_gate_pass": joint_per_key_gate,
                    "d39_joint_15_key_aggregate_strict_gate_pass": joint_aggregate_gate,
                    "d39_int8_fp32_outer_argmax_invariance_gate_pass": precision_gate,
                    "d39_internal_matched_fp32_gate_pass": internal_precision_gate,
                    "d39_explicit_fp32_candidate_gate_pass": explicit_fp32_candidate_gate,
                    "d39_old_base_radius_r0_prefix_gate_pass": radius_prefix_gate,
                    "d39_radius_source_protocol_gate_pass": radius_source_gate,
                    "d39_resource_protocol_gate_pass": resource_gate,
                    "outer_held_new_intrusion_count": d39_intrusion,
                    "d38_negative_outer_held_new_intrusion_count": d38_intrusion,
                    "strong_b3_outer_held_new_intrusion_count": diag_intrusion,
                    "mean_seen_new_accuracy": d39_seen_new,
                    "d38_negative_mean_seen_new_accuracy": d38_seen_new,
                    "new_new_confusion_count": d39_confusions,
                    "d38_negative_new_new_confusion_count": d38_confusions,
                    "mean_new_per_class_accuracy": d39_new_mean,
                    "d38_negative_mean_new_per_class_accuracy": d38_new_mean,
                }
            )
        decisions.append(decision)
    return (D39_INT8 if positive else IDENTITY_CANDIDATE), decisions


def _validate_d40_matrix_rows(
    folds_by_candidate: Mapping[str, Sequence[Mapping[str, Any]]],
) -> set[tuple[str, int]]:
    """Validate the exact 6x3x5 D40 matrix and held-physical closure."""

    if tuple(folds_by_candidate) != D40_CANDIDATES:
        raise D25RunnerError("D40 exact six-candidate lock drift")
    keyed: dict[str, set[tuple[str, int]]] = {}
    for candidate_id, rows in folds_by_candidate.items():
        if len(rows) != 15 or any(
            str(row["candidate_id"]) != candidate_id for row in rows
        ):
            raise D25RunnerError("D40 candidate row identity/cardinality drift")
        if any(
            int(row["fold_index"]) not in range(len(HELD_RANKS))
            or tuple(int(value) for value in row["held_ranks"])
            != tuple(HELD_RANKS[int(row["fold_index"])])
            or int(row["held_physical_token_count"]) <= 0
            or len(str(row["held_physical_token_sha256"])) != 64
            for row in rows
        ):
            raise D25RunnerError("D40 held-rank/physical identity drift")
        keys = {(str(row["scenario"]), int(row["fold_index"])) for row in rows}
        if len(keys) != 15:
            raise D25RunnerError("D40 duplicate scene/fold key drift")
        keyed[candidate_id] = keys
    expected_keys = set(keyed[IDENTITY_CANDIDATE])
    locked_keys = {
        (scenario, fold_index)
        for scenario in legacy.FORMAL_LEO_WEAK_SCENARIOS
        for fold_index in range(len(HELD_RANKS))
    }
    if expected_keys != locked_keys or any(
        keys != expected_keys for keys in keyed.values()
    ):
        raise D25RunnerError("D40 matched 15-key scene/fold closure drift")
    reference_physical = {
        (str(row["scenario"]), int(row["fold_index"])): (
            int(row["held_physical_token_count"]),
            str(row["held_physical_token_sha256"]),
        )
        for row in folds_by_candidate[IDENTITY_CANDIDATE]
    }
    if any(
        (
            int(row["held_physical_token_count"]),
            str(row["held_physical_token_sha256"]),
        )
        != reference_physical[(str(row["scenario"]), int(row["fold_index"]))]
        for rows in folds_by_candidate.values()
        for row in rows
    ):
        raise D25RunnerError("D40 matched held physical-token closure drift")
    return expected_keys


def _select_d40_candidate(
    folds_by_candidate: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[str, list[dict[str, Any]]]:
    """Apply every preregistered matched D40 gate; only int8 may promote."""

    expected_keys = _validate_d40_matrix_rows(folds_by_candidate)
    keyed = {
        candidate_id: {
            (str(row["scenario"]), int(row["fold_index"])): row for row in rows
        }
        for candidate_id, rows in folds_by_candidate.items()
    }
    first_key = next(iter(expected_keys))
    old_names = tuple(
        keyed[DIAG_CANDIDATE][first_key]["before_old"]["per_class_accuracy"]
    )
    new_names = tuple(
        keyed[DIAG_CANDIDATE][first_key]["after_new"]["per_class_accuracy"]
    )
    d40_rows = list(folds_by_candidate[D40_INT8])
    d38_rows = list(folds_by_candidate[D40_D38_B_INT8])
    diag_rows = list(folds_by_candidate[DIAG_CANDIDATE])

    before_classwise_gate = all(
        float(keyed[D40_INT8][key]["before_old"]["per_class_accuracy"][name])
        + 1.0e-12
        >= float(keyed[DIAG_CANDIDATE][key]["before_old"]["per_class_accuracy"][name])
        for key in expected_keys
        for name in old_names
    )
    before_aggregate_strict_gate = bool(
        np.mean([float(row["before_old"]["overall_accuracy"]) for row in d40_rows])
        > np.mean([float(row["before_old"]["overall_accuracy"]) for row in diag_rows])
        + 1.0e-12
    )
    before_gate = bool(before_classwise_gate and before_aggregate_strict_gate)
    after_old_gate = all(
        float(keyed[D40_INT8][key]["after_old"]["overall_accuracy"]) + 1.0e-12
        >= float(keyed[DIAG_CANDIDATE][key]["after_old"]["overall_accuracy"])
        and float(keyed[D40_INT8][key]["forgetting"])
        <= float(keyed[DIAG_CANDIDATE][key]["forgetting"]) + 1.0e-12
        and all(
            float(keyed[D40_INT8][key]["after_old"]["per_class_accuracy"][name])
            + 1.0e-12
            >= float(
                keyed[DIAG_CANDIDATE][key]["after_old"]["per_class_accuracy"][name]
            )
            for name in old_names
        )
        for key in expected_keys
    )
    d40_intrusion = sum(int(row["outer_held_new_intrusion_count"]) for row in d40_rows)
    d38_intrusion = sum(int(row["outer_held_new_intrusion_count"]) for row in d38_rows)
    diag_intrusion = sum(int(row["outer_held_new_intrusion_count"]) for row in diag_rows)
    intrusion_gate = bool(d40_intrusion < diag_intrusion and d40_intrusion <= 33)
    d40_confusions = sum(int(row["new_new_confusion_count"]) for row in d40_rows)
    d38_confusions = sum(int(row["new_new_confusion_count"]) for row in d38_rows)
    diag_confusions = sum(int(row["new_new_confusion_count"]) for row in diag_rows)
    d40_new_mean = {
        name: float(
            np.mean(
                [float(row["after_new"]["per_class_accuracy"][name]) for row in d40_rows]
            )
        )
        for name in new_names
    }
    diag_new_mean = {
        name: float(
            np.mean(
                [float(row["after_new"]["per_class_accuracy"][name]) for row in diag_rows]
            )
        )
        for name in new_names
    }
    new_gate = bool(
        all(
            float(keyed[D40_INT8][key]["after_new"]["overall_accuracy"])
            + 1.0e-12
            >= float(keyed[DIAG_CANDIDATE][key]["after_new"]["overall_accuracy"])
            for key in expected_keys
        )
        and d40_confusions < D40_NEW_NEW_CONFUSION_CAP
        and min(d40_new_mean.values()) > min(diag_new_mean.values()) + 1.0e-12
        and min(float(row["new_new_margin_min"]) for row in d40_rows)
        > min(float(row["new_new_margin_min"]) for row in diag_rows) + 1.0e-12
    )
    joint_per_key_gate = all(
        float(keyed[D40_INT8][key]["H_old_new"]) + 1.0e-12
        >= float(keyed[DIAG_CANDIDATE][key]["H_old_new"])
        and float(keyed[D40_INT8][key]["joint_floor"]) + 1.0e-12
        >= float(keyed[DIAG_CANDIDATE][key]["joint_floor"])
        for key in expected_keys
    )
    joint_aggregate_gate = bool(
        np.mean([float(row["H_old_new"]) for row in d40_rows])
        > np.mean([float(row["H_old_new"]) for row in diag_rows]) + 1.0e-12
        and np.mean([float(row["joint_floor"]) for row in d40_rows])
        > np.mean([float(row["joint_floor"]) for row in diag_rows]) + 1.0e-12
    )
    internal_precision_gate = all(
        int(row["matched_fp32_outer_argmax_change_count"]) == 0 for row in d40_rows
    )
    explicit_fp32_candidate_gate = all(
        str(keyed[D40_INT8][key]["deployment_precision"]) == "int8"
        and str(keyed[D40_FP32][key]["deployment_precision"]) == "fp32"
        and str(keyed[D40_INT8][key]["outer_prediction_sha256"])
        == str(keyed[D40_FP32][key]["outer_prediction_sha256"])
        and str(keyed[D40_INT8][key]["registration_before_prediction_sha256"])
        == str(keyed[D40_FP32][key]["registration_before_prediction_sha256"])
        and _canonical_bytes(keyed[D40_INT8][key]["training_trace"])
        == _canonical_bytes(keyed[D40_FP32][key]["training_trace"])
        for key in expected_keys
    )
    precision_gate = bool(internal_precision_gate and explicit_fp32_candidate_gate)
    prefix_gate = all(
        bool(row["old_score_columns_bitwise_unchanged"])
        and bool(row["old_base_prefix_bitwise_unchanged"])
        and bool(row["geometry_summary"]["old_prefix_bitwise_unchanged"])
        for row in d40_rows
    )
    protocol_gate = all(
        str(row["geometry_summary"]["stage2c_solver"])
        == "zero_step_synchronous_new_hnbr"
        and float(row["geometry_summary"]["hnbr_temperature"])
        == D40_HNBR_TEMPERATURE
        and bool(row["geometry_summary"]["stable_softmax_subtracts_row_max"])
        and bool(row["geometry_summary"]["positive_projection_only"])
        and bool(row["geometry_summary"]["old_hnbr_synchronous"])
        and bool(row["geometry_summary"]["new_hnbr_synchronous"])
        and not bool(
            row["geometry_summary"][
                "new_hnbr_uses_residualized_new_direction_as_negative"
            ]
        )
        and str(row["geometry_summary"]["new_hnbr_old_negative_precision"])
        == "int8_decoded"
        and bool(
            row["geometry_summary"][
                "new_hnbr_old_negative_matches_before_int8_decode"
            ]
        )
        and not bool(
            row["geometry_summary"]["old_fp32_reference_used_as_new_hnbr_negative"]
        )
        and bool(row["geometry_summary"]["label_permutation_equivariant"])
        and not bool(row["geometry_summary"]["class_id_specific_branch"])
        and not bool(
            row["geometry_summary"]["fp32_target_direction_stored_in_formal_state"]
        )
        and int(row["geometry_summary"]["query_rows_used"]) == 0
        and int(row["direction_source_audit"]["old_source_held_intersection_count"])
        == 0
        and int(row["direction_source_audit"]["new_source_held_intersection_count"])
        == 0
        and int(row["direction_source_audit"]["old_source_new_class_row_count"])
        == 0
        and int(row["direction_source_audit"]["new_source_old_class_row_count"])
        == 0
        and int(row["direction_source_audit"]["held_direction_fit_row_count"]) == 0
        and int(row["direction_source_audit"]["query_rows_used"]) == 0
        for row in d40_rows
    )
    resource_gate = all(
        int(row["resource"]["peak_trainable_parameters"]) <= 2016
        and int(row["resource"]["adaptation_epochs"]) == 20
        and int(row["resource"]["total_optimizer_steps"]) == 20
        and int(row["resource"]["stage2c_optimizer_steps"]) == 0
        and bool(row["resource"]["persistent_state_cap_pass"])
        and np.isfinite(float(row["resource"]["estimated_hnbr_support_macs"]))
        and float(row["resource"]["estimated_hnbr_support_macs"]) > 0.0
        and int(row["resource"]["resident_fp32_target_prototype_count"]) == 0
        and int(row["resource"]["dense_query_graph_bytes"]) == 0
        and int(row["resource"]["query_rows_used_for_fit"]) == 0
        and not bool(row["resource"]["query_labels_used_for_fit"])
        and not bool(row["resource"]["query_role_oracle_access"])
        and not bool(row["resource"]["query_true_batch_class_count_access"])
        and not bool(row["resource"]["query_class_quota_access"])
        and not bool(row["resource"]["query_batch_global_assignment"])
        and not bool(row["resource"]["clean_sample_access"])
        and not bool(row["resource"]["source_sample_access"])
        and bool(row["target_old_int8_prototypes_used_for_prediction"])
        and bool(row["target_new_int8_prototypes_used_for_prediction"])
        for row in d40_rows
    )
    positive = bool(
        before_gate
        and after_old_gate
        and intrusion_gate
        and new_gate
        and joint_per_key_gate
        and joint_aggregate_gate
        and precision_gate
        and prefix_gate
        and protocol_gate
        and resource_gate
    )
    decisions: list[dict[str, Any]] = []
    for candidate_id, rows in folds_by_candidate.items():
        decision: dict[str, Any] = {
            **legacy._aggregate_candidate(rows),
            "candidate_id": candidate_id,
            "family": (
                "d40_hnbr"
                if candidate_id in (D40_INT8, D40_FP32)
                else "d38_structural_negative"
                if candidate_id == D40_D38_B_INT8
                else "d40_protonet_cda"
                if candidate_id == D40_PROTONET_CDA
                else "control"
            ),
            "fallback": candidate_id == IDENTITY_CANDIDATE,
            "diagnostic_only": candidate_id != D40_INT8,
            "eligible_positive_route": candidate_id == D40_INT8 and positive,
        }
        if candidate_id == D40_INT8:
            decision.update(
                {
                    "d40_before_old_strong_b3_classwise_gate_pass": before_classwise_gate,
                    "d40_before_old_aggregate_strict_gate_pass": before_aggregate_strict_gate,
                    "d40_after_old_forgetting_classwise_gate_pass": after_old_gate,
                    "d40_intrusion_strong_b3_absolute_gate_pass": intrusion_gate,
                    "d40_new_accuracy_confusion_floor_margin_gate_pass": new_gate,
                    "d40_joint_per_key_gate_pass": joint_per_key_gate,
                    "d40_joint_15_key_aggregate_strict_gate_pass": joint_aggregate_gate,
                    "d40_int8_fp32_outer_argmax_invariance_gate_pass": precision_gate,
                    "d40_internal_matched_fp32_gate_pass": internal_precision_gate,
                    "d40_explicit_fp32_candidate_gate_pass": explicit_fp32_candidate_gate,
                    "d40_old_prefix_gate_pass": prefix_gate,
                    "d40_hnbr_source_protocol_gate_pass": protocol_gate,
                    "d40_resource_protocol_gate_pass": resource_gate,
                    "outer_held_new_intrusion_count": d40_intrusion,
                    "strong_b3_outer_held_new_intrusion_count": diag_intrusion,
                    "d38_negative_outer_held_new_intrusion_count": d38_intrusion,
                    "new_new_confusion_count": d40_confusions,
                    "strong_b3_new_new_confusion_count": diag_confusions,
                    "d38_negative_new_new_confusion_count": d38_confusions,
                    "mean_new_per_class_accuracy": d40_new_mean,
                    "strong_b3_mean_new_per_class_accuracy": diag_new_mean,
                }
            )
        decisions.append(decision)
    return (D40_INT8 if positive else IDENTITY_CANDIDATE), decisions


def _select_d38_candidate(
    folds_by_candidate: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[str, list[dict[str, Any]]]:
    """Apply the globally matched D38 selector; only B-int8 is promotable."""

    if tuple(folds_by_candidate) != D38_CANDIDATES:
        raise D25RunnerError("D38 exact six-candidate lock drift")
    by_candidate_key = {
        candidate_id: {
            (str(row["scenario"]), int(row["fold_index"])): row for row in rows
        }
        for candidate_id, rows in folds_by_candidate.items()
    }
    expected_keys = set(by_candidate_key[IDENTITY_CANDIDATE])
    if len(expected_keys) != 15 or any(
        set(rows) != expected_keys for rows in by_candidate_key.values()
    ):
        raise D25RunnerError("D38 matched scene/fold closure drift")
    old_names = tuple(
        by_candidate_key[DIAG_CANDIDATE][next(iter(expected_keys))]["before_old"][
            "per_class_accuracy"
        ]
    )
    new_names = tuple(
        by_candidate_key[DIAG_CANDIDATE][next(iter(expected_keys))]["after_new"][
            "per_class_accuracy"
        ]
    )
    a_rows = list(folds_by_candidate[D38_A_INT8])
    b_rows = list(folds_by_candidate[D38_B_INT8])
    mean = lambda rows, key: float(np.mean([float(row[key]) for row in rows]))
    a_mean_new = float(
        np.mean([float(row["after_new"]["overall_accuracy"]) for row in a_rows])
    )
    b_mean_new = float(
        np.mean([float(row["after_new"]["overall_accuracy"]) for row in b_rows])
    )
    a_min_new = min(float(row["after_new"]["class_floor_accuracy"]) for row in a_rows)
    b_min_new = min(float(row["after_new"]["class_floor_accuracy"]) for row in b_rows)
    a_margin = mean(a_rows, "new_new_margin_mean")
    b_margin = mean(b_rows, "new_new_margin_mean")
    a_confusions = sum(int(row["new_new_confusion_count"]) for row in a_rows)
    b_confusions = sum(int(row["new_new_confusion_count"]) for row in b_rows)
    a_h = mean(a_rows, "H_old_new")
    b_h = mean(b_rows, "H_old_new")
    b_improves_a = bool(
        b_mean_new > a_mean_new + 1.0e-12
        and b_min_new > a_min_new + 1.0e-12
        and b_margin > a_margin + 1.0e-12
        and b_confusions < a_confusions
        and b_h > a_h + 1.0e-12
    )
    decisions: list[dict[str, Any]] = []
    for candidate_id, raw_rows in folds_by_candidate.items():
        rows = list(raw_rows)
        decision: dict[str, Any] = {
            **legacy._aggregate_candidate(rows),
            "candidate_id": candidate_id,
            "family": (
                "d38_strong_b3_quantized"
                if candidate_id in D38_METHOD_CANDIDATES
                else "d38_protonet_cda"
                if candidate_id == D38_PROTONET_CDA
                else "control"
            ),
            "fallback": candidate_id == IDENTITY_CANDIDATE,
            "diagnostic_only": candidate_id != D38_B_INT8,
            "eligible_positive_route": False,
        }
        if candidate_id != D38_B_INT8:
            decisions.append(decision)
            continue
        old_before_gate = all(
            float(row["before_old"]["per_class_accuracy"][name]) + 1.0e-12
            >= float(
                by_candidate_key[DIAG_CANDIDATE][key]["before_old"][
                    "per_class_accuracy"
                ][name]
            )
            for key, row in by_candidate_key[candidate_id].items()
            for name in old_names
        )
        comparator_gate = True
        for key, row in by_candidate_key[candidate_id].items():
            comparators = [
                by_candidate_key[name][key]
                for name in (
                    IDENTITY_CANDIDATE,
                    D38_PROTONET_CDA,
                    DIAG_CANDIDATE,
                )
            ]
            comparator_gate = bool(
                comparator_gate
                and float(row["after_old"]["overall_accuracy"]) + 1.0e-12
                >= max(float(item["after_old"]["overall_accuracy"]) for item in comparators)
                and float(row["after_new"]["overall_accuracy"]) + 1.0e-12
                >= max(float(item["after_new"]["overall_accuracy"]) for item in comparators)
                and float(row["H_old_new"]) + 1.0e-12
                >= max(float(item["H_old_new"]) for item in comparators)
                and float(row["forgetting"])
                <= min(float(item["forgetting"]) for item in comparators) + 1.0e-12
                and float(row["joint_floor"]) + 1.0e-12
                >= max(float(item["joint_floor"]) for item in comparators)
                and all(
                    float(row["after_old"]["per_class_accuracy"][name]) + 1.0e-12
                    >= max(
                        float(item["after_old"]["per_class_accuracy"][name])
                        for item in comparators
                    )
                    for name in old_names
                )
                and all(
                    float(row["after_new"]["per_class_accuracy"][name]) + 1.0e-12
                    >= max(
                        float(item["after_new"]["per_class_accuracy"][name])
                        for item in comparators
                    )
                    for name in new_names
                )
            )
        intrusion_gate = all(
            int(row["outer_held_new_intrusion_count"])
            <= int(
                by_candidate_key[DIAG_CANDIDATE][key][
                    "outer_held_new_intrusion_count"
                ]
            )
            for key, row in by_candidate_key[candidate_id].items()
        )
        fp32_gate = all(
            int(row["matched_fp32_outer_argmax_change_count"]) == 0 for row in rows
        )
        prefix_gate = all(
            bool(row["old_score_columns_bitwise_unchanged"]) for row in rows
        )
        resource_gate = all(
            int(row["resource"]["peak_trainable_parameters"]) <= 80_000
            and int(row["resource"]["adaptation_epochs"]) <= 30
            and int(row["resource"]["total_optimizer_steps"]) <= 50
            and bool(row["resource"]["persistent_state_cap_pass"])
            and int(row["resource"]["dense_query_graph_bytes"]) == 0
            and int(row["resource"]["query_rows_used_for_fit"]) == 0
            and bool(row["target_old_int8_prototypes_used_for_prediction"])
            and bool(row["target_new_int8_prototypes_used_for_prediction"])
            for row in rows
        )
        positive = bool(
            old_before_gate
            and b_improves_a
            and comparator_gate
            and intrusion_gate
            and fp32_gate
            and prefix_gate
            and resource_gate
        )
        decision.update(
            {
                "d38_quantized_before_old_strong_b3_classwise_gate_pass": old_before_gate,
                "d38_b_improves_a_gate_pass": b_improves_a,
                "d38_matched_comparator_gate_pass": comparator_gate,
                "d38_intrusion_not_worse_than_strong_b3_gate_pass": intrusion_gate,
                "d38_int8_fp32_outer_argmax_invariance_gate_pass": fp32_gate,
                "d38_old_prefix_bitwise_gate_pass": prefix_gate,
                "d38_resource_protocol_gate_pass": resource_gate,
                "d38_a_vs_b": {
                    "a_mean_new": a_mean_new,
                    "b_mean_new": b_mean_new,
                    "a_min_new": a_min_new,
                    "b_min_new": b_min_new,
                    "a_mean_new_new_margin": a_margin,
                    "b_mean_new_new_margin": b_margin,
                    "a_new_new_confusion_count": a_confusions,
                    "b_new_new_confusion_count": b_confusions,
                    "a_mean_h": a_h,
                    "b_mean_h": b_h,
                },
                "eligible_positive_route": positive,
            }
        )
        decisions.append(decision)
    selected = D38_B_INT8 if any(
        row["candidate_id"] == D38_B_INT8 and row["eligible_positive_route"]
        for row in decisions
    ) else IDENTITY_CANDIDATE
    return selected, decisions


def _select_d37_candidate(
    folds_by_candidate: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[str, list[dict[str, Any]]]:
    """Require feasible OOF intervals plus strict B3/D33 joint gates."""

    comparator_ids = (IDENTITY_CANDIDATE, DIAG_CANDIDATE, D33_B3_FAST)
    comparators = {name: list(folds_by_candidate[name]) for name in comparator_ids}
    comparator_by_key = {
        candidate_id: {
            (str(row["scenario"]), int(row["fold_index"])): row
            for row in candidate_rows
        }
        for candidate_id, candidate_rows in comparators.items()
    }
    expected_keys = set(comparator_by_key[DIAG_CANDIDATE])
    if any(set(rows_by_key) != expected_keys for rows_by_key in comparator_by_key.values()):
        raise D25RunnerError("D37 matched comparator row closure drift")
    old_names = tuple(comparators[DIAG_CANDIDATE][0]["after_old"]["per_class_accuracy"])
    new_names = tuple(comparators[DIAG_CANDIDATE][0]["after_new"]["per_class_accuracy"])
    old_thresholds = {
        name: max(
            float(
                np.mean(
                    [
                        float(row["after_old"]["per_class_accuracy"][name])
                        for row in comparators[candidate_id]
                    ]
                )
            )
            for candidate_id in comparator_ids
        )
        for name in old_names
    }
    new_thresholds = {
        name: max(
            float(
                np.mean(
                    [
                        float(row["after_new"]["per_class_accuracy"][name])
                        for row in comparators[candidate_id]
                    ]
                )
            )
            for candidate_id in comparator_ids
        )
        for name in new_names
    }
    strong_b3_before_thresholds = {
        name: float(
            np.mean(
                [
                    float(row["before_old"]["per_class_accuracy"][name])
                    for row in comparators[DIAG_CANDIDATE]
                ]
            )
        )
        for name in old_names
    }
    joint = {
        "old": max(
            float(np.mean([float(row["after_old"]["overall_accuracy"]) for row in comparators[cid]]))
            for cid in comparator_ids
        ),
        "new": max(
            float(np.mean([float(row["after_new"]["overall_accuracy"]) for row in comparators[cid]]))
            for cid in comparator_ids
        ),
        "h": max(
            float(np.mean([float(row["H_old_new"]) for row in comparators[cid]]))
            for cid in comparator_ids
        ),
        "forgetting": min(
            float(np.mean([float(row["forgetting"]) for row in comparators[cid]]))
            for cid in comparator_ids
        ),
        "joint_floor": max(
            min(float(row["joint_floor"]) for row in comparators[cid])
            for cid in comparator_ids
        ),
    }
    decisions: list[dict[str, Any]] = []
    eligible: list[tuple[str, float, float, float, int]] = []
    for candidate_id, raw_rows in folds_by_candidate.items():
        rows = list(raw_rows)
        aggregate = legacy._aggregate_candidate(rows)
        decision: dict[str, Any] = {
            **aggregate,
            "candidate_id": candidate_id,
            "family": (
                "d37_b3_preserving_residual_int8"
                if candidate_id in D37_CANDIDATES
                else "d33_fast_negative_control"
                if candidate_id == D33_B3_FAST
                else "d25" if candidate_id == D25_C0 else "control"
            ),
            "fallback": candidate_id == D25_C0,
            "diagnostic_only": candidate_id in comparator_ids,
            "eligible_positive_route": False,
        }
        if candidate_id not in D37_CANDIDATES:
            decisions.append(decision)
            continue
        mean_before = {
            name: float(
                np.mean(
                    [float(row["before_old"]["per_class_accuracy"][name]) for row in rows]
                )
            )
            for name in old_names
        }
        mean_b3_reference = {
            name: float(
                np.mean(
                    [float(row["b3_reference_old"]["per_class_accuracy"][name]) for row in rows]
                )
            )
            for name in old_names
        }
        mean_old = {
            name: float(
                np.mean(
                    [float(row["after_old"]["per_class_accuracy"][name]) for row in rows]
                )
            )
            for name in old_names
        }
        mean_new = {
            name: float(
                np.mean(
                    [float(row["after_new"]["per_class_accuracy"][name]) for row in rows]
                )
            )
            for name in new_names
        }
        row_keys = [(str(row["scenario"]), int(row["fold_index"])) for row in rows]
        if set(row_keys) != expected_keys or len(row_keys) != len(expected_keys):
            raise D25RunnerError("D37 candidate/matched comparator row closure drift")
        internal_b3_gate = all(
            float(row["before_old"]["per_class_accuracy"][name]) + 1.0e-12
            >= float(row["b3_reference_old"]["per_class_accuracy"][name])
            for row in rows
            for name in old_names
        )
        strong_b3_gate = all(
            float(row["before_old"]["per_class_accuracy"][name]) + 1.0e-12
            >= float(
                comparator_by_key[DIAG_CANDIDATE][key]["before_old"][
                    "per_class_accuracy"
                ][name]
            )
            for row, key in zip(rows, row_keys, strict=True)
            for name in old_names
        )
        feasible_gate = all(bool(row["oof_feasible_interval_pass"]) for row in rows)
        safety_gate = all(bool(row["outer_held_zero_new_intrusion_pass"]) for row in rows)
        reachability_gate = all(bool(row["new_physical_loso_all_reachable"]) for row in rows)
        prefix_gate = all(bool(row["old_score_columns_bitwise_unchanged"]) for row in rows)
        int8_gate = all(
            bool(row["target_old_int8_prototypes_used_for_prediction"])
            and bool(row["target_new_int8_prototypes_used_for_prediction"])
            for row in rows
        )
        old_floor_gate = all(
            float(row["after_old"]["per_class_accuracy"][name]) + 1.0e-12
            >= max(
                float(
                    comparator_by_key[comparator_id][key]["after_old"][
                        "per_class_accuracy"
                    ][name]
                )
                for comparator_id in comparator_ids
            )
            for row, key in zip(rows, row_keys, strict=True)
            for name in old_names
        )
        new_floor_gate = all(
            float(row["after_new"]["per_class_accuracy"][name]) + 1.0e-12
            >= max(
                float(
                    comparator_by_key[comparator_id][key]["after_new"][
                        "per_class_accuracy"
                    ][name]
                )
                for comparator_id in comparator_ids
            )
            for row, key in zip(rows, row_keys, strict=True)
            for name in new_names
        )
        classwise_gate = bool(old_floor_gate and new_floor_gate)
        mean_after_old = float(np.mean([float(row["after_old"]["overall_accuracy"]) for row in rows]))
        mean_after_new = float(np.mean([float(row["after_new"]["overall_accuracy"]) for row in rows]))
        mean_h = float(np.mean([float(row["H_old_new"]) for row in rows]))
        mean_forgetting = float(np.mean([float(row["forgetting"]) for row in rows]))
        worst_joint_floor = min(float(row["joint_floor"]) for row in rows)
        joint_gate = all(
            float(row["after_old"]["overall_accuracy"]) + 1.0e-12
            >= max(
                float(comparator_by_key[cid][key]["after_old"]["overall_accuracy"])
                for cid in comparator_ids
            )
            and float(row["after_new"]["overall_accuracy"]) + 1.0e-12
            >= max(
                float(comparator_by_key[cid][key]["after_new"]["overall_accuracy"])
                for cid in comparator_ids
            )
            and float(row["H_old_new"]) + 1.0e-12
            >= max(
                float(comparator_by_key[cid][key]["H_old_new"])
                for cid in comparator_ids
            )
            and float(row["forgetting"]) <= min(
                float(comparator_by_key[cid][key]["forgetting"])
                for cid in comparator_ids
            ) + 1.0e-12
            and float(row["joint_floor"]) + 1.0e-12
            >= max(
                float(comparator_by_key[cid][key]["joint_floor"])
                for cid in comparator_ids
            )
            for row, key in zip(rows, row_keys, strict=True)
        )
        hard_gate = bool(
            internal_b3_gate
            and strong_b3_gate
            and feasible_gate
            and safety_gate
            and reachability_gate
            and prefix_gate
            and int8_gate
        )
        positive = bool(hard_gate and classwise_gate and joint_gate)
        decision.update(
            {
                "quantized_old_head_classwise_noninferior_to_b3": internal_b3_gate,
                "quantized_old_head_classwise_noninferior_to_strong_b3": strong_b3_gate,
                "oof_feasible_interval_all_folds": feasible_gate,
                "outer_held_zero_new_intrusion_all_folds": safety_gate,
                "all_new_classes_reachable_all_folds": reachability_gate,
                "old_score_prefix_bitwise_unchanged_all_folds": prefix_gate,
                "target_old_new_int8_gate_pass": int8_gate,
                "d37_all_old_class_floor_gate_pass": old_floor_gate,
                "d37_all_new_class_floor_gate_pass": new_floor_gate,
                "d37_classwise_comparator_gate_pass": classwise_gate,
                "d37_joint_comparator_gate_pass": joint_gate,
                "d37_hard_gate_pass": hard_gate,
                "mean_old_per_class_accuracy": mean_old,
                "mean_new_per_class_accuracy": mean_new,
                "mean_quantized_before_old_per_class_accuracy": mean_before,
                "mean_b3_reference_old_per_class_accuracy": mean_b3_reference,
                "strong_b3_before_per_class_thresholds": strong_b3_before_thresholds,
                "classwise_comparator_thresholds": {
                    "old": old_thresholds,
                    "new": new_thresholds,
                },
                "joint_comparator_thresholds": joint,
                "outer_held_new_intrusion_count": int(
                    sum(int(row["outer_held_new_intrusion_count"]) for row in rows)
                ),
                "unreachable_new_class_count": int(
                    sum(int(row["unreachable_new_class_count"]) for row in rows)
                ),
                "oof_infeasible_fold_count": int(
                    sum(not bool(row["oof_feasible_interval_pass"]) for row in rows)
                ),
                "eligible_positive_route": positive,
            }
        )
        decisions.append(decision)
        if positive:
            eligible.append(
                (
                    candidate_id,
                    worst_joint_floor,
                    mean_h,
                    mean_after_new,
                    -int(decision["outer_held_new_intrusion_count"]),
                )
            )
    selected = max(eligible, key=lambda item: item[1:])[0] if eligible else D25_C0
    return selected, decisions


def _apply_full_k10_d34_gate(
    selected_id: str,
    candidate_decisions: list[dict[str, Any]],
    deployment_resources: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> tuple[str, str | None]:
    """Require the same D34 old-safety and deployment gates on full K10."""

    for decision in candidate_decisions:
        candidate_id = str(decision["candidate_id"])
        if candidate_id not in D34_CANDIDATES:
            continue
        by_scenario: dict[str, dict[str, bool]] = {}
        for scenario in legacy.FORMAL_LEO_WEAK_SCENARIOS:
            resource = deployment_resources[candidate_id][scenario]
            by_scenario[scenario] = {
                "old_support_non_degradation": bool(
                    resource["old_support_non_degradation_pass"]
                ),
                "old_score_prefix_bitwise_unchanged": bool(
                    resource["old_score_prefix_bitwise_unchanged"]
                ),
                "all_new_classes_reachable": int(
                    resource["unreachable_edge_count"]
                )
                == 0,
                "resource_protocol": bool(
                    int(resource["peak_trainable_parameters"]) <= 50_000
                    and int(resource["total_optimizer_steps"]) <= 20
                    and bool(resource["persistent_state_cap_pass"])
                    and int(resource["dense_query_graph_bytes"]) == 0
                    and bool(resource["latency_includes_argmax"])
                    and int(resource["query_rows_used_for_fit"]) == 0
                ),
            }
        fold_old_loso_pass = bool(
            decision.get("old_loso_zero_intrusion_all_folds", False)
        )
        full_pass = bool(
            fold_old_loso_pass
            and all(all(values.values()) for values in by_scenario.values())
        )
        decision["full_k10_d34_gate_by_scenario"] = by_scenario
        decision["full_k10_uses_independent_fold_old_loso_hard_gate"] = (
            fold_old_loso_pass
        )
        decision["full_k10_d34_gate_pass"] = full_pass
        decision["eligible_positive_route"] = bool(
            decision.get("eligible_positive_route", False) and full_pass
        )
    if selected_id not in D34_CANDIDATES:
        return selected_id, None
    selected_decision = next(
        row for row in candidate_decisions if row["candidate_id"] == selected_id
    )
    if bool(selected_decision["full_k10_d34_gate_pass"]):
        return selected_id, None
    return D25_C0, "FULL_K10_D34_OLD_SAFETY_OR_RESOURCE_GATE_FAILED"


def _apply_full_k10_d35_gate(
    selected_id: str,
    candidate_decisions: list[dict[str, Any]],
    deployment_resources: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> tuple[str, str | None]:
    """Apply D35 full-K10 reachability, old-safety, and resource closure."""

    for decision in candidate_decisions:
        candidate_id = str(decision["candidate_id"])
        if candidate_id not in D35_CANDIDATES:
            continue
        by_scenario: dict[str, dict[str, bool]] = {}
        for scenario in legacy.FORMAL_LEO_WEAK_SCENARIOS:
            resource = deployment_resources[candidate_id][scenario]
            by_scenario[scenario] = {
                "old_support_non_degradation": bool(
                    resource["old_support_non_degradation_pass"]
                ),
                "old_score_prefix_bitwise_unchanged": bool(
                    resource["old_score_prefix_bitwise_unchanged"]
                ),
                "all_new_classes_reachable": int(
                    resource["unreachable_new_class_count"]
                )
                == 0,
                "all_new_classes_globally_visible": bool(
                    resource["all_new_classes_globally_visible"]
                ),
                "resource_protocol": bool(
                    int(resource["peak_trainable_parameters"]) <= 50_000
                    and int(resource["total_optimizer_steps"]) <= 20
                    and bool(resource["persistent_state_cap_pass"])
                    and int(resource["dense_query_graph_bytes"]) == 0
                    and bool(resource["latency_includes_argmax"])
                    and int(resource["query_rows_used_for_fit"]) == 0
                ),
            }
        outer_pass = bool(
            decision.get("outer_held_zero_new_intrusion_all_folds", False)
        )
        full_pass = bool(
            outer_pass and all(all(values.values()) for values in by_scenario.values())
        )
        decision["full_k10_d35_gate_by_scenario"] = by_scenario
        decision["full_k10_uses_outer_held_intrusion_hard_gate"] = outer_pass
        decision["full_k10_d35_gate_pass"] = full_pass
        decision["eligible_positive_route"] = bool(
            decision.get("eligible_positive_route", False) and full_pass
        )
    if selected_id not in D35_CANDIDATES:
        return selected_id, None
    selected_decision = next(
        row for row in candidate_decisions if row["candidate_id"] == selected_id
    )
    if bool(selected_decision["full_k10_d35_gate_pass"]):
        return selected_id, None
    return D25_C0, "FULL_K10_D35_SAFETY_REACHABILITY_OR_RESOURCE_GATE_FAILED"


def _apply_full_k10_d36_gate(
    selected_id: str,
    candidate_decisions: list[dict[str, Any]],
    deployment_resources: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> tuple[str, str | None]:
    """Finalize D36 positivity only after outer gates and five-fold full-K closure."""

    for decision in candidate_decisions:
        candidate_id = str(decision["candidate_id"])
        if candidate_id not in D36_CANDIDATES:
            continue
        by_scenario: dict[str, dict[str, bool]] = {}
        for scenario in legacy.FORMAL_LEO_WEAK_SCENARIOS:
            resource = deployment_resources[candidate_id][scenario]
            by_scenario[scenario] = {
                "quantized_old_head_b3_noninferior": bool(
                    resource["quantized_old_head_classwise_noninferior_to_b3"]
                ),
                "old_support_non_degradation": bool(
                    resource["old_support_non_degradation_pass"]
                ),
                "full_support_zero_old_to_new_intrusion": int(
                    resource["full_support_old_to_new_intrusion_count"]
                )
                == 0,
                "fivefold_crossfit_no_self_participation": bool(
                    resource["full_k10_crossfit_fold_count"] == 5
                    and resource["full_k10_crossfit_no_self_participation"]
                ),
                "target_old_new_int8": bool(
                    resource["target_old_int8_prototypes_used_for_prediction"]
                    and resource["target_new_int8_prototypes_used_for_prediction"]
                    and int(resource["resident_fp32_target_prototype_count"]) == 0
                ),
                "resource_protocol": bool(
                    int(resource["peak_trainable_parameters"]) <= 50_000
                    and int(resource["total_optimizer_steps"]) <= 20
                    and bool(resource["persistent_state_cap_pass"])
                    and int(resource["dense_query_graph_bytes"]) == 0
                    and bool(resource["latency_includes_argmax"])
                    and int(resource["query_rows_used_for_fit"]) == 0
                ),
            }
        outer_pass = bool(
            decision.get("d36_hard_gate_pass", False)
            and decision.get("d36_classwise_comparator_gate_pass", False)
            and decision.get("d36_generic_floor_gate_pass", False)
            and decision.get("d36_joint_comparator_gate_pass", False)
        )
        full_pass = bool(
            outer_pass and all(all(values.values()) for values in by_scenario.values())
        )
        decision["full_k10_d36_gate_by_scenario"] = by_scenario
        decision["full_k10_uses_outer_joint_safety_generic_floor_gate"] = outer_pass
        decision["full_k10_d36_gate_pass"] = full_pass
        decision["eligible_positive_route"] = bool(
            decision.get("eligible_positive_route", False) and full_pass
        )
    if selected_id not in D36_CANDIDATES:
        return selected_id, None
    selected_decision = next(
        row for row in candidate_decisions if row["candidate_id"] == selected_id
    )
    if bool(selected_decision["full_k10_d36_gate_pass"]):
        return selected_id, None
    return D25_C0, "FULL_K10_D36_OOF_SAFETY_GENERIC_FLOOR_OR_RESOURCE_GATE_FAILED"


def _apply_full_k10_d39_gate(
    selected_id: str,
    candidate_decisions: list[dict[str, Any]],
    deployment_resources: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> tuple[str, str | None]:
    """Require selected D39 int8 full-K10 radius/resource/protocol closure."""

    if selected_id != D39_INT8:
        return selected_id, None
    decision = next(
        row for row in candidate_decisions if row["candidate_id"] == D39_INT8
    )
    by_scenario: dict[str, dict[str, bool]] = {}
    for scenario in legacy.FORMAL_LEO_WEAK_SCENARIOS:
        resource = deployment_resources[D39_INT8][scenario]
        source = resource["radius_source_audit"]
        by_scenario[scenario] = {
            "resource_protocol": bool(
                int(resource["peak_trainable_parameters"]) <= 80_000
                and int(resource["adaptation_epochs"]) <= 30
                and int(resource["total_optimizer_steps"]) <= 50
                and bool(resource["persistent_state_cap_pass"])
                and int(resource["dense_query_graph_bytes"]) == 0
                and int(resource["query_rows_used_for_fit"]) == 0
                and not bool(resource["query_labels_used_for_fit"])
                and not bool(resource["query_role_oracle_access"])
                and not bool(resource["query_true_batch_class_count_access"])
                and not bool(resource["query_class_quota_access"])
                and not bool(resource["query_batch_global_assignment"])
                and not bool(resource["clean_sample_access"])
                and not bool(resource["source_sample_access"])
                and not bool(resource["class_id_specific_branch"])
                and bool(resource["radius_label_permutation_equivariant"])
            ),
            "formal_int8_state": bool(
                resource["deployment_precision"] == "int8"
                and resource["target_old_int8_prototypes_used_for_prediction"]
                and resource["target_new_int8_prototypes_used_for_prediction"]
                and int(resource["resident_fp32_target_prototype_count"]) == 0
                and bool(resource["formal_state_int8_only"])
                and str(resource["radius_storage_dtype"]) == "float16"
                and str(resource["r0_storage_dtype"]) == "float16"
            ),
            "radius_positive_finite": bool(
                resource["radius_positive_finite"]
                and np.isfinite(float(resource["r0_fp16"]))
                and float(resource["r0_fp16"]) > 0.0
            ),
            "old_base_radius_r0_prefix": bool(
                resource["old_prefix_bitwise_unchanged"]
                and resource["old_prototype_prefix_bitwise_unchanged"]
                and resource["old_radius_prefix_bitwise_unchanged"]
                and resource["r0_bitwise_unchanged"]
            ),
            "matched_fp32_radius_and_argmax": bool(
                resource["radius_fp16_shared_between_int8_fp32"]
                and int(resource["matched_fp32_full_k10_argmax_change_count"])
                == 0
            ),
            "radius_source_protocol": bool(
                bool(resource["old_radius_materialized_before_stage2c"])
                and int(resource["old_radius_materialization_hook_call_count"])
                == 1
                and int(
                    resource["old_radius_materialization_stage2b_trace_length"]
                )
                == 20
                and int(source["old_source_new_class_row_count"]) == 0
                and int(source["new_source_old_class_row_count"]) == 0
                and int(source["held_radius_fit_row_count"]) == 0
                and int(source["query_rows_used"]) == 0
            ),
            "latency_closed": bool(
                np.isfinite(float(resource["batch1_head_latency_mean_ms"]))
                and np.isfinite(float(resource["batch1_head_latency_p95_ms"]))
                and int(resource["batch1_head_latency_sample_count"]) > 0
                and bool(resource["latency_includes_argmax"])
            ),
            "full_k10_refit_only": bool(
                resource["full_k10_refit_only_no_candidate_change"]
            ),
        }
    full_pass = bool(
        decision.get("eligible_positive_route", False)
        and all(all(values.values()) for values in by_scenario.values())
    )
    decision["full_k10_d39_gate_by_scenario"] = by_scenario
    decision["full_k10_d39_gate_pass"] = full_pass
    decision["eligible_positive_route"] = full_pass
    if full_pass:
        return selected_id, None
    return IDENTITY_CANDIDATE, "FULL_K10_D39_RADIUS_PRECISION_RESOURCE_OR_PROTOCOL_GATE_FAILED"


def _apply_full_k10_d40_gate(
    selected_id: str,
    candidate_decisions: list[dict[str, Any]],
    deployment_resources: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> tuple[str, str | None]:
    """Require selected D40 int8 full-K10 HNBR/resource/protocol closure."""

    if selected_id != D40_INT8:
        return selected_id, None
    decision = next(
        row for row in candidate_decisions if row["candidate_id"] == D40_INT8
    )
    by_scenario: dict[str, dict[str, bool]] = {}
    for scenario in legacy.FORMAL_LEO_WEAK_SCENARIOS:
        resource = deployment_resources[D40_INT8][scenario]
        source = resource["direction_source_audit"]
        by_scenario[scenario] = {
            "resource_protocol": bool(
                int(resource["peak_trainable_parameters"]) <= 2016
                and int(resource["adaptation_epochs"]) == 20
                and int(resource["total_optimizer_steps"]) == 20
                and int(resource["stage2c_optimizer_steps"]) == 0
                and bool(resource["persistent_state_cap_pass"])
                and np.isfinite(float(resource["estimated_hnbr_support_macs"]))
                and float(resource["estimated_hnbr_support_macs"]) > 0.0
                and int(resource["dense_query_graph_bytes"]) == 0
                and int(resource["query_rows_used_for_fit"]) == 0
                and not bool(resource["query_labels_used_for_fit"])
                and not bool(resource["query_role_oracle_access"])
                and not bool(resource["query_true_batch_class_count_access"])
                and not bool(resource["query_class_quota_access"])
                and not bool(resource["query_batch_global_assignment"])
                and not bool(resource["clean_sample_access"])
                and not bool(resource["source_sample_access"])
            ),
            "formal_int8_state": bool(
                resource["deployment_precision"] == "int8"
                and resource["target_old_int8_prototypes_used_for_prediction"]
                and resource["target_new_int8_prototypes_used_for_prediction"]
                and int(resource["resident_fp32_target_prototype_count"]) == 0
                and bool(resource["formal_state_int8_only"])
            ),
            "old_prefix": bool(
                resource["old_prefix_bitwise_unchanged"]
                and resource["old_base_prefix_bitwise_unchanged"]
            ),
            "matched_fp32_argmax": bool(
                int(resource["matched_fp32_full_k10_argmax_change_count"]) == 0
            ),
            "hnbr_source_protocol": bool(
                int(source["old_source_new_class_row_count"]) == 0
                and int(source["new_source_old_class_row_count"]) == 0
                and int(source["held_direction_fit_row_count"]) == 0
                and int(source["query_rows_used"]) == 0
                and str(resource["new_hnbr_old_negative_precision"])
                == "int8_decoded"
                and bool(
                    resource[
                        "new_hnbr_old_negative_matches_before_int8_decode"
                    ]
                )
                and not bool(
                    resource["old_fp32_reference_used_as_new_hnbr_negative"]
                )
                and bool(resource["hnbr_label_permutation_equivariant"])
                and not bool(resource["hnbr_class_id_specific_branch"])
            ),
            "latency_closed": bool(
                np.isfinite(float(resource["batch1_head_latency_mean_ms"]))
                and np.isfinite(float(resource["batch1_head_latency_p95_ms"]))
                and int(resource["batch1_head_latency_sample_count"]) > 0
                and bool(resource["latency_includes_argmax"])
            ),
            "full_k10_refit_only": bool(
                resource["full_k10_refit_only_no_candidate_change"]
            ),
        }
    full_pass = bool(
        decision.get("eligible_positive_route", False)
        and all(all(values.values()) for values in by_scenario.values())
    )
    decision["full_k10_d40_gate_by_scenario"] = by_scenario
    decision["full_k10_d40_gate_pass"] = full_pass
    decision["eligible_positive_route"] = full_pass
    if full_pass:
        return selected_id, None
    return IDENTITY_CANDIDATE, "FULL_K10_D40_HNBR_PRECISION_RESOURCE_OR_PROTOCOL_GATE_FAILED"


def _apply_full_k10_d38_gate(
    selected_id: str,
    candidate_decisions: list[dict[str, Any]],
    deployment_resources: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> tuple[str, str | None]:
    """Require D38-B int8 full-K10 resource, precision, and protocol closure."""

    if selected_id != D38_B_INT8:
        return selected_id, None
    decision = next(
        row for row in candidate_decisions if row["candidate_id"] == D38_B_INT8
    )
    by_scenario: dict[str, dict[str, bool]] = {}
    for scenario in legacy.FORMAL_LEO_WEAK_SCENARIOS:
        resource = deployment_resources[D38_B_INT8][scenario]
        by_scenario[scenario] = {
            "resource_protocol": bool(
                int(resource["peak_trainable_parameters"]) <= 80_000
                and int(resource["adaptation_epochs"]) <= 30
                and int(resource["total_optimizer_steps"]) <= 50
                and bool(resource["persistent_state_cap_pass"])
                and int(resource["dense_query_graph_bytes"]) == 0
                and int(resource["query_rows_used_for_fit"]) == 0
            ),
            "formal_int8_state": bool(
                resource["deployment_precision"] == "int8"
                and resource["target_old_int8_prototypes_used_for_prediction"]
                and resource["target_new_int8_prototypes_used_for_prediction"]
                and int(resource["resident_fp32_target_prototype_count"]) == 0
            ),
            "old_prefix_bitwise_unchanged": bool(
                resource["old_prefix_bitwise_unchanged"]
            ),
            "matched_fp32_argmax_invariant": int(
                resource["matched_fp32_full_k10_argmax_change_count"]
            )
            == 0,
            "full_k10_refit_only": bool(
                resource["full_k10_refit_only_no_candidate_change"]
            ),
        }
    outer_pass = bool(decision.get("eligible_positive_route", False))
    full_pass = bool(
        outer_pass and all(all(values.values()) for values in by_scenario.values())
    )
    decision["full_k10_d38_gate_by_scenario"] = by_scenario
    decision["full_k10_d38_gate_pass"] = full_pass
    decision["eligible_positive_route"] = full_pass
    if full_pass:
        return selected_id, None
    return IDENTITY_CANDIDATE, "FULL_K10_D38_PRECISION_RESOURCE_OR_PROTOCOL_GATE_FAILED"


def _apply_full_k10_d37_gate(
    selected_id: str,
    candidate_decisions: list[dict[str, Any]],
    deployment_resources: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> tuple[str, str | None]:
    """Require feasible five-fold calibration and complete D37 deployment closure."""

    for decision in candidate_decisions:
        candidate_id = str(decision["candidate_id"])
        if candidate_id not in D37_CANDIDATES:
            continue
        by_scenario: dict[str, dict[str, bool]] = {}
        for scenario in legacy.FORMAL_LEO_WEAK_SCENARIOS:
            resource = deployment_resources[candidate_id][scenario]
            by_scenario[scenario] = {
                "oof_feasible_interval": bool(resource["oof_feasible_interval_pass"]),
                "quantized_old_head_b3_noninferior": bool(
                    resource["quantized_old_head_classwise_noninferior_to_b3"]
                ),
                "old_support_non_degradation": bool(
                    resource["old_support_non_degradation_pass"]
                ),
                "full_support_zero_old_to_new_intrusion": int(
                    resource["full_support_old_to_new_intrusion_count"]
                )
                == 0,
                "fivefold_crossfit_no_self_participation": bool(
                    resource["full_k10_crossfit_fold_count"] == 5
                    and resource["full_k10_crossfit_no_self_participation"]
                ),
                "old_prefix_bitwise_unchanged": bool(
                    resource["old_prefix_bitwise_unchanged"]
                    and resource["old_score_prefix_bitwise_unchanged"]
                ),
                "all_new_classes_reachable": int(
                    resource["unreachable_new_class_count"]
                )
                == 0,
                "target_old_new_int8": bool(
                    resource["target_old_int8_prototypes_used_for_prediction"]
                    and resource["target_new_int8_prototypes_used_for_prediction"]
                    and int(resource["resident_fp32_target_prototype_count"]) == 0
                ),
                "resource_protocol": bool(
                    int(resource["peak_trainable_parameters"]) <= 80_000
                    and int(resource["total_optimizer_steps"]) <= 30
                    and bool(resource["persistent_state_cap_pass"])
                    and int(resource["dense_query_graph_bytes"]) == 0
                    and bool(resource["latency_includes_argmax"])
                    and int(resource["query_rows_used_for_fit"]) == 0
                ),
            }
        outer_pass = bool(
            decision.get("d37_hard_gate_pass", False)
            and decision.get("d37_classwise_comparator_gate_pass", False)
            and decision.get("d37_joint_comparator_gate_pass", False)
        )
        full_pass = bool(
            outer_pass and all(all(values.values()) for values in by_scenario.values())
        )
        decision["full_k10_d37_gate_by_scenario"] = by_scenario
        decision["full_k10_uses_outer_strict_gate"] = outer_pass
        decision["full_k10_d37_gate_pass"] = full_pass
        decision["eligible_positive_route"] = bool(
            decision.get("eligible_positive_route", False) and full_pass
        )
    if selected_id not in D37_CANDIDATES:
        return selected_id, None
    selected_decision = next(
        row for row in candidate_decisions if row["candidate_id"] == selected_id
    )
    if bool(selected_decision["full_k10_d37_gate_pass"]):
        return selected_id, None
    return D25_C0, "FULL_K10_D37_OOF_B3_SAFETY_REACHABILITY_OR_RESOURCE_GATE_FAILED"


def _full_d25_state_audit(
    component: object,
    rows: Mapping[str, np.ndarray],
    z_id160: np.ndarray,
    fft96: np.ndarray,
    rf32: np.ndarray,
    *,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    config: MultimodalConcatConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    labels = np.asarray(rows["labels"]).astype(str)
    old = np.isin(labels, np.asarray(old_classes))
    new = np.isin(labels, np.asarray(new_classes))
    fit_started = time.perf_counter()
    before = fit_old_concat(
        component if config.use_ground_identity_fusion else None,
        z_id160[old],
        fft96[old],
        rf32[old],
        labels[old],
        registered_classes=old_classes,
        config=config,
    )
    after = append_new_classes_concat(
        before,
        z_id160[new],
        fft96[new],
        rf32[new],
        labels[new],
        registered_classes=new_classes,
    )
    if before.old_prefix_sha256 != after.old_prefix_sha256:
        raise D25RunnerError("D25 deployment old prefix drift")
    fit_elapsed_ms = (time.perf_counter() - fit_started) * 1000.0
    registered_feature = build_concat288(
        z_id160,
        fft96,
        rf32,
        block_energy=config.block_energy,
    )
    score_elapsed_ms: list[float] = []
    for feature in registered_feature:
        score_started = time.perf_counter()
        score_one_concat(after, feature)
        score_elapsed_ms.append((time.perf_counter() - score_started) * 1000.0)
    resource = dict(after.resource_audit())
    registered_count = len(old_classes) + len(new_classes)
    identity_qknn_macs = registered_count * 10 * 160
    identity_qknn_fp16_state_bytes = registered_count * 10 * 160 * 2
    identity_qknn_fp32_state_bytes = registered_count * 10 * 160 * 4
    resource.update(
        {
            "deployment_k_shot": 10,
            "registered_class_count": registered_count,
            "old_prefix_sha256": after.old_prefix_sha256,
            "old_score_columns_bitwise_unchanged_after_registration": True,
            "identity_single_qknn_estimated_score_macs_per_query": identity_qknn_macs,
            "identity_single_qknn_fp16_sample_state_bytes": identity_qknn_fp16_state_bytes,
            "identity_single_qknn_fp32_sample_state_bytes": identity_qknn_fp32_state_bytes,
            "estimated_score_mac_ratio_vs_identity_single_qknn": float(
                resource["estimated_head_macs_per_query"] / identity_qknn_macs
            ),
            "estimated_score_mac_reduction_vs_identity_single_qknn": float(
                1.0
                - resource["estimated_head_macs_per_query"] / identity_qknn_macs
            ),
            "persistent_state_ratio_vs_identity_single_qknn_fp16": float(
                resource["persistent_state_bytes"] / identity_qknn_fp16_state_bytes
            ),
            "closed_form_registration_elapsed_ms": fit_elapsed_ms,
            "batch1_head_latency_mean_ms": float(np.mean(score_elapsed_ms)),
            "batch1_head_latency_p95_ms": float(
                np.quantile(np.asarray(score_elapsed_ms, dtype=np.float64), 0.95)
            ),
            "batch1_head_latency_sample_count": len(score_elapsed_ms),
            "head_peak_cuda_memory_bytes": 0,
            "head_runtime": "numpy_cpu",
            "query_role_oracle_access": False,
            "query_true_batch_class_count_access": False,
            "query_class_quota_access": False,
            "query_batch_global_assignment": False,
            "source_sample_access": False,
            "clean_sample_access": False,
        }
    )
    return resource, after.geometry_audit()


def _full_c3_state_audit(
    rows: Mapping[str, np.ndarray],
    z_id160: np.ndarray,
    fft96: np.ndarray,
    rf32: np.ndarray,
    *,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    config: D25C3Config,
) -> tuple[dict[str, Any], dict[str, Any]]:
    labels = np.asarray(rows["labels"]).astype(str)
    old = np.isin(labels, np.asarray(old_classes))
    new = np.isin(labels, np.asarray(new_classes))
    features = build_concat288(z_id160, fft96, rf32)
    fit_started = time.perf_counter()
    before_fit = fit_stage2b_diag_floor(
        features[old], labels[old], old_classes, config=config
    )
    before = before_fit.state
    after_fit = append_stage2c_new_suffix(
        before, features[new], labels[new], new_classes
    )
    after = after_fit.state
    fit_elapsed_ms = (time.perf_counter() - fit_started) * 1000.0
    if (
        before.old_prefix_sha256 != after.old_prefix_sha256
        or before.shared_sha256 != after.shared_sha256
    ):
        raise D25RunnerError("C3 deployment frozen state drift")
    before_old_predictions = [predict_one_c3(before, row)[0] for row in features[old]]
    after_old_predictions = [predict_one_c3(after, row)[0] for row in features[old]]
    before_old_metric = legacy._metric_block(
        labels[old], before_old_predictions, old_classes
    )
    after_old_metric = legacy._metric_block(
        labels[old], after_old_predictions, old_classes
    )
    old_support_non_degradation = all(
        float(after_old_metric["per_class_accuracy"][label]) + 1.0e-12
        >= float(before_old_metric["per_class_accuracy"][label])
        for label in old_classes
    )
    score_elapsed_ms: list[float] = []
    for feature in features:
        score_started = time.perf_counter()
        score_one_c3(after, feature)
        score_elapsed_ms.append((time.perf_counter() - score_started) * 1000.0)
    resource = dict(after.resource_audit())
    registered_count = len(old_classes) + len(new_classes)
    identity_qknn_macs = registered_count * 10 * 160
    identity_qknn_fp16_state_bytes = registered_count * 10 * 160 * 2
    resource.update(
        {
            "deployment_k_shot": 10,
            "registered_class_count": registered_count,
            "old_prefix_sha256": after.old_prefix_sha256,
            "shared_sha256": after.shared_sha256,
            "old_score_columns_bitwise_unchanged_after_registration": True,
            "old_support_before_registration": before_old_metric,
            "old_support_after_registration": after_old_metric,
            "old_support_non_degradation_pass": old_support_non_degradation,
            "identity_single_qknn_estimated_score_macs_per_query": identity_qknn_macs,
            "identity_single_qknn_fp16_sample_state_bytes": identity_qknn_fp16_state_bytes,
            "estimated_score_mac_ratio_vs_identity_single_qknn": float(
                resource["estimated_head_macs_per_query"] / identity_qknn_macs
            ),
            "persistent_state_ratio_vs_identity_single_qknn_fp16": float(
                resource["persistent_state_bytes"]
                / identity_qknn_fp16_state_bytes
            ),
            "support_adaptation_and_registration_elapsed_ms": fit_elapsed_ms,
            "batch1_head_latency_mean_ms": float(np.mean(score_elapsed_ms)),
            "batch1_head_latency_p95_ms": float(
                np.quantile(np.asarray(score_elapsed_ms, dtype=np.float64), 0.95)
            ),
            "batch1_head_latency_sample_count": len(score_elapsed_ms),
            "head_peak_cuda_memory_bytes": 0,
            "head_runtime": "numpy_cpu_fp32",
            "complete_loss_trace": list(before_fit.training_trace)
            + list(after_fit.training_trace),
            "query_role_oracle_access": False,
            "query_true_batch_class_count_access": False,
            "query_class_quota_access": False,
            "query_batch_global_assignment": False,
            "source_sample_access": False,
            "clean_sample_access": False,
        }
    )
    return resource, _c3_geometry(after)


def _full_d26_state_audit(
    rows: Mapping[str, np.ndarray],
    z_id160: np.ndarray,
    fft96: np.ndarray,
    rf32: np.ndarray,
    *,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    config: D26CompactDiagConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    labels = np.asarray(rows["labels"]).astype(str)
    old = np.isin(labels, np.asarray(old_classes))
    new = np.isin(labels, np.asarray(new_classes))
    features = build_concat288(z_id160, fft96, rf32)
    fit_started = time.perf_counter()
    before_fit = fit_stage2b_compact_diag(
        features[old], labels[old], old_classes, config=config
    )
    before = before_fit.state
    after_fit = append_stage2c_d26(
        before,
        features[new],
        labels[new],
        new_classes,
        features[old],
        labels[old],
    )
    after = after_fit.state
    fit_elapsed_ms = (time.perf_counter() - fit_started) * 1000.0
    if (
        before.old_lock_sha256 != after.old_lock_sha256
        or before.log_diag.tobytes() != after.log_diag.tobytes()
        or before.weights.tobytes()
        != after.weights[: len(old_classes)].tobytes()
    ):
        raise D25RunnerError("D26 deployment frozen state drift")
    before_old_scores = score_all_d26(before, features[old])
    after_old_scores = score_all_d26(after, features[old])
    if not np.array_equal(
        before_old_scores, after_old_scores[:, : len(old_classes)]
    ):
        raise D25RunnerError("D26 deployment old raw score prefix drift")
    before_old_predictions = (
        predict_all_d26(before, features[old]).astype(str).tolist()
    )
    after_old_predictions = (
        predict_all_d26(after, features[old]).astype(str).tolist()
    )
    before_old_metric = legacy._metric_block(
        labels[old], before_old_predictions, old_classes
    )
    after_old_metric = legacy._metric_block(
        labels[old], after_old_predictions, old_classes
    )
    classwise_pass = all(
        float(after_old_metric["per_class_accuracy"][label]) + 1.0e-12
        >= float(before_old_metric["per_class_accuracy"][label])
        for label in old_classes
    )
    floor_pass = (
        float(after_old_metric["class_floor_accuracy"]) + 1.0e-12
        >= float(before_old_metric["class_floor_accuracy"])
    )
    old_support_non_degradation = bool(classwise_pass and floor_pass)
    score_elapsed_ms: list[float] = []
    for feature in features:
        score_started = time.perf_counter()
        score_all_d26(after, feature[None, :])
        score_elapsed_ms.append((time.perf_counter() - score_started) * 1000.0)
    resource = dict(after.resource_audit())
    registered_count = len(old_classes) + len(new_classes)
    identity_qknn_macs = registered_count * 10 * 160
    identity_qknn_fp16_state_bytes = registered_count * 10 * 160 * 2
    resource.update(
        {
            "deployment_k_shot": 10,
            "registered_class_count": registered_count,
            "old_prefix_sha256": after.old_lock_sha256,
            "old_score_columns_bitwise_unchanged_after_registration": True,
            "old_support_before_registration": before_old_metric,
            "old_support_after_registration": after_old_metric,
            "old_support_classwise_non_degradation_pass": classwise_pass,
            "old_support_floor_non_degradation_pass": floor_pass,
            "old_support_non_degradation_pass": old_support_non_degradation,
            "new_group_bias": float(after.new_group_bias),
            "new_class_biases": _d26_new_class_biases(after),
            "new_group_bias_support_only_audit": json.loads(
                after.bias_audit_json
            ),
            "identity_single_qknn_estimated_score_macs_per_query": identity_qknn_macs,
            "identity_single_qknn_fp16_sample_state_bytes": identity_qknn_fp16_state_bytes,
            "estimated_score_mac_ratio_vs_identity_single_qknn": float(
                resource["estimated_macs_per_query"] / identity_qknn_macs
            ),
            "persistent_state_ratio_vs_identity_single_qknn_fp16": float(
                resource["persistent_state_bytes"]
                / identity_qknn_fp16_state_bytes
            ),
            "support_adaptation_and_registration_elapsed_ms": fit_elapsed_ms,
            "batch1_head_latency_mean_ms": float(np.mean(score_elapsed_ms)),
            "batch1_head_latency_p95_ms": float(
                np.quantile(np.asarray(score_elapsed_ms, dtype=np.float64), 0.95)
            ),
            "batch1_head_latency_sample_count": len(score_elapsed_ms),
            "head_peak_cuda_memory_bytes": 0,
            "head_runtime": "numpy_cpu_fp32",
            "complete_loss_trace": list(before_fit.loss_trace)
            + list(after_fit.loss_trace),
            "query_role_oracle_access": False,
            "query_true_batch_class_count_access": False,
            "query_class_quota_access": False,
            "query_batch_global_assignment": False,
            "source_sample_access": False,
            "clean_sample_access": False,
        }
    )
    return resource, _d26_geometry(after)


def _full_d28_state_audit(
    rows: Mapping[str, np.ndarray],
    z_id160: np.ndarray,
    fft96: np.ndarray,
    rf32: np.ndarray,
    *,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    config: D28CandidateConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    labels = np.asarray(rows["labels"]).astype(str)
    ranks = np.asarray(rows["ranks"], dtype=np.int64)
    old = np.isin(labels, np.asarray(old_classes))
    new = np.isin(labels, np.asarray(new_classes))
    all_classes = old_classes + new_classes
    features = build_concat288(z_id160, fft96, rf32)
    fit_started = time.perf_counter()
    before_fit = fit_stage2b_compact_diag(
        features[old], labels[old], old_classes, config=config.base
    )
    before = before_fit.state
    after_fit = append_stage2c_d26(
        before,
        features[new],
        labels[new],
        new_classes,
        features[old],
        labels[old],
    )
    after = after_fit.state
    raw_support_scores = score_all_d26(after, features)
    gate = fit_support_evidence_gate(
        raw_support_scores,
        labels,
        _dense_fold_shot_ranks(labels, ranks, all_classes),
        all_classes,
        len(old_classes),
        config=config.gate,
    )
    fit_elapsed_ms = (time.perf_counter() - fit_started) * 1000.0
    if (
        before.old_lock_sha256 != after.old_lock_sha256
        or before.log_diag.tobytes() != after.log_diag.tobytes()
        or before.weights.tobytes()
        != after.weights[: len(old_classes)].tobytes()
    ):
        raise D25RunnerError("D28 deployment frozen base state drift")
    before_old_predictions = (
        predict_all_d26(before, features[old]).astype(str).tolist()
    )
    raw_old_scores = score_all_d26(after, features[old])
    adjusted_old_scores = apply_support_evidence_gate(gate, raw_old_scores)
    if not np.array_equal(
        adjusted_old_scores[:, : len(old_classes)],
        raw_old_scores[:, : len(old_classes)],
    ):
        raise D25RunnerError("D28 deployment old raw score prefix drift")
    after_old_predictions = predict_with_support_evidence_gate(
        gate, raw_old_scores
    ).astype(str).tolist()
    before_old_metric = legacy._metric_block(
        labels[old], before_old_predictions, old_classes
    )
    after_old_metric = legacy._metric_block(
        labels[old], after_old_predictions, old_classes
    )
    classwise_pass = all(
        float(after_old_metric["per_class_accuracy"][label]) + 1.0e-12
        >= float(before_old_metric["per_class_accuracy"][label])
        for label in old_classes
    )
    floor_pass = (
        float(after_old_metric["class_floor_accuracy"]) + 1.0e-12
        >= float(before_old_metric["class_floor_accuracy"])
    )
    old_support_non_degradation = bool(classwise_pass and floor_pass)
    score_elapsed_ms: list[float] = []
    for feature in features:
        score_started = time.perf_counter()
        raw_score = score_all_d26(after, feature[None, :])
        apply_support_evidence_gate(gate, raw_score)
        score_elapsed_ms.append((time.perf_counter() - score_started) * 1000.0)
    base_resource = dict(after.resource_audit())
    gate_resource = dict(gate.resource_audit())
    registered_count = len(all_classes)
    identity_qknn_macs = registered_count * 10 * 160
    identity_qknn_fp16_state_bytes = registered_count * 10 * 160 * 2
    combined_query_macs = int(base_resource["estimated_macs_per_query"]) + int(
        gate_resource["estimated_gate_macs_per_query"]
    )
    combined_state = int(base_resource["persistent_state_bytes"]) + int(
        gate_resource["deployable_predictor_state_bytes"]
    )
    resource = {
        **base_resource,
        "schema": "cvs.phase2.d28_combined_resource.v1",
        "base_d27_resource": base_resource,
        "gate_resource": gate_resource,
        "gate_enabled": bool(gate.enabled),
        "gate_fit_audit": json.loads(gate.audit_json),
        "gate_fitted_parameter_count": int(gate_resource["fitted_parameter_count"]),
        "active_adaptation_parameter_count": int(
            base_resource["peak_trainable_parameters"]
            + gate_resource["fitted_parameter_count"]
        ),
        "persistent_state_bytes": combined_state,
        "external_gate_evidence_audit_bytes": int(
            gate_resource["external_evidence_audit_bytes"]
        ),
        "persistent_state_cap_pass": combined_state <= 256 * 1024,
        "estimated_macs_per_query": combined_query_macs,
        "deployment_k_shot": 10,
        "registered_class_count": registered_count,
        "old_prefix_sha256": after.old_lock_sha256,
        "old_score_columns_bitwise_unchanged_after_registration": True,
        "old_support_before_registration": before_old_metric,
        "old_support_after_registration": after_old_metric,
        "old_support_classwise_non_degradation_pass": classwise_pass,
        "old_support_floor_non_degradation_pass": floor_pass,
        "old_support_non_degradation_pass": old_support_non_degradation,
        "new_group_bias": float(after.new_group_bias),
        "new_class_biases": _d26_new_class_biases(after),
        "new_group_bias_support_only_audit": json.loads(after.bias_audit_json),
        "identity_single_qknn_estimated_score_macs_per_query": identity_qknn_macs,
        "identity_single_qknn_fp16_sample_state_bytes": (
            identity_qknn_fp16_state_bytes
        ),
        "estimated_score_mac_ratio_vs_identity_single_qknn": float(
            combined_query_macs / identity_qknn_macs
        ),
        "persistent_state_ratio_vs_identity_single_qknn_fp16": float(
            combined_state / identity_qknn_fp16_state_bytes
        ),
        "support_adaptation_and_registration_elapsed_ms": fit_elapsed_ms,
        "batch1_head_latency_mean_ms": float(np.mean(score_elapsed_ms)),
        "batch1_head_latency_p95_ms": float(
            np.quantile(np.asarray(score_elapsed_ms, dtype=np.float64), 0.95)
        ),
        "batch1_head_latency_sample_count": len(score_elapsed_ms),
        "head_peak_cuda_memory_bytes": 0,
        "head_runtime": "numpy_cpu_fp32",
        "complete_loss_trace": list(before_fit.loss_trace)
        + list(after_fit.loss_trace),
        "query_role_oracle_access": False,
        "query_true_batch_class_count_access": False,
        "query_class_quota_access": False,
        "query_batch_global_assignment": False,
        "source_sample_access": False,
        "clean_sample_access": False,
    }
    geometry = _d26_geometry(after)
    geometry["schema"] = "cvs.phase2.d28_evidence_gate_geometry.v1"
    geometry["gate_enabled"] = bool(gate.enabled)
    geometry["gate_fit_audit"] = json.loads(gate.audit_json)
    return resource, geometry


def _full_d29_state_audit(
    rows: Mapping[str, np.ndarray],
    z_id160: np.ndarray,
    fft96: np.ndarray,
    rf32: np.ndarray,
    *,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    config: D29CandidateConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    labels = np.asarray(rows["labels"]).astype(str)
    ranks = np.asarray(rows["ranks"], dtype=np.int64)
    old = np.isin(labels, np.asarray(old_classes))
    new = np.isin(labels, np.asarray(new_classes))
    all_classes = old_classes + new_classes
    features = build_concat288(z_id160, fft96, rf32)
    fit_started = time.perf_counter()
    before_fit = fit_stage2b_compact_diag(
        features[old], labels[old], old_classes, config=config.base
    )
    before = before_fit.state
    after_fit = append_stage2c_d26(
        before,
        features[new],
        labels[new],
        new_classes,
        features[old],
        labels[old],
    )
    after = after_fit.state
    raw_support_scores = score_all_d26(after, features)
    release = fit_classwise_safe_release(
        raw_support_scores,
        labels,
        _dense_fold_shot_ranks(labels, ranks, all_classes),
        all_classes,
        len(old_classes),
        config=config.release,
    )
    fit_elapsed_ms = (time.perf_counter() - fit_started) * 1000.0
    if (
        before.old_lock_sha256 != after.old_lock_sha256
        or before.log_diag.tobytes() != after.log_diag.tobytes()
        or before.weights.tobytes()
        != after.weights[: len(old_classes)].tobytes()
    ):
        raise D25RunnerError("D29 deployment frozen base state drift")

    before_old_predictions = (
        predict_all_d26(before, features[old]).astype(str).tolist()
    )
    raw_old_scores = score_all_d26(after, features[old])
    adjusted_old_scores = apply_classwise_safe_release(release, raw_old_scores)
    if not np.array_equal(
        adjusted_old_scores[:, : len(old_classes)],
        raw_old_scores[:, : len(old_classes)],
    ):
        raise D25RunnerError("D29 deployment old raw score prefix drift")
    after_old_predictions = predict_with_classwise_safe_release(
        release, raw_old_scores
    ).astype(str).tolist()
    before_old_metric = legacy._metric_block(
        labels[old], before_old_predictions, old_classes
    )
    after_old_metric = legacy._metric_block(
        labels[old], after_old_predictions, old_classes
    )
    classwise_pass = all(
        float(after_old_metric["per_class_accuracy"][label]) + 1.0e-12
        >= float(before_old_metric["per_class_accuracy"][label])
        for label in old_classes
    )
    floor_pass = (
        float(after_old_metric["class_floor_accuracy"]) + 1.0e-12
        >= float(before_old_metric["class_floor_accuracy"])
    )
    old_support_non_degradation = bool(classwise_pass and floor_pass)

    score_elapsed_ms: list[float] = []
    for feature in features:
        score_started = time.perf_counter()
        raw_score = score_all_d26(after, feature[None, :])
        apply_classwise_safe_release(release, raw_score)
        score_elapsed_ms.append((time.perf_counter() - score_started) * 1000.0)
    base_resource = dict(after.resource_audit())
    release_resource = dict(release.resource_audit())
    registered_count = len(all_classes)
    identity_qknn_macs = registered_count * 10 * 160
    identity_qknn_fp16_state_bytes = registered_count * 10 * 160 * 2
    combined_query_macs = int(base_resource["estimated_macs_per_query"])
    release_scalar_ops = int(
        release_resource["estimated_release_scalar_ops_per_query"]
    )
    combined_state = int(base_resource["persistent_state_bytes"]) + int(
        release_resource["deployable_predictor_state_bytes"]
    )
    resource = {
        **base_resource,
        "schema": "cvs.phase2.d29_combined_resource.v1",
        "base_d27_resource": base_resource,
        "release_resource": release_resource,
        "release_enabled": bool(release.enabled),
        "release_fit_audit": json.loads(release.audit_json),
        "release_fitted_parameter_count": int(
            release_resource["fitted_parameter_count"]
        ),
        "active_adaptation_parameter_count": int(
            base_resource["peak_trainable_parameters"]
            + release_resource["fitted_parameter_count"]
        ),
        "persistent_state_bytes": combined_state,
        "external_release_evidence_audit_bytes": int(
            release_resource["external_evidence_audit_bytes"]
        ),
        "persistent_state_cap_pass": combined_state <= 256 * 1024,
        "estimated_macs_per_query": combined_query_macs,
        "estimated_row_local_scalar_ops_per_query": release_scalar_ops,
        "deployment_k_shot": 10,
        "registered_class_count": registered_count,
        "old_prefix_sha256": after.old_lock_sha256,
        "old_score_columns_bitwise_unchanged_after_registration": True,
        "old_support_before_registration": before_old_metric,
        "old_support_after_registration": after_old_metric,
        "old_support_classwise_non_degradation_pass": classwise_pass,
        "old_support_floor_non_degradation_pass": floor_pass,
        "old_support_non_degradation_pass": old_support_non_degradation,
        "new_group_bias": float(after.new_group_bias),
        "new_class_biases": _d26_new_class_biases(after),
        "new_group_bias_support_only_audit": json.loads(after.bias_audit_json),
        "identity_single_qknn_estimated_score_macs_per_query": identity_qknn_macs,
        "identity_single_qknn_fp16_sample_state_bytes": (
            identity_qknn_fp16_state_bytes
        ),
        "estimated_score_mac_ratio_vs_identity_single_qknn": float(
            combined_query_macs / identity_qknn_macs
        ),
        "estimated_scalar_op_ratio_vs_identity_single_qknn": float(
            release_scalar_ops / identity_qknn_macs
        ),
        "persistent_state_ratio_vs_identity_single_qknn_fp16": float(
            combined_state / identity_qknn_fp16_state_bytes
        ),
        "support_adaptation_and_registration_elapsed_ms": fit_elapsed_ms,
        "batch1_head_latency_mean_ms": float(np.mean(score_elapsed_ms)),
        "batch1_head_latency_p95_ms": float(
            np.quantile(np.asarray(score_elapsed_ms, dtype=np.float64), 0.95)
        ),
        "batch1_head_latency_sample_count": len(score_elapsed_ms),
        "head_peak_cuda_memory_bytes": 0,
        "head_runtime": "numpy_cpu_fp32",
        "complete_loss_trace": list(before_fit.loss_trace)
        + list(after_fit.loss_trace),
        "query_features_used_for_fit": False,
        "query_labels_used_for_fit": False,
        "query_role_oracle_access": False,
        "query_true_batch_class_count_access": False,
        "query_class_quota_access": False,
        "query_batch_global_assignment": False,
        "source_sample_access": False,
        "clean_sample_access": False,
    }
    geometry = _d26_geometry(after)
    geometry["schema"] = "cvs.phase2.d29_pcsr_geometry.v1"
    geometry["release_enabled"] = bool(release.enabled)
    geometry["release_fit_audit"] = json.loads(release.audit_json)
    return resource, geometry


def _full_d30_state_audit(
    component: Any,
    rows: Mapping[str, np.ndarray],
    z_id160: np.ndarray,
    direct_logits: np.ndarray,
    fft96: np.ndarray,
    rf32: np.ndarray,
    *,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    config: D30CandidateConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    labels = np.asarray(rows["labels"]).astype(str)
    ranks = np.asarray(rows["ranks"], dtype=np.int64)
    old = np.isin(labels, np.asarray(old_classes))
    new = np.isin(labels, np.asarray(new_classes))
    all_classes = old_classes + new_classes
    features = _d1_feature_from_blocks(z_id160, fft96, rf32)
    fit_started = time.perf_counter()
    before_fit = fit_stage2b_compact_diag(
        features[old], labels[old], old_classes, config=config.base
    )
    before = before_fit.state
    after_fit = append_stage2c_d26(
        before,
        features[new],
        labels[new],
        new_classes,
        features[old],
        labels[old],
    )
    after = after_fit.state
    dali_old = fit_old_dali(
        component,
        z_id160[old],
        labels[old],
        direct_logits[old],
        config=config.dali,
    )
    dali_state = register_new_dali(
        dali_old,
        z_id160[new],
        labels[new],
        registered_classes=new_classes,
    )
    raw_scores = score_all_d26(after, features)
    raw_old_scores = raw_scores[old]
    dali_old_scores = _d30_rerank_matrix(
        dali_state,
        raw_old_scores,
        z_id160[old],
        direct_logits[old],
    )
    dali_gate_pass, dali_gate_audit = _d30_old_support_gate(
        raw_old_scores,
        dali_old_scores,
        labels[old],
        old_classes,
    )
    dali_enabled = _d30_enable_dali(dali_state.k_shot, dali_gate_pass)
    dali_gate_audit["k_shot"] = int(dali_state.k_shot)
    dali_gate_audit["k1_exact_base_head_passthrough"] = bool(
        int(dali_state.k_shot) == 1
    )
    dali_gate_audit["enabled"] = dali_enabled
    dali_scores = (
        _d30_rerank_matrix(
            dali_state, raw_scores, z_id160, direct_logits
        )
        if dali_enabled
        else raw_scores.copy()
    )
    envelope = fit_max_envelope_calibration(
        dali_scores,
        labels,
        _dense_fold_shot_ranks(labels, ranks, all_classes),
        all_classes,
        len(old_classes),
        config=MaxEnvelopeCalibrationConfig(
            objective=config.envelope_objective,
            coordinate_passes=2,
        ),
    )
    adjusted_scores = apply_max_envelope_calibration(envelope, dali_scores)
    fit_elapsed_ms = (time.perf_counter() - fit_started) * 1000.0
    before_old_predictions = (
        predict_all_d26(before, features[old]).astype(str).tolist()
    )
    after_old_predictions = np.asarray(all_classes)[
        np.argmax(adjusted_scores[old], axis=1)
    ].tolist()
    before_old_metric = legacy._metric_block(
        labels[old], before_old_predictions, old_classes
    )
    after_old_metric = legacy._metric_block(
        labels[old], after_old_predictions, old_classes
    )
    classwise_pass = all(
        float(after_old_metric["per_class_accuracy"][name]) + 1.0e-12
        >= float(before_old_metric["per_class_accuracy"][name])
        for name in old_classes
    )
    floor_pass = (
        float(after_old_metric["class_floor_accuracy"]) + 1.0e-12
        >= float(before_old_metric["class_floor_accuracy"])
    )
    old_support_non_degradation = bool(classwise_pass and floor_pass)
    score_elapsed_ms: list[float] = []
    for index, feature in enumerate(features):
        score_started = time.perf_counter()
        row_raw = score_all_d26(after, feature[None, :])
        row_dali = (
            _d30_rerank_matrix(
                dali_state,
                row_raw,
                z_id160[index : index + 1],
                direct_logits[index : index + 1],
            )
            if dali_enabled
            else row_raw
        )
        apply_max_envelope_calibration(envelope, row_dali)
        score_elapsed_ms.append((time.perf_counter() - score_started) * 1000.0)
    base_resource = dict(after.resource_audit())
    dali_resource = dict(dali_state.resource_audit())
    envelope_resource = dict(envelope.resource_audit())
    dali_extra_macs = int(dali_resource["fixed_medoid_ground_macs_per_query"])
    scalar_ops = (
        (12 * len(old_classes) if dali_enabled else 0)
        + int(envelope_resource["estimated_scalar_ops_per_query"])
    )
    combined_state = (
        int(base_resource["persistent_state_bytes"])
        + int(dali_resource["persistent_state_bytes"])
        + int(envelope_resource["deployable_predictor_state_bytes"])
    )
    registered_count = len(all_classes)
    identity_qknn_macs = registered_count * 10 * 160
    identity_qknn_fp16_state_bytes = registered_count * 10 * 160 * 2
    combined_macs = int(base_resource["estimated_macs_per_query"]) + (
        dali_extra_macs if dali_enabled else 0
    )
    confusion_before_envelope = audit_envelope_confusions(
        dali_scores, labels, all_classes, len(old_classes)
    )
    confusion_after_envelope = audit_envelope_confusions(
        adjusted_scores, labels, all_classes, len(old_classes)
    )
    if (
        confusion_before_envelope["new_aggregate"]["old_win"]
        != confusion_after_envelope["new_aggregate"]["old_win"]
    ):
        raise D25RunnerError("D30 deployment envelope changed group counts")
    resource = {
        **base_resource,
        "schema": "cvs.phase2.d30_combined_resource.v1",
        "base_d27_b3_geometry_resource": base_resource,
        "dali_resource": dali_resource,
        "max_envelope_resource": envelope_resource,
        "dali_enabled_by_old_support_gate": dali_enabled,
        "dali_old_support_gate": dali_gate_audit,
        "max_envelope_enabled": bool(envelope.enabled),
        "actual_int8_component_used_for_prediction": dali_enabled,
        "int8_component_loaded_and_audited": True,
        "int8_component_state_bytes": int(dali_resource["int8_component_state_bytes"]),
        "active_adaptation_parameter_count": int(
            base_resource["peak_trainable_parameters"]
            + envelope_resource["fitted_parameter_count"]
        ),
        "persistent_state_bytes": combined_state,
        "persistent_state_cap_pass": combined_state <= 256 * 1024,
        "estimated_macs_per_query": combined_macs,
        "estimated_row_local_scalar_ops_per_query": scalar_ops,
        "deployment_k_shot": 10,
        "registered_class_count": registered_count,
        "old_support_before_registration": before_old_metric,
        "old_support_after_registration": after_old_metric,
        "old_support_classwise_non_degradation_pass": classwise_pass,
        "old_support_floor_non_degradation_pass": floor_pass,
        "old_support_non_degradation_pass": old_support_non_degradation,
        "identity_single_qknn_estimated_score_macs_per_query": identity_qknn_macs,
        "identity_single_qknn_fp16_sample_state_bytes": identity_qknn_fp16_state_bytes,
        "estimated_score_mac_ratio_vs_identity_single_qknn": float(
            combined_macs / identity_qknn_macs
        ),
        "persistent_state_ratio_vs_identity_single_qknn_fp16": float(
            combined_state / identity_qknn_fp16_state_bytes
        ),
        "support_adaptation_and_registration_elapsed_ms": fit_elapsed_ms,
        "batch1_head_latency_mean_ms": float(np.mean(score_elapsed_ms)),
        "batch1_head_latency_p95_ms": float(
            np.quantile(np.asarray(score_elapsed_ms, dtype=np.float64), 0.95)
        ),
        "batch1_head_latency_sample_count": len(score_elapsed_ms),
        "head_peak_cuda_memory_bytes": 0,
        "head_runtime": "numpy_cpu_fp32",
        "complete_loss_trace": list(before_fit.loss_trace) + list(after_fit.loss_trace),
        "feature_geometry": "b3_auxiliary_dominant_z160_fft96_rf32_v1",
        "feature_block_energy_target": {
            "z160": 1.0 / 17.0,
            "fft96_rf32_aux_total": 16.0 / 17.0,
        },
        "query_rows_used_for_fit": 0,
        "query_features_used_for_fit": False,
        "query_labels_used_for_fit": False,
        "query_role_oracle_access": False,
        "query_true_batch_class_count_access": False,
        "query_class_quota_access": False,
        "query_batch_global_assignment": False,
        "dense_query_graph_bytes": 0,
        "source_sample_access": False,
        "clean_sample_access": False,
    }
    geometry = _d26_geometry(after)
    geometry.update(
        {
            "schema": "cvs.phase2.d30_dual_envelope_geometry.v1",
            "feature_geometry": "b3_auxiliary_dominant_z160_fft96_rf32_v1",
            "observed_feature_block_energy": _d30_observed_block_energy(features),
            "dali_enabled": dali_enabled,
            "dali_old_support_gate": dali_gate_audit,
            "max_envelope_enabled": bool(envelope.enabled),
            "max_envelope_biases": [float(value) for value in envelope.biases],
            "max_envelope_fit_audit": json.loads(envelope.audit_json),
            "support_confusion_before_envelope": confusion_before_envelope,
            "support_confusion_after_envelope": confusion_after_envelope,
        }
    )
    return resource, geometry


def _full_d31_state_audit(
    component: Any,
    rows: Mapping[str, np.ndarray],
    z_id160: np.ndarray,
    direct_logits: np.ndarray,
    fft96: np.ndarray,
    rf32: np.ndarray,
    *,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    config: D31CandidateConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    labels = np.asarray(rows["labels"]).astype(str)
    old = np.isin(labels, np.asarray(old_classes))
    new = np.isin(labels, np.asarray(new_classes))
    features = _d1_feature_from_blocks(z_id160, fft96, rf32)
    fit_started = time.perf_counter()
    fit = _fit_d31_route(
        component,
        features,
        z_id160,
        direct_logits,
        labels,
        old,
        new,
        old_classes,
        new_classes,
        config,
    )
    fit_elapsed_ms = (time.perf_counter() - fit_started) * 1000.0
    before = fit["before"]
    after = fit["after"]
    before_predictions = predict_all_d26(before, features[old]).astype(str).tolist()
    final_predictions = np.asarray(old_classes + new_classes)[
        np.argmax(fit["adjusted_scores"], axis=1)
    ]
    before_metric = legacy._metric_block(labels[old], before_predictions, old_classes)
    after_metric = legacy._metric_block(
        labels[old], final_predictions[old], old_classes
    )
    classwise_pass = all(
        float(after_metric["per_class_accuracy"][name]) + 1.0e-12
        >= float(before_metric["per_class_accuracy"][name])
        for name in old_classes
    )
    floor_pass = (
        float(after_metric["class_floor_accuracy"]) + 1.0e-12
        >= float(before_metric["class_floor_accuracy"])
    )
    score_elapsed_ms: list[float] = []
    for index, feature in enumerate(features):
        score_started = time.perf_counter()
        row_raw = score_all_d31(after, feature[None, :])
        if fit["dali_enabled"]:
            _d30_rerank_matrix(
                fit["dali_state"],
                row_raw,
                z_id160[index : index + 1],
                direct_logits[index : index + 1],
            )
        score_elapsed_ms.append((time.perf_counter() - score_started) * 1000.0)
    base_resource = dict(after.resource_audit())
    accounting = _d31_dali_state_accounting(fit["dali_state"])
    dali_resource = accounting["dali_resource"]
    combined_resident = int(base_resource["persistent_state_bytes"]) + int(
        accounting["actual_current_dali_state_bytes"]
    )
    projected_active = int(base_resource["persistent_state_bytes"]) + int(
        accounting["projected_slim_dali_runtime_bytes"]
    )
    registered_count = len(old_classes) + len(new_classes)
    identity_macs = registered_count * 10 * 160
    combined_macs = int(base_resource["estimated_macs_per_query"]) + (
        int(dali_resource["fixed_medoid_ground_macs_per_query"])
        if fit["dali_enabled"]
        else 0
    )
    training_trace = list(fit["before_fit"].loss_trace) + list(
        fit["stage2c_fit"].loss_trace
    )
    resource = {
        **base_resource,
        "schema": "cvs.phase2.d31_combined_resource.v1",
        "d31_suffix_resource": base_resource,
        **accounting,
        "dali_enabled_by_old_support_gate": bool(fit["dali_enabled"]),
        "dali_old_support_gate": fit["dali_gate"],
        "actual_int8_component_used_for_prediction": bool(fit["dali_enabled"]),
        "full_bundle_resident_combined_state_bytes": combined_resident,
        "projected_slim_active_predictor_state_bytes": projected_active,
        "deployment_resource_primary_state_view": (
            "projected_slim_fixed_medoid_predictor_with_full_bundle_residency_disclosed"
        ),
        "deployable_predictor_state_bytes_projected_slim_medoid": projected_active,
        "persistent_state_bytes": combined_resident,
        "persistent_state_cap_pass": combined_resident <= 256 * 1024,
        "estimated_macs_per_query": combined_macs,
        "estimated_score_mac_ratio_vs_identity_single_qknn": float(
            combined_macs / identity_macs
        ),
        "total_optimizer_steps": int(after.stage2b_optimizer_steps)
        + int(after.stage2c_optimizer_steps),
        "total_adaptation_epochs": int(after.stage2b_optimizer_steps)
        + int(after.stage2c_optimizer_steps),
        "deployment_k_shot": 10,
        "registered_class_count": registered_count,
        "old_support_before_registration": before_metric,
        "old_support_after_registration": after_metric,
        "old_support_classwise_non_degradation_pass": classwise_pass,
        "old_support_floor_non_degradation_pass": floor_pass,
        "old_support_non_degradation_pass": bool(classwise_pass and floor_pass),
        "support_adaptation_and_registration_elapsed_ms": fit_elapsed_ms,
        "batch1_head_latency_mean_ms": float(np.mean(score_elapsed_ms)),
        "batch1_head_latency_p95_ms": float(
            np.quantile(np.asarray(score_elapsed_ms, dtype=np.float64), 0.95)
        ),
        "batch1_head_latency_sample_count": len(score_elapsed_ms),
        "head_peak_cuda_memory_bytes": 0,
        "head_runtime": "numpy_cpu_fp32",
        "complete_loss_trace": training_trace,
        "feature_geometry": "b3_auxiliary_dominant_z160_fft96_rf32_v1",
        "query_rows_used_for_fit": 0,
        "query_role_oracle_access": False,
        "query_true_batch_class_count_access": False,
        "query_class_quota_access": False,
        "query_batch_global_assignment": False,
        "dense_query_graph_bytes": 0,
        "clean_sample_access": False,
        "source_sample_access": False,
    }
    geometry = {
        "schema": "cvs.phase2.d31_all_registered_geometry.v1",
        "feature_geometry": "b3_auxiliary_dominant_z160_fft96_rf32_v1",
        "observed_feature_block_energy": _d30_observed_block_energy(features),
        "old_prefix_sha256": after.old_prefix_sha256,
        "dali_enabled": bool(fit["dali_enabled"]),
        "dali_old_support_gate": fit["dali_gate"],
        "raw_confusion": _d31_confusion_audit(
            fit["raw_scores"], labels, old_classes, new_classes
        ),
        "final_confusion": _d31_confusion_audit(
            fit["adjusted_scores"], labels, old_classes, new_classes
        ),
        "support_gate": json.loads(after.support_gate_json),
    }
    return resource, geometry


def _full_d33_state_audit(
    rows: Mapping[str, np.ndarray],
    z_id160: np.ndarray,
    fft96: np.ndarray,
    rf32: np.ndarray,
    *,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    config: D33CandidateConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Full K-shot D33 support and batch-1 deployable-head audit."""

    labels = np.asarray(rows["labels"]).astype(str)
    old = np.isin(labels, np.asarray(old_classes))
    new = np.isin(labels, np.asarray(new_classes))
    features = _d1_feature_from_blocks(z_id160, fft96, rf32)
    fit_started = time.perf_counter()
    fit = _fit_d33_route(
        features,
        labels,
        old,
        new,
        old_classes,
        new_classes,
        config,
    )
    fit_elapsed_ms = (time.perf_counter() - fit_started) * 1000.0
    after = fit["after"]
    all_classes = old_classes + new_classes
    before_scores = _score_d33_old_stage(fit, features[old])
    before_predictions = np.asarray(old_classes)[np.argmax(before_scores, axis=1)]
    final_scores = score_d33_spherical_registration(after, features)
    final_predictions = np.asarray(all_classes)[np.argmax(final_scores, axis=1)]
    before_metric = legacy._metric_block(
        labels[old], before_predictions.astype(str).tolist(), old_classes
    )
    after_metric = legacy._metric_block(
        labels[old], final_predictions[old].astype(str).tolist(), old_classes
    )
    classwise_pass = all(
        float(after_metric["per_class_accuracy"][name]) + 1.0e-12
        >= float(before_metric["per_class_accuracy"][name])
        for name in old_classes
    )
    floor_pass = (
        float(after_metric["class_floor_accuracy"]) + 1.0e-12
        >= float(before_metric["class_floor_accuracy"])
    )
    score_elapsed_ms: list[float] = []
    for feature in features:
        score_started = time.perf_counter()
        row_scores = score_d33_spherical_registration(after, feature[None, :])
        _ = int(np.argmax(row_scores[0]))
        score_elapsed_ms.append((time.perf_counter() - score_started) * 1000.0)
    resource = _d33_resource(fit, len(all_classes))
    resource.update(
        {
            "deployment_k_shot": 10,
            "registered_class_count": len(all_classes),
            "old_support_before_registration": before_metric,
            "old_support_after_registration": after_metric,
            "old_support_classwise_non_degradation_pass": classwise_pass,
            "old_support_floor_non_degradation_pass": floor_pass,
            "old_support_non_degradation_pass": bool(classwise_pass and floor_pass),
            "support_adaptation_and_registration_elapsed_ms": fit_elapsed_ms,
            "batch1_head_latency_mean_ms": float(np.mean(score_elapsed_ms)),
            "batch1_head_latency_p95_ms": float(
                np.quantile(np.asarray(score_elapsed_ms, dtype=np.float64), 0.95)
            ),
            "batch1_head_latency_sample_count": len(score_elapsed_ms),
            "head_latency_scope": "d33_spherical_score_plus_argmax",
            "latency_includes_argmax": True,
            "head_peak_cuda_memory_bytes": 0,
            "head_runtime": "numpy_cpu_fp32",
        }
    )
    before_all_old_scores = _score_d33_old_stage(fit, features)
    raw_old_unchanged = bool(
        np.array_equal(before_all_old_scores, final_scores[:, : len(old_classes)])
    )
    confusion = _d31_confusion_audit(
        final_scores, labels, old_classes, new_classes
    )
    geometry = {
        "schema": "cvs.phase2.d33_spherical_geometry.v1",
        "feature_geometry": "b3_auxiliary_dominant_z160_fft96_rf32_v1",
        "observed_feature_block_energy": _d30_observed_block_energy(features),
        "old_solver": config.old_solver,
        "selection_policy": config.registration.selection_policy,
        "raw_confusion": confusion,
        "final_confusion": confusion,
        "base_old_parameter_prefix_bitwise_unchanged": True,
        "raw_old_score_columns_bitwise_unchanged_after_registration": raw_old_unchanged,
        "final_old_score_columns_bitwise_unchanged_after_registration": raw_old_unchanged,
        "final_old_score_transform_policy": "none_after_spherical_score",
    }
    return resource, geometry


def _full_d34_state_audit(
    rows: Mapping[str, np.ndarray],
    z_id160: np.ndarray,
    fft96: np.ndarray,
    rf32: np.ndarray,
    *,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    config: D34CandidateConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Full-K10 D34 fit, frozen-prefix proof, latency, and resource audit."""

    labels = np.asarray(rows["labels"]).astype(str)
    old = np.isin(labels, np.asarray(old_classes))
    new = np.isin(labels, np.asarray(new_classes))
    features = _d1_feature_from_blocks(z_id160, fft96, rf32)
    fit_started = time.perf_counter()
    fit = _fit_d34_route(
        features,
        labels,
        old,
        new,
        old_classes,
        new_classes,
        config,
    )
    fit_elapsed_ms = (time.perf_counter() - fit_started) * 1000.0
    all_classes = old_classes + new_classes
    old_prefix, scores = _score_d34(fit, features)
    if not np.array_equal(old_prefix, scores[:, : len(old_classes)]):
        raise D25RunnerError("D34 full-K10 changed old score prefix")
    predictions = np.asarray(all_classes)[np.argmax(scores, axis=1)]
    before_predictions = np.asarray(old_classes)[np.argmax(old_prefix[old], axis=1)]
    before_metric = legacy._metric_block(
        labels[old], before_predictions.astype(str).tolist(), old_classes
    )
    after_metric = legacy._metric_block(
        labels[old], predictions[old].astype(str).tolist(), old_classes
    )
    classwise_pass = all(
        float(after_metric["per_class_accuracy"][name]) + 1.0e-12
        >= float(before_metric["per_class_accuracy"][name])
        for name in old_classes
    )
    floor_pass = (
        float(after_metric["class_floor_accuracy"]) + 1.0e-12
        >= float(before_metric["class_floor_accuracy"])
    )
    score_elapsed_ms: list[float] = []
    for feature in features:
        started = time.perf_counter()
        adapted, prefix = _d34_fast_unit_and_prefix(
            fit["old_state"], feature[None, :]
        )
        row_scores = score_d34_collision_local_registration(
            fit["after"], adapted, prefix
        )
        _ = int(np.argmax(row_scores[0]))
        score_elapsed_ms.append((time.perf_counter() - started) * 1000.0)
    resource = _d34_resource(fit, len(all_classes))
    resource.update(
        {
            "deployment_k_shot": 10,
            "registered_class_count": len(all_classes),
            "old_support_before_registration": before_metric,
            "old_support_after_registration": after_metric,
            "old_support_classwise_non_degradation_pass": classwise_pass,
            "old_support_floor_non_degradation_pass": floor_pass,
            "old_support_non_degradation_pass": bool(classwise_pass and floor_pass),
            "old_score_prefix_bitwise_unchanged": True,
            "support_adaptation_and_registration_elapsed_ms": fit_elapsed_ms,
            "batch1_head_latency_mean_ms": float(np.mean(score_elapsed_ms)),
            "batch1_head_latency_p95_ms": float(
                np.quantile(np.asarray(score_elapsed_ms, dtype=np.float64), 0.95)
            ),
            "batch1_head_latency_sample_count": len(score_elapsed_ms),
            "head_latency_scope": "FAST_prefix_plus_D34_collision_score_plus_argmax",
            "latency_includes_argmax": True,
            "head_peak_cuda_memory_bytes": 0,
            "head_runtime": "numpy_cpu_fp32_int8",
        }
    )
    geometry = {
        "schema": "cvs.phase2.d34_collision_local_geometry.v1",
        "feature_geometry": "b3_auxiliary_dominant_z160_fft96_rf32_v1",
        "observed_feature_block_energy": _d30_observed_block_energy(features),
        "arm": str(config.registration.arm),
        "old_score_prefix_bitwise_unchanged": True,
        **dict(fit["geometry"]),
    }
    return resource, geometry


def _full_d35_state_audit(
    rows: Mapping[str, np.ndarray],
    z_id160: np.ndarray,
    fft96: np.ndarray,
    rf32: np.ndarray,
    *,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    config: D35CandidateConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Full-K10 D35 fit, frozen-prefix proof, latency, and geometry audit."""

    labels = np.asarray(rows["labels"]).astype(str)
    old = np.isin(labels, np.asarray(old_classes))
    new = np.isin(labels, np.asarray(new_classes))
    features = _d1_feature_from_blocks(z_id160, fft96, rf32)
    fit_started = time.perf_counter()
    fit = _fit_d35_route(
        features, labels, old, new, old_classes, new_classes, config
    )
    fit_elapsed_ms = (time.perf_counter() - fit_started) * 1000.0
    all_classes = old_classes + new_classes
    old_prefix, scores = _score_d35(fit, features)
    if not np.array_equal(old_prefix, scores[:, : len(old_classes)]):
        raise D25RunnerError("D35 full-K10 changed old score prefix")
    predictions = np.asarray(all_classes)[np.argmax(scores, axis=1)]
    before_predictions = np.asarray(old_classes)[np.argmax(old_prefix[old], axis=1)]
    before_metric = legacy._metric_block(
        labels[old], before_predictions.astype(str).tolist(), old_classes
    )
    after_metric = legacy._metric_block(
        labels[old], predictions[old].astype(str).tolist(), old_classes
    )
    new_metric = legacy._metric_block(
        labels[new], predictions[new].astype(str).tolist(), new_classes
    )
    classwise_pass = all(
        float(after_metric["per_class_accuracy"][name]) + 1.0e-12
        >= float(before_metric["per_class_accuracy"][name])
        for name in old_classes
    )
    floor_pass = (
        float(after_metric["class_floor_accuracy"]) + 1.0e-12
        >= float(before_metric["class_floor_accuracy"])
    )
    score_elapsed_ms: list[float] = []
    for feature in features:
        started = time.perf_counter()
        adapted, prefix = _d34_fast_unit_and_prefix(
            fit["old_state"], feature[None, :]
        )
        row_scores = score_d35_dense_safe_registration(
            fit["after"], adapted, prefix
        )
        _ = int(np.argmax(row_scores[0]))
        score_elapsed_ms.append((time.perf_counter() - started) * 1000.0)
    resource = _d35_resource(fit, len(all_classes))
    resource.update(
        {
            "deployment_k_shot": 10,
            "registered_class_count": len(all_classes),
            "old_support_before_registration": before_metric,
            "old_support_after_registration": after_metric,
            "new_support_after_registration": new_metric,
            "old_support_classwise_non_degradation_pass": classwise_pass,
            "old_support_floor_non_degradation_pass": floor_pass,
            "old_support_non_degradation_pass": bool(classwise_pass and floor_pass),
            "old_score_prefix_bitwise_unchanged": True,
            "support_adaptation_and_registration_elapsed_ms": fit_elapsed_ms,
            "batch1_head_latency_mean_ms": float(np.mean(score_elapsed_ms)),
            "batch1_head_latency_p95_ms": float(
                np.quantile(np.asarray(score_elapsed_ms, dtype=np.float64), 0.95)
            ),
            "batch1_head_latency_sample_count": len(score_elapsed_ms),
            "head_latency_scope": "FAST_prefix_plus_D35_dense_safe_score_plus_argmax",
            "latency_includes_argmax": True,
            "head_peak_cuda_memory_bytes": 0,
            "head_runtime": "numpy_cpu_fp32_int8",
        }
    )
    geometry = {
        "schema": "cvs.phase2.d35_dense_safe_geometry.v1",
        "feature_geometry": "b3_auxiliary_dominant_z160_fft96_rf32_v1",
        "observed_feature_block_energy": _d30_observed_block_energy(features),
        "arm": str(config.registration.arm),
        "old_score_prefix_bitwise_unchanged": True,
        **dict(fit["geometry"]),
        "new_class_reachability": dict(resource["new_class_reachability"]),
        "unreachable_new_class_count": int(
            resource["unreachable_new_class_count"]
        ),
    }
    return resource, geometry


def _full_d38_state_audit(
    rows: Mapping[str, np.ndarray],
    z_id160: np.ndarray,
    fft96: np.ndarray,
    rf32: np.ndarray,
    *,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    config: D38CandidateConfig,
    seed: int,
    device: torch.device | str = "cpu",
    scenario: str = "unit_test_scene",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Full-K10 refit/audit without changing the outer-selected D38 arm."""

    labels = np.asarray(rows["labels"]).astype(str)
    ranks = np.asarray(rows["ranks"], dtype=np.int64)
    tokens = np.asarray(rows["tokens"]).astype(str)
    old = np.isin(labels, np.asarray(old_classes))
    new = np.isin(labels, np.asarray(new_classes))
    if int(np.sum(old)) != 10 * len(old_classes) or int(np.sum(new)) != 10 * len(
        new_classes
    ):
        raise D25RunnerError("D38 full-K10 class symmetry drift")
    features = _d1_feature_from_blocks(z_id160, fft96, rf32)
    started = time.perf_counter()
    result = fit_d38_strong_b3_quantized(
        features[old],
        labels[old],
        old_classes,
        features[new],
        labels[new],
        new_classes,
        seed=int(seed),
        device=device,
        config=config.core,
    )
    fit_elapsed_ms = (time.perf_counter() - started) * 1000.0
    deployed_state = (
        result.state if config.deploy_precision == "int8" else result.matched_fp32_state
    )
    all_classes = old_classes + new_classes
    before_scores = score_d38_strong_b3(result.before_state, features[old])
    after_scores = score_d38_strong_b3(deployed_state, features)
    int8_scores = score_d38_strong_b3(result.state, features)
    fp32_scores = score_d38_strong_b3(result.matched_fp32_state, features)
    before_predictions = np.asarray(old_classes)[np.argmax(before_scores, axis=1)]
    after_predictions = np.asarray(all_classes)[np.argmax(after_scores, axis=1)]
    before_old = legacy._metric_block(
        labels[old], before_predictions.astype(str).tolist(), old_classes
    )
    after_old = legacy._metric_block(
        labels[old], after_predictions[old].astype(str).tolist(), old_classes
    )
    after_new = legacy._metric_block(
        labels[new], after_predictions[new].astype(str).tolist(), new_classes
    )
    pairwise = pairwise_support_diagnostics_d38(
        deployed_state,
        features[new],
        labels[new],
        tokens[new],
        scenario=scenario,
        outer_fold=-1,
        physical_ranks=ranks[new],
    )
    margins = np.asarray([float(row["new_new_margin"]) for row in pairwise])
    intrusion = _d37_old_to_new_intrusion_count(after_predictions, old, new_classes)
    prefix_unchanged = old_prefix_bitwise_unchanged_d38(
        result.before_state, result.state
    )
    argmax_changes = int(
        np.sum(np.argmax(int8_scores, axis=1) != np.argmax(fp32_scores, axis=1))
    )
    latency_ms: list[float] = []
    for feature in features:
        score_started = time.perf_counter()
        row_scores = score_d38_strong_b3(deployed_state, feature[None, :])
        _ = int(np.argmax(row_scores[0]))
        latency_ms.append((time.perf_counter() - score_started) * 1000.0)
    resource = dict(result.resource_audit)
    deployed_bytes = int(deployed_state.persistent_state_bytes)
    classwise_old_pass = all(
        float(after_old["per_class_accuracy"][name]) + 1.0e-12
        >= float(before_old["per_class_accuracy"][name])
        for name in old_classes
    )
    resource.update(
        {
            "deployment_precision": config.deploy_precision,
            "deployment_k_shot": 10,
            "peak_trainable_parameters": int(resource["trainable_parameters"]),
            "total_optimizer_steps": int(resource["optimizer_steps"]),
            "persistent_state_bytes": deployed_bytes,
            "persistent_state_cap_pass": deployed_bytes <= 256 * 1024,
            "target_old_int8_prototypes_used_for_prediction": (
                config.deploy_precision == "int8"
            ),
            "target_new_int8_prototypes_used_for_prediction": (
                config.deploy_precision == "int8"
            ),
            "resident_fp32_target_prototype_count": (
                0 if config.deploy_precision == "int8" else len(all_classes)
            ),
            "old_prefix_bitwise_unchanged": prefix_unchanged,
            "old_support_before_registration": before_old,
            "old_support_after_registration": after_old,
            "new_support_after_registration": after_new,
            "old_support_classwise_non_degradation_pass": classwise_old_pass,
            "old_support_non_degradation_pass": bool(
                classwise_old_pass and intrusion == 0
            ),
            "full_support_old_to_new_intrusion_count": intrusion,
            "matched_fp32_full_k10_argmax_change_count": argmax_changes,
            "pairwise_support_diagnostic_row_count": len(pairwise),
            "new_new_confusion_count": int(np.sum(margins <= 0.0)),
            "new_new_margin_min": float(np.min(margins)),
            "support_adaptation_and_registration_elapsed_ms": fit_elapsed_ms,
            "batch1_head_latency_mean_ms": float(np.mean(latency_ms)),
            "batch1_head_latency_p95_ms": float(np.quantile(latency_ms, 0.95)),
            "batch1_head_latency_sample_count": len(latency_ms),
            "latency_includes_argmax": True,
            "full_k10_refit_only_no_candidate_change": True,
            "complete_loss_trace": [dict(row) for row in result.training_trace],
        }
    )
    geometry = {
        **dict(result.geometry_audit),
        "schema": "cvs.phase2.d38.full_k10_geometry.v1",
        "observed_feature_block_energy": _d30_observed_block_energy(features),
        "full_k10_refit_only_no_candidate_change": True,
        "pairwise_support_diagnostics": pairwise,
        "new_new_confusion_count": int(np.sum(margins <= 0.0)),
        "new_new_margin_min": float(np.min(margins)),
        "new_new_margin_mean": float(np.mean(margins)),
        "old_prefix_bitwise_unchanged": prefix_unchanged,
        "matched_fp32_full_k10_argmax_change_count": argmax_changes,
        "query_rows_used": 0,
    }
    return resource, geometry


def _full_d39_state_audit(
    rows: Mapping[str, np.ndarray],
    z_id160: np.ndarray,
    fft96: np.ndarray,
    rf32: np.ndarray,
    *,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    config: D39CandidateConfig,
    seed: int,
    device: torch.device | str = "cpu",
    scenario: str = "unit_test_scene",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Full-K10 D39 fit, latency, precision, radius-source and prefix audit."""

    labels = np.asarray(rows["labels"]).astype(str)
    tokens = np.asarray(rows["tokens"]).astype(str)
    ranks = np.asarray(rows["ranks"], dtype=np.int64)
    old = np.isin(labels, np.asarray(old_classes))
    new = np.isin(labels, np.asarray(new_classes))
    if int(np.sum(old)) != 10 * len(old_classes) or int(np.sum(new)) != 10 * len(
        new_classes
    ):
        raise D25RunnerError("D39 full-K10 class symmetry drift")
    features = _d1_feature_from_blocks(z_id160, fft96, rf32)
    started = time.perf_counter()
    result = fit_d39_angular_radius(
        features[old],
        labels[old],
        old_classes,
        features[new],
        labels[new],
        new_classes,
        seed=int(seed),
        device=device,
        config=config.core,
    )
    fit_elapsed_ms = (time.perf_counter() - started) * 1000.0
    deployed_state = (
        result.state if config.deploy_precision == "int8" else result.matched_fp32_state
    )
    all_classes = old_classes + new_classes
    before_scores = score_d39_angular_radius(result.before_state, features[old])
    after_scores = score_d39_angular_radius(deployed_state, features)
    int8_scores = score_d39_angular_radius(result.state, features)
    fp32_scores = score_d39_angular_radius(result.matched_fp32_state, features)
    before_predictions = np.asarray(old_classes)[np.argmax(before_scores, axis=1)]
    after_predictions = np.asarray(all_classes)[np.argmax(after_scores, axis=1)]
    before_old = legacy._metric_block(
        labels[old], before_predictions.astype(str).tolist(), old_classes
    )
    after_old = legacy._metric_block(
        labels[old], after_predictions[old].astype(str).tolist(), old_classes
    )
    after_new = legacy._metric_block(
        labels[new], after_predictions[new].astype(str).tolist(), new_classes
    )
    pairwise = pairwise_support_diagnostics_d39(
        deployed_state,
        features[new],
        labels[new],
        tokens[new],
        scenario=scenario,
        outer_fold=-1,
        physical_ranks=ranks[new],
    )
    margins = np.asarray([float(row["new_new_margin"]) for row in pairwise])
    intrusion = _d37_old_to_new_intrusion_count(after_predictions, old, new_classes)
    prefix_unchanged = old_prefix_bitwise_unchanged_d39(
        result.before_state, result.state
    )
    old_prototype_prefix_unchanged = old_prefix_bitwise_unchanged_d38(
        result.before_state.base_state, result.state.base_state
    )
    old_radius_prefix_unchanged = bool(
        np.array_equal(
            result.before_state.radius_fp16,
            result.state.radius_fp16[: len(old_classes)],
        )
    )
    r0_unchanged = bool(
        np.array_equal(result.before_state.r0_fp16, result.state.r0_fp16)
    )
    radius_positive_finite = bool(
        np.isfinite(result.state.radius_fp16).all()
        and np.all(result.state.radius_fp16 > 0)
        and np.isfinite(result.state.r0_fp16).all()
        and np.all(result.state.r0_fp16 > 0)
    )
    radius_shared = bool(
        np.array_equal(
            result.state.radius_fp16, result.matched_fp32_state.radius_fp16
        )
        and np.array_equal(result.state.r0_fp16, result.matched_fp32_state.r0_fp16)
    )
    argmax_changes = int(
        np.sum(np.argmax(int8_scores, axis=1) != np.argmax(fp32_scores, axis=1))
    )
    latency_ms: list[float] = []
    for feature in features:
        score_started = time.perf_counter()
        row_scores = score_d39_angular_radius(deployed_state, feature[None, :])
        _ = int(np.argmax(row_scores[0]))
        latency_ms.append((time.perf_counter() - score_started) * 1000.0)
    old_source_tokens = sorted(tokens[old].tolist())
    new_source_tokens = sorted(tokens[new].tolist())
    radius_source_audit = {
        "old_source_physical_token_sha256": hashlib.sha256(
            _canonical_bytes(old_source_tokens)
        ).hexdigest(),
        "new_source_physical_token_sha256": hashlib.sha256(
            _canonical_bytes(new_source_tokens)
        ).hexdigest(),
        "old_source_row_count": len(old_source_tokens),
        "new_source_row_count": len(new_source_tokens),
        "old_source_new_class_row_count": int(np.sum(old & new)),
        "new_source_old_class_row_count": int(np.sum(new & old)),
        "held_radius_fit_row_count": 0,
        "query_rows_used": 0,
    }
    resource = dict(result.resource_audit)
    deployed_bytes = int(deployed_state.persistent_state_bytes)
    classwise_old_pass = all(
        float(after_old["per_class_accuracy"][name]) + 1.0e-12
        >= float(before_old["per_class_accuracy"][name])
        for name in old_classes
    )
    resource.update(
        {
            "deployment_precision": config.deploy_precision,
            "deployment_k_shot": 10,
            "peak_trainable_parameters": int(resource["trainable_parameters"]),
            "total_optimizer_steps": int(resource["optimizer_steps"]),
            "persistent_state_bytes": deployed_bytes,
            "persistent_state_cap_pass": deployed_bytes <= 256 * 1024,
            "target_old_int8_prototypes_used_for_prediction": (
                config.deploy_precision == "int8"
            ),
            "target_new_int8_prototypes_used_for_prediction": (
                config.deploy_precision == "int8"
            ),
            "resident_fp32_target_prototype_count": (
                0 if config.deploy_precision == "int8" else len(all_classes)
            ),
            "clean_sample_access": False,
            "source_sample_access": False,
            "old_prefix_bitwise_unchanged": prefix_unchanged,
            "old_prototype_prefix_bitwise_unchanged": old_prototype_prefix_unchanged,
            "old_radius_prefix_bitwise_unchanged": old_radius_prefix_unchanged,
            "r0_bitwise_unchanged": r0_unchanged,
            "radius_positive_finite": radius_positive_finite,
            "radius_fp16_shared_between_int8_fp32": radius_shared,
            "radius_source_audit": radius_source_audit,
            "old_support_before_registration": before_old,
            "old_support_after_registration": after_old,
            "new_support_after_registration": after_new,
            "old_support_classwise_non_degradation_pass": classwise_old_pass,
            "old_support_non_degradation_pass": bool(
                classwise_old_pass and intrusion == 0
            ),
            "full_support_old_to_new_intrusion_count": intrusion,
            "matched_fp32_full_k10_argmax_change_count": argmax_changes,
            "pairwise_support_diagnostic_row_count": len(pairwise),
            "new_new_confusion_count": int(np.sum(margins <= 0.0)),
            "new_new_margin_min": float(np.min(margins)),
            "support_adaptation_and_registration_elapsed_ms": fit_elapsed_ms,
            "batch1_head_latency_mean_ms": float(np.mean(latency_ms)),
            "batch1_head_latency_p95_ms": float(np.quantile(latency_ms, 0.95)),
            "batch1_head_latency_sample_count": len(latency_ms),
            "latency_includes_argmax": True,
            "full_k10_refit_only_no_candidate_change": True,
            "complete_loss_trace": [dict(row) for row in result.training_trace],
        }
    )
    geometry = {
        **dict(result.geometry_audit),
        "schema": "cvs.phase2.d39.full_k10_geometry.v1",
        "observed_feature_block_energy": _d30_observed_block_energy(features),
        "full_k10_refit_only_no_candidate_change": True,
        "pairwise_support_diagnostics": pairwise,
        "new_new_confusion_count": int(np.sum(margins <= 0.0)),
        "new_new_margin_min": float(np.min(margins)),
        "new_new_margin_mean": float(np.mean(margins)),
        "old_prefix_bitwise_unchanged": prefix_unchanged,
        "old_prototype_prefix_bitwise_unchanged": old_prototype_prefix_unchanged,
        "old_radius_prefix_bitwise_unchanged": old_radius_prefix_unchanged,
        "r0_bitwise_unchanged": r0_unchanged,
        "radius_positive_finite": radius_positive_finite,
        "radius_fp16_shared_between_int8_fp32": radius_shared,
        "radius_source_audit": radius_source_audit,
        "matched_fp32_full_k10_argmax_change_count": argmax_changes,
        "query_rows_used": 0,
    }
    return resource, geometry


def _full_d40_state_audit(
    rows: Mapping[str, np.ndarray],
    z_id160: np.ndarray,
    fft96: np.ndarray,
    rf32: np.ndarray,
    *,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    config: D40CandidateConfig,
    seed: int,
    device: torch.device | str = "cpu",
    scenario: str = "unit_test_scene",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Selected-only full-K10 D40 fit, precision, geometry and resource audit."""

    labels = np.asarray(rows["labels"]).astype(str)
    tokens = np.asarray(rows["tokens"]).astype(str)
    ranks = np.asarray(rows["ranks"], dtype=np.int64)
    old = np.isin(labels, np.asarray(old_classes))
    new = np.isin(labels, np.asarray(new_classes))
    if int(np.sum(old)) != 10 * len(old_classes) or int(np.sum(new)) != 10 * len(
        new_classes
    ):
        raise D25RunnerError("D40 full-K10 class symmetry drift")
    features = _d1_feature_from_blocks(z_id160, fft96, rf32)
    started = time.perf_counter()
    result = fit_d40_hnbr(
        features[old],
        labels[old],
        old_classes,
        features[new],
        labels[new],
        new_classes,
        seed=int(seed),
        device=device,
        config=config.core,
    )
    fit_elapsed_ms = (time.perf_counter() - started) * 1000.0
    deployed_before = (
        result.before_state
        if config.deploy_precision == "int8"
        else result.matched_fp32_before_state
    )
    deployed_state = (
        result.state if config.deploy_precision == "int8" else result.matched_fp32_state
    )
    all_classes = old_classes + new_classes
    before_scores = score_d40_hnbr(deployed_before, features[old])
    after_scores = score_d40_hnbr(deployed_state, features)
    int8_scores = score_d40_hnbr(result.state, features)
    fp32_scores = score_d40_hnbr(result.matched_fp32_state, features)
    before_predictions = np.asarray(old_classes)[np.argmax(before_scores, axis=1)]
    after_predictions = np.asarray(all_classes)[np.argmax(after_scores, axis=1)]
    before_old = legacy._metric_block(
        labels[old], before_predictions.astype(str).tolist(), old_classes
    )
    after_old = legacy._metric_block(
        labels[old], after_predictions[old].astype(str).tolist(), old_classes
    )
    after_new = legacy._metric_block(
        labels[new], after_predictions[new].astype(str).tolist(), new_classes
    )
    pairwise = pairwise_support_diagnostics_d40(
        deployed_state,
        features[new],
        labels[new],
        tokens[new],
        scenario=scenario,
        outer_fold=-1,
        physical_ranks=ranks[new],
    )
    new_new_margins = np.asarray(
        [float(row["new_new_margin"]) for row in pairwise], dtype=np.float64
    )
    new_old_margins = np.asarray(
        [float(row["new_old_margin"]) for row in pairwise], dtype=np.float64
    )
    intrusion = _d37_old_to_new_intrusion_count(after_predictions, old, new_classes)
    prefix_unchanged = old_prefix_bitwise_unchanged_d40(
        result.before_state, result.state
    )
    old_base_prefix_unchanged = old_prefix_bitwise_unchanged_d38(
        result.before_state.base_state, result.state.base_state
    )
    argmax_changes = int(
        np.sum(np.argmax(int8_scores, axis=1) != np.argmax(fp32_scores, axis=1))
    )
    latency_ms: list[float] = []
    for feature in features:
        score_started = time.perf_counter()
        row_scores = score_d40_hnbr(deployed_state, feature[None, :])
        _ = int(np.argmax(row_scores[0]))
        latency_ms.append((time.perf_counter() - score_started) * 1000.0)
    direction_source_audit = {
        "old_source_physical_token_sha256": hashlib.sha256(
            _canonical_bytes(sorted(tokens[old].tolist()))
        ).hexdigest(),
        "new_source_physical_token_sha256": hashlib.sha256(
            _canonical_bytes(sorted(tokens[new].tolist()))
        ).hexdigest(),
        "old_source_row_count": int(np.sum(old)),
        "new_source_row_count": int(np.sum(new)),
        "old_source_new_class_row_count": int(np.sum(old & new)),
        "new_source_old_class_row_count": int(np.sum(new & old)),
        "held_direction_fit_row_count": 0,
        "query_rows_used": 0,
    }
    resource = dict(result.resource_audit)
    deployed_bytes = int(deployed_state.persistent_state_bytes)
    classwise_old_pass = all(
        float(after_old["per_class_accuracy"][name]) + 1.0e-12
        >= float(before_old["per_class_accuracy"][name])
        for name in old_classes
    )
    resource.update(
        {
            "deployment_precision": config.deploy_precision,
            "deployment_k_shot": 10,
            "peak_trainable_parameters": int(resource["trainable_parameters"]),
            "total_optimizer_steps": int(resource["optimizer_steps"]),
            "persistent_state_bytes": deployed_bytes,
            "persistent_state_cap_pass": deployed_bytes <= 256 * 1024,
            "target_old_int8_prototypes_used_for_prediction": bool(
                config.deploy_precision == "int8"
            ),
            "target_new_int8_prototypes_used_for_prediction": bool(
                config.deploy_precision == "int8"
            ),
            "resident_fp32_target_prototype_count": (
                0 if config.deploy_precision == "int8" else len(all_classes)
            ),
            "clean_sample_access": False,
            "source_sample_access": False,
            "old_prefix_bitwise_unchanged": prefix_unchanged,
            "old_base_prefix_bitwise_unchanged": old_base_prefix_unchanged,
            "new_hnbr_old_negative_precision": result.geometry_audit[
                "new_hnbr_old_negative_precision"
            ],
            "new_hnbr_old_negative_matches_before_int8_decode": bool(
                result.geometry_audit[
                    "new_hnbr_old_negative_matches_before_int8_decode"
                ]
            ),
            "old_fp32_reference_used_as_new_hnbr_negative": bool(
                result.geometry_audit[
                    "old_fp32_reference_used_as_new_hnbr_negative"
                ]
            ),
            "hnbr_label_permutation_equivariant": bool(
                result.geometry_audit["label_permutation_equivariant"]
            ),
            "hnbr_class_id_specific_branch": bool(
                result.geometry_audit["class_id_specific_branch"]
            ),
            "direction_source_audit": direction_source_audit,
            "old_support_before_registration": before_old,
            "old_support_after_registration": after_old,
            "new_support_after_registration": after_new,
            "old_support_classwise_non_degradation_pass": classwise_old_pass,
            "old_support_non_degradation_pass": bool(
                classwise_old_pass and intrusion == 0
            ),
            "full_support_old_to_new_intrusion_count": intrusion,
            "matched_fp32_full_k10_argmax_change_count": argmax_changes,
            "pairwise_support_diagnostic_row_count": len(pairwise),
            "new_new_confusion_count": int(np.sum(new_new_margins <= 0.0)),
            "new_new_margin_min": float(np.min(new_new_margins)),
            "new_old_margin_min": float(np.min(new_old_margins)),
            "support_adaptation_and_registration_elapsed_ms": fit_elapsed_ms,
            "batch1_head_latency_mean_ms": float(np.mean(latency_ms)),
            "batch1_head_latency_p95_ms": float(np.quantile(latency_ms, 0.95)),
            "batch1_head_latency_sample_count": len(latency_ms),
            "latency_includes_argmax": True,
            "full_k10_refit_only_no_candidate_change": True,
            "complete_loss_trace": [dict(row) for row in result.training_trace],
        }
    )
    geometry = {
        **dict(result.geometry_audit),
        "schema": "cvs.phase2.d40.full_k10_geometry.v1",
        "observed_feature_block_energy": _d30_observed_block_energy(features),
        "full_k10_refit_only_no_candidate_change": True,
        "pairwise_support_diagnostics": pairwise,
        "new_new_confusion_count": int(np.sum(new_new_margins <= 0.0)),
        "new_new_margin_min": float(np.min(new_new_margins)),
        "new_new_margin_mean": float(np.mean(new_new_margins)),
        "new_old_margin_min": float(np.min(new_old_margins)),
        "old_prefix_bitwise_unchanged": prefix_unchanged,
        "old_base_prefix_bitwise_unchanged": old_base_prefix_unchanged,
        "direction_source_audit": direction_source_audit,
        "matched_fp32_full_k10_argmax_change_count": argmax_changes,
        "query_rows_used": 0,
    }
    return resource, geometry


def _full_d37_state_audit(
    rows: Mapping[str, np.ndarray],
    z_id160: np.ndarray,
    fft96: np.ndarray,
    rf32: np.ndarray,
    *,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    config: D37CandidateConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Full-K10 D37 fit with five physical rank-pair OOF folds."""

    labels = np.asarray(rows["labels"]).astype(str)
    ranks = np.asarray(rows["ranks"], dtype=np.int64)
    old = np.isin(labels, np.asarray(old_classes))
    new = np.isin(labels, np.asarray(new_classes))
    features = _d1_feature_from_blocks(z_id160, fft96, rf32)
    started = time.perf_counter()
    fit = _fit_d37_route(
        features,
        labels,
        ranks,
        np.asarray(rows["tokens"]),
        old,
        new,
        old_classes,
        new_classes,
        config,
    )
    fit_elapsed_ms = (time.perf_counter() - started) * 1000.0
    all_classes = old_classes + new_classes
    before_scores, after_scores = _score_d37(fit, features)
    before_predictions = np.asarray(old_classes)[np.argmax(before_scores[old], axis=1)]
    after_predictions = np.asarray(all_classes)[np.argmax(after_scores, axis=1)]
    before_metric = legacy._metric_block(
        labels[old], before_predictions.astype(str).tolist(), old_classes
    )
    after_metric = legacy._metric_block(
        labels[old], after_predictions[old].astype(str).tolist(), old_classes
    )
    new_metric = legacy._metric_block(
        labels[new], after_predictions[new].astype(str).tolist(), new_classes
    )
    b3_scores = score_b3_fisher_closed_form(fit["fisher_fit"].state, features[old])
    b3_predictions = np.asarray(old_classes)[np.argmax(b3_scores, axis=1)]
    b3_metric = legacy._metric_block(
        labels[old], b3_predictions.astype(str).tolist(), old_classes
    )
    full_intrusion = _d37_old_to_new_intrusion_count(
        after_predictions, old, new_classes
    )
    classwise_non_degradation = all(
        float(after_metric["per_class_accuracy"][name]) + 1.0e-12
        >= float(before_metric["per_class_accuracy"][name])
        for name in old_classes
    )
    quantized_b3_noninferior = all(
        float(before_metric["per_class_accuracy"][name]) + 1.0e-12
        >= float(b3_metric["per_class_accuracy"][name])
        for name in old_classes
    )
    new_reachability_rows: list[dict[str, Any]] = []
    for index in np.flatnonzero(new).tolist():
        target = len(old_classes) + new_classes.index(str(labels[index]))
        scores = after_scores[index]
        margin = float(scores[target] - np.max(np.delete(scores, target)))
        new_reachability_rows.append(
            {
                "rank": int(ranks[index]),
                "new_class": str(labels[index]),
                "correct": bool(np.argmax(scores) == target),
                "margin": margin,
            }
        )
    new_reachability = {
        name: bool(values)
        and all(bool(row["correct"]) and float(row["margin"]) > 0.0 for row in values)
        for name in new_classes
        for values in [[row for row in new_reachability_rows if row["new_class"] == name]]
    }
    score_elapsed_ms: list[float] = []
    for feature in features:
        score_started = time.perf_counter()
        row_scores = (
            score_d37_b3_preserving_int8(fit["state"], feature[None, :])
            if bool(fit["oof_feasible_interval_pass"])
            else base_score_d37_b3_preserving_int8(
                fit["state_no_offset"], feature[None, :]
            )
        )
        _ = int(np.argmax(row_scores[0]))
        score_elapsed_ms.append((time.perf_counter() - score_started) * 1000.0)
    resource = _d37_resource(fit, len(all_classes))
    resource.update(
        {
            "deployment_k_shot": 10,
            "registered_class_count": len(all_classes),
            "old_support_before_registration": before_metric,
            "old_support_after_registration": after_metric,
            "new_support_after_registration": new_metric,
            "b3_reference_old_support": b3_metric,
            "quantized_old_head_classwise_noninferior_to_b3": quantized_b3_noninferior,
            "old_support_classwise_non_degradation_pass": classwise_non_degradation,
            "old_support_floor_non_degradation_pass": float(
                after_metric["class_floor_accuracy"]
            )
            + 1.0e-12
            >= float(before_metric["class_floor_accuracy"]),
            "old_support_non_degradation_pass": bool(
                classwise_non_degradation
                and full_intrusion == 0
                and fit["oof_feasible_interval_pass"]
            ),
            "old_score_prefix_bitwise_unchanged": bool(
                old_prefix_bitwise_unchanged_d37(
                    fit["before_state"], fit["state_no_offset"]
                )
            ),
            "old_score_column_max_abs_diff": float(
                np.max(np.abs(before_scores - after_scores[:, : len(old_classes)]))
            ),
            "full_support_old_to_new_intrusion_count": full_intrusion,
            "new_class_reachability": new_reachability,
            "unreachable_new_class_count": int(
                sum(not value for value in new_reachability.values())
            ),
            "full_k10_crossfit_fold_count": len(fit["inner_pairs"]),
            "full_k10_crossfit_no_self_participation": True,
            "support_adaptation_and_registration_elapsed_ms": fit_elapsed_ms,
            "batch1_head_latency_mean_ms": float(np.mean(score_elapsed_ms)),
            "batch1_head_latency_p95_ms": float(
                np.quantile(np.asarray(score_elapsed_ms, dtype=np.float64), 0.95)
            ),
            "batch1_head_latency_sample_count": len(score_elapsed_ms),
            "head_latency_scope": "D37_residual_int8_score_plus_argmax",
            "latency_includes_argmax": True,
            "head_peak_cuda_memory_bytes": 0,
            "head_runtime": "numpy_cpu_fp32_int8",
        }
    )
    geometry = {
        **dict(fit["core_result"].geometry_audit),
        "schema": "cvs.phase2.d37_b3_preserving_int8_geometry.v1",
        "feature_geometry": "b3_auxiliary_dominant_z160_fft96_rf32_v1",
        "observed_feature_block_energy": _d30_observed_block_energy(features),
        "full_k10_crossfit_rank_pairs": [list(pair) for pair in fit["inner_pairs"]],
        "full_k10_crossfit_no_self_participation": True,
        "oof_feasible_interval_pass": bool(fit["oof_feasible_interval_pass"]),
        "oof_feasible_interval_lower_bound": fit["oof_feasible_interval_lower_bound"],
        "oof_feasible_interval_upper_bound": fit["oof_feasible_interval_upper_bound"],
        "oof_failure_reason": fit["oof_failure_reason"],
        "full_support_old_to_new_intrusion_count": full_intrusion,
        "new_support_reachability_rows": new_reachability_rows,
        "new_class_reachability": new_reachability,
    }
    return resource, geometry


def _full_d36_state_audit(
    rows: Mapping[str, np.ndarray],
    z_id160: np.ndarray,
    fft96: np.ndarray,
    rf32: np.ndarray,
    *,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    config: D36CandidateConfig,
    ground_anchor: np.ndarray | None,
    ground_medoid_index: int | None,
    ground_anchor_sha256: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Full-K10 five-fold OOF calibration, state, latency, and resource audit."""

    labels = np.asarray(rows["labels"]).astype(str)
    ranks = np.asarray(rows["ranks"], dtype=np.int64)
    old = np.isin(labels, np.asarray(old_classes))
    new = np.isin(labels, np.asarray(new_classes))
    features = _d1_feature_from_blocks(z_id160, fft96, rf32)
    started = time.perf_counter()
    fit = _fit_d36_route(
        features,
        labels,
        ranks,
        old,
        new,
        old_classes,
        new_classes,
        config,
        ground_anchor,
    )
    fit_elapsed_ms = (time.perf_counter() - started) * 1000.0
    all_classes = old_classes + new_classes
    before_scores, after_scores = _score_d36(fit, features)
    before_predictions = np.asarray(old_classes)[np.argmax(before_scores[old], axis=1)]
    after_predictions = np.asarray(all_classes)[np.argmax(after_scores, axis=1)]
    before_metric = legacy._metric_block(
        labels[old], before_predictions.astype(str).tolist(), old_classes
    )
    after_metric = legacy._metric_block(
        labels[old], after_predictions[old].astype(str).tolist(), old_classes
    )
    new_metric = legacy._metric_block(
        labels[new], after_predictions[new].astype(str).tolist(), new_classes
    )
    b3_scores = score_b3_fisher_closed_form(fit["fisher_fit"].state, features[old])
    b3_predictions = np.asarray(old_classes)[np.argmax(b3_scores, axis=1)]
    b3_metric = legacy._metric_block(
        labels[old], b3_predictions.astype(str).tolist(), old_classes
    )
    before_correct = before_predictions.astype(str) == labels[old]
    old_after_predictions = after_predictions[old].astype(str)
    full_intrusion = int(
        np.sum(before_correct & np.isin(old_after_predictions, np.asarray(new_classes)))
    )
    classwise_non_degradation = all(
        float(after_metric["per_class_accuracy"][name]) + 1.0e-12
        >= float(before_metric["per_class_accuracy"][name])
        for name in old_classes
    )
    quantized_b3_noninferior = all(
        float(before_metric["per_class_accuracy"][name]) + 1.0e-12
        >= float(b3_metric["per_class_accuracy"][name])
        for name in old_classes
    )
    score_elapsed_ms: list[float] = []
    for feature in features:
        score_started = time.perf_counter()
        row_scores = score_d36_compiled_joint_int8(
            fit["state"], feature[None, :]
        )
        _ = int(np.argmax(row_scores[0]))
        score_elapsed_ms.append((time.perf_counter() - score_started) * 1000.0)
    resource = _d36_resource(fit, len(all_classes))
    resource.update(
        {
            "deployment_k_shot": 10,
            "registered_class_count": len(all_classes),
            "old_support_before_registration": before_metric,
            "old_support_after_registration": after_metric,
            "new_support_after_registration": new_metric,
            "b3_reference_old_support": b3_metric,
            "quantized_old_head_classwise_noninferior_to_b3": quantized_b3_noninferior,
            "old_support_classwise_non_degradation_pass": classwise_non_degradation,
            "old_support_floor_non_degradation_pass": float(after_metric["class_floor_accuracy"]) + 1.0e-12 >= float(before_metric["class_floor_accuracy"]),
            "old_support_non_degradation_pass": bool(classwise_non_degradation and full_intrusion == 0),
            "full_support_old_to_new_intrusion_count": full_intrusion,
            "full_k10_crossfit_fold_count": len(fit["inner_pairs"]),
            "full_k10_crossfit_no_self_participation": True,
            "support_adaptation_and_registration_elapsed_ms": fit_elapsed_ms,
            "batch1_head_latency_mean_ms": float(np.mean(score_elapsed_ms)),
            "batch1_head_latency_p95_ms": float(np.quantile(np.asarray(score_elapsed_ms, dtype=np.float64), 0.95)),
            "batch1_head_latency_sample_count": len(score_elapsed_ms),
            "head_latency_scope": "D36_compiled_int8_score_plus_argmax",
            "latency_includes_argmax": True,
            "head_peak_cuda_memory_bytes": 0,
            "head_runtime": "numpy_cpu_fp32_int8",
            "ground_anchor_medoid_index": ground_medoid_index if config.compiled.arm in ("B", "C") else None,
            "ground_anchor_sha256": ground_anchor_sha256 if config.compiled.arm in ("B", "C") else None,
            "ground_anchor_read_only": bool(config.compiled.arm == "A" or (ground_anchor is not None and not ground_anchor.flags.writeable)),
        }
    )
    geometry = {
        **dict(fit["core_result"].geometry_audit),
        "schema": "cvs.phase2.d36_compiled_joint_int8_geometry.v1",
        "feature_geometry": "b3_auxiliary_dominant_z160_fft96_rf32_v1",
        "observed_feature_block_energy": _d30_observed_block_energy(features),
        "full_k10_crossfit_rank_pairs": [list(pair) for pair in fit["inner_pairs"]],
        "full_k10_crossfit_no_self_participation": True,
        "full_support_old_to_new_intrusion_count": full_intrusion,
        "compiled_int8_quantization_error_mean": float(fit["core_result"].geometry_audit["compiled_int8_quantization_error_mean"]),
        "compiled_int8_quantization_error_max": float(fit["core_result"].geometry_audit["compiled_int8_quantization_error_max"]),
        "ground_anchor_medoid_index": resource["ground_anchor_medoid_index"],
        "ground_anchor_sha256": resource["ground_anchor_sha256"],
        "ground_anchor_read_only": resource["ground_anchor_read_only"],
    }
    return resource, geometry


def _full_d32_state_audit(
    component: Any,
    rows: Mapping[str, np.ndarray],
    z_id160: np.ndarray,
    direct_logits: np.ndarray,
    fft96: np.ndarray,
    rf32: np.ndarray,
    *,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    config: D32CandidateConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Full K-shot D32 support audit with complete row-local latency."""

    labels = np.asarray(rows["labels"]).astype(str)
    old = np.isin(labels, np.asarray(old_classes))
    new = np.isin(labels, np.asarray(new_classes))
    features = _d1_feature_from_blocks(z_id160, fft96, rf32)
    fit_started = time.perf_counter()
    fit = _fit_d32_route(
        component,
        features,
        z_id160,
        direct_logits,
        labels,
        old,
        new,
        old_classes,
        new_classes,
        config,
    )
    fit_elapsed_ms = (time.perf_counter() - fit_started) * 1000.0
    before = fit["before"]
    after = fit["after"]
    all_classes = old_classes + new_classes
    before_predictions = predict_all_d26(before, features[old]).astype(str).tolist()
    final_predictions = np.asarray(all_classes)[np.argmax(fit["adjusted_scores"], axis=1)]
    before_metric = legacy._metric_block(labels[old], before_predictions, old_classes)
    after_metric = legacy._metric_block(
        labels[old], final_predictions[old].tolist(), old_classes
    )
    classwise_pass = all(
        float(after_metric["per_class_accuracy"][name]) + 1.0e-12
        >= float(before_metric["per_class_accuracy"][name])
        for name in old_classes
    )
    floor_pass = (
        float(after_metric["class_floor_accuracy"]) + 1.0e-12
        >= float(before_metric["class_floor_accuracy"])
    )
    score_elapsed_ms: list[float] = []
    for index, feature in enumerate(features):
        score_started = time.perf_counter()
        row_scores = score_all_d32(after, feature[None, :])
        if fit["dali_enabled"]:
            row_scores = _d30_rerank_matrix(
                fit["dali_state"],
                row_scores,
                z_id160[index : index + 1],
                direct_logits[index : index + 1],
            )
        _ = int(np.argmax(row_scores[0]))
        score_elapsed_ms.append((time.perf_counter() - score_started) * 1000.0)
    base_resource = dict(after.resource_audit())
    accounting = _d31_dali_state_accounting(fit["dali_state"])
    dali_resource = accounting["dali_resource"]
    combined_resident = int(base_resource["persistent_state_bytes"]) + int(
        accounting["actual_current_dali_state_bytes"]
    )
    projected_active = int(base_resource["persistent_state_bytes"]) + int(
        accounting["projected_slim_dali_runtime_bytes"]
    )
    registered_count = len(all_classes)
    identity_macs = registered_count * 10 * 160
    base_head_macs = int(base_resource["estimated_macs_per_query"])
    dali_macs = (
        int(dali_resource["fixed_medoid_ground_macs_per_query"])
        if fit["dali_enabled"]
        else 0
    )
    total_macs = base_head_macs + dali_macs
    argmax_ops = max(0, registered_count - 1)
    dali_scalar_ops = 12 * len(old_classes) if fit["dali_enabled"] else 0
    training_trace = list(fit["before_fit"].loss_trace) + list(
        fit["stage2c_fit"].loss_trace
    )
    resource = {
        **base_resource,
        "schema": "cvs.phase2.d32_combined_resource.v1",
        "d32_suffix_resource": base_resource,
        **accounting,
        "dali_enabled_by_old_support_gate": bool(fit["dali_enabled"]),
        "dali_old_support_gate": fit["dali_gate"],
        "actual_int8_component_used_for_prediction": bool(fit["dali_enabled"]),
        "authorized_full_bundle_state_bytes": int(
            accounting["authorized_full_bundle_state_bytes"]
        ),
        "full_bundle_resident_combined_state_bytes": combined_resident,
        "projected_slim_active_predictor_state_bytes": projected_active,
        "slim_runtime_projection_only": True,
        "current_formal_bundle_rebuilt_as_slim_medoid": False,
        "deployment_resource_primary_state_view": (
            "projected_slim_fixed_medoid_predictor_with_full_bundle_residency_disclosed"
        ),
        "deployable_predictor_state_bytes_projected_slim_medoid": projected_active,
        "persistent_state_bytes": combined_resident,
        "persistent_state_cap_pass": combined_resident <= 256 * 1024,
        "stage2b_adaptation_macs": int(
            base_resource["estimated_stage2b_adaptation_macs"]
        ),
        "stage2c_adaptation_macs": int(
            base_resource["estimated_stage2c_adaptation_macs"]
        ),
        "total_adaptation_macs": int(base_resource["estimated_adaptation_macs"]),
        "base_head_macs_per_query": base_head_macs,
        "d32_extra_scalar_bias_adds_per_query": int(
            base_resource["estimated_scalar_bias_adds_per_query"]
        ),
        "dali_medoid_macs_per_query": dali_macs,
        "argmax_scalar_comparisons_per_query": argmax_ops,
        "total_post_backbone_macs_per_query": total_macs,
        "estimated_macs_per_query": total_macs,
        "estimated_row_local_scalar_ops_per_query": int(
            base_resource["estimated_scalar_bias_adds_per_query"]
            + dali_scalar_ops
            + argmax_ops
        ),
        "identity_single_qknn_macs_same_registered_count": identity_macs,
        "estimated_score_mac_ratio_vs_identity_single_qknn": float(
            total_macs / identity_macs
        ),
        "total_optimizer_steps": int(after.stage2b_optimizer_steps)
        + int(after.stage2c_optimizer_steps),
        "total_adaptation_epochs": int(after.stage2b_optimizer_steps)
        + int(after.stage2c_optimizer_steps),
        "deployment_k_shot": 10,
        "registered_class_count": registered_count,
        "old_support_before_registration": before_metric,
        "old_support_after_registration": after_metric,
        "old_support_classwise_non_degradation_pass": classwise_pass,
        "old_support_floor_non_degradation_pass": floor_pass,
        "old_support_non_degradation_pass": bool(classwise_pass and floor_pass),
        "support_adaptation_and_registration_elapsed_ms": fit_elapsed_ms,
        "batch1_head_latency_mean_ms": float(np.mean(score_elapsed_ms)),
        "batch1_head_latency_p95_ms": float(
            np.quantile(np.asarray(score_elapsed_ms, dtype=np.float64), 0.95)
        ),
        "batch1_head_latency_sample_count": len(score_elapsed_ms),
        "head_latency_scope": "d32_score_plus_safe_cap_bias_plus_optional_dali_plus_argmax",
        "latency_includes_argmax": True,
        "head_peak_cuda_memory_bytes": 0,
        "head_runtime": "numpy_cpu_fp32",
        "complete_loss_trace": training_trace,
        "feature_geometry": "b3_auxiliary_dominant_z160_fft96_rf32_v1",
        "query_rows_used_for_fit": 0,
        "query_role_oracle_access": False,
        "query_true_batch_class_count_access": False,
        "query_class_quota_access": False,
        "query_batch_global_assignment": False,
        "dense_query_graph_bytes": 0,
        "clean_sample_access": False,
        "source_sample_access": False,
    }
    geometry = {
        "schema": "cvs.phase2.d32_inloop_safe_cap_geometry.v1",
        "feature_geometry": "b3_auxiliary_dominant_z160_fft96_rf32_v1",
        "observed_feature_block_energy": _d30_observed_block_energy(features),
        "old_prefix_sha256": after.old_prefix_sha256,
        "dali_enabled": bool(fit["dali_enabled"]),
        "dali_old_support_gate": fit["dali_gate"],
        "raw_confusion": _d31_confusion_audit(
            fit["raw_scores"], labels, old_classes, new_classes
        ),
        "final_confusion": _d31_confusion_audit(
            fit["adjusted_scores"], labels, old_classes, new_classes
        ),
        "support_gate": json.loads(after.support_gate_json),
        "base_old_parameter_prefix_bitwise_unchanged": True,
        "final_old_score_columns_bitwise_unchanged": bool(
            np.array_equal(
                fit["raw_scores"][:, : len(old_classes)],
                fit["adjusted_scores"][:, : len(old_classes)],
            )
        ),
        "dali_max_old_preserved": True,
    }
    return resource, geometry


def run(
    *,
    before_root: Path,
    before_seal: Path,
    expected_before_seal_sha256: str,
    before_formal_policy: Path,
    before_formal_policy_authorization: Path,
    before_signed_policy_authorization_envelope: Path,
    expected_before_signed_policy_authorization_envelope_sha256: str,
    after_root: Path,
    after_seal: Path,
    expected_after_seal_sha256: str,
    after_formal_policy: Path,
    after_formal_policy_authorization: Path,
    after_signed_policy_authorization_envelope: Path,
    expected_after_signed_policy_authorization_envelope_sha256: str,
    component_dir: Path,
    expected_component_manifest_sha256: str,
    class_binding_path: Path,
    expected_class_binding_sha256: str,
    output: Path,
    device_name: str = "auto",
    mode: str = MODE,
    candidate_set: str = CANDIDATE_SET_D25_V4,
) -> dict[str, Any]:
    if mode != MODE:
        raise D25RunnerError("D25 runner is development support-only")
    if output.exists():
        raise D25RunnerError("output path already exists")
    candidates = preregistered_candidates(candidate_set)
    candidate_lock = _candidate_lock(candidates, candidate_set)

    before_preopen_manifest = legacy._preopen_manifest(
        before_root,
        before_seal,
        expected_seal_sha256=expected_before_seal_sha256,
    )
    after_preopen_manifest = legacy._preopen_manifest(
        after_root,
        after_seal,
        expected_seal_sha256=expected_after_seal_sha256,
    )
    legacy._manifest_binding(before_preopen_manifest, after_preopen_manifest)
    if candidate_set == CANDIDATE_SET_D38_V1:
        _require_d38_development_cell(
            before_preopen_manifest, after_preopen_manifest
        )
    if candidate_set == CANDIDATE_SET_D39_V1:
        _require_d39_development_cell(
            before_preopen_manifest, after_preopen_manifest
        )
    if candidate_set == CANDIDATE_SET_D40_V1:
        _require_d40_development_cell(
            before_preopen_manifest, after_preopen_manifest
        )
    preopen_old_classes = legacy._registered_handles(before_preopen_manifest)
    component, component_audit = legacy._load_component(
        component_dir,
        expected_manifest_sha256=expected_component_manifest_sha256,
        expected_checkpoint_sha256=str(
            before_preopen_manifest["phase1_checkpoint_sha256"]
        ),
        bound_old_handles=preopen_old_classes,
        class_binding_path=class_binding_path,
        expected_class_binding_sha256=expected_class_binding_sha256,
    )
    d36_ground_anchor: np.ndarray | None = None
    d36_ground_medoid_index: int | None = None
    d36_ground_anchor_sha256: str | None = None
    if candidate_set == CANDIDATE_SET_D36_V1:
        (
            d36_ground_anchor,
            d36_ground_medoid_index,
            d36_ground_anchor_sha256,
        ) = _d36_fixed_ground_anchor(component)
    device = torch.device(
        "cuda:0"
        if device_name == "auto" and torch.cuda.is_available()
        else ("cpu" if device_name == "auto" else device_name)
    )
    model = legacy.load_torchscript_backbone_same_fd(
        before_root,
        legacy._member(before_preopen_manifest, "feature_runtime"),
        device=device,
    )
    runtime_direct_logit_binding_audit = legacy._verify_runtime_direct_logit_binding(
        model,
        before_preopen_manifest,
        component_audit["column_binding"],
    )

    before_evidence = legacy.materialize_somph_enrollment_with_signed_authority(
        before_root,
        detached_seal_path=before_seal,
        expected_seal_sha256=expected_before_seal_sha256,
        formal_policy_path=before_formal_policy,
        formal_policy_authorization_path=before_formal_policy_authorization,
        signed_policy_authorization_envelope_path=(
            before_signed_policy_authorization_envelope
        ),
        expected_signed_policy_authorization_envelope_sha256=(
            expected_before_signed_policy_authorization_envelope_sha256
        ),
    )
    after_evidence = legacy.materialize_somph_enrollment_with_signed_authority(
        after_root,
        detached_seal_path=after_seal,
        expected_seal_sha256=expected_after_seal_sha256,
        formal_policy_path=after_formal_policy,
        formal_policy_authorization_path=after_formal_policy_authorization,
        signed_policy_authorization_envelope_path=(
            after_signed_policy_authorization_envelope
        ),
        expected_signed_policy_authorization_envelope_sha256=(
            expected_after_signed_policy_authorization_envelope_sha256
        ),
    )
    before_authority = (
        legacy.finalize_somph_enrollment_authority_after_materialization(
            before_evidence
        )
    )
    after_authority = (
        legacy.finalize_somph_enrollment_authority_after_materialization(
            after_evidence
        )
    )
    legacy._require_post_materialization_authority(before_authority, after_authority)
    before_manifest = before_evidence.manifest
    after_manifest = after_evidence.manifest
    legacy._manifest_binding(before_manifest, after_manifest)
    old_classes = legacy._registered_handles(before_manifest)
    all_classes = legacy._registered_handles(after_manifest)
    if all_classes[: len(old_classes)] != old_classes:
        raise D25RunnerError("after registry does not append new classes")
    new_classes = all_classes[len(old_classes) :]
    if old_classes != preopen_old_classes:
        raise D25RunnerError("post-materialization registry differs from pre-open binding")

    before_overlay, before_overlay_audit = legacy._overlay_index(
        before_root, before_manifest
    )
    after_overlay, after_overlay_audit = legacy._overlay_index(
        after_root, after_manifest
    )
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    scene_rows: dict[str, dict[str, np.ndarray]] = {}
    scene_z: dict[str, np.ndarray] = {}
    scene_logits: dict[str, np.ndarray] = {}
    scene_fft: dict[str, np.ndarray] = {}
    scene_rf: dict[str, np.ndarray] = {}
    scene_b3: dict[str, np.ndarray] = {}
    extraction_audits: dict[str, Any] = {}
    old_reuse_audits: dict[str, Any] = {}
    for scenario in legacy.FORMAL_LEO_WEAK_SCENARIOS:
        before_rows = legacy._rows_with_overlay(
            before_evidence.materialized_payloads[scenario],
            before_manifest,
            before_overlay,
            scenario=scenario,
        )
        after_rows = legacy._rows_with_overlay(
            after_evidence.materialized_payloads[scenario],
            after_manifest,
            after_overlay,
            scenario=scenario,
        )
        legacy._old_reuse(before_rows, after_rows)
        backbone_started = time.perf_counter()
        z_id160, direct_logits, extraction = legacy._extract_scene_signals(
            model,
            device,
            after_rows,
            component_audit["column_binding"]["direct_logit_indices"],
        )
        backbone_elapsed_ms = (time.perf_counter() - backbone_started) * 1000.0
        fft_started = time.perf_counter()
        fft96 = spectral_logmag_sketch(after_rows["iq"])
        fft_elapsed_ms = (time.perf_counter() - fft_started) * 1000.0
        rf_started = time.perf_counter()
        rf32 = rf_statistics(after_rows["iq"])
        rf_elapsed_ms = (time.perf_counter() - rf_started) * 1000.0
        if not len(z_id160) == len(fft96) == len(rf32) == len(after_rows["iq"]):
            raise D25RunnerError("D25 feature block row alignment drift")
        b3_features = _d1_feature_from_blocks(z_id160, fft96, rf32)
        scene_rows[scenario] = after_rows
        scene_z[scenario] = z_id160
        scene_logits[scenario] = direct_logits
        scene_fft[scenario] = fft96
        scene_rf[scenario] = rf32
        scene_b3[scenario] = b3_features
        extraction.update(
            {
                "feature_operator_count": 3,
                "feature_operator_ids": [
                    "adv3b02_zid160_base_v1",
                    "same_received_iq_fft96_v1",
                    "same_received_iq_rf32_v1",
                ],
                "support_view_count": 1,
                "support_row_multiplicity": 1,
                "derived_support_rows": 0,
                "additional_physical_sample_count": 0,
                "additional_leo_overlay_count": 0,
                "same_received_iq_fft96_extractions": int(len(fft96)),
                "same_received_iq_rf32_extractions": int(len(rf32)),
                "fft96_sha256": _row_hashes(fft96),
                "rf32_sha256": _row_hashes(rf32),
                "d25_block_dimensions": [160, 96, 32],
                "d25_concatenated_feature_dimension": 288,
                "b3_registered_feature_dimension": int(b3_features.shape[1]),
                "additional_backbone_forwards_for_fft_rf": 0,
                "backbone_elapsed_ms": backbone_elapsed_ms,
                "backbone_mean_ms_per_physical_sample": float(
                    backbone_elapsed_ms / len(after_rows["iq"])
                ),
                "fft96_elapsed_ms": fft_elapsed_ms,
                "fft96_mean_ms_per_physical_sample": float(
                    fft_elapsed_ms / len(after_rows["iq"])
                ),
                "rf32_elapsed_ms": rf_elapsed_ms,
                "rf32_mean_ms_per_physical_sample": float(
                    rf_elapsed_ms / len(after_rows["iq"])
                ),
                "feature_operator_lineage": _operator_lineage(after_rows),
            }
        )
        extraction_audits[scenario] = extraction
        old_reuse_audits[scenario] = {
            "old_support_exact_reuse": True,
            "before_old_rows": int(len(before_rows["labels"])),
            "after_total_rows": int(len(after_rows["labels"])),
        }
    cross_scene = legacy._cross_scene_disjointness(scene_rows)

    training_log: list[dict[str, Any]] = []
    folds_by_candidate: dict[str, list[dict[str, Any]]] = {
        candidate_id: [] for candidate_id in candidates
    }
    diag_caches: dict[str, dict[tuple[str, tuple[int, int]], dict[str, Any]]] = {
        scenario: {} for scenario in legacy.FORMAL_LEO_WEAK_SCENARIOS
    }
    for candidate_id, config in candidates.items():
        for scenario in legacy.FORMAL_LEO_WEAK_SCENARIOS:
            for fold_index, held_ranks in enumerate(HELD_RANKS):
                if isinstance(config, D40CandidateConfig):
                    row = _evaluate_d40_fold(
                        scene_rows[scenario],
                        scene_z[scenario],
                        scene_fft[scenario],
                        scene_rf[scenario],
                        old_classes=old_classes,
                        new_classes=new_classes,
                        held_ranks=held_ranks,
                        candidate_id=candidate_id,
                        config=config,
                        seed=int(before_manifest["seed"]) + fold_index,
                        device=device,
                        scenario=scenario,
                        outer_fold=fold_index,
                    )
                elif isinstance(config, D39CandidateConfig):
                    row = _evaluate_d39_fold(
                        scene_rows[scenario],
                        scene_z[scenario],
                        scene_fft[scenario],
                        scene_rf[scenario],
                        old_classes=old_classes,
                        new_classes=new_classes,
                        held_ranks=held_ranks,
                        candidate_id=candidate_id,
                        config=config,
                        seed=int(before_manifest["seed"]) + fold_index,
                        device=device,
                        scenario=scenario,
                        outer_fold=fold_index,
                    )
                elif isinstance(config, D38CandidateConfig):
                    row = _evaluate_d38_fold(
                        scene_rows[scenario],
                        scene_z[scenario],
                        scene_fft[scenario],
                        scene_rf[scenario],
                        old_classes=old_classes,
                        new_classes=new_classes,
                        held_ranks=held_ranks,
                        candidate_id=candidate_id,
                        config=config,
                        seed=int(before_manifest["seed"]) + fold_index,
                        device=device,
                        scenario=scenario,
                        outer_fold=fold_index,
                    )
                elif isinstance(config, D38ProtoNetCDAConfig):
                    row = legacy._evaluate_fold(
                        component,
                        scene_rows[scenario],
                        scene_z[scenario],
                        scene_logits[scenario],
                        scene_b3[scenario],
                        old_classes=old_classes,
                        new_classes=new_classes,
                        held_ranks=held_ranks,
                        candidate_id=IDENTITY_CANDIDATE,
                        config=candidates[IDENTITY_CANDIDATE],
                        fit_seed=int(before_manifest["seed"]) + fold_index,
                        device=device,
                        diag_cache=diag_caches[scenario],
                    )
                    row["candidate_id"] = candidate_id
                    row["baseline_equivalence_audit"] = {
                        "independent_candidate_row": True,
                        "mathematical_equivalence_expected": True,
                        "feature_geometry": config.feature_geometry,
                        "identity_candidate_id": IDENTITY_CANDIDATE,
                        "query_rows_used": 0,
                    }
                elif isinstance(config, D37CandidateConfig):
                    row = _evaluate_d37_fold(
                        scene_rows[scenario],
                        scene_z[scenario],
                        scene_fft[scenario],
                        scene_rf[scenario],
                        old_classes=old_classes,
                        new_classes=new_classes,
                        held_ranks=held_ranks,
                        candidate_id=candidate_id,
                        config=config,
                    )
                elif isinstance(config, D36CandidateConfig):
                    row = _evaluate_d36_fold(
                        scene_rows[scenario],
                        scene_z[scenario],
                        scene_fft[scenario],
                        scene_rf[scenario],
                        old_classes=old_classes,
                        new_classes=new_classes,
                        held_ranks=held_ranks,
                        candidate_id=candidate_id,
                        config=config,
                        ground_anchor=d36_ground_anchor,
                        ground_medoid_index=d36_ground_medoid_index,
                        ground_anchor_sha256=d36_ground_anchor_sha256,
                    )
                elif isinstance(config, D35CandidateConfig):
                    row = _evaluate_d35_fold(
                        scene_rows[scenario],
                        scene_z[scenario],
                        scene_fft[scenario],
                        scene_rf[scenario],
                        old_classes=old_classes,
                        new_classes=new_classes,
                        held_ranks=held_ranks,
                        candidate_id=candidate_id,
                        config=config,
                    )
                elif isinstance(config, D34CandidateConfig):
                    row = _evaluate_d34_fold(
                        scene_rows[scenario],
                        scene_z[scenario],
                        scene_fft[scenario],
                        scene_rf[scenario],
                        old_classes=old_classes,
                        new_classes=new_classes,
                        held_ranks=held_ranks,
                        candidate_id=candidate_id,
                        config=config,
                    )
                elif isinstance(config, D33CandidateConfig):
                    row = _evaluate_d33_fold(
                        scene_rows[scenario],
                        scene_z[scenario],
                        scene_fft[scenario],
                        scene_rf[scenario],
                        old_classes=old_classes,
                        new_classes=new_classes,
                        held_ranks=held_ranks,
                        candidate_id=candidate_id,
                        config=config,
                    )
                elif isinstance(config, D32CandidateConfig):
                    row = _evaluate_d32_fold(
                        component,
                        scene_rows[scenario],
                        scene_z[scenario],
                        scene_logits[scenario],
                        scene_fft[scenario],
                        scene_rf[scenario],
                        old_classes=old_classes,
                        new_classes=new_classes,
                        held_ranks=held_ranks,
                        candidate_id=candidate_id,
                        config=config,
                    )
                elif isinstance(config, D31CandidateConfig):
                    row = _evaluate_d31_fold(
                        component,
                        scene_rows[scenario],
                        scene_z[scenario],
                        scene_logits[scenario],
                        scene_fft[scenario],
                        scene_rf[scenario],
                        old_classes=old_classes,
                        new_classes=new_classes,
                        held_ranks=held_ranks,
                        candidate_id=candidate_id,
                        config=config,
                    )
                elif isinstance(config, D30CandidateConfig):
                    row = _evaluate_d30_fold(
                        component,
                        scene_rows[scenario],
                        scene_z[scenario],
                        scene_logits[scenario],
                        scene_fft[scenario],
                        scene_rf[scenario],
                        old_classes=old_classes,
                        new_classes=new_classes,
                        held_ranks=held_ranks,
                        candidate_id=candidate_id,
                        config=config,
                    )
                elif isinstance(config, D29CandidateConfig):
                    row = _evaluate_d29_fold(
                        scene_rows[scenario],
                        scene_z[scenario],
                        scene_fft[scenario],
                        scene_rf[scenario],
                        old_classes=old_classes,
                        new_classes=new_classes,
                        held_ranks=held_ranks,
                        candidate_id=candidate_id,
                        config=config,
                    )
                elif isinstance(config, D28CandidateConfig):
                    row = _evaluate_d28_fold(
                        scene_rows[scenario],
                        scene_z[scenario],
                        scene_fft[scenario],
                        scene_rf[scenario],
                        old_classes=old_classes,
                        new_classes=new_classes,
                        held_ranks=held_ranks,
                        candidate_id=candidate_id,
                        config=config,
                    )
                elif isinstance(config, D25C3Config):
                    row = _evaluate_c3_fold(
                        scene_rows[scenario],
                        scene_z[scenario],
                        scene_fft[scenario],
                        scene_rf[scenario],
                        old_classes=old_classes,
                        new_classes=new_classes,
                        held_ranks=held_ranks,
                        candidate_id=candidate_id,
                        config=config,
                    )
                elif isinstance(config, D26CompactDiagConfig):
                    row = _evaluate_d26_fold(
                        scene_rows[scenario],
                        scene_z[scenario],
                        scene_fft[scenario],
                        scene_rf[scenario],
                        old_classes=old_classes,
                        new_classes=new_classes,
                        held_ranks=held_ranks,
                        candidate_id=candidate_id,
                        config=config,
                    )
                elif candidate_id in D25_CANDIDATES:
                    if not isinstance(config, MultimodalConcatConfig):
                        raise D25RunnerError("D25 candidate config drift")
                    row = _evaluate_d25_fold(
                        component,
                        scene_rows[scenario],
                        scene_z[scenario],
                        scene_fft[scenario],
                        scene_rf[scenario],
                        old_classes=old_classes,
                        new_classes=new_classes,
                        held_ranks=held_ranks,
                        candidate_id=candidate_id,
                        config=config,
                    )
                else:
                    row = legacy._evaluate_fold(
                        component,
                        scene_rows[scenario],
                        scene_z[scenario],
                        scene_logits[scenario],
                        scene_b3[scenario],
                        old_classes=old_classes,
                        new_classes=new_classes,
                        held_ranks=held_ranks,
                        candidate_id=candidate_id,
                        config=config,
                        fit_seed=int(before_manifest["seed"]) + fold_index,
                        device=device,
                        diag_cache=diag_caches[scenario],
                    )
                if candidate_set in (
                    CANDIDATE_SET_D38_V1,
                    CANDIDATE_SET_D39_V1,
                    CANDIDATE_SET_D40_V1,
                ):
                    labels = np.asarray(scene_rows[scenario]["labels"]).astype(str)
                    ranks = np.asarray(scene_rows[scenario]["ranks"], dtype=np.int64)
                    physical_tokens = np.asarray(
                        scene_rows[scenario]["tokens"]
                    ).astype(str)
                    held_mask = np.isin(
                        ranks, np.asarray(held_ranks, dtype=np.int64)
                    )
                    train_mask = ~held_mask
                    old_mask = np.isin(labels, np.asarray(old_classes))
                    new_mask = np.isin(labels, np.asarray(new_classes))
                    if candidate_id in (
                        IDENTITY_CANDIDATE,
                        D38_PROTONET_CDA,
                        D39_PROTONET_CDA,
                        D40_PROTONET_CDA,
                    ):
                        old_proto = legacy._target_support_centroids(
                            scene_z[scenario][train_mask & old_mask],
                            labels[train_mask & old_mask],
                            old_classes,
                        )
                        new_proto = legacy._target_support_centroids(
                            scene_z[scenario][train_mask & new_mask],
                            labels[train_mask & new_mask],
                            new_classes,
                        )
                        baseline_scores = legacy._normalize_matrix(
                            scene_z[scenario][held_mask & old_mask]
                        ) @ np.concatenate([old_proto, new_proto], axis=0).T
                    elif candidate_id == DIAG_CANDIDATE:
                        diag_state = diag_caches[scenario][
                            (str(held_ranks), held_ranks)
                        ]
                        baseline_scores = legacy._diag_scores(
                            diag_state,
                            scene_b3[scenario][held_mask & old_mask],
                            include_new=True,
                        )
                        if candidate_set == CANDIDATE_SET_D40_V1:
                            _enrich_d40_strong_b3_pairwise(
                                row,
                                scene_rows[scenario],
                                scene_b3[scenario],
                                diag_state,
                                old_classes=old_classes,
                                new_classes=new_classes,
                                held_ranks=held_ranks,
                                scenario=scenario,
                                outer_fold=fold_index,
                            )
                    else:
                        baseline_scores = None
                    if baseline_scores is not None:
                        baseline_predictions = np.asarray(old_classes + new_classes)[
                            np.argmax(baseline_scores, axis=1)
                        ]
                        row["outer_held_new_intrusion_count"] = int(
                            np.sum(
                                np.isin(
                                    baseline_predictions, np.asarray(new_classes)
                                )
                            )
                        )
                    row["direct_adv3b02_old_only_anchor"] = _d38_direct_old_anchor(
                        scene_rows[scenario],
                        scene_logits[scenario],
                        old_classes=old_classes,
                        held_ranks=held_ranks,
                    )
                    held_physical_tokens = sorted(physical_tokens[held_mask].tolist())
                    row["held_physical_token_count"] = len(held_physical_tokens)
                    row["held_physical_token_sha256"] = hashlib.sha256(
                        _canonical_bytes(held_physical_tokens)
                    ).hexdigest()
                row.update(
                    {
                        "schema": _artifact_schema(candidate_set, "support_fold"),
                        "scenario": scenario,
                        "fold_index": fold_index,
                        "held_ranks": list(held_ranks),
                        "query_opened": False,
                        "formal_metric_claim_allowed": False,
                        "performance_claim_allowed": False,
                    }
                )
                folds_by_candidate[candidate_id].append(row)
                training_log.append(row)
    if candidate_set in (
        CANDIDATE_SET_D38_V1,
        CANDIDATE_SET_D39_V1,
        CANDIDATE_SET_D40_V1,
    ):
        protonet_candidate = (
            D40_PROTONET_CDA
            if candidate_set == CANDIDATE_SET_D40_V1
            else D39_PROTONET_CDA
            if candidate_set == CANDIDATE_SET_D39_V1
            else D38_PROTONET_CDA
        )
        for key in (
            (scenario, fold_index)
            for scenario in legacy.FORMAL_LEO_WEAK_SCENARIOS
            for fold_index, _ in enumerate(HELD_RANKS)
        ):
            identity_row = next(
                row
                for row in folds_by_candidate[IDENTITY_CANDIDATE]
                if (str(row["scenario"]), int(row["fold_index"])) == key
            )
            protonet_row = next(
                row
                for row in folds_by_candidate[protonet_candidate]
                if (str(row["scenario"]), int(row["fold_index"])) == key
            )
            compared_fields = (
                "before_old",
                "after_old",
                "after_new",
                "H_old_new",
                "forgetting",
                "joint_floor",
            )
            equivalent = all(
                _canonical_bytes(identity_row[name])
                == _canonical_bytes(protonet_row[name])
                for name in compared_fields
            )
            protonet_row["baseline_equivalence_audit"].update(
                {
                    "compared_fields": list(compared_fields),
                    "same_row_metrics_equal": equivalent,
                    "equivalence_audit_pass": equivalent,
                }
            )
            if not equivalent:
                raise D25RunnerError("D38/D39 ProtoNet/identity equivalence audit drift")
    expected_rows = (
        len(candidates)
        * len(legacy.FORMAL_LEO_WEAK_SCENARIOS)
        * len(HELD_RANKS)
    )
    if len(training_log) != expected_rows:
        raise D25RunnerError("D25 training-log cardinality drift")
    if candidate_set == CANDIDATE_SET_D25_V4 and expected_rows != 75:
        raise D25RunnerError("D25 training-log cardinality drift")
    if candidate_set == CANDIDATE_SET_C3_V1 and expected_rows != 90:
        raise D25RunnerError("D25 training-log cardinality drift")
    if candidate_set == CANDIDATE_SET_D26_V1 and expected_rows != 90:
        raise D25RunnerError("D26 training-log cardinality drift")
    if candidate_set == CANDIDATE_SET_D26_V2 and expected_rows != 90:
        raise D25RunnerError("D26-v2 training-log cardinality drift")
    if candidate_set == CANDIDATE_SET_D27_V1 and expected_rows != 90:
        raise D25RunnerError("D27 training-log cardinality drift")
    if candidate_set == CANDIDATE_SET_D28_V1 and expected_rows != 90:
        raise D25RunnerError("D28 training-log cardinality drift")
    if candidate_set == CANDIDATE_SET_D29_V1 and expected_rows != 90:
        raise D25RunnerError("D29 training-log cardinality drift")
    if candidate_set == CANDIDATE_SET_D30_V1 and expected_rows != 90:
        raise D25RunnerError("D30 training-log cardinality drift")
    if candidate_set == CANDIDATE_SET_D31_V1 and expected_rows != 90:
        raise D25RunnerError("D31 training-log cardinality drift")
    if candidate_set == CANDIDATE_SET_D32_V1 and expected_rows != 90:
        raise D25RunnerError("D32 training-log cardinality drift")
    if candidate_set == CANDIDATE_SET_D33_V1 and expected_rows != 105:
        raise D25RunnerError("D33 training-log cardinality drift")
    if candidate_set == CANDIDATE_SET_D34_V1 and expected_rows != 105:
        raise D25RunnerError("D34 training-log cardinality drift")
    if candidate_set == CANDIDATE_SET_D35_V1 and expected_rows != 105:
        raise D25RunnerError("D35 training-log cardinality drift")
    if candidate_set == CANDIDATE_SET_D36_V1 and expected_rows != 105:
        raise D25RunnerError("D36 training-log cardinality drift")
    if candidate_set == CANDIDATE_SET_D37_V1 and expected_rows != 105:
        raise D25RunnerError("D37 training-log cardinality drift")
    if candidate_set == CANDIDATE_SET_D38_V1 and expected_rows != 90:
        raise D25RunnerError("D38 training-log cardinality drift")
    if candidate_set == CANDIDATE_SET_D39_V1 and expected_rows != 90:
        raise D25RunnerError("D39 training-log cardinality drift")
    if candidate_set == CANDIDATE_SET_D40_V1 and expected_rows != 90:
        raise D25RunnerError("D40 training-log cardinality drift")
    selected_id, candidate_decisions = (
        _select_d40_candidate(folds_by_candidate)
        if candidate_set == CANDIDATE_SET_D40_V1
        else _select_d39_candidate(folds_by_candidate)
        if candidate_set == CANDIDATE_SET_D39_V1
        else _select_d38_candidate(folds_by_candidate)
        if candidate_set == CANDIDATE_SET_D38_V1
        else _select_d37_candidate(folds_by_candidate)
        if candidate_set == CANDIDATE_SET_D37_V1
        else _select_d36_candidate(folds_by_candidate)
        if candidate_set == CANDIDATE_SET_D36_V1
        else _select_d35_candidate(folds_by_candidate)
        if candidate_set == CANDIDATE_SET_D35_V1
        else _select_d34_candidate(folds_by_candidate)
        if candidate_set == CANDIDATE_SET_D34_V1
        else _select_c3_candidate(folds_by_candidate)
        if candidate_set == CANDIDATE_SET_C3_V1
        else _select_d26_candidate(
            folds_by_candidate,
            D27_CANDIDATES
            if candidate_set == CANDIDATE_SET_D27_V1
            else D28_CANDIDATES
            if candidate_set == CANDIDATE_SET_D28_V1
            else D29_CANDIDATES
            if candidate_set == CANDIDATE_SET_D29_V1
            else D30_CANDIDATES
            if candidate_set == CANDIDATE_SET_D30_V1
            else D31_CANDIDATES
            if candidate_set == CANDIDATE_SET_D31_V1
            else D32_CANDIDATES
            if candidate_set == CANDIDATE_SET_D32_V1
            else D33_CANDIDATES
            if candidate_set == CANDIDATE_SET_D33_V1
            else D26_CANDIDATES,
        )
        if candidate_set in (CANDIDATE_SET_D26_V1, CANDIDATE_SET_D26_V2)
        or candidate_set
        in (
            CANDIDATE_SET_D27_V1,
            CANDIDATE_SET_D28_V1,
            CANDIDATE_SET_D29_V1,
            CANDIDATE_SET_D30_V1,
            CANDIDATE_SET_D31_V1,
            CANDIDATE_SET_D32_V1,
            CANDIDATE_SET_D33_V1,
            CANDIDATE_SET_D34_V1,
            CANDIDATE_SET_D35_V1,
            CANDIDATE_SET_D36_V1,
            CANDIDATE_SET_D37_V1,
            CANDIDATE_SET_D38_V1,
            CANDIDATE_SET_D39_V1,
            CANDIDATE_SET_D40_V1,
        )
        else _select_candidate(folds_by_candidate)
    )

    deployment_resources: dict[str, dict[str, Any]] = {
        candidate_id: {} for candidate_id in candidates
    }
    geometry_ids = (
        D40_CANDIDATES
        if candidate_set == CANDIDATE_SET_D40_V1
        else D39_CANDIDATES
        if candidate_set == CANDIDATE_SET_D39_V1
        else D38_CANDIDATES
        if candidate_set == CANDIDATE_SET_D38_V1
        else (D25_C0,) + C3_CANDIDATES
        if candidate_set == CANDIDATE_SET_C3_V1
        else (D25_C0,) + D27_CANDIDATES
        if candidate_set == CANDIDATE_SET_D27_V1
        else (D25_C0,) + D28_CANDIDATES
        if candidate_set == CANDIDATE_SET_D28_V1
        else (D25_C0,) + D29_CANDIDATES
        if candidate_set == CANDIDATE_SET_D29_V1
        else (D25_C0,) + D30_CANDIDATES
        if candidate_set == CANDIDATE_SET_D30_V1
        else (D25_C0,) + D31_CANDIDATES
        if candidate_set == CANDIDATE_SET_D31_V1
        else (D25_C0,) + D32_CANDIDATES
        if candidate_set == CANDIDATE_SET_D32_V1
        else (D25_C0,) + D33_CANDIDATES
        if candidate_set == CANDIDATE_SET_D33_V1
        else (D25_C0, D33_B3_FAST) + D34_CANDIDATES
        if candidate_set == CANDIDATE_SET_D34_V1
        else (D25_C0, D33_B3_FAST) + D35_CANDIDATES
        if candidate_set == CANDIDATE_SET_D35_V1
        else (D25_C0, D33_B3_FAST) + D36_CANDIDATES
        if candidate_set == CANDIDATE_SET_D36_V1
        else (D25_C0, D33_B3_FAST) + D37_CANDIDATES
        if candidate_set == CANDIDATE_SET_D37_V1
        else (D25_C0,) + D26_CANDIDATES
        if candidate_set in (CANDIDATE_SET_D26_V1, CANDIDATE_SET_D26_V2)
        else D25_CANDIDATES
    )
    geometry_matrix: dict[str, dict[str, Any]] = {
        candidate_id: {} for candidate_id in geometry_ids
    }
    for candidate_id, config in candidates.items():
        for scenario in legacy.FORMAL_LEO_WEAK_SCENARIOS:
            if not _full_state_refit_required(
                candidate_set, candidate_id, selected_id
            ):
                deployment_resources[candidate_id][scenario] = {
                    "schema": (
                        "cvs.phase2.d40.full_k10_not_refit.v1"
                        if candidate_set == CANDIDATE_SET_D40_V1
                        else "cvs.phase2.d39.full_k10_not_refit.v1"
                        if candidate_set == CANDIDATE_SET_D39_V1
                        else "cvs.phase2.d38.full_k10_not_refit.v1"
                    ),
                    "full_k10_refit_performed": False,
                    "reason": "not_globally_selected_by_outer_6x3x5_matrix",
                    "selected_candidate_id": selected_id,
                    "query_rows_used_for_fit": 0,
                }
                geometry_matrix[candidate_id][scenario] = {
                    "schema": (
                        "cvs.phase2.d40.full_k10_not_refit_geometry.v1"
                        if candidate_set == CANDIDATE_SET_D40_V1
                        else "cvs.phase2.d39.full_k10_not_refit_geometry.v1"
                        if candidate_set == CANDIDATE_SET_D39_V1
                        else "cvs.phase2.d38.full_k10_not_refit_geometry.v1"
                    ),
                    "full_k10_refit_performed": False,
                    "selected_candidate_id": selected_id,
                    "query_rows_used": 0,
                }
                continue
            if isinstance(config, D40CandidateConfig):
                resource, geometry = _full_d40_state_audit(
                    scene_rows[scenario],
                    scene_z[scenario],
                    scene_fft[scenario],
                    scene_rf[scenario],
                    old_classes=old_classes,
                    new_classes=new_classes,
                    config=config,
                    seed=int(before_manifest["seed"]),
                    device=device,
                    scenario=scenario,
                )
                resource["direct_adv3b02_old_only_anchor"] = (
                    _d38_direct_old_anchor(
                        scene_rows[scenario],
                        scene_logits[scenario],
                        old_classes=old_classes,
                        held_ranks=None,
                    )
                )
                deployment_resources[candidate_id][scenario] = resource
                geometry_matrix[candidate_id][scenario] = geometry
            elif isinstance(config, D39CandidateConfig):
                resource, geometry = _full_d39_state_audit(
                    scene_rows[scenario],
                    scene_z[scenario],
                    scene_fft[scenario],
                    scene_rf[scenario],
                    old_classes=old_classes,
                    new_classes=new_classes,
                    config=config,
                    seed=int(before_manifest["seed"]),
                    device=device,
                    scenario=scenario,
                )
                resource["direct_adv3b02_old_only_anchor"] = (
                    _d38_direct_old_anchor(
                        scene_rows[scenario],
                        scene_logits[scenario],
                        old_classes=old_classes,
                        held_ranks=None,
                    )
                )
                deployment_resources[candidate_id][scenario] = resource
                geometry_matrix[candidate_id][scenario] = geometry
            elif isinstance(config, D38CandidateConfig):
                resource, geometry = _full_d38_state_audit(
                    scene_rows[scenario],
                    scene_z[scenario],
                    scene_fft[scenario],
                    scene_rf[scenario],
                    old_classes=old_classes,
                    new_classes=new_classes,
                    config=config,
                    seed=int(before_manifest["seed"]),
                    device=device,
                    scenario=scenario,
                )
                resource["direct_adv3b02_old_only_anchor"] = (
                    _d38_direct_old_anchor(
                        scene_rows[scenario],
                        scene_logits[scenario],
                        old_classes=old_classes,
                        held_ranks=None,
                    )
                )
                deployment_resources[candidate_id][scenario] = resource
                geometry_matrix[candidate_id][scenario] = geometry
            elif isinstance(config, D38ProtoNetCDAConfig):
                resource = legacy._deployment_state_audit(
                    component,
                    scene_rows[scenario],
                    scene_z[scenario],
                    scene_logits[scenario],
                    scene_b3[scenario],
                    old_classes=old_classes,
                    new_classes=new_classes,
                    candidate_id=IDENTITY_CANDIDATE,
                    config=candidates[IDENTITY_CANDIDATE],
                    fit_seed=int(before_manifest["seed"]),
                    device=device,
                )
                resource["baseline_equivalence_audit"] = {
                    "independent_candidate_row": True,
                    "mathematical_equivalence_expected": True,
                    "identity_candidate_id": IDENTITY_CANDIDATE,
                    "query_rows_used": 0,
                }
                resource["direct_adv3b02_old_only_anchor"] = (
                    _d38_direct_old_anchor(
                        scene_rows[scenario],
                        scene_logits[scenario],
                        old_classes=old_classes,
                        held_ranks=None,
                    )
                )
                deployment_resources[candidate_id][scenario] = resource
                geometry_matrix[candidate_id][scenario] = {
                    "schema": "cvs.phase2.d38.protonet_cda_geometry.v1",
                    "feature_geometry": config.feature_geometry,
                    "independent_candidate_row": True,
                    "equivalence_audit_required": True,
                    "query_rows_used": 0,
                }
            elif isinstance(config, D37CandidateConfig):
                resource, geometry = _full_d37_state_audit(
                    scene_rows[scenario],
                    scene_z[scenario],
                    scene_fft[scenario],
                    scene_rf[scenario],
                    old_classes=old_classes,
                    new_classes=new_classes,
                    config=config,
                )
                deployment_resources[candidate_id][scenario] = resource
                geometry_matrix[candidate_id][scenario] = geometry
            elif isinstance(config, D36CandidateConfig):
                resource, geometry = _full_d36_state_audit(
                    scene_rows[scenario],
                    scene_z[scenario],
                    scene_fft[scenario],
                    scene_rf[scenario],
                    old_classes=old_classes,
                    new_classes=new_classes,
                    config=config,
                    ground_anchor=d36_ground_anchor,
                    ground_medoid_index=d36_ground_medoid_index,
                    ground_anchor_sha256=d36_ground_anchor_sha256,
                )
                deployment_resources[candidate_id][scenario] = resource
                geometry_matrix[candidate_id][scenario] = geometry
            elif isinstance(config, D35CandidateConfig):
                resource, geometry = _full_d35_state_audit(
                    scene_rows[scenario],
                    scene_z[scenario],
                    scene_fft[scenario],
                    scene_rf[scenario],
                    old_classes=old_classes,
                    new_classes=new_classes,
                    config=config,
                )
                deployment_resources[candidate_id][scenario] = resource
                geometry_matrix[candidate_id][scenario] = geometry
            elif isinstance(config, D34CandidateConfig):
                resource, geometry = _full_d34_state_audit(
                    scene_rows[scenario],
                    scene_z[scenario],
                    scene_fft[scenario],
                    scene_rf[scenario],
                    old_classes=old_classes,
                    new_classes=new_classes,
                    config=config,
                )
                deployment_resources[candidate_id][scenario] = resource
                geometry_matrix[candidate_id][scenario] = geometry
            elif isinstance(config, D33CandidateConfig):
                resource, geometry = _full_d33_state_audit(
                    scene_rows[scenario],
                    scene_z[scenario],
                    scene_fft[scenario],
                    scene_rf[scenario],
                    old_classes=old_classes,
                    new_classes=new_classes,
                    config=config,
                )
                deployment_resources[candidate_id][scenario] = resource
                geometry_matrix[candidate_id][scenario] = geometry
            elif isinstance(config, D32CandidateConfig):
                resource, geometry = _full_d32_state_audit(
                    component,
                    scene_rows[scenario],
                    scene_z[scenario],
                    scene_logits[scenario],
                    scene_fft[scenario],
                    scene_rf[scenario],
                    old_classes=old_classes,
                    new_classes=new_classes,
                    config=config,
                )
                deployment_resources[candidate_id][scenario] = resource
                geometry_matrix[candidate_id][scenario] = geometry
            elif isinstance(config, D31CandidateConfig):
                resource, geometry = _full_d31_state_audit(
                    component,
                    scene_rows[scenario],
                    scene_z[scenario],
                    scene_logits[scenario],
                    scene_fft[scenario],
                    scene_rf[scenario],
                    old_classes=old_classes,
                    new_classes=new_classes,
                    config=config,
                )
                deployment_resources[candidate_id][scenario] = resource
                geometry_matrix[candidate_id][scenario] = geometry
            elif isinstance(config, D30CandidateConfig):
                resource, geometry = _full_d30_state_audit(
                    component,
                    scene_rows[scenario],
                    scene_z[scenario],
                    scene_logits[scenario],
                    scene_fft[scenario],
                    scene_rf[scenario],
                    old_classes=old_classes,
                    new_classes=new_classes,
                    config=config,
                )
                deployment_resources[candidate_id][scenario] = resource
                geometry_matrix[candidate_id][scenario] = geometry
            elif isinstance(config, D29CandidateConfig):
                resource, geometry = _full_d29_state_audit(
                    scene_rows[scenario],
                    scene_z[scenario],
                    scene_fft[scenario],
                    scene_rf[scenario],
                    old_classes=old_classes,
                    new_classes=new_classes,
                    config=config,
                )
                deployment_resources[candidate_id][scenario] = resource
                geometry_matrix[candidate_id][scenario] = geometry
            elif isinstance(config, D28CandidateConfig):
                resource, geometry = _full_d28_state_audit(
                    scene_rows[scenario],
                    scene_z[scenario],
                    scene_fft[scenario],
                    scene_rf[scenario],
                    old_classes=old_classes,
                    new_classes=new_classes,
                    config=config,
                )
                deployment_resources[candidate_id][scenario] = resource
                geometry_matrix[candidate_id][scenario] = geometry
            elif isinstance(config, D25C3Config):
                resource, geometry = _full_c3_state_audit(
                    scene_rows[scenario],
                    scene_z[scenario],
                    scene_fft[scenario],
                    scene_rf[scenario],
                    old_classes=old_classes,
                    new_classes=new_classes,
                    config=config,
                )
                deployment_resources[candidate_id][scenario] = resource
                geometry_matrix[candidate_id][scenario] = geometry
            elif isinstance(config, D26CompactDiagConfig):
                resource, geometry = _full_d26_state_audit(
                    scene_rows[scenario],
                    scene_z[scenario],
                    scene_fft[scenario],
                    scene_rf[scenario],
                    old_classes=old_classes,
                    new_classes=new_classes,
                    config=config,
                )
                deployment_resources[candidate_id][scenario] = resource
                geometry_matrix[candidate_id][scenario] = geometry
            elif candidate_id in D25_CANDIDATES:
                resource, geometry = _full_d25_state_audit(
                    component,
                    scene_rows[scenario],
                    scene_z[scenario],
                    scene_fft[scenario],
                    scene_rf[scenario],
                    old_classes=old_classes,
                    new_classes=new_classes,
                    config=config,
                )
                deployment_resources[candidate_id][scenario] = resource
                geometry_matrix[candidate_id][scenario] = geometry
            else:
                deployment_resources[candidate_id][scenario] = (
                    legacy._deployment_state_audit(
                        component,
                        scene_rows[scenario],
                        scene_z[scenario],
                        scene_logits[scenario],
                        scene_b3[scenario],
                        old_classes=old_classes,
                        new_classes=new_classes,
                        candidate_id=candidate_id,
                        config=config,
                        fit_seed=int(before_manifest["seed"]),
                        device=device,
                    )
                )
                if candidate_set in (
                    CANDIDATE_SET_D38_V1,
                    CANDIDATE_SET_D39_V1,
                    CANDIDATE_SET_D40_V1,
                ):
                    deployment_resources[candidate_id][scenario][
                        "direct_adv3b02_old_only_anchor"
                    ] = _d38_direct_old_anchor(
                        scene_rows[scenario],
                        scene_logits[scenario],
                        old_classes=old_classes,
                        held_ranks=None,
                    )
                    geometry_matrix[candidate_id][scenario] = {
                        "schema": _artifact_schema(
                            candidate_set, "matched_baseline_geometry"
                        ),
                        "candidate_id": candidate_id,
                        "matched_scene": scenario,
                        "query_rows_used": 0,
                    }

    pre_full_k10_selected_id = selected_id
    full_k10_fallback_reason: str | None = None
    if candidate_set == CANDIDATE_SET_C3_V1:
        selected_id, full_k10_fallback_reason = (
            _apply_full_k10_c3_old_support_gate(
                selected_id, candidate_decisions, deployment_resources
            )
        )
    elif candidate_set == CANDIDATE_SET_D38_V1:
        selected_id, full_k10_fallback_reason = _apply_full_k10_d38_gate(
            selected_id, candidate_decisions, deployment_resources
        )
    elif candidate_set == CANDIDATE_SET_D39_V1:
        selected_id, full_k10_fallback_reason = _apply_full_k10_d39_gate(
            selected_id, candidate_decisions, deployment_resources
        )
    elif candidate_set == CANDIDATE_SET_D40_V1:
        selected_id, full_k10_fallback_reason = _apply_full_k10_d40_gate(
            selected_id, candidate_decisions, deployment_resources
        )
    elif candidate_set == CANDIDATE_SET_D34_V1:
        selected_id, full_k10_fallback_reason = _apply_full_k10_d34_gate(
            selected_id, candidate_decisions, deployment_resources
        )
    elif candidate_set == CANDIDATE_SET_D35_V1:
        selected_id, full_k10_fallback_reason = _apply_full_k10_d35_gate(
            selected_id, candidate_decisions, deployment_resources
        )
    elif candidate_set == CANDIDATE_SET_D36_V1:
        selected_id, full_k10_fallback_reason = _apply_full_k10_d36_gate(
            selected_id, candidate_decisions, deployment_resources
        )
    elif candidate_set == CANDIDATE_SET_D37_V1:
        selected_id, full_k10_fallback_reason = _apply_full_k10_d37_gate(
            selected_id, candidate_decisions, deployment_resources
        )
    elif candidate_set in (
        CANDIDATE_SET_D26_V1,
        CANDIDATE_SET_D26_V2,
        CANDIDATE_SET_D27_V1,
        CANDIDATE_SET_D28_V1,
        CANDIDATE_SET_D29_V1,
        CANDIDATE_SET_D30_V1,
        CANDIDATE_SET_D31_V1,
        CANDIDATE_SET_D32_V1,
        CANDIDATE_SET_D33_V1,
    ):
        selected_id, full_k10_fallback_reason = (
            _apply_full_k10_d26_old_support_gate(
                selected_id,
                candidate_decisions,
                deployment_resources,
                D27_CANDIDATES
                if candidate_set == CANDIDATE_SET_D27_V1
                else D28_CANDIDATES
                if candidate_set == CANDIDATE_SET_D28_V1
                else D29_CANDIDATES
                if candidate_set == CANDIDATE_SET_D29_V1
                else D30_CANDIDATES
                if candidate_set == CANDIDATE_SET_D30_V1
                else D31_CANDIDATES
                if candidate_set == CANDIDATE_SET_D31_V1
                else D32_CANDIDATES
                if candidate_set == CANDIDATE_SET_D32_V1
                else D33_CANDIDATES
                if candidate_set == CANDIDATE_SET_D33_V1
                else D26_CANDIDATES,
            )
        )

    elapsed = time.perf_counter() - start
    support_audit = {
        "schema": _artifact_schema(candidate_set, "support_audit"),
        "status": "DEVELOPMENT_SUPPORT_ONLY_USER_AUTHORIZED_PREBUNDLE_INT8_SCREEN",
        "formal_launch_authority": False,
        "formal_metric_claim_allowed": False,
        "performance_claim_allowed": False,
        "query_opened": False,
        "query_rows_opened": 0,
        "query_labels_opened": 0,
        "support_query_disjointness_status": SUPPORT_QUERY_DISJOINTNESS_STATUS,
        "phase2_query_decision_policy": "per_sample_all_registered_classes",
        "phase2_query_role_oracle_access": False,
        "phase2_query_true_batch_class_count_access": False,
        "phase2_query_class_quota_access": False,
        "phase2_query_batch_global_assignment": False,
        "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "phase2_clean_dataset_reachable": False,
        "phase2_clean_cache_reachable": False,
        "phase2_clean_control_flow_reachable": False,
        "source_sample_access": False,
        "phase2_source_sample_access": False,
        "phase2_source_cache_access": False,
        "phase2_source_label_access": False,
        "phase2_unapproved_source_derived_signal_access": False,
        "phase2_source_replay": False,
        "phase2_external_source_adapter_access": False,
        "phase2_pretrained_artifact_policy": (
            "sealed_phase1_deployment_bundle_with_optional_int8_"
            "domain_class_prototypes_v1"
        ),
        "sample_level_source_feature_access": False,
        "authorized_int8_phase1_aggregate_component_access": True,
        "int8_component_update_access": False,
        "one_physical_support_one_leo_channel_observation": True,
        "support_view_count": 1,
        "support_row_multiplicity": 1,
        "feature_operator_count": 3,
        "derived_support_rows": 0,
        "additional_physical_sample_count": 0,
        "additional_leo_overlay_count": 0,
        "feature_operators_count_toward_k": False,
        "candidate_lock": candidate_lock,
        "old_reuse_by_scenario": old_reuse_audits,
        "cross_scene_disjointness": cross_scene,
        "before_overlay_audit": before_overlay_audit,
        "after_overlay_audit": after_overlay_audit,
        "feature_extraction": extraction_audits,
        "runtime_direct_logit_binding": runtime_direct_logit_binding_audit,
        "component": component_audit,
        "before_post_materialization_audit_sha256": before_authority[
            "post_materialization_audit_sha256"
        ],
        "after_post_materialization_audit_sha256": after_authority[
            "post_materialization_audit_sha256"
        ],
    }
    training_log_sha256 = legacy._write_jsonl(
        output / "training_log.jsonl", training_log
    )
    support_audit_sha256 = legacy._write_json(
        output / "support_audit.json", support_audit
    )
    selected_decision = next(
        (
            row
            for row in candidate_decisions
            if str(row["candidate_id"]) == str(selected_id)
        ),
        None,
    )
    selected_positive_route = (
        bool(
            selected_decision is not None
            and selected_id in _positive_route_candidates(candidate_set)
            and selected_decision.get("eligible_positive_route", False)
        )
        if candidate_set in (
            CANDIDATE_SET_D34_V1,
            CANDIDATE_SET_D35_V1,
            CANDIDATE_SET_D36_V1,
            CANDIDATE_SET_D37_V1,
            CANDIDATE_SET_D38_V1,
            CANDIDATE_SET_D39_V1,
            CANDIDATE_SET_D40_V1,
        )
        else selected_id in _positive_route_candidates(candidate_set)
    )
    eligible_candidate_ids = (
        [
            str(row["candidate_id"])
            for row in candidate_decisions
            if bool(row.get("eligible_positive_route", False))
        ]
        if candidate_set in (
            CANDIDATE_SET_D34_V1,
            CANDIDATE_SET_D35_V1,
            CANDIDATE_SET_D36_V1,
            CANDIDATE_SET_D37_V1,
            CANDIDATE_SET_D38_V1,
            CANDIDATE_SET_D39_V1,
            CANDIDATE_SET_D40_V1,
        )
        else list(_positive_route_candidates(candidate_set))
    )
    selection = {
        "schema": _artifact_schema(candidate_set, "selection"),
        "selected_candidate_id": selected_id,
        "pre_full_k10_selected_candidate_id": pre_full_k10_selected_id,
        "full_k10_fallback_reason": full_k10_fallback_reason,
        "selected_positive_route": selected_positive_route,
        "fallback_to_identity": selected_id == IDENTITY_CANDIDATE,
        "selection_baseline": (
            D25_C0
            if candidate_set
            in (
                CANDIDATE_SET_C3_V1,
                CANDIDATE_SET_D26_V1,
                CANDIDATE_SET_D26_V2,
                CANDIDATE_SET_D27_V1,
                CANDIDATE_SET_D28_V1,
                CANDIDATE_SET_D29_V1,
                CANDIDATE_SET_D30_V1,
                CANDIDATE_SET_D31_V1,
                CANDIDATE_SET_D32_V1,
                CANDIDATE_SET_D33_V1,
                CANDIDATE_SET_D34_V1,
                CANDIDATE_SET_D35_V1,
                CANDIDATE_SET_D36_V1,
                CANDIDATE_SET_D37_V1,
            )
            else IDENTITY_CANDIDATE
        ),
        "diagnostic_comparator": DIAG_CANDIDATE,
        "eligible_candidate_ids": eligible_candidate_ids,
        "selection_rule": (
            "C3:_all_fold_old_support_non_degradation,_per_scenario_pooled_"
            "old_and_new_floor_gain>=0.10,_per_class_drop<=0.10,_H_and_"
            "forgetting_noninferior_vs_C0;_B3_diagnostic_only"
            if candidate_set == CANDIDATE_SET_C3_V1
            else "D27:_per-new-class_pre-registration_old-only_safety_caps,_"
            "support-LOO_coordinate_bias_selection,_all_fold_old_support_non-"
            "degradation,_per_scenario_pooled_old_and_new_floor_gain>=0.10,_"
            "per_class_drop<=0.10,_H_and_forgetting_noninferior_vs_C0;_B3_"
            "performance_reference_only"
            if candidate_set == CANDIDATE_SET_D27_V1
            else "D28:_D27-B_plus_support-only_shot-rank_cross-fitted_"
            "row-local_E5_ridge_gate,_old_score_columns_unchanged,_all_fold_"
            "old_support_non-degradation,_per_scenario_pooled_old_and_new_"
            "floor_gain>=0.10,_per_class_drop<=0.10,_H_and_forgetting_"
            "noninferior_vs_C0;_B3_performance_reference_only"
            if candidate_set == CANDIDATE_SET_D28_V1
            else "D29:_D27-B_plus_support-only_shot-rank_cross-fitted_"
            "per-class_safe_release,_old_score_columns_unchanged,_full-refit_"
            "safety_and_strict-new-gain_or_atomic_passthrough,_all_fold_old_"
            "support_non-degradation,_per_scenario_pooled_old_and_new_floor_"
            "gain>=0.10,_per_class_drop<=0.10,_H_and_forgetting_noninferior_"
            "vs_C0;_B3_performance_reference_only"
            if candidate_set == CANDIDATE_SET_D29_V1
            else "D30:_B3_auxiliary-dominant_geometry_plus_D27-B,_support-"
            "old-safe_fixed-int8-medoid_DALI_max-old_rerank,_fivefold_OOF_"
            "max-new-envelope_identity_calibration,_atomic_passthrough,_all_"
            "fold_old_support_non-degradation,_per-scenario_pooled_old_and_"
            "new_floor_gain>=0.10,_per-class_drop<=0.10,_H_and_forgetting_"
            "noninferior_vs_C0;_B3_performance_reference_only"
            if candidate_set == CANDIDATE_SET_D30_V1
            else "D31:_B3_auxiliary-dominant_geometry,_15-step_frozen_"
            "Stage2-B,_10/15-step_all-registered-support_new-suffix,_fixed-"
            "medoid_DALI_old-internal_rerank,_old-support_safety_gates,_all_"
            "fold_old_support_non-degradation,_per-scenario_pooled_old_and_"
            "new_floor_gain>=0.10,_per-class_drop<=0.10,_H_and_forgetting_"
            "noninferior_vs_C0;_B3_performance_reference_only"
            if candidate_set == CANDIDATE_SET_D31_V1
            else "D34:_FAST_Fisher_frozen_old_score_prefix,_support-only_"
            "collision-local_sparse_registration,_old-support_non-degradation_"
            "and_old-LOSO_zero-intrusion_hard_gates,_then_worst_joint-floor,_"
            "worst-new-floor,_H,_and_fewer-edges_ranking,_full-K10_resource_"
            "closure;_historical_B3_and_D33-FAST_are_negative_controls"
            if candidate_set == CANDIDATE_SET_D34_V1
            else "D36:_B3_FAST_initialized_12-step_compiled_joint_old/new_"
            "int8_head,_four-inner-fold_OOF_margin_calibration_per_outer_"
            "fold,_fivefold_full-K10_crossfit,_optional_read-only_fixed_"
            "Phase1_maximin-medoid_anchor,_quantized-old_B3_noninferiority,_"
            "zero_outer-old-intrusion,_physical-new-LOO,_all-registered-class_"
            "generic-floor_and_classwise_joint-comparator,_resource_closure"
            if candidate_set == CANDIDATE_SET_D36_V1
            else "D40:_exact_six-candidate_matched_3x5_outer_matrix,_D38-B20_"
            "stage2b_then_zero-step_synchronous_HNBR,_exact-strong-B3_pairwise_"
            "old/new/floor/H/forgetting/intrusion_noninferiority_and_strict-"
            "aggregate-gain,_new-new_confusion<32,_int8-vs-matched-FP32_"
            "argmax,_held-physical/source-closure,_selected-only_full-K10_"
            "state-resource-geometry_closure"
            if candidate_set == CANDIDATE_SET_D40_V1
            else "D39:_exact_six-candidate_matched_3x5_outer_matrix,_exact_D38-B_"
            "trajectory,_all-class_angular-Gaussian_radius_nu4_epsilon001,_"
            "old-before-state_radius_source,_append-only_base-radius-r0,_"
            "pairwise_new-new/new-old,_D38-B_structural-negative,_int8-vs-"
            "matched-FP32_argmax,_15-key_old/new/floor/H/forgetting/intrusion_"
            "selector,_selected-only_full-K10_state-resource-geometry_closure"
            if candidate_set == CANDIDATE_SET_D39_V1
            else "D38:_exact_six-candidate_matched_3x5_outer_matrix,_fullbatch_"
            "B3-geometry_A0-vs-B10_global_selection,_exact-legacy-strong-B3_"
            "and_identity-or-ProtoNet_matched_gates,_pairwise-new-new/new-old_"
            "support_diagnostics,_direct-ADV3B02-old-only_anchor,_old-prefix_"
            "bitwise-invariance,_B-int8-vs-matched-FP32_argmax-invariance,_"
            "full-K10-refit_and_resource-closure"
            if candidate_set == CANDIDATE_SET_D38_V1
            else "D37:_B3-preserving_append-only_two-level_residual-int8_"
            "old/new_head,_physical-rank-pair_OOF_base-scores_and-labels,_"
            "shared-new-offset_non-empty-feasible-interval_hard-gate,_old_"
            "prefix_bitwise-invariance,_strong-B3_and_D33-FAST_classwise_"
            "joint-comparator,_zero-old-intrusion,_new-reachability,_full-"
            "K10_fivefold_OOF_and_resource-closure"
            if candidate_set == CANDIDATE_SET_D37_V1
            else "D35:_FAST_Fisher_frozen_old_score_prefix,_globally-visible_"
            "int8_dense-safe_new_registration,_fit-old_non-degradation,_outer-"
            "held_zero-new-intrusion_and_physical-new-LOO-reachability_hard_"
            "gates,_joint_B3_and_D33-FAST_old/new/H/forgetting/joint-floor_"
            "comparator,_full-K10_resource_and_geometry_closure"
            if candidate_set == CANDIDATE_SET_D35_V1
            else "D33:_B3_auxiliary-dominant_geometry,_Adam15_or_Fisher_"
            "old_diagonal,_support-only_symmetric_spherical_registration,_"
            "classwise_robust_radius_LOSO_selection,_no_int8_predictor,_"
            "complete_resource_and_argmax_latency_gate,_all-fold_old-support_"
            "non-degradation,_per-scenario_pooled_old-and-new-floor_gain>=0.10,_"
            "per-class-drop<=0.10,_H-and-forgetting_noninferior_vs_C0;_"
            "historical_B3_performance_reference_only"
            if candidate_set == CANDIDATE_SET_D33_V1
            else "D32:_B3_auxiliary-dominant_geometry,_15-step_frozen_"
            "Stage2-B,_10/15-step_all-registered-support_in-loop_safe-cap_"
            "new-suffix,_support-only_checkpoint_selection,_fixed-medoid_DALI_"
            "old-internal_rerank,_complete_resource_and_argmax_latency_gate,_"
            "all-fold_old-support_non-degradation,_per-scenario_pooled_old-and-"
            "new-floor_gain>=0.10,_per-class-drop<=0.10,_H-and-forgetting_"
            "noninferior_vs_C0;_B3_performance_reference_only"
            if candidate_set == CANDIDATE_SET_D32_V1
            else "D26-v2:_pre-registration_old-only_per-class_and_correct-row_"
            "bias_guard,_all_fold_old_support_non_degradation,_per_scenario_"
            "pooled_old_and_new_floor_gain>=0.10,_per_class_drop<=0.10,_H_"
            "and_forgetting_noninferior_vs_C0;_B3_performance_reference_only"
            if candidate_set == CANDIDATE_SET_D26_V2
            else "D26:_all_fold_old_support_non_degradation,_per_scenario_"
            "pooled_old_and_new_floor_gain>=0.10,_per_class_drop<=0.10,_H_"
            "and_forgetting_noninferior_vs_C0;_B3_performance_reference_only"
            if candidate_set == CANDIDATE_SET_D26_V1
            else "all_15_folds_classwise_noninferior_vs_Z0_and_strict_worst_"
            "after_old_or_joint_floor_improvement;_B3_is_diagnostic_only"
        ),
        "candidate_lock_sha256": candidate_lock["sha256"],
        "candidate_decisions": candidate_decisions,
    }
    selection_sha256 = legacy._write_json(output / "selection.json", selection)
    resource_sha256 = legacy._write_json(
        output / "resource_audit.json",
        {
            "schema": _artifact_schema(candidate_set, "resource_matrix"),
            "selected_candidate_id": selected_id,
            "by_candidate_by_scenario": deployment_resources,
        },
    )
    geometry_sha256 = legacy._write_json(
        output / "geometry_audit.json",
        {
            "schema": _artifact_schema(candidate_set, "geometry_matrix"),
            "query_rows_used": 0,
            "by_candidate_by_scenario": geometry_matrix,
        },
    )

    current_source_closure = _candidate_lock(candidates, candidate_set)[
        "source_closure"
    ]
    if current_source_closure != candidate_lock["source_closure"]:
        raise D25RunnerError("D25 source closure changed after support opening")
    receipt = {
        "schema": _artifact_schema(candidate_set, "receipt"),
        "status": (
            "DEVELOPMENT_SUPPORT_ONLY_COMPLETE"
            if candidate_set
            not in (
                CANDIDATE_SET_D38_V1,
                CANDIDATE_SET_D39_V1,
                CANDIDATE_SET_D40_V1,
            )
            or selected_positive_route
            else "DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE"
        ),
        "mode": mode,
        "core_commit": CORE_COMMIT,
        **(
            {
                "phase1_core_commit": CORE_COMMIT,
                "d26_core_git_commit": D26_CORE_GIT_COMMIT,
            }
            if candidate_set
            in (
                CANDIDATE_SET_D26_V1,
                CANDIDATE_SET_D26_V2,
                CANDIDATE_SET_D27_V1,
                CANDIDATE_SET_D28_V1,
                CANDIDATE_SET_D29_V1,
                CANDIDATE_SET_D30_V1,
                CANDIDATE_SET_D31_V1,
                CANDIDATE_SET_D32_V1,
                CANDIDATE_SET_D33_V1,
                CANDIDATE_SET_D34_V1,
                CANDIDATE_SET_D35_V1,
                CANDIDATE_SET_D36_V1,
                CANDIDATE_SET_D37_V1,
                CANDIDATE_SET_D38_V1,
                CANDIDATE_SET_D39_V1,
                CANDIDATE_SET_D40_V1,
            )
            else {}
        ),
        **candidate_lock["source_closure"],
        "source_closure_unchanged_after_support": True,
        "candidate_lock_sha256": candidate_lock["sha256"],
        "selected_candidate_id": selected_id,
        "pre_full_k10_selected_candidate_id": pre_full_k10_selected_id,
        "full_k10_fallback_reason": full_k10_fallback_reason,
        "selected_positive_route": selected_positive_route,
        "candidate_set": candidate_set,
        "formal_launch_authority": False,
        "formal_metric_claim_allowed": False,
        "performance_claim_allowed": False,
        "query_opened": False,
        "support_query_disjointness_status": SUPPORT_QUERY_DISJOINTNESS_STATUS,
        "receiver": str(before_manifest["receiver"]),
        "seed": int(before_manifest["seed"]),
        "k_shot": int(before_manifest["k_shot"]),
        "old_class_count": len(old_classes),
        "new_class_count": len(new_classes),
        "scenarios": list(legacy.FORMAL_LEO_WEAK_SCENARIOS),
        "candidate_count": len(candidates),
        "folds_per_candidate": len(legacy.FORMAL_LEO_WEAK_SCENARIOS)
        * len(HELD_RANKS),
        "training_log_row_count": len(training_log),
        "elapsed_seconds": elapsed,
        "training_log_sha256": training_log_sha256,
        "support_audit_sha256": support_audit_sha256,
        "selection_sha256": selection_sha256,
        "resource_audit_sha256": resource_sha256,
        "geometry_audit_sha256": geometry_sha256,
        "component_manifest_sha256": expected_component_manifest_sha256,
        "component_npz_sha256": component_audit["manifest"][
            "component_npz_sha256"
        ],
        "component_provenance_status": component_audit["manifest"][
            "provenance_status"
        ],
    }
    receipt_sha256 = legacy._write_json(output / "RECEIPT.json", receipt)
    return {"receipt_sha256": receipt_sha256, **receipt}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-root", type=Path, required=True)
    parser.add_argument("--before-seal", type=Path, required=True)
    parser.add_argument("--before-seal-sha256", required=True)
    parser.add_argument("--before-formal-policy", type=Path, required=True)
    parser.add_argument("--before-formal-policy-authorization", type=Path, required=True)
    parser.add_argument(
        "--before-signed-policy-authorization-envelope", type=Path, required=True
    )
    parser.add_argument(
        "--before-signed-policy-authorization-envelope-sha256", required=True
    )
    parser.add_argument("--after-root", type=Path, required=True)
    parser.add_argument("--after-seal", type=Path, required=True)
    parser.add_argument("--after-seal-sha256", required=True)
    parser.add_argument("--after-formal-policy", type=Path, required=True)
    parser.add_argument("--after-formal-policy-authorization", type=Path, required=True)
    parser.add_argument(
        "--after-signed-policy-authorization-envelope", type=Path, required=True
    )
    parser.add_argument(
        "--after-signed-policy-authorization-envelope-sha256", required=True
    )
    parser.add_argument("--component-dir", type=Path, required=True)
    parser.add_argument("--component-manifest-sha256", required=True)
    parser.add_argument("--class-binding", type=Path, required=True)
    parser.add_argument("--class-binding-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mode", choices=(MODE,), required=True)
    parser.add_argument(
        "--candidate-set",
        choices=(
            CANDIDATE_SET_D25_V4,
            CANDIDATE_SET_C3_V1,
            CANDIDATE_SET_D26_V1,
            CANDIDATE_SET_D26_V2,
            CANDIDATE_SET_D27_V1,
            CANDIDATE_SET_D28_V1,
            CANDIDATE_SET_D29_V1,
            CANDIDATE_SET_D30_V1,
            CANDIDATE_SET_D31_V1,
            CANDIDATE_SET_D32_V1,
            CANDIDATE_SET_D33_V1,
            CANDIDATE_SET_D34_V1,
            CANDIDATE_SET_D35_V1,
            CANDIDATE_SET_D36_V1,
            CANDIDATE_SET_D37_V1,
            CANDIDATE_SET_D38_V1,
            CANDIDATE_SET_D39_V1,
            CANDIDATE_SET_D40_V1,
        ),
        default=CANDIDATE_SET_D25_V4,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run(
        before_root=args.before_root,
        before_seal=args.before_seal,
        expected_before_seal_sha256=args.before_seal_sha256,
        before_formal_policy=args.before_formal_policy,
        before_formal_policy_authorization=args.before_formal_policy_authorization,
        before_signed_policy_authorization_envelope=(
            args.before_signed_policy_authorization_envelope
        ),
        expected_before_signed_policy_authorization_envelope_sha256=(
            args.before_signed_policy_authorization_envelope_sha256
        ),
        after_root=args.after_root,
        after_seal=args.after_seal,
        expected_after_seal_sha256=args.after_seal_sha256,
        after_formal_policy=args.after_formal_policy,
        after_formal_policy_authorization=args.after_formal_policy_authorization,
        after_signed_policy_authorization_envelope=(
            args.after_signed_policy_authorization_envelope
        ),
        expected_after_signed_policy_authorization_envelope_sha256=(
            args.after_signed_policy_authorization_envelope_sha256
        ),
        component_dir=args.component_dir,
        expected_component_manifest_sha256=args.component_manifest_sha256,
        class_binding_path=args.class_binding,
        expected_class_binding_sha256=args.class_binding_sha256,
        output=args.output,
        device_name=args.device,
        mode=args.mode,
        candidate_set=args.candidate_set,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
