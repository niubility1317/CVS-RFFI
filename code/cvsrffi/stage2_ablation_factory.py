"""Frozen catalog and support-only factory for Phase2 T1 ablation methods."""

from __future__ import annotations

import hashlib
import inspect
import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping


STAGE2_ABLATION_SCHEMA = "cvs.stage2_ablation.config.v1"
PROTOCOL_SCHEMA = "p2_min_v1"


class Stage2AblationConfigError(ValueError):
    """Raised when a frozen Stage2 catalog entry is invalid."""


class Stage2AblationNotImplementedError(RuntimeError):
    """Raised instead of silently executing an unimplemented method."""


@dataclass(frozen=True)
class Stage2ArmSpec:
    """Immutable identity and one-factor comparison declaration."""

    ablation_id: str
    stage: str
    table: str
    grade: str
    factor: str
    reference_id: str
    declared_diff: tuple[str, ...]
    alias_of: str | None = None


_FULL_CONFIG: dict[str, Any] = {
    "schema": STAGE2_ABLATION_SCHEMA,
    "protocol_schema": PROTOCOL_SCHEMA,
    "phase1_bundle_access": "immutable_jointly_sealed_only",
    "target_input_access": "fixed_received_iq_and_current_row_support_only",
    "clean_source_runtime_access": False,
    "query_fit_access": False,
    "query_decision_policy": "per_sample_all_registered_classes",
    "class_policy": "label_permutation_equivariant",
    "state_profile": "stage2c_old_new_kshot",
    "method_family": "rtb_idr_d92",
    "feature_profile": "identity160_fft96_rf32_beta4_blocknorm_globalnorm",
    "center_profile": "d81_ground_spectrum_cauchy",
    "covariance_profile": "d92_old_new_task_balanced_ledoit_wolf",
    "geometry_profile": "d46_full_block_classwise_support_loo",
    "fisher_profile": "d62_bounded_pareto_atomic",
    "head_profile": "d42_equal_prior_affine",
    "quantization_profile": (
        "f3_dual_residual_int8_fp16_block_scale_bias_fp16_diag_metric"
    ),
    "fallback_profile": "p2_fallback_kle2",
}


def _spec(
    ablation_id: str,
    *,
    stage: str,
    table: str,
    factor: str,
    diff: str | None,
    reference_id: str = "P2-FULL",
    alias_of: str | None = None,
) -> Stage2ArmSpec:
    return Stage2ArmSpec(
        ablation_id=ablation_id,
        stage=stage,
        table=table,
        grade="M",
        factor=factor,
        reference_id=reference_id,
        declared_diff=() if diff is None else (diff,),
        alias_of=alias_of,
    )


# The order is frozen to the manuscript's T1 release order.
STAGE2_STATE_ARMS: tuple[Stage2ArmSpec, ...] = (
    _spec("P2-S2A", stage="stage2a", table="state", factor="zero_label_deployment", diff="state_profile"),
    _spec("P2-S2B-PROTO", stage="stage2b", table="state", factor="old_class_prototype", diff="state_profile"),
    _spec("P2-S2B-DIAGOFF", stage="stage2b", table="state", factor="old_class_metric", diff="state_profile"),
    _spec("P2-S2B-FULL", stage="stage2b", table="state", factor="old_class_adaptation", diff="state_profile"),
)

STAGE2_BASELINE_ARMS: tuple[Stage2ArmSpec, ...] = (
    _spec("P2-BASE-COSINE", stage="stage2c", table="baseline", factor="baseline", diff="method_family"),
    _spec("P2-BASE-EUCLIDEAN", stage="stage2c", table="baseline", factor="baseline", diff="method_family"),
    _spec("P2-BASE-QKNN", stage="stage2c", table="baseline", factor="baseline", diff="method_family"),
    _spec("P2-BASE-DIAG-LDA", stage="stage2c", table="baseline", factor="baseline", diff="method_family"),
    _spec("P2-BASE-POOLED-LW-LDA", stage="stage2c", table="baseline", factor="baseline", diff="method_family"),
    _spec("P2-BASE-FULL-BLOCK-LDA", stage="stage2c", table="baseline", factor="baseline", diff="method_family"),
    _spec("P2-BASE-ADAPTER-HEAD", stage="stage2c", table="baseline", factor="baseline", diff="method_family"),
)

STAGE2_MAIN_ARMS: tuple[Stage2ArmSpec, ...] = (
    _spec("P2-FULL", stage="stage2c", table="main", factor="reference", diff=None),
    *STAGE2_BASELINE_ARMS,
    _spec("P2-A0", stage="stage2c", table="main", factor="joint_feature", diff="feature_profile"),
    _spec("P2-A1", stage="stage2c", table="main", factor="joint_feature", diff="feature_profile"),
    _spec("P2-A2", stage="stage2c", table="main", factor="joint_feature", diff="feature_profile"),
    _spec("P2-B0", stage="stage2c", table="main", factor="robust_center", diff="center_profile"),
    _spec("P2-C3", stage="stage2c", table="main", factor="task_covariance", diff="covariance_profile"),
    _spec("P2-D0", stage="stage2c", table="main", factor="dual_geometry", diff="geometry_profile"),
    _spec("P2-D1", stage="stage2c", table="main", factor="dual_geometry", diff="geometry_profile"),
    _spec("P2-D2", stage="stage2c", table="main", factor="crossfit_fusion", diff="geometry_profile"),
    _spec("P2-E0", stage="stage2c", table="main", factor="fisher_safety", diff="fisher_profile"),
    _spec("P2-F0", stage="stage2c", table="main", factor="quantization", diff="quantization_profile"),
    _spec("P2-F1", stage="stage2c", table="main", factor="quantization", diff="quantization_profile"),
    _spec("P2-F2", stage="stage2c", table="main", factor="quantization", diff="quantization_profile"),
    _spec(
        "P2-F3",
        stage="stage2c",
        table="main",
        factor="quantization",
        diff=None,
        alias_of="P2-FULL",
    ),
)

STAGE2_T1_ARMS: tuple[Stage2ArmSpec, ...] = (*STAGE2_STATE_ARMS, *STAGE2_MAIN_ARMS)


# This is intentionally a separate current-method screen rather than a change
# to the historical T1 catalogue.  Every arm starts from identity160+FFT96;
# it never includes the FP32 F0 quantization comparison.
STAGE2_E0_256_ABLATION_ARMS: tuple[Stage2ArmSpec, ...] = (
    _spec(
        "P2-256-FULL",
        stage="stage2c",
        table="e0_256_module_screen",
        factor="reference",
        diff=None,
        reference_id="P2-256-FULL",
    ),
    _spec(
        "P2-256-A0",
        stage="stage2c",
        table="e0_256_module_screen",
        factor="joint_feature",
        diff="feature_profile",
        reference_id="P2-256-FULL",
    ),
    _spec(
        "P2-256-B0",
        stage="stage2c",
        table="e0_256_module_screen",
        factor="robust_center",
        diff="center_profile",
        reference_id="P2-256-FULL",
    ),
    _spec(
        "P2-256-S0",
        stage="stage2c",
        table="e0_256_module_screen",
        factor="auto_shrinkage",
        diff="covariance_profile",
        reference_id="P2-256-FULL",
    ),
    _spec(
        "P2-256-C3",
        stage="stage2c",
        table="e0_256_module_screen",
        factor="task_covariance",
        diff="covariance_profile",
        reference_id="P2-256-FULL",
    ),
    _spec(
        "P2-256-D0",
        stage="stage2c",
        table="e0_256_module_screen",
        factor="dual_geometry",
        diff="geometry_profile",
        reference_id="P2-256-FULL",
    ),
    _spec(
        "P2-256-D2",
        stage="stage2c",
        table="e0_256_module_screen",
        factor="crossfit_fusion",
        diff="geometry_profile",
        reference_id="P2-256-FULL",
    ),
)

STAGE2_ALL_ARMS: tuple[Stage2ArmSpec, ...] = (
    *STAGE2_T1_ARMS,
    *STAGE2_E0_256_ABLATION_ARMS,
)


_OVERRIDES: dict[str, dict[str, Any]] = {
    "P2-FULL": {},
    "P2-S2A": {"state_profile": "stage2a_zero_label_frozen_bundle"},
    "P2-S2B-PROTO": {"state_profile": "stage2b_old_support_cosine_prototype"},
    "P2-S2B-DIAGOFF": {"state_profile": "stage2b_old_full_diag_metric_off"},
    "P2-S2B-FULL": {"state_profile": "stage2b_old_full_diag_metric_20_steps"},
    "P2-BASE-COSINE": {"method_family": "cosine_nearest_centroid"},
    "P2-BASE-EUCLIDEAN": {"method_family": "euclidean_protonet"},
    "P2-BASE-QKNN": {"method_family": "single_qknn"},
    "P2-BASE-DIAG-LDA": {"method_family": "diagonal_lda"},
    "P2-BASE-POOLED-LW-LDA": {"method_family": "pooled_ledoit_wolf_lda"},
    "P2-BASE-FULL-BLOCK-LDA": {"method_family": "full_block_shrinkage_lda_no_robust_center"},
    "P2-BASE-ADAPTER-HEAD": {"method_family": "frozen_lightweight_adapter_equal_prior_head"},
    "P2-A0": {"feature_profile": "identity160_only"},
    "P2-A1": {"feature_profile": "identity160_fft96_beta4_blocknorm_globalnorm"},
    "P2-A2": {"feature_profile": "identity160_rf32_beta4_blocknorm_globalnorm"},
    "P2-B0": {"center_profile": "support_plain_mean_no_ground_spectrum"},
    "P2-C3": {"covariance_profile": "d81_all_classes_equal_ledoit_wolf"},
    "P2-D0": {"geometry_profile": "full_only"},
    "P2-D1": {"geometry_profile": "block3_only"},
    "P2-D2": {"geometry_profile": "full_block_fixed_half"},
    "P2-E0": {"fisher_profile": "off"},
    "P2-F0": {"quantization_profile": "f0_fp32_weight_fp32_bias"},
    "P2-F1": {"quantization_profile": "f1_fp16_weight_fp16_bias"},
    "P2-F2": {"quantization_profile": "f2_single_int8_fp16_scale"},
    # P2-F3 is a logical manuscript arm and a physical alias of P2-FULL.
    "P2-F3": {},
    # The current 256-D screen preserves the compiled F3 deployment state for
    # every arm.  It deliberately has no P2-256-F0 arm.
    "P2-256-FULL": {
        "feature_profile": "identity160_fft96_beta4_blocknorm_globalnorm",
    },
    "P2-256-A0": {"feature_profile": "identity160_only_compact"},
    "P2-256-B0": {
        "feature_profile": "identity160_fft96_beta4_blocknorm_globalnorm",
        "center_profile": "support_plain_mean_no_ground_spectrum",
    },
    "P2-256-S0": {
        "feature_profile": "identity160_fft96_beta4_blocknorm_globalnorm",
        "covariance_profile": (
            "d92_old_new_task_balanced_empirical_fixed_ridge"
        ),
    },
    "P2-256-C3": {
        "feature_profile": "identity160_fft96_beta4_blocknorm_globalnorm",
        "covariance_profile": "d81_all_classes_equal_ledoit_wolf",
    },
    "P2-256-D0": {
        "feature_profile": "identity160_fft96_beta4_blocknorm_globalnorm",
        "geometry_profile": "full_only",
    },
    "P2-256-D2": {
        "feature_profile": "identity160_fft96_beta4_blocknorm_globalnorm",
        "geometry_profile": "full_block_fixed_half",
    },
}

_SPECS_BY_ID: dict[str, Stage2ArmSpec] = {
    spec.ablation_id: spec for spec in STAGE2_ALL_ARMS
}


def get_stage2_arm(ablation_id: str) -> Stage2ArmSpec:
    """Return a frozen arm declaration or fail on an unknown identity."""

    try:
        return _SPECS_BY_ID[ablation_id]
    except KeyError as exc:
        raise Stage2AblationConfigError(
            f"unknown frozen Stage2 ablation ID: {ablation_id!r}"
        ) from exc


def resolve_stage2_config(ablation_id: str) -> dict[str, Any]:
    """Resolve the method-only configuration for one logical arm."""

    get_stage2_arm(ablation_id)
    resolved = deepcopy(_FULL_CONFIG)
    resolved.update(deepcopy(_OVERRIDES[ablation_id]))
    return resolved


def canonical_stage2_config_json(ablation_id: str) -> str:
    """Return stable JSON used for physical-config identity."""

    return json.dumps(
        resolve_stage2_config(ablation_id),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def resolved_stage2_config_hash(ablation_id: str) -> str:
    """Return the SHA-256 identity of the effective method configuration."""

    return hashlib.sha256(canonical_stage2_config_json(ablation_id).encode("utf-8")).hexdigest()


def stage2_config_diff(
    ablation_id: str, reference_id: str | None = None
) -> dict[str, tuple[Any, Any]]:
    """Return exact resolved differences as ``key: (reference, arm)``."""

    spec = get_stage2_arm(ablation_id)
    reference = resolve_stage2_config(reference_id or spec.reference_id)
    candidate = resolve_stage2_config(ablation_id)
    keys = set(reference) | set(candidate)
    return {
        key: (reference.get(key), candidate.get(key))
        for key in sorted(keys)
        if reference.get(key) != candidate.get(key)
    }


class Stage2AblationMethod:
    """Frozen arm identity with a query-inaccessible numerical fit boundary."""

    def __init__(self, ablation_id: str):
        self.spec = get_stage2_arm(ablation_id)
        self.config = resolve_stage2_config(ablation_id)
        self.resolved_config_hash = resolved_stage2_config_hash(ablation_id)

    def fit(
        self,
        *,
        deployment_bundle: Mapping[str, Any],
        old_support_features: Any | None,
        old_support_labels: Any | None,
        old_classes: Any,
        new_support_features: Any | None = None,
        new_support_labels: Any | None = None,
        new_classes: Any = (),
        seed: int,
        device: Any = "cpu",
        module2_mode: str = "baseline",
    ) -> Any:
        """Fit from immutable deployment state and legal support only."""

        from cvsrffi.stage2_ablation_executors import fit_stage2_ablation

        return fit_stage2_ablation(
            ablation_id=self.spec.ablation_id,
            old_support_features=old_support_features,
            old_support_labels=old_support_labels,
            old_classes=old_classes,
            new_support_features=new_support_features,
            new_support_labels=new_support_labels,
            new_classes=new_classes,
            deployment_prototypes=deployment_bundle.get("deployment_prototypes"),
            ground_basis=deployment_bundle.get("ground_basis"),
            ground_spectral_weights=deployment_bundle.get(
                "ground_spectral_weights"
            ),
            ground_audit=deployment_bundle.get("ground_audit"),
            ground_class_centers=deployment_bundle.get("ground_class_centers"),
            ground_full_centers=deployment_bundle.get("ground_full_centers"),
            ground_class_registry=deployment_bundle.get("ground_class_registry"),
            module2_mode=module2_mode,
            seed=int(seed),
            device=device,
        )


def build_stage2_method(ablation_id: str) -> Stage2AblationMethod:
    """Build a frozen, support-only numerical arm."""

    return Stage2AblationMethod(ablation_id)


def validate_stage2_catalog() -> None:
    """Validate counts, aliases, and the one-factor resolved-diff contract."""

    ids = [spec.ablation_id for spec in STAGE2_ALL_ARMS]
    if len(ids) != len(set(ids)):
        raise Stage2AblationConfigError("Stage2 catalog contains duplicate IDs")
    if len(STAGE2_STATE_ARMS) != 4:
        raise Stage2AblationConfigError("Stage2 state catalog must contain 4 arms")
    if len(STAGE2_MAIN_ARMS) != 21:
        raise Stage2AblationConfigError("Stage2 main catalog must contain 21 logical arms")
    if len(STAGE2_BASELINE_ARMS) != 7:
        raise Stage2AblationConfigError("Stage2 baseline catalog must contain 7 arms")
    if len(STAGE2_E0_256_ABLATION_ARMS) != 7:
        raise Stage2AblationConfigError(
            "Stage2 current-256D catalog must contain 7 logical arms"
        )
    if set(ids) != set(_OVERRIDES):
        raise Stage2AblationConfigError("Stage2 specs and overrides do not cover the same IDs")

    for spec in STAGE2_ALL_ARMS:
        diff_keys = tuple(stage2_config_diff(spec.ablation_id))
        if spec.ablation_id == spec.reference_id:
            if diff_keys or spec.declared_diff:
                raise Stage2AblationConfigError(
                    f"{spec.ablation_id} must be the zero-diff reference"
                )
            continue
        if spec.alias_of is not None:
            if diff_keys or spec.declared_diff:
                raise Stage2AblationConfigError(
                    f"{spec.ablation_id} alias must have no physical config diff"
                )
            if resolved_stage2_config_hash(spec.ablation_id) != resolved_stage2_config_hash(
                spec.alias_of
            ):
                raise Stage2AblationConfigError(
                    f"{spec.ablation_id} alias hash differs from {spec.alias_of}"
                )
            continue
        if diff_keys != spec.declared_diff or len(diff_keys) != 1:
            raise Stage2AblationConfigError(
                f"{spec.ablation_id} resolved diff {diff_keys!r} does not match "
                f"declared single diff {spec.declared_diff!r}"
            )

    fit_parameters = inspect.signature(Stage2AblationMethod.fit).parameters
    if any("query" in name.lower() for name in fit_parameters):
        raise Stage2AblationConfigError("Stage2 fit API must not accept query inputs")


validate_stage2_catalog()


__all__ = [
    "PROTOCOL_SCHEMA",
    "STAGE2_ABLATION_SCHEMA",
    "STAGE2_ALL_ARMS",
    "STAGE2_BASELINE_ARMS",
    "STAGE2_E0_256_ABLATION_ARMS",
    "STAGE2_MAIN_ARMS",
    "STAGE2_STATE_ARMS",
    "STAGE2_T1_ARMS",
    "Stage2AblationConfigError",
    "Stage2AblationMethod",
    "Stage2AblationNotImplementedError",
    "Stage2ArmSpec",
    "build_stage2_method",
    "canonical_stage2_config_json",
    "get_stage2_arm",
    "resolve_stage2_config",
    "resolved_stage2_config_hash",
    "stage2_config_diff",
    "validate_stage2_catalog",
]
