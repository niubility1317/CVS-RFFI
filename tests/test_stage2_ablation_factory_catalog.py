from __future__ import annotations

import inspect

import numpy as np
import pytest

from cvsrffi.stage2_ablation_factory import (
    STAGE2_BASELINE_ARMS,
    STAGE2_MAIN_ARMS,
    STAGE2_STATE_ARMS,
    STAGE2_T1_ARMS,
    Stage2AblationConfigError,
    Stage2AblationMethod,
    build_stage2_method,
    get_stage2_arm,
    resolve_stage2_config,
    resolved_stage2_config_hash,
    stage2_config_diff,
    validate_stage2_catalog,
)


EXPECTED_MAIN_IDS = (
    "P2-FULL",
    "P2-BASE-COSINE",
    "P2-BASE-EUCLIDEAN",
    "P2-BASE-QKNN",
    "P2-BASE-DIAG-LDA",
    "P2-BASE-POOLED-LW-LDA",
    "P2-BASE-FULL-BLOCK-LDA",
    "P2-BASE-ADAPTER-HEAD",
    "P2-A0",
    "P2-B0",
    "P2-C3",
    "P2-D0",
    "P2-D1",
    "P2-D2",
    "P2-E0",
    "P2-F0",
    "P2-F1",
    "P2-F2",
    "P2-F3",
)


def test_catalog_has_frozen_state_main_and_baseline_membership() -> None:
    assert tuple(spec.ablation_id for spec in STAGE2_STATE_ARMS) == (
        "P2-S2A",
        "P2-S2B-PROTO",
        "P2-S2B-DIAGOFF",
        "P2-S2B-FULL",
    )
    assert tuple(spec.ablation_id for spec in STAGE2_MAIN_ARMS) == EXPECTED_MAIN_IDS
    assert len(STAGE2_BASELINE_ARMS) == 7
    assert len(STAGE2_T1_ARMS) == 23
    validate_stage2_catalog()


def test_every_non_alias_arm_has_exactly_its_declared_single_diff() -> None:
    for spec in STAGE2_T1_ARMS:
        diff = stage2_config_diff(spec.ablation_id)
        if spec.ablation_id == "P2-FULL" or spec.alias_of is not None:
            assert diff == {}
            assert spec.declared_diff == ()
        else:
            assert tuple(diff) == spec.declared_diff
            assert len(diff) == 1


def test_full_and_f3_are_logical_arms_with_one_effective_config() -> None:
    full = get_stage2_arm("P2-FULL")
    f3 = get_stage2_arm("P2-F3")
    assert full.alias_of is None
    assert f3.alias_of == "P2-FULL"
    assert resolve_stage2_config("P2-F3") == resolve_stage2_config("P2-FULL")
    assert resolved_stage2_config_hash("P2-F3") == resolved_stage2_config_hash("P2-FULL")


def test_resolved_full_config_closes_protocol_and_query_permissions() -> None:
    config = resolve_stage2_config("P2-FULL")
    assert config["protocol_schema"] == "p2_min_v1"
    assert config["phase1_bundle_access"] == "immutable_jointly_sealed_only"
    assert config["clean_source_runtime_access"] is False
    assert config["query_fit_access"] is False
    assert config["query_decision_policy"] == "per_sample_all_registered_classes"
    assert config["class_policy"] == "label_permutation_equivariant"
    assert config["head_profile"] == "d42_equal_prior_affine"
    assert config["fallback_profile"] == "p2_fallback_kle2"


def test_fit_signature_has_no_query_and_factory_reaches_numerical_stage2a() -> None:
    signature = inspect.signature(Stage2AblationMethod.fit)
    assert not any("query" in name.lower() for name in signature.parameters)
    method = build_stage2_method("P2-S2A")
    prototypes = np.eye(6, 288, dtype=np.float32)
    state = method.fit(
        deployment_bundle={"deployment_prototypes": prototypes},
        old_support_features=None,
        old_support_labels=None,
        old_classes=[f"old-{index}" for index in range(6)],
        seed=820001,
    )
    assert state.stage == "stage2a"
    assert state.predict(prototypes).tolist() == [
        f"old-{index}" for index in range(6)
    ]


def test_unknown_arm_fails_closed() -> None:
    with pytest.raises(Stage2AblationConfigError, match="unknown frozen Stage2"):
        resolve_stage2_config("P2-NOT-REGISTERED")
