"""Behavioral checks for the approved Phase1 MIRAGE-OWDG data policy."""

from __future__ import annotations

import importlib

import pytest


def _policy_api():
    """Import the public policy API inside tests so RED reports a missing feature."""

    module = importlib.import_module("cvsrffi.phase1_mirage.protocol")
    return (
        module.Permission,
        module.Phase1DataPolicy,
        module.Phase1PolicyError,
        module.ProxyRole,
        module.SampleIdentity,
        module.SourcePartition,
        module.TargetRole,
    )


def test_partition_counts_follow_the_7_63_15_15_source_policy():
    _, Phase1DataPolicy, _, _, _, SourcePartition, _ = _policy_api()

    counts = Phase1DataPolicy().partition_counts(100)

    assert counts == {
        SourcePartition.L_S: 7,
        SourcePartition.U_S: 63,
        SourcePartition.V_CAL: 15,
        SourcePartition.V_SELECT: 15,
    }


def test_source_partitions_allow_shared_tx_identity_when_physical_ids_differ():
    _, Phase1DataPolicy, _, _, SampleIdentity, _, _ = _policy_api()

    Phase1DataPolicy().validate_source_partitions(
        l_s=(SampleIdentity("l-01", "source-tx-7"),),
        u_s=(SampleIdentity("u-01", "source-tx-7"),),
        v_cal=(SampleIdentity("vc-01", "source-tx-7"),),
        v_select=(SampleIdentity("vs-01", "source-tx-7"),),
    )


def test_source_partitions_reject_reused_physical_sample_ids():
    _, Phase1DataPolicy, Phase1PolicyError, _, SampleIdentity, _, _ = _policy_api()

    with pytest.raises(Phase1PolicyError, match="physical_sample_id"):
        Phase1DataPolicy().validate_source_partitions(
            l_s=(SampleIdentity("shared-01", "source-tx-1"),),
            u_s=(SampleIdentity("u-01", "source-tx-1"),),
            v_cal=(SampleIdentity("shared-01", "source-tx-1"),),
            v_select=(SampleIdentity("vs-01", "source-tx-1"),),
        )


def test_proxy_train_is_labeled_training_only_and_can_receive_rejection_gradients():
    Permission, Phase1DataPolicy, Phase1PolicyError, ProxyRole, _, SourcePartition, _ = _policy_api()
    policy = Phase1DataPolicy()

    assert policy.proxy_origin_is_allowed(ProxyRole.PROXY_TRAIN, SourcePartition.L_S)
    assert policy.allows(ProxyRole.PROXY_TRAIN, Permission.REJECTION_GRADIENT)
    assert {
        action for action in Permission if policy.allows(ProxyRole.PROXY_TRAIN, action)
    } == {Permission.REJECTION_GRADIENT}
    with pytest.raises(Phase1PolicyError, match="L_s"):
        policy.require_proxy_origin(ProxyRole.PROXY_TRAIN, SourcePartition.U_S)


def test_validation_proxies_keep_calibration_and_model_selection_separate():
    Permission, Phase1DataPolicy, Phase1PolicyError, ProxyRole, _, SourcePartition, _ = _policy_api()
    policy = Phase1DataPolicy()

    assert policy.proxy_origin_is_allowed(ProxyRole.P_CAL, SourcePartition.V_CAL)
    assert policy.proxy_origin_is_allowed(ProxyRole.P_SELECT, SourcePartition.V_SELECT)
    assert policy.allows(ProxyRole.P_CAL, Permission.CALIBRATE)
    assert policy.allows(ProxyRole.P_SELECT, Permission.SELECT_MODEL)
    assert {action for action in Permission if policy.allows(ProxyRole.P_CAL, action)} == {
        Permission.CALIBRATE
    }
    assert {action for action in Permission if policy.allows(ProxyRole.P_SELECT, action)} == {
        Permission.SELECT_MODEL
    }
    with pytest.raises(Phase1PolicyError, match="V_cal"):
        policy.require_proxy_origin(ProxyRole.P_CAL, SourcePartition.V_SELECT)
    with pytest.raises(Phase1PolicyError, match="V_select"):
        policy.require_proxy_origin(ProxyRole.P_SELECT, SourcePartition.V_CAL)


@pytest.mark.parametrize("overlapping_tx", ("source-train-tx", "source-validation-tx"))
def test_target_unknown_tx_must_be_disjoint_from_source_train_and_validation_tx(overlapping_tx: str):
    _, Phase1DataPolicy, Phase1PolicyError, _, _, _, _ = _policy_api()
    policy = Phase1DataPolicy()

    with pytest.raises(Phase1PolicyError, match="target unknown TX"):
        policy.validate_target_unknown_identities(
            target_unknown_tx_ids={"target-unknown", overlapping_tx},
            source_train_tx_ids={"source-train-tx"},
            source_validation_tx_ids={"source-validation-tx"},
        )


def test_target_unknown_tx_allows_disjoint_source_train_and_validation_tx():
    _, Phase1DataPolicy, _, _, _, _, _ = _policy_api()

    result = Phase1DataPolicy().validate_target_unknown_identities(
        target_unknown_tx_ids={"target-unknown-1", "target-unknown-2"},
        source_train_tx_ids={"source-train-tx"},
        source_validation_tx_ids={"source-validation-tx"},
    )

    assert result is None


def test_target_roles_fail_closed_for_candidate_reranking():
    Permission, Phase1DataPolicy, Phase1PolicyError, _, _, _, TargetRole = _policy_api()
    policy = Phase1DataPolicy()

    for target_role in TargetRole:
        assert not policy.allows(target_role, Permission.CANDIDATE_RERANK)
        with pytest.raises(Phase1PolicyError, match="target"):
            policy.require_permission(target_role, Permission.CANDIDATE_RERANK)


def test_target_roles_cannot_change_training_calibration_selection_or_retries():
    Permission, Phase1DataPolicy, Phase1PolicyError, _, _, _, TargetRole = _policy_api()
    policy = Phase1DataPolicy()

    for target_role in TargetRole:
        for action in Permission:
            assert not policy.allows(target_role, action)
            with pytest.raises(Phase1PolicyError, match="target"):
                policy.require_permission(target_role, action)
