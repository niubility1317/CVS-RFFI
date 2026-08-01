from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

import cvsrffi.stage2_d106_four_arm as four_arm


def _sha(character: str) -> str:
    return character * 64


def _binding(**changes: object) -> four_arm.D106FourArmJobBinding:
    values: dict[str, object] = {
        "job_id": "target25-row-0007",
        "scenario_id": "leo_low_elev_weak",
        "capsule_id": _sha("1"),
        "split_id": _sha("2"),
        "validator_receipt_sha256": _sha("3"),
        "support_physical_root_sha256": _sha("4"),
        "query_physical_root_sha256": _sha("5"),
        "row_id": "rx7-seed713102-k5-new20",
        "seed": 713102,
        "active_k": 5,
        "registered_class_count": 6,
    }
    values.update(changes)
    return four_arm.D106FourArmJobBinding(**values)


def _component_receipts() -> dict[str, str]:
    return {
        "M0": _sha("a"),
        "M_DA": _sha("b"),
        "M_HEAD": _sha("c"),
        "M_JOINT": _sha("d"),
    }


def _assert_request_only(state: four_arm.D106FourArmState) -> None:
    assert state.same_job_scenario_support_query_binding is False
    assert state.same_job_scenario_support_query_binding_requested is True
    assert state.same_job_scenario_support_query_binding_verified is False
    assert state.request_receipt_only_not_data_lineage_authority is True
    assert state.data_lineage_authority_verified is False


def test_fixed_two_by_two_arm_map_and_simple_effect_trace() -> None:
    state = four_arm.build_d106_four_arm_state(_binding())

    assert tuple(state.arm_map) == four_arm.ARMS
    assert state.arm_map["M0"] == {
        "arm_id": "M0",
        "da_factor": "identity",
        "head_factor": "baseline_qknn",
    }
    assert state.arm_map["M_DA"]["da_factor"] == "rdce"
    assert state.arm_map["M_HEAD"]["head_factor"] == "rcmr_2v"
    assert state.arm_map["M_JOINT"] == {
        "arm_id": "M_JOINT",
        "da_factor": "rdce",
        "head_factor": "rcmr_2v",
    }
    assert {
        effect_id: (effect["before_arm"], effect["after_arm"])
        for effect_id, effect in state.simple_effect_trace.items()
    } == {
        "DA_AT_BASELINE_HEAD": ("M0", "M_DA"),
        "DA_AT_RCMR_HEAD": ("M_HEAD", "M_JOINT"),
        "HEAD_AT_IDENTITY_DA": ("M0", "M_HEAD"),
        "HEAD_AT_RDCE_DA": ("M_DA", "M_JOINT"),
    }
    _assert_request_only(state)
    assert state.binding.query_physical_root_sha256 == _sha("5")


def test_receipt_is_deterministic_and_receipt_state_is_immutable() -> None:
    first = four_arm.build_d106_four_arm_state(_binding())
    second = four_arm.build_d106_four_arm_state(_binding())

    assert first.state_receipt_sha256 == second.state_receipt_sha256
    assert first.binding.receipt_sha256 == second.binding.receipt_sha256
    assert first.arm_map == second.arm_map
    with pytest.raises(FrozenInstanceError):
        first.binding.job_id = "replacement"  # type: ignore[misc]
    with pytest.raises(TypeError):
        first.arm_map["M0"]["da_factor"] = "rdce"  # type: ignore[index]
    with pytest.raises(TypeError):
        first.simple_effect_trace["DA_AT_BASELINE_HEAD"] = {}  # type: ignore[index]

    audit = four_arm.audit_d106_four_arm_resources(first)
    assert audit["additional_persistent_numeric_bytes"] == 0
    assert audit["additional_query_numeric_buffers"] == 0
    assert audit["parameter_scan_dimensions"] == 0
    assert audit["query_execution_capability"] is False
    assert audit["performance_scorer_attached"] is False
    assert audit["formal_components_bound"] is False
    assert audit["actual_canonical_payload_bytes"] > 0
    assert (
        audit["actual_canonical_payload_bytes"]
        <= audit["canonical_payload_hard_cap_bytes"]
    )
    assert audit["external_component_resources"] == (
        "not_aggregated_by_composition_layer"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("protocol_schema", "p2_other"),
        ("phase2_data_status", "UNVALIDATED"),
        ("support_query_disjoint", False),
        ("active_k", 2),
        ("query_decision_policy", "batch_quota"),
        ("clean_source_runtime_access", True),
        ("source_runtime_access", True),
        ("query_truth_access", True),
        ("query_role_access", True),
        ("query_batch_count_access", True),
        ("query_class_quota_access", True),
        ("query_fit_access", True),
        ("query_state_updates", 1),
        ("performance_scorer_attached", True),
    ),
)
def test_protocol_and_query_capability_negatives_fail_closed(
    field: str, value: object
) -> None:
    with pytest.raises(four_arm.D106FourArmError):
        _binding(**{field: value})


def test_no_scan_fallback_or_verified_binding_can_be_constructed() -> None:
    state = four_arm.build_d106_four_arm_state(_binding())
    with pytest.raises(four_arm.D106FourArmError):
        replace(state, parameter_scan_dimensions=1)
    with pytest.raises(four_arm.D106FourArmError):
        replace(state, fallback_allowed=True)
    with pytest.raises(four_arm.D106FourArmError):
        replace(state, query_execution_capability=True)
    with pytest.raises(four_arm.D106FourArmError):
        replace(state, same_job_scenario_support_query_binding=True)
    with pytest.raises(four_arm.D106FourArmError):
        replace(state, same_job_scenario_support_query_binding_verified=True)
    with pytest.raises(four_arm.D106FourArmError):
        replace(state, data_lineage_authority_verified=True)


def test_formal_prepare_is_unconditionally_fail_closed() -> None:
    expected = (
        "D106AdaptedSupportBankReceipt.*identity-view authority.*RDCE-view "
        "authority.*RCMR strict-loader-wire authority"
    )

    with pytest.raises(four_arm.D106FourArmAuthorityError, match=expected):
        four_arm.prepare_d106_four_arm_formal_handoff()
    assert not hasattr(four_arm, "D106FourArmFormalHandoff")
    assert "D106FourArmFormalHandoff" not in four_arm.__all__


def test_duplicate_m0_as_m_da_or_same_config_bank_cannot_be_formalized() -> None:
    state = four_arm.build_d106_four_arm_state(_binding())
    shared_bank = object()

    with pytest.raises(
        four_arm.D106FourArmAuthorityError,
        match="D106AdaptedSupportBankReceipt",
    ):
        four_arm.prepare_d106_four_arm_formal_handoff(
            state,
            rdce_runtime={"same_config": True},
            m0_baseline_qknn_bank=shared_bank,
            m_da_baseline_qknn_bank=shared_bank,
            m_head_rcmr_state={"same_config": True},
            m_joint_rcmr_state={"same_config": True},
        )


def test_public_rcmr_state_cannot_be_formalized() -> None:
    state = four_arm.build_d106_four_arm_state(_binding())
    public_rcmr_state = {
        "public_state_receipt": _sha("e"),
        "is_formal": True,
        "strict_loader_wire": "unverified",
    }

    with pytest.raises(
        four_arm.D106FourArmAuthorityError,
        match="RCMR strict-loader-wire authority",
    ):
        four_arm.prepare_d106_four_arm_formal_handoff(
            state,
            rdce_runtime=object(),
            m0_baseline_qknn_bank=object(),
            m_da_baseline_qknn_bank=object(),
            m_head_rcmr_state=public_rcmr_state,
            m_joint_rcmr_state=public_rcmr_state,
        )


def test_nonformal_component_declaration_is_explicit_and_receipt_only() -> None:
    state = four_arm.build_d106_four_arm_state(_binding())
    declaration = four_arm.declare_d106_four_arm_components_nonformal(
        state,
        caller_component_receipts_sha256=_component_receipts(),
    )

    assert declaration.status == four_arm.COMPONENT_DECLARATION_STATUS
    assert declaration.formal_components_bound is False
    assert declaration.data_lineage_authority_verified is False
    assert declaration.query_execution_capability is False
    assert declaration.performance_scorer_attached is False
    assert declaration.runner_consumable is False
    assert dict(declaration.caller_component_receipts) == _component_receipts()
    assert declaration.receipt["declaration_scope"] == (
        "caller_provided_component_receipts_only"
    )
    assert declaration.receipt["formal_components_bound"] is False
    assert declaration.receipt["data_lineage_authority_verified"] is False

    audit = four_arm.audit_d106_four_arm_resources(declaration)
    assert audit["payload_kind"] == "non_formal_component_declaration"
    assert audit["formal_components_bound"] is False
    assert audit["actual_canonical_payload_bytes"] > 0
    assert (
        audit["actual_canonical_payload_bytes"]
        <= audit["canonical_payload_hard_cap_bytes"]
    )


def test_nonformal_declaration_cannot_be_consumed_by_formal_runner() -> None:
    declaration = four_arm.declare_d106_four_arm_components_nonformal(
        four_arm.build_d106_four_arm_state(_binding()),
        caller_component_receipts_sha256=_component_receipts(),
    )

    with pytest.raises(
        four_arm.D106FourArmAuthorityError,
        match="non-formal component declaration cannot be consumed by a formal runner",
    ):
        four_arm.reject_d106_four_arm_formal_runner_consumption(declaration)


def test_nonformal_declaration_requires_exact_caller_arm_labels() -> None:
    state = four_arm.build_d106_four_arm_state(_binding())

    with pytest.raises(four_arm.D106FourArmError, match="exactly M0"):
        four_arm.declare_d106_four_arm_components_nonformal(
            state,
            caller_component_receipts_sha256={"M0": _sha("a")},
        )
    with pytest.raises(four_arm.D106FourArmError, match="lowercase SHA256"):
        four_arm.declare_d106_four_arm_components_nonformal(
            state,
            caller_component_receipts_sha256={
                **_component_receipts(),
                "M_DA": "not-a-receipt",
            },
        )


def test_identity_factor_label_receipt_is_not_identity_view_authority() -> None:
    base = four_arm.build_d106_four_arm_state(_binding())
    same = four_arm.build_d106_four_arm_state(_binding())
    changed = four_arm.build_d106_four_arm_state(
        _binding(support_physical_root_sha256=_sha("6"))
    )

    assert (
        four_arm.derive_d106_four_arm_identity_factor_label_receipt(base)
        == four_arm.derive_d106_four_arm_identity_factor_label_receipt(same)
    )
    assert (
        four_arm.derive_d106_four_arm_identity_factor_label_receipt(base)
        != four_arm.derive_d106_four_arm_identity_factor_label_receipt(changed)
    )
    assert not hasattr(four_arm, "derive_d106_four_arm_identity_da_receipt")


def test_query_execution_and_performance_scoring_are_explicitly_refused() -> None:
    state = four_arm.build_d106_four_arm_state(_binding())
    declaration = four_arm.declare_d106_four_arm_components_nonformal(
        state,
        caller_component_receipts_sha256=_component_receipts(),
    )

    assert not hasattr(four_arm, "score_d106_four_arm_query")
    for value in (state, declaration):
        with pytest.raises(
            four_arm.D106FourArmQueryCapabilityError,
            match="no query capability",
        ):
            four_arm.reject_d106_four_arm_query_execution(
                value,
                opaque_query_id="query-17",
                query_truth="forbidden",
                batch_class_quota="forbidden",
            )


def test_query_validator_and_scenario_drift_are_request_only_not_verified() -> None:
    base = _binding()
    same = _binding()
    changed_query = _binding(query_physical_root_sha256=_sha("6"))
    changed_validator = _binding(validator_receipt_sha256=_sha("7"))
    changed_scenario = _binding(scenario_id="leo_rain_weak")
    states = tuple(
        four_arm.build_d106_four_arm_state(binding)
        for binding in (
            base,
            same,
            changed_query,
            changed_validator,
            changed_scenario,
        )
    )

    assert base.receipt_sha256 == same.receipt_sha256
    assert base.receipt_sha256 != changed_query.receipt_sha256
    assert base.receipt_sha256 != changed_validator.receipt_sha256
    assert base.receipt_sha256 != changed_scenario.receipt_sha256
    assert states[0].state_receipt_sha256 == states[1].state_receipt_sha256
    assert states[0].state_receipt_sha256 != states[2].state_receipt_sha256
    assert states[0].state_receipt_sha256 != states[3].state_receipt_sha256
    assert states[0].state_receipt_sha256 != states[4].state_receipt_sha256
    for state in states:
        _assert_request_only(state)
