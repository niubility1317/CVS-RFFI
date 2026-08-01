from __future__ import annotations

from collections.abc import Sequence
from dataclasses import FrozenInstanceError, replace

import pytest

import cvsrffi.stage2_d106_matrix_protocol as matrix_protocol
from cvsrffi.stage2_d106_matrix_protocol import (
    ARMS,
    D106MatrixArtifactReceipt,
    D106MatrixProtocolError,
    INCOMPLETE_FAIL_CLOSED,
    LEO_SCENARIOS,
    MATCHED_ARM_PAIR_COUNT,
    OUTER_JOB_COUNT,
    RECEIVERS,
    SCENARIO_ROW_COUNT,
    STATES,
    STATE_SURFACE_COUNT,
    STRUCTURAL_ID_COVERAGE_ONLY,
    STRUCTURAL_RECORD_COUNT,
    TARGET25_SEED,
    TARGET25_SLICES,
    audit_d106_matrix_structural_id_coverage,
    canonical_sha256,
    estimate_d106_matrix_resources,
    freeze_d106_matrix_protocol,
    reject_d106_matrix_artifact_completion,
    reject_d106_matrix_structural_coverage_consumption,
    validate_d106_matrix_protocol,
)


def _sha(character: str) -> str:
    return character * 64


class _OversizedSequence(Sequence[str]):
    """An input that must be rejected from len() without any item read."""

    def __len__(self) -> int:
        return matrix_protocol.MAX_COMPLETION_OBSERVATIONS + 1

    def __getitem__(self, index: int) -> str:
        raise AssertionError(f"cap check read item {index}")


class _OversizedJobSequence(Sequence[object]):
    """Plan construction must reject this by length before item access."""

    def __len__(self) -> int:
        return OUTER_JOB_COUNT + 1

    def __getitem__(self, index: int) -> object:
        raise AssertionError(f"plan cap check read item {index}")


class _EvilString(str):
    """A str subclass that made loose equality and membership accept evil-* tokens."""

    def __eq__(self, other: object) -> bool:
        del other
        return True

    def __ne__(self, other: object) -> bool:
        del other
        return False

    __hash__ = str.__hash__


def _all_surface_ids() -> tuple[str, ...]:
    return tuple(value.surface_id for value in freeze_d106_matrix_protocol().state_surfaces)


def _typed_artifact_claims() -> tuple[D106MatrixArtifactReceipt, ...]:
    plan = freeze_d106_matrix_protocol()
    return tuple(
        D106MatrixArtifactReceipt(
            surface_id=surface_id,
            matrix_receipt_sha256=plan.matrix_receipt_sha256,
            artifact_digest_sha256=_sha("a"),
        )
        for surface_id in _all_surface_ids()
    )


def test_frozen_matrix_has_exact_25_75_300_600_coverage() -> None:
    plan = freeze_d106_matrix_protocol()

    assert len(plan.jobs) == OUTER_JOB_COUNT == 25
    assert len(plan.scenario_rows) == SCENARIO_ROW_COUNT == 75
    assert len(plan.arm_pairs) == MATCHED_ARM_PAIR_COUNT == 300
    assert len(plan.state_surfaces) == STATE_SURFACE_COUNT == 600
    assert len(plan.jobs) + len(plan.scenario_rows) + len(plan.arm_pairs) + len(
        plan.state_surfaces
    ) == STRUCTURAL_RECORD_COUNT == 1000
    assert tuple(
        (job.receiver, job.k_shot, job.new_count) for job in plan.jobs
    ) == tuple(
        (receiver, k_shot, new_count)
        for receiver in RECEIVERS
        for k_shot, new_count in TARGET25_SLICES
    )
    assert {job.seed for job in plan.jobs} == {TARGET25_SEED}
    assert tuple(row.scenario for row in plan.scenario_rows) == LEO_SCENARIOS * 25
    assert tuple(pair.arm_id for pair in plan.arm_pairs) == ARMS * 75
    assert tuple(surface.state for surface in plan.state_surfaces) == STATES * 300
    assert len({job.job_id for job in plan.jobs}) == 25
    assert len({row.scenario_row_id for row in plan.scenario_rows}) == 75
    assert len({pair.arm_pair_id for pair in plan.arm_pairs}) == 300
    assert len({surface.surface_id for surface in plan.state_surfaces}) == 600


def test_order_receipt_and_all_sequences_are_deterministic_and_immutable() -> None:
    first = freeze_d106_matrix_protocol()
    second = freeze_d106_matrix_protocol()

    assert first == second
    assert first.matrix_receipt_sha256 == canonical_sha256(first.receipt_payload())
    assert tuple(value.index for value in first.jobs) == tuple(range(25))
    assert tuple(value.index for value in first.scenario_rows) == tuple(range(75))
    assert tuple(value.index for value in first.arm_pairs) == tuple(range(300))
    assert tuple(value.index for value in first.state_surfaces) == tuple(range(600))
    assert isinstance(first.jobs, tuple)
    assert isinstance(first.scenario_rows, tuple)
    assert isinstance(first.arm_pairs, tuple)
    assert isinstance(first.state_surfaces, tuple)
    with pytest.raises(FrozenInstanceError):
        first.seed = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        first.jobs[0].receiver = "tampered"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "tampered_value"),
    (
        ("clean_source_runtime_access", True),
        ("performance_values_allowed", True),
        ("query_truth_access", True),
        ("query_role_access", True),
        ("query_true_batch_class_count_access", True),
        ("query_class_quota_access", True),
        ("query_global_reassignment", True),
        ("query_fit_access", True),
        ("query_update_access", True),
        ("partial_favorable_selection", True),
        ("query_decision_policy", "batch_quota"),
    ),
)
def test_low_level_policy_tamper_and_rehashed_plan_fail_closed(
    field: str,
    tampered_value: object,
) -> None:
    plan = freeze_d106_matrix_protocol()
    object.__setattr__(plan.policy, field, tampered_value)
    object.__setattr__(
        plan,
        "matrix_receipt_sha256",
        canonical_sha256(plan.receipt_payload()),
    )

    with pytest.raises(D106MatrixProtocolError):
        validate_d106_matrix_protocol(plan)


@pytest.mark.parametrize(
    ("collection", "field"),
    (
        ("jobs", "job_id"),
        ("jobs", "receiver"),
        ("scenario_rows", "scenario_row_id"),
        ("scenario_rows", "job_id"),
        ("scenario_rows", "receiver"),
        ("scenario_rows", "scenario"),
        ("arm_pairs", "arm_pair_id"),
        ("arm_pairs", "scenario_row_id"),
        ("arm_pairs", "job_id"),
        ("arm_pairs", "receiver"),
        ("arm_pairs", "scenario"),
        ("arm_pairs", "arm_id"),
        ("state_surfaces", "surface_id"),
        ("state_surfaces", "arm_pair_id"),
        ("state_surfaces", "scenario_row_id"),
        ("state_surfaces", "job_id"),
        ("state_surfaces", "receiver"),
        ("state_surfaces", "scenario"),
        ("state_surfaces", "arm_id"),
        ("state_surfaces", "state"),
    ),
)
def test_record_construction_rejects_evil_str_subclasses(
    collection: str,
    field: str,
) -> None:
    record = getattr(freeze_d106_matrix_protocol(), collection)[0]

    with pytest.raises(D106MatrixProtocolError):
        replace(record, **{field: _EvilString(f"evil-{field}")})


@pytest.mark.parametrize(
    ("collection", "field"),
    (
        ("jobs", "scenario_row_ids"),
        ("scenario_rows", "arm_pair_ids"),
        ("arm_pairs", "state_surface_ids"),
    ),
)
def test_record_construction_rejects_evil_nested_id_subclasses(
    collection: str,
    field: str,
) -> None:
    record = getattr(freeze_d106_matrix_protocol(), collection)[0]
    identifiers = getattr(record, field)
    tampered = (_EvilString(f"evil-{field}"),) + identifiers[1:]

    with pytest.raises(D106MatrixProtocolError):
        replace(record, **{field: tampered})


@pytest.mark.parametrize(
    "field",
    ("schema", "protocol_schema", "matrix_receipt_sha256"),
)
def test_plan_construction_rejects_evil_scalar_str_subclasses(field: str) -> None:
    plan = freeze_d106_matrix_protocol()

    with pytest.raises(D106MatrixProtocolError):
        replace(plan, **{field: _EvilString(f"evil-{field}")})


@pytest.mark.parametrize("field", ("receivers", "scenarios", "states", "arms"))
def test_plan_construction_rejects_evil_enum_str_subclasses(field: str) -> None:
    plan = freeze_d106_matrix_protocol()
    values = getattr(plan, field)

    with pytest.raises(D106MatrixProtocolError):
        replace(plan, **{field: (_EvilString(f"evil-{field}"),) + values[1:]})


def test_policy_and_slice_construction_require_exact_builtin_types() -> None:
    plan = freeze_d106_matrix_protocol()

    with pytest.raises(D106MatrixProtocolError):
        replace(
            plan.policy,
            query_decision_policy=_EvilString("evil-policy"),
        )
    with pytest.raises(D106MatrixProtocolError):
        replace(plan, slices=((True, 5),) + plan.slices[1:])
    with pytest.raises(D106MatrixProtocolError):
        replace(plan.jobs[0], k_shot=True)


@pytest.mark.parametrize(
    ("collection", "field"),
    (
        ("jobs", "job_id"),
        ("jobs", "receiver"),
        ("jobs", "scenario_row_ids"),
        ("scenario_rows", "scenario_row_id"),
        ("scenario_rows", "job_id"),
        ("scenario_rows", "receiver"),
        ("scenario_rows", "scenario"),
        ("scenario_rows", "arm_pair_ids"),
        ("arm_pairs", "arm_pair_id"),
        ("arm_pairs", "scenario_row_id"),
        ("arm_pairs", "job_id"),
        ("arm_pairs", "receiver"),
        ("arm_pairs", "scenario"),
        ("arm_pairs", "arm_id"),
        ("arm_pairs", "state_surface_ids"),
        ("state_surfaces", "surface_id"),
        ("state_surfaces", "arm_pair_id"),
        ("state_surfaces", "scenario_row_id"),
        ("state_surfaces", "job_id"),
        ("state_surfaces", "receiver"),
        ("state_surfaces", "scenario"),
        ("state_surfaces", "arm_id"),
        ("state_surfaces", "state"),
    ),
)
def test_record_evil_str_low_level_tamper_survives_rehash_but_not_revalidation(
    collection: str,
    field: str,
) -> None:
    plan = freeze_d106_matrix_protocol()
    record = getattr(plan, collection)[0]
    original = getattr(record, field)
    tampered = (
        (_EvilString(f"evil-{field}"),) + original[1:]
        if type(original) is tuple
        else _EvilString(f"evil-{field}")
    )
    object.__setattr__(record, field, tampered)
    object.__setattr__(
        plan,
        "matrix_receipt_sha256",
        canonical_sha256(plan.receipt_payload()),
    )

    with pytest.raises(D106MatrixProtocolError, match="coverage/order"):
        validate_d106_matrix_protocol(plan)


@pytest.mark.parametrize(
    ("field", "tampered"),
    (
        ("schema", _EvilString("evil-schema")),
        ("protocol_schema", _EvilString("evil-protocol")),
        ("receivers", (_EvilString("evil-receiver"),) + RECEIVERS[1:]),
        ("slices", ((True, 5),) + TARGET25_SLICES[1:]),
        ("scenarios", (_EvilString("evil-scenario"),) + LEO_SCENARIOS[1:]),
        ("states", (_EvilString("evil-state"),) + STATES[1:]),
        ("arms", (_EvilString("evil-arm"),) + ARMS[1:]),
    ),
)
def test_top_level_evil_type_low_level_tamper_and_rehash_fail_closed(
    field: str,
    tampered: object,
) -> None:
    plan = freeze_d106_matrix_protocol()
    object.__setattr__(plan, field, tampered)
    object.__setattr__(
        plan,
        "matrix_receipt_sha256",
        canonical_sha256(plan.receipt_payload()),
    )

    with pytest.raises(D106MatrixProtocolError):
        validate_d106_matrix_protocol(plan)


def test_evil_policy_str_low_level_tamper_and_rehash_fail_closed() -> None:
    plan = freeze_d106_matrix_protocol()
    object.__setattr__(
        plan.policy,
        "query_decision_policy",
        _EvilString("evil-policy"),
    )
    object.__setattr__(
        plan,
        "matrix_receipt_sha256",
        canonical_sha256(plan.receipt_payload()),
    )

    with pytest.raises(D106MatrixProtocolError):
        validate_d106_matrix_protocol(plan)


def test_every_arm_pair_binds_same_row_before_and_after() -> None:
    plan = freeze_d106_matrix_protocol()
    surfaces = {value.surface_id: value for value in plan.state_surfaces}
    scenario_rows = {value.scenario_row_id: value for value in plan.scenario_rows}
    jobs = {value.job_id: value for value in plan.jobs}

    for pair in plan.arm_pairs:
        before, after = (surfaces[value] for value in pair.state_surface_ids)
        assert (before.state, after.state) == STATES
        assert before.registration_state == "BEFORE_REGISTRATION"
        assert after.registration_state == "AFTER_REGISTRATION"
        assert (
            before.job_id,
            before.scenario_row_id,
            before.arm_pair_id,
            before.receiver,
            before.seed,
            before.k_shot,
            before.new_count,
            before.scenario,
            before.arm_id,
        ) == (
            after.job_id,
            after.scenario_row_id,
            after.arm_pair_id,
            after.receiver,
            after.seed,
            after.k_shot,
            after.new_count,
            after.scenario,
            after.arm_id,
        )
        assert pair.scenario_row_id in jobs[pair.job_id].scenario_row_ids
        assert pair.arm_pair_id in scenario_rows[pair.scenario_row_id].arm_pair_ids


def test_plan_tamper_missing_duplicate_reorder_and_receipt_drift_fail_closed() -> None:
    plan = freeze_d106_matrix_protocol()

    with pytest.raises(D106MatrixProtocolError, match="coverage/order"):
        replace(plan, jobs=plan.jobs[:-1])
    with pytest.raises(D106MatrixProtocolError, match="coverage/order"):
        replace(
            plan,
            scenario_rows=plan.scenario_rows[:-1] + (plan.scenario_rows[0],),
        )
    with pytest.raises(D106MatrixProtocolError, match="coverage/order"):
        replace(
            plan,
            arm_pairs=(plan.arm_pairs[1], plan.arm_pairs[0]) + plan.arm_pairs[2:],
        )
    with pytest.raises(D106MatrixProtocolError, match="canonical receipt"):
        replace(plan, matrix_receipt_sha256="0" * 64)
    with pytest.raises(D106MatrixProtocolError, match="hard item cap"):
        replace(plan, jobs=_OversizedJobSequence())  # type: ignore[arg-type]


def test_complete_id_set_is_explicitly_structural_not_artifact_completion() -> None:
    plan = freeze_d106_matrix_protocol()
    coverage = audit_d106_matrix_structural_id_coverage(plan, _all_surface_ids())

    assert coverage.status == STRUCTURAL_ID_COVERAGE_ONLY
    assert coverage.completed_job_count == 25
    assert coverage.completed_scenario_row_count == 75
    assert coverage.completed_arm_pair_count == 300
    assert coverage.completed_surface_count == 600
    assert coverage.diagnostic_payload()["coverage_scope"] == (
        "surface_id_structure_only_no_artifact_verification"
    )
    assert coverage.diagnostic_payload()["runner_authority"] is False
    assert coverage.diagnostic_payload()["artifact_authority"] is False
    assert coverage.diagnostic_payload()["protocol_authority"] is False
    assert coverage.diagnostic_payload()["downstream_consumable"] is False
    assert coverage.diagnostic_payload()["artifact_receipt_verified"] is False
    assert coverage.diagnostic_payload()["artifact_digest_verified"] is False
    assert coverage.diagnostic_payload()["performance_values_read"] is False
    assert coverage.diagnostic_payload()["query_truth_access"] is False
    assert coverage.diagnostic_payload()["partial_favorable_selection"] is False
    assert "_D106MatrixStructuralCoverageDiagnostic" not in matrix_protocol.__all__
    with pytest.raises(D106MatrixProtocolError, match="cannot be consumed downstream"):
        reject_d106_matrix_structural_coverage_consumption(coverage)

    with pytest.raises(
        D106MatrixProtocolError,
        match="typed D106MatrixArtifactReceipt.*artifact digest.*matrix binding",
    ):
        reject_d106_matrix_artifact_completion(plan)


def test_typed_artifact_claims_with_digest_and_matrix_binding_still_fail_closed() -> None:
    plan = freeze_d106_matrix_protocol()
    claims = _typed_artifact_claims()

    with pytest.raises(
        D106MatrixProtocolError,
        match="no strict artifact verifier authority",
    ):
        reject_d106_matrix_artifact_completion(plan, claims)
    with pytest.raises(
        D106MatrixProtocolError,
        match="full typed artifact receipt coverage",
    ):
        reject_d106_matrix_artifact_completion(plan, claims[:-1])
    with pytest.raises(D106MatrixProtocolError, match="artifact matrix binding drift"):
        reject_d106_matrix_artifact_completion(
            plan,
            (replace(claims[0], matrix_receipt_sha256=_sha("b")),) + claims[1:],
        )
    with pytest.raises(D106MatrixProtocolError, match="artifact digest"):
        D106MatrixArtifactReceipt(
            surface_id=claims[0].surface_id,
            matrix_receipt_sha256=plan.matrix_receipt_sha256,
            artifact_digest_sha256="not-a-digest",
        )
    tampered = D106MatrixArtifactReceipt(
        surface_id=claims[0].surface_id,
        matrix_receipt_sha256=plan.matrix_receipt_sha256,
        artifact_digest_sha256=_sha("c"),
    )
    object.__setattr__(tampered, "artifact_digest_sha256", "not-a-digest")
    with pytest.raises(D106MatrixProtocolError, match="artifact digest"):
        reject_d106_matrix_artifact_completion(
            plan,
            (tampered,) + claims[1:],
        )


@pytest.mark.parametrize(
    "field",
    (
        "surface_id",
        "matrix_receipt_sha256",
        "artifact_digest_sha256",
        "artifact_kind",
    ),
)
def test_artifact_receipt_construction_rejects_all_evil_string_fields(
    field: str,
) -> None:
    claim = _typed_artifact_claims()[0]

    with pytest.raises(D106MatrixProtocolError):
        replace(claim, **{field: _EvilString(f"evil-{field}")})


@pytest.mark.parametrize(
    "field",
    (
        "surface_id",
        "matrix_receipt_sha256",
        "artifact_digest_sha256",
        "artifact_kind",
    ),
)
def test_artifact_evil_string_low_level_tamper_fails_gate_revalidation(
    field: str,
) -> None:
    plan = freeze_d106_matrix_protocol()
    claims = _typed_artifact_claims()
    object.__setattr__(
        claims[0],
        field,
        _EvilString(f"evil-{field}"),
    )

    with pytest.raises(D106MatrixProtocolError):
        reject_d106_matrix_artifact_completion(plan, claims)


def test_partial_duplicate_and_unexpected_id_coverage_fail_closed() -> None:
    plan = freeze_d106_matrix_protocol()
    all_ids = _all_surface_ids()

    partial = audit_d106_matrix_structural_id_coverage(plan, all_ids[:-1])
    assert partial.status == INCOMPLETE_FAIL_CLOSED
    assert partial.missing_surface_ids == (all_ids[-1],)
    assert partial.completed_job_count == 24
    with pytest.raises(D106MatrixProtocolError, match="STRUCTURAL_ID_COVERAGE_ONLY"):
        replace(partial, status=STRUCTURAL_ID_COVERAGE_ONLY)

    duplicate = audit_d106_matrix_structural_id_coverage(
        plan, all_ids + (all_ids[0],)
    )
    assert duplicate.status == INCOMPLETE_FAIL_CLOSED
    assert duplicate.duplicate_surface_ids == (all_ids[0],)

    unexpected = audit_d106_matrix_structural_id_coverage(
        plan, all_ids[:-1] + ("unexpected",)
    )
    assert unexpected.status == INCOMPLETE_FAIL_CLOSED
    assert unexpected.missing_surface_ids == (all_ids[-1],)
    assert unexpected.unexpected_surface_ids == ("unexpected",)


def test_replace_cannot_mint_artifact_authority_or_copy_a_token() -> None:
    plan = freeze_d106_matrix_protocol()
    partial = audit_d106_matrix_structural_id_coverage(
        plan, _all_surface_ids()[:-1]
    )
    rewritten = replace(
        partial,
        status=STRUCTURAL_ID_COVERAGE_ONLY,
        observed_record_count=STATE_SURFACE_COUNT,
        completed_surface_count=STATE_SURFACE_COUNT,
        completed_arm_pair_count=MATCHED_ARM_PAIR_COUNT,
        completed_scenario_row_count=SCENARIO_ROW_COUNT,
        completed_job_count=OUTER_JOB_COUNT,
        missing_surface_ids=(),
        duplicate_surface_ids=(),
        unexpected_surface_ids=(),
    )

    assert rewritten.status == STRUCTURAL_ID_COVERAGE_ONLY
    assert "_construction_token" not in rewritten.__dataclass_fields__
    assert rewritten.diagnostic_payload()["runner_authority"] is False
    assert rewritten.diagnostic_payload()["artifact_authority"] is False
    assert rewritten.diagnostic_payload()["protocol_authority"] is False
    assert rewritten.diagnostic_payload()["artifact_receipt_verified"] is False
    assert rewritten.diagnostic_payload()["artifact_digest_verified"] is False
    with pytest.raises(D106MatrixProtocolError, match="cannot be consumed downstream"):
        reject_d106_matrix_structural_coverage_consumption(rewritten)
    with pytest.raises(D106MatrixProtocolError, match="artifact receipts"):
        reject_d106_matrix_artifact_completion(plan, rewritten)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "authority_field",
    (
        "runner_authority",
        "artifact_authority",
        "protocol_authority",
        "downstream_consumable",
    ),
)
def test_structural_diagnostic_authority_tamper_fails_closed(
    authority_field: str,
) -> None:
    diagnostic = audit_d106_matrix_structural_id_coverage(
        freeze_d106_matrix_protocol(),
        _all_surface_ids(),
    )

    with pytest.raises(D106MatrixProtocolError, match="cannot carry authority"):
        replace(diagnostic, **{authority_field: True})
    object.__setattr__(diagnostic, authority_field, True)
    with pytest.raises(D106MatrixProtocolError, match="cannot carry authority"):
        diagnostic.diagnostic_payload()


def test_artifacts_complete_evil_status_is_rejected_direct_and_after_tamper() -> None:
    diagnostic = audit_d106_matrix_structural_id_coverage(
        freeze_d106_matrix_protocol(),
        _all_surface_ids(),
    )
    evil_status = _EvilString("ARTIFACTS_COMPLETE")

    with pytest.raises(D106MatrixProtocolError, match="status"):
        replace(diagnostic, status=evil_status)
    object.__setattr__(diagnostic, "status", evil_status)
    with pytest.raises(D106MatrixProtocolError, match="status"):
        diagnostic.diagnostic_payload()


def test_three_anomaly_lists_share_one_count_and_byte_budget() -> None:
    diagnostic = audit_d106_matrix_structural_id_coverage(
        freeze_d106_matrix_protocol(),
        _all_surface_ids()[:-1],
    )
    missing = tuple(f"missing-{index:04d}" for index in range(400))
    duplicates = tuple(f"duplicate-{index:04d}" for index in range(400))
    unexpected = tuple(f"unexpected-{index:04d}" for index in range(401))

    with pytest.raises(D106MatrixProtocolError, match="shared hard item cap"):
        replace(
            diagnostic,
            missing_surface_ids=missing,
            duplicate_surface_ids=duplicates,
            unexpected_surface_ids=unexpected,
        )


def test_completion_input_hard_cap_is_checked_before_item_access() -> None:
    plan = freeze_d106_matrix_protocol()

    with pytest.raises(D106MatrixProtocolError, match="hard item cap"):
        audit_d106_matrix_structural_id_coverage(plan, _OversizedSequence())
    with pytest.raises(D106MatrixProtocolError, match="structural ID byte cap"):
        audit_d106_matrix_structural_id_coverage(
            plan,
            _all_surface_ids()[:-1] + ("x" * 257,),
        )


def test_partial_anomalies_are_canonical_and_arrival_order_independent() -> None:
    plan = freeze_d106_matrix_protocol()
    all_ids = _all_surface_ids()
    first = audit_d106_matrix_structural_id_coverage(
        plan,
        all_ids[:-2] + ("unexpected-b", "unexpected-a"),
    )
    second = audit_d106_matrix_structural_id_coverage(
        plan,
        all_ids[:-2] + ("unexpected-a", "unexpected-b"),
    )

    assert first.status == second.status == INCOMPLETE_FAIL_CLOSED
    assert first.unexpected_surface_ids == second.unexpected_surface_ids == (
        "unexpected-a",
        "unexpected-b",
    )
    assert first.missing_surface_ids == second.missing_surface_ids
    assert first.diagnostic_sha256 == second.diagnostic_sha256


def test_negative_capabilities_and_resource_upper_bounds_are_explicit() -> None:
    plan = freeze_d106_matrix_protocol()
    policy = plan.policy
    assert policy.clean_source_runtime_access is False
    assert policy.performance_values_allowed is False
    assert policy.query_truth_access is False
    assert policy.query_role_access is False
    assert policy.query_true_batch_class_count_access is False
    assert policy.query_class_quota_access is False
    assert policy.query_global_reassignment is False
    assert policy.query_fit_access is False
    assert policy.query_update_access is False
    assert policy.partial_favorable_selection is False
    assert policy.query_decision_policy == "per_sample_all_registered_classes"

    estimate = estimate_d106_matrix_resources(plan)
    assert estimate.structural_record_count == 1000
    assert estimate.canonical_plan_bytes_exact <= (
        estimate.canonical_plan_hard_cap_bytes
    )
    assert estimate.canonical_plan_bytes_exact > 0
    assert estimate.primary_identifier_utf8_bytes_exact > 0
    assert estimate.completion_observation_hard_cap >= 600
    assert estimate.completion_identifier_utf8_hard_cap_bytes == 1200 * 256
    assert estimate.anomaly_identifier_shared_hard_cap == 1200
    assert estimate.anomaly_identifier_utf8_shared_hard_cap_bytes == 1200 * 256
    assert estimate.performance_value_fields == 0
    assert estimate.truth_or_query_role_fields == 0
    assert "RSS" in estimate.unaccounted_overhead
    validate_d106_matrix_protocol(plan)
