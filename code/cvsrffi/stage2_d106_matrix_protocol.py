"""Frozen, truth-free structural protocol for the D106 Target25 matrix.

This module enumerates identities and structural ID coverage only. Structural
coverage outputs are non-authoritative diagnostics and cannot be consumed as
runner, artifact, or protocol evidence. The module has no dataset, predictor,
scorer, arm implementation, filesystem, or N607 dependency.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Any, NoReturn


SCHEMA = "cvs.phase2.d106.matrix_protocol.v1"
PROTOCOL_SCHEMA = "p2_min_v1"
TARGET25_SEED = 713102
RECEIVERS = ("20-1", "3-19", "7-14", "7-7", "8-8")
TARGET25_SLICES = ((10, 5), (10, 10), (10, 20), (5, 20), (1, 20))
LEO_SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
STATES = ("before", "after")
ARMS = ("M0", "M_DA", "M_HEAD", "M_JOINT")

OUTER_JOB_COUNT = 25
SCENARIO_ROW_COUNT = 75
MATCHED_ARM_PAIR_COUNT = 300
STATE_SURFACE_COUNT = 600
STRUCTURAL_RECORD_COUNT = (
    OUTER_JOB_COUNT
    + SCENARIO_ROW_COUNT
    + MATCHED_ARM_PAIR_COUNT
    + STATE_SURFACE_COUNT
)

STRUCTURAL_ID_COVERAGE_ONLY = "STRUCTURAL_ID_COVERAGE_ONLY"
INCOMPLETE_FAIL_CLOSED = "INCOMPLETE_FAIL_CLOSED"
MAX_IDENTIFIER_UTF8_BYTES = 256
MAX_COMPLETION_OBSERVATIONS = 1200
MAX_COMPLETION_IDENTIFIER_UTF8_BYTES = (
    MAX_COMPLETION_OBSERVATIONS * MAX_IDENTIFIER_UTF8_BYTES
)
MAX_CANONICAL_PLAN_BYTES = 1_048_576


class D106MatrixProtocolError(ValueError):
    """Raised when the frozen D106 matrix or its coverage drifts."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Return the canonical JSON SHA256 used by D106 structural receipts."""

    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_text(value: Any, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise D106MatrixProtocolError(f"{name} must be non-empty trimmed text")
    if len(value.encode("utf-8")) > MAX_IDENTIFIER_UTF8_BYTES:
        raise D106MatrixProtocolError(f"{name} exceeds the structural ID byte cap")
    return value


def _require_sha256(value: Any, name: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise D106MatrixProtocolError(f"{name} must be a lowercase SHA256")
    if any(character not in "0123456789abcdef" for character in value):
        raise D106MatrixProtocolError(f"{name} must be a lowercase SHA256")
    return value


def _bounded_sequence_length(value: Any, name: str, maximum_items: int) -> int:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise D106MatrixProtocolError(f"{name} must be an indexable sequence")
    length = len(value)
    if type(length) is not int or length < 0:
        raise D106MatrixProtocolError(f"{name} has an invalid sequence length")
    if length > maximum_items:
        raise D106MatrixProtocolError(f"{name} exceeds its hard item cap")
    return length


def _bounded_sequence(
    value: Any, name: str, maximum_items: int
) -> tuple[Any, ...]:
    """Read at most a declared count before touching sequence elements."""

    length = _bounded_sequence_length(value, name, maximum_items)
    values: list[Any] = []
    for index in range(length):
        try:
            values.append(value[index])
        except (IndexError, KeyError) as error:
            raise D106MatrixProtocolError(
                f"{name} length/index contract drift"
            ) from error
    return tuple(values)


def _bounded_text_sequence(
    value: Any,
    name: str,
    maximum_items: int,
    maximum_total_utf8_bytes: int,
) -> tuple[str, ...]:
    values = _bounded_sequence(value, name, maximum_items)
    total_bytes = 0
    texts: list[str] = []
    for item in values:
        text = _require_text(item, name)
        total_bytes += len(text.encode("utf-8"))
        if total_bytes > maximum_total_utf8_bytes:
            raise D106MatrixProtocolError(f"{name} exceeds its aggregate UTF-8 byte cap")
        texts.append(text)
    return tuple(texts)


def _require_index(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise D106MatrixProtocolError(f"{name} must be a non-negative integer")
    return value


def _require_exact_string(value: Any, name: str, expected: str) -> str:
    """Require a builtin string before comparing it with a canonical literal."""

    if type(expected) is not str:
        raise D106MatrixProtocolError(f"{name} canonical literal is not builtin str")
    if type(value) is not str or value != expected:
        raise D106MatrixProtocolError(f"{name} exact string drift")
    return value


def _require_exact_string_member(
    value: Any,
    name: str,
    expected_values: Sequence[str],
) -> str:
    """Reject string subclasses before checking a frozen enumeration."""

    if type(value) is not str:
        raise D106MatrixProtocolError(f"{name} must be builtin str")
    for expected in expected_values:
        if type(expected) is not str:
            raise D106MatrixProtocolError(
                f"{name} canonical enumeration contains a non-builtin str"
            )
        if value == expected:
            return value
    raise D106MatrixProtocolError(f"{name} is outside the frozen enumeration")


def _require_exact_integer(value: Any, name: str, expected: int) -> int:
    if type(expected) is not int or type(value) is not int or value != expected:
        raise D106MatrixProtocolError(f"{name} exact integer drift")
    return value


def _require_frozen_slice(k_shot: Any, new_count: Any, name: str) -> None:
    if type(k_shot) is not int or type(new_count) is not int:
        raise D106MatrixProtocolError(f"{name} must contain builtin integers")
    for expected_k, expected_new in TARGET25_SLICES:
        if k_shot == expected_k and new_count == expected_new:
            return
    raise D106MatrixProtocolError(f"{name} drift")


def _require_exact_string_sequence(
    value: Any,
    name: str,
    expected_values: Sequence[str],
) -> tuple[str, ...]:
    values = _bounded_sequence(value, name, len(expected_values))
    if len(values) != len(expected_values):
        raise D106MatrixProtocolError(f"{name} exact sequence length drift")
    validated: list[str] = []
    for index, expected in enumerate(expected_values):
        validated.append(
            _require_exact_string(values[index], f"{name}[{index}]", expected)
        )
    return tuple(validated)


def _require_exact_slice_sequence(
    value: Any,
    name: str,
    expected_values: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    values = _bounded_sequence(value, name, len(expected_values))
    if len(values) != len(expected_values):
        raise D106MatrixProtocolError(f"{name} exact sequence length drift")
    validated: list[tuple[int, int]] = []
    for index, expected in enumerate(expected_values):
        pair = _bounded_sequence(values[index], f"{name}[{index}]", 2)
        if len(pair) != 2:
            raise D106MatrixProtocolError(f"{name}[{index}] must contain two integers")
        k_shot = _require_exact_integer(
            pair[0], f"{name}[{index}].k_shot", expected[0]
        )
        new_count = _require_exact_integer(
            pair[1], f"{name}[{index}].new_count", expected[1]
        )
        validated.append((k_shot, new_count))
    return tuple(validated)


def _safe_receiver(receiver: str) -> str:
    return _require_exact_string_member(
        receiver, "receiver", RECEIVERS
    ).replace("-", "_")


def _job_id(receiver: str, k_shot: int, new_count: int) -> str:
    return (
        f"d106-rx-{_safe_receiver(receiver)}__seed-{TARGET25_SEED}"
        f"__k-{k_shot}__new-{new_count}"
    )


def _scenario_row_id(job_id: str, scenario: str) -> str:
    return (
        f"{_require_text(job_id, 'job ID')}__scenario-"
        f"{_require_exact_string_member(scenario, 'scenario', LEO_SCENARIOS)}"
    )


def _arm_pair_id(scenario_row_id: str, arm_id: str) -> str:
    return (
        f"{_require_text(scenario_row_id, 'scenario-row ID')}__arm-"
        f"{_require_exact_string_member(arm_id, 'arm', ARMS)}"
    )


def _surface_id(arm_pair_id: str, state: str) -> str:
    return (
        f"{_require_text(arm_pair_id, 'arm-pair ID')}__state-"
        f"{_require_exact_string_member(state, 'state', STATES)}"
    )


@dataclass(frozen=True, slots=True)
class D106MatrixAccessPolicy:
    """Frozen negative capability declaration for the structural matrix."""

    clean_source_runtime_access: bool = False
    performance_values_allowed: bool = False
    query_truth_access: bool = False
    query_role_access: bool = False
    query_true_batch_class_count_access: bool = False
    query_class_quota_access: bool = False
    query_global_reassignment: bool = False
    query_fit_access: bool = False
    query_update_access: bool = False
    partial_favorable_selection: bool = False
    query_decision_policy: str = "per_sample_all_registered_classes"

    def __post_init__(self) -> None:
        booleans = (
            self.clean_source_runtime_access,
            self.performance_values_allowed,
            self.query_truth_access,
            self.query_role_access,
            self.query_true_batch_class_count_access,
            self.query_class_quota_access,
            self.query_global_reassignment,
            self.query_fit_access,
            self.query_update_access,
            self.partial_favorable_selection,
        )
        if any(value is not False for value in booleans):
            raise D106MatrixProtocolError("D106 matrix negative capability drift")
        _require_exact_string(
            self.query_decision_policy,
            "D106 query decision policy",
            "per_sample_all_registered_classes",
        )

    def receipt_payload(self) -> dict[str, Any]:
        return {
            "clean_source_runtime_access": self.clean_source_runtime_access,
            "partial_favorable_selection": self.partial_favorable_selection,
            "performance_values_allowed": self.performance_values_allowed,
            "query_class_quota_access": self.query_class_quota_access,
            "query_decision_policy": self.query_decision_policy,
            "query_fit_access": self.query_fit_access,
            "query_global_reassignment": self.query_global_reassignment,
            "query_role_access": self.query_role_access,
            "query_true_batch_class_count_access": (
                self.query_true_batch_class_count_access
            ),
            "query_truth_access": self.query_truth_access,
            "query_update_access": self.query_update_access,
        }


def _validate_d106_matrix_access_policy(value: Any) -> D106MatrixAccessPolicy:
    """Re-read every policy field; frozen dataclasses are not an authority boundary."""

    if type(value) is not D106MatrixAccessPolicy:
        raise D106MatrixProtocolError("exact D106MatrixAccessPolicy required")
    revalidated = D106MatrixAccessPolicy(
        clean_source_runtime_access=value.clean_source_runtime_access,
        performance_values_allowed=value.performance_values_allowed,
        query_truth_access=value.query_truth_access,
        query_role_access=value.query_role_access,
        query_true_batch_class_count_access=(
            value.query_true_batch_class_count_access
        ),
        query_class_quota_access=value.query_class_quota_access,
        query_global_reassignment=value.query_global_reassignment,
        query_fit_access=value.query_fit_access,
        query_update_access=value.query_update_access,
        partial_favorable_selection=value.partial_favorable_selection,
        query_decision_policy=value.query_decision_policy,
    )
    return revalidated


@dataclass(frozen=True, slots=True)
class D106MatrixJob:
    index: int
    job_id: str
    receiver: str
    seed: int
    k_shot: int
    new_count: int
    scenario_row_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_index(self.index, "job index")
        _require_text(self.job_id, "job ID")
        _require_exact_string_member(self.receiver, "job receiver", RECEIVERS)
        _require_exact_integer(self.seed, "job seed", TARGET25_SEED)
        _require_frozen_slice(self.k_shot, self.new_count, "job slice")
        scenario_ids = _bounded_text_sequence(
            self.scenario_row_ids,
            "scenario-row ID",
            len(LEO_SCENARIOS),
            len(LEO_SCENARIOS) * MAX_IDENTIFIER_UTF8_BYTES,
        )
        if len(scenario_ids) != len(LEO_SCENARIOS):
            raise D106MatrixProtocolError("job must bind exactly three scenario rows")
        if len(set(scenario_ids)) != len(scenario_ids):
            raise D106MatrixProtocolError("job scenario-row IDs must be unique")
        object.__setattr__(self, "scenario_row_ids", scenario_ids)

    def receipt_payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "job_id": self.job_id,
            "receiver": self.receiver,
            "seed": self.seed,
            "k_shot": self.k_shot,
            "new_count": self.new_count,
            "scenario_row_ids": list(self.scenario_row_ids),
        }


@dataclass(frozen=True, slots=True)
class D106ScenarioRow:
    index: int
    scenario_row_id: str
    job_id: str
    receiver: str
    seed: int
    k_shot: int
    new_count: int
    scenario: str
    arm_pair_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_index(self.index, "scenario-row index")
        _require_text(self.scenario_row_id, "scenario-row ID")
        _require_text(self.job_id, "job ID")
        _require_exact_string_member(
            self.receiver, "scenario-row receiver", RECEIVERS
        )
        _require_exact_string_member(
            self.scenario, "scenario-row scenario", LEO_SCENARIOS
        )
        _require_exact_integer(self.seed, "scenario-row seed", TARGET25_SEED)
        _require_frozen_slice(
            self.k_shot, self.new_count, "scenario-row slice"
        )
        pair_ids = _bounded_text_sequence(
            self.arm_pair_ids,
            "arm-pair ID",
            len(ARMS),
            len(ARMS) * MAX_IDENTIFIER_UTF8_BYTES,
        )
        if len(pair_ids) != len(ARMS) or len(set(pair_ids)) != len(pair_ids):
            raise D106MatrixProtocolError("scenario row must bind four unique arm pairs")
        object.__setattr__(self, "arm_pair_ids", pair_ids)

    def receipt_payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "scenario_row_id": self.scenario_row_id,
            "job_id": self.job_id,
            "receiver": self.receiver,
            "seed": self.seed,
            "k_shot": self.k_shot,
            "new_count": self.new_count,
            "scenario": self.scenario,
            "arm_pair_ids": list(self.arm_pair_ids),
        }


@dataclass(frozen=True, slots=True)
class D106MatchedArmPair:
    index: int
    arm_pair_id: str
    scenario_row_id: str
    job_id: str
    receiver: str
    seed: int
    k_shot: int
    new_count: int
    scenario: str
    arm_id: str
    state_surface_ids: tuple[str, str]

    def __post_init__(self) -> None:
        _require_index(self.index, "arm-pair index")
        for value, name in (
            (self.arm_pair_id, "arm-pair ID"),
            (self.scenario_row_id, "scenario-row ID"),
            (self.job_id, "job ID"),
        ):
            _require_text(value, name)
        _require_exact_string_member(self.receiver, "arm-pair receiver", RECEIVERS)
        _require_exact_string_member(
            self.scenario, "arm-pair scenario", LEO_SCENARIOS
        )
        _require_exact_string_member(self.arm_id, "arm-pair arm", ARMS)
        _require_exact_integer(self.seed, "arm-pair seed", TARGET25_SEED)
        _require_frozen_slice(self.k_shot, self.new_count, "arm-pair slice")
        surfaces = _bounded_text_sequence(
            self.state_surface_ids,
            "state-surface ID",
            len(STATES),
            len(STATES) * MAX_IDENTIFIER_UTF8_BYTES,
        )
        if len(surfaces) != len(STATES) or len(set(surfaces)) != len(surfaces):
            raise D106MatrixProtocolError("arm pair must bind one before/after surface")
        object.__setattr__(self, "state_surface_ids", surfaces)

    def receipt_payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "arm_pair_id": self.arm_pair_id,
            "scenario_row_id": self.scenario_row_id,
            "job_id": self.job_id,
            "receiver": self.receiver,
            "seed": self.seed,
            "k_shot": self.k_shot,
            "new_count": self.new_count,
            "scenario": self.scenario,
            "arm_id": self.arm_id,
            "state_surface_ids": list(self.state_surface_ids),
        }


@dataclass(frozen=True, slots=True)
class D106StateSurface:
    index: int
    surface_id: str
    arm_pair_id: str
    scenario_row_id: str
    job_id: str
    receiver: str
    seed: int
    k_shot: int
    new_count: int
    scenario: str
    arm_id: str
    state: str

    def __post_init__(self) -> None:
        _require_index(self.index, "state-surface index")
        for value, name in (
            (self.surface_id, "state-surface ID"),
            (self.arm_pair_id, "arm-pair ID"),
            (self.scenario_row_id, "scenario-row ID"),
            (self.job_id, "job ID"),
        ):
            _require_text(value, name)
        _require_exact_string_member(
            self.receiver, "state-surface receiver", RECEIVERS
        )
        _require_exact_string_member(
            self.scenario, "state-surface scenario", LEO_SCENARIOS
        )
        _require_exact_string_member(self.arm_id, "state-surface arm", ARMS)
        _require_exact_string_member(self.state, "state-surface state", STATES)
        _require_exact_integer(self.seed, "state-surface seed", TARGET25_SEED)
        _require_frozen_slice(self.k_shot, self.new_count, "state-surface slice")

    @property
    def registration_state(self) -> str:
        return (
            "BEFORE_REGISTRATION"
            if self.state == "before"
            else "AFTER_REGISTRATION"
        )

    def receipt_payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "surface_id": self.surface_id,
            "arm_pair_id": self.arm_pair_id,
            "scenario_row_id": self.scenario_row_id,
            "job_id": self.job_id,
            "receiver": self.receiver,
            "seed": self.seed,
            "k_shot": self.k_shot,
            "new_count": self.new_count,
            "scenario": self.scenario,
            "arm_id": self.arm_id,
            "state": self.state,
            "registration_state": self.registration_state,
        }


@dataclass(frozen=True, slots=True)
class _D106MatrixComponents:
    jobs: tuple[D106MatrixJob, ...]
    scenario_rows: tuple[D106ScenarioRow, ...]
    arm_pairs: tuple[D106MatchedArmPair, ...]
    state_surfaces: tuple[D106StateSurface, ...]


def _enumerate_components() -> _D106MatrixComponents:
    jobs: list[D106MatrixJob] = []
    scenario_rows: list[D106ScenarioRow] = []
    arm_pairs: list[D106MatchedArmPair] = []
    state_surfaces: list[D106StateSurface] = []
    for receiver in RECEIVERS:
        for k_shot, new_count in TARGET25_SLICES:
            job_id = _job_id(receiver, k_shot, new_count)
            job_scenario_ids: list[str] = []
            for scenario in LEO_SCENARIOS:
                scenario_id = _scenario_row_id(job_id, scenario)
                job_scenario_ids.append(scenario_id)
                scenario_pair_ids: list[str] = []
                for arm_id in ARMS:
                    pair_id = _arm_pair_id(scenario_id, arm_id)
                    scenario_pair_ids.append(pair_id)
                    surface_ids: list[str] = []
                    for state in STATES:
                        surface_id = _surface_id(pair_id, state)
                        surface_ids.append(surface_id)
                        state_surfaces.append(
                            D106StateSurface(
                                index=len(state_surfaces),
                                surface_id=surface_id,
                                arm_pair_id=pair_id,
                                scenario_row_id=scenario_id,
                                job_id=job_id,
                                receiver=receiver,
                                seed=TARGET25_SEED,
                                k_shot=k_shot,
                                new_count=new_count,
                                scenario=scenario,
                                arm_id=arm_id,
                                state=state,
                            )
                        )
                    arm_pairs.append(
                        D106MatchedArmPair(
                            index=len(arm_pairs),
                            arm_pair_id=pair_id,
                            scenario_row_id=scenario_id,
                            job_id=job_id,
                            receiver=receiver,
                            seed=TARGET25_SEED,
                            k_shot=k_shot,
                            new_count=new_count,
                            scenario=scenario,
                            arm_id=arm_id,
                            state_surface_ids=(surface_ids[0], surface_ids[1]),
                        )
                    )
                scenario_rows.append(
                    D106ScenarioRow(
                        index=len(scenario_rows),
                        scenario_row_id=scenario_id,
                        job_id=job_id,
                        receiver=receiver,
                        seed=TARGET25_SEED,
                        k_shot=k_shot,
                        new_count=new_count,
                        scenario=scenario,
                        arm_pair_ids=tuple(scenario_pair_ids),
                    )
                )
            jobs.append(
                D106MatrixJob(
                    index=len(jobs),
                    job_id=job_id,
                    receiver=receiver,
                    seed=TARGET25_SEED,
                    k_shot=k_shot,
                    new_count=new_count,
                    scenario_row_ids=tuple(job_scenario_ids),
                )
            )
    return _D106MatrixComponents(
        jobs=tuple(jobs),
        scenario_rows=tuple(scenario_rows),
        arm_pairs=tuple(arm_pairs),
        state_surfaces=tuple(state_surfaces),
    )


def _validate_component_record_exact(
    value: Any,
    expected: Any,
    record_type: type[Any],
    name: str,
    *,
    integer_fields: Sequence[str],
    string_fields: Sequence[str],
    string_sequence_fields: Sequence[str] = (),
) -> Any:
    """Compare one record with its canonical enumeration field by field."""

    if type(value) is not record_type or type(expected) is not record_type:
        raise D106MatrixProtocolError(f"{name} exact record type drift")
    for field in integer_fields:
        _require_exact_integer(
            getattr(value, field),
            f"{name}.{field}",
            getattr(expected, field),
        )
    for field in string_fields:
        _require_exact_string(
            getattr(value, field),
            f"{name}.{field}",
            getattr(expected, field),
        )
    for field in string_sequence_fields:
        _require_exact_string_sequence(
            getattr(value, field),
            f"{name}.{field}",
            getattr(expected, field),
        )
    return value


def _validate_component_sequence_exact(
    value: Any,
    expected_values: Sequence[Any],
    record_type: type[Any],
    name: str,
    *,
    integer_fields: Sequence[str],
    string_fields: Sequence[str],
    string_sequence_fields: Sequence[str] = (),
) -> tuple[Any, ...]:
    values = _bounded_sequence(value, name, len(expected_values))
    if len(values) != len(expected_values):
        raise D106MatrixProtocolError(
            "D106 matrix coverage/order/duplicate/missing closure drift"
        )
    validated: list[Any] = []
    for index, expected in enumerate(expected_values):
        try:
            validated.append(
                _validate_component_record_exact(
                    values[index],
                    expected,
                    record_type,
                    f"{name}[{index}]",
                    integer_fields=integer_fields,
                    string_fields=string_fields,
                    string_sequence_fields=string_sequence_fields,
                )
            )
        except D106MatrixProtocolError as error:
            raise D106MatrixProtocolError(
                "D106 matrix coverage/order/duplicate/missing closure drift"
            ) from error
    return tuple(validated)


@dataclass(frozen=True, slots=True)
class D106MatrixPlan:
    schema: str
    protocol_schema: str
    seed: int
    receivers: tuple[str, ...]
    slices: tuple[tuple[int, int], ...]
    scenarios: tuple[str, ...]
    states: tuple[str, ...]
    arms: tuple[str, ...]
    policy: D106MatrixAccessPolicy
    jobs: tuple[D106MatrixJob, ...]
    scenario_rows: tuple[D106ScenarioRow, ...]
    arm_pairs: tuple[D106MatchedArmPair, ...]
    state_surfaces: tuple[D106StateSurface, ...]
    matrix_receipt_sha256: str

    def __post_init__(self) -> None:
        for name, maximum_items in (
            ("receivers", len(RECEIVERS)),
            ("slices", len(TARGET25_SLICES)),
            ("scenarios", len(LEO_SCENARIOS)),
            ("states", len(STATES)),
            ("arms", len(ARMS)),
            ("jobs", OUTER_JOB_COUNT),
            ("scenario_rows", SCENARIO_ROW_COUNT),
            ("arm_pairs", MATCHED_ARM_PAIR_COUNT),
            ("state_surfaces", STATE_SURFACE_COUNT),
        ):
            object.__setattr__(
                self,
                name,
                _bounded_sequence(getattr(self, name), name, maximum_items),
            )
        _validate_d106_matrix_protocol(self)

    def receipt_payload(self) -> dict[str, Any]:
        return _matrix_payload(
            policy=self.policy,
            jobs=self.jobs,
            scenario_rows=self.scenario_rows,
            arm_pairs=self.arm_pairs,
            state_surfaces=self.state_surfaces,
        )


def _matrix_payload(
    *,
    policy: D106MatrixAccessPolicy,
    jobs: Sequence[D106MatrixJob],
    scenario_rows: Sequence[D106ScenarioRow],
    arm_pairs: Sequence[D106MatchedArmPair],
    state_surfaces: Sequence[D106StateSurface],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "protocol_schema": PROTOCOL_SCHEMA,
        "seed": TARGET25_SEED,
        "receivers": list(RECEIVERS),
        "target25_slices": [list(value) for value in TARGET25_SLICES],
        "leo_scenarios": list(LEO_SCENARIOS),
        "states": list(STATES),
        "arms": list(ARMS),
        "outer_job_count": OUTER_JOB_COUNT,
        "scenario_row_count": SCENARIO_ROW_COUNT,
        "matched_arm_pair_count": MATCHED_ARM_PAIR_COUNT,
        "state_surface_count": STATE_SURFACE_COUNT,
        "policy": policy.receipt_payload(),
        "jobs": [value.receipt_payload() for value in jobs],
        "scenario_rows": [value.receipt_payload() for value in scenario_rows],
        "arm_pairs": [value.receipt_payload() for value in arm_pairs],
        "state_surfaces": [value.receipt_payload() for value in state_surfaces],
    }


def _validate_d106_matrix_protocol(plan: D106MatrixPlan) -> None:
    if type(plan) is not D106MatrixPlan:
        raise D106MatrixProtocolError("exact D106MatrixPlan required")
    canonical_policy = _validate_d106_matrix_access_policy(plan.policy)
    _require_exact_string(plan.schema, "matrix schema", SCHEMA)
    _require_exact_string(
        plan.protocol_schema, "matrix protocol schema", PROTOCOL_SCHEMA
    )
    _require_exact_integer(plan.seed, "matrix seed", TARGET25_SEED)
    _require_exact_string_sequence(plan.receivers, "matrix receivers", RECEIVERS)
    _require_exact_slice_sequence(plan.slices, "matrix slices", TARGET25_SLICES)
    _require_exact_string_sequence(
        plan.scenarios, "matrix scenarios", LEO_SCENARIOS
    )
    _require_exact_string_sequence(plan.states, "matrix states", STATES)
    _require_exact_string_sequence(plan.arms, "matrix arms", ARMS)
    expected = _enumerate_components()
    jobs = _validate_component_sequence_exact(
        plan.jobs,
        expected.jobs,
        D106MatrixJob,
        "jobs",
        integer_fields=("index", "seed", "k_shot", "new_count"),
        string_fields=("job_id", "receiver"),
        string_sequence_fields=("scenario_row_ids",),
    )
    scenario_rows = _validate_component_sequence_exact(
        plan.scenario_rows,
        expected.scenario_rows,
        D106ScenarioRow,
        "scenario_rows",
        integer_fields=("index", "seed", "k_shot", "new_count"),
        string_fields=("scenario_row_id", "job_id", "receiver", "scenario"),
        string_sequence_fields=("arm_pair_ids",),
    )
    arm_pairs = _validate_component_sequence_exact(
        plan.arm_pairs,
        expected.arm_pairs,
        D106MatchedArmPair,
        "arm_pairs",
        integer_fields=("index", "seed", "k_shot", "new_count"),
        string_fields=(
            "arm_pair_id",
            "scenario_row_id",
            "job_id",
            "receiver",
            "scenario",
            "arm_id",
        ),
        string_sequence_fields=("state_surface_ids",),
    )
    state_surfaces = _validate_component_sequence_exact(
        plan.state_surfaces,
        expected.state_surfaces,
        D106StateSurface,
        "state_surfaces",
        integer_fields=("index", "seed", "k_shot", "new_count"),
        string_fields=(
            "surface_id",
            "arm_pair_id",
            "scenario_row_id",
            "job_id",
            "receiver",
            "scenario",
            "arm_id",
            "state",
        ),
    )
    payload = _matrix_payload(
        policy=canonical_policy,
        jobs=jobs,
        scenario_rows=scenario_rows,
        arm_pairs=arm_pairs,
        state_surfaces=state_surfaces,
    )
    if len(_canonical_bytes(payload)) > MAX_CANONICAL_PLAN_BYTES:
        raise D106MatrixProtocolError("D106 canonical matrix exceeds its byte cap")
    _require_exact_string(
        plan.matrix_receipt_sha256,
        "D106 matrix canonical receipt",
        canonical_sha256(payload),
    )


def validate_d106_matrix_protocol(plan: D106MatrixPlan) -> None:
    """Revalidate exact coverage, deterministic order, and canonical receipt."""

    _validate_d106_matrix_protocol(plan)


def freeze_d106_matrix_protocol() -> D106MatrixPlan:
    """Enumerate the single frozen 25-job D106 structural matrix."""

    components = _enumerate_components()
    policy = D106MatrixAccessPolicy()
    payload = _matrix_payload(
        policy=policy,
        jobs=components.jobs,
        scenario_rows=components.scenario_rows,
        arm_pairs=components.arm_pairs,
        state_surfaces=components.state_surfaces,
    )
    return D106MatrixPlan(
        schema=SCHEMA,
        protocol_schema=PROTOCOL_SCHEMA,
        seed=TARGET25_SEED,
        receivers=RECEIVERS,
        slices=TARGET25_SLICES,
        scenarios=LEO_SCENARIOS,
        states=STATES,
        arms=ARMS,
        policy=policy,
        jobs=components.jobs,
        scenario_rows=components.scenario_rows,
        arm_pairs=components.arm_pairs,
        state_surfaces=components.state_surfaces,
        matrix_receipt_sha256=canonical_sha256(payload),
    )


@dataclass(frozen=True, slots=True)
class D106MatrixArtifactReceipt:
    """Typed artifact-claim shape; this structural module cannot verify it."""

    surface_id: str
    matrix_receipt_sha256: str
    artifact_digest_sha256: str
    artifact_kind: str = "immutable_prediction_artifact"

    def __post_init__(self) -> None:
        _require_text(self.surface_id, "artifact receipt surface ID")
        _require_sha256(self.matrix_receipt_sha256, "artifact receipt matrix binding")
        _require_sha256(self.artifact_digest_sha256, "artifact digest")
        _require_exact_string(
            self.artifact_kind,
            "artifact receipt kind",
            "immutable_prediction_artifact",
        )


@dataclass(frozen=True, slots=True)
class _D106MatrixStructuralCoverageDiagnostic:
    """Private ID-only diagnostic with no runner, artifact, or protocol authority."""

    matrix_receipt_sha256: str
    status: str
    expected_surface_count: int
    observed_record_count: int
    completed_surface_count: int
    completed_arm_pair_count: int
    completed_scenario_row_count: int
    completed_job_count: int
    missing_surface_ids: tuple[str, ...]
    duplicate_surface_ids: tuple[str, ...]
    unexpected_surface_ids: tuple[str, ...]
    runner_authority: bool = False
    artifact_authority: bool = False
    protocol_authority: bool = False
    downstream_consumable: bool = False

    def __post_init__(self) -> None:
        _require_sha256(self.matrix_receipt_sha256, "coverage receipt matrix binding")
        if (
            self.runner_authority is not False
            or self.artifact_authority is not False
            or self.protocol_authority is not False
            or self.downstream_consumable is not False
        ):
            raise D106MatrixProtocolError(
                "structural coverage diagnostic cannot carry authority"
            )
        status = _require_exact_string_member(
            self.status,
            "coverage receipt status",
            (STRUCTURAL_ID_COVERAGE_ONLY, INCOMPLETE_FAIL_CLOSED),
        )
        for name in (
            "expected_surface_count",
            "observed_record_count",
            "completed_surface_count",
            "completed_arm_pair_count",
            "completed_scenario_row_count",
            "completed_job_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise D106MatrixProtocolError("coverage receipt count drift")
        anomaly_names = (
            "missing_surface_ids",
            "duplicate_surface_ids",
            "unexpected_surface_ids",
        )
        anomaly_lengths = tuple(
            _bounded_sequence_length(
                getattr(self, name),
                "coverage anomaly surface ID",
                MAX_COMPLETION_OBSERVATIONS,
            )
            for name in anomaly_names
        )
        if sum(anomaly_lengths) > MAX_COMPLETION_OBSERVATIONS:
            raise D106MatrixProtocolError(
                "coverage anomaly IDs exceed the shared hard item cap"
            )
        remaining_utf8_bytes = MAX_COMPLETION_IDENTIFIER_UTF8_BYTES
        for name, length in zip(anomaly_names, anomaly_lengths, strict=True):
            values = _bounded_text_sequence(
                getattr(self, name),
                "coverage anomaly surface ID",
                length,
                remaining_utf8_bytes,
            )
            if len(set(values)) != len(values) or values != tuple(sorted(values)):
                raise D106MatrixProtocolError(
                    "coverage anomaly IDs must be unique canonical order"
                )
            used_utf8_bytes = sum(len(value.encode("utf-8")) for value in values)
            remaining_utf8_bytes -= used_utf8_bytes
            object.__setattr__(self, name, values)
        structural_complete = (
            self.expected_surface_count == STATE_SURFACE_COUNT
            and self.observed_record_count == STATE_SURFACE_COUNT
            and self.completed_surface_count == STATE_SURFACE_COUNT
            and self.completed_arm_pair_count == MATCHED_ARM_PAIR_COUNT
            and self.completed_scenario_row_count == SCENARIO_ROW_COUNT
            and self.completed_job_count == OUTER_JOB_COUNT
            and not self.missing_surface_ids
            and not self.duplicate_surface_ids
            and not self.unexpected_surface_ids
        )
        if (status == STRUCTURAL_ID_COVERAGE_ONLY) is not structural_complete:
            raise D106MatrixProtocolError(
                "partial matrix cannot carry STRUCTURAL_ID_COVERAGE_ONLY"
            )

    def _revalidated(self) -> _D106MatrixStructuralCoverageDiagnostic:
        return _D106MatrixStructuralCoverageDiagnostic(
            matrix_receipt_sha256=self.matrix_receipt_sha256,
            status=self.status,
            expected_surface_count=self.expected_surface_count,
            observed_record_count=self.observed_record_count,
            completed_surface_count=self.completed_surface_count,
            completed_arm_pair_count=self.completed_arm_pair_count,
            completed_scenario_row_count=self.completed_scenario_row_count,
            completed_job_count=self.completed_job_count,
            missing_surface_ids=self.missing_surface_ids,
            duplicate_surface_ids=self.duplicate_surface_ids,
            unexpected_surface_ids=self.unexpected_surface_ids,
            runner_authority=self.runner_authority,
            artifact_authority=self.artifact_authority,
            protocol_authority=self.protocol_authority,
            downstream_consumable=self.downstream_consumable,
        )

    @property
    def diagnostic_sha256(self) -> str:
        return canonical_sha256(self.diagnostic_payload())

    def diagnostic_payload(self) -> dict[str, Any]:
        validated = self._revalidated()
        return {
            "schema": SCHEMA + ".structural_id_coverage_diagnostic",
            "matrix_receipt_sha256": validated.matrix_receipt_sha256,
            "status": validated.status,
            "coverage_scope": "surface_id_structure_only_no_artifact_verification",
            "runner_authority": validated.runner_authority,
            "artifact_authority": validated.artifact_authority,
            "protocol_authority": validated.protocol_authority,
            "downstream_consumable": validated.downstream_consumable,
            "artifact_receipt_verified": False,
            "artifact_digest_verified": False,
            "expected_surface_count": validated.expected_surface_count,
            "observed_record_count": validated.observed_record_count,
            "completed_surface_count": validated.completed_surface_count,
            "completed_arm_pair_count": validated.completed_arm_pair_count,
            "completed_scenario_row_count": validated.completed_scenario_row_count,
            "completed_job_count": validated.completed_job_count,
            "missing_surface_ids": list(validated.missing_surface_ids),
            "duplicate_surface_ids": list(validated.duplicate_surface_ids),
            "unexpected_surface_ids": list(validated.unexpected_surface_ids),
            "performance_values_read": False,
            "query_truth_access": False,
            "partial_favorable_selection": False,
        }


def audit_d106_matrix_structural_id_coverage(
    plan: D106MatrixPlan, observed_surface_ids: Sequence[str]
) -> _D106MatrixStructuralCoverageDiagnostic:
    """Audit bounded ID-only coverage with arrival-order-independent receipts."""

    validate_d106_matrix_protocol(plan)
    observed = _bounded_text_sequence(
        observed_surface_ids,
        "observed state-surface ID",
        MAX_COMPLETION_OBSERVATIONS,
        MAX_COMPLETION_IDENTIFIER_UTF8_BYTES,
    )
    expected = tuple(value.surface_id for value in plan.state_surfaces)
    expected_set = set(expected)
    counts = Counter(observed)
    missing = tuple(sorted(value for value in expected if counts[value] == 0))
    duplicate_set = {value for value, count in counts.items() if count > 1}
    duplicates = tuple(sorted(duplicate_set))
    unexpected = tuple(sorted(set(observed) - expected_set))
    completed_surfaces = {value for value in expected if counts[value] == 1}
    completed_pairs = {
        pair.arm_pair_id
        for pair in plan.arm_pairs
        if set(pair.state_surface_ids).issubset(completed_surfaces)
    }
    completed_scenarios = {
        row.scenario_row_id
        for row in plan.scenario_rows
        if set(row.arm_pair_ids).issubset(completed_pairs)
    }
    completed_jobs = {
        job.job_id
        for job in plan.jobs
        if set(job.scenario_row_ids).issubset(completed_scenarios)
    }
    structural_complete = (
        len(observed) == STATE_SURFACE_COUNT
        and len(completed_surfaces) == STATE_SURFACE_COUNT
        and len(completed_pairs) == MATCHED_ARM_PAIR_COUNT
        and len(completed_scenarios) == SCENARIO_ROW_COUNT
        and len(completed_jobs) == OUTER_JOB_COUNT
        and not missing
        and not duplicates
        and not unexpected
    )
    return _D106MatrixStructuralCoverageDiagnostic(
        matrix_receipt_sha256=plan.matrix_receipt_sha256,
        status=(
            STRUCTURAL_ID_COVERAGE_ONLY
            if structural_complete
            else INCOMPLETE_FAIL_CLOSED
        ),
        expected_surface_count=STATE_SURFACE_COUNT,
        observed_record_count=len(observed),
        completed_surface_count=len(completed_surfaces),
        completed_arm_pair_count=len(completed_pairs),
        completed_scenario_row_count=len(completed_scenarios),
        completed_job_count=len(completed_jobs),
        missing_surface_ids=missing,
        duplicate_surface_ids=duplicates,
        unexpected_surface_ids=unexpected,
    )


def reject_d106_matrix_structural_coverage_consumption(
    value: Any,
    *args: Any,
    **kwargs: Any,
) -> NoReturn:
    """Reject every runner, artifact, or protocol use of an ID-only diagnostic."""

    del args, kwargs
    if type(value) is not _D106MatrixStructuralCoverageDiagnostic:
        raise D106MatrixProtocolError(
            "structural coverage consumption rejection requires the exact diagnostic"
        )
    raise D106MatrixProtocolError(
        "structural ID coverage diagnostic has no runner, artifact, or protocol "
        "authority and cannot be consumed downstream"
    )


def reject_d106_matrix_artifact_completion(
    plan: D106MatrixPlan,
    artifact_receipts: Sequence[D106MatrixArtifactReceipt] | None = None,
) -> NoReturn:
    """Fail closed because this module has no trusted artifact verifier authority."""

    validate_d106_matrix_protocol(plan)
    if artifact_receipts is None:
        raise D106MatrixProtocolError(
            "artifact completion unavailable: exact typed D106MatrixArtifactReceipt "
            "with artifact digest and matrix binding is required"
        )
    receipts = _bounded_sequence(
        artifact_receipts,
        "artifact receipts",
        STATE_SURFACE_COUNT,
    )
    if len(receipts) != STATE_SURFACE_COUNT:
        raise D106MatrixProtocolError(
            "artifact completion unavailable: full typed artifact receipt coverage required"
        )
    expected_surface_ids = {value.surface_id for value in plan.state_surfaces}
    receipt_surface_ids: set[str] = set()
    for receipt in receipts:
        if type(receipt) is not D106MatrixArtifactReceipt:
            raise D106MatrixProtocolError(
                "artifact completion unavailable: exact typed artifact receipt required"
            )
        _require_text(receipt.surface_id, "artifact receipt surface ID")
        _require_sha256(
            receipt.matrix_receipt_sha256,
            "artifact receipt matrix binding",
        )
        _require_sha256(receipt.artifact_digest_sha256, "artifact digest")
        try:
            _require_exact_string(
                receipt.artifact_kind,
                "artifact receipt kind",
                "immutable_prediction_artifact",
            )
        except D106MatrixProtocolError as error:
            raise D106MatrixProtocolError(
                "artifact completion unavailable: artifact kind drift"
            ) from error
        try:
            _require_exact_string(
                receipt.matrix_receipt_sha256,
                "artifact receipt matrix binding",
                plan.matrix_receipt_sha256,
            )
        except D106MatrixProtocolError as error:
            raise D106MatrixProtocolError(
                "artifact completion unavailable: artifact matrix binding drift"
            ) from error
        if receipt.surface_id not in expected_surface_ids:
            raise D106MatrixProtocolError(
                "artifact completion unavailable: unexpected artifact surface ID"
            )
        if receipt.surface_id in receipt_surface_ids:
            raise D106MatrixProtocolError(
                "artifact completion unavailable: duplicate artifact surface ID"
            )
        receipt_surface_ids.add(receipt.surface_id)
    if receipt_surface_ids != expected_surface_ids:
        raise D106MatrixProtocolError(
            "artifact completion unavailable: missing artifact surface receipt"
        )
    raise D106MatrixProtocolError(
        "artifact completion unavailable: typed artifact claims remain unverified; "
        "no strict artifact verifier authority is wired"
    )


@dataclass(frozen=True, slots=True)
class D106MatrixResourceEstimate:
    outer_job_count: int
    scenario_row_count: int
    matched_arm_pair_count: int
    state_surface_count: int
    structural_record_count: int
    canonical_plan_bytes_exact: int
    canonical_plan_hard_cap_bytes: int
    primary_identifier_utf8_bytes_exact: int
    completion_observation_hard_cap: int
    completion_identifier_utf8_hard_cap_bytes: int
    anomaly_identifier_shared_hard_cap: int
    anomaly_identifier_utf8_shared_hard_cap_bytes: int
    max_identifier_utf8_bytes: int
    performance_value_fields: int
    truth_or_query_role_fields: int
    unaccounted_overhead: str


def estimate_d106_matrix_resources(plan: D106MatrixPlan) -> D106MatrixResourceEstimate:
    """Report deterministic structural bytes; Python allocator/RSS is excluded."""

    validate_d106_matrix_protocol(plan)
    identifiers = (
        [value.job_id for value in plan.jobs]
        + [value.scenario_row_id for value in plan.scenario_rows]
        + [value.arm_pair_id for value in plan.arm_pairs]
        + [value.surface_id for value in plan.state_surfaces]
    )
    return D106MatrixResourceEstimate(
        outer_job_count=len(plan.jobs),
        scenario_row_count=len(plan.scenario_rows),
        matched_arm_pair_count=len(plan.arm_pairs),
        state_surface_count=len(plan.state_surfaces),
        structural_record_count=(
            len(plan.jobs)
            + len(plan.scenario_rows)
            + len(plan.arm_pairs)
            + len(plan.state_surfaces)
        ),
        canonical_plan_bytes_exact=len(_canonical_bytes(plan.receipt_payload())),
        canonical_plan_hard_cap_bytes=MAX_CANONICAL_PLAN_BYTES,
        primary_identifier_utf8_bytes_exact=sum(
            len(value.encode("utf-8")) for value in identifiers
        ),
        completion_observation_hard_cap=MAX_COMPLETION_OBSERVATIONS,
        completion_identifier_utf8_hard_cap_bytes=(
            MAX_COMPLETION_IDENTIFIER_UTF8_BYTES
        ),
        anomaly_identifier_shared_hard_cap=MAX_COMPLETION_OBSERVATIONS,
        anomaly_identifier_utf8_shared_hard_cap_bytes=(
            MAX_COMPLETION_IDENTIFIER_UTF8_BYTES
        ),
        max_identifier_utf8_bytes=MAX_IDENTIFIER_UTF8_BYTES,
        performance_value_fields=0,
        truth_or_query_role_fields=0,
        unaccounted_overhead=(
            "Python objects, allocator behavior, interpreter metadata, and RSS are not "
            "estimated; canonical structural bytes and primary ID bytes are exact"
        ),
    )


__all__ = [
    "ARMS",
    "STRUCTURAL_ID_COVERAGE_ONLY",
    "D106MatrixArtifactReceipt",
    "D106MatchedArmPair",
    "D106MatrixAccessPolicy",
    "D106MatrixJob",
    "D106MatrixPlan",
    "D106MatrixProtocolError",
    "D106MatrixResourceEstimate",
    "D106ScenarioRow",
    "D106StateSurface",
    "INCOMPLETE_FAIL_CLOSED",
    "LEO_SCENARIOS",
    "MATCHED_ARM_PAIR_COUNT",
    "MAX_CANONICAL_PLAN_BYTES",
    "MAX_COMPLETION_IDENTIFIER_UTF8_BYTES",
    "OUTER_JOB_COUNT",
    "PROTOCOL_SCHEMA",
    "RECEIVERS",
    "SCENARIO_ROW_COUNT",
    "SCHEMA",
    "STATES",
    "STATE_SURFACE_COUNT",
    "STRUCTURAL_RECORD_COUNT",
    "TARGET25_SEED",
    "TARGET25_SLICES",
    "audit_d106_matrix_structural_id_coverage",
    "canonical_sha256",
    "estimate_d106_matrix_resources",
    "freeze_d106_matrix_protocol",
    "reject_d106_matrix_artifact_completion",
    "reject_d106_matrix_structural_coverage_consumption",
    "validate_d106_matrix_protocol",
]
