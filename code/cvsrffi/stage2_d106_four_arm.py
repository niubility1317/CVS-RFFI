"""Fixed D106 2x2 arm declaration with no prediction or formal-handoff surface.

This module owns only the fixed causal labels:

M0       identity DA + baseline Student-t qKNN
M_DA     RDCE DA + baseline Student-t qKNN
M_HEAD   identity DA + RCMR-2V qKNN
M_JOINT  RDCE DA + RCMR-2V qKNN

It can record a same-row request and caller-provided component receipt labels.
It cannot establish data lineage, construct adapted support banks, authorize
identity/RDCE views, wire a strict RCMR loader, execute a query, or produce a
formal runner handoff. The legacy formal-prepare entry point is an
unconditional fail-closed gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping, NoReturn


CANDIDATE_ID = "D106-FOUR-ARM-BINDING/r1"
PROTOCOL_SCHEMA = "p2_min_v1"
VALIDATED_ONCE = "VALIDATED_ONCE"
JOB_BINDING_SCHEMA = "cvs.phase2.d106.four_arm_job_binding.v1"
STATE_SCHEMA = "cvs.phase2.d106.four_arm_state.v2"
NON_FORMAL_DECLARATION_SCHEMA = "cvs.phase2.d106.four_arm_component_declaration.v1"
RESOURCE_RECEIPT_SCHEMA = "cvs.phase2.d106.four_arm_resource_receipt.v1"
FACTOR_LABEL_RECEIPT_SCHEMA = "cvs.phase2.d106.four_arm_factor_label_receipt.v1"

ARMS = ("M0", "M_DA", "M_HEAD", "M_JOINT")
ALLOWED_K = (1, 5, 10)
MAX_REGISTERED_CLASSES = 26
MAX_JOB_ID_BYTES = 128
MAX_SCENARIO_ID_BYTES = 128
MAX_ROW_ID_BYTES = 64
MAX_CANONICAL_STATE_RECEIPT_BYTES = 8192
MAX_CANONICAL_DECLARATION_BYTES = 8192

PER_SAMPLE_POLICY = "per_sample_all_registered_classes"
COMPONENT_DECLARATION_STATUS = "COMPONENTS_DECLARED_NON_FORMAL_NO_QUERY_SCORER"
MISSING_AUTHORITY_CODE = "MISSING_FORMAL_COMPONENT_AUTHORITY"
NO_QUERY_CAPABILITY_CODE = "NO_QUERY_CAPABILITY"

_DECLARATION_TOKEN = object()


class D106FourArmError(ValueError):
    """Raised when the frozen arm map or its request receipt drifts."""


class D106FourArmAuthorityError(D106FourArmError):
    """Raised when an operation would assert unavailable formal authority."""

    code = MISSING_AUTHORITY_CODE


class D106FourArmQueryCapabilityError(D106FourArmError):
    """Raised because this declaration module has no query execution API."""

    code = NO_QUERY_CAPABILITY_CODE


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise D106FourArmError(f"{name} must be a lowercase SHA256")
    if any(character not in "0123456789abcdef" for character in value):
        raise D106FourArmError(f"{name} must be a lowercase SHA256")
    return value


def _require_token(value: Any, name: str, maximum_bytes: int) -> str:
    if type(value) is not str or not value:
        raise D106FourArmError(f"{name} must be a non-empty exact string")
    if len(value.encode("utf-8")) > maximum_bytes:
        raise D106FourArmError(f"{name} exceeds the fixed receipt bound")
    return value


@dataclass(frozen=True, slots=True)
class D106FourArmJobBinding:
    """Requested same-job inputs for all four causal labels.

    The hashes describe what a caller requested. They are not validator,
    scenario, query-root, support-root, model, or data-lineage authority.
    """

    job_id: str
    scenario_id: str
    capsule_id: str
    split_id: str
    validator_receipt_sha256: str
    support_physical_root_sha256: str
    query_physical_root_sha256: str
    row_id: str
    seed: int
    active_k: int
    registered_class_count: int
    protocol_schema: str = PROTOCOL_SCHEMA
    phase2_data_status: str = VALIDATED_ONCE
    support_query_disjoint: bool = True
    query_decision_policy: str = PER_SAMPLE_POLICY
    clean_source_runtime_access: bool = False
    source_runtime_access: bool = False
    query_truth_access: bool = False
    query_role_access: bool = False
    query_batch_count_access: bool = False
    query_class_quota_access: bool = False
    query_fit_access: bool = False
    query_state_updates: int = 0
    performance_scorer_attached: bool = False
    schema: str = JOB_BINDING_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != JOB_BINDING_SCHEMA
            or self.protocol_schema != PROTOCOL_SCHEMA
            or self.phase2_data_status != VALIDATED_ONCE
            or self.support_query_disjoint is not True
            or self.query_decision_policy != PER_SAMPLE_POLICY
            or type(self.seed) is not int
            or self.seed < 0
            or type(self.active_k) is not int
            or self.active_k not in ALLOWED_K
            or type(self.registered_class_count) is not int
            or not 2 <= self.registered_class_count <= MAX_REGISTERED_CLASSES
            or self.clean_source_runtime_access is not False
            or self.source_runtime_access is not False
            or self.query_truth_access is not False
            or self.query_role_access is not False
            or self.query_batch_count_access is not False
            or self.query_class_quota_access is not False
            or self.query_fit_access is not False
            or self.query_state_updates != 0
            or type(self.query_state_updates) is not int
            or self.performance_scorer_attached is not False
        ):
            raise D106FourArmError("four-arm job binding protocol/capability drift")
        object.__setattr__(
            self, "job_id", _require_token(self.job_id, "job_id", MAX_JOB_ID_BYTES)
        )
        object.__setattr__(
            self,
            "scenario_id",
            _require_token(self.scenario_id, "scenario_id", MAX_SCENARIO_ID_BYTES),
        )
        object.__setattr__(
            self, "row_id", _require_token(self.row_id, "row_id", MAX_ROW_ID_BYTES)
        )
        for name in (
            "capsule_id",
            "split_id",
            "validator_receipt_sha256",
            "support_physical_root_sha256",
            "query_physical_root_sha256",
        ):
            object.__setattr__(self, name, _require_sha256(getattr(self, name), name))

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "scenario_id": self.scenario_id,
            "capsule_id": self.capsule_id,
            "split_id": self.split_id,
            "validator_receipt_sha256": self.validator_receipt_sha256,
            "support_physical_root_sha256": self.support_physical_root_sha256,
            "query_physical_root_sha256": self.query_physical_root_sha256,
            "row_id": self.row_id,
            "seed": self.seed,
            "active_k": self.active_k,
            "registered_class_count": self.registered_class_count,
            "protocol_schema": self.protocol_schema,
            "phase2_data_status": self.phase2_data_status,
            "support_query_disjoint": self.support_query_disjoint,
            "query_decision_policy": self.query_decision_policy,
            "clean_source_runtime_access": self.clean_source_runtime_access,
            "source_runtime_access": self.source_runtime_access,
            "query_truth_access": self.query_truth_access,
            "query_role_access": self.query_role_access,
            "query_batch_count_access": self.query_batch_count_access,
            "query_class_quota_access": self.query_class_quota_access,
            "query_fit_access": self.query_fit_access,
            "query_state_updates": self.query_state_updates,
            "performance_scorer_attached": self.performance_scorer_attached,
        }

    @property
    def receipt_sha256(self) -> str:
        """A deterministic request receipt, not verification evidence."""

        return _sha256({"schema": self.schema, "binding": self.as_dict()})


@dataclass(frozen=True, slots=True)
class D106FourArmSpec:
    arm_id: str
    da_factor: str
    head_factor: str

    def __post_init__(self) -> None:
        expected = {
            "M0": ("identity", "baseline_qknn"),
            "M_DA": ("rdce", "baseline_qknn"),
            "M_HEAD": ("identity", "rcmr_2v"),
            "M_JOINT": ("rdce", "rcmr_2v"),
        }
        if type(self.arm_id) is not str or expected.get(self.arm_id) != (
            self.da_factor,
            self.head_factor,
        ):
            raise D106FourArmError("fixed 2x2 arm mapping drift")

    def as_dict(self) -> dict[str, str]:
        return {
            "arm_id": self.arm_id,
            "da_factor": self.da_factor,
            "head_factor": self.head_factor,
        }


@dataclass(frozen=True, slots=True)
class D106FourArmSimpleEffect:
    effect_id: str
    before_arm: str
    after_arm: str
    held_factor: str

    def __post_init__(self) -> None:
        expected = {
            "DA_AT_BASELINE_HEAD": ("M0", "M_DA", "baseline_qknn"),
            "DA_AT_RCMR_HEAD": ("M_HEAD", "M_JOINT", "rcmr_2v"),
            "HEAD_AT_IDENTITY_DA": ("M0", "M_HEAD", "identity"),
            "HEAD_AT_RDCE_DA": ("M_DA", "M_JOINT", "rdce"),
        }
        if type(self.effect_id) is not str or expected.get(self.effect_id) != (
            self.before_arm,
            self.after_arm,
            self.held_factor,
        ):
            raise D106FourArmError("fixed simple-effect trace drift")

    def as_dict(self) -> dict[str, str]:
        return {
            "effect_id": self.effect_id,
            "before_arm": self.before_arm,
            "after_arm": self.after_arm,
            "held_factor": self.held_factor,
        }


ARM_SPECS = (
    D106FourArmSpec("M0", "identity", "baseline_qknn"),
    D106FourArmSpec("M_DA", "rdce", "baseline_qknn"),
    D106FourArmSpec("M_HEAD", "identity", "rcmr_2v"),
    D106FourArmSpec("M_JOINT", "rdce", "rcmr_2v"),
)
SIMPLE_EFFECTS = (
    D106FourArmSimpleEffect("DA_AT_BASELINE_HEAD", "M0", "M_DA", "baseline_qknn"),
    D106FourArmSimpleEffect("DA_AT_RCMR_HEAD", "M_HEAD", "M_JOINT", "rcmr_2v"),
    D106FourArmSimpleEffect("HEAD_AT_IDENTITY_DA", "M0", "M_HEAD", "identity"),
    D106FourArmSimpleEffect("HEAD_AT_RDCE_DA", "M_DA", "M_JOINT", "rdce"),
)


def _state_payload(binding: D106FourArmJobBinding) -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "binding": binding.as_dict(),
        "binding_request_receipt_sha256": binding.receipt_sha256,
        "arms": [spec.as_dict() for spec in ARM_SPECS],
        "simple_effects": [effect.as_dict() for effect in SIMPLE_EFFECTS],
        "same_job_scenario_support_query_binding": False,
        "same_job_scenario_support_query_binding_requested": True,
        "same_job_scenario_support_query_binding_verified": False,
        "request_receipt_only_not_data_lineage_authority": True,
        "data_lineage_authority_verified": False,
        "parameter_scan_dimensions": 0,
        "fallback_allowed": False,
        "query_execution_capability": False,
        "performance_scorer_attached": False,
        "source_runtime_access": False,
        "clean_source_runtime_access": False,
    }


@dataclass(frozen=True, slots=True)
class D106FourArmState:
    """Immutable request-level four-arm state; it owns no model or query data."""

    binding: D106FourArmJobBinding
    state_receipt_sha256: str
    schema: str = STATE_SCHEMA
    candidate_id: str = CANDIDATE_ID
    # This legacy field now means verified binding and must remain false.
    same_job_scenario_support_query_binding: bool = False
    same_job_scenario_support_query_binding_requested: bool = True
    same_job_scenario_support_query_binding_verified: bool = False
    request_receipt_only_not_data_lineage_authority: bool = True
    data_lineage_authority_verified: bool = False
    parameter_scan_dimensions: int = 0
    fallback_allowed: bool = False
    query_execution_capability: bool = False
    performance_scorer_attached: bool = False
    source_runtime_access: bool = False
    clean_source_runtime_access: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.binding) is not D106FourArmJobBinding
            or self.schema != STATE_SCHEMA
            or self.candidate_id != CANDIDATE_ID
            or self.same_job_scenario_support_query_binding is not False
            or self.same_job_scenario_support_query_binding_requested is not True
            or self.same_job_scenario_support_query_binding_verified is not False
            or self.request_receipt_only_not_data_lineage_authority is not True
            or self.data_lineage_authority_verified is not False
            or self.parameter_scan_dimensions != 0
            or type(self.parameter_scan_dimensions) is not int
            or self.fallback_allowed is not False
            or self.query_execution_capability is not False
            or self.performance_scorer_attached is not False
            or self.source_runtime_access is not False
            or self.clean_source_runtime_access is not False
        ):
            raise D106FourArmError("four-arm state capability/factor drift")
        raw = _canonical_bytes(_state_payload(self.binding))
        if len(raw) > MAX_CANONICAL_STATE_RECEIPT_BYTES:
            raise D106FourArmError("four-arm state receipt exceeds the fixed bound")
        if self.state_receipt_sha256 != hashlib.sha256(raw).hexdigest():
            raise D106FourArmError("four-arm state receipt mismatch")
        object.__setattr__(
            self,
            "state_receipt_sha256",
            _require_sha256(self.state_receipt_sha256, "state_receipt_sha256"),
        )

    @property
    def arm_map(self) -> Mapping[str, Mapping[str, str]]:
        return MappingProxyType(
            {spec.arm_id: MappingProxyType(spec.as_dict()) for spec in ARM_SPECS}
        )

    @property
    def simple_effect_trace(self) -> Mapping[str, Mapping[str, str]]:
        return MappingProxyType(
            {
                effect.effect_id: MappingProxyType(effect.as_dict())
                for effect in SIMPLE_EFFECTS
            }
        )


def build_d106_four_arm_state(binding: D106FourArmJobBinding) -> D106FourArmState:
    """Create the fixed 2x2 request state without accepting query data."""

    if type(binding) is not D106FourArmJobBinding:
        raise D106FourArmError("four-arm construction requires an exact job binding")
    raw = _canonical_bytes(_state_payload(binding))
    if len(raw) > MAX_CANONICAL_STATE_RECEIPT_BYTES:
        raise D106FourArmError("four-arm state receipt exceeds the fixed bound")
    return D106FourArmState(
        binding=binding,
        state_receipt_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _require_state(value: Any) -> D106FourArmState:
    if type(value) is not D106FourArmState:
        raise D106FourArmError("four-arm operation requires an exact typed state")
    D106FourArmState(
        binding=value.binding,
        state_receipt_sha256=value.state_receipt_sha256,
        schema=value.schema,
        candidate_id=value.candidate_id,
        same_job_scenario_support_query_binding=value.same_job_scenario_support_query_binding,
        same_job_scenario_support_query_binding_requested=(
            value.same_job_scenario_support_query_binding_requested
        ),
        same_job_scenario_support_query_binding_verified=(
            value.same_job_scenario_support_query_binding_verified
        ),
        request_receipt_only_not_data_lineage_authority=(
            value.request_receipt_only_not_data_lineage_authority
        ),
        data_lineage_authority_verified=value.data_lineage_authority_verified,
        parameter_scan_dimensions=value.parameter_scan_dimensions,
        fallback_allowed=value.fallback_allowed,
        query_execution_capability=value.query_execution_capability,
        performance_scorer_attached=value.performance_scorer_attached,
        source_runtime_access=value.source_runtime_access,
        clean_source_runtime_access=value.clean_source_runtime_access,
    )
    return value


def derive_d106_four_arm_identity_factor_label_receipt(
    state: D106FourArmState,
) -> str:
    """Return a factor label, not an identity-view or data-lineage proof."""

    typed_state = _require_state(state)
    return _sha256(
        {
            "schema": FACTOR_LABEL_RECEIPT_SCHEMA,
            "factor_label": "identity_z_id",
            "four_arm_state_receipt_sha256": typed_state.state_receipt_sha256,
            "binding_request_receipt_sha256": typed_state.binding.receipt_sha256,
            "receipt_scope": "factor_label_only_not_identity_view_or_data_lineage",
            "data_lineage_authority_verified": False,
        }
    )


def _canonical_component_receipt_pairs(
    caller_component_receipts_sha256: Mapping[str, str],
) -> tuple[tuple[str, str], ...]:
    if not isinstance(caller_component_receipts_sha256, Mapping):
        raise D106FourArmError(
            "caller_component_receipts_sha256 must be a mapping keyed by the four arms"
        )
    if set(caller_component_receipts_sha256) != set(ARMS):
        raise D106FourArmError(
            "caller_component_receipts_sha256 must contain exactly M0, M_DA, M_HEAD, M_JOINT"
        )
    return tuple(
        (
            arm_id,
            _require_sha256(
                caller_component_receipts_sha256[arm_id],
                f"caller_component_receipts_sha256[{arm_id}]",
            ),
        )
        for arm_id in ARMS
    )


def _declaration_payload_from_receipts(
    *,
    four_arm_state_receipt_sha256: str,
    binding_request_receipt_sha256: str,
    caller_component_receipts_sha256: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    return {
        "schema": NON_FORMAL_DECLARATION_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "four_arm_state_receipt_sha256": four_arm_state_receipt_sha256,
        "binding_request_receipt_sha256": binding_request_receipt_sha256,
        "caller_component_receipts_sha256": [
            {"arm_id": arm_id, "receipt_sha256": receipt_sha256}
            for arm_id, receipt_sha256 in caller_component_receipts_sha256
        ],
        "declaration_scope": "caller_provided_component_receipts_only",
        "formal_components_bound": False,
        "data_lineage_authority_verified": False,
        "query_execution_capability": False,
        "performance_scorer_attached": False,
        "runner_consumable": False,
        "status": COMPONENT_DECLARATION_STATUS,
    }


@dataclass(frozen=True, slots=True)
class D106FourArmComponentDeclaration:
    """Non-formal caller declaration that cannot authorize a runner or scorer."""

    four_arm_state_receipt_sha256: str
    binding_request_receipt_sha256: str
    caller_component_receipts_sha256: tuple[tuple[str, str], ...]
    declaration_receipt_sha256: str
    status: str = COMPONENT_DECLARATION_STATUS
    formal_components_bound: bool = False
    data_lineage_authority_verified: bool = False
    query_execution_capability: bool = False
    performance_scorer_attached: bool = False
    runner_consumable: bool = False
    schema: str = NON_FORMAL_DECLARATION_SCHEMA
    _mint_token: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            self.schema != NON_FORMAL_DECLARATION_SCHEMA
            or self.status != COMPONENT_DECLARATION_STATUS
            or self.formal_components_bound is not False
            or self.data_lineage_authority_verified is not False
            or self.query_execution_capability is not False
            or self.performance_scorer_attached is not False
            or self.runner_consumable is not False
        ):
            raise D106FourArmError(
                "non-formal component declaration capability/status drift"
            )
        if self._mint_token is not _DECLARATION_TOKEN:
            raise D106FourArmAuthorityError(
                "non-formal component declaration requires the declaration API mint token"
            )
        for name in (
            "four_arm_state_receipt_sha256",
            "binding_request_receipt_sha256",
            "declaration_receipt_sha256",
        ):
            object.__setattr__(self, name, _require_sha256(getattr(self, name), name))
        pairs = _canonical_component_receipt_pairs(
            dict(self.caller_component_receipts_sha256)
        )
        if self.caller_component_receipts_sha256 != pairs:
            raise D106FourArmError(
                "caller component receipts must use the fixed canonical arm order"
            )
        raw = _canonical_bytes(
            _declaration_payload_from_receipts(
                four_arm_state_receipt_sha256=self.four_arm_state_receipt_sha256,
                binding_request_receipt_sha256=self.binding_request_receipt_sha256,
                caller_component_receipts_sha256=pairs,
            )
        )
        if len(raw) > MAX_CANONICAL_DECLARATION_BYTES:
            raise D106FourArmError(
                "non-formal component declaration exceeds the fixed receipt bound"
            )
        if self.declaration_receipt_sha256 != hashlib.sha256(raw).hexdigest():
            raise D106FourArmError("non-formal component declaration receipt mismatch")

    @property
    def caller_component_receipts(self) -> Mapping[str, str]:
        """Caller-supplied labels only; they are not causal-lineage authority."""

        return MappingProxyType(dict(self.caller_component_receipts_sha256))

    @property
    def receipt(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                **_declaration_payload_from_receipts(
                    four_arm_state_receipt_sha256=self.four_arm_state_receipt_sha256,
                    binding_request_receipt_sha256=(
                        self.binding_request_receipt_sha256
                    ),
                    caller_component_receipts_sha256=(
                        self.caller_component_receipts_sha256
                    ),
                ),
                "declaration_receipt_sha256": self.declaration_receipt_sha256,
            }
        )


def declare_d106_four_arm_components_nonformal(
    state: D106FourArmState,
    *,
    caller_component_receipts_sha256: Mapping[str, str],
) -> D106FourArmComponentDeclaration:
    """Record caller-provided component labels without asserting their lineage."""

    typed_state = _require_state(state)
    pairs = _canonical_component_receipt_pairs(caller_component_receipts_sha256)
    raw = _canonical_bytes(
        _declaration_payload_from_receipts(
            four_arm_state_receipt_sha256=typed_state.state_receipt_sha256,
            binding_request_receipt_sha256=typed_state.binding.receipt_sha256,
            caller_component_receipts_sha256=pairs,
        )
    )
    if len(raw) > MAX_CANONICAL_DECLARATION_BYTES:
        raise D106FourArmError(
            "non-formal component declaration exceeds the fixed receipt bound"
        )
    return D106FourArmComponentDeclaration(
        four_arm_state_receipt_sha256=typed_state.state_receipt_sha256,
        binding_request_receipt_sha256=typed_state.binding.receipt_sha256,
        caller_component_receipts_sha256=pairs,
        declaration_receipt_sha256=hashlib.sha256(raw).hexdigest(),
        _mint_token=_DECLARATION_TOKEN,
    )


def prepare_d106_four_arm_formal_handoff(*args: Any, **kwargs: Any) -> NoReturn:
    """Unconditionally reject formal handoff construction from this module.

    No current input combination can close the authority gaps, including a
    duplicate M0/M_DA bank or a public RCMR state object.
    """

    del args, kwargs
    raise D106FourArmAuthorityError(
        "formal four-arm handoff is unavailable: missing "
        "D106AdaptedSupportBankReceipt, identity-view authority, RDCE-view "
        "authority, and RCMR strict-loader-wire authority"
    )


def reject_d106_four_arm_formal_runner_consumption(
    value: Any,
    *args: Any,
    **kwargs: Any,
) -> NoReturn:
    """Unconditionally reject runner consumption of request/declaration receipts."""

    del args, kwargs
    if type(value) is D106FourArmComponentDeclaration:
        raise D106FourArmAuthorityError(
            "non-formal component declaration cannot be consumed by a formal runner"
        )
    if type(value) is D106FourArmState:
        raise D106FourArmAuthorityError(
            "four-arm request state cannot be consumed by a formal runner"
        )
    raise D106FourArmError(
        "formal runner rejection requires a typed four-arm request or declaration"
    )


def reject_d106_four_arm_query_execution(
    value: D106FourArmState | D106FourArmComponentDeclaration,
    *args: Any,
    **kwargs: Any,
) -> NoReturn:
    """Fail closed for every attempted query, prediction, or scoring invocation."""

    if type(value) not in {D106FourArmState, D106FourArmComponentDeclaration}:
        raise D106FourArmError("query rejection requires a typed four-arm receipt")
    del args, kwargs
    raise D106FourArmQueryCapabilityError(
        "D106 four-arm composition has no query capability or performance scorer"
    )


def audit_d106_four_arm_resources(
    value: D106FourArmState | D106FourArmComponentDeclaration,
) -> Mapping[str, Any]:
    """Report only local receipt bytes; external execution resources are not summed."""

    if type(value) is D106FourArmState:
        canonical_payload = _canonical_bytes(_state_payload(value.binding))
        anchor = value.state_receipt_sha256
        maximum = MAX_CANONICAL_STATE_RECEIPT_BYTES
        payload_kind = "request_state"
    elif type(value) is D106FourArmComponentDeclaration:
        canonical_payload = _canonical_bytes(
            _declaration_payload_from_receipts(
                four_arm_state_receipt_sha256=value.four_arm_state_receipt_sha256,
                binding_request_receipt_sha256=value.binding_request_receipt_sha256,
                caller_component_receipts_sha256=value.caller_component_receipts_sha256,
            )
        )
        anchor = value.declaration_receipt_sha256
        maximum = MAX_CANONICAL_DECLARATION_BYTES
        payload_kind = "non_formal_component_declaration"
    else:
        raise D106FourArmError("resource audit requires a typed four-arm receipt")
    receipt = {
        "schema": RESOURCE_RECEIPT_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "receipt_anchor_sha256": anchor,
        "payload_kind": payload_kind,
        "actual_canonical_payload_bytes": len(canonical_payload),
        "canonical_payload_hard_cap_bytes": maximum,
        "formal_components_bound": False,
        "data_lineage_authority_verified": False,
        "additional_persistent_numeric_bytes": 0,
        "additional_query_numeric_buffers": 0,
        "parameter_scan_dimensions": 0,
        "query_execution_capability": False,
        "performance_scorer_attached": False,
        "external_component_resources": "not_aggregated_by_composition_layer",
    }
    return MappingProxyType({**receipt, "resource_receipt_sha256": _sha256(receipt)})


__all__ = [
    "ALLOWED_K",
    "ARMS",
    "ARM_SPECS",
    "CANDIDATE_ID",
    "COMPONENT_DECLARATION_STATUS",
    "D106FourArmAuthorityError",
    "D106FourArmComponentDeclaration",
    "D106FourArmError",
    "D106FourArmJobBinding",
    "D106FourArmQueryCapabilityError",
    "D106FourArmSimpleEffect",
    "D106FourArmSpec",
    "D106FourArmState",
    "MAX_CANONICAL_DECLARATION_BYTES",
    "MAX_CANONICAL_STATE_RECEIPT_BYTES",
    "MISSING_AUTHORITY_CODE",
    "NO_QUERY_CAPABILITY_CODE",
    "SIMPLE_EFFECTS",
    "audit_d106_four_arm_resources",
    "build_d106_four_arm_state",
    "declare_d106_four_arm_components_nonformal",
    "derive_d106_four_arm_identity_factor_label_receipt",
    "prepare_d106_four_arm_formal_handoff",
    "reject_d106_four_arm_formal_runner_consumption",
    "reject_d106_four_arm_query_execution",
]
