"""Truth-blind D105 four-arm evaluation over sealed D92 row packages.

The evaluator consumes the already sealed before/after support and query
packages, reconstructs the exact Phase1 checkpoint model, extracts D105 taps
from received IQ, and emits immutable predictions. Query truth and query role
are deliberately absent from every public input and output type.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from .somph_diagnostic_bundle_loader import load_verified_somph_predictor_bundle
from .somph_predictor_bundle import (
    FORMAL_LEO_WEAK_SCENARIOS,
    QUERY_NPZ_MEMBERS,
    SUPPORT_NPZ_MEMBERS,
    preflight_somph_predictor_bundle_with_authority,
)
from .stage2_d105_cbrc import (
    compute_d105_support_binding_root,
    validate_d105_physical_split,
)
from .stage2_d105_feature_tap import extract_d105_feature_tap
from .stage2_d105_four_arm import (
    ARMS,
    audit_d105_four_arm_resources,
    build_d105_four_arm_state,
    score_d105_four_arm_logits,
)
from .stage2_d105_phase1_bundle import (
    _tensor_from_d105_float32_c_iq,
    build_d105_exact_model_from_checkpoint,
    load_d105_candidate_method_lock,
    load_d105_candidate_runtime_manifest,
    load_d105_phase1_asset,
    make_d105_phase1_runtime_handle,
)
from .stage2_diag_cosine_exploration import _validate_matched_packages
from .stage2_lpo_rc_qknn import TypedValidatedOnceP2SplitHandle
from .stage2_zid_student_t_qknn import Phase1ZIDStudentTLock


SCHEMA = "cvs.phase2.d105.query_row_evaluation.v1"
STATE_SCHEMA = "cvs.phase2.d105.state_prediction.v1"
PAIR_SCHEMA = "cvs.phase2.d105.scenario_prediction_pair.v1"
REGISTRATION_STATES = ("BEFORE_REGISTRATION", "AFTER_REGISTRATION")
PREDICTION_CONTEXT_SCHEMA = "cvs.phase2.d105.prediction_context.v2"
PACKAGE_ROOT_KEYS = (
    "before_enrollment",
    "before_apply",
    "after_enrollment",
    "after_apply",
)
_PACKAGE_STATES = {
    "BEFORE_REGISTRATION": ("before", "S_B", "stage2b"),
    "AFTER_REGISTRATION": ("after", "S_C", "stage2c"),
}


class D105QueryEvaluationError(ValueError):
    """Raised when a D105 row input or immutable prediction binding drifts."""


def _canonical_bytes(value: Any) -> bytes:
    def convert(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): convert(member) for key, member in item.items()}
        if isinstance(item, (tuple, list)):
            return [convert(member) for member in item]
        if isinstance(item, np.generic):
            return item.item()
        return item

    return json.dumps(
        convert(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    text = str(value)
    if (
        len(text) != 64
        or text != text.lower()
        or any(char not in "0123456789abcdef" for char in text)
    ):
        raise D105QueryEvaluationError(f"{name} must be a lowercase SHA256")
    return text


def _regular_bytes(path: str | Path, *, name: str) -> bytes:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise D105QueryEvaluationError(f"{name} must be a regular non-symlink file")
    return source.read_bytes()


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    def freeze(item: Any) -> Any:
        if isinstance(item, Mapping):
            return MappingProxyType(
                {str(key): freeze(member) for key, member in item.items()}
            )
        if isinstance(item, (tuple, list)):
            return tuple(freeze(member) for member in item)
        return item

    return freeze(value)


def _physical_root(values: Sequence[str], name: str) -> str:
    rows = tuple(str(value) for value in values)
    if not rows or any(not value for value in rows) or len(set(rows)) != len(rows):
        raise D105QueryEvaluationError(
            f"{name} must contain unique non-empty physical tokens"
        )
    return _sha256(sorted(rows))


def _registered_handles(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    rows = manifest.get("registered_classes")
    if not isinstance(rows, list):
        raise D105QueryEvaluationError("registered class manifest drift")
    handles = tuple(
        str(row.get("class_handle", "")) if isinstance(row, Mapping) else str(row)
        for row in rows
    )
    if not handles or any(not value for value in handles) or len(set(handles)) != len(
        handles
    ):
        raise D105QueryEvaluationError("registered class handles drift")
    return handles


def build_d105_prediction_context(
    *,
    registration_state: str,
    stage: str,
    scenario: str,
    receiver: str,
    seed: int,
    active_k: int,
    registered_classes: Sequence[str],
    capsule_id: str,
    split_id: str,
    split_validator_receipt_sha256: str,
    support_physical_root_sha256: str,
    query_physical_root_sha256: str,
    package_root_sha256: Mapping[str, str],
    phase1_bundle_manifest_sha256: str,
    validated_bundle_id_sha256: str,
    bundle_content_root_sha256: str,
    bundle_validator_receipt_sha256: str,
    checkpoint_sha256: str,
    data_feature_runtime_sha256: str,
    data_materialization_lock_sha256: str,
    d105_candidate_runtime_manifest_sha256: str,
    d105_candidate_method_lock_sha256: str,
    qknn_lock_digest: str,
) -> tuple[Mapping[str, Any], str]:
    """Build the only canonical, truth-free D105 prediction context."""

    expected_state = (
        "BEFORE_REGISTRATION" if stage == "S_B" else "AFTER_REGISTRATION"
    )
    classes = tuple(str(value) for value in registered_classes)
    receiver_id = str(receiver)
    if (
        stage not in {"S_B", "S_C"}
        or registration_state != expected_state
        or scenario not in FORMAL_LEO_WEAK_SCENARIOS
        or not receiver_id
        or type(seed) is not int
        or type(active_k) is not int
        or active_k not in (1, 5, 10)
        or not classes
        or any(not value for value in classes)
        or len(set(classes)) != len(classes)
        or not isinstance(package_root_sha256, Mapping)
        or set(package_root_sha256) != set(PACKAGE_ROOT_KEYS)
    ):
        raise D105QueryEvaluationError(
            "prediction-context lifecycle/package schema drift"
        )
    hashes = {
        "capsule_id": capsule_id,
        "split_id": split_id,
        "split_validator_receipt_sha256": split_validator_receipt_sha256,
        "support_physical_root_sha256": support_physical_root_sha256,
        "query_physical_root_sha256": query_physical_root_sha256,
        "phase1_bundle_manifest_sha256": phase1_bundle_manifest_sha256,
        "validated_bundle_id_sha256": validated_bundle_id_sha256,
        "bundle_content_root_sha256": bundle_content_root_sha256,
        "bundle_validator_receipt_sha256": bundle_validator_receipt_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "data_feature_runtime_sha256": data_feature_runtime_sha256,
        "data_materialization_lock_sha256": data_materialization_lock_sha256,
        "d105_candidate_runtime_manifest_sha256": (
            d105_candidate_runtime_manifest_sha256
        ),
        "d105_candidate_method_lock_sha256": (
            d105_candidate_method_lock_sha256
        ),
        "qknn_lock_digest": qknn_lock_digest,
    }
    normalized_hashes = {
        name: _require_sha256(value, f"prediction context {name}")
        for name, value in hashes.items()
    }
    roots = {
        name: _require_sha256(
            package_root_sha256[name],
            f"prediction context package root {name}",
        )
        for name in PACKAGE_ROOT_KEYS
    }
    payload = {
        "schema": PREDICTION_CONTEXT_SCHEMA,
        "registration_state": registration_state,
        "stage": stage,
        "scenario": scenario,
        "receiver": receiver_id,
        "seed": seed,
        "active_k": active_k,
        "registered_classes": list(classes),
        "capsule_id": normalized_hashes["capsule_id"],
        "split_id": normalized_hashes["split_id"],
        "split_validator_receipt_sha256": normalized_hashes[
            "split_validator_receipt_sha256"
        ],
        "support_physical_root_sha256": normalized_hashes[
            "support_physical_root_sha256"
        ],
        "query_physical_root_sha256": normalized_hashes[
            "query_physical_root_sha256"
        ],
        "package_root_sha256": roots,
        "phase1_bundle_manifest_sha256": normalized_hashes[
            "phase1_bundle_manifest_sha256"
        ],
        "validated_bundle_id_sha256": normalized_hashes[
            "validated_bundle_id_sha256"
        ],
        "bundle_content_root_sha256": normalized_hashes[
            "bundle_content_root_sha256"
        ],
        "bundle_validator_receipt_sha256": normalized_hashes[
            "bundle_validator_receipt_sha256"
        ],
        "checkpoint_sha256": normalized_hashes["checkpoint_sha256"],
        "data_feature_runtime_sha256": normalized_hashes[
            "data_feature_runtime_sha256"
        ],
        "data_materialization_lock_sha256": normalized_hashes[
            "data_materialization_lock_sha256"
        ],
        "d105_candidate_runtime_manifest_sha256": normalized_hashes[
            "d105_candidate_runtime_manifest_sha256"
        ],
        "d105_candidate_method_lock_sha256": normalized_hashes[
            "d105_candidate_method_lock_sha256"
        ],
        "qknn_lock_digest": normalized_hashes["qknn_lock_digest"],
        "query_truth_present": False,
        "query_role_present": False,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
    }
    return _freeze_mapping(payload), _sha256(payload)


@dataclass(frozen=True, slots=True)
class D105SealedPackageRef:
    package_root: str | Path
    detached_seal_path: str | Path
    expected_seal_sha256: str
    formal_policy_path: str | Path
    formal_policy_authorization_path: str | Path
    signed_policy_authorization_envelope_path: str | Path
    expected_signed_policy_authorization_envelope_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.expected_seal_sha256, "package seal SHA256")
        _require_sha256(
            self.expected_signed_policy_authorization_envelope_sha256,
            "signed policy authorization envelope SHA256",
        )


@dataclass(frozen=True, slots=True)
class D105SplitAuthority:
    registration_state: str
    scenario: str
    capsule_id: str
    split_id: str
    validator_receipt_sha256: str
    support_token_root_sha256: str
    query_token_root_sha256: str
    protocol_schema: str = "p2_min_v1"
    phase2_data_status: str = "VALIDATED_ONCE"

    def __post_init__(self) -> None:
        if (
            self.registration_state not in REGISTRATION_STATES
            or self.scenario not in FORMAL_LEO_WEAK_SCENARIOS
            or self.protocol_schema != "p2_min_v1"
            or self.phase2_data_status != "VALIDATED_ONCE"
        ):
            raise D105QueryEvaluationError("D105 split authority lifecycle drift")
        for name in (
            "capsule_id",
            "split_id",
            "validator_receipt_sha256",
            "support_token_root_sha256",
            "query_token_root_sha256",
        ):
            _require_sha256(getattr(self, name), f"split authority {name}")


@dataclass(frozen=True, slots=True)
class D105Phase1BundleAuthority:
    """External D105 validator handle adapted from any current builder manifest."""

    bundle_dir: str | Path
    manifest_sha256: str
    bundle_wire_sha256: str
    validated_bundle_id_sha256: str
    validator_receipt_sha256: str
    expected_content_root_sha256: str
    checkpoint_sha256: str
    candidate_runtime_manifest_path: str | Path
    candidate_method_lock_path: str | Path
    d105_candidate_runtime_manifest_sha256: str
    d105_candidate_method_lock_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "manifest_sha256",
            "bundle_wire_sha256",
            "validated_bundle_id_sha256",
            "validator_receipt_sha256",
            "expected_content_root_sha256",
            "checkpoint_sha256",
            "d105_candidate_runtime_manifest_sha256",
            "d105_candidate_method_lock_sha256",
        ):
            _require_sha256(getattr(self, name), f"Phase1 authority {name}")


@dataclass(frozen=True, slots=True)
class D105QueryEvaluationContext:
    before_enrollment: D105SealedPackageRef
    before_apply: D105SealedPackageRef
    after_enrollment: D105SealedPackageRef
    after_apply: D105SealedPackageRef
    split_authorities: tuple[D105SplitAuthority, ...]
    phase1_bundle: D105Phase1BundleAuthority
    checkpoint_path: str | Path
    checkpoint_sha256: str
    data_feature_runtime_sha256: str
    data_materialization_lock_sha256: str
    qknn_lock: Phase1ZIDStudentTLock
    device: str = "cpu"
    feature_batch_size: int = 64
    score_chunk_size: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "checkpoint_sha256",
            "data_feature_runtime_sha256",
            "data_materialization_lock_sha256",
        ):
            _require_sha256(getattr(self, name), f"context {name}")
        if type(self.qknn_lock) is not Phase1ZIDStudentTLock:
            raise D105QueryEvaluationError("exact Phase1 qKNN lock is required")
        if type(self.feature_batch_size) is not int or self.feature_batch_size < 1:
            raise D105QueryEvaluationError("feature_batch_size must be positive")
        if self.score_chunk_size is not None and (
            type(self.score_chunk_size) is not int or self.score_chunk_size < 1
        ):
            raise D105QueryEvaluationError("score_chunk_size must be positive or None")
        keys = {
            (authority.registration_state, authority.scenario)
            for authority in self.split_authorities
        }
        expected = {
            (state, scenario)
            for state in REGISTRATION_STATES
            for scenario in FORMAL_LEO_WEAK_SCENARIOS
        }
        if len(self.split_authorities) != len(expected) or keys != expected:
            raise D105QueryEvaluationError(
                "exactly one split authority per state/scenario is required"
            )


@dataclass(frozen=True, slots=True)
class D105StatePrediction:
    registration_state: str
    stage: str
    scenario: str
    registered_classes: tuple[str, ...]
    query_physical_ids: tuple[str, ...]
    capsule_id: str
    split_id: str
    split_validator_receipt_sha256: str
    support_physical_root_sha256: str
    query_physical_root_sha256: str
    state_receipt_sha256: str
    prediction_context_sha256: str
    arm_predictions: Mapping[str, tuple[str, ...]]
    arm_prediction_sha256: Mapping[str, str]
    logit_sha256: Mapping[str, str]
    feature_receipt_sha256: str
    resource_receipt_sha256: str
    data_feature_runtime_sha256: str
    data_materialization_lock_sha256: str
    d105_candidate_runtime_manifest_sha256: str
    d105_candidate_method_lock_sha256: str
    receipt_sha256: str
    schema: str = STATE_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != STATE_SCHEMA
            or self.registration_state not in REGISTRATION_STATES
            or self.stage not in {"S_B", "S_C"}
            or self.scenario not in FORMAL_LEO_WEAK_SCENARIOS
            or tuple(self.arm_predictions) != ARMS
            or tuple(self.arm_prediction_sha256) != ARMS
            or tuple(self.logit_sha256) != ARMS
        ):
            raise D105QueryEvaluationError("state prediction schema drift")
        if (
            not self.registered_classes
            or len(set(self.registered_classes)) != len(self.registered_classes)
            or not self.query_physical_ids
            or len(set(self.query_physical_ids)) != len(self.query_physical_ids)
        ):
            raise D105QueryEvaluationError("state registry/query identity drift")
        for name in (
            "capsule_id",
            "split_id",
            "split_validator_receipt_sha256",
            "support_physical_root_sha256",
            "query_physical_root_sha256",
            "state_receipt_sha256",
            "prediction_context_sha256",
            "feature_receipt_sha256",
            "resource_receipt_sha256",
            "data_feature_runtime_sha256",
            "data_materialization_lock_sha256",
            "d105_candidate_runtime_manifest_sha256",
            "d105_candidate_method_lock_sha256",
            "receipt_sha256",
        ):
            _require_sha256(getattr(self, name), f"state prediction {name}")
        frozen_predictions = MappingProxyType(
            {
                arm: tuple(str(value) for value in self.arm_predictions[arm])
                for arm in ARMS
            }
        )
        row_count = len(self.query_physical_ids)
        if any(len(frozen_predictions[arm]) != row_count for arm in ARMS):
            raise D105QueryEvaluationError("arm prediction row alignment drift")
        if any(
            value not in self.registered_classes
            for arm in ARMS
            for value in frozen_predictions[arm]
        ):
            raise D105QueryEvaluationError(
                "arm prediction falls outside the registered classes"
            )
        for arm in ARMS:
            _require_sha256(
                self.arm_prediction_sha256[arm],
                f"{arm} prediction SHA256",
            )
            _require_sha256(self.logit_sha256[arm], f"{arm} logit SHA256")
            if self.arm_prediction_sha256[arm] != _sha256(
                list(frozen_predictions[arm])
            ):
                raise D105QueryEvaluationError(
                    f"{arm} immutable prediction SHA256 drift"
                )
        object.__setattr__(self, "arm_predictions", frozen_predictions)
        object.__setattr__(
            self,
            "arm_prediction_sha256",
            MappingProxyType(dict(self.arm_prediction_sha256)),
        )
        object.__setattr__(self, "logit_sha256", MappingProxyType(dict(self.logit_sha256)))


@dataclass(frozen=True, slots=True)
class D105ScenarioPredictionPair:
    scenario: str
    before: D105StatePrediction
    after: D105StatePrediction
    pair_receipt_sha256: str
    schema: str = PAIR_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != PAIR_SCHEMA
            or self.scenario not in FORMAL_LEO_WEAK_SCENARIOS
            or self.before.scenario != self.scenario
            or self.after.scenario != self.scenario
            or self.before.registration_state != "BEFORE_REGISTRATION"
            or self.after.registration_state != "AFTER_REGISTRATION"
            or not set(self.before.query_physical_ids)
            < set(self.after.query_physical_ids)
        ):
            raise D105QueryEvaluationError("before/after scenario-pair binding drift")


@dataclass(frozen=True, slots=True)
class D105QueryRowEvaluation:
    receiver: str
    seed: int
    k_shot: int
    old_classes: tuple[str, ...]
    new_classes: tuple[str, ...]
    scenario_pairs: tuple[D105ScenarioPredictionPair, ...]
    checkpoint_load_receipt_sha256: str
    package_binding_receipt_sha256: str
    predictor_receipt_sha256: str
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != SCHEMA
            or tuple(pair.scenario for pair in self.scenario_pairs)
            != tuple(FORMAL_LEO_WEAK_SCENARIOS)
        ):
            raise D105QueryEvaluationError("row evaluation scenario closure drift")

    def state_prediction(
        self, scenario: str, registration_state: str
    ) -> D105StatePrediction:
        """Return one bound state without discarding the paired row artifact."""

        for pair in self.scenario_pairs:
            if pair.scenario == scenario:
                if registration_state == "BEFORE_REGISTRATION":
                    return pair.before
                if registration_state == "AFTER_REGISTRATION":
                    return pair.after
        raise D105QueryEvaluationError("unknown scenario or registration state")

    def target25_output_for(self, request: Any) -> Any:
        """Adapt one member of this complete pair-set to the Target25 callback.

        The complete evaluation is produced first. This method only selects one
        already sealed state requested by the runner; it cannot execute an
        after-only evaluation path.
        """

        prediction = self.state_prediction(
            str(request.scenario), str(request.registration_state)
        )
        expected_new = (
            () if prediction.stage == "S_B" else tuple(self.new_classes)
        )
        if (
            str(request.receiver) != self.receiver
            or int(request.seed) != self.seed
            or int(request.k_shot) != self.k_shot
            or str(request.stage) != prediction.stage
            or str(request.capsule_id) != prediction.capsule_id
            or str(request.split_id) != prediction.split_id
            or str(request.authority_receipt_sha256)
            != prediction.split_validator_receipt_sha256
            or tuple(request.query_physical_ids) != prediction.query_physical_ids
            or _physical_root(
                tuple(request.support_physical_ids), "Target25 support physical IDs"
            )
            != prediction.support_physical_root_sha256
            or tuple(request.registered_classes) != prediction.registered_classes
            or tuple(request.old_classes) != self.old_classes
            or tuple(request.new_classes) != expected_new
            or str(request.prediction_context_sha256)
            != prediction.prediction_context_sha256
            or str(request.data_feature_runtime_sha256)
            != prediction.data_feature_runtime_sha256
            or str(request.data_materialization_lock_sha256)
            != prediction.data_materialization_lock_sha256
            or str(request.d105_candidate_runtime_manifest_sha256)
            != prediction.d105_candidate_runtime_manifest_sha256
            or str(request.d105_candidate_method_lock_sha256)
            != prediction.d105_candidate_method_lock_sha256
        ):
            raise D105QueryEvaluationError(
                "Target25 request does not match the complete paired evaluation"
            )
        from .stage2_d105_target25_runner import D105Target25PredictionOutput

        return D105Target25PredictionOutput(
            stage=prediction.stage,
            registration_state=prediction.registration_state,
            arm_predictions=prediction.arm_predictions,
            state_receipt_sha256=prediction.state_receipt_sha256,
            predictor_receipt_sha256=prediction.receipt_sha256,
            feature_receipt_sha256=prediction.feature_receipt_sha256,
            resource_receipt_sha256=prediction.resource_receipt_sha256,
            logit_sha256_by_arm=prediction.logit_sha256,
            arm_prediction_sha256_by_arm=prediction.arm_prediction_sha256,
        )


PackageLoader = Callable[
    [D105SealedPackageRef],
    tuple[dict[str, dict[str, np.ndarray]], dict[str, Any], dict[str, Any]],
]
ModelLoader = Callable[[bytes, int, torch.device], tuple[torch.nn.Module, Mapping[str, Any]]]
FeatureExtractor = Callable[[torch.nn.Module, torch.Tensor], Any]


def _default_package_loader(reference: D105SealedPackageRef):
    authority_manifest, _authority_seal, authority_audit = (
        preflight_somph_predictor_bundle_with_authority(
            reference.package_root,
            detached_seal_path=reference.detached_seal_path,
            expected_seal_sha256=reference.expected_seal_sha256,
            formal_policy_path=reference.formal_policy_path,
            formal_policy_authorization_path=(
                reference.formal_policy_authorization_path
            ),
            signed_policy_authorization_envelope_path=(
                reference.signed_policy_authorization_envelope_path
            ),
            expected_signed_policy_authorization_envelope_sha256=(
                reference.expected_signed_policy_authorization_envelope_sha256
            ),
        )
    )
    payloads, materialized_manifest, materialization_audit = (
        load_verified_somph_predictor_bundle(
            reference.package_root,
            detached_seal_path=reference.detached_seal_path,
            expected_seal_sha256=reference.expected_seal_sha256,
        )
    )
    required_authority_hashes = (
        "authority_commit_sha256",
        "package_root_sha256",
        "package_detached_seal_sha256",
        "signed_policy_authorization_envelope_sha256",
    )
    if (
        authority_manifest != materialized_manifest
        or authority_audit.get("signed_path_free_runtime_authorization_verified")
        is not True
        or authority_audit.get("iq_open_authorized") is not True
        or any(
            _require_sha256(
                authority_audit.get(name), f"D92 authority audit {name}"
            )
            != (
                reference.expected_seal_sha256
                if name == "package_detached_seal_sha256"
                else (
                    reference.expected_signed_policy_authorization_envelope_sha256
                    if name
                    == "signed_policy_authorization_envelope_sha256"
                    else authority_audit.get(name)
                )
            )
            for name in required_authority_hashes
        )
        or authority_audit.get("package_root_sha256")
        != authority_manifest.get("package_root_sha256")
    ):
        raise D105QueryEvaluationError(
            "D92 signed authority/materialization binding drift"
        )
    return payloads, materialized_manifest, {
        "schema": "cvs.phase2.d105.authorized_materialization_audit.v1",
        "authority": dict(authority_audit),
        "materialization": dict(materialization_audit),
        "authority_commit_sha256": authority_audit[
            "authority_commit_sha256"
        ],
        "package_root_sha256": authority_manifest["package_root_sha256"],
        "package_detached_seal_sha256": reference.expected_seal_sha256,
        "signed_policy_authorization_envelope_sha256": (
            reference.expected_signed_policy_authorization_envelope_sha256
        ),
        "receiver": authority_manifest["receiver"],
        "seed": authority_manifest["seed"],
        "stage": authority_manifest["stage"],
        "registration_state": authority_manifest["registration_state"],
        "k_shot": authority_manifest["k_shot"],
    }


def _default_model_loader(
    checkpoint_bytes: bytes, input_len: int, device: torch.device
) -> tuple[torch.nn.Module, Mapping[str, Any]]:
    try:
        from baseline_origin_sat_view import SatViewStage
    except ImportError as error:
        raise D105QueryEvaluationError(
            "checkpoint safe-global dependency is unavailable"
        ) from error
    safe_globals = getattr(torch.serialization, "safe_globals", None)
    try:
        if safe_globals is not None:
            with safe_globals([SatViewStage]):
                checkpoint = torch.load(
                    io.BytesIO(checkpoint_bytes),
                    map_location="cpu",
                    weights_only=True,
                )
            checkpoint_policy = "weights_only_with_explicit_safe_globals"
        else:
            try:
                checkpoint = torch.load(
                    io.BytesIO(checkpoint_bytes),
                    map_location="cpu",
                    weights_only=False,
                )
            except TypeError:
                checkpoint = torch.load(
                    io.BytesIO(checkpoint_bytes), map_location="cpu"
                )
            checkpoint_policy = "legacy_exact_sha_bound_bytes"
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise D105QueryEvaluationError(
            "exact checkpoint deserialization failed closed"
        ) from error
    if not isinstance(checkpoint, Mapping):
        raise D105QueryEvaluationError("checkpoint payload must be a mapping")
    model, audit = build_d105_exact_model_from_checkpoint(
        checkpoint, input_len=input_len, device=device
    )
    model.to(device)
    model.eval()
    return model, {**dict(audit), "checkpoint_policy": checkpoint_policy}


def _device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if not value.startswith("cuda:") or not torch.cuda.is_available():
        raise D105QueryEvaluationError(f"requested device is unavailable: {value}")
    result = torch.device(value)
    try:
        index = int(value.split(":", 1)[1])
    except (IndexError, ValueError) as error:
        raise D105QueryEvaluationError("invalid CUDA device") from error
    if index >= torch.cuda.device_count():
        raise D105QueryEvaluationError(f"requested device is unavailable: {value}")
    return result


def _load_candidate_identity(
    authority: D105Phase1BundleAuthority,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Verify canonical candidate artifacts and every declared core source."""

    try:
        runtime_loaded = load_d105_candidate_runtime_manifest(
            authority.candidate_runtime_manifest_path,
            expected_checkpoint_sha256=authority.checkpoint_sha256,
        )
        runtime_sha256 = runtime_loaded[
            "d105_candidate_runtime_manifest_sha256"
        ]
        lock_loaded = load_d105_candidate_method_lock(
            authority.candidate_method_lock_path,
            expected_checkpoint_sha256=authority.checkpoint_sha256,
            expected_runtime_sha256=runtime_sha256,
        )
    except ValueError as error:
        raise D105QueryEvaluationError(
            "D105 candidate implementation identity validation failed"
        ) from error
    if (
        runtime_sha256
        != authority.d105_candidate_runtime_manifest_sha256
        or lock_loaded["d105_candidate_method_lock_sha256"]
        != authority.d105_candidate_method_lock_sha256
    ):
        raise D105QueryEvaluationError(
            "D105 candidate implementation/lock SHA256 drift"
        )
    runtime_manifest = runtime_loaded["manifest"]
    code_root = Path(__file__).resolve().parents[1]
    for relative_path, expected_sha256 in runtime_manifest[
        "core_file_sha256"
    ].items():
        raw = Path(str(relative_path))
        if raw.is_absolute() or ".." in raw.parts:
            raise D105QueryEvaluationError(
                "candidate core-file manifest path is unsafe"
            )
        unresolved = code_root / raw
        if unresolved.is_symlink():
            raise D105QueryEvaluationError(
                "candidate core-file manifest points to a symbolic link"
            )
        try:
            source = unresolved.resolve(strict=True)
        except FileNotFoundError as error:
            raise D105QueryEvaluationError(
                f"candidate core file is missing: {relative_path}"
            ) from error
        try:
            source.relative_to(code_root)
        except ValueError as error:
            raise D105QueryEvaluationError(
                "candidate core-file manifest escapes the code root"
            ) from error
        actual = _sha256_bytes(
            _regular_bytes(source, name=f"candidate core file {relative_path}")
        )
        if actual != expected_sha256:
            raise D105QueryEvaluationError(
                f"candidate core-file SHA256 drift: {relative_path}"
            )
    return runtime_manifest, lock_loaded["lock"]


def _validate_qknn_lock_identity(
    qknn_lock: Phase1ZIDStudentTLock, candidate_lock: Mapping[str, Any]
) -> None:
    expected = candidate_lock["student_t_qknn"]
    actual = {
        "student_nu": float(qknn_lock.student_nu),
        "kernel_effective_dim": int(qknn_lock.kernel_effective_dim),
        "kernel_volume_gamma": float(qknn_lock.kernel_volume_gamma),
        "shared_h0": float(qknn_lock.shared_h0),
        "scale_prior_strength": float(qknn_lock.scale_prior_strength),
        "scale_min_ratio": float(qknn_lock.scale_min_ratio),
        "scale_max_ratio": float(qknn_lock.scale_max_ratio),
        "temperature": float(qknn_lock.temperature),
        "support_storage": "int8_fp16_scale",
    }
    if actual != expected:
        raise D105QueryEvaluationError(
            "runtime qKNN configuration differs from the candidate method lock"
        )


def _cross_state_lock(
    before_enrollment: Mapping[str, Any],
    before_apply: Mapping[str, Any],
    after_enrollment: Mapping[str, Any],
    after_apply: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...], int, str, int]:
    for field in (
        "receiver",
        "seed",
        "k_shot",
        "phase1_checkpoint_sha256",
        "feature_runtime_sha256",
        "method_lock_sha256",
        "row_handle",
        "row_manifest_sha256",
    ):
        before_value = before_apply.get(field, before_enrollment.get(field))
        after_value = after_apply.get(field, after_enrollment.get(field))
        if before_value != after_value:
            raise D105QueryEvaluationError(f"before/after {field} drift")
    for manifest, expected_state, expected_stage in (
        (before_enrollment, "before", "stage2b"),
        (before_apply, "before", "stage2b"),
        (after_enrollment, "after", "stage2c"),
        (after_apply, "after", "stage2c"),
    ):
        if (
            manifest.get("registration_state") != expected_state
            or manifest.get("stage") != expected_stage
        ):
            raise D105QueryEvaluationError("before/after lifecycle stage drift")
    old_classes = _registered_handles(before_enrollment)
    all_classes = _registered_handles(after_enrollment)
    if all_classes[: len(old_classes)] != old_classes:
        raise D105QueryEvaluationError("old registered class prefix drift")
    new_count = len(all_classes) - len(old_classes)
    k_shot = int(after_enrollment.get("k_shot", -1))
    if len(old_classes) != 6 or new_count not in (5, 10, 20):
        raise D105QueryEvaluationError("D105 Target25 class-count lock drift")
    if k_shot not in (1, 5, 10):
        raise D105QueryEvaluationError("D105 K-shot lock drift")
    return (
        old_classes,
        all_classes,
        k_shot,
        str(after_enrollment.get("receiver")),
        int(after_enrollment.get("seed")),
    )


def _support_rows(
    payload: Mapping[str, np.ndarray],
    *,
    registered_classes: tuple[str, ...],
    active_k: int,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    if set(payload) != set(SUPPORT_NPZ_MEMBERS):
        raise D105QueryEvaluationError("support payload exact allowlist drift")
    ranks = np.asarray(payload["support_rank_within_class"])
    indices = np.asarray(payload["support_class_indices"])
    tokens = np.asarray(payload["support_tokens"]).astype(str)
    iq = np.asarray(payload["support_leo_weak_iq"])
    if (
        ranks.dtype.kind not in "iu"
        or indices.dtype.kind not in "iu"
        or ranks.ndim != 1
        or indices.shape != ranks.shape
        or tokens.shape != ranks.shape
        or len(iq) != len(ranks)
        or iq.dtype != np.float32
        or iq.ndim != 3
        or iq.shape[1] != 2
        or not np.isfinite(iq).all()
    ):
        raise D105QueryEvaluationError("support IQ/index/token contract drift")
    mask = ranks.astype(np.int64) < active_k
    selected_indices = indices.astype(np.int64)[mask]
    if (
        len(selected_indices) != active_k * len(registered_classes)
        or len(selected_indices) == 0
        or int(selected_indices.min()) != 0
        or int(selected_indices.max()) != len(registered_classes) - 1
        or any(
            int(np.sum(selected_indices == index)) != active_k
            for index in range(len(registered_classes))
        )
    ):
        raise D105QueryEvaluationError("support balanced K-shot assignment drift")
    selected_tokens = tuple(tokens[mask].tolist())
    _physical_root(selected_tokens, "support tokens")
    labels = tuple(registered_classes[index] for index in selected_indices.tolist())
    return np.ascontiguousarray(iq[mask], dtype=np.float32), labels, selected_tokens


def _query_rows(
    payload: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, tuple[str, ...]]:
    if set(payload) != set(QUERY_NPZ_MEMBERS):
        raise D105QueryEvaluationError(
            "query payload exact allowlist drift; truth/role fields are forbidden"
        )
    iq = np.asarray(payload["query_leo_weak_iq"])
    tokens = tuple(np.asarray(payload["query_tokens"]).astype(str).tolist())
    if (
        iq.dtype != np.float32
        or iq.ndim != 3
        or iq.shape[1] != 2
        or len(iq) != len(tokens)
        or len(iq) == 0
        or not np.isfinite(iq).all()
    ):
        raise D105QueryEvaluationError("query IQ/token contract drift")
    _physical_root(tokens, "query tokens")
    return np.ascontiguousarray(iq, dtype=np.float32), tokens


def _tap_rows(
    model: torch.nn.Module,
    iq: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    feature_extractor: FeatureExtractor,
) -> tuple[np.ndarray, np.ndarray, str]:
    pre_relu: list[np.ndarray] = []
    z_dom: list[np.ndarray] = []
    receipts: list[str] = []
    for start in range(0, len(iq), batch_size):
        batch = np.ascontiguousarray(iq[start : start + batch_size], dtype=np.float32)
        tensor = _tensor_from_d105_float32_c_iq(
            batch,
            torch_module=torch,
            device=device,
            error_type=D105QueryEvaluationError,
            name="D105 query batch",
        )
        tapped = feature_extractor(model, tensor)
        pre = np.asarray(tapped.pre_relu)
        domain = np.asarray(tapped.z_dom)
        if (
            pre.dtype != np.float32
            or domain.dtype != np.float32
            or pre.shape != (len(tensor), 160)
            or domain.shape != (len(tensor), 160)
            or not np.isfinite(pre).all()
            or not np.isfinite(domain).all()
        ):
            raise D105QueryEvaluationError("D105 feature tap output drift")
        pre_relu.append(np.ascontiguousarray(pre, dtype=np.float32))
        z_dom.append(np.ascontiguousarray(domain, dtype=np.float32))
        receipts.append(_require_sha256(tapped.receipt_sha256, "feature tap receipt"))
    merged_pre = np.concatenate(pre_relu, axis=0)
    merged_domain = np.concatenate(z_dom, axis=0)
    receipt = _sha256(
        {
            "schema": "cvs.phase2.d105.feature_tap_batch_binding.v1",
            "rows": len(iq),
            "chunk_receipts": receipts,
            "query_rows_used_for_fit": 0,
            "query_state_updates": 0,
        }
    )
    return merged_pre, merged_domain, receipt


def _package_binding(
    references: Sequence[D105SealedPackageRef],
    manifests: Sequence[Mapping[str, Any]],
    audits: Sequence[Mapping[str, Any]],
) -> str:
    rows: list[dict[str, Any]] = []
    for reference, manifest, audit in zip(
        references, manifests, audits, strict=True
    ):
        authority_commit = _require_sha256(
            audit.get("authority_commit_sha256"),
            "D92 package authority commit SHA256",
        )
        package_root = _require_sha256(
            manifest.get("package_root_sha256"), "D92 package root SHA256"
        )
        if (
            audit.get("package_root_sha256") != package_root
            or audit.get("package_detached_seal_sha256")
            != reference.expected_seal_sha256
            or audit.get("signed_policy_authorization_envelope_sha256")
            != reference.expected_signed_policy_authorization_envelope_sha256
            or audit.get("receiver") != manifest.get("receiver")
            or audit.get("seed") != manifest.get("seed")
            or audit.get("stage") != manifest.get("stage")
            or audit.get("registration_state")
            != manifest.get("registration_state")
            or audit.get("k_shot") != manifest.get("k_shot")
        ):
            raise D105QueryEvaluationError(
                "D92 authority audit/package lifecycle binding drift"
            )
        rows.append(
            {
                "package_root_sha256": package_root,
                "expected_seal_sha256": reference.expected_seal_sha256,
                "authority_commit_sha256": authority_commit,
                "signed_policy_authorization_envelope_sha256": _require_sha256(
                    audit.get(
                        "signed_policy_authorization_envelope_sha256"
                    ),
                    "D92 signed policy authorization envelope SHA256",
                ),
                "profile": manifest.get("profile"),
                "receiver": manifest.get("receiver"),
                "seed": manifest.get("seed"),
                "registration_state": manifest.get("registration_state"),
                "stage": manifest.get("stage"),
                "k_shot": manifest.get("k_shot"),
                "data_feature_runtime_sha256": manifest.get(
                    "feature_runtime_sha256"
                ),
                "data_materialization_lock_sha256": manifest.get(
                    "method_lock_sha256"
                ),
                "audit_sha256": _sha256(audit),
            }
        )
    return _sha256(
        {
            "schema": "cvs.phase2.d105.d92_package_binding.v1",
            "packages": rows,
        }
    )


def _validate_authority_split_receipt_binding(
    audits: Sequence[Mapping[str, Any]],
    split_authorities: Sequence[D105SplitAuthority],
) -> str:
    authority_commits = {
        _require_sha256(
            audit.get("authority_commit_sha256"),
            "D92 package authority commit SHA256",
        )
        for audit in audits
    }
    split_receipts = {
        _require_sha256(
            authority.validator_receipt_sha256,
            "D105 split validator receipt SHA256",
        )
        for authority in split_authorities
    }
    if (
        len(audits) != len(PACKAGE_ROOT_KEYS)
        or len(split_authorities)
        != len(REGISTRATION_STATES) * len(FORMAL_LEO_WEAK_SCENARIOS)
        or len(authority_commits) != 1
        or len(split_receipts) != 1
        or authority_commits != split_receipts
    ):
        raise D105QueryEvaluationError(
            "D92 authority commit/split validator receipt binding drift"
        )
    return next(iter(authority_commits))


def _preflight_d105_model_inputs(
    payloads: Sequence[Mapping[str, Mapping[str, np.ndarray]]],
    *,
    old_classes: tuple[str, ...],
    all_classes: tuple[str, ...],
    active_k: int,
    split_by_key: Mapping[tuple[str, str], D105SplitAuthority],
) -> int:
    """Validate every IQ/token input before reconstructing the checkpoint.

    Model construction is deliberately deferred until the complete sealed
    package surface is known to be usable.  This keeps malformed package
    payloads, forbidden query metadata, split-root drift, and before/after
    query mismatches from opening the Phase1 checkpoint at all.
    """

    if len(payloads) != len(PACKAGE_ROOT_KEYS) or any(
        set(payload) != set(FORMAL_LEO_WEAK_SCENARIOS) for payload in payloads
    ):
        raise D105QueryEvaluationError("formal three-scenario package closure drift")
    input_lengths: set[int] = set()
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        before_support_iq, _before_labels, before_support_tokens = _support_rows(
            payloads[0][scenario],
            registered_classes=old_classes,
            active_k=active_k,
        )
        before_query_iq, before_query_tokens = _query_rows(payloads[1][scenario])
        after_support_iq, _after_labels, after_support_tokens = _support_rows(
            payloads[2][scenario],
            registered_classes=all_classes,
            active_k=active_k,
        )
        after_query_iq, after_query_tokens = _query_rows(payloads[3][scenario])
        for registration_state, support_iq, support_tokens, query_iq, query_tokens in (
            (
                "BEFORE_REGISTRATION",
                before_support_iq,
                before_support_tokens,
                before_query_iq,
                before_query_tokens,
            ),
            (
                "AFTER_REGISTRATION",
                after_support_iq,
                after_support_tokens,
                after_query_iq,
                after_query_tokens,
            ),
        ):
            validate_d105_physical_split(support_tokens, query_tokens)
            support_root = _physical_root(support_tokens, "support tokens")
            query_root = _physical_root(query_tokens, "query tokens")
            authority = split_by_key[(registration_state, scenario)]
            if (
                authority.support_token_root_sha256 != support_root
                or authority.query_token_root_sha256 != query_root
            ):
                raise D105QueryEvaluationError(
                    "VALIDATED_ONCE split token-root drift"
                )
            input_lengths.add(int(support_iq.shape[-1]))
            input_lengths.add(int(query_iq.shape[-1]))
        after_index = {
            token: index for index, token in enumerate(after_query_tokens)
        }
        if not set(before_query_tokens) < set(after_index):
            raise D105QueryEvaluationError(
                "before old-query tokens must be an after-query subset"
            )
        for index, token in enumerate(before_query_tokens):
            if not np.array_equal(
                before_query_iq[index], after_query_iq[after_index[token]]
            ):
                raise D105QueryEvaluationError(
                    "shared before/after query IQ bytes drift"
                )
    if len(input_lengths) != 1:
        raise D105QueryEvaluationError(
            "received IQ input length drift across packages"
        )
    return next(iter(input_lengths))


def _evaluate_state(
    *,
    registration_state: str,
    scenario: str,
    support_payload: Mapping[str, np.ndarray],
    query_payload: Mapping[str, np.ndarray],
    registered_classes: tuple[str, ...],
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    active_k: int,
    split_authority: D105SplitAuthority,
    model: torch.nn.Module,
    device: torch.device,
    feature_batch_size: int,
    score_chunk_size: int | None,
    feature_extractor: FeatureExtractor,
    bundle: Any,
    bundle_handle: Any,
    qknn_lock: Phase1ZIDStudentTLock,
    package_binding_sha256: str,
    package_root_sha256: Mapping[str, str],
    phase1_authority: D105Phase1BundleAuthority,
    receiver: str,
    seed: int,
    data_feature_runtime_sha256: str,
    data_materialization_lock_sha256: str,
) -> tuple[D105StatePrediction, Mapping[str, np.ndarray]]:
    _, stage, _ = _PACKAGE_STATES[registration_state]
    support_iq, support_labels, support_tokens = _support_rows(
        support_payload,
        registered_classes=registered_classes,
        active_k=active_k,
    )
    query_iq, query_tokens = _query_rows(query_payload)
    validate_d105_physical_split(support_tokens, query_tokens)
    support_root = _physical_root(support_tokens, "support tokens")
    query_root = _physical_root(query_tokens, "query tokens")
    if (
        split_authority.support_token_root_sha256 != support_root
        or split_authority.query_token_root_sha256 != query_root
    ):
        raise D105QueryEvaluationError("VALIDATED_ONCE split token-root drift")
    split_handle = TypedValidatedOnceP2SplitHandle(
        capsule_id=split_authority.capsule_id,
        split_id=split_authority.split_id,
        validator_receipt_sha256=split_authority.validator_receipt_sha256,
        support_physical_root_sha256=support_root,
        query_physical_root_sha256=query_root,
        support_query_disjoint=True,
    )
    support_pre, support_zdom, support_tap_receipt = _tap_rows(
        model,
        support_iq,
        device=device,
        batch_size=feature_batch_size,
        feature_extractor=feature_extractor,
    )
    query_pre, _query_zdom, query_tap_receipt = _tap_rows(
        model,
        query_iq,
        device=device,
        batch_size=feature_batch_size,
        feature_extractor=feature_extractor,
    )
    support_receipt = compute_d105_support_binding_root(
        support_pre,
        support_zdom,
        support_labels,
        support_tokens,
        registered_classes,
        old_classes,
        new_classes,
        active_k=active_k,
        stage=stage,
    )
    state = build_d105_four_arm_state(
        bundle,
        bundle_handle,
        support_pre,
        support_zdom,
        support_labels,
        support_tokens,
        registered_classes,
        old_classes,
        new_classes,
        config=qknn_lock,
        split_handle=split_handle,
        active_k=active_k,
        stage=stage,
        support_receipt_sha256=support_receipt,
    )
    before_state_receipt = state.receipt_sha256
    logits = score_d105_four_arm_logits(
        state,
        query_pre,
        query_physical_ids=query_tokens,
        chunk_size=score_chunk_size,
    )
    resource = audit_d105_four_arm_resources(state)
    if (
        state.receipt_sha256 != before_state_receipt
        or int(resource.get("query_rows_used_for_fit", -1)) != 0
        or int(resource.get("query_state_updates", -1)) != 0
    ):
        raise D105QueryEvaluationError("query fit/update closure drift")
    predictions = {
        arm: tuple(
            registered_classes[int(index)]
            for index in np.argmax(logits.by_arm[arm], axis=1).tolist()
        )
        for arm in ARMS
    }
    logit_hashes = {
        arm: _sha256_bytes(np.ascontiguousarray(logits.by_arm[arm]).tobytes())
        for arm in ARMS
    }
    prediction_hashes = {arm: _sha256(list(predictions[arm])) for arm in ARMS}
    if active_k == 1 and (
        logit_hashes["M_HEAD"] != logit_hashes["M0"]
        or logit_hashes["M_JOINT"] != logit_hashes["M_DA"]
        or prediction_hashes["M_HEAD"] != prediction_hashes["M0"]
        or prediction_hashes["M_JOINT"] != prediction_hashes["M_DA"]
    ):
        raise D105QueryEvaluationError("K1 exact four-arm identity drift")
    feature_receipt = _sha256(
        {
            "support_tap_receipt_sha256": support_tap_receipt,
            "query_tap_receipt_sha256": query_tap_receipt,
            "support_rows": len(support_tokens),
            "query_rows": len(query_tokens),
            "query_rows_used_for_fit": 0,
            "query_state_updates": 0,
        }
    )
    resource_receipt = _sha256(resource)
    prediction_context_payload, prediction_context = (
        build_d105_prediction_context(
            registration_state=registration_state,
            stage=stage,
            scenario=scenario,
            receiver=receiver,
            seed=seed,
            active_k=active_k,
            registered_classes=registered_classes,
            capsule_id=split_authority.capsule_id,
            split_id=split_authority.split_id,
            split_validator_receipt_sha256=(
                split_authority.validator_receipt_sha256
            ),
            support_physical_root_sha256=support_root,
            query_physical_root_sha256=query_root,
            package_root_sha256=package_root_sha256,
            phase1_bundle_manifest_sha256=phase1_authority.manifest_sha256,
            validated_bundle_id_sha256=(
                phase1_authority.validated_bundle_id_sha256
            ),
            bundle_content_root_sha256=(
                phase1_authority.expected_content_root_sha256
            ),
            bundle_validator_receipt_sha256=(
                phase1_authority.validator_receipt_sha256
            ),
            checkpoint_sha256=phase1_authority.checkpoint_sha256,
            data_feature_runtime_sha256=data_feature_runtime_sha256,
            data_materialization_lock_sha256=(
                data_materialization_lock_sha256
            ),
            d105_candidate_runtime_manifest_sha256=(
                phase1_authority.d105_candidate_runtime_manifest_sha256
            ),
            d105_candidate_method_lock_sha256=(
                phase1_authority.d105_candidate_method_lock_sha256
            ),
            qknn_lock_digest=qknn_lock.lock_digest,
        )
    )
    receipt_payload = {
        "schema": STATE_SCHEMA,
        "prediction_context_sha256": prediction_context,
        "prediction_context_payload_sha256": _sha256(
            prediction_context_payload
        ),
        "package_binding_sha256": package_binding_sha256,
        "state_receipt_sha256": state.receipt_sha256,
        "arm_prediction_sha256": prediction_hashes,
        "logit_sha256": logit_hashes,
        "feature_receipt_sha256": feature_receipt,
        "resource_receipt_sha256": resource_receipt,
        "data_feature_runtime_sha256": data_feature_runtime_sha256,
        "data_materialization_lock_sha256": data_materialization_lock_sha256,
        "d105_candidate_runtime_manifest_sha256": (
            phase1_authority.d105_candidate_runtime_manifest_sha256
        ),
        "d105_candidate_method_lock_sha256": (
            phase1_authority.d105_candidate_method_lock_sha256
        ),
        "query_truth_present": False,
        "query_role_present": False,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
    }
    return (
        D105StatePrediction(
            registration_state=registration_state,
            stage=stage,
            scenario=scenario,
            registered_classes=registered_classes,
            query_physical_ids=query_tokens,
            capsule_id=split_authority.capsule_id,
            split_id=split_authority.split_id,
            split_validator_receipt_sha256=(
                split_authority.validator_receipt_sha256
            ),
            support_physical_root_sha256=support_root,
            query_physical_root_sha256=query_root,
            state_receipt_sha256=state.receipt_sha256,
            prediction_context_sha256=prediction_context,
            arm_predictions=predictions,
            arm_prediction_sha256=prediction_hashes,
            logit_sha256=logit_hashes,
            feature_receipt_sha256=feature_receipt,
            resource_receipt_sha256=resource_receipt,
            data_feature_runtime_sha256=data_feature_runtime_sha256,
            data_materialization_lock_sha256=data_materialization_lock_sha256,
            d105_candidate_runtime_manifest_sha256=(
                phase1_authority.d105_candidate_runtime_manifest_sha256
            ),
            d105_candidate_method_lock_sha256=(
                phase1_authority.d105_candidate_method_lock_sha256
            ),
            receipt_sha256=_sha256(receipt_payload),
        ),
        logits.by_arm,
    )


def evaluate_d105_query_row(
    context: D105QueryEvaluationContext,
    *,
    package_loader: PackageLoader = _default_package_loader,
    model_loader: ModelLoader = _default_model_loader,
    feature_extractor: FeatureExtractor = extract_d105_feature_tap,
) -> D105QueryRowEvaluation:
    """Evaluate all three scenarios and both registration states truth-blind."""

    if type(context) is not D105QueryEvaluationContext:
        raise D105QueryEvaluationError("exact D105 query context is required")
    references = (
        context.before_enrollment,
        context.before_apply,
        context.after_enrollment,
        context.after_apply,
    )
    loaded = tuple(package_loader(reference) for reference in references)
    payloads = tuple(item[0] for item in loaded)
    manifests = tuple(item[1] for item in loaded)
    audits = tuple(item[2] for item in loaded)
    _validate_authority_split_receipt_binding(
        audits,
        context.split_authorities,
    )
    if any(set(payload) != set(FORMAL_LEO_WEAK_SCENARIOS) for payload in payloads):
        raise D105QueryEvaluationError("formal three-scenario package closure drift")
    _validate_matched_packages(manifests[0], manifests[1])
    _validate_matched_packages(manifests[2], manifests[3])
    old_classes, all_classes, active_k, receiver, seed = _cross_state_lock(
        *manifests
    )
    expected_common = {
        "phase1_checkpoint_sha256": context.checkpoint_sha256,
        "feature_runtime_sha256": context.data_feature_runtime_sha256,
        "method_lock_sha256": context.data_materialization_lock_sha256,
    }
    for manifest in manifests:
        if any(manifest.get(key) != value for key, value in expected_common.items()):
            raise D105QueryEvaluationError("D92 package/context authority drift")
    authority = context.phase1_bundle
    if authority.checkpoint_sha256 != context.checkpoint_sha256:
        raise D105QueryEvaluationError("Phase1 bundle/checkpoint authority drift")
    _runtime_manifest, candidate_lock = _load_candidate_identity(authority)
    _validate_qknn_lock_identity(context.qknn_lock, candidate_lock)
    try:
        asset = load_d105_phase1_asset(
            authority.bundle_dir, require_formal_phase2_eligible=True
        )
        bundle_handle = make_d105_phase1_runtime_handle(asset)
    except ValueError as error:
        raise D105QueryEvaluationError(
            "formal D105 Phase1 asset validation failed"
        ) from error
    manifest = asset.manifest
    bundle = asset.bundle
    if (
        asset.manifest_sha256 != authority.manifest_sha256
        or manifest.get("bundle_wire_sha256") != authority.bundle_wire_sha256
        or str(asset.validated_bundle_id_sha256)
        != authority.validated_bundle_id_sha256
        or str(asset.validator_receipt_sha256)
        != authority.validator_receipt_sha256
        or bundle.content_root_sha256 != authority.expected_content_root_sha256
        or bundle.checkpoint_sha256 != authority.checkpoint_sha256
        or manifest.get("d105_candidate_runtime_manifest_sha256")
        != authority.d105_candidate_runtime_manifest_sha256
        or manifest.get("d105_candidate_method_lock_sha256")
        != authority.d105_candidate_method_lock_sha256
        or bundle.runtime_sha256
        != authority.d105_candidate_runtime_manifest_sha256
        or bundle.method_lock_sha256
        != authority.d105_candidate_method_lock_sha256
    ):
        raise D105QueryEvaluationError("formal Phase1 asset/context authority drift")
    package_binding = _package_binding(references, manifests, audits)
    package_root_sha256 = {
        name: _require_sha256(
            manifest.get("package_root_sha256"),
            f"{name} package_root_sha256",
        )
        for name, manifest in zip(PACKAGE_ROOT_KEYS, manifests, strict=True)
    }
    split_by_key = {
        (split_authority.registration_state, split_authority.scenario): split_authority
        for split_authority in context.split_authorities
    }
    input_len = _preflight_d105_model_inputs(
        payloads,
        old_classes=old_classes,
        all_classes=all_classes,
        active_k=active_k,
        split_by_key=split_by_key,
    )
    checkpoint_bytes = _regular_bytes(
        context.checkpoint_path, name="exact Phase1 checkpoint"
    )
    if _sha256_bytes(checkpoint_bytes) != context.checkpoint_sha256:
        raise D105QueryEvaluationError("exact Phase1 checkpoint SHA256 drift")
    runtime_device = _device(context.device)
    model, checkpoint_audit = model_loader(
        checkpoint_bytes, input_len, runtime_device
    )
    if not isinstance(model, torch.nn.Module) or model.training:
        raise D105QueryEvaluationError("exact checkpoint loader must return eval model")
    checkpoint_load_receipt = _sha256(checkpoint_audit)
    scenario_pairs: list[D105ScenarioPredictionPair] = []
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        before, _before_logits = _evaluate_state(
            registration_state="BEFORE_REGISTRATION",
            scenario=scenario,
            support_payload=payloads[0][scenario],
            query_payload=payloads[1][scenario],
            registered_classes=old_classes,
            old_classes=old_classes,
            new_classes=(),
            active_k=active_k,
            split_authority=split_by_key[("BEFORE_REGISTRATION", scenario)],
            model=model,
            device=runtime_device,
            feature_batch_size=context.feature_batch_size,
            score_chunk_size=context.score_chunk_size,
            feature_extractor=feature_extractor,
            bundle=bundle,
            bundle_handle=bundle_handle,
            qknn_lock=context.qknn_lock,
            package_binding_sha256=package_binding,
            package_root_sha256=package_root_sha256,
            phase1_authority=authority,
            receiver=receiver,
            seed=seed,
            data_feature_runtime_sha256=context.data_feature_runtime_sha256,
            data_materialization_lock_sha256=(
                context.data_materialization_lock_sha256
            ),
        )
        after, _after_logits = _evaluate_state(
            registration_state="AFTER_REGISTRATION",
            scenario=scenario,
            support_payload=payloads[2][scenario],
            query_payload=payloads[3][scenario],
            registered_classes=all_classes,
            old_classes=old_classes,
            new_classes=all_classes[len(old_classes) :],
            active_k=active_k,
            split_authority=split_by_key[("AFTER_REGISTRATION", scenario)],
            model=model,
            device=runtime_device,
            feature_batch_size=context.feature_batch_size,
            score_chunk_size=context.score_chunk_size,
            feature_extractor=feature_extractor,
            bundle=bundle,
            bundle_handle=bundle_handle,
            qknn_lock=context.qknn_lock,
            package_binding_sha256=package_binding,
            package_root_sha256=package_root_sha256,
            phase1_authority=authority,
            receiver=receiver,
            seed=seed,
            data_feature_runtime_sha256=context.data_feature_runtime_sha256,
            data_materialization_lock_sha256=(
                context.data_materialization_lock_sha256
            ),
        )
        pair_receipt = _sha256(
            {
                "schema": PAIR_SCHEMA,
                "scenario": scenario,
                "before_receipt_sha256": before.receipt_sha256,
                "after_receipt_sha256": after.receipt_sha256,
                "before_query_is_after_subset": True,
                "shared_query_iq_exact": True,
            }
        )
        scenario_pairs.append(
            D105ScenarioPredictionPair(
                scenario=scenario,
                before=before,
                after=after,
                pair_receipt_sha256=pair_receipt,
            )
        )
    predictor_receipt = _sha256(
        {
            "schema": SCHEMA,
            "receiver": receiver,
            "seed": seed,
            "k_shot": active_k,
            "old_classes": old_classes,
            "new_classes": all_classes[len(old_classes) :],
            "scenario_pair_receipts": [
                pair.pair_receipt_sha256 for pair in scenario_pairs
            ],
            "checkpoint_load_receipt_sha256": checkpoint_load_receipt,
            "package_binding_receipt_sha256": package_binding,
            "data_feature_runtime_sha256": context.data_feature_runtime_sha256,
            "data_materialization_lock_sha256": (
                context.data_materialization_lock_sha256
            ),
            "d105_candidate_runtime_manifest_sha256": (
                authority.d105_candidate_runtime_manifest_sha256
            ),
            "d105_candidate_method_lock_sha256": (
                authority.d105_candidate_method_lock_sha256
            ),
            "query_truth_present": False,
            "query_role_present": False,
            "query_rows_used_for_fit": 0,
            "query_state_updates": 0,
            "per_sample_all_registered_classes": True,
        }
    )
    return D105QueryRowEvaluation(
        receiver=receiver,
        seed=seed,
        k_shot=active_k,
        old_classes=old_classes,
        new_classes=all_classes[len(old_classes) :],
        scenario_pairs=tuple(scenario_pairs),
        checkpoint_load_receipt_sha256=checkpoint_load_receipt,
        package_binding_receipt_sha256=package_binding,
        predictor_receipt_sha256=predictor_receipt,
    )


__all__ = [
    "D105Phase1BundleAuthority",
    "D105QueryEvaluationContext",
    "D105QueryEvaluationError",
    "D105QueryRowEvaluation",
    "D105ScenarioPredictionPair",
    "D105SealedPackageRef",
    "D105SplitAuthority",
    "D105StatePrediction",
    "build_d105_prediction_context",
    "evaluate_d105_query_row",
]
