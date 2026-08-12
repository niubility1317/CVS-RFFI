"""Fail-closed primitives for Phase1 CLIC target LEO confirmation.

This module deliberately contains no cache builder, channel simulator, target
adaptation, threshold fitting, or truth-side scoring loop.  It provides the
small immutable helpers shared by the offline target sealer and the separate
truth-side evaluator.  The target predictor is allowed to see only the
sealed received-IQ package; labels, roles, TX, receiver, and day metadata stay
in the isolated sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import re
import zipfile

import numpy as np

from cvsrffi.leo_weak_cache import (
    FORMAL_LEO_WEAK_SCENARIOS,
    canonical_json_sha256,
    sha256_file,
)
from cvsrffi import phase1_clic as _clic


TARGET_PACKAGE_SCHEMA = "cvs.phase1.clic_target_iq_only_package.v1"
TARGET_TRUTH_SCHEMA = "cvs.phase1.clic_target_truth_sidecar.v1"
ADV3B02_REFERENCE_SCHEMA = "cvs.phase1.adv3b02_target_known_reference.v1"
ADV3B02_METRICS_SCHEMA = "cvs.phase1.adv3b02_target_known_metrics.v1"
PREDICTOR_STATE_SCHEMA = "cvs.phase1.clic_predictor_state.v1"
_TRAIN_CONFIG_SCHEMA = "cvs.phase1.clic_train_data_config.v1"
_PAIR_SCHEMA = "cvs.phase1.clic_postfreeze_pair.v1"
_PAIR_RAW_ARTIFACT_STEMS = (
    "checkpoint",
    "terminal",
    "clean",
    "leo",
    "leo_binding",
    "common_receipt",
    "proxy_diagnostic",
)


class CLICTargetProtocolError(RuntimeError):
    """Raised when a target artifact cannot satisfy the frozen protocol."""


class CLICTargetGateError(CLICTargetProtocolError):
    """Raised when a non-compensating target confirmation gate fails."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic JSON bytes while rejecting non-finite state."""

    def convert(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return convert(item.tolist())
        if isinstance(item, np.generic):
            return convert(item.item())
        if isinstance(item, Mapping):
            return {str(key): convert(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(value) for value in item]
        if isinstance(item, float):
            if not math.isfinite(item):
                raise CLICTargetProtocolError("non-finite value cannot enter sealed target state")
            return item
        if isinstance(item, (str, int, bool)) or item is None:
            return item
        raise CLICTargetProtocolError(
            f"unsupported sealed target state type: {type(item).__name__}"
        )

    try:
        return json.dumps(
            convert(value),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CLICTargetProtocolError("cannot canonicalize sealed target state") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise CLICTargetProtocolError(f"{label} SHA256 is invalid")
    try:
        int(value, 16)
    except ValueError as exc:
        raise CLICTargetProtocolError(f"{label} SHA256 is invalid") from exc
    if value.lower() != value:
        raise CLICTargetProtocolError(f"{label} SHA256 must be lowercase")
    return value


def read_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"{label} is missing: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CLICTargetProtocolError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise CLICTargetProtocolError(f"{label} must be a JSON object")
    # Exercise the serializer at read time, rather than letting NaN leak into a
    # later manifest or comparison receipt.
    canonical_json_bytes(value)
    return dict(value)


def _require_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CLICTargetProtocolError(f"{label} must be a JSON object")
    return {str(key): item for key, item in value.items()}


def _validate_exact_keys(
    payload: Mapping[str, Any], *, required: tuple[str, ...], ignored: frozenset[str], label: str
) -> dict[str, Any]:
    fields = {str(key): value for key, value in payload.items()}
    missing = [field for field in required if field not in fields]
    if missing:
        raise CLICTargetProtocolError(f"{label} is missing required data-config fields: {missing}")
    unexpected = sorted(set(fields) - set(required) - set(ignored))
    if unexpected:
        raise CLICTargetProtocolError(
            f"{label} has unsupported data-config fields and cannot be normalized safely: {unexpected}"
        )
    return {field: fields[field] for field in required}


# These are deliberately narrow data schemas.  Everything outside the schema
# either has no bearing on a training/test data configuration and is explicitly
# ignored, or fails closed rather than silently weakening the equality audit.
_TRAIN_DATA_FIELDS = (
    "dataset_provenance",
    "source_train_tx_ids",
    "source_validation_tx_ids",
    "source_proxy_tx_ids",
    "source_receiver_ids",
    "source_day_ids",
    "split_mode",
    "role_construction",
    "physical_row_selection",
    "preprocessing",
    "single_leo_training_scenes",
)
_TRAIN_IGNORED_FIELDS = frozenset(
    {
        "schema",
        "normalized_sha256",
        "capsule_id",
        "physical_sample_ids",
        "physical_sample_ids_sha256",
        "received_iq_sha256",
        "received_iq_sha256_root",
        "scene_seed",
        "seed",
        "random_seed",
        "epoch",
        "epochs",
        "optimizer",
        "loss",
        "model",
        "model_architecture",
        "model_state",
        "checkpoint",
        "checkpoint_sha256",
    }
)
_KNOWN_TEST_DATA_FIELDS = (
    "target_receiver_ids",
    "target_day_ids",
    "target_known_tx_ids",
    "class_order",
    "scenes",
    "leo_weak_channel",
    "preprocessing",
    "zero_adaptation",
    "metric_definitions",
)
_KNOWN_TEST_IGNORED_FIELDS = frozenset(
    {
        "schema",
        "normalized_sha256",
        "capsule_id",
        "physical_sample_ids",
        "physical_sample_ids_sha256",
        "received_iq_sha256",
        "received_iq_sha256_root",
        "scene_seed",
        "seed",
        "random_seed",
        "checkpoint",
        "checkpoint_sha256",
        "model",
        "model_architecture",
        "epoch",
        "epochs",
        "optimizer",
        "loss",
    }
)

_REQUIRED_ROLE_CONSTRUCTION = {
    "split_mode": "tx_rx_day_1_6_3",
    "labeled_ratio": 0.07,
    "unlabeled_ratio": 0.63,
    "source_val_ratio": 0.30,
}


# Only this semantic surface may participate in an ADV/CLIC training-data
# equivalence key.  Receipt, manifest and selected-row hashes establish that a
# particular arm has not changed after sealing, but they do *not* establish a
# different experimental configuration: the comparison intentionally permits
# distinct physical rows/received-IQ realisations and seeds.
_SEMANTIC_DATASET_PROVENANCE_FIELDS = (
    "dataset_schema",
    "wisig_pkl_sha256",
)
_SEMANTIC_PHYSICAL_SELECTION_FIELDS = (
    "selection_policy",
    "group_axes",
)


def _normalize_semantic_dataset_provenance(value: Any) -> Any:
    """Keep a data identity, never per-arm receipt/manifest provenance.

    A short legacy string remains a valid semantic dataset identifier for
    already-sealed test/legacy descriptors.  New real C/G/ADV manifests must
    use the explicit WiSig mapping, which binds the frozen dataset bytes but
    not an arm-specific source split receipt.
    """

    if isinstance(value, str) and value:
        return value
    payload = _require_mapping(value, label="training dataset_provenance")
    allowed = {
        frozenset({"dataset_schema"}),
        frozenset(_SEMANTIC_DATASET_PROVENANCE_FIELDS),
    }
    if frozenset(payload) not in allowed:
        raise CLICTargetProtocolError(
            "training dataset_provenance must contain only semantic "
            "dataset_schema and optional wisig_pkl_sha256 fields"
        )
    if str(payload["dataset_schema"]) != "WiSig":
        raise CLICTargetProtocolError("training dataset_provenance dataset schema drift")
    normalized = {"dataset_schema": "WiSig"}
    if "wisig_pkl_sha256" in payload:
        normalized["wisig_pkl_sha256"] = require_sha256(
            payload["wisig_pkl_sha256"], label="training WiSig dataset"
        )
    return normalized


def _normalize_semantic_physical_row_selection(value: Any) -> Any:
    """Keep the selection policy/axes, never concrete selected rows.

    A historical scalar policy is semantic (and contains no row identity), so
    it stays readable for prior C/ADV descriptors.  Mapped configurations are
    intentionally exact and reject receipt/index/hash fields rather than
    accidentally turning them into an equivalence key.
    """

    if isinstance(value, str) and value:
        return value
    payload = _require_mapping(value, label="training physical_row_selection")
    if set(payload) != set(_SEMANTIC_PHYSICAL_SELECTION_FIELDS):
        raise CLICTargetProtocolError(
            "training physical_row_selection must contain only semantic "
            "selection_policy and group_axes fields"
        )
    policy = str(payload["selection_policy"])
    axes = payload["group_axes"]
    if not policy or not isinstance(axes, list) or tuple(str(item) for item in axes) != (
        "tx_id",
        "rx_id",
        "day_id",
        "eq_id",
    ):
        raise CLICTargetProtocolError("training physical-row semantic selection drift")
    return {
        "selection_policy": policy,
        "group_axes": ["tx_id", "rx_id", "day_id", "eq_id"],
    }


def _require_fixed_role_construction(value: Any) -> dict[str, Any]:
    """Reject opaque role labels and historical ratios before comparison.

    A string such as ``source_only_labeled_unlabeled_validation`` does not
    establish which active fractions produced the checkpoint.  The current
    Phase1 data contract fixes the three roles at 0.07/0.63/0.30, so a legacy
    0.2 training configuration must never enter the matched ADV3B02 gate.
    """

    role = _require_mapping(value, label="training role_construction")
    expected_fields = set(_REQUIRED_ROLE_CONSTRUCTION)
    if set(role) != expected_fields:
        raise CLICTargetProtocolError(
            "training role_construction must exactly declare split_mode and "
            "labeled/unlabeled/source-validation ratios"
        )
    if str(role["split_mode"]) != _REQUIRED_ROLE_CONSTRUCTION["split_mode"]:
        raise CLICTargetProtocolError("training role_construction split_mode drift")
    normalized: dict[str, Any] = {"split_mode": str(role["split_mode"])}
    for field in ("labeled_ratio", "unlabeled_ratio", "source_val_ratio"):
        raw = role[field]
        if isinstance(raw, bool):
            raise CLICTargetProtocolError(f"training role_construction {field} must be numeric")
        try:
            number = float(raw)
        except (TypeError, ValueError) as exc:
            raise CLICTargetProtocolError(
                f"training role_construction {field} must be numeric"
            ) from exc
        if not math.isfinite(number) or not math.isclose(
            number,
            float(_REQUIRED_ROLE_CONSTRUCTION[field]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise CLICTargetProtocolError(
                "training role_construction ratio drift: "
                f"{field}={number!r}, expected={_REQUIRED_ROLE_CONSTRUCTION[field]!r}"
            )
        normalized[field] = float(_REQUIRED_ROLE_CONSTRUCTION[field])
    if not math.isclose(
        sum(float(normalized[field]) for field in ("labeled_ratio", "unlabeled_ratio", "source_val_ratio")),
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise CLICTargetProtocolError("training role_construction ratios must sum to one")
    return normalized


def normalize_train_data_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize exactly the frozen *training data* configuration surface."""

    normalized = _validate_exact_keys(
        _require_mapping(config, label="candidate/baseline train config"),
        required=_TRAIN_DATA_FIELDS,
        ignored=_TRAIN_IGNORED_FIELDS,
        label="candidate/baseline train config",
    )
    normalized["role_construction"] = _require_fixed_role_construction(
        normalized["role_construction"]
    )
    normalized["dataset_provenance"] = _normalize_semantic_dataset_provenance(
        normalized["dataset_provenance"]
    )
    normalized["physical_row_selection"] = _normalize_semantic_physical_row_selection(
        normalized["physical_row_selection"]
    )
    if str(normalized["split_mode"]) != normalized["role_construction"]["split_mode"]:
        raise CLICTargetProtocolError(
            "training split_mode and role_construction split_mode must agree"
        )
    canonical_json_bytes(normalized)
    return normalized


def normalize_known_test_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize exactly the frozen target-known test data configuration."""

    normalized = _validate_exact_keys(
        _require_mapping(config, label="candidate/baseline known-test config"),
        required=_KNOWN_TEST_DATA_FIELDS,
        ignored=_KNOWN_TEST_IGNORED_FIELDS,
        label="candidate/baseline known-test config",
    )
    if not isinstance(normalized["zero_adaptation"], bool):
        raise CLICTargetProtocolError("known-test zero_adaptation must be boolean")
    scenes = normalized["scenes"]
    if not isinstance(scenes, list) or tuple(str(item) for item in scenes) != FORMAL_LEO_WEAK_SCENARIOS:
        raise CLICTargetProtocolError("known-test LEO weak scene order/configuration drift")
    canonical_json_bytes(normalized)
    return normalized


def unwrap_manifest_normalized_config(payload: Mapping[str, Any], *, schema: str, label: str) -> tuple[dict[str, Any], str]:
    """Read a byte-sealed configuration manifest and verify its declared SHA."""

    if str(payload.get("schema", "")) != schema:
        raise CLICTargetProtocolError(f"{label} schema drift")
    normalized = _require_mapping(payload.get("normalized"), label=f"{label} normalized")
    declared_sha = require_sha256(payload.get("normalized_sha256"), label=f"{label} normalized")
    actual_sha = canonical_sha256(normalized)
    if actual_sha != declared_sha:
        raise CLICTargetProtocolError(f"{label} normalized SHA drift")
    return normalized, declared_sha


def read_verified_config_manifest(
    path: str | Path,
    *,
    expected_schema: str,
    expected_raw_sha256: str | None = None,
    label: str,
) -> dict[str, Any]:
    """Open a config manifest only after verifying raw and canonical identities."""

    source = Path(path).resolve()
    payload = read_json_object(source, label=label)
    raw_sha = sha256_file(source)
    if expected_raw_sha256 is not None:
        if raw_sha != require_sha256(expected_raw_sha256, label=f"{label} raw"):
            raise CLICTargetProtocolError(f"{label} raw SHA drift")
    normalized, declared_sha = unwrap_manifest_normalized_config(
        payload, schema=expected_schema, label=label
    )
    return {
        "path": str(source),
        "raw_sha256": raw_sha,
        "normalized": normalized,
        "normalized_sha256": declared_sha,
    }


def opaque_token(*, lineage_sha256: str, scene: str, ordinal: int, received_iq_sha256: str) -> str:
    """Generate a non-semantic token for an IQ-only predictor package row."""

    if str(scene) not in FORMAL_LEO_WEAK_SCENARIOS:
        raise CLICTargetProtocolError("opaque token scene is not a formal LEO weak scenario")
    if int(ordinal) < 0:
        raise CLICTargetProtocolError("opaque token ordinal must be nonnegative")
    return canonical_json_sha256(
        {
            "lineage_sha256": require_sha256(lineage_sha256, label="target lineage"),
            "ordinal": int(ordinal),
            "received_iq_sha256": require_sha256(
                received_iq_sha256, label="received IQ"
            ),
            "scene": str(scene),
        }
    )


def join_prediction_and_truth_by_opaque_token(
    prediction_rows: Any,
    truth_rows: Any,
) -> list[dict[str, Any]]:
    """Perform the scorer-only exact opaque-token join without interpreting data.

    This helper intentionally knows nothing about labels, roles, class order,
    scores, or metrics.  It preserves predictor row order, rejects duplicate or
    unmatched tokens, and requires the predictor-safe scene tag to agree with
    the evaluator-only sidecar before returning nested original row copies.
    """

    if not isinstance(prediction_rows, (list, tuple)) or not isinstance(truth_rows, (list, tuple)):
        raise CLICTargetProtocolError("opaque-token join requires prediction and truth row sequences")

    def indexed(rows: list[Any] | tuple[Any, ...], *, label: str) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for index, raw in enumerate(rows):
            row = _require_mapping(raw, label=f"opaque-token join {label} row {index}")
            token = require_sha256(row.get("opaque_token"), label=f"opaque-token join {label}")
            scene = str(row.get("scene", ""))
            if scene not in FORMAL_LEO_WEAK_SCENARIOS:
                raise CLICTargetProtocolError(
                    f"opaque-token join {label} row {index} has an invalid scene"
                )
            if token in result:
                raise CLICTargetProtocolError(f"opaque-token join has duplicate {label} token")
            result[token] = row
        return result

    prediction_by_token = indexed(prediction_rows, label="prediction")
    truth_by_token = indexed(truth_rows, label="truth")
    if set(prediction_by_token) != set(truth_by_token):
        raise CLICTargetProtocolError("opaque-token join has missing or extra token")
    joined: list[dict[str, Any]] = []
    for raw in prediction_rows:
        prediction = _require_mapping(raw, label="opaque-token join prediction row")
        token = str(prediction["opaque_token"])
        truth = truth_by_token[token]
        if str(prediction["scene"]) != str(truth["scene"]):
            raise CLICTargetProtocolError("opaque-token join scene mismatch")
        joined.append(
            {
                "opaque_token": token,
                "prediction": dict(prediction),
                "truth": dict(truth),
            }
        )
    return joined


def _resolve_descriptor_path(descriptor_path: Path, raw_path: Any, *, label: str) -> Path:
    candidate = Path(str(raw_path))
    resolved = candidate if candidate.is_absolute() else descriptor_path.parent / candidate
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    return resolved


def _load_bundle_member_json(
    bundle_path: str | Path,
    *,
    member_name: str,
    expected_sha256: str,
    label: str,
) -> dict[str, Any]:
    """Open one named immutable member after its byte digest has been bound."""

    source = Path(bundle_path).resolve()
    expected = require_sha256(expected_sha256, label=label)
    try:
        with zipfile.ZipFile(source, "r") as archive:
            names = archive.namelist()
            if names.count(member_name) != 1:
                raise CLICTargetProtocolError(f"{label} member is absent or duplicated")
            raw = archive.read(member_name)
    except (OSError, zipfile.BadZipFile) as exc:
        raise CLICTargetProtocolError(f"{label} bundle archive is unreadable") from exc
    if hashlib.sha256(raw).hexdigest() != expected:
        raise CLICTargetProtocolError(f"{label} member SHA drift")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CLICTargetProtocolError(f"{label} member JSON is invalid") from exc
    if not isinstance(payload, dict):
        raise CLICTargetProtocolError(f"{label} member must be a JSON object")
    canonical_json_bytes(payload)
    return dict(payload)


def _descriptor_fold(descriptor: Mapping[str, Any], checkpoint_path: Path) -> int:
    value = descriptor.get("fold_index")
    if value is not None:
        if type(value) is not int or value not in range(1, 7):
            raise CLICTargetProtocolError("C predictor descriptor fold_index is invalid")
        return int(value)
    match = re.fullmatch(r"F([1-6])C_CLIC12", checkpoint_path.parent.name)
    if match is None:
        raise CLICTargetProtocolError("C predictor descriptor/checkpoint fold binding is invalid")
    return int(match.group(1))


def _strict_received_iq(received_i: Any, *, input_len: int):
    """Materialize exactly one finite `[1,2,T]` received-IQ forward input."""

    import torch

    if torch.is_tensor(received_i):
        source = received_i.detach().cpu().float().contiguous()
        try:
            values = np.asarray(source.tolist(), dtype=np.float32)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise CLICTargetProtocolError(
                "target received-IQ tensor cannot cross the safe Torch/NumPy boundary"
            ) from exc
        if values.shape != tuple(source.shape) or not values.flags.c_contiguous:
            raise CLICTargetProtocolError("target received-IQ tensor shape/contiguity drift")
    else:
        try:
            values = np.asarray(received_i)
        except (TypeError, ValueError) as exc:
            raise CLICTargetProtocolError("target received-IQ cannot be materialized") from exc
    if values.dtype.kind != "f":
        raise CLICTargetProtocolError("target received-IQ must be floating point")
    if values.ndim == 2:
        values = values[None, ...]
    if values.shape != (1, 2, int(input_len)) or not np.isfinite(values).all():
        raise CLICTargetProtocolError("target received-IQ shape/non-finite drift")
    source = np.ascontiguousarray(values, dtype=np.float32)
    try:
        # Buffer protocol avoids the legacy NumPy ndarray C API; clone makes
        # the predictor input independent of the IQ-package allocation.
        tensor = torch.frombuffer(memoryview(source), dtype=torch.float32)
        return tensor.reshape(source.shape).clone()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise CLICTargetProtocolError(
            "target received-IQ safe NumPy/Torch conversion failed"
        ) from exc


def _tensor_to_numpy_float64(value: Any, *, label: str) -> np.ndarray:
    """Copy one model tensor without Tensor.numpy() on Torch 2.1/NumPy 2.x."""

    import torch

    if not torch.is_tensor(value):
        raise CLICTargetProtocolError(f"{label} is not a tensor")
    source = value.detach().cpu().contiguous()
    try:
        result = np.asarray(source.tolist(), dtype=np.float64)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise CLICTargetProtocolError(f"{label} safe tensor conversion failed") from exc
    if result.shape != tuple(source.shape) or not result.flags.c_contiguous:
        raise CLICTargetProtocolError(f"{label} tensor conversion shape/contiguity drift")
    if result.size <= 0 or not np.isfinite(result).all():
        raise CLICTargetProtocolError(f"{label} tensor conversion is empty or non-finite")
    return result


class _CLICTargetPredictorRuntime:
    """One sealed C/G predictor state with an IQ-only, no-update forward API."""

    def __init__(
        self,
        *,
        arm: str,
        operator: str,
        state_sha256: str,
        source_frozen_rule_sha256: str,
        train_config_manifest_path: str,
        train_config_raw_sha256: str,
        train_config_normalized_sha256: str,
        train_config_member_name: str | None,
        forward_impl,
        fold_index: int | None = None,
        source_class_order: Sequence[str] | None = None,
        source_class_order_sha256: str | None = None,
    ) -> None:
        self.arm = arm
        self.operator = operator
        self.state_sha256 = require_sha256(state_sha256, label="predictor state")
        self.source_frozen_rule_sha256 = require_sha256(
            source_frozen_rule_sha256, label="predictor source-frozen rule"
        )
        self.train_config_manifest_path = str(train_config_manifest_path)
        self.train_config_raw_sha256 = require_sha256(
            train_config_raw_sha256, label="predictor train config raw"
        )
        self.train_config_normalized_sha256 = require_sha256(
            train_config_normalized_sha256, label="predictor train config normalized"
        )
        self.train_config_member_name = train_config_member_name
        if fold_index is not None and (type(fold_index) is not int or fold_index not in range(1, 7)):
            raise CLICTargetProtocolError("target predictor fold_index is invalid")
        # The production C/G loaders always populate this from their sealed
        # source-policy state.  ``None`` remains only for deliberately
        # path-loaded test doubles; no production path treats it as a fold.
        self.fold_index = fold_index
        if (source_class_order is None) != (source_class_order_sha256 is None):
            raise CLICTargetProtocolError(
                "target predictor source class order/SHA binding is incomplete"
            )
        if source_class_order is None:
            self.source_class_order = None
            self.source_class_order_sha256 = None
        else:
            order = _validated_source_class_order(
                source_class_order,
                source_class_order_sha256,
                label="target predictor source class order",
            )
            self.source_class_order = list(order)
            self.source_class_order_sha256 = str(source_class_order_sha256)
        self._forward_impl = forward_impl

    def forward_once(self, received_i: Any, *, scene: str) -> dict[str, Any]:
        if scene not in FORMAL_LEO_WEAK_SCENARIOS:
            raise CLICTargetProtocolError("target predictor requires one formal LEO weak scene")
        output = self._forward_impl(received_i, scene=scene)
        if not isinstance(output, Mapping):
            raise CLICTargetProtocolError("target predictor forward output is not a mapping")
        required = {"z_id", "z_dom", "q_clic", "tx_logits", "e_unknown", "decision"}
        missing = required - set(output)
        if missing:
            raise CLICTargetProtocolError(f"target predictor forward output is incomplete: {sorted(missing)}")
        decision = str(output["decision"])
        if decision not in {"registered", "unknown", "defer"}:
            raise CLICTargetProtocolError("target predictor emitted an invalid decision")
        try:
            energy = float(output["e_unknown"])
        except (TypeError, ValueError) as exc:
            raise CLICTargetProtocolError("target predictor unknown energy is invalid") from exc
        if not math.isfinite(energy):
            raise CLICTargetProtocolError("target predictor unknown energy is non-finite")
        checked = dict(output)
        checked["e_unknown"] = energy
        checked["decision"] = decision
        return checked


def _write_new_immutable_json(path: Path, payload: Mapping[str, Any], *, label: str) -> None:
    """Create one deterministic JSON evidence artifact without replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json_bytes(dict(payload)).decode("utf-8"))
            handle.write("\n")
    except FileExistsError as exc:
        raise CLICTargetProtocolError(f"{label} already exists and is immutable: {path}") from exc


def _reopen_c_checkpoint_terminal(
    checkpoint_path: Path, terminal_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Strictly reopen C's final-only checkpoint and terminal before sealing/loading."""

    try:
        import torch
        import export_phase1_clic_features as clean

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("args"), Mapping):
            raise CLICTargetProtocolError("C predictor checkpoint payload is malformed")
        raw_args = checkpoint["args"]
        source_tx_ids = clean._parse_csv(
            raw_args.get("phase1_source_train_tx_ids", ""), label="C predictor source-L TX IDs"
        )
        known_validation_tx_ids = clean._parse_csv(
            raw_args.get("phase1_source_known_validation_tx_ids", ""),
            label="C predictor held-validation TX IDs",
        )
        proxy_unknown_tx_ids = clean._parse_csv(
            raw_args.get("phase1_source_proxy_unknown_tx_ids", ""),
            label="C predictor fixed-proxy TX IDs",
        )
        args, terminal, arm = clean.validate_clic_training_checkpoint(
            checkpoint,
            checkpoint_path=checkpoint_path,
            terminal_receipt_path=terminal_path,
            source_tx_ids=source_tx_ids,
            known_validation_tx_ids=known_validation_tx_ids,
            proxy_unknown_tx_ids=proxy_unknown_tx_ids,
        )
    except CLICTargetProtocolError:
        raise
    except Exception as exc:
        raise CLICTargetProtocolError("C predictor checkpoint/terminal strict reopen failed") from exc
    if arm != "C":
        raise CLICTargetProtocolError("C predictor checkpoint arm binding drifted")
    return (
        dict(checkpoint),
        dict(args),
        dict(terminal),
        tuple(str(value) for value in source_tx_ids),
        tuple(str(value) for value in known_validation_tx_ids),
        tuple(str(value) for value in proxy_unknown_tx_ids),
    )


def _reopen_c_pair_authority(
    pair_path: Path,
    *,
    checkpoint_path: Path,
    terminal_path: Path,
    checkpoint_sha256: str,
    terminal_receipt_sha256: str,
    fold_index: int,
    source_tx_ids: Sequence[str],
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    """Read PAIR's immutable C policy state, never an independent sidecar."""

    raw_before = sha256_file(pair_path)
    payload = read_json_object(pair_path, label="C predictor PAIR artifact")
    raw_after = sha256_file(pair_path)
    if raw_before != raw_after:
        raise CLICTargetProtocolError("C predictor PAIR artifact changed while opening")
    try:
        import evaluate_phase1_clic_postfreeze_pair as pair

        if (
            payload.get("schema") != _PAIR_SCHEMA
            or payload.get("same_fold") is not True
            or payload.get("source_only") is not True
            or payload.get("target_artifacts_present") is not False
            or payload.get("fold_index") != fold_index
        ):
            raise CLICTargetProtocolError("C predictor PAIR schema/fold/source-only binding drifted")
        if tuple(str(value) for value in payload.get("source_tx_ids", ())) != tuple(source_tx_ids):
            raise CLICTargetProtocolError("C predictor PAIR source-TX binding drifted")
        states = payload.get("clic_source_policy_state")
        raw_artifacts = payload.get("raw_artifacts")
        if (
            not isinstance(states, Mapping)
            or set(str(key) for key in states) != {"C", "G"}
            or not isinstance(raw_artifacts, Mapping)
            or set(str(key) for key in raw_artifacts) != {"C", "G"}
        ):
            raise CLICTargetProtocolError("C predictor PAIR C/G authority map is incomplete")
        raw_c = raw_artifacts.get("C")
        if not isinstance(raw_c, Mapping):
            raise CLICTargetProtocolError("C predictor PAIR C raw-artifact map is invalid")
        raw_c_paths = _verified_pair_raw_artifact_paths(
            pair_path, raw_c, label="C predictor PAIR"
        )
        paired_checkpoint = raw_c_paths["checkpoint"]
        paired_terminal = raw_c_paths["terminal"]
        if paired_checkpoint != checkpoint_path or paired_terminal != terminal_path:
            raise CLICTargetProtocolError("C predictor PAIR checkpoint/terminal path binding drifted")
        source_policy = pair._validated_clic_source_policy_state(
            states.get("C"),
            fold_index=fold_index,
            arm="C",
            checkpoint_sha256=checkpoint_sha256,
            terminal_receipt_sha256=terminal_receipt_sha256,
        )
    except CLICTargetProtocolError:
        raise
    except Exception as exc:
        raise CLICTargetProtocolError("C predictor PAIR source policy strict reopen failed") from exc
    return dict(payload), raw_before, dict(source_policy), dict(raw_c)


def _ordered_nonempty_strings(
    value: Any, *, label: str, reject_duplicates: bool = False
) -> list[str]:
    raw_rows = np.asarray(value)
    if raw_rows.ndim != 1 or raw_rows.size <= 0:
        raise CLICTargetProtocolError(f"{label} must be a nonempty one-dimensional identifier sequence")
    rows = np.asarray(raw_rows, dtype=str).reshape(-1)
    result: list[str] = []
    seen: set[str] = set()
    for raw in rows.tolist():
        item = str(raw)
        if not item:
            raise CLICTargetProtocolError(f"{label} contains an empty identifier")
        if item not in seen:
            seen.add(item)
            result.append(item)
        elif reject_duplicates:
            raise CLICTargetProtocolError(f"{label} contains duplicate identifiers")
    if not result:
        raise CLICTargetProtocolError(f"{label} is empty")
    return result


def _validated_source_class_order(
    value: Any, sha256: Any, *, label: str
) -> tuple[str, ...]:
    """Validate the sealed local-four source class order and its JSON root."""

    order = _ordered_nonempty_strings(value, label=label, reject_duplicates=True)
    if len(order) != 4:
        raise CLICTargetProtocolError(f"{label} must contain exactly the local-four source TX IDs")
    expected_sha = require_sha256(sha256, label=label)
    if canonical_sha256(order) != expected_sha:
        raise CLICTargetProtocolError(f"{label} SHA binding drifted")
    return tuple(order)


def _verified_pair_raw_artifact_paths(
    pair_path: Path, raw_artifacts: Mapping[str, Any], *, label: str
) -> dict[str, Path]:
    """Reopen every PAIR authority input under its immutable raw byte SHA."""

    expected_fields = set(_PAIR_RAW_ARTIFACT_STEMS) | {
        f"{stem}_sha256" for stem in _PAIR_RAW_ARTIFACT_STEMS
    }
    if set(str(key) for key in raw_artifacts) != expected_fields:
        raise CLICTargetProtocolError(f"{label} raw-artifact field set drifted")
    resolved: dict[str, Path] = {}
    for stem in _PAIR_RAW_ARTIFACT_STEMS:
        artifact = _resolve_descriptor_path(
            pair_path, raw_artifacts.get(stem), label=f"{label} {stem}"
        )
        expected_sha = require_sha256(
            raw_artifacts.get(f"{stem}_sha256"), label=f"{label} {stem}"
        )
        if sha256_file(artifact) != expected_sha:
            raise CLICTargetProtocolError(f"{label} {stem} raw artifact SHA drifted")
        resolved[stem] = artifact
    return resolved


def _c_train_config_from_pair_clean(
    *,
    pair_path: Path,
    pair_sha256: str,
    raw_c: Mapping[str, Any],
    checkpoint_path: Path,
    terminal_path: Path,
    checkpoint_sha256: str,
    terminal_receipt_sha256: str,
    checkpoint: Mapping[str, Any],
    args: Mapping[str, Any],
    source_tx_ids: Sequence[str],
    known_validation_tx_ids: Sequence[str],
    proxy_unknown_tx_ids: Sequence[str],
) -> dict[str, Any]:
    """Derive a source-only C training-config manifest from PAIR-bound clean evidence.

    Only aggregate receiver/day/TX sets and fixed preprocessing are sealed.  No
    source rows, sample IDs, features, logits, IQ, or target material leave the
    clean artifact.
    """

    clean_path = _resolve_descriptor_path(pair_path, raw_c.get("clean"), label="C predictor PAIR clean export")
    clean_before = sha256_file(clean_path)
    try:
        with np.load(clean_path, allow_pickle=False) as archive:
            required = {"manifest_json", "tx_ids", "rx_ids", "day_ids", "dataset_role"}
            if not required.issubset(set(archive.files)):
                raise CLICTargetProtocolError("C predictor PAIR clean export lacks aggregate metadata")
            raw_manifest = np.asarray(archive["manifest_json"])
            manifest = json.loads(str(raw_manifest.item()))
            tx_ids = np.asarray(archive["tx_ids"], dtype=str).reshape(-1)
            receiver_ids = np.asarray(archive["rx_ids"], dtype=str).reshape(-1)
            day_ids = np.asarray(archive["day_ids"], dtype=str).reshape(-1)
            roles = np.asarray(archive["dataset_role"], dtype=str).reshape(-1)
    except CLICTargetProtocolError:
        raise
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise CLICTargetProtocolError("C predictor PAIR clean export is unreadable") from exc
    clean_after = sha256_file(clean_path)
    if clean_before != clean_after:
        raise CLICTargetProtocolError("C predictor PAIR clean export changed while opening")
    if not isinstance(manifest, Mapping):
        raise CLICTargetProtocolError("C predictor PAIR clean manifest is invalid")
    if (
        manifest.get("schema") != "cvs.phase1.clic_lv_export.v1"
        or manifest.get("source_only") is not True
        or manifest.get("source_checkpoint_sha256") != checkpoint_sha256
        or manifest.get("terminal_receipt_sha256") != terminal_receipt_sha256
        or tuple(str(value) for value in manifest.get("source_tx_ids", ())) != tuple(source_tx_ids)
        or tuple(str(value) for value in manifest.get("known_validation_tx_ids", ())) != tuple(known_validation_tx_ids)
        or tuple(str(value) for value in manifest.get("proxy_unknown_tx_ids", ())) != tuple(proxy_unknown_tx_ids)
    ):
        raise CLICTargetProtocolError("C predictor PAIR clean manifest provenance binding drifted")
    if not (tx_ids.size and tx_ids.size == receiver_ids.size == day_ids.size == roles.size):
        raise CLICTargetProtocolError("C predictor PAIR clean aggregate metadata alignment drifted")
    labeled = roles == "labeled_fit"
    validation = roles == "source_validation_known"
    proxy = roles == "proxy_unknown"
    if not (np.any(labeled) and np.any(validation) and np.any(proxy)):
        raise CLICTargetProtocolError("C predictor PAIR clean source role coverage drifted")
    if set(tx_ids[labeled].tolist()) != set(source_tx_ids):
        raise CLICTargetProtocolError("C predictor PAIR clean source-L TX coverage drifted")
    if set(tx_ids[validation].tolist()) != set(source_tx_ids):
        raise CLICTargetProtocolError("C predictor PAIR clean source-V local4 TX coverage drifted")
    if set(tx_ids[proxy].tolist()) != set(proxy_unknown_tx_ids):
        raise CLICTargetProtocolError("C predictor PAIR clean proxy TX coverage drifted")

    source_receivers = _ordered_nonempty_strings(receiver_ids[labeled | validation], label="C predictor source receivers")
    source_days = _ordered_nonempty_strings(day_ids[labeled | validation], label="C predictor source days")
    split_info = checkpoint.get("split_info")
    checkpoint_split = split_info.get("source_split_receipt") if isinstance(split_info, Mapping) else None
    manifest_split = manifest.get("source_split_receipt")
    source_split = manifest_split if isinstance(manifest_split, Mapping) else checkpoint_split
    if isinstance(source_split, Mapping):
        if source_split.get("schema") != "cvs.phase1.source_split_receipt.v1":
            raise CLICTargetProtocolError("C predictor source split receipt schema drifted")
        receipt_receivers = _ordered_nonempty_strings(
            source_split.get("source_receivers"),
            label="C predictor source split receivers",
            reject_duplicates=True,
        )
        receipt_days = _ordered_nonempty_strings(
            source_split.get("source_days"),
            label="C predictor source split days",
            reject_duplicates=True,
        )
        if set(receipt_receivers) != set(source_receivers) or set(receipt_days) != set(source_days):
            raise CLICTargetProtocolError("C predictor source split/clean aggregate axis drifted")
        source_receivers = receipt_receivers
        source_days = receipt_days
    for field, observed in (("source_receiver_ids", source_receivers), ("source_day_ids", source_days)):
        if field in manifest:
            declared = _ordered_nonempty_strings(
                manifest.get(field),
                label=f"C predictor clean manifest {field}",
                reject_duplicates=True,
            )
            if declared != observed:
                raise CLICTargetProtocolError(f"C predictor clean manifest {field} binding drifted")

    # Real v5 source artifacts must state their data schema and input width.
    # The narrow split_info fallback exists only for the pre-v5 mechanical
    # fixture shape, whose terminal/checkpoint contract predates those args.
    legacy_checkpoint_copy = isinstance(checkpoint.get("split_info"), Mapping)
    raw_dataset = args.get("dataset")
    if raw_dataset is None and legacy_checkpoint_copy:
        raw_dataset = "wisig"
    if str(raw_dataset or "").casefold() != "wisig":
        raise CLICTargetProtocolError("C predictor training dataset schema drifted")
    dataset_provenance: dict[str, Any] = {"dataset_schema": "WiSig"}
    if args.get("wisig_pkl_sha256") is not None:
        dataset_provenance["wisig_pkl_sha256"] = require_sha256(
            args.get("wisig_pkl_sha256"), label="C predictor frozen WiSig dataset"
        )
    input_len = args.get("wisig_out_len")
    if input_len is None and legacy_checkpoint_copy:
        input_len = int(_clic.CLIC_INPUT_LENGTH)
    if type(input_len) is not int or int(input_len) != int(_clic.CLIC_INPUT_LENGTH):
        raise CLICTargetProtocolError("C predictor frozen input length drifted")
    normalized = {
        "dataset_provenance": dataset_provenance,
        "source_train_tx_ids": [str(value) for value in source_tx_ids],
        "source_validation_tx_ids": [str(value) for value in known_validation_tx_ids],
        "source_proxy_tx_ids": [str(value) for value in proxy_unknown_tx_ids],
        "source_receiver_ids": source_receivers,
        "source_day_ids": source_days,
        "split_mode": "tx_rx_day_1_6_3",
        "role_construction": {
            "split_mode": "tx_rx_day_1_6_3",
            "labeled_ratio": float(args.get("labeled_ratio")),
            "unlabeled_ratio": float(args.get("unlabeled_ratio")),
            "source_val_ratio": float(args.get("source_val_ratio")),
        },
        "physical_row_selection": {
            "selection_policy": "pre_registered_tx_rx_day_eq_split_by_sig_i",
            "group_axes": ["tx_id", "rx_id", "day_id", "eq_id"],
        },
        "preprocessing": {"input_len": int(input_len), "iq_dtype": "float32"},
        "single_leo_training_scenes": list(FORMAL_LEO_WEAK_SCENARIOS),
    }
    normalized = normalize_train_data_config(normalized)
    return {
        "schema": _TRAIN_CONFIG_SCHEMA,
        "checkpoint_sha256": checkpoint_sha256,
        "terminal_receipt_sha256": terminal_receipt_sha256,
        "pair_artifact_sha256": pair_sha256,
        "source_clean_npz_sha256": clean_before,
        "normalized": normalized,
        "normalized_sha256": canonical_sha256(normalized),
    }


def seal_clic_c_predictor_state(
    checkpoint_path: str | Path,
    terminal_receipt_path: str | Path,
    pair_artifact_path: str | Path,
    output_path: str | Path,
    *,
    fold_index: int,
) -> Path:
    """Seal one C predictor descriptor from the immutable PAIR output only."""

    if type(fold_index) is not int or fold_index not in range(1, 7):
        raise CLICTargetProtocolError("C predictor descriptor fold_index is invalid")
    checkpoint_file = Path(checkpoint_path).resolve()
    terminal_file = Path(terminal_receipt_path).resolve()
    pair_file = Path(pair_artifact_path).resolve()
    descriptor_file = Path(output_path).resolve()
    if not checkpoint_file.is_file() or not terminal_file.is_file() or not pair_file.is_file():
        raise FileNotFoundError("C predictor descriptor requires existing checkpoint, terminal, and PAIR artifacts")
    train_config_file = descriptor_file.with_name(descriptor_file.stem + ".train_config.json")
    if descriptor_file.exists() or train_config_file.exists():
        raise CLICTargetProtocolError("C predictor descriptor or its train config already exists and is immutable")
    checkpoint_sha = sha256_file(checkpoint_file)
    terminal_sha = sha256_file(terminal_file)
    checkpoint, args, _terminal, source_tx_ids, known_validation_tx_ids, proxy_unknown_tx_ids = _reopen_c_checkpoint_terminal(
        checkpoint_file, terminal_file
    )
    _pair_payload, pair_sha, source_policy, raw_c = _reopen_c_pair_authority(
        pair_file,
        checkpoint_path=checkpoint_file,
        terminal_path=terminal_file,
        checkpoint_sha256=checkpoint_sha,
        terminal_receipt_sha256=terminal_sha,
        fold_index=fold_index,
        source_tx_ids=source_tx_ids,
    )
    source_class_order = _validated_source_class_order(
        source_policy.get("geometry", {}).get("class_order"),
        canonical_sha256(source_policy.get("geometry", {}).get("class_order", ())),
        label="C predictor PAIR source class order",
    )
    if source_class_order != tuple(source_tx_ids):
        raise CLICTargetProtocolError("C predictor PAIR source class order/TX binding drifted")
    source_class_order_sha = canonical_sha256(source_class_order)
    train_config = _c_train_config_from_pair_clean(
        pair_path=pair_file,
        pair_sha256=pair_sha,
        raw_c=raw_c,
        checkpoint_path=checkpoint_file,
        terminal_path=terminal_file,
        checkpoint_sha256=checkpoint_sha,
        terminal_receipt_sha256=terminal_sha,
        checkpoint=checkpoint,
        args=args,
        source_tx_ids=source_tx_ids,
        known_validation_tx_ids=known_validation_tx_ids,
        proxy_unknown_tx_ids=proxy_unknown_tx_ids,
    )
    clean_file = _resolve_descriptor_path(
        pair_file, raw_c.get("clean"), label="C predictor PAIR clean export"
    )
    def assert_source_inputs_unchanged() -> None:
        # A changed source input must leave neither half of a descriptor pair
        # behind for later callers to misinterpret as sealed evidence.
        if (
            sha256_file(checkpoint_file) != checkpoint_sha
            or sha256_file(terminal_file) != terminal_sha
            or sha256_file(pair_file) != pair_sha
            or sha256_file(clean_file) != train_config["source_clean_npz_sha256"]
        ):
            raise CLICTargetProtocolError("C predictor descriptor source artifact changed while sealing")
        _verified_pair_raw_artifact_paths(
            pair_file, raw_c, label="C predictor PAIR"
        )

    train_config_created = False
    descriptor_created = False
    try:
        assert_source_inputs_unchanged()
        _write_new_immutable_json(train_config_file, train_config, label="C predictor train config")
        train_config_created = True
        assert_source_inputs_unchanged()
        train_config_raw_sha = sha256_file(train_config_file)
        descriptor = {
            "schema": PREDICTOR_STATE_SCHEMA,
            "arm": "C",
            "operator": "raw_phase_control",
            "fold_index": fold_index,
            "checkpoint_path": str(checkpoint_file),
            "checkpoint_sha256": checkpoint_sha,
            "terminal_receipt_path": str(terminal_file),
            "terminal_receipt_sha256": terminal_sha,
            "pair_artifact_path": str(pair_file),
            "pair_artifact_sha256": pair_sha,
            "pair_policy_state_sha256": source_policy["state_sha256"],
            "source_class_order": list(source_class_order),
            "source_class_order_sha256": source_class_order_sha,
            "train_config_manifest_path": str(train_config_file),
            "train_config_raw_sha256": train_config_raw_sha,
            "train_config_normalized_sha256": train_config["normalized_sha256"],
            "immutable": True,
        }
        _write_new_immutable_json(descriptor_file, descriptor, label="C predictor descriptor")
        descriptor_created = True
        assert_source_inputs_unchanged()
    except Exception:
        if descriptor_created and descriptor_file.is_file():
            descriptor_file.unlink()
        if train_config_created and train_config_file.is_file():
            train_config_file.unlink()
        raise
    return descriptor_file


def _load_c_predictor_state(path: Path) -> _CLICTargetPredictorRuntime:
    """Strictly reopen a C descriptor via its immutable PAIR authority."""

    descriptor = read_json_object(path, label="C predictor descriptor")
    required = {
        "schema",
        "arm",
        "operator",
        "fold_index",
        "checkpoint_path",
        "checkpoint_sha256",
        "terminal_receipt_path",
        "terminal_receipt_sha256",
        "pair_artifact_path",
        "pair_artifact_sha256",
        "pair_policy_state_sha256",
        "source_class_order",
        "source_class_order_sha256",
        "train_config_manifest_path",
        "train_config_raw_sha256",
        "train_config_normalized_sha256",
        "immutable",
    }
    if set(descriptor) != required:
        raise CLICTargetProtocolError("C predictor descriptor fields drift")
    if (
        descriptor.get("schema") != PREDICTOR_STATE_SCHEMA
        or descriptor.get("arm") != "C"
        or descriptor.get("operator") != "raw_phase_control"
        or descriptor.get("immutable") is not True
    ):
        raise CLICTargetProtocolError("C predictor descriptor schema/arm/operator/immutable binding drift")
    checkpoint_path = _resolve_descriptor_path(path, descriptor.get("checkpoint_path"), label="C predictor checkpoint")
    terminal_path = _resolve_descriptor_path(path, descriptor.get("terminal_receipt_path"), label="C predictor terminal")
    checkpoint_sha = require_sha256(descriptor.get("checkpoint_sha256"), label="C predictor checkpoint")
    terminal_sha = require_sha256(descriptor.get("terminal_receipt_sha256"), label="C predictor terminal")
    if sha256_file(checkpoint_path) != checkpoint_sha or sha256_file(terminal_path) != terminal_sha:
        raise CLICTargetProtocolError("C predictor checkpoint/terminal byte SHA drift")
    fold = _descriptor_fold(descriptor, checkpoint_path)
    checkpoint, checkpoint_args, _terminal, source_tx_ids, known_validation, proxy_unknown = _reopen_c_checkpoint_terminal(
        checkpoint_path, terminal_path
    )
    pair_path = _resolve_descriptor_path(path, descriptor.get("pair_artifact_path"), label="C predictor PAIR artifact")
    pair_sha = require_sha256(descriptor.get("pair_artifact_sha256"), label="C predictor PAIR artifact")
    if sha256_file(pair_path) != pair_sha:
        raise CLICTargetProtocolError("C predictor PAIR artifact byte SHA drift")
    _pair_payload, reopened_pair_sha, source_policy, _raw_c = _reopen_c_pair_authority(
        pair_path,
        checkpoint_path=checkpoint_path,
        terminal_path=terminal_path,
        checkpoint_sha256=checkpoint_sha,
        terminal_receipt_sha256=terminal_sha,
        fold_index=fold,
        source_tx_ids=source_tx_ids,
    )
    if reopened_pair_sha != pair_sha:
        raise CLICTargetProtocolError("C predictor PAIR artifact reopening SHA drift")
    if source_policy["state_sha256"] != require_sha256(
        descriptor.get("pair_policy_state_sha256"), label="C predictor PAIR source policy"
    ):
        raise CLICTargetProtocolError("C predictor PAIR source policy binding drift")
    source_class_order = _validated_source_class_order(
        descriptor.get("source_class_order"),
        descriptor.get("source_class_order_sha256"),
        label="C predictor descriptor source class order",
    )
    pair_source_class_order = _validated_source_class_order(
        source_policy.get("geometry", {}).get("class_order"),
        canonical_sha256(source_policy.get("geometry", {}).get("class_order", ())),
        label="C predictor PAIR source class order",
    )
    if source_class_order != pair_source_class_order or source_class_order != tuple(source_tx_ids):
        raise CLICTargetProtocolError("C predictor source class order PAIR/checkpoint binding drift")
    train_path = _resolve_descriptor_path(path, descriptor.get("train_config_manifest_path"), label="C predictor train config")
    train_raw_sha = require_sha256(descriptor.get("train_config_raw_sha256"), label="C predictor train config")
    train = read_verified_config_manifest(
        train_path,
        expected_schema=_TRAIN_CONFIG_SCHEMA,
        expected_raw_sha256=train_raw_sha,
        label="C predictor train config",
    )
    train_payload = read_json_object(train_path, label="C predictor train config")
    expected_train_fields = {
        "schema",
        "checkpoint_sha256",
        "terminal_receipt_sha256",
        "pair_artifact_sha256",
        "source_clean_npz_sha256",
        "normalized",
        "normalized_sha256",
    }
    if set(train_payload) != expected_train_fields:
        raise CLICTargetProtocolError("C predictor train config fields drift")
    if (
        train_payload.get("checkpoint_sha256") != checkpoint_sha
        or train_payload.get("terminal_receipt_sha256") != terminal_sha
        or train_payload.get("pair_artifact_sha256") != pair_sha
    ):
        raise CLICTargetProtocolError("C predictor train config PAIR/checkpoint binding drift")
    require_sha256(train_payload.get("source_clean_npz_sha256"), label="C predictor train config clean export")
    recomputed_train = _c_train_config_from_pair_clean(
        pair_path=pair_path,
        pair_sha256=pair_sha,
        raw_c=_raw_c,
        checkpoint_path=checkpoint_path,
        terminal_path=terminal_path,
        checkpoint_sha256=checkpoint_sha,
        terminal_receipt_sha256=terminal_sha,
        checkpoint=checkpoint,
        args=checkpoint_args,
        source_tx_ids=source_tx_ids,
        known_validation_tx_ids=known_validation,
        proxy_unknown_tx_ids=proxy_unknown,
    )
    if train_payload != recomputed_train:
        raise CLICTargetProtocolError("C predictor train config is not PAIR-derived authority")
    if (
        sha256_file(checkpoint_path) != checkpoint_sha
        or sha256_file(terminal_path) != terminal_sha
        or sha256_file(pair_path) != pair_sha
    ):
        raise CLICTargetProtocolError("C predictor source artifact changed during strict reopen")
    normalized_train = normalize_train_data_config(train["normalized"])
    data_sha = canonical_sha256(normalized_train)
    if data_sha != require_sha256(
        descriptor.get("train_config_normalized_sha256"), label="C predictor train config normalized"
    ):
        raise CLICTargetProtocolError("C predictor train config normalized SHA drift")
    try:
        import torch
        import evaluate_phase1_clic_postfreeze_pair as pair
        from cvsrffi.checkpoint_loading import build_exact_ssdg_model_from_checkpoint

        input_len = int(normalized_train["preprocessing"]["input_len"])
        if checkpoint["args"].get("wisig_out_len", input_len) != input_len:
            raise CLICTargetProtocolError("C predictor checkpoint/config input-length binding drift")
        model, _audit = build_exact_ssdg_model_from_checkpoint(
            checkpoint, input_len=input_len, device=torch.device("cpu")
        )
        model.eval()
        parameter = next(model.parameters())
    except CLICTargetProtocolError:
        raise
    except Exception as exc:
        raise CLICTargetProtocolError("C predictor strict checkpoint/model reopen failed") from exc

    def forward(received_i: Any, *, scene: str) -> dict[str, Any]:
        tensor = _strict_received_iq(received_i, input_len=input_len).to(
            dtype=parameter.dtype, device=torch.device("cpu")
        )
        try:
            with torch.no_grad():
                output = model(tensor, y_tx=None, grl_lambda=1.0, return_aux=True)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise CLICTargetProtocolError("C predictor received-IQ forward failed") from exc
        if not isinstance(output, Mapping):
            raise CLICTargetProtocolError("C predictor model forward output is invalid")
        try:
            scored = pair.score_clic_open_set(
                source_policy["geometry"],
                source_policy["policies"][scene],
                _tensor_to_numpy_float64(output["z_id"], label="C predictor z_id"),
                _tensor_to_numpy_float64(
                    output["tx_logits"], label="C predictor tx_logits"
                ),
                scene,
            )
        except Exception as exc:
            raise CLICTargetProtocolError("C predictor source-frozen scoring failed") from exc
        return {
            "z_id": _tensor_to_numpy_float64(output["z_id"], label="C predictor z_id"),
            "z_dom": _tensor_to_numpy_float64(output["z_dom"], label="C predictor z_dom"),
            "q_clic": _tensor_to_numpy_float64(output["q_clic"], label="C predictor q_clic"),
            "tx_logits": _tensor_to_numpy_float64(
                output["tx_logits"], label="C predictor tx_logits"
            ),
            "e_unknown": float(np.asarray(scored["e_unknown"], dtype=np.float64).reshape(-1)[0]),
            "decision": str(np.asarray(scored["decision"]).reshape(-1)[0]),
        }

    return _CLICTargetPredictorRuntime(
        arm="C",
        operator="raw_phase_control",
        state_sha256=canonical_sha256(
            {
                "descriptor_sha256": sha256_file(path),
                "pair_artifact_sha256": pair_sha,
                "source_policy_state_sha256": source_policy["state_sha256"],
            }
        ),
        source_frozen_rule_sha256=source_policy["state_sha256"],
        train_config_manifest_path=train["path"],
        train_config_raw_sha256=train["raw_sha256"],
        train_config_normalized_sha256=data_sha,
        train_config_member_name=None,
        forward_impl=forward,
        fold_index=fold,
        source_class_order=source_class_order,
        source_class_order_sha256=descriptor["source_class_order_sha256"],
    )


def _load_g_predictor_bundle(path: Path) -> _CLICTargetPredictorRuntime:
    """Open G only through the verified deployment bundle/reload API."""

    try:
        import export_phase1_clic_deployment_bundle as bundle

        verified = bundle.verify_clic_bundle(path)
    except Exception as exc:
        raise CLICTargetProtocolError("G predictor deployment bundle strict verification failed") from exc
    if (
        verified.get("state_origin") != "checkpoint_model_exact"
        or verified.get("real_checkpoint_state_rebuild_verified") is not True
        or verified.get("real_checkpoint_reload_verified") is not False
    ):
        raise CLICTargetProtocolError("G predictor requires a real verified deployment bundle")
    candidate = verified.get("candidate_train_data_config")
    if not isinstance(candidate, Mapping) or candidate.get("real_checkpoint_config") is not True:
        raise CLICTargetProtocolError("G predictor bundle candidate train config is not real")
    normalized = candidate.get("normalized")
    if not isinstance(normalized, Mapping):
        raise CLICTargetProtocolError("G predictor bundle candidate train config is malformed")
    # The bundle keeps `input_len` redundantly for audit readability.  The
    # target comparison normalizer consumes the canonical preprocessing map.
    normalized_for_target = dict(normalized)
    normalized_for_target.pop("input_len", None)
    try:
        data_sha = canonical_sha256(normalize_train_data_config(normalized_for_target))
    except CLICTargetProtocolError as exc:
        raise CLICTargetProtocolError("G predictor bundle train data config failed normalization") from exc
    member_name = str(verified.get("train_config_member_name", ""))
    raw_sha = str(verified.get("train_config_raw_sha256", ""))
    if member_name != "candidate_train_data_config.json":
        raise CLICTargetProtocolError("G predictor bundle train config member binding drift")
    # Reopen this member once more through its descriptor before an IQ package
    # can be opened; later scorer reopening repeats the same byte check.
    reopened = _load_bundle_member_json(
        path,
        member_name=member_name,
        expected_sha256=raw_sha,
        label="G predictor candidate train config",
    )
    if reopened != candidate:
        raise CLICTargetProtocolError("G predictor bundle train config member/verification drift")
    if candidate.get("normalized_sha256") != canonical_sha256(dict(normalized)):
        raise CLICTargetProtocolError("G predictor bundle train config manifest normalized SHA drift")
    source_policy = verified.get("clic_source_policy_state")
    if not isinstance(source_policy, Mapping) or type(source_policy.get("fold_index")) is not int:
        raise CLICTargetProtocolError("G predictor bundle source-policy fold binding drift")
    source_class_order = _validated_source_class_order(
        verified.get("source_class_order"),
        verified.get("source_class_order_sha256"),
        label="G predictor bundle source class order",
    )
    pair_geometry = source_policy.get("geometry")
    if not isinstance(pair_geometry, Mapping):
        raise CLICTargetProtocolError("G predictor bundle source-policy geometry is invalid")
    pair_source_class_order = _validated_source_class_order(
        pair_geometry.get("class_order"),
        canonical_sha256(pair_geometry.get("class_order", ())),
        label="G predictor PAIR source class order",
    )
    if source_class_order != pair_source_class_order:
        raise CLICTargetProtocolError("G predictor bundle source class order/policy binding drifted")
    fold = int(source_policy["fold_index"])
    if fold not in range(1, 7):
        raise CLICTargetProtocolError("G predictor bundle source-policy fold is invalid")

    def forward(received_i: Any, *, scene: str) -> dict[str, Any]:
        try:
            return dict(bundle.reload_forward(path, received_i, scene=scene))
        except Exception as exc:
            raise CLICTargetProtocolError("G predictor verified received-IQ forward failed") from exc

    return _CLICTargetPredictorRuntime(
        arm="G",
        operator="complex_local_invariant_curvature",
        state_sha256=str(verified["state_sha256"]),
        source_frozen_rule_sha256=str(verified["source_frozen_unknown_rule_sha256"]),
        train_config_manifest_path=str(path.resolve()),
        train_config_raw_sha256=raw_sha,
        train_config_normalized_sha256=data_sha,
        train_config_member_name=member_name,
        forward_impl=forward,
        fold_index=fold,
        source_class_order=source_class_order,
        source_class_order_sha256=verified["source_class_order_sha256"],
    )


def load_verified_clic_predictor_state(path: str | Path) -> _CLICTargetPredictorRuntime:
    """Load a predictor from one immutable path, never a caller-injected state."""

    if not isinstance(path, (str, Path)):
        raise CLICTargetProtocolError("CLIC predictor input must be one immutable path artifact")
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"CLIC predictor artifact is missing: {source}")
    if source.suffix.lower() == ".zip":
        return _load_g_predictor_bundle(source)
    return _load_c_predictor_state(source)


def build_c_predictor_state_parser() -> argparse.ArgumentParser:
    """Build the deliberately narrow, path-only C descriptor sealer CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal-c-predictor-state", action="store_true")
    parser.add_argument("--checkpoint", "--checkpoint-path", dest="checkpoint_path")
    parser.add_argument(
        "--terminal-receipt-json",
        "--terminal-receipt-path",
        dest="terminal_receipt_path",
    )
    parser.add_argument(
        "--pair-artifact-json", "--pair-artifact-path", dest="pair_artifact_path"
    )
    parser.add_argument("--output", "--output-path", dest="output_path")
    parser.add_argument("--fold-index", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Seal one C predictor descriptor; no model/state/config injection exists."""

    parser = build_c_predictor_state_parser()
    args = parser.parse_args(argv)
    if args.seal_c_predictor_state is not True:
        parser.error("--seal-c-predictor-state is required")
    missing = [
        flag
        for flag, value in (
            ("--checkpoint", args.checkpoint_path),
            ("--terminal-receipt-json", args.terminal_receipt_path),
            ("--pair-artifact-json", args.pair_artifact_path),
            ("--output", args.output_path),
            ("--fold-index", args.fold_index),
        )
        if value is None
    ]
    if missing:
        parser.error("C predictor descriptor sealer is missing required arguments: " + ", ".join(missing))
    result = seal_clic_c_predictor_state(
        args.checkpoint_path,
        args.terminal_receipt_path,
        args.pair_artifact_path,
        args.output_path,
        fold_index=args.fold_index,
    )
    print(str(result))
    return 0


__all__ = [
    "ADV3B02_METRICS_SCHEMA",
    "ADV3B02_REFERENCE_SCHEMA",
    "CLICTargetGateError",
    "CLICTargetProtocolError",
    "PREDICTOR_STATE_SCHEMA",
    "TARGET_PACKAGE_SCHEMA",
    "TARGET_TRUTH_SCHEMA",
    "canonical_json_bytes",
    "canonical_sha256",
    "build_c_predictor_state_parser",
    "join_prediction_and_truth_by_opaque_token",
    "load_verified_clic_predictor_state",
    "main",
    "normalize_known_test_config",
    "normalize_train_data_config",
    "opaque_token",
    "read_json_object",
    "read_verified_config_manifest",
    "require_sha256",
    "seal_clic_c_predictor_state",
]


if __name__ == "__main__":
    raise SystemExit(main())
