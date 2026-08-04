"""Immutable truth-free prepare/smoke/shard/merge runner for D108 Target125.

The real materializer reopens only the sealed D92 packages, runs their sealed
TorchScript identity runtime, and reconstructs the exact D92 288-dimensional
``registered_feature``.  Support inference uses batches of 64 and every query
row is forwarded alone.  Eight immutable shards are required; only an exact,
duplicate-free merge may publish the 3,000-surface sealed manifest.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any

import numpy as np

from .stage2_d108_matrix_protocol import (
    ACCESS_LEDGER,
    ARMS,
    ARM_PAIR_COUNT,
    CANDIDATE_ID,
    FEATURE_WIDTH,
    OLD_CLASS_COUNT,
    OUTER_JOB_COUNT,
    PHASES,
    PROTOCOL_SCHEMA,
    SCENES,
    SCENE_ROW_COUNT,
    SURFACE_COUNT,
    audit_surface_coverage,
    canonical_bytes,
    canonical_sha256,
    freeze_d108_target125_matrix,
)
from .stage2_d108_target125_inputs import (
    CONTEXT_SCHEMA,
    PLAN_SCHEMA,
    prepare_d108_target125_inputs,
)


PREDICTION_MANIFEST_SCHEMA = (
    "cvs.phase2.d108.cbrrc_smme.target125.prediction_manifest.v1"
)
PREDICTION_ARTIFACT_SCHEMA = (
    "cvs.phase2.d108.cbrrc_smme.target125.prediction_artifact.v1"
)
PREDICTION_SHARD_SCHEMA = (
    "cvs.phase2.d108.cbrrc_smme.target125.prediction_shard.v1"
)
SMOKE_RECEIPT_SCHEMA = "cvs.phase2.d108.cbrrc_smme.target125.smoke_receipt.v1"
SMOKE_PREDICTIONS_SCHEMA = (
    "cvs.phase2.d108.cbrrc_smme.target125.smoke_predictions.v1"
)
SHARD_COUNT = 8

_PLAN_FIELDS = {
    "schema", "candidate_id", "protocol_schema", "matrix_protocol",
    "identity", "rows", "plan_receipt_sha256",
}
_CONTEXT_FIELDS = {
    "schema", "candidate_id", "protocol_schema", "plan_receipt_sha256",
    "identity", "rows", "context_receipt_sha256",
}
_IDENTITY_FIELDS = {
    "matrix_receipt_sha256", "d92_matrix_manifest", "d92_output_root",
    "d92_sealed_runtime_sha256", "checkpoint", "d108_method_lock",
    "ground_component",
}
_ROW_FIELDS = {
    "outer_id", "source_d92_job_id", "receiver", "seed", "k_shot",
    "active_k", "new_count", "source_pool_k", "k5_prefix_from_matched_k10",
    "packages", "authority_bundle",
}
_PACKAGE_FIELDS = {"package_root", "detached_seal_path", "expected_seal_sha256"}
_AUTHORITY_FIELDS = {"directory", "commit_path", "commit_sha256"}
_MATERIALIZED_FIELDS = {
    "support_features", "support_labels", "registered_classes",
    "support_physical_ids", "query_features", "query_physical_ids",
}
_OUTER_ROW_FIELDS = {
    "outer_id", "receiver", "seed", "k_shot", "new_count",
    "old_classes", "new_classes",
}
_SURFACE_FIELDS = {
    "surface_id", "outer_id", "receiver", "seed", "k_shot", "new_count",
    "scene", "arm", "phase", "registered_classes", "prediction_artifact",
    "prediction_artifact_sha256", "ordered_query_physical_ids",
    "ordered_query_physical_ids_sha256", "predicted_labels",
    "predicted_labels_sha256", "access_ledger", "truth_open", "immutable",
}
_ARTIFACT_FIELDS = (
    _SURFACE_FIELDS - {"prediction_artifact", "prediction_artifact_sha256"}
) | {"schema", "artifact_receipt_sha256"}
_MANIFEST_FIELDS = {
    "schema", "candidate_id", "protocol_schema", "manifest_sealed",
    "truth_open", "outer_job_count", "scene_row_count", "arm_pair_count",
    "surface_count", "scenes", "arms", "phases", "outer_rows",
    "access_ledger", "shard_count", "shard_receipts", "surfaces",
    "manifest_sha256",
}


class D108Target125RunnerError(ValueError):
    """Raised when D108 execution cannot preserve its sealed boundary."""


StateMaterializer = Callable[[Mapping[str, Any]], Mapping[str, Any]]
PairBuilder = Callable[..., Any]
QueryScorer = Callable[..., np.ndarray]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise D108Target125RunnerError(f"{name} must be a lowercase SHA256")
    return value


def _regular_file(path: Path, name: str) -> Path:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise D108Target125RunnerError(f"{name} must be a regular non-symlink file")
    return source.resolve(strict=True)


def _read_json(
    path: Path, *, name: str, expected_file_sha256: str | None = None
) -> dict[str, Any]:
    source = _regular_file(path, name)
    if expected_file_sha256 is not None and _sha256_file(source) != _sha(
        expected_file_sha256, f"expected {name} SHA256"
    ):
        raise D108Target125RunnerError(f"{name} SHA mismatch")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise D108Target125RunnerError(f"{name} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise D108Target125RunnerError(f"{name} must contain an object")
    return value


def _write_bytes_new(path: Path, raw: bytes) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"immutable output already exists: {path}")
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, stat.S_IREAD)
    return hashlib.sha256(raw).hexdigest()


def _write_json_new(path: Path, value: Mapping[str, Any]) -> str:
    return _write_bytes_new(path, canonical_bytes(value) + b"\n")


def _receipt(document: Mapping[str, Any], field: str, name: str) -> str:
    receipt = _sha(document.get(field), f"{name} {field}")
    if canonical_sha256({key: value for key, value in document.items() if key != field}) != receipt:
        raise D108Target125RunnerError(f"{name} canonical receipt drift")
    return receipt


def _tokens(
    value: Any, name: str, rows: int, *, require_unique: bool = True
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != rows:
        raise D108Target125RunnerError(f"{name} must align to its rows")
    result = tuple(str(item) for item in value)
    if any(not item for item in result) or (require_unique and len(set(result)) != len(result)):
        raise D108Target125RunnerError(f"{name} token closure drift")
    return result


def _access_ledger(value: Any) -> dict[str, bool]:
    if not isinstance(value, Mapping) or set(value) != set(ACCESS_LEDGER):
        raise D108Target125RunnerError("access-ledger field closure drift")
    if any(value.get(name) is not False for name in ACCESS_LEDGER):
        raise D108Target125RunnerError("access ledger grants forbidden access")
    return dict(ACCESS_LEDGER)


def _prepared_inputs(
    *,
    plan_manifest_path: Path,
    expected_plan_file_sha256: str,
    context_manifest_path: Path,
    expected_context_file_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = _read_json(
        plan_manifest_path, name="D108 Target125 plan",
        expected_file_sha256=expected_plan_file_sha256,
    )
    context = _read_json(
        context_manifest_path, name="D108 Target125 context",
        expected_file_sha256=expected_context_file_sha256,
    )
    if set(plan) != _PLAN_FIELDS or set(context) != _CONTEXT_FIELDS:
        raise D108Target125RunnerError("prepared plan/context field closure drift")
    if (
        plan.get("schema") != PLAN_SCHEMA
        or context.get("schema") != CONTEXT_SCHEMA
        or plan.get("candidate_id") != CANDIDATE_ID
        or context.get("candidate_id") != CANDIDATE_ID
        or plan.get("protocol_schema") != PROTOCOL_SCHEMA
        or context.get("protocol_schema") != PROTOCOL_SCHEMA
    ):
        raise D108Target125RunnerError("prepared plan/context identity drift")
    plan_receipt = _receipt(plan, "plan_receipt_sha256", "plan")
    _receipt(context, "context_receipt_sha256", "context")
    matrix = freeze_d108_target125_matrix()
    if (
        plan.get("matrix_protocol") != matrix.receipt_payload()
        or context.get("plan_receipt_sha256") != plan_receipt
        or plan.get("identity") != context.get("identity")
    ):
        raise D108Target125RunnerError("prepared matrix/identity binding drift")
    identity = plan.get("identity")
    if not isinstance(identity, Mapping) or set(identity) != _IDENTITY_FIELDS:
        raise D108Target125RunnerError("prepared identity field closure drift")
    if identity.get("matrix_receipt_sha256") != matrix.matrix_receipt_sha256:
        raise D108Target125RunnerError("prepared matrix receipt drift")
    _sha(identity.get("d92_sealed_runtime_sha256"), "D92 sealed runtime SHA256")
    for name in ("checkpoint", "d108_method_lock"):
        ref = identity.get(name)
        if not isinstance(ref, Mapping) or set(ref) != {"path", "sha256"}:
            raise D108Target125RunnerError(f"prepared {name} binding drift")
        _sha(ref.get("sha256"), f"{name} SHA256")
    ground = identity.get("ground_component")
    if not isinstance(ground, Mapping) or set(ground) != {
        "directory", "manifest_path", "manifest_sha256"
    }:
        raise D108Target125RunnerError("prepared ground-component binding drift")
    _sha(ground.get("manifest_sha256"), "ground manifest SHA256")
    rows = plan.get("rows")
    if not isinstance(rows, list) or len(rows) != OUTER_JOB_COUNT or rows != context.get("rows"):
        raise D108Target125RunnerError("prepared row closure drift")
    for row, expected in zip(rows, matrix.outer_rows, strict=True):
        if not isinstance(row, Mapping) or set(row) != _ROW_FIELDS:
            raise D108Target125RunnerError("prepared row field closure drift")
        if any(row.get(name) != getattr(expected, name) for name in (
            "outer_id", "receiver", "seed", "k_shot", "new_count"
        )) or row.get("active_k") != expected.k_shot:
            raise D108Target125RunnerError("prepared row matrix binding drift")
        expected_pool = 10 if (expected.k_shot, expected.new_count) == (5, 20) else expected.k_shot
        if row.get("source_pool_k") != expected_pool:
            raise D108Target125RunnerError("prepared source-pool K drift")
        packages = row.get("packages")
        if not isinstance(packages, Mapping) or set(packages) != {
            "before_enrollment", "before_apply", "after_enrollment", "after_apply"
        }:
            raise D108Target125RunnerError("prepared four-package closure drift")
        for reference in packages.values():
            if not isinstance(reference, Mapping) or set(reference) != _PACKAGE_FIELDS:
                raise D108Target125RunnerError("prepared package-reference drift")
            _sha(reference.get("expected_seal_sha256"), "package seal SHA256")
        authority = row.get("authority_bundle")
        if not isinstance(authority, Mapping) or set(authority) != _AUTHORITY_FIELDS:
            raise D108Target125RunnerError("prepared per-row authority binding drift")
        _sha(authority.get("commit_sha256"), "authority COMMIT SHA256")
    return plan, context


def prepare_d108_target125_run(**kwargs: Any) -> dict[str, Any]:
    return prepare_d108_target125_inputs(**kwargs)


def _readonly_features(
    value: Any, name: str, *, feature_width: int = FEATURE_WIDTH
) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.dtype != np.float32 or array.ndim != 2 or array.shape[0] < 1
        or array.shape[1] != feature_width or not np.isfinite(array).all()
    ):
        raise D108Target125RunnerError(
            f"{name} must be finite float32 [N,{feature_width}]"
        )
    copied = np.ascontiguousarray(array, dtype=np.float32).copy()
    norms = np.linalg.norm(copied.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, rtol=1.0e-5, atol=1.0e-5):
        raise D108Target125RunnerError(f"{name} must retain D92 row normalization")
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True, slots=True)
class _MaterializedState:
    support_features: np.ndarray
    support_labels: tuple[str, ...]
    registered_classes: tuple[str, ...]
    support_physical_ids: tuple[str, ...]
    query_features: np.ndarray
    query_physical_ids: tuple[str, ...]


def _coerce_materialized_state(
    value: Mapping[str, Any], *, request: Mapping[str, Any], feature_width: int = FEATURE_WIDTH
) -> _MaterializedState:
    if not isinstance(value, Mapping) or set(value) != _MATERIALIZED_FIELDS:
        raise D108Target125RunnerError("state materializer field closure drift")
    if any(any(token in str(name).lower() for token in (
        "truth", "role", "metric", "score", "quota"
    )) for name in value):
        raise D108Target125RunnerError("state materializer exposed forbidden data")
    support = _readonly_features(
        value["support_features"], "support_features", feature_width=feature_width
    )
    query = _readonly_features(
        value["query_features"], "query_features", feature_width=feature_width
    )
    registered = _tokens(
        value["registered_classes"], "registered_classes",
        len(value["registered_classes"]),
    )
    labels = _tokens(
        value["support_labels"], "support_labels", len(support), require_unique=False
    )
    active_k = request.get("k_shot")
    if (
        type(active_k) is not int or any(label not in registered for label in labels)
        or any(labels.count(label) != active_k for label in registered)
    ):
        raise D108Target125RunnerError("support does not close balanced active K")
    support_ids = _tokens(value["support_physical_ids"], "support physical IDs", len(support))
    query_ids = _tokens(value["query_physical_ids"], "query physical IDs", len(query))
    if set(support_ids).intersection(query_ids):
        raise D108Target125RunnerError("support/query physical IDs overlap")
    return _MaterializedState(
        support, labels, registered, support_ids, query, query_ids
    )


def _verify_bound_file(reference: Mapping[str, Any], *, path_key: str, sha_key: str, name: str) -> Path:
    source = _regular_file(Path(str(reference.get(path_key))), name)
    if _sha256_file(source) != _sha(reference.get(sha_key), f"{name} SHA256"):
        raise D108Target125RunnerError(f"{name} SHA drift")
    return source


def _package_payloads(reference: Mapping[str, Any]):
    try:
        from .somph_diagnostic_bundle_loader import load_verified_somph_predictor_bundle
        payloads, manifest, audit = load_verified_somph_predictor_bundle(
            Path(str(reference["package_root"])),
            detached_seal_path=Path(str(reference["detached_seal_path"])),
            expected_seal_sha256=_sha(
                reference["expected_seal_sha256"], "D92 package seal SHA256"
            ),
        )
    except Exception as error:
        raise D108Target125RunnerError("sealed D92 package verification failed") from error
    if not isinstance(payloads, Mapping) or not isinstance(manifest, Mapping):
        raise D108Target125RunnerError("sealed D92 package materialization drift")
    return payloads, dict(manifest), dict(audit)


def _support_rows(
    payload: Mapping[str, Any], *, registered_classes: tuple[str, ...], active_k: int
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    try:
        from .somph_predictor_bundle import SUPPORT_NPZ_MEMBERS
    except Exception as error:  # pragma: no cover - import closure
        raise D108Target125RunnerError("D92 support schema unavailable") from error
    if set(payload) != set(SUPPORT_NPZ_MEMBERS) - {"manifest_json"}:
        raise D108Target125RunnerError("D92 support payload allowlist drift")
    ranks = np.asarray(payload["support_rank_within_class"])
    indices = np.asarray(payload["support_class_indices"])
    ids = tuple(np.asarray(payload["support_tokens"]).astype(str).tolist())
    iq = np.asarray(payload["support_leo_weak_iq"])
    if (
        ranks.dtype.kind not in "iu" or indices.dtype.kind not in "iu"
        or ranks.ndim != 1 or ranks.shape != indices.shape or len(ids) != len(ranks)
        or iq.dtype != np.float32 or iq.ndim != 3 or iq.shape[1] != 2
        or len(iq) != len(ranks) or not np.isfinite(iq).all()
    ):
        raise D108Target125RunnerError("D92 support IQ/index contract drift")
    mask = ranks.astype(np.int64) < active_k
    selected = indices.astype(np.int64)[mask]
    if (
        len(selected) != active_k * len(registered_classes) or len(selected) == 0
        or int(selected.min()) != 0 or int(selected.max()) != len(registered_classes) - 1
        or any(int(np.sum(selected == index)) != active_k for index in range(len(registered_classes)))
    ):
        raise D108Target125RunnerError("D92 support balanced active-K drift")
    selected_ids = tuple(item for item, use in zip(ids, mask, strict=True) if use)
    if len(set(selected_ids)) != len(selected_ids) or any(not item for item in selected_ids):
        raise D108Target125RunnerError("D92 support physical-ID drift")
    labels = tuple(registered_classes[index] for index in selected.tolist())
    return np.ascontiguousarray(iq[mask], dtype=np.float32), labels, selected_ids


def _query_rows(payload: Mapping[str, Any]) -> tuple[np.ndarray, tuple[str, ...]]:
    try:
        from .somph_predictor_bundle import QUERY_NPZ_MEMBERS
    except Exception as error:  # pragma: no cover - import closure
        raise D108Target125RunnerError("D92 query schema unavailable") from error
    if set(payload) != set(QUERY_NPZ_MEMBERS) - {"manifest_json"}:
        raise D108Target125RunnerError(
            "D92 query payload allowlist drift; truth/role fields are forbidden"
        )
    iq = np.asarray(payload["query_leo_weak_iq"])
    ids = tuple(np.asarray(payload["query_tokens"]).astype(str).tolist())
    if (
        iq.dtype != np.float32 or iq.ndim != 3 or iq.shape[1] != 2
        or len(iq) != len(ids) or len(iq) < 1 or not np.isfinite(iq).all()
        or len(set(ids)) != len(ids) or any(not item for item in ids)
    ):
        raise D108Target125RunnerError("D92 query IQ/ID contract drift")
    return np.ascontiguousarray(iq, dtype=np.float32), ids


class _D108RealStateMaterializer:
    """Strict D92 sealed-runtime adapter; query forwards are singleton-only."""

    def __init__(self, *, plan: Mapping[str, Any], device: str, support_batch_size: int) -> None:
        if type(support_batch_size) is not int or support_batch_size != 64:
            raise D108Target125RunnerError("D108 support_batch_size must equal 64")
        self.plan = plan
        self.support_batch_size = 64
        self.package_cache: dict[tuple[tuple[str, str], ...], tuple[Any, dict[str, Any], dict[str, Any]]] = {}
        self.model_cache: dict[tuple[str, str], Any] = {}
        identity = plan["identity"]
        _verify_bound_file(identity["checkpoint"], path_key="path", sha_key="sha256", name="checkpoint")
        _verify_bound_file(identity["d108_method_lock"], path_key="path", sha_key="sha256", name="D108 method lock")
        ground = identity["ground_component"]
        _verify_bound_file(ground, path_key="manifest_path", sha_key="manifest_sha256", name="ground manifest")
        try:
            from .stage2_diag_cosine_exploration import _device
            self.device = _device(device)
            from . import stage2_d42_unified_shrinkage_lda as d42
            from scripts.probe_d92_registration_balanced_covariance import (
                build_d92_fit,
                load_ground_basis,
            )
            basis, spectral, ground_audit = load_ground_basis(
                Path(str(ground["directory"])),
                str(ground["manifest_sha256"]),
                int(d42.FEATURE_DIM),
            )
            self.d92_fit, self.d92_transform_records, self.d92_component_records = (
                build_d92_fit(d42, basis, spectral, ground_audit)
            )
        except Exception as error:  # pragma: no cover - environment closure
            raise D108Target125RunnerError("D92 runtime/ground fit is unavailable") from error

    def _package(self, reference: Mapping[str, Any]):
        key = tuple(sorted((str(name), str(value)) for name, value in reference.items()))
        if key not in self.package_cache:
            try:
                self.package_cache[key] = _package_payloads(reference)
            except Exception as error:
                raise D108Target125RunnerError("sealed D92 package verification failed") from error
        return self.package_cache[key]

    def _model(self, package_root: str, manifest: Mapping[str, Any]):
        try:
            from .stage2_diag_cosine_exploration import _descriptor
            from .stage2_predictor_runtime import load_torchscript_backbone_same_fd
            descriptor = _descriptor(manifest, "feature_runtime")
            runtime_sha = _sha(descriptor.get("sha256"), "sealed runtime SHA256")
            if runtime_sha != self.plan["identity"]["d92_sealed_runtime_sha256"]:
                raise D108Target125RunnerError("D92 sealed runtime identity drift")
            key = (str(package_root), runtime_sha)
            if key not in self.model_cache:
                self.model_cache[key] = load_torchscript_backbone_same_fd(
                    package_root, descriptor, device=self.device
                )
            return self.model_cache[key]
        except D108Target125RunnerError:
            raise
        except Exception as error:
            raise D108Target125RunnerError("D92 sealed runtime load failed") from error

    def _features(
        self, *, iq: np.ndarray, package_root: str, manifest: Mapping[str, Any], batch_size: int
    ) -> np.ndarray:
        try:
            from .stage2_diag_cosine_exploration import forward_zid160, registered_feature
            zid = forward_zid160(
                self._model(package_root, manifest), iq,
                device=self.device, batch_size=batch_size,
            )
            result = registered_feature(iq, zid)
        except Exception as error:
            raise D108Target125RunnerError("D92 registered_feature materialization failed") from error
        return _readonly_features(np.ascontiguousarray(result, dtype=np.float32), "registered features")

    def __call__(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        phase = request.get("phase")
        if phase not in PHASES:
            raise D108Target125RunnerError("state request phase drift")
        packages = request.get("packages")
        if not isinstance(packages, Mapping):
            raise D108Target125RunnerError("state request packages missing")
        support_ref = packages[f"{phase}_enrollment"]
        query_ref = packages[f"{phase}_apply"]
        support_payloads, support_manifest, _ = self._package(support_ref)
        query_payloads, query_manifest, _ = self._package(query_ref)
        try:
            from .stage2_diag_cosine_exploration import _validate_matched_packages
            _validate_matched_packages(support_manifest, query_manifest)
        except Exception as error:
            raise D108Target125RunnerError("D92 support/query package pairing drift") from error
        scene = request["scene"]
        if scene not in support_payloads or scene not in query_payloads:
            raise D108Target125RunnerError("D92 package scene is missing")
        registry = tuple(
            str(item.get("class_handle", ""))
            for item in support_manifest.get("registered_classes", [])
            if isinstance(item, Mapping)
        )
        if not registry or len(set(registry)) != len(registry):
            raise D108Target125RunnerError("D92 registered-class contract drift")
        for manifest in (support_manifest, query_manifest):
            if (
                manifest.get("receiver") != request["receiver"]
                or manifest.get("seed") != request["seed"]
                or manifest.get("k_shot") != request["source_pool_k"]
            ):
                raise D108Target125RunnerError("D92 package row binding drift")
        support_iq, labels, support_ids = _support_rows(
            support_payloads[scene], registered_classes=registry,
            active_k=request["k_shot"],
        )
        query_iq, query_ids = _query_rows(query_payloads[scene])
        if support_iq.shape[1:] != query_iq.shape[1:]:
            raise D108Target125RunnerError("support/query IQ shape drift")
        return {
            "support_features": self._features(
                iq=support_iq, package_root=str(support_ref["package_root"]),
                manifest=support_manifest, batch_size=64,
            ),
            "support_labels": labels,
            "registered_classes": registry,
            "support_physical_ids": support_ids,
            "query_features": self._features(
                iq=query_iq, package_root=str(query_ref["package_root"]),
                manifest=query_manifest, batch_size=1,
            ),
            "query_physical_ids": query_ids,
        }


def _state_request(row: Mapping[str, Any], scene: str, phase: str) -> dict[str, Any]:
    return {
        "schema": "cvs.phase2.d108.cbrrc_smme.target125.state_request.v1",
        "outer_id": row["outer_id"], "receiver": row["receiver"],
        "seed": row["seed"], "k_shot": row["k_shot"],
        "active_k": row["active_k"], "new_count": row["new_count"],
        "source_pool_k": row["source_pool_k"], "scene": scene, "phase": phase,
        "packages": row["packages"], "authority_bundle": row["authority_bundle"],
        "access_ledger": dict(ACCESS_LEDGER),
    }


def _materialize_pair(
    materializer: StateMaterializer, row: Mapping[str, Any], scene: str
) -> tuple[_MaterializedState, _MaterializedState]:
    authority = row["authority_bundle"]
    _verify_bound_file(
        authority, path_key="commit_path", sha_key="commit_sha256",
        name="per-row authority COMMIT",
    )
    before_request = _state_request(row, scene, "before")
    after_request = _state_request(row, scene, "after")
    feature_width = getattr(materializer, "feature_width", FEATURE_WIDTH)
    if type(feature_width) is not int or feature_width not in (160, FEATURE_WIDTH):
        raise D108Target125RunnerError("typed materializer feature width is not permitted")
    before = _coerce_materialized_state(
        materializer(before_request), request=before_request, feature_width=feature_width
    )
    after = _coerce_materialized_state(
        materializer(after_request), request=after_request, feature_width=feature_width
    )
    if len(before.registered_classes) != OLD_CLASS_COUNT:
        raise D108Target125RunnerError("before registry must retain six old classes")
    if (
        after.registered_classes[:OLD_CLASS_COUNT] != before.registered_classes
        or len(after.registered_classes) != OLD_CLASS_COUNT + row["new_count"]
    ):
        raise D108Target125RunnerError("before/after registry registration drift")
    after_old_indices = [
        index for index, label in enumerate(after.support_labels)
        if label in before.registered_classes
    ]
    after_old_ids = tuple(after.support_physical_ids[index] for index in after_old_indices)
    after_old_features = after.support_features[after_old_indices]
    if (
        after_old_ids != before.support_physical_ids
        or not np.array_equal(after_old_features, before.support_features)
    ):
        raise D108Target125RunnerError(
            "before/after old support physical IDs or registered features drift"
        )
    return before, after


def _resolve_core(
    pair_builder: PairBuilder | None, query_scorer: QueryScorer | None
) -> tuple[PairBuilder, QueryScorer]:
    if pair_builder is None or query_scorer is None:
        try:
            from .stage2_d108_d92_core import (
                build_d108_d92_pair,
                score,
            )
        except Exception as error:  # pragma: no cover - independent landing
            raise D108Target125RunnerError("D108 D92 pair/score API is unavailable") from error
        pair_builder = build_d108_d92_pair if pair_builder is None else pair_builder
        query_scorer = score if query_scorer is None else query_scorer
    if not callable(pair_builder) or not callable(query_scorer):
        raise D108Target125RunnerError("D108 pair/score API must be callable")
    return pair_builder, query_scorer


def _build_pair(
    before: _MaterializedState,
    after: _MaterializedState,
    *, row: Mapping[str, Any], scene: str, plan: Mapping[str, Any],
    pair_builder: PairBuilder, device: Any, d92_fit: Callable[..., Any],
) -> Any:
    old_mask = np.asarray(
        [label in before.registered_classes for label in after.support_labels],
        dtype=bool,
    )
    new_mask = ~old_mask
    if int(old_mask.sum()) != OLD_CLASS_COUNT * row["k_shot"] or int(new_mask.sum()) != row["new_count"] * row["k_shot"]:
        raise D108Target125RunnerError("after old/new support split drift")
    try:
        return pair_builder(
            after.support_features[old_mask],
            tuple(
                label for label, selected in zip(after.support_labels, old_mask, strict=True) if selected
            ),
            before.registered_classes,
            after.support_features[new_mask],
            tuple(
                label for label, selected in zip(after.support_labels, new_mask, strict=True) if selected
            ),
            after.registered_classes[OLD_CLASS_COUNT:],
            seed=int(row["seed"]) + SCENES.index(scene),
            device=device,
            d92_fit=d92_fit,
        )
    except Exception as error:
        raise D108Target125RunnerError("D108 pair construction failed closed") from error


def _predict_labels(
    pair: Any, materialized: _MaterializedState, *, arm: str, phase: str,
    query_scorer: QueryScorer,
) -> tuple[str, ...]:
    try:
        logits = np.asarray(
            query_scorer(pair, phase, arm, materialized.query_features)
        )
    except Exception as error:
        raise D108Target125RunnerError("D108 query scoring failed closed") from error
    expected = (len(materialized.query_features), len(materialized.registered_classes))
    if logits.dtype.kind not in "f" or logits.shape != expected or not np.isfinite(logits).all():
        raise D108Target125RunnerError("D108 query logits contract drift")
    maxima = np.max(logits, axis=1, keepdims=True)
    if np.any(np.sum(logits == maxima, axis=1) != 1):
        raise D108Target125RunnerError("D108 exact top tie must fail closed")
    indices = np.argmax(logits, axis=1)
    return tuple(materialized.registered_classes[int(index)] for index in indices.tolist())


def _pair_runtime_bindings(
    materializer: StateMaterializer,
    *, device: str,
    d92_fit: Callable[..., Any] | None,
) -> tuple[Any, Callable[..., Any]]:
    resolved_fit = d92_fit if d92_fit is not None else getattr(materializer, "d92_fit", None)
    if not callable(resolved_fit):
        raise D108Target125RunnerError(
            "injected materializer requires an explicit callable d92_fit"
        )
    return getattr(materializer, "device", device), resolved_fit


def _surface_record(
    *, surface: Any, materialized: _MaterializedState,
    predicted_labels: tuple[str, ...], prediction_root: Path,
) -> dict[str, Any]:
    query_ids = list(materialized.query_physical_ids)
    labels = list(predicted_labels)
    if len(labels) != len(query_ids) or any(label not in materialized.registered_classes for label in labels):
        raise D108Target125RunnerError("predicted-label closure drift")
    payload: dict[str, Any] = {
        "schema": PREDICTION_ARTIFACT_SCHEMA,
        "surface_id": surface.surface_id, "outer_id": surface.outer_id,
        "receiver": surface.receiver, "seed": surface.seed,
        "k_shot": surface.k_shot, "new_count": surface.new_count,
        "scene": surface.scene, "arm": surface.arm, "phase": surface.phase,
        "registered_classes": list(materialized.registered_classes),
        "ordered_query_physical_ids": query_ids,
        "ordered_query_physical_ids_sha256": canonical_sha256(query_ids),
        "predicted_labels": labels,
        "predicted_labels_sha256": canonical_sha256(labels),
        "access_ledger": dict(ACCESS_LEDGER), "truth_open": False,
        "immutable": True,
    }
    payload["artifact_receipt_sha256"] = canonical_sha256(payload)
    artifact_path = prediction_root / f"{surface.surface_id}.json"
    artifact_file_sha = _write_json_new(artifact_path, payload)
    return {
        **{name: value for name, value in payload.items() if name not in {"schema", "artifact_receipt_sha256"}},
        "prediction_artifact": f"predictions/{surface.surface_id}.json",
        "prediction_artifact_sha256": artifact_file_sha,
    }


def _output_dir_new(path: Path, name: str) -> Path:
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"immutable {name} output already exists: {destination}")
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise D108Target125RunnerError(f"unsafe {name} output parent")
    destination.mkdir()
    return destination


def smoke_d108_target125_prepared_state(
    *, plan_manifest_path: Path, expected_plan_file_sha256: str,
    context_manifest_path: Path, expected_context_file_sha256: str,
    output_dir: Path, row_index: int = 0, scene_index: int = 0,
    device: str = "cpu", feature_batch_size: int = 64,
    state_materializer: StateMaterializer | None = None,
    pair_builder: PairBuilder | None = None,
    query_scorer: QueryScorer | None = None,
    d92_fit: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    plan, context = _prepared_inputs(
        plan_manifest_path=plan_manifest_path,
        expected_plan_file_sha256=expected_plan_file_sha256,
        context_manifest_path=context_manifest_path,
        expected_context_file_sha256=expected_context_file_sha256,
    )
    if type(row_index) is not int or row_index not in range(OUTER_JOB_COUNT) or type(scene_index) is not int or scene_index not in range(len(SCENES)):
        raise D108Target125RunnerError("smoke row/scene index drift")
    destination = _output_dir_new(output_dir, "smoke")
    materializer = state_materializer or _D108RealStateMaterializer(
        plan=plan, device=device, support_batch_size=feature_batch_size
    )
    builder, scorer = _resolve_core(pair_builder, query_scorer)
    pair_device, resolved_d92_fit = _pair_runtime_bindings(
        materializer, device=device, d92_fit=d92_fit
    )
    row = context["rows"][row_index]
    scene = SCENES[scene_index]
    before, after = _materialize_pair(materializer, row, scene)
    pair = _build_pair(
        before, after, row=row, scene=scene, plan=plan, pair_builder=builder,
        device=pair_device, d92_fit=resolved_d92_fit,
    )
    smoke_surfaces: list[dict[str, Any]] = []
    for arm in ARMS:
        for phase, state in (("before", before), ("after", after)):
            labels = _predict_labels(
                pair, state, arm=arm, phase=phase, query_scorer=scorer
            )
            query_ids = list(state.query_physical_ids)
            predicted = list(labels)
            smoke_surfaces.append(
                {
                    "arm": arm,
                    "phase": phase,
                    "registered_classes": list(state.registered_classes),
                    "ordered_query_physical_ids": query_ids,
                    "ordered_query_physical_ids_sha256": canonical_sha256(query_ids),
                    "predicted_labels": predicted,
                    "predicted_labels_sha256": canonical_sha256(predicted),
                }
            )
    predictions: dict[str, Any] = {
        "schema": SMOKE_PREDICTIONS_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "protocol_schema": PROTOCOL_SCHEMA,
        "outer_id": row["outer_id"],
        "receiver": row["receiver"],
        "seed": row["seed"],
        "k_shot": row["k_shot"],
        "new_count": row["new_count"],
        "scene": scene,
        "surfaces": smoke_surfaces,
        "access_ledger": dict(ACCESS_LEDGER),
        "truth_open": False,
        "immutable": True,
    }
    predictions["smoke_predictions_receipt_sha256"] = canonical_sha256(predictions)
    predictions_path = destination / "smoke_predictions.json"
    predictions_file_sha = _write_json_new(predictions_path, predictions)
    receipt: dict[str, Any] = {
        "schema": SMOKE_RECEIPT_SCHEMA, "candidate_id": CANDIDATE_ID,
        "protocol_schema": PROTOCOL_SCHEMA, "status": "D108_REAL_CHECKPOINT_NO_QUERY_FIT_SMOKE_PASS",
        "outer_id": row["outer_id"], "scene": scene, "arms": list(ARMS),
        "phases": list(PHASES), "support_batch_size": feature_batch_size,
        "query_batch_size": 1, "query_truth_access": False,
        "query_fit_access": False, "query_update_access": False,
    }
    receipt["smoke_receipt_sha256"] = canonical_sha256(receipt)
    path = destination / "smoke_receipt.json"
    file_sha = _write_json_new(path, receipt)
    return {
        **receipt,
        "smoke_receipt": str(path),
        "smoke_receipt_file_sha256": file_sha,
        "smoke_predictions": str(predictions_path),
        "smoke_predictions_file_sha256": predictions_file_sha,
        "smoke_predictions_receipt_sha256": predictions[
            "smoke_predictions_receipt_sha256"
        ],
    }


def _shard_outer_indices(shard_index: int) -> tuple[int, ...]:
    if type(shard_index) is not int or shard_index not in range(SHARD_COUNT):
        raise D108Target125RunnerError("shard_index must be an integer in 0..7")
    return tuple(index for index in range(OUTER_JOB_COUNT) if index % SHARD_COUNT == shard_index)


def _predict_shard(
    *, plan: Mapping[str, Any], context: Mapping[str, Any], output_dir: Path,
    shard_index: int, device: str, feature_batch_size: int,
    state_materializer: StateMaterializer | None,
    pair_builder: PairBuilder | None, query_scorer: QueryScorer | None,
    d92_fit: Callable[..., Any] | None,
) -> dict[str, Any]:
    indices = _shard_outer_indices(shard_index)
    destination = _output_dir_new(output_dir, f"shard {shard_index}")
    prediction_root = destination / "predictions"
    prediction_root.mkdir()
    materializer = state_materializer or _D108RealStateMaterializer(
        plan=plan, device=device, support_batch_size=feature_batch_size
    )
    builder, scorer = _resolve_core(pair_builder, query_scorer)
    pair_device, resolved_d92_fit = _pair_runtime_bindings(
        materializer, device=device, d92_fit=d92_fit
    )
    matrix = freeze_d108_target125_matrix()
    surface_by_key = {
        (surface.outer_id, surface.scene, surface.arm, surface.phase): surface
        for surface in matrix.surfaces
    }
    outer_rows: list[dict[str, Any]] = []
    surfaces: list[dict[str, Any]] = []
    for index in indices:
        row = context["rows"][index]
        expected_outer = matrix.outer_rows[index]
        frozen_old: tuple[str, ...] | None = None
        frozen_new: tuple[str, ...] | None = None
        for scene in SCENES:
            before, after = _materialize_pair(materializer, row, scene)
            old = before.registered_classes
            new = after.registered_classes[OLD_CLASS_COUNT:]
            if frozen_old is None:
                frozen_old, frozen_new = old, new
            elif frozen_old != old or frozen_new != new:
                raise D108Target125RunnerError("outer registry differs across scenes")
            pair = _build_pair(
                before, after, row=row, scene=scene, plan=plan,
                pair_builder=builder, device=pair_device,
                d92_fit=resolved_d92_fit,
            )
            for arm in ARMS:
                for phase, state in (("before", before), ("after", after)):
                    surface = surface_by_key[(row["outer_id"], scene, arm, phase)]
                    labels = _predict_labels(
                        pair, state, arm=arm, phase=phase, query_scorer=scorer
                    )
                    surfaces.append(_surface_record(
                        surface=surface, materialized=state,
                        predicted_labels=labels, prediction_root=prediction_root,
                    ))
        if frozen_old is None or frozen_new is None:
            raise D108Target125RunnerError("outer scene loop did not materialize")
        outer_rows.append({
            "outer_id": expected_outer.outer_id, "receiver": expected_outer.receiver,
            "seed": expected_outer.seed, "k_shot": expected_outer.k_shot,
            "new_count": expected_outer.new_count, "old_classes": list(frozen_old),
            "new_classes": list(frozen_new),
        })
    shard: dict[str, Any] = {
        "schema": PREDICTION_SHARD_SCHEMA, "candidate_id": CANDIDATE_ID,
        "protocol_schema": PROTOCOL_SCHEMA, "truth_open": False,
        "shard_index": shard_index, "shard_count": SHARD_COUNT,
        "matrix_receipt_sha256": plan["identity"]["matrix_receipt_sha256"],
        "plan_receipt_sha256": plan["plan_receipt_sha256"],
        "context_receipt_sha256": context["context_receipt_sha256"],
        "outer_indices": list(indices), "outer_rows": outer_rows,
        "surface_count": len(surfaces), "access_ledger": dict(ACCESS_LEDGER),
        "surfaces": surfaces,
    }
    shard["shard_receipt_sha256"] = canonical_sha256(shard)
    path = destination / "prediction_shard_manifest.json"
    file_sha = _write_json_new(path, shard)
    return {
        "prediction_shard_manifest": str(path),
        "prediction_shard_manifest_file_sha256": file_sha,
        "shard_receipt_sha256": shard["shard_receipt_sha256"],
        "shard_index": shard_index, "outer_job_count": len(indices),
        "surface_count": len(surfaces),
    }


def _validate_shard(path: Path) -> tuple[dict[str, Any], Path]:
    source = _regular_file(path, "prediction shard manifest")
    shard = _read_json(source, name="prediction shard manifest")
    required = {
        "schema", "candidate_id", "protocol_schema", "truth_open", "shard_index",
        "shard_count", "matrix_receipt_sha256", "plan_receipt_sha256",
        "context_receipt_sha256", "outer_indices", "outer_rows", "surface_count",
        "access_ledger", "surfaces", "shard_receipt_sha256",
    }
    if set(shard) != required or shard.get("schema") != PREDICTION_SHARD_SCHEMA or shard.get("candidate_id") != CANDIDATE_ID or shard.get("protocol_schema") != PROTOCOL_SCHEMA or shard.get("truth_open") is not False or shard.get("shard_count") != SHARD_COUNT:
        raise D108Target125RunnerError("prediction shard identity drift")
    _receipt(shard, "shard_receipt_sha256", "prediction shard")
    _access_ledger(shard.get("access_ledger"))
    expected_indices = _shard_outer_indices(shard.get("shard_index"))
    if shard.get("outer_indices") != list(expected_indices):
        raise D108Target125RunnerError("prediction shard outer coverage drift")
    surfaces = shard.get("surfaces")
    if not isinstance(surfaces, list) or shard.get("surface_count") != len(surfaces) or len(surfaces) != len(expected_indices) * len(SCENES) * len(ARMS) * len(PHASES):
        raise D108Target125RunnerError("prediction shard surface-count drift")
    matrix = freeze_d108_target125_matrix()
    expected_outers = [matrix.outer_rows[index] for index in expected_indices]
    outer_rows = shard.get("outer_rows")
    if not isinstance(outer_rows, list) or len(outer_rows) != len(expected_outers):
        raise D108Target125RunnerError("prediction shard outer-row count drift")
    registries: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for row, expected in zip(outer_rows, expected_outers, strict=True):
        if not isinstance(row, Mapping) or set(row) != _OUTER_ROW_FIELDS or any(
            row.get(name) != getattr(expected, name)
            for name in ("outer_id", "receiver", "seed", "k_shot", "new_count")
        ):
            raise D108Target125RunnerError("prediction shard outer-row identity drift")
        old = _tokens(row.get("old_classes"), "shard old classes", OLD_CLASS_COUNT)
        new = _tokens(row.get("new_classes"), "shard new classes", expected.new_count)
        if set(old).intersection(new):
            raise D108Target125RunnerError("prediction shard registry overlap")
        registries[expected.outer_id] = (old, new)
    outer_ids = {row.outer_id for row in expected_outers}
    expected_surfaces = [surface for surface in matrix.surfaces if surface.outer_id in outer_ids]
    query_by_scope: dict[tuple[str, str, str], tuple[str, ...]] = {}
    for surface, expected in zip(surfaces, expected_surfaces, strict=True):
        if not isinstance(surface, Mapping) or set(surface) != _SURFACE_FIELDS or any(
            surface.get(name) != getattr(expected, name)
            for name in (
                "surface_id", "outer_id", "receiver", "seed", "k_shot",
                "new_count", "scene", "arm", "phase",
            )
        ):
            raise D108Target125RunnerError("prediction shard surface identity drift")
        old, new = registries[expected.outer_id]
        registry = _tokens(
            surface.get("registered_classes"), "shard surface registry",
            len(surface.get("registered_classes", [])),
        )
        if registry != (old if expected.phase == "before" else old + new):
            raise D108Target125RunnerError("prediction shard phase registry drift")
        query_ids = _tokens(
            surface.get("ordered_query_physical_ids"), "shard query IDs",
            len(surface.get("ordered_query_physical_ids", [])),
        )
        labels = _tokens(
            surface.get("predicted_labels"), "shard predicted labels", len(query_ids),
            require_unique=False,
        )
        if (
            surface.get("ordered_query_physical_ids_sha256") != canonical_sha256(list(query_ids))
            or surface.get("predicted_labels_sha256") != canonical_sha256(list(labels))
            or any(label not in registry for label in labels)
            or surface.get("truth_open") is not False
            or surface.get("immutable") is not True
        ):
            raise D108Target125RunnerError("prediction shard query/prediction closure drift")
        _access_ledger(surface.get("access_ledger"))
        scope = (expected.outer_id, expected.scene, expected.phase)
        if scope in query_by_scope and query_by_scope[scope] != query_ids:
            raise D108Target125RunnerError("prediction shard four-arm query order drift")
        query_by_scope[scope] = query_ids
        _validate_artifact(surface=surface, root=source.parent)
    return shard, source.parent


def _merge_shards(*, shard_manifest_paths: Sequence[Path], output_dir: Path) -> dict[str, Any]:
    if isinstance(shard_manifest_paths, (str, bytes)) or len(shard_manifest_paths) != SHARD_COUNT:
        raise D108Target125RunnerError("merge requires exactly eight shard manifests")
    loaded = [_validate_shard(Path(path)) for path in shard_manifest_paths]
    by_index: dict[int, tuple[dict[str, Any], Path]] = {}
    for shard, root in loaded:
        index = shard["shard_index"]
        if index in by_index:
            raise D108Target125RunnerError("merge has a duplicate shard index")
        by_index[index] = (shard, root)
    if set(by_index) != set(range(SHARD_COUNT)):
        raise D108Target125RunnerError("merge shard-index coverage drift")
    identity_fields = (
        "matrix_receipt_sha256", "plan_receipt_sha256", "context_receipt_sha256"
    )
    baseline = by_index[0][0]
    if any(any(shard[name] != baseline[name] for name in identity_fields) for shard, _ in by_index.values()):
        raise D108Target125RunnerError("merge shard identity mismatch")
    matrix = freeze_d108_target125_matrix()
    surfaces_by_id: dict[str, tuple[dict[str, Any], Path]] = {}
    outer_by_id: dict[str, dict[str, Any]] = {}
    for index in range(SHARD_COUNT):
        shard, root = by_index[index]
        for outer in shard["outer_rows"]:
            outer_id = outer.get("outer_id") if isinstance(outer, Mapping) else None
            if outer_id in outer_by_id:
                raise D108Target125RunnerError("merge duplicate outer row")
            outer_by_id[str(outer_id)] = dict(outer)
        for surface in shard["surfaces"]:
            surface_id = surface.get("surface_id") if isinstance(surface, Mapping) else None
            if type(surface_id) is not str or surface_id in surfaces_by_id:
                raise D108Target125RunnerError("merge duplicate/invalid surface")
            surfaces_by_id[surface_id] = (dict(surface), root)
    audit_surface_coverage(surfaces_by_id)
    if set(outer_by_id) != {row.outer_id for row in matrix.outer_rows}:
        raise D108Target125RunnerError("merge outer coverage drift")
    # Validate every source artifact before creating the final output directory.
    source_artifacts: list[tuple[dict[str, Any], bytes]] = []
    for expected in matrix.surfaces:
        surface, root = surfaces_by_id[expected.surface_id]
        relative = surface.get("prediction_artifact")
        if relative != f"predictions/{expected.surface_id}.json":
            raise D108Target125RunnerError("shard artifact relative path drift")
        source = _regular_file(root / str(relative), "shard prediction artifact")
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != _sha(
            surface.get("prediction_artifact_sha256"), "shard artifact SHA256"
        ):
            raise D108Target125RunnerError("shard artifact SHA mismatch")
        source_artifacts.append((surface, raw))
    destination = _output_dir_new(output_dir, "merged prediction")
    prediction_root = destination / "predictions"
    prediction_root.mkdir()
    merged_surfaces: list[dict[str, Any]] = []
    for (surface, raw), expected in zip(source_artifacts, matrix.surfaces, strict=True):
        target = prediction_root / f"{expected.surface_id}.json"
        file_sha = _write_bytes_new(target, raw)
        copied = dict(surface)
        copied["prediction_artifact"] = f"predictions/{expected.surface_id}.json"
        copied["prediction_artifact_sha256"] = file_sha
        merged_surfaces.append(copied)
    manifest: dict[str, Any] = {
        "schema": PREDICTION_MANIFEST_SCHEMA, "candidate_id": CANDIDATE_ID,
        "protocol_schema": PROTOCOL_SCHEMA, "manifest_sealed": True,
        "truth_open": False, "outer_job_count": OUTER_JOB_COUNT,
        "scene_row_count": SCENE_ROW_COUNT, "arm_pair_count": ARM_PAIR_COUNT,
        "surface_count": SURFACE_COUNT, "scenes": list(SCENES),
        "arms": list(ARMS), "phases": list(PHASES),
        "outer_rows": [outer_by_id[row.outer_id] for row in matrix.outer_rows],
        "access_ledger": dict(ACCESS_LEDGER), "shard_count": SHARD_COUNT,
        "shard_receipts": [by_index[index][0]["shard_receipt_sha256"] for index in range(SHARD_COUNT)],
        "surfaces": merged_surfaces,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    path = destination / "prediction_manifest.json"
    file_sha = _write_json_new(path, manifest)
    validate_d108_target125_prediction_manifest(
        prediction_manifest_path=path,
        expected_prediction_manifest_file_sha256=file_sha,
    )
    return {
        "prediction_manifest": str(path),
        "prediction_manifest_file_sha256": file_sha,
        "prediction_manifest_sha256": manifest["manifest_sha256"],
        "outer_job_count": OUTER_JOB_COUNT, "scene_row_count": SCENE_ROW_COUNT,
        "arm_pair_count": ARM_PAIR_COUNT, "surface_count": SURFACE_COUNT,
        "shard_count": SHARD_COUNT,
    }


def predict_d108_target125(
    *, plan_manifest_path: Path | None = None,
    expected_plan_file_sha256: str | None = None,
    context_manifest_path: Path | None = None,
    expected_context_file_sha256: str | None = None,
    output_dir: Path, device: str = "cpu", feature_batch_size: int = 64,
    shard_index: int | None = None,
    shard_manifest_paths: Sequence[Path] | None = None,
    state_materializer: StateMaterializer | None = None,
    pair_builder: PairBuilder | None = None,
    query_scorer: QueryScorer | None = None,
    d92_fit: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run one immutable shard or merge exactly eight completed shards."""

    if shard_manifest_paths is not None:
        if any(value is not None for value in (
            plan_manifest_path, expected_plan_file_sha256,
            context_manifest_path, expected_context_file_sha256, shard_index,
            state_materializer, pair_builder, query_scorer,
            d92_fit,
        )):
            raise D108Target125RunnerError("merge accepts only shard manifests and output_dir")
        return _merge_shards(
            shard_manifest_paths=shard_manifest_paths, output_dir=output_dir
        )
    if any(value is None for value in (
        plan_manifest_path, expected_plan_file_sha256,
        context_manifest_path, expected_context_file_sha256, shard_index,
    )):
        raise D108Target125RunnerError("shard predict requires prepared inputs and shard_index")
    plan, context = _prepared_inputs(
        plan_manifest_path=Path(plan_manifest_path),
        expected_plan_file_sha256=str(expected_plan_file_sha256),
        context_manifest_path=Path(context_manifest_path),
        expected_context_file_sha256=str(expected_context_file_sha256),
    )
    return _predict_shard(
        plan=plan, context=context, output_dir=output_dir,
        shard_index=int(shard_index), device=device,
        feature_batch_size=feature_batch_size,
        state_materializer=state_materializer, pair_builder=pair_builder,
        query_scorer=query_scorer, d92_fit=d92_fit,
    )


def _validate_outer_rows(value: Any) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    if not isinstance(value, list) or len(value) != OUTER_JOB_COUNT:
        raise D108Target125RunnerError("prediction outer-row count drift")
    matrix = freeze_d108_target125_matrix()
    result: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for row, expected in zip(value, matrix.outer_rows, strict=True):
        if not isinstance(row, Mapping) or set(row) != _OUTER_ROW_FIELDS or any(
            row.get(name) != getattr(expected, name)
            for name in ("outer_id", "receiver", "seed", "k_shot", "new_count")
        ):
            raise D108Target125RunnerError("prediction outer-row identity drift")
        old = _tokens(row.get("old_classes"), "old classes", OLD_CLASS_COUNT)
        new = _tokens(row.get("new_classes"), "new classes", expected.new_count)
        if set(old).intersection(new):
            raise D108Target125RunnerError("old/new registry overlap")
        result[expected.outer_id] = (old, new)
    return result


def _validate_artifact(*, surface: Mapping[str, Any], root: Path) -> None:
    relative = surface.get("prediction_artifact")
    if relative != f"predictions/{surface['surface_id']}.json":
        raise D108Target125RunnerError("prediction artifact relative path drift")
    source = _regular_file(root / str(relative), "prediction artifact")
    if _sha256_file(source) != _sha(surface.get("prediction_artifact_sha256"), "artifact SHA256"):
        raise D108Target125RunnerError("prediction artifact file SHA drift")
    artifact = _read_json(source, name="prediction artifact")
    if set(artifact) != _ARTIFACT_FIELDS or artifact.get("schema") != PREDICTION_ARTIFACT_SCHEMA:
        raise D108Target125RunnerError("prediction artifact field/schema drift")
    _receipt(artifact, "artifact_receipt_sha256", "prediction artifact")
    projected = {name: value for name, value in artifact.items() if name not in {"schema", "artifact_receipt_sha256"}}
    expected = {name: value for name, value in surface.items() if name not in {"prediction_artifact", "prediction_artifact_sha256"}}
    if projected != expected:
        raise D108Target125RunnerError("artifact/manifest content drift")


def validate_d108_target125_prediction_manifest(
    *, prediction_manifest_path: Path,
    expected_prediction_manifest_file_sha256: str | None = None,
) -> dict[str, Any]:
    path = _regular_file(prediction_manifest_path, "prediction manifest")
    manifest = _read_json(
        path, name="prediction manifest",
        expected_file_sha256=expected_prediction_manifest_file_sha256,
    )
    if set(manifest) != _MANIFEST_FIELDS:
        raise D108Target125RunnerError("prediction manifest field closure drift")
    if (
        manifest.get("schema") != PREDICTION_MANIFEST_SCHEMA
        or manifest.get("candidate_id") != CANDIDATE_ID
        or manifest.get("protocol_schema") != PROTOCOL_SCHEMA
        or manifest.get("manifest_sealed") is not True
        or manifest.get("truth_open") is not False
        or manifest.get("outer_job_count") != OUTER_JOB_COUNT
        or manifest.get("scene_row_count") != SCENE_ROW_COUNT
        or manifest.get("arm_pair_count") != ARM_PAIR_COUNT
        or manifest.get("surface_count") != SURFACE_COUNT
        or manifest.get("scenes") != list(SCENES)
        or manifest.get("arms") != list(ARMS)
        or manifest.get("phases") != list(PHASES)
        or manifest.get("shard_count") != SHARD_COUNT
        or not isinstance(manifest.get("shard_receipts"), list)
        or len(manifest["shard_receipts"]) != SHARD_COUNT
        or len(set(manifest["shard_receipts"])) != SHARD_COUNT
    ):
        raise D108Target125RunnerError("prediction manifest identity/count drift")
    for receipt_value in manifest["shard_receipts"]:
        _sha(receipt_value, "shard receipt")
    _receipt(manifest, "manifest_sha256", "prediction manifest")
    _access_ledger(manifest.get("access_ledger"))
    registries = _validate_outer_rows(manifest.get("outer_rows"))
    surfaces = manifest.get("surfaces")
    if not isinstance(surfaces, list) or len(surfaces) != SURFACE_COUNT:
        raise D108Target125RunnerError("prediction surface count drift")
    matrix = freeze_d108_target125_matrix()
    audit_surface_coverage(
        surface.get("surface_id") if isinstance(surface, Mapping) else ""
        for surface in surfaces
    )
    query_by_scope: dict[tuple[str, str, str], tuple[str, ...]] = {}
    for surface, expected in zip(surfaces, matrix.surfaces, strict=True):
        if not isinstance(surface, Mapping) or set(surface) != _SURFACE_FIELDS or any(
            surface.get(name) != getattr(expected, name)
            for name in (
                "surface_id", "outer_id", "receiver", "seed", "k_shot",
                "new_count", "scene", "arm", "phase",
            )
        ):
            raise D108Target125RunnerError("prediction surface matrix drift")
        old, new = registries[expected.outer_id]
        registry = _tokens(
            surface.get("registered_classes"), "surface registry",
            len(surface.get("registered_classes", [])),
        )
        if registry != (old if expected.phase == "before" else old + new):
            raise D108Target125RunnerError("surface registry phase drift")
        query_ids = _tokens(
            surface.get("ordered_query_physical_ids"), "query physical IDs",
            len(surface.get("ordered_query_physical_ids", [])),
        )
        if surface.get("ordered_query_physical_ids_sha256") != canonical_sha256(list(query_ids)):
            raise D108Target125RunnerError("query physical-ID receipt drift")
        labels = _tokens(
            surface.get("predicted_labels"), "predicted labels", len(query_ids),
            require_unique=False,
        )
        if any(label not in registry for label in labels) or surface.get("predicted_labels_sha256") != canonical_sha256(list(labels)):
            raise D108Target125RunnerError("predicted-label receipt/registry drift")
        if surface.get("truth_open") is not False or surface.get("immutable") is not True:
            raise D108Target125RunnerError("surface truth/immutability drift")
        _access_ledger(surface.get("access_ledger"))
        scope = (expected.outer_id, expected.scene, expected.phase)
        if scope in query_by_scope and query_by_scope[scope] != query_ids:
            raise D108Target125RunnerError("four-arm query order differs")
        query_by_scope[scope] = query_ids
        _validate_artifact(surface=surface, root=path.parent)
    if len(query_by_scope) != SCENE_ROW_COUNT * len(PHASES):
        raise D108Target125RunnerError("prediction query-scope coverage drift")
    return manifest


__all__ = [
    "D108Target125RunnerError", "PREDICTION_ARTIFACT_SCHEMA",
    "PREDICTION_MANIFEST_SCHEMA", "PREDICTION_SHARD_SCHEMA", "SHARD_COUNT",
    "SMOKE_PREDICTIONS_SCHEMA", "SMOKE_RECEIPT_SCHEMA", "predict_d108_target125",
    "prepare_d108_target125_run", "smoke_d108_target125_prepared_state",
    "validate_d108_target125_prediction_manifest",
]
