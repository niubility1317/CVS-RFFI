"""Immutable truth-free prepare/smoke/predict runner for D107 Target125.

The runner opens only D92 sealed received-IQ packages and the pinned checkpoint.
For each state it performs one checkpoint forward over support plus query and
then splits signed ``z_id`` rows.  It invokes the SCMKRR public API separately
for all four frozen arms; no K router, truth catalog, role, quota, or query
adaptation surface exists here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
from typing import Any

import numpy as np

from .stage2_d107_matrix_protocol import (
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
    canonical_bytes,
    canonical_sha256,
    freeze_d107_target125_matrix,
)
from .stage2_d107_target125_inputs import (
    CONTEXT_SCHEMA,
    PLAN_SCHEMA,
    prepare_d107_target125_inputs,
)


PREDICTION_MANIFEST_SCHEMA = "cvs.phase2.d107.scmkrr.target125.prediction_manifest.v1"
PREDICTION_ARTIFACT_SCHEMA = "cvs.phase2.d107.scmkrr.target125.prediction_artifact.v1"
SMOKE_RECEIPT_SCHEMA = "cvs.phase2.d107.scmkrr.target125.smoke_receipt.v1"

_PLAN_FIELDS = {
    "schema",
    "candidate_id",
    "protocol_schema",
    "matrix_protocol",
    "identity",
    "rows",
    "plan_receipt_sha256",
}
_CONTEXT_FIELDS = {
    "schema",
    "candidate_id",
    "protocol_schema",
    "plan_receipt_sha256",
    "identity",
    "rows",
    "context_receipt_sha256",
}
_ROW_FIELDS = {
    "outer_id",
    "source_d92_job_id",
    "receiver",
    "seed",
    "k_shot",
    "active_k",
    "new_count",
    "source_pool_k",
    "k5_prefix_from_matched_k10",
    "packages",
}
_PACKAGE_FIELDS = {"package_root", "detached_seal_path", "expected_seal_sha256"}
_OUTER_ROW_FIELDS = {
    "outer_id",
    "receiver",
    "seed",
    "k_shot",
    "new_count",
    "old_classes",
    "new_classes",
}
_SURFACE_FIELDS = {
    "surface_id",
    "outer_id",
    "receiver",
    "seed",
    "k_shot",
    "new_count",
    "scene",
    "arm",
    "phase",
    "registered_classes",
    "prediction_artifact",
    "prediction_artifact_sha256",
    "ordered_query_physical_ids",
    "ordered_query_physical_ids_sha256",
    "predicted_labels",
    "predicted_labels_sha256",
    "access_ledger",
    "truth_open",
    "immutable",
}
_ARTIFACT_FIELDS = (_SURFACE_FIELDS - {"prediction_artifact", "prediction_artifact_sha256"}) | {
    "schema",
    "artifact_receipt_sha256",
}
_MANIFEST_FIELDS = {
    "schema",
    "candidate_id",
    "protocol_schema",
    "manifest_sealed",
    "truth_open",
    "outer_job_count",
    "scene_row_count",
    "arm_pair_count",
    "surface_count",
    "scenes",
    "arms",
    "phases",
    "outer_rows",
    "access_ledger",
    "surfaces",
    "manifest_sha256",
}
_MATERIALIZED_FIELDS = {
    "support_signed",
    "support_labels",
    "registered_classes",
    "support_physical_ids",
    "query_signed",
    "query_physical_ids",
    "tau",
    "spectrum",
}


class D107Target125RunnerError(ValueError):
    """Raised when D107 execution cannot preserve its sealed protocol boundary."""


StateMaterializer = Callable[[Mapping[str, Any]], Mapping[str, Any]]
StateBuilder = Callable[..., Any]
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
        raise D107Target125RunnerError(f"{name} must be a lowercase SHA256")
    return value


def _regular_file(path: Path, name: str) -> Path:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise D107Target125RunnerError(f"{name} must be a regular non-symlink file")
    return source.resolve(strict=True)


def _read_json(
    path: Path, *, name: str, expected_file_sha256: str | None = None
) -> dict[str, Any]:
    source = _regular_file(path, name)
    if expected_file_sha256 is not None and _sha256_file(source) != _sha(
        expected_file_sha256, f"expected {name} SHA256"
    ):
        raise D107Target125RunnerError(f"{name} SHA mismatch")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise D107Target125RunnerError(f"{name} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise D107Target125RunnerError(f"{name} must contain an object")
    return value


def _write_json_new(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"immutable output already exists: {path}")
    raw = canonical_bytes(value) + b"\n"
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, stat.S_IREAD)
    return hashlib.sha256(raw).hexdigest()


def _receipt(document: Mapping[str, Any], field: str, name: str) -> str:
    receipt = _sha(document.get(field), f"{name} {field}")
    payload = {key: value for key, value in document.items() if key != field}
    if canonical_sha256(payload) != receipt:
        raise D107Target125RunnerError(f"{name} canonical receipt drift")
    return receipt


def _sequence(value: Any, name: str, expected_len: int | None = None) -> list[Any]:
    if not isinstance(value, list) or (
        expected_len is not None and len(value) != expected_len
    ):
        suffix = "" if expected_len is None else f" with {expected_len} items"
        raise D107Target125RunnerError(f"{name} must be a list{suffix}")
    return value


def _access_ledger(value: Any, name: str = "access ledger") -> dict[str, bool]:
    if not isinstance(value, Mapping) or set(value) != set(ACCESS_LEDGER):
        raise D107Target125RunnerError(f"{name} field closure drift")
    if any(value.get(field) is not False for field in ACCESS_LEDGER):
        raise D107Target125RunnerError(f"{name} grants forbidden access")
    return dict(ACCESS_LEDGER)


def _prepared_inputs(
    *,
    plan_manifest_path: Path,
    expected_plan_file_sha256: str,
    context_manifest_path: Path,
    expected_context_file_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = _read_json(
        plan_manifest_path,
        name="D107 Target125 plan",
        expected_file_sha256=expected_plan_file_sha256,
    )
    context = _read_json(
        context_manifest_path,
        name="D107 Target125 context",
        expected_file_sha256=expected_context_file_sha256,
    )
    if set(plan) != _PLAN_FIELDS or set(context) != _CONTEXT_FIELDS:
        raise D107Target125RunnerError("prepared plan/context field closure drift")
    if (
        plan.get("schema") != PLAN_SCHEMA
        or context.get("schema") != CONTEXT_SCHEMA
        or plan.get("candidate_id") != CANDIDATE_ID
        or context.get("candidate_id") != CANDIDATE_ID
        or plan.get("protocol_schema") != PROTOCOL_SCHEMA
        or context.get("protocol_schema") != PROTOCOL_SCHEMA
    ):
        raise D107Target125RunnerError("prepared plan/context identity drift")
    plan_receipt = _receipt(plan, "plan_receipt_sha256", "plan")
    context_receipt = _receipt(context, "context_receipt_sha256", "context")
    matrix = freeze_d107_target125_matrix()
    if (
        plan.get("matrix_protocol") != matrix.receipt_payload()
        or context.get("plan_receipt_sha256") != plan_receipt
        or plan.get("identity") != context.get("identity")
        or not context_receipt
    ):
        raise D107Target125RunnerError("prepared matrix/identity binding drift")
    identity = plan.get("identity")
    if not isinstance(identity, Mapping) or set(identity) != {
        "matrix_receipt_sha256",
        "d92_matrix_manifest",
        "d92_output_root",
        "d92_sealed_runtime_sha256",
        "checkpoint",
        "d107_method_lock",
        "rdce_asset",
    }:
        raise D107Target125RunnerError("prepared identity field closure drift")
    if identity.get("matrix_receipt_sha256") != matrix.matrix_receipt_sha256:
        raise D107Target125RunnerError("prepared matrix receipt drift")
    _sha(identity.get("d92_sealed_runtime_sha256"), "D92 sealed runtime SHA256")
    plan_rows = _sequence(plan.get("rows"), "prepared plan rows", OUTER_JOB_COUNT)
    context_rows = _sequence(
        context.get("rows"), "prepared context rows", OUTER_JOB_COUNT
    )
    if plan_rows != context_rows:
        raise D107Target125RunnerError("prepared plan/context row drift")
    expected_rows = matrix.outer_rows
    for row, expected in zip(plan_rows, expected_rows, strict=True):
        if not isinstance(row, Mapping) or set(row) != _ROW_FIELDS:
            raise D107Target125RunnerError("prepared D92 row field closure drift")
        if any(
            row.get(name) != getattr(expected, name)
            for name in ("outer_id", "receiver", "seed", "k_shot", "new_count")
        ) or row.get("active_k") != expected.k_shot:
            raise D107Target125RunnerError("prepared D92 row binding drift")
        expected_pool_k = 10 if (expected.k_shot, expected.new_count) == (5, 20) else expected.k_shot
        if (
            row.get("source_pool_k") != expected_pool_k
            or row.get("k5_prefix_from_matched_k10")
            != (expected.k_shot == 5 and expected.new_count == 20)
        ):
            raise D107Target125RunnerError("prepared K5/K10 nesting binding drift")
        packages = row.get("packages")
        if not isinstance(packages, Mapping) or set(packages) != {
            "before_enrollment",
            "before_apply",
            "after_enrollment",
            "after_apply",
        }:
            raise D107Target125RunnerError("prepared four-package ordering drift")
        for package in packages.values():
            if not isinstance(package, Mapping) or set(package) != _PACKAGE_FIELDS:
                raise D107Target125RunnerError("prepared package reference closure drift")
            _sha(package.get("expected_seal_sha256"), "prepared package seal SHA256")
    return plan, context


def prepare_d107_target125_run(**kwargs: Any) -> dict[str, Any]:
    """Forward the immutable prepare stage to the D92 input binder."""

    return prepare_d107_target125_inputs(**kwargs)


def _readonly_rows(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.dtype.kind not in "f"
        or array.ndim != 2
        or array.shape[0] < 1
        or array.shape[1] != FEATURE_WIDTH
        or not np.isfinite(array).all()
    ):
        raise D107Target125RunnerError(
            f"{name} must be finite floating [N,{FEATURE_WIDTH}]"
        )
    copied = np.ascontiguousarray(array, dtype=np.float32).copy()
    norms = np.linalg.norm(copied.astype(np.float64), axis=1)
    if not np.all(np.isfinite(norms)) or np.any(
        ~np.isclose(norms, 1.0, rtol=1.0e-5, atol=1.0e-5)
    ):
        raise D107Target125RunnerError(f"{name} must be signed L2-normalized z_id")
    copied.setflags(write=False)
    return copied


def _readonly_positive_vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.dtype.kind not in "f"
        or array.ndim != 1
        or len(array) < 1
        or not np.isfinite(array).all()
        or np.any(array <= 0.0)
    ):
        raise D107Target125RunnerError(f"{name} must be finite positive float vector")
    copied = np.ascontiguousarray(array, dtype=np.float64).copy()
    copied.setflags(write=False)
    return copied


def _tokens(
    value: Any, name: str, rows: int, *, require_unique: bool = True
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != rows:
        raise D107Target125RunnerError(f"{name} must align to its feature rows")
    tokens = tuple(str(item) for item in value)
    if any(not item for item in tokens) or (
        require_unique and len(set(tokens)) != len(tokens)
    ):
        qualifier = "non-empty and unique" if require_unique else "non-empty"
        raise D107Target125RunnerError(f"{name} must be {qualifier}")
    return tokens


@dataclass(frozen=True, slots=True)
class _MaterializedState:
    support_signed: np.ndarray
    support_labels: tuple[str, ...]
    registered_classes: tuple[str, ...]
    support_physical_ids: tuple[str, ...]
    query_signed: np.ndarray
    query_physical_ids: tuple[str, ...]
    tau: np.ndarray
    spectrum: np.ndarray


def _coerce_materialized_state(
    value: Mapping[str, Any], *, request: Mapping[str, Any]
) -> _MaterializedState:
    if not isinstance(value, Mapping) or set(value) != _MATERIALIZED_FIELDS:
        raise D107Target125RunnerError("state materializer field closure drift")
    for name in value:
        lowered = str(name).lower()
        if any(token in lowered for token in ("truth", "role", "metric", "score", "quota")):
            raise D107Target125RunnerError("state materializer exposed forbidden data")
    support = _readonly_rows(value["support_signed"], "support_signed")
    query = _readonly_rows(value["query_signed"], "query_signed")
    registered = _tokens(
        value["registered_classes"], "registered_classes", len(value["registered_classes"])
    )
    if len(registered) < 1:
        raise D107Target125RunnerError("registered_classes must be non-empty")
    labels = _tokens(
        value["support_labels"], "support_labels", len(support), require_unique=False
    )
    if any(label not in registered for label in labels):
        raise D107Target125RunnerError("support labels are outside the registry")
    active_k = request.get("k_shot")
    if type(active_k) is not int or any(labels.count(label) != active_k for label in registered):
        raise D107Target125RunnerError("support rows do not close balanced active K")
    support_ids = _tokens(value["support_physical_ids"], "support_physical_ids", len(support))
    query_ids = _tokens(value["query_physical_ids"], "query_physical_ids", len(query))
    if set(support_ids).intersection(query_ids):
        raise D107Target125RunnerError("support/query physical IDs overlap")
    tau = _readonly_positive_vector(value["tau"], "tau")
    spectrum = _readonly_positive_vector(value["spectrum"], "spectrum")
    if tau.shape != spectrum.shape:
        raise D107Target125RunnerError("tau/spectrum shape drift")
    return _MaterializedState(
        support_signed=support,
        support_labels=labels,
        registered_classes=registered,
        support_physical_ids=support_ids,
        query_signed=query,
        query_physical_ids=query_ids,
        tau=tau,
        spectrum=spectrum,
    )


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _signed_zid_l2_from_tap(tapped: Any, rows: int) -> np.ndarray:
    """Select the signed pre-ReLU identity feature and normalize each row."""

    pre_relu = np.asarray(getattr(tapped, "pre_relu", None))
    if (
        pre_relu.dtype != np.float32
        or pre_relu.ndim != 2
        or pre_relu.shape != (rows, FEATURE_WIDTH)
        or not np.isfinite(pre_relu).all()
    ):
        raise D107Target125RunnerError("signed pre-ReLU feature tap contract drift")
    norms = np.linalg.norm(pre_relu.astype(np.float64), axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1.0e-12):
        raise D107Target125RunnerError("signed pre-ReLU feature has a degenerate row")
    return np.ascontiguousarray(pre_relu / norms[:, None], dtype=np.float32)


def _package_payloads(reference: Mapping[str, Any]):
    try:
        from .somph_diagnostic_bundle_loader import load_verified_somph_predictor_bundle
    except Exception as error:  # pragma: no cover - environment/import closure
        raise D107Target125RunnerError("D92 package loader is unavailable") from error
    try:
        payloads, manifest, audit = load_verified_somph_predictor_bundle(
            Path(str(reference["package_root"])),
            detached_seal_path=Path(str(reference["detached_seal_path"])),
            expected_seal_sha256=_sha(
                reference["expected_seal_sha256"], "D92 package seal SHA256"
            ),
        )
    except Exception as error:
        raise D107Target125RunnerError("sealed D92 package verification failed") from error
    if not isinstance(payloads, Mapping) or not isinstance(manifest, Mapping):
        raise D107Target125RunnerError("sealed D92 package materialization drift")
    return payloads, dict(manifest), dict(audit)


def _support_rows(
    payload: Mapping[str, Any], *, registered_classes: tuple[str, ...], active_k: int
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    try:
        from .somph_predictor_bundle import SUPPORT_NPZ_MEMBERS
    except Exception as error:  # pragma: no cover - import closure
        raise D107Target125RunnerError("D92 support package schema is unavailable") from error
    if set(payload) != set(SUPPORT_NPZ_MEMBERS) - {"manifest_json"}:
        raise D107Target125RunnerError("D92 support payload allowlist drift")
    ranks = np.asarray(payload["support_rank_within_class"])
    indices = np.asarray(payload["support_class_indices"])
    ids = tuple(np.asarray(payload["support_tokens"]).astype(str).tolist())
    iq = np.asarray(payload["support_leo_weak_iq"])
    if (
        ranks.dtype.kind not in "iu"
        or indices.dtype.kind not in "iu"
        or ranks.ndim != 1
        or ranks.shape != indices.shape
        or len(ids) != len(ranks)
        or iq.dtype != np.float32
        or iq.ndim != 3
        or iq.shape[1] != 2
        or len(iq) != len(ranks)
        or not np.isfinite(iq).all()
    ):
        raise D107Target125RunnerError("D92 support IQ/index contract drift")
    mask = ranks.astype(np.int64) < active_k
    selected_indices = indices.astype(np.int64)[mask]
    if (
        len(selected_indices) != active_k * len(registered_classes)
        or len(selected_indices) == 0
        or int(selected_indices.min()) != 0
        or int(selected_indices.max()) != len(registered_classes) - 1
        or any(int(np.sum(selected_indices == index)) != active_k for index in range(len(registered_classes)))
    ):
        raise D107Target125RunnerError("D92 support balanced active-K binding drift")
    selected_ids = tuple(item for item, use in zip(ids, mask, strict=True) if use)
    if len(set(selected_ids)) != len(selected_ids) or any(not item for item in selected_ids):
        raise D107Target125RunnerError("D92 support physical ID drift")
    labels = tuple(registered_classes[index] for index in selected_indices.tolist())
    return np.ascontiguousarray(iq[mask], dtype=np.float32), labels, selected_ids


def _query_rows(payload: Mapping[str, Any]) -> tuple[np.ndarray, tuple[str, ...]]:
    try:
        from .somph_predictor_bundle import QUERY_NPZ_MEMBERS
    except Exception as error:  # pragma: no cover - import closure
        raise D107Target125RunnerError("D92 query package schema is unavailable") from error
    if set(payload) != set(QUERY_NPZ_MEMBERS) - {"manifest_json"}:
        raise D107Target125RunnerError(
            "D92 query payload allowlist drift; truth/role fields are forbidden"
        )
    iq = np.asarray(payload["query_leo_weak_iq"])
    ids = tuple(np.asarray(payload["query_tokens"]).astype(str).tolist())
    if (
        iq.dtype != np.float32
        or iq.ndim != 3
        or iq.shape[1] != 2
        or len(iq) != len(ids)
        or len(iq) < 1
        or not np.isfinite(iq).all()
        or len(set(ids)) != len(ids)
        or any(not item for item in ids)
    ):
        raise D107Target125RunnerError("D92 query IQ/ID contract drift")
    return np.ascontiguousarray(iq, dtype=np.float32), ids


class _D107RealStateMaterializer:
    """Strict D92/package/checkpoint adapter with no target labels or roles."""

    def __init__(self, *, plan: Mapping[str, Any], device: str, feature_batch_size: int) -> None:
        if type(feature_batch_size) is not int or feature_batch_size < 1:
            raise D107Target125RunnerError("feature_batch_size must be positive")
        self.plan = plan
        self.feature_batch_size = feature_batch_size
        identity = plan["identity"]
        checkpoint = identity.get("checkpoint")
        method_lock = identity.get("d107_method_lock")
        rdce = identity.get("rdce_asset")
        if not all(isinstance(value, Mapping) for value in (checkpoint, method_lock, rdce)):
            raise D107Target125RunnerError("prepared runtime asset binding missing")
        self.checkpoint_path = _regular_file(Path(str(checkpoint.get("path"))), "checkpoint")
        self.checkpoint_sha256 = _sha(checkpoint.get("sha256"), "checkpoint SHA256")
        if _sha256_file(self.checkpoint_path) != self.checkpoint_sha256:
            raise D107Target125RunnerError("checkpoint SHA drift")
        lock_path = _regular_file(Path(str(method_lock.get("path"))), "D107 method lock")
        if _sha256_file(lock_path) != _sha(method_lock.get("sha256"), "D107 method-lock SHA256"):
            raise D107Target125RunnerError("D107 method-lock SHA drift")
        self.checkpoint_bytes = self.checkpoint_path.read_bytes()
        self.package_cache: dict[tuple[tuple[str, str], ...], tuple[Any, dict[str, Any], dict[str, Any]]] = {}
        self.model: Any = None
        self.model_input_len: int | None = None
        try:
            from .stage2_d105_query_evaluation import _device
        except Exception as error:  # pragma: no cover - environment/import closure
            raise D107Target125RunnerError("D105 checkpoint device adapter unavailable") from error
        self.device = _device(device)
        self.tau, self.spectrum = self._rdce_summary(rdce)

    def _rdce_summary(self, reference: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        try:
            from .stage2_d106_rdce_asset import (
                ASSET_WIRE_NAME,
                D106RDCEAssetLineage,
                WIRE_MAGIC,
                decode_d106_rdce_spectrum,
                decode_d106_rdce_tau,
                deserialize_d106_rdce_asset,
            )
        except Exception as error:  # pragma: no cover - import closure
            raise D107Target125RunnerError("RDCE public asset API is unavailable") from error
        if set(reference) != {"directory", "wire_sha256"}:
            raise D107Target125RunnerError("RDCE asset plan field closure drift")
        try:
            directory = Path(str(reference["directory"]))
            if not directory.is_dir() or directory.is_symlink():
                raise D107Target125RunnerError("RDCE asset directory drift")
            if {member.name for member in directory.iterdir()} != {ASSET_WIRE_NAME}:
                raise D107Target125RunnerError("RDCE asset directory member drift")
            wire_path = _regular_file(directory / ASSET_WIRE_NAME, "RDCE wire")
            wire = wire_path.read_bytes()
            expected_wire = _sha(reference["wire_sha256"], "RDCE wire SHA256")
            if hashlib.sha256(wire).hexdigest() != expected_wire or not wire.startswith(WIRE_MAGIC):
                raise D107Target125RunnerError("RDCE wire SHA/magic drift")
            offset = len(WIRE_MAGIC)
            if len(wire) < offset + 4:
                raise D107Target125RunnerError("RDCE wire header is truncated")
            header_size = struct.unpack(">I", wire[offset : offset + 4])[0]
            offset += 4
            if header_size < 1 or offset + header_size > len(wire):
                raise D107Target125RunnerError("RDCE wire header length drift")
            header_raw = wire[offset : offset + header_size]
            header = json.loads(header_raw.decode("utf-8"))
            if (
                not isinstance(header, Mapping)
                or canonical_bytes(header) != header_raw
                or not isinstance(header.get("asset"), Mapping)
            ):
                raise D107Target125RunnerError("RDCE wire canonical header drift")
            lineage_names = (
                "checkpoint_sha256",
                "runtime_sha256",
                "method_lock_sha256",
                "split_id",
                "tap_sha256",
                "construction_code_sha256",
                "content_root_sha256",
                "source_receipt_sha256",
                "tap_receipt_sha256",
                "tap_authority_sha256",
            )
            asset_header = header["asset"]
            if any(name not in asset_header for name in lineage_names):
                raise D107Target125RunnerError("RDCE wire lineage field closure drift")
            lineage = D106RDCEAssetLineage(
                **{name: asset_header[name] for name in lineage_names}
            )
            asset = deserialize_d106_rdce_asset(
                wire,
                expected_wire_sha256=expected_wire,
                expected_lineage=lineage,
            )
            tau = decode_d106_rdce_tau(asset)
            spectrum = decode_d106_rdce_spectrum(asset)
        except Exception as error:
            raise D107Target125RunnerError("RDCE asset/lineage verification failed") from error
        if asset.checkpoint_sha256 != self.checkpoint_sha256:
            raise D107Target125RunnerError("RDCE/checkpoint binding drift")
        return (
            _readonly_positive_vector(tau, "RDCE tau"),
            _readonly_positive_vector(spectrum, "RDCE spectrum"),
        )

    def _package(self, reference: Mapping[str, Any]):
        key = tuple(sorted((str(name), str(value)) for name, value in reference.items()))
        cached = self.package_cache.get(key)
        if cached is None:
            cached = _package_payloads(reference)
            self.package_cache[key] = cached
        return cached

    def _model_for(self, input_len: int):
        if self.model is None:
            try:
                from .stage2_d105_query_evaluation import _default_model_loader

                self.model, _audit = _default_model_loader(
                    self.checkpoint_bytes, input_len, self.device
                )
            except Exception as error:
                raise D107Target125RunnerError("checkpoint safe loader failed") from error
            self.model_input_len = input_len
        elif self.model_input_len != input_len:
            raise D107Target125RunnerError("received-IQ input length drift")
        return self.model

    def _signed_tap(self, iq: np.ndarray) -> np.ndarray:
        try:
            import torch

            from .stage2_d105_feature_tap import extract_d105_feature_tap
        except Exception as error:  # pragma: no cover - import closure
            raise D107Target125RunnerError("signed z_id feature tap is unavailable") from error
        model = self._model_for(int(iq.shape[-1]))
        rows: list[np.ndarray] = []
        for start in range(0, len(iq), self.feature_batch_size):
            batch = np.ascontiguousarray(iq[start : start + self.feature_batch_size], dtype=np.float32)
            tensor = torch.as_tensor(batch, dtype=torch.float32, device=self.device)
            try:
                with torch.no_grad():
                    tapped = extract_d105_feature_tap(model, tensor)
            except Exception as error:
                raise D107Target125RunnerError("one-pass signed z_id feature tap failed") from error
            rows.append(_signed_zid_l2_from_tap(tapped, len(batch)))
        return np.ascontiguousarray(np.concatenate(rows, axis=0), dtype=np.float32)

    def __call__(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        phase = request.get("phase")
        if phase not in PHASES:
            raise D107Target125RunnerError("state request phase drift")
        packages = request.get("packages")
        if not isinstance(packages, Mapping):
            raise D107Target125RunnerError("state request packages missing")
        support_payloads, support_manifest, _support_audit = self._package(
            packages[f"{phase}_enrollment"]
        )
        query_payloads, query_manifest, _query_audit = self._package(
            packages[f"{phase}_apply"]
        )
        try:
            from .stage2_diag_cosine_exploration import _validate_matched_packages

            _validate_matched_packages(support_manifest, query_manifest)
        except Exception as error:
            raise D107Target125RunnerError("D92 support/query package pairing drift") from error
        scene = request["scene"]
        if scene not in support_payloads or scene not in query_payloads:
            raise D107Target125RunnerError("D92 package scene is missing")
        registry = tuple(
            str(item.get("class_handle", ""))
            for item in support_manifest.get("registered_classes", [])
            if isinstance(item, Mapping)
        )
        if not registry or len(set(registry)) != len(registry):
            raise D107Target125RunnerError("D92 registered-class contract drift")
        for manifest in (support_manifest, query_manifest):
            if (
                manifest.get("receiver") != request["receiver"]
                or manifest.get("seed") != request["seed"]
                or manifest.get("k_shot") != request["source_pool_k"]
            ):
                raise D107Target125RunnerError("D92 package row binding drift")
        support_iq, labels, support_ids = _support_rows(
            support_payloads[scene],
            registered_classes=registry,
            active_k=request["k_shot"],
        )
        query_iq, query_ids = _query_rows(query_payloads[scene])
        if support_iq.shape[1:] != query_iq.shape[1:]:
            raise D107Target125RunnerError("support/query IQ shape drift")
        # Exactly one checkpoint-forward input per phase: split only after it
        # returns signed z_id.  Query rows never feed a state-construction API.
        signed = self._signed_tap(
            np.ascontiguousarray(np.concatenate((support_iq, query_iq), axis=0), dtype=np.float32)
        )
        return {
            "support_signed": signed[: len(support_iq)],
            "support_labels": labels,
            "registered_classes": registry,
            "support_physical_ids": support_ids,
            "query_signed": signed[len(support_iq) :],
            "query_physical_ids": query_ids,
            "tau": self.tau,
            "spectrum": self.spectrum,
        }


def _state_request(row: Mapping[str, Any], scene: str, phase: str) -> dict[str, Any]:
    if scene not in SCENES or phase not in PHASES:
        raise D107Target125RunnerError("internal scene/phase request drift")
    return {
        "schema": "cvs.phase2.d107.scmkrr.target125.state_request.v1",
        "outer_id": row["outer_id"],
        "receiver": row["receiver"],
        "seed": row["seed"],
        "k_shot": row["k_shot"],
        "active_k": row["active_k"],
        "new_count": row["new_count"],
        "source_pool_k": row["source_pool_k"],
        "scene": scene,
        "phase": phase,
        "packages": row["packages"],
        "access_ledger": dict(ACCESS_LEDGER),
    }


def _materialize_pair(
    materializer: StateMaterializer, row: Mapping[str, Any], scene: str
) -> tuple[_MaterializedState, _MaterializedState]:
    before_request = _state_request(row, scene, "before")
    after_request = _state_request(row, scene, "after")
    before = _coerce_materialized_state(materializer(before_request), request=before_request)
    after = _coerce_materialized_state(materializer(after_request), request=after_request)
    if len(before.registered_classes) != OLD_CLASS_COUNT:
        raise D107Target125RunnerError("before state must retain exactly six old classes")
    if (
        after.registered_classes[:OLD_CLASS_COUNT] != before.registered_classes
        or len(after.registered_classes) != OLD_CLASS_COUNT + row["new_count"]
    ):
        raise D107Target125RunnerError("before/after registry registration drift")
    if not np.array_equal(before.tau, after.tau) or not np.array_equal(
        before.spectrum, after.spectrum
    ):
        raise D107Target125RunnerError("before/after RDCE summary drift")
    return before, after


def _resolve_core(
    state_builder: StateBuilder | None, query_scorer: QueryScorer | None
) -> tuple[StateBuilder, QueryScorer]:
    if state_builder is None or query_scorer is None:
        try:
            from .stage2_d107_scmkrr import (
                ARMS as core_arms,
                build_scmkrr_state,
                score_scmkrr_query,
            )
        except Exception as error:  # pragma: no cover - core lands independently
            raise D107Target125RunnerError("D107 SCMKRR public API is unavailable") from error
        if tuple(core_arms) != ARMS:
            raise D107Target125RunnerError("D107 SCMKRR public arm contract drift")
        state_builder = build_scmkrr_state if state_builder is None else state_builder
        query_scorer = score_scmkrr_query if query_scorer is None else query_scorer
    if not callable(state_builder) or not callable(query_scorer):
        raise D107Target125RunnerError("SCMKRR state/query API must be callable")
    return state_builder, query_scorer


def _predict_labels(
    materialized: _MaterializedState,
    *,
    anchor_signed: np.ndarray,
    arm: str,
    state_builder: StateBuilder,
    query_scorer: QueryScorer,
) -> tuple[str, ...]:
    try:
        state = state_builder(
            support_signed=materialized.support_signed,
            labels=materialized.support_labels,
            registered_classes=materialized.registered_classes,
            anchor_signed=anchor_signed,
            tau=materialized.tau,
            spectrum=materialized.spectrum,
            arm=arm,
        )
        logits = np.asarray(query_scorer(state, materialized.query_signed))
    except Exception as error:
        raise D107Target125RunnerError("SCMKRR state/query evaluation failed closed") from error
    if (
        logits.dtype.kind not in "f"
        or logits.shape != (len(materialized.query_signed), len(materialized.registered_classes))
        or not np.isfinite(logits).all()
    ):
        raise D107Target125RunnerError("SCMKRR query logits contract drift")
    maxima = np.max(logits, axis=1, keepdims=True)
    if np.any(np.sum(logits == maxima, axis=1) != 1):
        raise D107Target125RunnerError("SCMKRR exact top tie must fail closed")
    indices = np.argmax(logits, axis=1)
    return tuple(materialized.registered_classes[int(index)] for index in indices.tolist())


def _surface_record(
    *,
    surface: Any,
    materialized: _MaterializedState,
    predicted_labels: tuple[str, ...],
    prediction_root: Path,
) -> dict[str, Any]:
    query_ids = list(materialized.query_physical_ids)
    labels = list(predicted_labels)
    if len(labels) != len(query_ids) or any(label not in materialized.registered_classes for label in labels):
        raise D107Target125RunnerError("SCMKRR predicted-label closure drift")
    payload: dict[str, Any] = {
        "schema": PREDICTION_ARTIFACT_SCHEMA,
        "surface_id": surface.surface_id,
        "outer_id": surface.outer_id,
        "receiver": surface.receiver,
        "seed": surface.seed,
        "k_shot": surface.k_shot,
        "new_count": surface.new_count,
        "scene": surface.scene,
        "arm": surface.arm,
        "phase": surface.phase,
        "registered_classes": list(materialized.registered_classes),
        "ordered_query_physical_ids": query_ids,
        "ordered_query_physical_ids_sha256": canonical_sha256(query_ids),
        "predicted_labels": labels,
        "predicted_labels_sha256": canonical_sha256(labels),
        "access_ledger": dict(ACCESS_LEDGER),
        "truth_open": False,
        "immutable": True,
    }
    payload["artifact_receipt_sha256"] = canonical_sha256(payload)
    artifact_path = prediction_root / f"{surface.surface_id}.json"
    artifact_file_sha = _write_json_new(artifact_path, payload)
    return {
        "surface_id": surface.surface_id,
        "outer_id": surface.outer_id,
        "receiver": surface.receiver,
        "seed": surface.seed,
        "k_shot": surface.k_shot,
        "new_count": surface.new_count,
        "scene": surface.scene,
        "arm": surface.arm,
        "phase": surface.phase,
        "registered_classes": list(materialized.registered_classes),
        "prediction_artifact": f"predictions/{surface.surface_id}.json",
        "prediction_artifact_sha256": artifact_file_sha,
        "ordered_query_physical_ids": query_ids,
        "ordered_query_physical_ids_sha256": canonical_sha256(query_ids),
        "predicted_labels": labels,
        "predicted_labels_sha256": canonical_sha256(labels),
        "access_ledger": dict(ACCESS_LEDGER),
        "truth_open": False,
        "immutable": True,
    }


def smoke_d107_target125_prepared_state(
    *,
    plan_manifest_path: Path,
    expected_plan_file_sha256: str,
    context_manifest_path: Path,
    expected_context_file_sha256: str,
    output_dir: Path,
    row_index: int = 0,
    scene_index: int = 0,
    device: str = "cpu",
    feature_batch_size: int = 64,
    state_materializer: StateMaterializer | None = None,
    state_builder: StateBuilder | None = None,
    query_scorer: QueryScorer | None = None,
) -> dict[str, Any]:
    """Exercise both states and all four arms for one sealed Target125 row."""

    plan, context = _prepared_inputs(
        plan_manifest_path=plan_manifest_path,
        expected_plan_file_sha256=expected_plan_file_sha256,
        context_manifest_path=context_manifest_path,
        expected_context_file_sha256=expected_context_file_sha256,
    )
    if (
        type(row_index) is not int
        or type(scene_index) is not int
        or row_index not in range(OUTER_JOB_COUNT)
        or scene_index not in range(len(SCENES))
    ):
        raise D107Target125RunnerError("smoke row/scene index drift")
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"immutable smoke output already exists: {destination}")
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise D107Target125RunnerError("unsafe smoke output parent")
    destination.mkdir()
    materializer = (
        _D107RealStateMaterializer(
            plan=plan, device=device, feature_batch_size=feature_batch_size
        )
        if state_materializer is None
        else state_materializer
    )
    builder, scorer = _resolve_core(state_builder, query_scorer)
    row = context["rows"][row_index]
    scene = SCENES[scene_index]
    before, after = _materialize_pair(materializer, row, scene)
    for arm in ARMS:
        _predict_labels(
            before,
            anchor_signed=before.support_signed,
            arm=arm,
            state_builder=builder,
            query_scorer=scorer,
        )
        _predict_labels(
            after,
            anchor_signed=before.support_signed,
            arm=arm,
            state_builder=builder,
            query_scorer=scorer,
        )
    receipt: dict[str, Any] = {
        "schema": SMOKE_RECEIPT_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "protocol_schema": PROTOCOL_SCHEMA,
        "outer_id": row["outer_id"],
        "scene": scene,
        "phase_count": len(PHASES),
        "arm_count": len(ARMS),
        "before_anchor_signed_sha256": _array_sha256(before.support_signed),
        "before_query_ids_sha256": canonical_sha256(list(before.query_physical_ids)),
        "after_query_ids_sha256": canonical_sha256(list(after.query_physical_ids)),
        "access_ledger": dict(ACCESS_LEDGER),
    }
    receipt["smoke_receipt_sha256"] = canonical_sha256(receipt)
    receipt_path = destination / "smoke_receipt.json"
    receipt_file_sha = _write_json_new(receipt_path, receipt)
    return {
        **receipt,
        "smoke_receipt": str(receipt_path),
        "smoke_receipt_file_sha256": receipt_file_sha,
    }


def predict_d107_target125(
    *,
    plan_manifest_path: Path,
    expected_plan_file_sha256: str,
    context_manifest_path: Path,
    expected_context_file_sha256: str,
    output_dir: Path,
    device: str = "cpu",
    feature_batch_size: int = 64,
    state_materializer: StateMaterializer | None = None,
    state_builder: StateBuilder | None = None,
    query_scorer: QueryScorer | None = None,
) -> dict[str, Any]:
    """Seal all 3,000 D107 prediction surfaces before any truth can open."""

    plan, context = _prepared_inputs(
        plan_manifest_path=plan_manifest_path,
        expected_plan_file_sha256=expected_plan_file_sha256,
        context_manifest_path=context_manifest_path,
        expected_context_file_sha256=expected_context_file_sha256,
    )
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"immutable prediction output already exists: {destination}")
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise D107Target125RunnerError("unsafe prediction output parent")
    destination.mkdir()
    prediction_root = destination / "predictions"
    prediction_root.mkdir()
    materializer = (
        _D107RealStateMaterializer(
            plan=plan, device=device, feature_batch_size=feature_batch_size
        )
        if state_materializer is None
        else state_materializer
    )
    builder, scorer = _resolve_core(state_builder, query_scorer)
    matrix = freeze_d107_target125_matrix()
    surface_by_key = {
        (surface.outer_id, surface.scene, surface.arm, surface.phase): surface
        for surface in matrix.surfaces
    }
    outer_rows: list[dict[str, Any]] = []
    manifest_surfaces: list[dict[str, Any]] = []
    for row, expected_outer in zip(context["rows"], matrix.outer_rows, strict=True):
        frozen_old: tuple[str, ...] | None = None
        frozen_new: tuple[str, ...] | None = None
        for scene in SCENES:
            before, after = _materialize_pair(materializer, row, scene)
            old_classes = before.registered_classes
            new_classes = after.registered_classes[len(old_classes) :]
            if frozen_old is None:
                frozen_old, frozen_new = old_classes, new_classes
            elif frozen_old != old_classes or frozen_new != new_classes:
                raise D107Target125RunnerError("outer registry differs across scenes")
            for arm in ARMS:
                for phase, materialized in (("before", before), ("after", after)):
                    surface = surface_by_key[
                        (row["outer_id"], scene, arm, phase)
                    ]
                    labels = _predict_labels(
                        materialized,
                        anchor_signed=before.support_signed,
                        arm=arm,
                        state_builder=builder,
                        query_scorer=scorer,
                    )
                    manifest_surfaces.append(
                        _surface_record(
                            surface=surface,
                            materialized=materialized,
                            predicted_labels=labels,
                            prediction_root=prediction_root,
                        )
                    )
        if frozen_old is None or frozen_new is None:
            raise D107Target125RunnerError("outer scene loop did not materialize")
        outer_rows.append(
            {
                "outer_id": expected_outer.outer_id,
                "receiver": expected_outer.receiver,
                "seed": expected_outer.seed,
                "k_shot": expected_outer.k_shot,
                "new_count": expected_outer.new_count,
                "old_classes": list(frozen_old),
                "new_classes": list(frozen_new),
            }
        )
    manifest: dict[str, Any] = {
        "schema": PREDICTION_MANIFEST_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "protocol_schema": PROTOCOL_SCHEMA,
        "manifest_sealed": True,
        "truth_open": False,
        "outer_job_count": OUTER_JOB_COUNT,
        "scene_row_count": SCENE_ROW_COUNT,
        "arm_pair_count": ARM_PAIR_COUNT,
        "surface_count": SURFACE_COUNT,
        "scenes": list(SCENES),
        "arms": list(ARMS),
        "phases": list(PHASES),
        "outer_rows": outer_rows,
        "access_ledger": dict(ACCESS_LEDGER),
        "surfaces": manifest_surfaces,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_path = destination / "prediction_manifest.json"
    manifest_file_sha = _write_json_new(manifest_path, manifest)
    validate_d107_target125_prediction_manifest(
        prediction_manifest_path=manifest_path,
        expected_prediction_manifest_file_sha256=manifest_file_sha,
    )
    return {
        "prediction_manifest": str(manifest_path),
        "prediction_manifest_file_sha256": manifest_file_sha,
        "prediction_manifest_sha256": manifest["manifest_sha256"],
        "outer_job_count": OUTER_JOB_COUNT,
        "scene_row_count": SCENE_ROW_COUNT,
        "arm_pair_count": ARM_PAIR_COUNT,
        "surface_count": SURFACE_COUNT,
    }


def _validate_outer_rows(value: Any) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    rows = _sequence(value, "prediction outer_rows", OUTER_JOB_COUNT)
    matrix = freeze_d107_target125_matrix()
    result: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for row, expected in zip(rows, matrix.outer_rows, strict=True):
        if not isinstance(row, Mapping) or set(row) != _OUTER_ROW_FIELDS:
            raise D107Target125RunnerError("prediction outer row field closure drift")
        if any(
            row.get(name) != getattr(expected, name)
            for name in ("outer_id", "receiver", "seed", "k_shot", "new_count")
        ):
            raise D107Target125RunnerError("prediction outer row identity drift")
        old = _tokens(row.get("old_classes"), "old_classes", len(row.get("old_classes", [])))
        new = _tokens(row.get("new_classes"), "new_classes", len(row.get("new_classes", [])))
        if len(old) != OLD_CLASS_COUNT or set(old).intersection(new) or len(new) != expected.new_count:
            raise D107Target125RunnerError("prediction old/new registry closure drift")
        result[expected.outer_id] = (old, new)
    return result


def _validate_artifact(
    *, surface: Mapping[str, Any], root: Path
) -> None:
    relative = surface.get("prediction_artifact")
    if type(relative) is not str or relative != f"predictions/{surface['surface_id']}.json":
        raise D107Target125RunnerError("prediction artifact relative path drift")
    path = root / relative
    source = _regular_file(path, "prediction artifact")
    if _sha256_file(source) != _sha(
        surface.get("prediction_artifact_sha256"), "prediction artifact SHA256"
    ):
        raise D107Target125RunnerError("prediction artifact file SHA drift")
    artifact = _read_json(source, name="prediction artifact")
    if set(artifact) != _ARTIFACT_FIELDS or artifact.get("schema") != PREDICTION_ARTIFACT_SCHEMA:
        raise D107Target125RunnerError("prediction artifact field/schema drift")
    _receipt(artifact, "artifact_receipt_sha256", "prediction artifact")
    projected = {
        name: value
        for name, value in artifact.items()
        if name not in {"schema", "artifact_receipt_sha256"}
    }
    expected = {
        name: value
        for name, value in surface.items()
        if name not in {"prediction_artifact", "prediction_artifact_sha256"}
    }
    if projected != expected:
        raise D107Target125RunnerError("prediction artifact/manifest content drift")


def validate_d107_target125_prediction_manifest(
    *,
    prediction_manifest_path: Path,
    expected_prediction_manifest_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify full pre-truth closure, immutable artifacts, and query equality."""

    path = _regular_file(prediction_manifest_path, "prediction manifest")
    manifest = _read_json(
        path,
        name="prediction manifest",
        expected_file_sha256=expected_prediction_manifest_file_sha256,
    )
    if set(manifest) != _MANIFEST_FIELDS:
        raise D107Target125RunnerError("prediction manifest field closure drift")
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
    ):
        raise D107Target125RunnerError("prediction manifest identity/count closure drift")
    _receipt(manifest, "manifest_sha256", "prediction manifest")
    _access_ledger(manifest.get("access_ledger"), "prediction manifest access ledger")
    outer_registries = _validate_outer_rows(manifest.get("outer_rows"))
    surfaces = _sequence(manifest.get("surfaces"), "prediction surfaces", SURFACE_COUNT)
    matrix = freeze_d107_target125_matrix()
    query_by_scope: dict[tuple[str, str, str], tuple[str, ...]] = {}
    for surface, expected in zip(surfaces, matrix.surfaces, strict=True):
        if not isinstance(surface, Mapping) or set(surface) != _SURFACE_FIELDS:
            raise D107Target125RunnerError("prediction surface field closure drift")
        if any(
            surface.get(name) != getattr(expected, name)
            for name in (
                "surface_id",
                "outer_id",
                "receiver",
                "seed",
                "k_shot",
                "new_count",
                "scene",
                "arm",
                "phase",
            )
        ):
            raise D107Target125RunnerError("prediction surface matrix identity drift")
        old, new = outer_registries[expected.outer_id]
        registry = _tokens(
            surface.get("registered_classes"),
            "surface registered_classes",
            len(surface.get("registered_classes", [])),
        )
        expected_registry = old if expected.phase == "before" else old + new
        if registry != expected_registry:
            raise D107Target125RunnerError("surface before/after registry drift")
        query_ids = _tokens(
            surface.get("ordered_query_physical_ids"),
            "ordered query physical IDs",
            len(surface.get("ordered_query_physical_ids", [])),
        )
        if surface.get("ordered_query_physical_ids_sha256") != canonical_sha256(list(query_ids)):
            raise D107Target125RunnerError("ordered query physical-ID receipt drift")
        labels = _tokens(
            surface.get("predicted_labels"),
            "predicted labels",
            len(query_ids),
            require_unique=False,
        )
        if any(label not in registry for label in labels):
            raise D107Target125RunnerError("predicted label is outside its registry")
        if surface.get("predicted_labels_sha256") != canonical_sha256(list(labels)):
            raise D107Target125RunnerError("predicted-label receipt drift")
        if surface.get("truth_open") is not False or surface.get("immutable") is not True:
            raise D107Target125RunnerError("surface truth/immutability flag drift")
        _access_ledger(surface.get("access_ledger"), "surface access ledger")
        scope = (expected.outer_id, expected.scene, expected.phase)
        known = query_by_scope.get(scope)
        if known is None:
            query_by_scope[scope] = query_ids
        elif known != query_ids:
            raise D107Target125RunnerError("four-arm query order differs within a phase")
        _validate_artifact(surface=surface, root=path.parent)
    if len(query_by_scope) != SCENE_ROW_COUNT * len(PHASES):
        raise D107Target125RunnerError("prediction query-scope coverage drift")
    return manifest


__all__ = [
    "D107Target125RunnerError",
    "PREDICTION_ARTIFACT_SCHEMA",
    "PREDICTION_MANIFEST_SCHEMA",
    "SMOKE_RECEIPT_SCHEMA",
    "predict_d107_target125",
    "prepare_d107_target125_run",
    "smoke_d107_target125_prepared_state",
    "validate_d107_target125_prediction_manifest",
]
