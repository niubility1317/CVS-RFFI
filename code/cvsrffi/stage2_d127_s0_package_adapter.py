"""Strict D92-retry2 package adapter for the frozen D127 S0 matrix.

This is deliberately only the target-input half of D127.  It opens the six
already sealed D92 retry2 package pairs selected by the D106 context, produces
the two complete truth-free S0 state matrices, and records the K5-from-K10
prefix proof.  It neither loads truth nor calls a model, scorer, optimiser, or
remote service.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from .stage2_d106_matrix_protocol import LEO_SCENARIOS
from .stage2_d106_target25_runner import (
    _d106_query_rows,
    _d106_support_rows,
    _load_raw_package,
)
from .stage2_diag_cosine_exploration import _validate_matched_packages
from .stage2_d127_s0_entry import D127S0Row
from . import stage2_d127_s0_entry as entry
from . import stage2_d127_phase1_release as phase1_release
from .stage2_zid_student_t_qknn import Phase1ZIDStudentTLock


METHOD_LOCK_SCHEMA = "cvs.stage2.d127.joint_s0.method_lock.v1"
D106_CONTEXT_SCHEMA = "cvs.phase2.d106.target25_input_context.v2"
PREFIX_RECEIPT_SCHEMA = "cvs.stage2.d127.s0.k5_k10_prefix_receipt.v1"
CAPSULE_SCHEMA = "cvs.stage2.d127.s0.d92_capsule.v1"
SPLIT_SCHEMA = "cvs.stage2.d127.s0.d92_split.v1"
STATE_INPUT_SCHEMA = "cvs.stage2.d127.s0.state_input.v1"
PREPARED_PLAN_SCHEMA = "cvs.stage2.d127.s0.prepared_plan.v1"
CANDIDATE_WORKER_SCHEMA = "cvs.stage2.d127.s0.candidate_worker_pair.v1"
PAIRED_PREDICTION_SCHEMA = "cvs.stage2.d127.s0.paired_prediction.v1"
SCORER_PAIR_MANIFEST_SCHEMA = "cvs.stage2.d127.s0.scorer_pair_manifest.v1"
PHASE1_MANIFEST_RECEIPT_SCHEMA = "cvs.stage2.d127.s0.phase1_manifest_receipt.v1"

S0_SEED = 713102
S0_RECEIVERS = ("20-1", "3-19", "7-14")
S0_K_NEW = ((1, 20), (5, 20))
S0_STATES = ("before", "after")
S0_SCENES = tuple(LEO_SCENARIOS)
S0_ROW_COUNT = 18
_RAW_ROW_FIELDS = {
    "job_id",
    "source_d92_job_id",
    "receiver",
    "seed",
    "k_shot",
    "source_pool_k",
    "new_count",
    "packages",
}
_PACKAGE_NAMES = {
    "before_enrollment",
    "before_apply",
    "after_enrollment",
    "after_apply",
}
_PACKAGE_REF_FIELDS = {"package_root", "detached_seal_path", "expected_seal_sha256"}
_ARM_IDS = ("M0", "M_DA", "M_L92", "M_JOINT")
_COMMON_ARM_IDS = ("M0", "M_L92")
_CANDIDATE_ARM_IDS = ("M_DA", "M_JOINT")
_FORBIDDEN_NORMALIZED_KEYS = frozenset(
    {
        "truth",
        "querytruth",
        "role",
        "roles",
        "queryrole",
        "quota",
        "classquota",
        "globalreassignment",
    }
)


class D127S0PackageAdapterError(ValueError):
    """Raised when the S0 package locator or materialized state drifts."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise D127S0PackageAdapterError(message)


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise D127S0PackageAdapterError("canonical JSON value is invalid") from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


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
        raise D127S0PackageAdapterError(f"{name} must be a lowercase SHA256")
    return value


def _read_json_sha(path: str | Path, expected_sha256: str, name: str) -> tuple[dict[str, Any], str]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise D127S0PackageAdapterError(f"{name} must be a regular non-symlink file")
    source = source.resolve(strict=True)
    expected = _sha(expected_sha256, f"expected {name} SHA256")
    if _sha256_file(source) != expected:
        raise D127S0PackageAdapterError(f"{name} SHA mismatch")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise D127S0PackageAdapterError(f"{name} is not valid UTF-8 JSON") from exc
    if type(value) is not dict:
        raise D127S0PackageAdapterError(f"{name} must contain a JSON object")
    return value, expected


def _forbidden_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            text = str(key)
            normalized = "".join(character for character in text.lower() if character.isalnum())
            if normalized in _FORBIDDEN_NORMALIZED_KEYS:
                found.add(text)
            found.update(_forbidden_keys(item))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            found.update(_forbidden_keys(item))
    return found


def _opaque_root(values: Sequence[str]) -> str:
    frozen = tuple(str(value) for value in values)
    _require(bool(frozen) and len(set(frozen)) == len(frozen) and all(frozen), "opaque token closure drift")
    # The receipt intentionally stores only the digest, never the tokens.
    return _canonical_sha256(sorted(frozen))


def _registry(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    values = manifest.get("registered_classes")
    _require(isinstance(values, list) and len(values) >= 2, "D92 registry is missing")
    result = tuple(
        str(item.get("class_handle", ""))
        for item in values
        if isinstance(item, Mapping)
    )
    _require(
        len(result) == len(values) and len(set(result)) == len(result) and all(result),
        "D92 registry handle closure drift",
    )
    return result


def _validate_method_lock(value: Mapping[str, Any]) -> None:
    _require(value.get("schema") == METHOD_LOCK_SCHEMA, "D127 method-lock schema drift")
    _require(value.get("protocol_schema") == "p2_min_v1", "D127 protocol-schema drift")
    _require(value.get("claim_scope") == "TARGET_DEVELOPMENT_S0_ONLY", "D127 S0 claim scope drift")
    checkpoint = value.get("checkpoint")
    _require(isinstance(checkpoint, Mapping), "D127 checkpoint lock missing")
    _sha(checkpoint.get("sha256"), "D127 checkpoint SHA256")
    phase1 = value.get("phase1_asset_build")
    _require(isinstance(phase1, Mapping), "D127 Phase1 asset-build lock missing")
    for name in (
        "source_received_iq_sha256",
        "source_received_iq_receipt_sha256",
        "source_label_join_archive_sha256",
    ):
        _sha(phase1.get(name), f"D127 Phase1 {name}")
    matrix = value.get("s0_matrix")
    _require(isinstance(matrix, Mapping), "D127 S0 matrix lock missing")
    _require(
        matrix.get("seed") == S0_SEED
        and tuple(matrix.get("receivers", ())) == S0_RECEIVERS
        and tuple(tuple(item) for item in matrix.get("k_new_count", ())) == S0_K_NEW
        and tuple(matrix.get("scenes", ())) == S0_SCENES
        and matrix.get("row_pair_count") == S0_ROW_COUNT
        and tuple(matrix.get("registration_states", ())) == S0_STATES
        and matrix.get("k5_source_pool_k") == 10
        and matrix.get("k5_is_ordered_k10_prefix") is True,
        "D127 frozen S0 matrix drift",
    )
    _require(
        matrix.get("d92_retry2_manifest_sha256")
        == "b70045e7cd45a6029bc0a1a47ada0bb72d16fdb6bc7662c43bd253bfc7e4bc5c",
        "D127 formal D92 retry2 manifest binding drift",
    )
    qknn = value.get("student_t_qknn")
    _require(isinstance(qknn, Mapping), "D127 qKNN lock missing")
    _require(tuple(qknn.get("active_k", ())) == (1, 5), "D127 qKNN active-K drift")
    _require(
        qknn.get("student_nu") == 3
        and qknn.get("kernel_effective_dim") == 12
        and qknn.get("kernel_volume_gamma") == 1
        and qknn.get("shared_h0") == 0.35
        and qknn.get("scale_prior_strength") == 2
        and qknn.get("scale_min_ratio") == 0.5
        and qknn.get("scale_max_ratio") == 2
        and qknn.get("temperature") == 0.85,
        "D127 qKNN numerical lock drift",
    )
    _sha(qknn.get("phase1_lodo_receipt_sha256"), "D127 qKNN LODO receipt")
    _sha(qknn.get("quantization_margin_audit_sha256"), "D127 qKNN quantization receipt")
    da = value.get("domain_adaptation")
    _require(isinstance(da, Mapping), "D127 DA lock missing")
    _require(
        da.get("query_fit_count") == 0
        and da.get("query_update_count") == 0
        and da.get("query_selection_count") == 0,
        "D127 method lock permits query use",
    )


def load_d127_s0_method_lock(
    path: str | Path, *, expected_sha256: str
) -> tuple[dict[str, Any], str, dict[int, Phase1ZIDStudentTLock]]:
    """Load the frozen method lock and derive its two typed qKNN locks."""

    document, digest = _read_json_sha(path, expected_sha256, "D127 method lock")
    _require(not _forbidden_keys(document), "D127 method lock contains forbidden predictor field")
    _validate_method_lock(document)
    qknn = document["student_t_qknn"]
    locks: dict[int, Phase1ZIDStudentTLock] = {}
    for active_k in (1, 5):
        locks[active_k] = Phase1ZIDStudentTLock(
            active_k=active_k,
            student_nu=float(qknn["student_nu"]),
            kernel_effective_dim=int(qknn["kernel_effective_dim"]),
            kernel_volume_gamma=float(qknn["kernel_volume_gamma"]),
            shared_h0=float(qknn["shared_h0"]),
            scale_prior_strength=float(qknn["scale_prior_strength"]),
            scale_min_ratio=float(qknn["scale_min_ratio"]),
            scale_max_ratio=float(qknn["scale_max_ratio"]),
            temperature=float(qknn["temperature"]),
            phase1_lodo_receipt_sha256=str(qknn["phase1_lodo_receipt_sha256"]),
            quantization_margin_audit_sha256=str(qknn["quantization_margin_audit_sha256"]),
        )
    return document, digest, locks


@dataclass(frozen=True, slots=True)
class D127S0StateInput:
    """One materialized state row plus opaque lineage needed by later pairing."""

    state: str
    row: D127S0Row
    source_d92_job_id: str
    source_pool_k: int
    capsule_id: str
    split_id: str
    support_token_root_sha256: str
    query_token_root_sha256: str
    registered_class_root_sha256: str
    state_input_receipt_sha256: str

    def __post_init__(self) -> None:
        _require(self.state in S0_STATES, "invalid D127 S0 registration state")
        _require(self.source_pool_k in (1, 10), "D127 source-pool K drift")
        for value, name in (
            (self.capsule_id, "capsule ID"),
            (self.split_id, "split ID"),
            (self.support_token_root_sha256, "support root"),
            (self.query_token_root_sha256, "query root"),
            (self.registered_class_root_sha256, "registry root"),
            (self.state_input_receipt_sha256, "state receipt"),
        ):
            _sha(value, name)

    def binding(self) -> dict[str, Any]:
        return {
            "schema": STATE_INPUT_SCHEMA,
            "row_id": self.row.row_id,
            "state": self.state,
            "receiver": self.row.receiver_id,
            "k_shot": self.row.k_shot,
            "scene": self.row.scene,
            "source_d92_job_id": self.source_d92_job_id,
            "source_pool_k": self.source_pool_k,
            "capsule_id": self.capsule_id,
            "split_id": self.split_id,
            "support_token_root_sha256": self.support_token_root_sha256,
            "query_token_root_sha256": self.query_token_root_sha256,
            "query_token_count": len(self.row.opaque_query_ids),
            "query_token_ordered_sha256": _canonical_sha256(list(self.row.opaque_query_ids)),
            "registered_class_root_sha256": self.registered_class_root_sha256,
            "qknn_lock_digest": self.row.qknn_lock.lock_digest,
            "state_input_receipt_sha256": self.state_input_receipt_sha256,
        }


@dataclass(frozen=True, slots=True)
class D127S0PreparedPackageRows:
    """Complete, typed, truth-free before/after S0 inputs and compact receipt."""

    method_lock_sha256: str
    checkpoint_sha256: str
    phase1_asset_expected_binding: Mapping[str, Any]
    context_sha256: str
    qknn_locks: Mapping[int, Phase1ZIDStudentTLock]
    before: tuple[D127S0StateInput, ...]
    after: tuple[D127S0StateInput, ...]
    prefix_receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        _sha(self.method_lock_sha256, "method lock SHA256")
        _sha(self.checkpoint_sha256, "D127 checkpoint SHA256")
        _sha(self.context_sha256, "D106 context SHA256")
        _require(len(self.qknn_locks) == 2 and set(self.qknn_locks) == {1, 5}, "D127 qKNN lock closure drift")
        _validate_phase1_asset_expected_binding(self.phase1_asset_expected_binding)
        _require(len(self.before) == S0_ROW_COUNT and len(self.after) == S0_ROW_COUNT, "D127 S0 state rows incomplete")
        _require(
            tuple(item.row.row_id for item in self.before) == tuple(item.row.row_id for item in self.after),
            "D127 before/after row pairing drift",
        )

    @property
    def pair_bindings(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "row_id": left.row.row_id,
                "receiver": left.row.receiver_id,
                "k_shot": left.row.k_shot,
                "scene": left.row.scene,
                "before": left.binding(),
                "after": right.binding(),
                "before_query_is_after_ordered_subset": _ordered_subset(
                    left.row.opaque_query_ids, right.row.opaque_query_ids
                ),
                "formal_d92_reference": {
                    "source_d92_job_id": left.source_d92_job_id,
                    "pipeline_receipt_required": True,
                },
            }
            for left, right in zip(self.before, self.after, strict=True)
        )


def _ordered_subset(left: Sequence[str], right: Sequence[str]) -> bool:
    """Return whether ``left`` occurs in ``right`` in the same order."""

    iterator = iter(right)
    return all(any(candidate == value for candidate in iterator) for value in left)


def _phase1_asset_expected_binding(
    method_lock: Mapping[str, Any], *, method_lock_sha256: str, qknn_locks: Mapping[int, Phase1ZIDStudentTLock]
) -> dict[str, Any]:
    phase1 = method_lock["phase1_asset_build"]
    qknn = method_lock["student_t_qknn"]
    value = {
        "method_lock_sha256": method_lock_sha256,
        "checkpoint_sha256": method_lock["checkpoint"]["sha256"],
        "source_binding": {
            "checkpoint_sha256": method_lock["checkpoint"]["sha256"],
            "method_lock_sha256": method_lock_sha256,
            "selected_received_iq_sha256": phase1["source_received_iq_sha256"],
            "selected_received_iq_receipt_sha256": phase1["source_received_iq_receipt_sha256"],
            "source_label_join_archive_sha256": phase1["source_label_join_archive_sha256"],
        },
        "qknn_lock_binding": {
            "phase1_lodo_receipt_sha256": qknn["phase1_lodo_receipt_sha256"],
            "quantization_margin_audit_sha256": qknn["quantization_margin_audit_sha256"],
            "lock_digest_by_k": {str(k): qknn_locks[k].lock_digest for k in (1, 5)},
        },
    }
    _validate_phase1_asset_expected_binding(value)
    return value


def _validate_phase1_asset_expected_binding(value: Any) -> None:
    _require(
        isinstance(value, Mapping)
        and set(value) == {"method_lock_sha256", "checkpoint_sha256", "source_binding", "qknn_lock_binding"},
        "D127 Phase1 expected binding closure drift",
    )
    _sha(value["method_lock_sha256"], "D127 Phase1 expected method-lock SHA256")
    _sha(value["checkpoint_sha256"], "D127 Phase1 expected checkpoint SHA256")
    source = value["source_binding"]
    _require(
        isinstance(source, Mapping)
        and set(source) == {
            "checkpoint_sha256",
            "method_lock_sha256",
            "selected_received_iq_sha256",
            "selected_received_iq_receipt_sha256",
            "source_label_join_archive_sha256",
        },
        "D127 Phase1 expected source binding closure drift",
    )
    _require(
        source["method_lock_sha256"] == value["method_lock_sha256"]
        and source["checkpoint_sha256"] == value["checkpoint_sha256"],
        "D127 Phase1 expected source method/checkpoint drift",
    )
    for name in source:
        _sha(source[name], f"D127 Phase1 expected source {name}")
    qknn = value["qknn_lock_binding"]
    _require(
        isinstance(qknn, Mapping)
        and set(qknn) == {"phase1_lodo_receipt_sha256", "quantization_margin_audit_sha256", "lock_digest_by_k"},
        "D127 Phase1 expected qKNN binding closure drift",
    )
    _sha(qknn["phase1_lodo_receipt_sha256"], "D127 Phase1 expected LODO receipt")
    _sha(qknn["quantization_margin_audit_sha256"], "D127 Phase1 expected quantization receipt")
    locks = qknn["lock_digest_by_k"]
    _require(isinstance(locks, Mapping) and len(locks) == 2 and set(locks) == {"1", "5"}, "D127 Phase1 expected qKNN lock closure drift")
    for k in ("1", "5"):
        _sha(locks[k], f"D127 Phase1 expected qKNN K{k} digest")


PackageLoader = Callable[[Mapping[str, Any]], tuple[Any, Mapping[str, Any], Mapping[str, Any]]]


def _target_context_rows(document: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    _require(document.get("schema") == D106_CONTEXT_SCHEMA, "D106 context schema drift")
    rows = document.get("rows")
    _require(isinstance(rows, list), "D106 context rows are missing")
    _require(not _forbidden_keys(document), "D106 context contains forbidden predictor field")
    wanted = {(receiver, k, new_count) for receiver in S0_RECEIVERS for k, new_count in S0_K_NEW}
    selected: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _RAW_ROW_FIELDS:
            raise D127S0PackageAdapterError("D106 context raw-row closure drift")
        key = (str(row["receiver"]), int(row["k_shot"]), int(row["new_count"]))
        if key not in wanted:
            continue
        if key in selected:
            raise D127S0PackageAdapterError("D127 S0 source row is duplicated")
        _require(row["seed"] == S0_SEED, "D127 S0 seed drift")
        expected_source_pool = 10 if key[1] == 5 else 1
        _require(row["source_pool_k"] == expected_source_pool, "D127 S0 source-pool K drift")
        source_job = str(row["source_d92_job_id"])
        expected_fragment = f"_k_{expected_source_pool}__new_20"
        _require(expected_fragment in source_job, "D127 S0 source D92 job K drift")
        packages = row["packages"]
        _require(isinstance(packages, Mapping) and set(packages) == _PACKAGE_NAMES, "D127 S0 four-package closure drift")
        for reference in packages.values():
            _require(isinstance(reference, Mapping) and set(reference) == _PACKAGE_REF_FIELDS, "D127 package reference closure drift")
            _sha(reference["expected_seal_sha256"], "D92 package seal SHA256")
        selected[key] = row
    ordered = tuple(selected[key] for key in ((receiver, k, n) for receiver in S0_RECEIVERS for k, n in S0_K_NEW))
    _require(len(ordered) == 6, "D127 S0 requires exactly six frozen D92 raw rows")
    return ordered


def _strict_support_prefix(
    payload: Mapping[str, np.ndarray], *, registry: tuple[str, ...], active_k: int, source_pool_k: int
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...], dict[str, Any]]:
    """Use D106's strict selector and additionally prove ordered rank prefixes."""

    try:
        support_iq, labels, tokens = _d106_support_rows(
            payload, registered_classes=registry, active_k=active_k
        )
    except Exception as exc:
        raise D127S0PackageAdapterError("D92 strict support/prefix materialization failed") from exc
    ranks = np.asarray(payload["support_rank_within_class"], dtype=np.int64)
    indices = np.asarray(payload["support_class_indices"], dtype=np.int64)
    raw_tokens = np.asarray(payload["support_tokens"]).astype(str)
    _require(len(ranks) == len(indices) == len(raw_tokens), "D92 support rank/token length drift")
    class_records: list[dict[str, Any]] = []
    for class_index in range(len(registry)):
        class_mask = indices == class_index
        available = sorted(ranks[class_mask].tolist())
        _require(available == list(range(source_pool_k)), "D92 ordered source support ranks drift")
        prefix_mask = class_mask & (ranks < active_k)
        prefix_ranks = sorted(ranks[prefix_mask].tolist())
        _require(prefix_ranks == list(range(active_k)), "D92 ordered support prefix drift")
        prefix_tokens = tuple(raw_tokens[prefix_mask].tolist())
        class_records.append(
            {
                "registry_index": class_index,
                "prefix_count": active_k,
                "prefix_token_root_sha256": _opaque_root(prefix_tokens),
                "available_rank_count": source_pool_k,
            }
        )
    _require(tuple(tokens) == tuple(raw_tokens[ranks < active_k].tolist()), "D106 selected-prefix token drift")
    return support_iq, labels, tokens, {"class_prefixes": class_records}


def _materialize_state(
    *,
    raw_row: Mapping[str, Any],
    scene: str,
    state: str,
    qknn_lock: Phase1ZIDStudentTLock,
    loader: PackageLoader,
    package_cache: dict[tuple[tuple[str, str], ...], tuple[Any, Mapping[str, Any], Mapping[str, Any]]],
) -> tuple[D127S0StateInput, dict[str, Any]]:
    packages = raw_row["packages"]
    enrollment_ref = packages[f"{state}_enrollment"]
    apply_ref = packages[f"{state}_apply"]

    def loaded(reference: Mapping[str, Any]):
        key = tuple(sorted((str(name), str(value)) for name, value in reference.items()))
        if key not in package_cache:
            package_cache[key] = loader(reference)
        return package_cache[key]

    support_payloads, support_manifest, _support_audit = loaded(enrollment_ref)
    query_payloads, query_manifest, _query_audit = loaded(apply_ref)
    _require(isinstance(support_payloads, Mapping) and isinstance(query_payloads, Mapping), "D92 sealed package payload drift")
    _require(isinstance(support_manifest, Mapping) and isinstance(query_manifest, Mapping), "D92 sealed package manifest drift")
    try:
        _validate_matched_packages(support_manifest, query_manifest)
    except Exception as exc:
        raise D127S0PackageAdapterError("D92 support/query package pairing drift") from exc
    _require(scene in support_payloads and scene in query_payloads, "D92 S0 scene package missing")
    source_pool_k = int(raw_row["source_pool_k"])
    for manifest in (support_manifest, query_manifest):
        _require(
            manifest.get("receiver") == raw_row["receiver"]
            and manifest.get("seed") == S0_SEED
            and manifest.get("k_shot") == source_pool_k
            and manifest.get("registration_state") == state,
            "D92 S0 package identity drift",
        )
        _sha(manifest.get("package_root_sha256"), "D92 package root SHA256")
    registry = _registry(support_manifest)
    _require(_registry(query_manifest) == registry, "D92 enrollment/apply registry drift")
    active_k = int(raw_row["k_shot"])
    support_iq, support_labels, support_tokens, prefix = _strict_support_prefix(
        support_payloads[scene], registry=registry, active_k=active_k, source_pool_k=source_pool_k
    )
    try:
        query_iq, query_tokens = _d106_query_rows(query_payloads[scene])
    except Exception as exc:
        raise D127S0PackageAdapterError("D92 strict query materialization failed") from exc
    _require(not set(support_tokens).intersection(query_tokens), "D92 support/query opaque IDs overlap")
    support_root = _opaque_root(support_tokens)
    query_root = _opaque_root(query_tokens)
    registry_root = _opaque_root(registry)
    capsule_id = _canonical_sha256(
        {
            "schema": CAPSULE_SCHEMA,
            "source_d92_job_id": raw_row["source_d92_job_id"],
            "state": state,
            "support_seal_sha256": enrollment_ref["expected_seal_sha256"],
            "query_seal_sha256": apply_ref["expected_seal_sha256"],
            "support_package_root_sha256": support_manifest["package_root_sha256"],
            "query_package_root_sha256": query_manifest["package_root_sha256"],
        }
    )
    split_id = _canonical_sha256(
        {
            "schema": SPLIT_SCHEMA,
            "capsule_id": capsule_id,
            "receiver": raw_row["receiver"],
            "seed": S0_SEED,
            "k_shot": active_k,
            "scene": scene,
            "state": state,
            "support_token_root_sha256": support_root,
            "query_token_root_sha256": query_root,
            "registered_class_root_sha256": registry_root,
        }
    )
    row_id = f"d127-s0-rx-{str(raw_row['receiver']).replace('-', '_')}__seed-{S0_SEED}__k-{active_k}__new-20__scene-{scene}"
    row = D127S0Row(
        row_id=row_id,
        receiver_id=str(raw_row["receiver"]),
        k_shot=active_k,
        scene=scene,
        support_iq=torch.from_numpy(support_iq),
        query_iq=torch.from_numpy(query_iq),
        support_labels=support_labels,
        registered_classes=registry,
        opaque_query_ids=query_tokens,
        qknn_lock=qknn_lock,
    )
    core = {
        "schema": STATE_INPUT_SCHEMA,
        "row_id": row_id,
        "state": state,
        "source_d92_job_id": raw_row["source_d92_job_id"],
        "source_pool_k": source_pool_k,
        "capsule_id": capsule_id,
        "split_id": split_id,
        "support_token_root_sha256": support_root,
        "query_token_root_sha256": query_root,
        "registered_class_root_sha256": registry_root,
        "qknn_lock_digest": qknn_lock.lock_digest,
    }
    item = D127S0StateInput(
        state=state,
        row=row,
        source_d92_job_id=str(raw_row["source_d92_job_id"]),
        source_pool_k=source_pool_k,
        capsule_id=capsule_id,
        split_id=split_id,
        support_token_root_sha256=support_root,
        query_token_root_sha256=query_root,
        registered_class_root_sha256=registry_root,
        state_input_receipt_sha256=_canonical_sha256(core),
    )
    prefix_record = {
        "receiver": row.receiver_id,
        "k_shot": active_k,
        "scene": scene,
        "state": state,
        "source_pool_k": source_pool_k,
        "support_token_count": len(support_tokens),
        "support_token_root_sha256": support_root,
        "query_token_count": len(query_tokens),
        "query_token_root_sha256": query_root,
        "registered_class_count": len(registry),
        **prefix,
    }
    return item, prefix_record


def materialize_d127_s0_package_rows(
    *,
    method_lock_path: str | Path,
    expected_method_lock_sha256: str,
    d106_context_path: str | Path,
    expected_d106_context_sha256: str,
    device: torch.device | str = "cpu",
    package_loader: PackageLoader = _load_raw_package,
) -> D127S0PreparedPackageRows:
    """Materialize frozen D92 packages into exactly 18 before and 18 after rows.

    ``package_loader`` exists solely for hermetic local tests.  Production uses
    D106's sealed-bundle loader by default.
    """

    method_lock, method_lock_sha256, qknn_locks = load_d127_s0_method_lock(
        method_lock_path, expected_sha256=expected_method_lock_sha256
    )
    phase1_expected_binding = _phase1_asset_expected_binding(
        method_lock, method_lock_sha256=method_lock_sha256, qknn_locks=qknn_locks
    )
    context, context_sha256 = _read_json_sha(
        d106_context_path, expected_d106_context_sha256, "D106 Target25 context"
    )
    raw_rows = _target_context_rows(context)
    package_cache: dict[tuple[tuple[str, str], ...], tuple[Any, Mapping[str, Any], Mapping[str, Any]]] = {}
    state_inputs: dict[str, list[D127S0StateInput]] = {state: [] for state in S0_STATES}
    prefix_records: list[dict[str, Any]] = []
    runtime_device = torch.device(device)
    for raw_row in raw_rows:
        for scene in S0_SCENES:
            for state in S0_STATES:
                item, record = _materialize_state(
                    raw_row=raw_row,
                    scene=scene,
                    state=state,
                    qknn_lock=qknn_locks[int(raw_row["k_shot"])],
                    loader=package_loader,
                    package_cache=package_cache,
                )
                # Inputs are fixed received-IQ views; moving them does not add K.
                moved = D127S0Row(
                    row_id=item.row.row_id,
                    receiver_id=item.row.receiver_id,
                    k_shot=item.row.k_shot,
                    scene=item.row.scene,
                    support_iq=item.row.support_iq.to(runtime_device),
                    query_iq=item.row.query_iq.to(runtime_device),
                    support_labels=item.row.support_labels,
                    registered_classes=item.row.registered_classes,
                    opaque_query_ids=item.row.opaque_query_ids,
                    qknn_lock=item.row.qknn_lock,
                )
                state_inputs[state].append(
                    D127S0StateInput(
                        state=item.state,
                        row=moved,
                        source_d92_job_id=item.source_d92_job_id,
                        source_pool_k=item.source_pool_k,
                        capsule_id=item.capsule_id,
                        split_id=item.split_id,
                        support_token_root_sha256=item.support_token_root_sha256,
                        query_token_root_sha256=item.query_token_root_sha256,
                        registered_class_root_sha256=item.registered_class_root_sha256,
                        state_input_receipt_sha256=item.state_input_receipt_sha256,
                    )
                )
                if moved.k_shot == 5:
                    prefix_records.append(record)
    before = tuple(sorted(state_inputs["before"], key=lambda item: (item.row.receiver_id, item.row.k_shot, item.row.scene, item.row.row_id)))
    after = tuple(sorted(state_inputs["after"], key=lambda item: (item.row.receiver_id, item.row.k_shot, item.row.scene, item.row.row_id)))
    _require(len(before) == S0_ROW_COUNT and len(after) == S0_ROW_COUNT, "D127 S0 state count drift")
    for left, right in zip(before, after, strict=True):
        _require(left.row.row_id == right.row.row_id, "D127 state pair row-ID drift")
        _require(
            left.row.registered_classes == right.row.registered_classes[: len(left.row.registered_classes)],
            "D127 before registry is not an after-registry prefix",
        )
        _require(left.row.k_shot == right.row.k_shot and left.row.scene == right.row.scene, "D127 paired K/scene drift")
    prefix_receipt: dict[str, Any] = {
        "schema": PREFIX_RECEIPT_SCHEMA,
        "method_lock_sha256": method_lock_sha256,
        "d106_context_sha256": context_sha256,
        "policy": "K5_uses_rank_lt_5_from_same_K10_source_pool",
        "record_count": len(prefix_records),
        "records": prefix_records,
    }
    _require(len(prefix_records) == 18, "D127 K5 prefix receipt coverage drift")
    pairs = D127S0PreparedPackageRows(
        method_lock_sha256=method_lock_sha256,
        checkpoint_sha256=method_lock["checkpoint"]["sha256"],
        phase1_asset_expected_binding=phase1_expected_binding,
        context_sha256=context_sha256,
        qknn_locks=qknn_locks,
        before=before,
        after=after,
        prefix_receipt={},
    ).pair_bindings
    _require(
        all(pair["before_query_is_after_ordered_subset"] for pair in pairs),
        "D127 before query IDs are not an ordered after-query subset",
    )
    for pair in pairs:
        pair["formal_d92_reference"]["d92_retry2_manifest_sha256"] = method_lock["s0_matrix"][
            "d92_retry2_manifest_sha256"
        ]
    prefix_receipt["pair_bindings"] = list(pairs)
    prefix_receipt["receipt_sha256"] = _canonical_sha256(prefix_receipt)
    return D127S0PreparedPackageRows(
        method_lock_sha256=method_lock_sha256,
        checkpoint_sha256=method_lock["checkpoint"]["sha256"],
        phase1_asset_expected_binding=phase1_expected_binding,
        context_sha256=context_sha256,
        qknn_locks=qknn_locks,
        before=before,
        after=after,
        prefix_receipt=prefix_receipt,
    )


def write_d127_s0_prefix_receipt_exclusive(path: str | Path, receipt: Mapping[str, Any]) -> Path:
    """Persist the compact, opaque-token-only K5 prefix receipt once."""

    target = Path(path)
    _require(receipt.get("schema") == PREFIX_RECEIPT_SCHEMA, "prefix receipt schema drift")
    _require(not _forbidden_keys(receipt), "prefix receipt contains forbidden predictor field")
    digest = receipt.get("receipt_sha256")
    payload = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    _require(digest == _canonical_sha256(payload), "prefix receipt digest drift")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as stream:
            stream.write(_canonical_bytes(dict(receipt)) + b"\n")
    except FileExistsError as exc:
        raise D127S0PackageAdapterError("prefix receipt output already exists") from exc
    return target


def _write_json_exclusive(path: str | Path, payload: Mapping[str, Any], *, name: str) -> Path:
    target = Path(path)
    _require(not target.is_symlink(), f"{name} output cannot be a symlink")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as stream:
            stream.write(_canonical_bytes(dict(payload)) + b"\n")
    except FileExistsError as exc:
        raise D127S0PackageAdapterError(f"{name} output already exists") from exc
    return target


def _verify_prefix_receipt(receipt: Mapping[str, Any]) -> None:
    _require(receipt.get("schema") == PREFIX_RECEIPT_SCHEMA, "prefix receipt schema drift")
    _require(not _forbidden_keys(receipt), "prefix receipt contains forbidden field")
    signed = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    _require(receipt.get("receipt_sha256") == _canonical_sha256(signed), "prefix receipt digest drift")
    records = receipt.get("records")
    pairs = receipt.get("pair_bindings")
    _require(isinstance(records, list) and len(records) == 18, "prefix receipt K5 coverage drift")
    _require(isinstance(pairs, list) and len(pairs) == S0_ROW_COUNT, "prefix receipt pair coverage drift")
    for record in records:
        _require(isinstance(record, Mapping), "prefix receipt record type drift")
        _require(
            record.get("k_shot") == 5
            and record.get("source_pool_k") == 10
            and record.get("support_token_count") == record.get("registered_class_count", 0) * 5,
            "prefix receipt K5 count/source-pool drift",
        )
        _sha(record.get("support_token_root_sha256"), "prefix receipt support root")
        _sha(record.get("query_token_root_sha256"), "prefix receipt query root")
        classes = record.get("class_prefixes")
        _require(isinstance(classes, list) and len(classes) == record.get("registered_class_count"), "prefix receipt class coverage drift")
        for item in classes:
            _require(isinstance(item, Mapping) and item.get("prefix_count") == 5, "prefix receipt class K5 drift")
            _sha(item.get("prefix_token_root_sha256"), "prefix receipt class root")
    for pair in pairs:
        _validate_pair_binding(pair)


def _validate_pair_binding(value: Any) -> None:
    _require(isinstance(value, Mapping), "S0 pair binding must be an object")
    _require(
        isinstance(value.get("row_id"), str)
        and value.get("receiver") in S0_RECEIVERS
        and value.get("k_shot") in (1, 5)
        and value.get("scene") in S0_SCENES
        and value.get("before_query_is_after_ordered_subset") is True,
        "S0 pair identity/order binding drift",
    )
    formal = value.get("formal_d92_reference")
    _require(isinstance(formal, Mapping), "S0 formal D92 reference is missing")
    _require(
        isinstance(formal.get("source_d92_job_id"), str)
        and formal.get("pipeline_receipt_required") is True,
        "S0 formal D92 job/pipeline reference drift",
    )
    _sha(formal.get("d92_retry2_manifest_sha256"), "S0 formal D92 manifest SHA256")
    for state in S0_STATES:
        item = value.get(state)
        _require(isinstance(item, Mapping), "S0 paired state binding is missing")
        _require(
            item.get("row_id") == value["row_id"]
            and item.get("receiver") == value["receiver"]
            and item.get("k_shot") == value["k_shot"]
            and item.get("scene") == value["scene"]
            and item.get("state") == state,
            "S0 paired state identity drift",
        )
        for name in (
            "capsule_id",
            "split_id",
            "support_token_root_sha256",
            "query_token_root_sha256",
            "query_token_ordered_sha256",
            "registered_class_root_sha256",
            "qknn_lock_digest",
            "state_input_receipt_sha256",
        ):
            _sha(item.get(name), f"S0 state {name}")
        _require(type(item.get("query_token_count")) is int and item["query_token_count"] > 0, "S0 query count drift")


def build_d127_s0_prepared_plan(prepared: D127S0PreparedPackageRows) -> dict[str, Any]:
    """Return the small serializable plan used by later truth-free workers."""

    _verify_prefix_receipt(prepared.prefix_receipt)
    pairs = prepared.prefix_receipt["pair_bindings"]
    plan: dict[str, Any] = {
        "schema": PREPARED_PLAN_SCHEMA,
        "method_lock_sha256": prepared.method_lock_sha256,
        "checkpoint_sha256": prepared.checkpoint_sha256,
        "phase1_asset_expected_binding": prepared.phase1_asset_expected_binding,
        "d106_context_sha256": prepared.context_sha256,
        "qknn_lock_digests": {str(k): prepared.qknn_locks[k].lock_digest for k in (1, 5)},
        "row_pair_count": S0_ROW_COUNT,
        "state_row_count": S0_ROW_COUNT * len(S0_STATES),
        "truth_loaded": False,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "prefix_receipt_sha256": prepared.prefix_receipt["receipt_sha256"],
        "pair_bindings": pairs,
    }
    plan["prepared_plan_sha256"] = _canonical_sha256(plan)
    return plan


def _validate_prepared_plan(plan: Mapping[str, Any]) -> None:
    _require(plan.get("schema") == PREPARED_PLAN_SCHEMA, "D127 prepared plan schema drift")
    _require(not _forbidden_keys(plan), "D127 prepared plan contains forbidden field")
    signed = {key: value for key, value in plan.items() if key != "prepared_plan_sha256"}
    _require(plan.get("prepared_plan_sha256") == _canonical_sha256(signed), "D127 prepared plan digest drift")
    _sha(plan.get("method_lock_sha256"), "D127 prepared method-lock SHA256")
    _sha(plan.get("checkpoint_sha256"), "D127 prepared checkpoint SHA256")
    _validate_phase1_asset_expected_binding(plan.get("phase1_asset_expected_binding"))
    _sha(plan.get("d106_context_sha256"), "D127 prepared context SHA256")
    _sha(plan.get("prefix_receipt_sha256"), "D127 prepared prefix receipt SHA256")
    _require(
        plan.get("row_pair_count") == S0_ROW_COUNT
        and plan.get("state_row_count") == 36
        and plan.get("truth_loaded") is False
        and plan.get("query_rows_used_for_fit") == 0
        and plan.get("query_state_updates") == 0
        and plan.get("query_selection_count") == 0,
        "D127 prepared plan count/access drift",
    )
    qknn = plan.get("qknn_lock_digests")
    _require(isinstance(qknn, Mapping) and set(qknn) == {"1", "5"}, "D127 prepared qKNN binding drift")
    for value in qknn.values():
        _sha(value, "D127 prepared qKNN lock digest")
    pairs = plan.get("pair_bindings")
    _require(isinstance(pairs, list) and len(pairs) == S0_ROW_COUNT, "D127 prepared pair count drift")
    seen: set[str] = set()
    expected_order: list[tuple[str, int, str, str]] = []
    for pair in pairs:
        _validate_pair_binding(pair)
        _require(pair["row_id"] not in seen, "D127 prepared pair row ID is duplicated")
        seen.add(pair["row_id"])
        expected_order.append((pair["receiver"], pair["k_shot"], pair["scene"], pair["row_id"]))
    _require(expected_order == sorted(expected_order), "D127 prepared pair order drift")


def write_d127_s0_prepared_plan_exclusive(path: str | Path, plan: Mapping[str, Any]) -> Path:
    _validate_prepared_plan(plan)
    return _write_json_exclusive(path, plan, name="D127 prepared plan")


def load_d127_s0_prepared_plan(path: str | Path, *, expected_sha256: str) -> tuple[dict[str, Any], str]:
    plan, digest = _read_json_sha(path, expected_sha256, "D127 prepared plan")
    _validate_prepared_plan(plan)
    return plan, digest


def _assert_prepared_matches_plan(prepared: D127S0PreparedPackageRows, plan: Mapping[str, Any]) -> None:
    _validate_prepared_plan(plan)
    _require(
        plan["method_lock_sha256"] == prepared.method_lock_sha256
        and plan["checkpoint_sha256"] == prepared.checkpoint_sha256
        and plan["phase1_asset_expected_binding"] == prepared.phase1_asset_expected_binding
        and plan["d106_context_sha256"] == prepared.context_sha256
        and plan["prefix_receipt_sha256"] == prepared.prefix_receipt.get("receipt_sha256"),
        "D127 prepared plan/input lineage drift",
    )
    _require(
        plan["qknn_lock_digests"] == {str(k): prepared.qknn_locks[k].lock_digest for k in (1, 5)},
        "D127 prepared plan qKNN lineage drift",
    )
    _require(plan["pair_bindings"] == prepared.prefix_receipt.get("pair_bindings"), "D127 prepared pair binding drift")


def load_d127_s0_candidate_asset(
    *,
    bundle_dir: str | Path,
    expected_manifest_sha256: str,
    candidate_id: str,
    device: torch.device | str,
    prepared_plan: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Load only one decoded candidate from a complete immutable A/B/C bundle."""

    _validate_prepared_plan(prepared_plan)
    _require(candidate_id in entry.CANDIDATE_IDS, "D127 worker candidate is unknown")
    _sha(expected_manifest_sha256, "D127 Phase1 asset manifest SHA256")
    try:
        assets = phase1_release.load_d127_phase1_asset_bundle(bundle_dir, expected_manifest_sha256)
        manifest_path = Path(bundle_dir) / phase1_release.MANIFEST_FILE_NAME
        raw_manifest = manifest_path.read_bytes()
        _require(_sha256_file(manifest_path) == expected_manifest_sha256, "D127 Phase1 manifest reread SHA drift")
        manifest = json.loads(raw_manifest.decode("utf-8"))
        _require(type(manifest) is dict and raw_manifest == _canonical_bytes(manifest), "D127 Phase1 manifest reread canonical drift")
        expected = prepared_plan["phase1_asset_expected_binding"]
        _require(manifest.get("bundle_kind") == "merged_complete", "D127 Phase1 bundle is not merged complete")
        _require(manifest.get("candidate_ids") == list(entry.CANDIDATE_IDS), "D127 Phase1 merged candidate closure drift")
        for name in ("method_lock_sha256", "checkpoint_sha256", "source_binding", "qknn_lock_binding"):
            _require(manifest.get(name) == expected[name], f"D127 Phase1 manifest {name} lineage drift")
        asset = assets[candidate_id].decode(device=torch.device(device))
    except Exception as exc:
        raise D127S0PackageAdapterError("D127 Phase1 asset bundle load/decode failed") from exc
    receipt: dict[str, Any] = {
        "schema": PHASE1_MANIFEST_RECEIPT_SCHEMA,
        "manifest_sha256": expected_manifest_sha256,
        "candidate_id": candidate_id,
        "method_lock_sha256": manifest["method_lock_sha256"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "source_binding": manifest["source_binding"],
        "qknn_lock_binding": manifest["qknn_lock_binding"],
        "episode_manifest_sha256": manifest["episode_manifest_sha256"],
        "episode_contract_sha256": manifest["episode_contract_sha256"],
        "candidate_asset": manifest["candidate_assets"][candidate_id],
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    _validate_phase1_manifest_receipt(receipt, prepared_plan=prepared_plan, candidate_id=candidate_id)
    return asset, receipt


def _validate_phase1_manifest_receipt(
    value: Any, *, prepared_plan: Mapping[str, Any], candidate_id: str
) -> None:
    _require(isinstance(value, Mapping), "D127 Phase1 manifest receipt missing")
    _require(
        set(value) == {
            "schema", "manifest_sha256", "candidate_id", "method_lock_sha256", "checkpoint_sha256",
            "source_binding", "qknn_lock_binding", "episode_manifest_sha256", "episode_contract_sha256",
            "candidate_asset", "receipt_sha256",
        },
        "D127 Phase1 manifest receipt closure drift",
    )
    signed = {key: item for key, item in value.items() if key != "receipt_sha256"}
    _require(value.get("receipt_sha256") == _canonical_sha256(signed), "D127 Phase1 manifest receipt digest drift")
    expected = prepared_plan["phase1_asset_expected_binding"]
    _require(
        value.get("schema") == PHASE1_MANIFEST_RECEIPT_SCHEMA
        and value.get("candidate_id") == candidate_id
        and value.get("method_lock_sha256") == expected["method_lock_sha256"]
        and value.get("checkpoint_sha256") == expected["checkpoint_sha256"]
        and value.get("source_binding") == expected["source_binding"]
        and value.get("qknn_lock_binding") == expected["qknn_lock_binding"],
        "D127 Phase1 manifest receipt lineage drift",
    )
    for name in ("manifest_sha256", "episode_manifest_sha256", "episode_contract_sha256"):
        _sha(value.get(name), f"D127 Phase1 manifest receipt {name}")
    candidate_asset = value.get("candidate_asset")
    _require(
        isinstance(candidate_asset, Mapping)
        and candidate_asset.get("candidate_id") == candidate_id
        and candidate_asset.get("persistent_fp32_sidecar") is False,
        "D127 Phase1 manifest candidate-asset receipt drift",
    )


def _validate_local_worker_state(
    value: Any, *, candidate_id: str, state: str, bindings: Sequence[Mapping[str, Any]]
) -> None:
    _require(isinstance(value, Mapping), "D127 candidate worker state is missing")
    _require(
        value.get("schema") == entry.LOCAL_WORKER_SCHEMA
        and value.get("candidate_id") == candidate_id
        and value.get("truth_loaded") is False
        and value.get("row_count") == S0_ROW_COUNT
        and value.get("rows_complete") is True
        and value.get("query_rows_used_for_fit") == 0
        and value.get("query_state_updates") == 0
        and value.get("query_selection_count") == 0
        and value.get("phase2_optimizer_steps") == 0,
        "D127 candidate worker state access/count drift",
    )
    signed = {key: item for key, item in value.items() if key != "prediction_sha256"}
    _require(value.get("prediction_sha256") == entry._sha256(signed), "D127 candidate worker state digest drift")
    rows = value.get("rows")
    _require(isinstance(rows, list) and len(rows) == S0_ROW_COUNT, "D127 candidate worker rows incomplete")
    for row, pair in zip(rows, bindings, strict=True):
        expected = pair[state]
        _require(isinstance(row, Mapping), "D127 candidate worker row type drift")
        _require(
            row.get("row_id") == pair["row_id"]
            and row.get("receiver_id") == pair["receiver"]
            and row.get("k_shot") == pair["k_shot"]
            and row.get("scene") == pair["scene"],
            "D127 candidate worker row identity drift",
        )
        query_ids = row.get("opaque_query_ids")
        _require(isinstance(query_ids, list) and len(query_ids) == expected["query_token_count"], "D127 candidate worker query count drift")
        _require(
            _canonical_sha256(query_ids) == expected["query_token_ordered_sha256"]
            and _opaque_root(query_ids) == expected["query_token_root_sha256"],
            "D127 candidate worker query-root drift",
        )
        arms = row.get("arms")
        _require(isinstance(arms, Mapping) and len(arms) == len(_ARM_IDS) and set(arms) == set(_ARM_IDS), "D127 candidate worker arm closure drift")
        registry: tuple[str, ...] | None = None
        for arm_id in _ARM_IDS:
            arm = arms[arm_id]
            _require(isinstance(arm, Mapping), "D127 candidate worker arm type drift")
            classes = arm.get("classes")
            predictions = arm.get("predictions")
            _require(isinstance(classes, list) and isinstance(predictions, list), "D127 candidate worker arm payload drift")
            _require(len(predictions) == len(query_ids), "D127 candidate worker prediction count drift")
            if registry is None:
                registry = tuple(str(item) for item in classes)
            _require(tuple(str(item) for item in classes) == registry, "D127 candidate worker registry drift")
            _require(all(str(item) in registry for item in predictions), "D127 candidate worker prediction outside registry")
        _require(registry is not None and _opaque_root(registry) == expected["registered_class_root_sha256"], "D127 candidate worker registry-root drift")
    resource = value.get("resource")
    _require(isinstance(resource, Mapping), "D127 candidate worker resource receipt missing")
    for name in ("total_id_backbone_forwards", "total_query_rows", "total_adapter_macs_support_plus_query"):
        _require(type(resource.get(name)) is int and resource[name] >= 0, "D127 candidate worker resource drift")


def run_d127_s0_candidate_worker_pair(
    *,
    model: Any,
    candidate_id: str,
    asset: Any,
    prepared: D127S0PreparedPackageRows,
    prepared_plan: Mapping[str, Any],
    phase1_asset_manifest_sha256: str,
    phase1_manifest_receipt: Mapping[str, Any],
    checkpoint_sha256: str,
) -> dict[str, Any]:
    """Run one frozen candidate on the two S0 states, without opening truth."""

    _assert_prepared_matches_plan(prepared, prepared_plan)
    _require(candidate_id in entry.CANDIDATE_IDS, "D127 worker candidate is unknown")
    _sha(phase1_asset_manifest_sha256, "D127 Phase1 asset manifest SHA256")
    _validate_phase1_manifest_receipt(
        phase1_manifest_receipt, prepared_plan=prepared_plan, candidate_id=candidate_id
    )
    _require(
        phase1_manifest_receipt["manifest_sha256"] == phase1_asset_manifest_sha256,
        "D127 worker Phase1 manifest receipt/hash drift",
    )
    _sha(checkpoint_sha256, "D127 checkpoint SHA256")
    _require(
        checkpoint_sha256 == prepared_plan["checkpoint_sha256"],
        "D127 worker checkpoint binding drift",
    )
    before = entry._run_d127_s0_candidate_worker(
        model=model, candidate_id=candidate_id, asset=asset, rows=tuple(item.row for item in prepared.before)
    )
    after = entry._run_d127_s0_candidate_worker(
        model=model, candidate_id=candidate_id, asset=asset, rows=tuple(item.row for item in prepared.after)
    )
    pair_bindings = prepared_plan["pair_bindings"]
    _validate_local_worker_state(before, candidate_id=candidate_id, state="before", bindings=pair_bindings)
    _validate_local_worker_state(after, candidate_id=candidate_id, state="after", bindings=pair_bindings)
    payload: dict[str, Any] = {
        "schema": CANDIDATE_WORKER_SCHEMA,
        "candidate_id": candidate_id,
        "evaluation_scope": "TARGET_DEVELOPMENT_S0_18_CANDIDATE_WORKER",
        "truth_loaded": False,
        "method_lock_sha256": prepared_plan["method_lock_sha256"],
        "d106_context_sha256": prepared_plan["d106_context_sha256"],
        "phase1_asset_manifest_sha256": phase1_asset_manifest_sha256,
        "phase1_manifest_receipt": phase1_manifest_receipt,
        "checkpoint_sha256": checkpoint_sha256,
        "prepared_plan_sha256": prepared_plan["prepared_plan_sha256"],
        "prefix_receipt_sha256": prepared_plan["prefix_receipt_sha256"],
        "row_pair_count": S0_ROW_COUNT,
        "state_row_count": 36,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "pair_bindings": pair_bindings,
        "states": {"before": before, "after": after},
        "physical_execution": {
            "candidate_worker_mode": True,
            "common_base_forward_reused_across_candidates": False,
            "physical_base_forwards_are_repeated_per_candidate": True,
            "total_id_backbone_forwards": int(before["resource"]["total_id_backbone_forwards"]) + int(after["resource"]["total_id_backbone_forwards"]),
        },
    }
    payload["candidate_worker_sha256"] = _canonical_sha256(payload)
    return payload


def _validate_candidate_worker_pair(value: Mapping[str, Any], *, plan: Mapping[str, Any]) -> None:
    _validate_prepared_plan(plan)
    _require(value.get("schema") == CANDIDATE_WORKER_SCHEMA, "D127 worker-pair schema drift")
    _require(not _forbidden_keys(value), "D127 worker-pair contains forbidden field")
    signed = {key: item for key, item in value.items() if key != "candidate_worker_sha256"}
    _require(value.get("candidate_worker_sha256") == _canonical_sha256(signed), "D127 worker-pair digest drift")
    _require(
        value.get("candidate_id") in entry.CANDIDATE_IDS
        and value.get("truth_loaded") is False
        and value.get("method_lock_sha256") == plan["method_lock_sha256"]
        and value.get("checkpoint_sha256") == plan["checkpoint_sha256"]
        and value.get("d106_context_sha256") == plan["d106_context_sha256"]
        and value.get("prepared_plan_sha256") == plan["prepared_plan_sha256"]
        and value.get("prefix_receipt_sha256") == plan["prefix_receipt_sha256"]
        and value.get("row_pair_count") == S0_ROW_COUNT
        and value.get("state_row_count") == 36
        and value.get("query_rows_used_for_fit") == 0
        and value.get("query_state_updates") == 0
        and value.get("query_selection_count") == 0
        and value.get("pair_bindings") == plan["pair_bindings"],
        "D127 worker-pair lineage/access drift",
    )
    _sha(value.get("phase1_asset_manifest_sha256"), "D127 worker asset manifest SHA256")
    _validate_phase1_manifest_receipt(
        value.get("phase1_manifest_receipt"), prepared_plan=plan, candidate_id=str(value["candidate_id"])
    )
    _require(
        value["phase1_manifest_receipt"]["manifest_sha256"] == value["phase1_asset_manifest_sha256"],
        "D127 worker asset manifest receipt/hash drift",
    )
    _sha(value.get("checkpoint_sha256"), "D127 worker checkpoint SHA256")
    states = value.get("states")
    _require(isinstance(states, Mapping) and len(states) == len(S0_STATES) and set(states) == set(S0_STATES), "D127 worker-pair state closure drift")
    for state in S0_STATES:
        _validate_local_worker_state(
            states[state], candidate_id=str(value["candidate_id"]), state=state, bindings=plan["pair_bindings"]
        )
    physical = value.get("physical_execution")
    _require(
        isinstance(physical, Mapping)
        and physical.get("candidate_worker_mode") is True
        and physical.get("common_base_forward_reused_across_candidates") is False
        and physical.get("physical_base_forwards_are_repeated_per_candidate") is True
        and type(physical.get("total_id_backbone_forwards")) is int,
        "D127 worker-pair physical resource disclosure drift",
    )


def _build_scorer_pair_manifest(prepared_plan: Mapping[str, Any]) -> dict[str, Any]:
    """Create the sole scorer binding plane from the frozen prepared plan."""

    _validate_prepared_plan(prepared_plan)
    manifest: dict[str, Any] = {
        "schema": SCORER_PAIR_MANIFEST_SCHEMA,
        "method_lock_sha256": prepared_plan["method_lock_sha256"],
        "checkpoint_sha256": prepared_plan["checkpoint_sha256"],
        "d106_context_sha256": prepared_plan["d106_context_sha256"],
        "qknn_lock_digests": prepared_plan["qknn_lock_digests"],
        "prefix_receipt_sha256": prepared_plan["prefix_receipt_sha256"],
        "prepared_plan_sha256": prepared_plan["prepared_plan_sha256"],
        "row_pair_count": S0_ROW_COUNT,
        "pair_bindings": prepared_plan["pair_bindings"],
    }
    manifest["pair_manifest_sha256"] = _canonical_sha256(manifest)
    return manifest


def _validate_scorer_pair_manifest(value: Any, *, prepared_plan: Mapping[str, Any]) -> None:
    _require(isinstance(value, Mapping), "D127 scorer pair manifest is missing")
    _require(value.get("schema") == SCORER_PAIR_MANIFEST_SCHEMA, "D127 scorer pair manifest schema drift")
    _require(not _forbidden_keys(value), "D127 scorer pair manifest contains forbidden field")
    signed = {key: item for key, item in value.items() if key != "pair_manifest_sha256"}
    _require(value.get("pair_manifest_sha256") == _canonical_sha256(signed), "D127 scorer pair manifest digest drift")
    expected = _build_scorer_pair_manifest(prepared_plan)
    _require(dict(value) == expected, "D127 scorer pair manifest lineage drift")


def write_d127_s0_candidate_worker_exclusive(path: str | Path, payload: Mapping[str, Any]) -> Path:
    # Structural validation is repeated by merge with the frozen plan; writing
    # here only protects the truth-free file and immutable digest boundary.
    _require(payload.get("schema") == CANDIDATE_WORKER_SCHEMA and payload.get("truth_loaded") is False, "invalid D127 worker-pair payload")
    _require(not _forbidden_keys(payload), "D127 worker-pair contains forbidden field")
    signed = {key: value for key, value in payload.items() if key != "candidate_worker_sha256"}
    _require(payload.get("candidate_worker_sha256") == _canonical_sha256(signed), "D127 worker-pair digest drift")
    return _write_json_exclusive(path, payload, name="D127 candidate worker")


def load_d127_s0_candidate_worker(path: str | Path, *, expected_sha256: str) -> tuple[dict[str, Any], str]:
    return _read_json_sha(path, expected_sha256, "D127 candidate worker")


def merge_d127_s0_candidate_workers(
    *, prepared_plan: Mapping[str, Any], workers: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Merge A/B/C workers, retaining exactly one verified public common pair."""

    _validate_prepared_plan(prepared_plan)
    frozen = tuple(workers)
    _require(len(frozen) == len(entry.CANDIDATE_IDS), "D127 merge requires exactly three candidate workers")
    for worker in frozen:
        _validate_candidate_worker_pair(worker, plan=prepared_plan)
    by_candidate = {str(worker["candidate_id"]): worker for worker in frozen}
    _require(set(by_candidate) == set(entry.CANDIDATE_IDS), "D127 merge candidate coverage drift")
    shared_asset_hashes = {worker["phase1_asset_manifest_sha256"] for worker in frozen}
    shared_checkpoint_hashes = {worker["checkpoint_sha256"] for worker in frozen}
    _require(len(shared_asset_hashes) == 1 and len(shared_checkpoint_hashes) == 1, "D127 merge asset/checkpoint lineage drift")
    state_payloads: dict[str, dict[str, Any]] = {}
    worker_resources: dict[str, Any] = {}
    total_physical_forwards = 0
    phase1_manifest_receipts: dict[str, Any] = {}
    for candidate_id in entry.CANDIDATE_IDS:
        worker = by_candidate[candidate_id]
        phase1_manifest_receipts[candidate_id] = worker["phase1_manifest_receipt"]
        worker_resources[candidate_id] = {}
        for state in S0_STATES:
            resource = worker["states"][state]["resource"]
            worker_resources[candidate_id][state] = resource
            total_physical_forwards += int(resource["total_id_backbone_forwards"])
    for state in S0_STATES:
        rows: list[dict[str, Any]] = []
        for index, pair in enumerate(prepared_plan["pair_bindings"]):
            candidate_rows = {candidate: by_candidate[candidate]["states"][state]["rows"][index] for candidate in entry.CANDIDATE_IDS}
            owner = candidate_rows[entry.CANDIDATE_IDS[0]]
            common: dict[str, Any] = {}
            for arm_id in _COMMON_ARM_IDS:
                arm_hash = _canonical_sha256(owner["arms"][arm_id])
                for candidate_id in entry.CANDIDATE_IDS[1:]:
                    _require(
                        _canonical_sha256(candidate_rows[candidate_id]["arms"][arm_id]) == arm_hash,
                        "D127 candidate worker common-arm value/hash drift",
                    )
                common[arm_id] = owner["arms"][arm_id]
            candidates: dict[str, Any] = {}
            for candidate_id in entry.CANDIDATE_IDS:
                worker_row = candidate_rows[candidate_id]
                candidates[candidate_id] = {
                    "arms": {arm_id: worker_row["arms"][arm_id] for arm_id in _CANDIDATE_ARM_IDS},
                    "joint_receipt": worker_row["joint_receipt"],
                    "hook_receipt": worker_row["hook_receipt"],
                    "da_resource": worker_row["da_resource"],
                }
            row = {
                "row_id": pair["row_id"],
                "receiver": pair["receiver"],
                "k_shot": pair["k_shot"],
                "scene": pair["scene"],
                "state": state,
                "opaque_query_ids": owner["opaque_query_ids"],
                "common_arms": common,
                "candidates": candidates,
            }
            row["row_sha256"] = _canonical_sha256(row)
            rows.append(row)
        state_payload = {"state": state, "row_count": S0_ROW_COUNT, "rows": rows}
        state_payload["state_sha256"] = _canonical_sha256(state_payload)
        state_payloads[state] = state_payload
    payload: dict[str, Any] = {
        "schema": PAIRED_PREDICTION_SCHEMA,
        "evaluation_scope": "TARGET_DEVELOPMENT_S0_18",
        "truth_loaded": False,
        "method_lock_sha256": prepared_plan["method_lock_sha256"],
        "qknn_lock_digests": prepared_plan["qknn_lock_digests"],
        "d106_context_sha256": prepared_plan["d106_context_sha256"],
        "phase1_asset_manifest_sha256": next(iter(shared_asset_hashes)),
        "phase1_manifest_receipts": phase1_manifest_receipts,
        "checkpoint_sha256": next(iter(shared_checkpoint_hashes)),
        "prepared_plan_sha256": prepared_plan["prepared_plan_sha256"],
        "prefix_receipt_sha256": prepared_plan["prefix_receipt_sha256"],
        "pair_manifest": _build_scorer_pair_manifest(prepared_plan),
        "candidate_ids": list(entry.CANDIDATE_IDS),
        "row_pair_count": S0_ROW_COUNT,
        "state_row_count": 36,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "pair_bindings": prepared_plan["pair_bindings"],
        "states": state_payloads,
        "physical_execution": {
            "candidate_workers": len(entry.CANDIDATE_IDS),
            "common_base_forward_reused_across_candidates": False,
            "physical_base_forwards_are_repeated_per_candidate": True,
            "total_id_backbone_forwards": total_physical_forwards,
            "resource_by_candidate_and_state": worker_resources,
        },
    }
    payload["paired_prediction_sha256"] = _canonical_sha256(payload)
    validate_d127_s0_prediction_pairs(payload, prepared_plan=prepared_plan)
    return payload


def validate_d127_s0_prediction_pairs(
    prediction: Mapping[str, Any], *, prepared_plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify closure before a scorer is allowed to open any truth sidecar."""

    _validate_prepared_plan(prepared_plan)
    _require(prediction.get("schema") == PAIRED_PREDICTION_SCHEMA, "D127 paired prediction schema drift")
    _require(not _forbidden_keys(prediction), "D127 paired prediction contains forbidden field")
    signed = {key: value for key, value in prediction.items() if key != "paired_prediction_sha256"}
    _require(prediction.get("paired_prediction_sha256") == _canonical_sha256(signed), "D127 paired prediction digest drift")
    _require(
        prediction.get("truth_loaded") is False
        and prediction.get("method_lock_sha256") == prepared_plan["method_lock_sha256"]
        and prediction.get("checkpoint_sha256") == prepared_plan["checkpoint_sha256"]
        and prediction.get("qknn_lock_digests") == prepared_plan["qknn_lock_digests"]
        and prediction.get("d106_context_sha256") == prepared_plan["d106_context_sha256"]
        and prediction.get("prepared_plan_sha256") == prepared_plan["prepared_plan_sha256"]
        and prediction.get("prefix_receipt_sha256") == prepared_plan["prefix_receipt_sha256"]
        and tuple(prediction.get("candidate_ids", ())) == entry.CANDIDATE_IDS
        and prediction.get("row_pair_count") == S0_ROW_COUNT
        and prediction.get("state_row_count") == 36
        and prediction.get("query_rows_used_for_fit") == 0
        and prediction.get("query_state_updates") == 0
        and prediction.get("query_selection_count") == 0
        and prediction.get("pair_bindings") == prepared_plan["pair_bindings"],
        "D127 paired prediction lineage/access closure drift",
    )
    _validate_scorer_pair_manifest(prediction.get("pair_manifest"), prepared_plan=prepared_plan)
    _sha(prediction.get("phase1_asset_manifest_sha256"), "D127 paired asset manifest SHA256")
    receipts = prediction.get("phase1_manifest_receipts")
    _require(
        isinstance(receipts, Mapping)
        and len(receipts) == len(entry.CANDIDATE_IDS)
        and set(receipts) == set(entry.CANDIDATE_IDS),
        "D127 paired Phase1 manifest receipt closure drift",
    )
    for candidate_id in entry.CANDIDATE_IDS:
        _validate_phase1_manifest_receipt(
            receipts[candidate_id], prepared_plan=prepared_plan, candidate_id=candidate_id
        )
        _require(
            receipts[candidate_id]["manifest_sha256"] == prediction["phase1_asset_manifest_sha256"],
            "D127 paired Phase1 manifest receipt/hash drift",
        )
    _sha(prediction.get("checkpoint_sha256"), "D127 paired checkpoint SHA256")
    states = prediction.get("states")
    _require(isinstance(states, Mapping) and len(states) == len(S0_STATES) and set(states) == set(S0_STATES), "D127 paired prediction state closure drift")
    query_by_state: dict[str, dict[str, tuple[str, ...]]] = {state: {} for state in S0_STATES}
    for state in S0_STATES:
        surface = states[state]
        _require(isinstance(surface, Mapping) and surface.get("state") == state and surface.get("row_count") == S0_ROW_COUNT, "D127 paired state count drift")
        signed_state = {key: value for key, value in surface.items() if key != "state_sha256"}
        _require(surface.get("state_sha256") == _canonical_sha256(signed_state), "D127 paired state digest drift")
        rows = surface.get("rows")
        _require(isinstance(rows, list) and len(rows) == S0_ROW_COUNT, "D127 paired state row closure drift")
        for row, pair in zip(rows, prepared_plan["pair_bindings"], strict=True):
            signed_row = {key: value for key, value in row.items() if key != "row_sha256"}
            _require(row.get("row_sha256") == _canonical_sha256(signed_row), "D127 paired row digest drift")
            expected = pair[state]
            _require(
                row.get("row_id") == pair["row_id"]
                and row.get("receiver") == pair["receiver"]
                and row.get("k_shot") == pair["k_shot"]
                and row.get("scene") == pair["scene"]
                and row.get("state") == state,
                "D127 paired row identity drift",
            )
            query_ids = row.get("opaque_query_ids")
            _require(isinstance(query_ids, list) and len(query_ids) == expected["query_token_count"], "D127 paired query count drift")
            _require(
                _canonical_sha256(query_ids) == expected["query_token_ordered_sha256"]
                and _opaque_root(query_ids) == expected["query_token_root_sha256"],
                "D127 paired query-root drift",
            )
            query_by_state[state][pair["row_id"]] = tuple(query_ids)
            common = row.get("common_arms")
            candidates = row.get("candidates")
            _require(isinstance(common, Mapping) and len(common) == len(_COMMON_ARM_IDS) and set(common) == set(_COMMON_ARM_IDS), "D127 paired common-arm closure drift")
            _require(isinstance(candidates, Mapping) and len(candidates) == len(entry.CANDIDATE_IDS) and set(candidates) == set(entry.CANDIDATE_IDS), "D127 paired candidate closure drift")
            registry: tuple[str, ...] | None = None
            for arm_id in _COMMON_ARM_IDS:
                arm = common[arm_id]
                _require(isinstance(arm, Mapping), "D127 paired common arm type drift")
                classes = tuple(str(item) for item in arm.get("classes", ()))
                predictions = arm.get("predictions")
                _require(bool(classes) and isinstance(predictions, list) and len(predictions) == len(query_ids), "D127 paired common arm payload drift")
                registry = classes if registry is None else registry
                _require(classes == registry and all(str(item) in registry for item in predictions), "D127 paired common-arm registry drift")
            _require(registry is not None and _opaque_root(registry) == expected["registered_class_root_sha256"], "D127 paired registry root drift")
            for candidate_id in entry.CANDIDATE_IDS:
                candidate = candidates[candidate_id]
                _require(isinstance(candidate, Mapping), "D127 paired candidate payload type drift")
                arms = candidate.get("arms")
                _require(isinstance(arms, Mapping) and len(arms) == len(_CANDIDATE_ARM_IDS) and set(arms) == set(_CANDIDATE_ARM_IDS), "D127 paired adapted-arm closure drift")
                for arm_id in _CANDIDATE_ARM_IDS:
                    arm = arms[arm_id]
                    _require(tuple(str(item) for item in arm.get("classes", ())) == registry, "D127 paired adapted registry drift")
                    predictions = arm.get("predictions")
                    _require(isinstance(predictions, list) and len(predictions) == len(query_ids) and all(str(item) in registry for item in predictions), "D127 paired adapted prediction drift")
    for pair in prepared_plan["pair_bindings"]:
        _require(
            _ordered_subset(query_by_state["before"][pair["row_id"]], query_by_state["after"][pair["row_id"]]),
            "D127 paired before-query ordered-subset drift",
        )
    physical = prediction.get("physical_execution")
    _require(
        isinstance(physical, Mapping)
        and physical.get("candidate_workers") == 3
        and physical.get("common_base_forward_reused_across_candidates") is False
        and physical.get("physical_base_forwards_are_repeated_per_candidate") is True
        and type(physical.get("total_id_backbone_forwards")) is int,
        "D127 paired physical resource disclosure drift",
    )
    return {
        "status": "D127_S0_TRUTH_FREE_PREDICTION_PAIRS_VALIDATED",
        "row_pair_count": S0_ROW_COUNT,
        "state_row_count": 36,
        "truth_loaded": False,
        "method_lock_sha256": prediction["method_lock_sha256"],
        "prepared_plan_sha256": prediction["prepared_plan_sha256"],
        "paired_prediction_sha256": prediction["paired_prediction_sha256"],
    }


def write_d127_s0_paired_prediction_exclusive(path: str | Path, payload: Mapping[str, Any], *, prepared_plan: Mapping[str, Any]) -> Path:
    validate_d127_s0_prediction_pairs(payload, prepared_plan=prepared_plan)
    return _write_json_exclusive(path, payload, name="D127 paired prediction")


def load_d127_s0_paired_prediction(
    path: str | Path, *, expected_sha256: str, prepared_plan: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
    payload, digest = _read_json_sha(path, expected_sha256, "D127 paired prediction")
    validate_d127_s0_prediction_pairs(payload, prepared_plan=prepared_plan)
    return payload, digest


__all__ = [
    "CANDIDATE_WORKER_SCHEMA",
    "D127S0PackageAdapterError",
    "D127S0PreparedPackageRows",
    "D127S0StateInput",
    "PAIRED_PREDICTION_SCHEMA",
    "PHASE1_MANIFEST_RECEIPT_SCHEMA",
    "PREFIX_RECEIPT_SCHEMA",
    "PREPARED_PLAN_SCHEMA",
    "SCORER_PAIR_MANIFEST_SCHEMA",
    "S0_K_NEW",
    "S0_RECEIVERS",
    "S0_ROW_COUNT",
    "S0_SCENES",
    "S0_SEED",
    "build_d127_s0_prepared_plan",
    "load_d127_s0_candidate_asset",
    "load_d127_s0_candidate_worker",
    "load_d127_s0_method_lock",
    "load_d127_s0_paired_prediction",
    "load_d127_s0_prepared_plan",
    "materialize_d127_s0_package_rows",
    "merge_d127_s0_candidate_workers",
    "run_d127_s0_candidate_worker_pair",
    "validate_d127_s0_prediction_pairs",
    "write_d127_s0_candidate_worker_exclusive",
    "write_d127_s0_paired_prediction_exclusive",
    "write_d127_s0_prepared_plan_exclusive",
    "write_d127_s0_prefix_receipt_exclusive",
]
