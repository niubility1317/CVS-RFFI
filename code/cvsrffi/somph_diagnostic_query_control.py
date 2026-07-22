"""Seal D8b query-only packages bound to one explicitly selected support commit.

Predictor roots contain only opaque query payloads and protocol manifests.
Query truth, roles, TX labels, and selection details remain under the
physically separate scorer root. This module never runs prediction or scoring.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi import somph_diagnostic_enrollment_control as enrollment
from cvsrffi import somph_offline_target_package as producer
from cvsrffi import somph_predictor_bundle as bundle
from cvsrffi.somph_formal_matrix import NEW_TX_IDS, OLD_TX_IDS
from cvsrffi.stage2_metric_scorer import TRUTH_SIDECAR_SCHEMA
from cvsrffi.stage2_predictor_bundle import sha256_file


SCHEMA = "cvs.phase2.somph_diagnostic_query_control.v1"
QUERY_MANIFEST_SCHEMA = "cvs.phase2.somph_diagnostic_query_package.v1"
QUERY_SEAL_SCHEMA = "cvs.phase2.somph_diagnostic_query_package_seal.v1"
QUERY_PER_CLASS = 20


class SomphDiagnosticQueryControlError(ValueError):
    """Raised when query construction could leak truth or overlap support."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _select_support_query(
    arrays_by_scenario: Mapping[str, Mapping[str, np.ndarray]],
    *,
    receiver: str,
    seed: int,
    k_shot: int,
    query_per_class: int,
    labels: Sequence[tuple[str, str]],
) -> tuple[
    dict[str, dict[tuple[str, str], tuple[list[int], list[int]]]],
    dict[str, Any],
]:
    selected: dict[
        str, dict[tuple[str, str], tuple[list[int], list[int]]]
    ] = {}
    audit: dict[str, Any] = {}
    observed: set[str] = set()
    for scenario in bundle.FORMAL_LEO_WEAK_SCENARIOS:
        arrays = arrays_by_scenario[scenario]
        roles = np.asarray(arrays["dataset_role"]).astype(str)
        tx_ids = np.asarray(arrays["tx_ids"]).astype(str)
        rx_ids = np.asarray(arrays["rx_ids"]).astype(str)
        sample_ids = np.asarray(arrays["sample_ids"]).astype(str)
        scenario_selected: dict[
            tuple[str, str], tuple[list[int], list[int]]
        ] = {}
        rows: list[dict[str, Any]] = []
        scenario_all: set[str] = set()
        for role, tx_label in labels:
            candidates = np.flatnonzero(
                (roles == role)
                & (tx_ids == tx_label)
                & (rx_ids == receiver)
            )
            ordered = producer._selection_order(
                sample_ids,
                candidates,
                receiver=receiver,
                seed=seed,
                role=role,
                tx_label=tx_label,
            )
            support = ordered[:k_shot]
            query = ordered[k_shot : k_shot + query_per_class]
            if len(support) != k_shot or len(query) != query_per_class:
                raise SomphDiagnosticQueryControlError(
                    f"insufficient support/query rows: {scenario}/{role}/{tx_label}"
                )
            support_ids = [str(sample_ids[index]) for index in support]
            query_ids = [str(sample_ids[index]) for index in query]
            if set(support_ids).intersection(query_ids):
                raise SomphDiagnosticQueryControlError(
                    f"support/query overlap: {scenario}/{role}/{tx_label}"
                )
            scenario_all.update(support_ids)
            scenario_all.update(query_ids)
            scenario_selected[(role, tx_label)] = (support, query)
            rows.append(
                {
                    "dataset_role": role,
                    "tx_label": tx_label,
                    "support_physical_sample_ids": support_ids,
                    "query_physical_sample_ids": query_ids,
                    "support_root_sha256": _canonical_sha256(support_ids),
                    "query_root_sha256": _canonical_sha256(query_ids),
                }
            )
        if len(scenario_all) != len(labels) * (k_shot + query_per_class):
            raise SomphDiagnosticQueryControlError(
                f"duplicate selected physical sample within {scenario}"
            )
        if observed.intersection(scenario_all):
            raise SomphDiagnosticQueryControlError(
                "selected physical samples overlap across scenarios"
            )
        observed.update(scenario_all)
        selected[scenario] = scenario_selected
        audit[scenario] = rows
    return selected, audit


def _descriptor(path: Path, root: Path) -> dict[str, Any]:
    relative = str(path.relative_to(root)).replace("\\", "/")
    item = {
        "relative_path": relative,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            item["npz_members"] = list(archive.files)
    else:
        item["npz_members"] = []
    return item


def _seal_query_root(
    root: Path,
    *,
    seal_path: Path,
    stage: str,
    registration_state: str,
    receiver: str,
    seed: int,
    k_shot: int,
    query_per_class: int,
    registered_class_count: int,
    support_candidate_id: str,
    support_candidate_commit_sha256: str,
    support_candidate_state_sha256_by_scenario: Mapping[
        str, Mapping[str, str]
    ],
    strict_enrollment_package_root_sha256: str,
    strict_enrollment_package_seal_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = {
        "sealed_feature_runtime.pt",
        "method_lock.json",
        "overlay_provenance.json",
        *{
            f"query_{scenario}.npz"
            for scenario in bundle.FORMAL_LEO_WEAK_SCENARIOS
        },
    }
    actual = {
        str(path.relative_to(root)).replace("\\", "/")
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual != expected:
        raise SomphDiagnosticQueryControlError(
            f"query predictor allowlist drift: {sorted(actual ^ expected)}"
        )
    members = [
        _descriptor(root / relative, root)
        for relative in sorted(expected)
    ]
    package_root_sha256 = _canonical_sha256(members)
    manifest = {
        "schema": QUERY_MANIFEST_SCHEMA,
        "profile": "query_only_no_truth",
        "stage": stage,
        "registration_state": registration_state,
        "receiver": receiver,
        "seed": seed,
        "k_shot": k_shot,
        "query_per_registered_class_per_scenario": query_per_class,
        "registered_class_count": registered_class_count,
        "target_channel_scenarios": list(bundle.FORMAL_LEO_WEAK_SCENARIOS),
        "support_candidate_id": support_candidate_id,
        "support_candidate_commit_sha256": (
            support_candidate_commit_sha256
        ),
        "support_candidate_state_sha256_by_scenario": {
            key: dict(value)
            for key, value in (
                support_candidate_state_sha256_by_scenario.items()
            )
        },
        "strict_enrollment_package_root_sha256": (
            strict_enrollment_package_root_sha256
        ),
        "strict_enrollment_package_seal_sha256": (
            strict_enrollment_package_seal_sha256
        ),
        "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "phase2_query_decision_policy": (
            "per_sample_all_registered_classes"
        ),
        "phase2_query_role_oracle_access": False,
        "phase2_query_true_batch_class_count_access": False,
        "phase2_query_class_quota_access": False,
        "phase2_query_batch_global_assignment": False,
        "query_truth_access": False,
        "query_label_fit_access": False,
        "members": members,
        "package_root_sha256": package_root_sha256,
    }
    manifest_path = root / "query_package_manifest.json"
    manifest_bytes = _canonical_bytes(manifest) + b"\n"
    with manifest_path.open("xb") as handle:
        handle.write(manifest_bytes)
    seal = {
        "schema": QUERY_SEAL_SCHEMA,
        "manifest_relative_path": manifest_path.name,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "manifest_size_bytes": len(manifest_bytes),
        "package_root_sha256": package_root_sha256,
    }
    seal_path.parent.mkdir(parents=True, exist_ok=True)
    with seal_path.open("xb") as handle:
        handle.write(_canonical_bytes(seal) + b"\n")
    return manifest, seal


def _verify_query_root(
    root: Path,
    *,
    seal_path: Path,
) -> dict[str, Any]:
    manifest_path = root / "query_package_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if (
        seal.get("schema") != QUERY_SEAL_SCHEMA
        or seal.get("manifest_sha256")
        != hashlib.sha256(manifest_bytes).hexdigest()
        or seal.get("manifest_size_bytes") != len(manifest_bytes)
        or manifest.get("package_root_sha256")
        != seal.get("package_root_sha256")
    ):
        raise SomphDiagnosticQueryControlError("query manifest/seal drift")
    expected_paths = {
        item["relative_path"] for item in manifest["members"]
    } | {"query_package_manifest.json"}
    actual_paths = {
        str(path.relative_to(root)).replace("\\", "/")
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual_paths != expected_paths:
        raise SomphDiagnosticQueryControlError(
            "query package exact member allowlist drift"
        )
    rows_by_scenario: dict[str, int] = {}
    tokens: set[str] = set()
    for item in manifest["members"]:
        path = root / item["relative_path"]
        if (
            sha256_file(path) != item["sha256"]
            or path.stat().st_size != item["size_bytes"]
        ):
            raise SomphDiagnosticQueryControlError(
                f"query member digest drift: {item['relative_path']}"
            )
        if not item["relative_path"].startswith("query_"):
            continue
        with np.load(path, allow_pickle=False) as archive:
            forbidden = [
                key
                for key in archive.files
                if any(
                    token in key.lower()
                    for token in ("truth", "role", "quota", "label", "tx")
                )
            ]
            if forbidden:
                raise SomphDiagnosticQueryControlError(
                    f"query predictor contains truth-like keys: {forbidden}"
                )
            current = np.asarray(archive["query_tokens"]).astype(str).tolist()
            scenario = str(np.asarray(archive["manifest_json"]).item())
            embedded = json.loads(scenario)
            scenario = embedded["scenario"]
        if tokens.intersection(current):
            raise SomphDiagnosticQueryControlError(
                "query token reuse across scenarios"
            )
        tokens.update(current)
        rows_by_scenario[scenario] = len(current)
    return {
        "manifest": manifest,
        "seal_sha256": sha256_file(seal_path),
        "query_rows_by_scenario": rows_by_scenario,
        "query_token_count": len(tokens),
        "truth_like_npz_members": [],
    }


def build_diagnostic_query_control(
    *,
    cache_set_manifest_path: str | Path,
    lineage_receipt_path: str | Path,
    lineage_seal_path: str | Path,
    expected_lineage_receipt_sha256: str,
    expected_lineage_seal_sha256: str,
    strict_enrollment_receipt_path: str | Path,
    support_candidate_id: str,
    support_candidate_commit_path: str | Path,
    support_candidate_state_root: str | Path,
    expected_support_commit_status: str,
    sealed_feature_runtime_path: str | Path,
    method_lock_path: str | Path,
    output_root: str | Path,
    receiver: str,
    seed: int,
    k_shot: int,
    query_per_class: int = QUERY_PER_CLASS,
    old_tx_ids: Sequence[str] = OLD_TX_IDS,
    new_tx_ids: Sequence[str] = NEW_TX_IDS[:5],
    token_secret: bytes | None = None,
) -> dict[str, Any]:
    """Build and seal query-only before/after packages; do not predict."""

    old_labels, new_labels = enrollment._validate_control(
        receiver=receiver,
        seed=seed,
        k_shot=k_shot,
        old_tx_ids=old_tx_ids,
        new_tx_ids=new_tx_ids,
    )
    if query_per_class != QUERY_PER_CLASS:
        raise SomphDiagnosticQueryControlError("D8b query count is frozen at Q20")
    if token_secret is None:
        token_secret = os.urandom(32)
    if not isinstance(token_secret, bytes) or len(token_secret) < 32:
        raise SomphDiagnosticQueryControlError(
            "token_secret must contain at least 256 bits"
        )
    context = producer.load_verified_lineage_context_from_receipt_seal(
        cache_set_manifest_path=cache_set_manifest_path,
        lineage_receipt_path=lineage_receipt_path,
        lineage_seal_path=lineage_seal_path,
        expected_lineage_receipt_sha256=expected_lineage_receipt_sha256,
        expected_lineage_seal_sha256=expected_lineage_seal_sha256,
    )
    arrays_by_scenario = producer._load_scenario_caches(
        cache_set_manifest_path,
        cache_set=context["cache_set"],
        receipt=context["receipt"],
    )
    all_labels = old_labels + new_labels
    selected, selection_rows = _select_support_query(
        arrays_by_scenario,
        receiver=receiver,
        seed=seed,
        k_shot=k_shot,
        query_per_class=query_per_class,
        labels=all_labels,
    )
    strict_receipt_path = Path(strict_enrollment_receipt_path)
    strict_receipt = json.loads(strict_receipt_path.read_text(encoding="utf-8"))
    if (
        strict_receipt.get("k_shot") != k_shot
        or strict_receipt.get("reachable_support_pool_max_k") != k_shot
    ):
        raise SomphDiagnosticQueryControlError(
            "strict enrollment receipt is not exact K10"
        )
    if not support_candidate_id:
        raise SomphDiagnosticQueryControlError(
            "support_candidate_id must be nonempty"
        )
    candidate_commit_path = Path(support_candidate_commit_path)
    candidate_commit = json.loads(
        candidate_commit_path.read_text(encoding="utf-8")
    )
    if (
        candidate_commit.get("status") != expected_support_commit_status
        or candidate_commit.get("query_package_opened") is not False
        or candidate_commit.get("query_truth_opened") is not False
        or candidate_commit.get("scorer_opened") is not False
        or not isinstance(candidate_commit.get("state_sha256"), dict)
    ):
        raise SomphDiagnosticQueryControlError(
            "selected support candidate commit state drift"
        )
    candidate_commit_sha = sha256_file(candidate_commit_path)
    candidate_root = Path(support_candidate_state_root)
    candidate_state_bindings: dict[
        str, dict[str, dict[str, str]]
    ] = {
        "before": {},
        "after": {},
    }
    for state in ("before", "after"):
        labels = old_labels if state == "before" else all_labels
        for scenario in bundle.FORMAL_LEO_WEAK_SCENARIOS:
            key = f"{scenario}:{state}:k10"
            expected = candidate_commit["state_sha256"][key]
            state_path = (
                candidate_root / f"state_{scenario}_{state}_k10.json"
            )
            if sha256_file(state_path) != expected["metadata_sha256"]:
                raise SomphDiagnosticQueryControlError(
                    f"support candidate state metadata digest drift: {key}"
                )
            candidate_state = json.loads(
                state_path.read_text(encoding="utf-8")
            )
            candidate_support_hashes = {
                row["parent_received_iq_sha256"]
                for row in candidate_state["support_lineage"]
            }
            cache_hashes = np.asarray(
                arrays_by_scenario[scenario]["post_channel_iq_sha256"]
            ).astype(str)
            reproduced = {
                str(cache_hashes[index])
                for label in labels
                for index in selected[scenario][label][0]
            }
            if reproduced != candidate_support_hashes:
                raise SomphDiagnosticQueryControlError(
                    "reproduced strict support does not bind selected "
                    f"candidate state: {key}"
                )
            candidate_state_bindings[state][scenario] = dict(expected)

    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite query control root: {output}")
    predictor_root = output / "predictor"
    scorer_root = output / "scorer"
    seals_root = output / "seals"
    predictor_root.mkdir(parents=True)
    scorer_root.mkdir()
    seals_root.mkdir()
    state_results: dict[str, Any] = {}
    truth_by_state: dict[str, list[dict[str, Any]]] = {}
    for state, labels in (("before", old_labels), ("after", all_labels)):
        stage = "stage2b" if state == "before" else "stage2c"
        root = predictor_root / state / "query_only"
        producer._prepare_root(
            root,
            sealed_feature_runtime_path=sealed_feature_runtime_path,
            method_lock_path=method_lock_path,
        )
        _registry, truth_rows = producer._write_profile_payloads(
            root,
            profile=bundle.APPLY_ONLY,
            registration_state=state,
            receiver=receiver,
            seed=seed,
            labels=list(labels),
            selected_by_scenario=selected,
            arrays_by_scenario=arrays_by_scenario,
            secret=token_secret,
            lineage_receipt_sha256=context["lineage_receipt_sha256"],
            cache_sha256_by_scenario=context["cache_set"][
                "cache_sha256_by_scenario"
            ],
        )
        seal_path = seals_root / f"{state}_query.seal.json"
        manifest, _seal = _seal_query_root(
            root,
            seal_path=seal_path,
            stage=stage,
            registration_state=state,
            receiver=receiver,
            seed=seed,
            k_shot=k_shot,
            query_per_class=query_per_class,
            registered_class_count=len(labels),
            support_candidate_id=support_candidate_id,
            support_candidate_commit_sha256=candidate_commit_sha,
            support_candidate_state_sha256_by_scenario=(
                candidate_state_bindings[state]
            ),
            strict_enrollment_package_root_sha256=strict_receipt["states"][
                state
            ]["enrollment_package_root_sha256"],
            strict_enrollment_package_seal_sha256=strict_receipt["states"][
                state
            ]["enrollment_package_seal_sha256"],
        )
        verification = _verify_query_root(root, seal_path=seal_path)
        truth_path = scorer_root / f"{state}_truth_sidecar.json"
        producer._write_json_new(
            truth_path,
            {
                "schema": TRUTH_SIDECAR_SCHEMA,
                "stage": stage,
                "receiver": receiver,
                "seed": seed,
                "rows": truth_rows,
            },
        )
        truth_by_state[state] = truth_rows
        state_results[state] = {
            "stage": stage,
            "query_package_root": str(root),
            "query_package_root_sha256": manifest["package_root_sha256"],
            "query_package_seal": str(seal_path),
            "query_package_seal_sha256": verification["seal_sha256"],
            "query_rows_by_scenario": verification[
                "query_rows_by_scenario"
            ],
            "truth_sidecar": str(truth_path),
            "truth_sidecar_sha256": sha256_file(truth_path),
        }
    before_old = {
        row["query_token"]: row["physical_sample_id"]
        for row in truth_by_state["before"]
    }
    after_old = {
        row["query_token"]: row["physical_sample_id"]
        for row in truth_by_state["after"]
        if row["evaluation_role"] == "target_old"
    }
    if before_old != after_old:
        raise SomphDiagnosticQueryControlError(
            "before/after old query mapping drift"
        )
    after_new = [
        row
        for row in truth_by_state["after"]
        if row["evaluation_role"] == "target_new"
    ]
    if not after_new:
        raise SomphDiagnosticQueryControlError("after query lacks new classes")
    selection_audit = {
        "schema": "cvs.phase2.d8b_query_selection_audit.v1",
        "status": "PASS_QUERY20_SUPPORT_DISJOINT",
        "receiver": receiver,
        "seed": seed,
        "k_shot": k_shot,
        "query_per_class": query_per_class,
        "support_candidate_id": support_candidate_id,
        "support_candidate_commit_sha256": candidate_commit_sha,
        "support_candidate_commit_status": candidate_commit["status"],
        "support_candidate_lineage_reproduction": (
            "PASS_ALL_STATES_SCENARIOS"
        ),
        "selection_by_scenario": selection_rows,
        "before_after_old_query_exact_reuse": "PASS",
        "after_new_query_append_only": "PASS",
        "support_query_physical_intersection_count": 0,
        "cross_scenario_selected_physical_intersection_count": 0,
        "predictor_contains_truth_role_quota": False,
        "prediction_created": False,
        "scorer_invoked": False,
    }
    selection_path = scorer_root / "query_selection_audit.json"
    producer._write_json_new(selection_path, selection_audit)
    receipt = {
        "schema": SCHEMA,
        "status": "SEALED_QUERY_ONLY_READY_NO_PREDICTION",
        "classification": "development_control_sensitivity_only",
        "receiver": receiver,
        "seed": seed,
        "k_shot": k_shot,
        "query_per_class": query_per_class,
        "support_candidate_id": support_candidate_id,
        "support_candidate_commit": str(candidate_commit_path),
        "support_candidate_commit_sha256": candidate_commit_sha,
        "expected_support_commit_status": expected_support_commit_status,
        "strict_enrollment_receipt_sha256": sha256_file(
            strict_receipt_path
        ),
        "states": state_results,
        "query_selection_audit": str(selection_path),
        "query_selection_audit_sha256": sha256_file(selection_path),
        "predictor_scorer_roots_physically_distinct": True,
        "predictor_truth_role_quota_access": False,
        "prediction_created": False,
        "scorer_invoked": False,
        "token_secret_persisted": False,
        "formal_confirmation_evidence_allowed": False,
        "formal_launch_authority": False,
    }
    receipt_path = output / "query_only_build_receipt.json"
    producer._write_json_new(receipt_path, receipt)
    return {
        **receipt,
        "query_only_build_receipt": str(receipt_path),
        "query_only_build_receipt_sha256": sha256_file(receipt_path),
    }


__all__ = [
    "SCHEMA",
    "SomphDiagnosticQueryControlError",
    "build_diagnostic_query_control",
]
