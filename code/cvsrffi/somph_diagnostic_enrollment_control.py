"""Support-only SOMP-H packages for non-formal development controls.

This controller-side module deliberately stops before query materialization.
It accepts only a verified diagnostic lineage receipt and publishes matched
before/after enrollment-only bundles whose physically reachable support is
exactly the declared K-shot count.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi import somph_offline_target_package as producer
from cvsrffi import somph_predictor_bundle as bundle
from cvsrffi.somph_formal_matrix import (
    FORMAL_RECEIVERS,
    OLD_TX_IDS,
    NEW_TX_IDS,
)
from cvsrffi.stage2_predictor_bundle import sha256_file


SCHEMA = "cvs.phase2.somph_diagnostic_enrollment_control.v1"
SOURCE_RECEIVERS = {
    "1-1",
    "1-19",
    "14-7",
    "18-2",
    "19-2",
    "2-1",
    "2-19",
}


class SomphDiagnosticEnrollmentControlError(ValueError):
    """Raised when a development control could reach query or formal claims."""


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _validate_control(
    *,
    receiver: str,
    seed: int,
    k_shot: int,
    old_tx_ids: Sequence[str],
    new_tx_ids: Sequence[str],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    if receiver in SOURCE_RECEIVERS or receiver in FORMAL_RECEIVERS:
        raise SomphDiagnosticEnrollmentControlError(
            "development control receiver must be non-source and non-formal"
        )
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise SomphDiagnosticEnrollmentControlError(
            "development control seed must be a nonnegative integer"
        )
    if k_shot != 10:
        raise SomphDiagnosticEnrollmentControlError(
            "D8b development control is frozen at K=10"
        )
    if tuple(old_tx_ids) != tuple(OLD_TX_IDS):
        raise SomphDiagnosticEnrollmentControlError(
            "development control old TX tuple drift"
        )
    if tuple(new_tx_ids) != tuple(NEW_TX_IDS[:5]):
        raise SomphDiagnosticEnrollmentControlError(
            "development control frozen new5 tuple drift"
        )
    old_labels = [("target_old", value) for value in old_tx_ids]
    new_labels = [("target_new", value) for value in new_tx_ids]
    return old_labels, new_labels


def _support_selection(
    arrays_by_scenario: Mapping[str, Mapping[str, np.ndarray]],
    *,
    receiver: str,
    seed: int,
    k_shot: int,
    labels: Sequence[tuple[str, str]],
) -> tuple[
    dict[str, dict[tuple[str, str], tuple[list[int], list[int]]]],
    dict[str, str],
]:
    selected: dict[
        str, dict[tuple[str, str], tuple[list[int], list[int]]]
    ] = {}
    roots: dict[str, str] = {}
    observed: set[str] = set()
    for scenario in bundle.FORMAL_LEO_WEAK_SCENARIOS:
        arrays = arrays_by_scenario[scenario]
        roles = np.asarray(arrays["dataset_role"]).astype(str)
        tx_ids = np.asarray(arrays["tx_ids"]).astype(str)
        rx_ids = np.asarray(arrays["rx_ids"]).astype(str)
        sample_ids = np.asarray(arrays["sample_ids"]).astype(str)
        current: dict[
            tuple[str, str], tuple[list[int], list[int]]
        ] = {}
        physical: list[str] = []
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
            if len(support) != k_shot:
                raise SomphDiagnosticEnrollmentControlError(
                    f"insufficient support rows: {scenario}/{role}/{tx_label}"
                )
            current[(role, tx_label)] = (support, [])
            physical.extend(str(sample_ids[index]) for index in support)
        if len(physical) != len(set(physical)):
            raise SomphDiagnosticEnrollmentControlError(
                f"duplicate support physical root within {scenario}"
            )
        if observed.intersection(physical):
            raise SomphDiagnosticEnrollmentControlError(
                "support physical roots overlap across scenarios"
            )
        observed.update(physical)
        selected[scenario] = current
        roots[scenario] = _canonical_sha256(sorted(physical))
    return selected, roots


def _assert_exact_k_reachability(
    root: Path,
    *,
    registered_class_count: int,
    k_shot: int,
) -> dict[str, int]:
    rows_by_scenario: dict[str, int] = {}
    expected_pairs = [
        (class_index, rank)
        for class_index in range(registered_class_count)
        for rank in range(k_shot)
    ]
    for scenario in bundle.FORMAL_LEO_WEAK_SCENARIOS:
        path = root / f"support_{scenario}.npz"
        with np.load(path, allow_pickle=False) as archive:
            labels = np.asarray(archive["support_class_indices"])
            ranks = np.asarray(archive["support_rank_within_class"])
            embedded = json.loads(str(np.asarray(archive["manifest_json"]).item()))
        actual_pairs = list(zip(labels.tolist(), ranks.tolist()))
        if actual_pairs != expected_pairs:
            raise SomphDiagnosticEnrollmentControlError(
                "manifest K and physically reachable per-class support differ: "
                f"{scenario}"
            )
        if embedded.get("support_pool_max_k") != k_shot:
            raise SomphDiagnosticEnrollmentControlError(
                f"embedded reachable support K drift: {scenario}"
            )
        rows_by_scenario[scenario] = len(actual_pairs)
    return rows_by_scenario


def build_diagnostic_enrollment_control(
    *,
    cache_set_manifest_path: str | Path,
    lineage_receipt_path: str | Path,
    lineage_seal_path: str | Path,
    expected_lineage_receipt_sha256: str,
    expected_lineage_seal_sha256: str,
    sealed_feature_runtime_path: str | Path,
    method_lock_path: str | Path,
    output_root: str | Path,
    receiver: str,
    seed: int,
    k_shot: int,
    old_tx_ids: Sequence[str] = OLD_TX_IDS,
    new_tx_ids: Sequence[str] = NEW_TX_IDS[:5],
    token_secret: bytes | None = None,
) -> dict[str, Any]:
    """Publish before/after enrollment bundles without creating query data."""

    old_labels, new_labels = _validate_control(
        receiver=receiver,
        seed=seed,
        k_shot=k_shot,
        old_tx_ids=old_tx_ids,
        new_tx_ids=new_tx_ids,
    )
    if token_secret is None:
        token_secret = os.urandom(32)
    if not isinstance(token_secret, bytes) or len(token_secret) < 32:
        raise SomphDiagnosticEnrollmentControlError(
            "token_secret must contain at least 256 bits"
        )
    context = producer.load_verified_lineage_context_from_receipt_seal(
        cache_set_manifest_path=cache_set_manifest_path,
        lineage_receipt_path=lineage_receipt_path,
        lineage_seal_path=lineage_seal_path,
        expected_lineage_receipt_sha256=expected_lineage_receipt_sha256,
        expected_lineage_seal_sha256=expected_lineage_seal_sha256,
    )
    if (
        context.get("external_authority_lock_verified") is not False
        or context.get("formal_launch_authority") is not False
    ):
        raise SomphDiagnosticEnrollmentControlError(
            "development control requires diagnostic-only lineage"
        )
    arrays_by_scenario = producer._load_scenario_caches(
        cache_set_manifest_path,
        cache_set=context["cache_set"],
        receipt=context["receipt"],
    )
    all_labels = old_labels + new_labels
    selected, support_roots = _support_selection(
        arrays_by_scenario,
        receiver=receiver,
        seed=seed,
        k_shot=k_shot,
        labels=all_labels,
    )
    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite enrollment control root: {output}"
        )
    output.mkdir(parents=True, exist_ok=False)
    predictor_root = output / "predictor"
    seals_root = output / "seals"
    predictor_root.mkdir()
    seals_root.mkdir()
    runtime_sha = sha256_file(sealed_feature_runtime_path)
    method_sha = sha256_file(method_lock_path)
    states: dict[str, Any] = {}
    for state, labels in (
        ("before", old_labels),
        ("after", all_labels),
    ):
        stage = "stage2b" if state == "before" else "stage2c"
        root = predictor_root / state / bundle.ENROLLMENT_ONLY
        copied_runtime, copied_method = producer._prepare_root(
            root,
            sealed_feature_runtime_path=sealed_feature_runtime_path,
            method_lock_path=method_lock_path,
        )
        if copied_runtime != runtime_sha or copied_method != method_sha:
            raise SomphDiagnosticEnrollmentControlError(
                "runtime or method lock copy digest drift"
            )
        registry, truth_rows = producer._write_profile_payloads(
            root,
            profile=bundle.ENROLLMENT_ONLY,
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
            support_pool_k=k_shot,
        )
        if truth_rows:
            raise SomphDiagnosticEnrollmentControlError(
                "support-only writer unexpectedly produced truth rows"
            )
        overlay_sha = sha256_file(root / "overlay_provenance.json")
        support_rows_by_scenario = _assert_exact_k_reachability(
            root,
            registered_class_count=len(registry),
            k_shot=k_shot,
        )
        seal_path = seals_root / f"{state}_enrollment.seal.json"
        _manifest_path, _seal_path, manifest, _seal = (
            bundle.write_somph_predictor_bundle(
                root,
                profile=bundle.ENROLLMENT_ONLY,
                stage=stage,
                registration_state=state,
                receiver=receiver,
                seed=seed,
                k_shot=k_shot,
                registered_classes=registry,
                expected_method_lock_sha256=method_sha,
                expected_overlay_provenance_sha256=overlay_sha,
                detached_seal_path=seal_path,
                support_pool_max_k=k_shot,
            )
        )
        states[state] = {
            "stage": stage,
            "registered_class_count": len(registry),
            "enrollment_package_root": str(root),
            "enrollment_package_root_sha256": manifest[
                "package_root_sha256"
            ],
            "enrollment_package_seal": str(seal_path),
            "enrollment_package_seal_sha256": sha256_file(seal_path),
            "support_rows_by_scenario": support_rows_by_scenario,
            "physical_support_per_class_per_scenario": k_shot,
        }
    forbidden = [
        str(path.relative_to(output))
        for path in output.rglob("*")
        if path.is_file()
        and any(
            token in path.name.lower()
            for token in ("query", "truth", "score", "prediction")
        )
    ]
    if forbidden:
        raise SomphDiagnosticEnrollmentControlError(
            f"support-only output contains forbidden members: {forbidden}"
        )
    receipt = {
        "schema": SCHEMA,
        "status": "SEALED_ENROLLMENT_ONLY_CONTROL_READY",
        "classification": "development_control_sensitivity_only",
        "formal_confirmation_evidence_allowed": False,
        "formal_launch_authority": False,
        "receiver": receiver,
        "seed": seed,
        "k_shot": k_shot,
        "reachable_support_pool_max_k": k_shot,
        "nested_k_prefixes": [1, 5, 10],
        "old_tx_ids": list(old_tx_ids),
        "new_tx_ids": list(new_tx_ids),
        "states": states,
        "support_physical_ids_sha256_by_scenario": support_roots,
        "query_payload_created": False,
        "query_truth_opened": False,
        "prediction_created": False,
        "scorer_invoked": False,
        "token_secret_persisted": False,
        "lineage_receipt_sha256": context["lineage_receipt_sha256"],
        "lineage_seal_sha256": context["lineage_seal_sha256"],
        "cache_set_manifest_sha256": context[
            "cache_set_manifest_sha256"
        ],
        "sealed_feature_runtime_sha256": runtime_sha,
        "method_lock_sha256": method_sha,
    }
    receipt_path = output / "support_only_build_receipt.json"
    producer._write_json_new(receipt_path, receipt)
    return {
        **receipt,
        "support_only_build_receipt": str(receipt_path),
        "support_only_build_receipt_sha256": sha256_file(receipt_path),
    }


__all__ = [
    "SCHEMA",
    "SomphDiagnosticEnrollmentControlError",
    "build_diagnostic_enrollment_control",
]
