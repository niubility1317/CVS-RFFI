#!/usr/bin/env python
"""Build and sign path-free SOMP-H before/after runtime authorizations offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi import somph_cache_build_matrix as cache_matrix  # noqa: E402
from cvsrffi import somph_lineage_authority as authority  # noqa: E402
from cvsrffi import somph_predictor_bundle as predictor  # noqa: E402
from cvsrffi import somph_runtime_trust as runtime_trust  # noqa: E402
from cvsrffi.leo_weak_cache import (  # noqa: E402
    FORMAL_LEO_WEAK_SCENARIOS,
    load_verified_leo_weak_cache_set,
    post_channel_iq_sha256,
)
from scripts import sign_cvs_somph_authority_lock as lock_signer  # noqa: E402


AUTHORIZATION_NAMES = {
    "before": "before_formal_policy_authorization.v2.json",
    "after": "after_formal_policy_authorization.v2.json",
}
ENVELOPE_NAMES = {
    "before": "before_signed_policy_authorization_envelope.v2.json",
    "after": "after_signed_policy_authorization_envelope.v2.json",
}
_FORBIDDEN_OUTPUT_FRAGMENTS = (
    "path",
    "raw",
    "clean",
    "build_spec",
    "cache_build",
    "dataset_member",
    "dataset_selector",
    "loader",
)


class SomphRuntimeAuthorizationSigningError(RuntimeError):
    """Raised before publication when the offline v2 trust bridge is incomplete."""


def _json_line(payload: Mapping[str, Any]) -> bytes:
    return authority.canonical_json_bytes(dict(payload)) + b"\n"


def _member_sha(commit: Mapping[str, Any], name: str) -> str:
    rows = commit.get("members")
    if not isinstance(rows, list):
        raise SomphRuntimeAuthorizationSigningError(
            "authority commit member list missing"
        )
    matches = [row for row in rows if isinstance(row, dict) and row.get("name") == name]
    if len(matches) != 1:
        raise SomphRuntimeAuthorizationSigningError(
            f"authority commit member binding missing: {name}"
        )
    try:
        return authority._require_sha256(matches[0].get("sha256"), field=name)
    except authority.SomphLineageAuthorityError as exc:
        raise SomphRuntimeAuthorizationSigningError(str(exc)) from exc


def _read_actual_formal_manifest(
    path: str | Path,
    *,
    expected_sha256: str,
    lock: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    try:
        manifest, _raw, digest, _size = authority._read_external_json(
            path, context="actual SOMP-H 30-cell cache manifest"
        )
        if digest != expected_sha256:
            raise SomphRuntimeAuthorizationSigningError(
                "actual 30-cell manifest is not the committed manifest"
            )
        cache_matrix.validate_cache_build_manifest(
            manifest, manifest_root=Path(path).resolve().parent
        )
    except (
        authority.SomphLineageAuthorityError,
        cache_matrix.SomphCacheBuildMatrixError,
    ) as exc:
        raise SomphRuntimeAuthorizationSigningError(
            "actual 30-cell manifest exact validation failed"
        ) from exc

    cell_id = f"rx_{str(lock['receiver']).replace('-', '_')}_seed_{lock['seed']}"
    cells = manifest.get("cells")
    matches = [
        row
        for row in cells
        if isinstance(row, dict) and row.get("cell_id") == cell_id
    ] if isinstance(cells, list) else []
    if len(matches) != 1:
        raise SomphRuntimeAuthorizationSigningError(
            "actual manifest corresponding authority cell missing"
        )
    cell = matches[0]
    if (
        cell.get("receiver") != lock.get("receiver")
        or cell.get("seed") != lock.get("seed")
        or cell.get("cache_scope") != lock.get("cache_scope")
        or cell.get("spec_file_sha256")
        != lock.get("build_spec", {}).get("file_sha256")
        or cell.get("spec_canonical_sha256")
        != lock.get("build_spec", {}).get("canonical_sha256")
    ):
        raise SomphRuntimeAuthorizationSigningError(
            "actual manifest cell/verified authority lock binding drift"
        )
    return manifest, digest


def _runtime_code_closure() -> tuple[list[dict[str, str]], str]:
    """Hash the exact files imported by the production predictor process."""

    imported = {
        "somph_predictor_bundle.py": Path(predictor.__file__),
        "somph_runtime_trust.py": Path(runtime_trust.__file__),
        "stage2_predictor_bundle.py": Path(predictor.stage2_bundle_module.__file__),
    }
    if tuple(imported) != predictor.CODE_CLOSURE_LOGICAL_MEMBERS:
        raise SomphRuntimeAuthorizationSigningError(
            "actual imported runtime code closure order drift"
        )
    members: list[dict[str, str]] = []
    for logical_name, imported_path in imported.items():
        if imported_path.is_symlink():
            raise SomphRuntimeAuthorizationSigningError(
                f"runtime code closure member is a symlink: {logical_name}"
            )
        candidate = imported_path.resolve(strict=True)
        if not candidate.is_file():
            raise SomphRuntimeAuthorizationSigningError(
                f"runtime code closure member missing: {logical_name}"
            )
        try:
            raw, digest, _size = authority._read_external_bytes(
                candidate, context=f"runtime code closure:{logical_name}"
            )
        except authority.SomphLineageAuthorityError as exc:
            raise SomphRuntimeAuthorizationSigningError(str(exc)) from exc
        if not raw:
            raise SomphRuntimeAuthorizationSigningError(
                f"runtime code closure member empty: {logical_name}"
            )
        members.append({"logical_name": logical_name, "sha256": digest})
    return members, authority.sha256_bytes(authority.canonical_json_bytes(members))


def _read_policy(path: str | Path) -> tuple[dict[str, Any], str]:
    try:
        policy, _raw, digest, _size = authority._read_external_json(
            path, context="actual SOMP-H formal execution policy v2"
        )
        predictor._validate_formal_policy(policy)
    except (authority.SomphLineageAuthorityError, predictor.PredictorPackageError) as exc:
        raise SomphRuntimeAuthorizationSigningError(str(exc)) from exc
    return policy, digest


def _preflight_enrollment_package(
    package_root: str | Path,
    detached_seal_path: str | Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    str,
    dict[str, dict[str, np.ndarray]],
]:
    try:
        _seal_raw, seal_sha, _seal_size = authority._read_external_bytes(
            detached_seal_path, context="SOMP-H detached package seal"
        )
        manifest, seal, _audit, provenance = predictor._preflight(
            package_root,
            detached_seal_path=detached_seal_path,
            expected_seal_sha256=seal_sha,
            inspect_iq_members=False,
            load_npz_control_members=False,
        )
    except (authority.SomphLineageAuthorityError, predictor.PredictorPackageError) as exc:
        raise SomphRuntimeAuthorizationSigningError(str(exc)) from exc
    if manifest.get("profile") != predictor.ENROLLMENT_ONLY:
        raise SomphRuntimeAuthorizationSigningError(
            "runtime authorization bridge accepts enrollment-only packages"
        )
    root = Path(package_root).resolve(strict=True)
    descriptors = {item["kind"]: item for item in manifest["members"]}
    payloads: dict[str, dict[str, np.ndarray]] = {}
    try:
        for scenario in FORMAL_LEO_WEAK_SCENARIOS:
            arrays, embedded = predictor._materialize_iq(
                root, descriptors[f"support:{scenario}"]
            )
            predictor._validate_support_payload(
                arrays,
                embedded,
                manifest=manifest,
                scenario=scenario,
                provenance=provenance[scenario],
            )
            payloads[scenario] = arrays
        predictor._actual_materialized_roots(payloads, manifest)
    except (KeyError, predictor.PredictorPackageError) as exc:
        raise SomphRuntimeAuthorizationSigningError(str(exc)) from exc
    return manifest, seal, provenance, seal_sha, payloads


def _require_package_pair(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> list[str]:
    common = (
        "receiver",
        "seed",
        "k_shot",
        "phase1_checkpoint_sha256",
        "feature_runtime_sha256",
        "method_lock_sha256",
    )
    if any(before.get(key) != after.get(key) for key in common):
        raise SomphRuntimeAuthorizationSigningError(
            "before/after package row identity drift"
        )
    if (
        before.get("registration_state") != "before"
        or before.get("stage") != "stage2b"
        or before.get("registered_class_count") != len(predictor.OLD_TX_IDS)
        or after.get("registration_state") != "after"
        or after.get("stage") != "stage2c"
    ):
        raise SomphRuntimeAuthorizationSigningError(
            "before/after stage-registration binding drift"
        )
    new_count = int(after.get("registered_class_count", 0)) - len(predictor.OLD_TX_IDS)
    if new_count not in predictor.FORMAL_NEW_CLASS_COUNTS:
        raise SomphRuntimeAuthorizationSigningError(
            "after package new-class count is not formal"
        )
    before_registry = before.get("registered_classes")
    after_registry = after.get("registered_classes")
    if (
        not isinstance(before_registry, list)
        or not isinstance(after_registry, list)
        or after_registry[: len(before_registry)] != before_registry
    ):
        raise SomphRuntimeAuthorizationSigningError(
            "after package does not preserve the old registered-class prefix"
        )
    return list(predictor.NEW_TX_IDS[:new_count])


def _require_package_authority_source_binding(
    provenance: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    lock: Mapping[str, Any],
    attestation: Mapping[str, Any],
) -> None:
    for scenario in predictor.FORMAL_LEO_WEAK_SCENARIOS:
        rows = provenance.get(scenario)
        if not isinstance(rows, dict) or not rows:
            raise SomphRuntimeAuthorizationSigningError(
                "sealed package provenance scenario missing"
            )
        expected_cache = lock["cache_sha256_by_scenario"][scenario]
        expected_receipt = attestation["structural_receipt_sha256"]
        if any(
            row.get("source_leo_cache_sha256") != expected_cache
            or row.get("source_leo_provenance_sha256") != expected_receipt
            for row in rows.values()
        ):
            raise SomphRuntimeAuthorizationSigningError(
                "sealed package provenance is detached from verified authority"
            )


def _verified_cache_manifest_path(
    verified_cache_root: str | Path, lock: Mapping[str, Any]
) -> Path:
    candidate = Path(verified_cache_root)
    if candidate.is_symlink():
        raise SomphRuntimeAuthorizationSigningError(
            "verified cache root must not be a symlink"
        )
    if candidate.is_dir():
        locked = lock.get("cache_set_manifest", {}).get("path")
        if not isinstance(locked, str) or not locked:
            raise SomphRuntimeAuthorizationSigningError(
                "verified authority cache-set manifest identity missing"
            )
        candidate = candidate / Path(locked).name
    resolved = candidate.resolve(strict=True)
    if candidate.is_symlink() or not resolved.is_file():
        raise SomphRuntimeAuthorizationSigningError(
            "verified cache root must resolve to one regular cache-set manifest"
        )
    return resolved


def _load_authority_bound_cache_set(
    verified_cache_root: str | Path,
    *,
    lock: Mapping[str, Any],
    attestation: Mapping[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    supplied_root = Path(verified_cache_root)
    mirror_root = supplied_root.resolve(strict=True) if supplied_root.is_dir() else None
    manifest_path = _verified_cache_manifest_path(verified_cache_root, lock)
    try:
        manifest, _manifest_raw, _manifest_file_sha, _manifest_size = (
            authority._read_external_json(
                manifest_path, context="verified cache-set mirror manifest"
            )
        )
        _raw, manifest_sha, _size = authority._read_external_bytes(
            manifest_path, context="verified cache-set mirror manifest"
        )
        load_path = manifest_path
        temporary: tempfile.TemporaryDirectory[str] | None = None
        if mirror_root is not None:
            relocated = dict(manifest)
            declared = manifest.get("cache_npz_by_scenario")
            if not isinstance(declared, dict):
                raise ValueError("verified cache-set scenario map missing")
            relocated_map: dict[str, str] = {}
            observed_files: set[Path] = set()
            for scenario in FORMAL_LEO_WEAK_SCENARIOS:
                raw = Path(str(declared.get(scenario, "")))
                candidate = (
                    mirror_root / raw.name if raw.is_absolute() else mirror_root / raw
                )
                resolved = candidate.resolve(strict=True)
                if (
                    resolved.parent != mirror_root
                    and mirror_root not in resolved.parents
                ) or resolved in observed_files or not resolved.is_file():
                    raise ValueError("verified cache mirror member missing or ambiguous")
                observed_files.add(resolved)
                relocated_map[scenario] = str(resolved)
            relocated["cache_npz_by_scenario"] = relocated_map
            temporary = tempfile.TemporaryDirectory(
                prefix="somph-verified-cache-relocation-"
            )
            load_path = Path(temporary.name) / "cache_set.json"
            with load_path.open("xb") as handle:
                handle.write(_json_line(relocated))
                handle.flush()
                os.fsync(handle.fileno())
        try:
            arrays, _loaded_manifest, audit = load_verified_leo_weak_cache_set(
                load_path,
                expected_scope="stage2_registered",
                allowed_roles={"target_old", "target_new"},
            )
        finally:
            if temporary is not None:
                temporary.cleanup()
    except (OSError, TypeError, ValueError, authority.SomphLineageAuthorityError) as exc:
        raise SomphRuntimeAuthorizationSigningError(
            "verified cache-set full validation failed"
        ) from exc
    expected_manifest = lock.get("cache_set_manifest", {})
    if (
        manifest_sha != expected_manifest.get("sha256")
        or manifest.get("cache_scope") != lock.get("cache_scope")
        or manifest.get("cache_sha256_by_scenario")
        != lock.get("cache_sha256_by_scenario")
        or audit.get("physical_sample_ids_sha256_by_scenario")
        != attestation.get("physical_sample_ids_sha256_by_scenario")
        or audit.get("physical_sample_scenario_assignment_sha256")
        != attestation.get("physical_sample_scenario_assignment_sha256")
    ):
        raise SomphRuntimeAuthorizationSigningError(
            "verified cache-set manifest/authority root binding drift"
        )
    cache_audits = audit.get("cache_audits")
    if not isinstance(cache_audits, dict):
        raise SomphRuntimeAuthorizationSigningError(
            "verified cache-set per-scenario audit missing"
        )
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        row = cache_audits.get(scenario, {})
        if (
            row.get("sha256") != lock["cache_sha256_by_scenario"][scenario]
            or row.get("physical_sample_ids_sha256")
            != attestation["physical_sample_ids_sha256_by_scenario"][scenario]
            or row.get("post_channel_iq_sha256_root")
            != lock["post_channel_iq_sha256_root_by_scenario"][scenario]
            or row.get("overlay_ids_sha256")
            != lock["overlay_ids_sha256_by_scenario"][scenario]
        ):
            raise SomphRuntimeAuthorizationSigningError(
                f"verified cache/authority roots drift for {scenario}"
            )
    return arrays


def _verify_support_cache_membership(
    *,
    state: str,
    manifest: Mapping[str, Any],
    payloads: Mapping[str, Mapping[str, np.ndarray]],
    cache_arrays: Mapping[str, Mapping[str, np.ndarray]],
    new_tx_ids: list[str],
) -> tuple[dict[tuple[str, int, int], str], dict[str, dict[str, str]]]:
    expected_registry = list(predictor.OLD_TX_IDS) + (
        [] if state == "before" else list(new_tx_ids)
    )
    if manifest.get("support_pool_max_k") != manifest.get("k_shot"):
        raise SomphRuntimeAuthorizationSigningError(
            "formal enrollment support_pool_max_k must equal K-shot"
        )
    selected_physical_ids: dict[str, set[str]] = {}
    assignments: dict[tuple[str, int, int], str] = {}
    assignment_rows: list[dict[str, Any]] = []
    physical_roots: dict[str, str] = {}
    overlay_roots: dict[str, str] = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        support = payloads[scenario]
        cache = cache_arrays[scenario]
        cache_hashes = np.asarray(cache["post_channel_iq_sha256"]).astype(str)
        cache_iq = np.asarray(cache["leo_weak_iq"], dtype=np.float32)
        cache_seeds = np.asarray(cache["satellite_seeds"]).astype(np.int64)
        cache_tx = np.asarray(cache["tx_ids"]).astype(str)
        cache_roles = np.asarray(cache["dataset_role"]).astype(str)
        cache_receivers = np.asarray(cache["rx_ids"]).astype(str)
        cache_scenarios = np.asarray(cache["sat_scenarios"]).astype(str)
        cache_physical = np.asarray(cache["sample_ids"]).astype(str)
        cache_overlays = np.asarray(cache["overlay_ids"]).astype(str)
        support_iq = np.asarray(support["support_leo_weak_iq"], dtype=np.float32)
        support_hashes = np.asarray(
            support["support_post_channel_iq_sha256"]
        ).astype(str)
        support_seeds = np.asarray(support["support_satellite_seeds"]).astype(
            np.int64
        )
        class_indices = np.asarray(support["support_class_indices"]).astype(np.int64)
        ranks = np.asarray(support["support_rank_within_class"]).astype(np.int64)
        seen: set[str] = set()
        ordered_physical: list[str] = []
        ordered_overlays: list[str] = []
        expected_pairs = [
            (class_index, rank)
            for class_index in range(len(expected_registry))
            for rank in range(int(manifest["k_shot"]))
        ]
        if list(zip(class_indices.tolist(), ranks.tolist())) != expected_pairs:
            raise SomphRuntimeAuthorizationSigningError(
                f"support exact-K/rank assignment drift for {state}/{scenario}"
            )
        for row_index, (class_index, declared_hash) in enumerate(
            zip(class_indices.tolist(), support_hashes.tolist())
        ):
            actual_hash = post_channel_iq_sha256(support_iq[row_index])
            if actual_hash != declared_hash:
                raise SomphRuntimeAuthorizationSigningError(
                    f"support IQ digest drift for {state}/{scenario}"
                )
            matches = np.flatnonzero(cache_hashes == actual_hash)
            if matches.shape != (1,):
                raise SomphRuntimeAuthorizationSigningError(
                    f"support IQ has no unique verified-cache membership for {state}/{scenario}"
                )
            cache_index = int(matches[0])
            if not np.array_equal(support_iq[row_index], cache_iq[cache_index]):
                raise SomphRuntimeAuthorizationSigningError(
                    f"support/cache IQ bytes drift for {state}/{scenario}"
                )
            expected_tx = expected_registry[int(class_index)]
            expected_role = (
                "target_old"
                if int(class_index) < len(predictor.OLD_TX_IDS)
                else "target_new"
            )
            if (
                int(support_seeds[row_index]) != int(cache_seeds[cache_index])
                or cache_tx[cache_index] != expected_tx
                or cache_roles[cache_index] != expected_role
                or cache_receivers[cache_index] != manifest["receiver"]
                or cache_scenarios[cache_index] != scenario
                or not cache_overlays[cache_index]
            ):
                raise SomphRuntimeAuthorizationSigningError(
                    f"support/cache seed-TX-role-overlay binding drift for {state}/{scenario}"
                )
            physical_id = cache_physical[cache_index]
            if physical_id in seen:
                raise SomphRuntimeAuthorizationSigningError(
                    f"support repeats one physical sample within {state}/{scenario}"
                )
            seen.add(physical_id)
            rank = int(ranks[row_index])
            assignments[(scenario, int(class_index), rank)] = physical_id
            assignment_rows.append(
                {
                    "scenario": scenario,
                    "class_index": int(class_index),
                    "rank_within_class": rank,
                    "physical_sample_id": physical_id,
                    "overlay_id": cache_overlays[cache_index],
                    "tx_id": expected_tx,
                    "role": expected_role,
                    "satellite_seed": int(cache_seeds[cache_index]),
                    "post_channel_iq_sha256": actual_hash,
                }
            )
            ordered_physical.append(physical_id)
            ordered_overlays.append(cache_overlays[cache_index])
        selected_physical_ids[scenario] = seen
        physical_roots[scenario] = authority.sha256_bytes(
            authority.canonical_json_bytes(ordered_physical)
        )
        overlay_roots[scenario] = authority.sha256_bytes(
            authority.canonical_json_bytes(ordered_overlays)
        )
    for left_index, left in enumerate(FORMAL_LEO_WEAK_SCENARIOS):
        for right in FORMAL_LEO_WEAK_SCENARIOS[left_index + 1 :]:
            if selected_physical_ids[left] & selected_physical_ids[right]:
                raise SomphRuntimeAuthorizationSigningError(
                    f"support reuses physical samples across scenarios for {state}"
                )
    return assignments, {
        "selected_physical_sample_sha256_by_scenario": physical_roots,
        "selected_overlay_sha256_by_scenario": overlay_roots,
        "selected_membership_assignment_sha256": authority.sha256_bytes(
            authority.canonical_json_bytes(assignment_rows)
        ),
        "support_query_disjointness_status": "SUPPORT_ONLY_NO_QUERY_CLAIM",
    }


def _require_before_after_old_membership_stable(
    before: Mapping[tuple[str, int, int], str],
    after: Mapping[tuple[str, int, int], str],
) -> None:
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        for class_index in range(len(predictor.OLD_TX_IDS)):
            before_ranks = sorted(
                rank
                for current_scenario, current_class, rank in before
                if current_scenario == scenario and current_class == class_index
            )
            for rank in before_ranks:
                key = (scenario, class_index, rank)
                if after.get(key) != before[key]:
                    raise SomphRuntimeAuthorizationSigningError(
                        "before/after old-class support membership drift"
                    )


def _authorization(
    *,
    manifest: Mapping[str, Any],
    seal: Mapping[str, Any],
    seal_sha256: str,
    provenance: Mapping[str, Mapping[str, Mapping[str, Any]]],
    new_tx_ids: list[str],
    formal_policy_sha256: str,
    code_members: list[dict[str, str]],
    code_closure_sha256: str,
    lock: Mapping[str, Any],
    attestation: Mapping[str, Any],
    commit: Mapping[str, Any],
    authority_commit_sha256: str,
    membership_roots: Mapping[str, Any],
) -> dict[str, Any]:
    _require_package_authority_source_binding(
        provenance, lock=lock, attestation=attestation
    )
    package_roots = predictor._package_control_roots(
        manifest, provenance, new_tx_ids=new_tx_ids
    )
    preflight_code_sha256 = next(
        row["sha256"]
        for row in code_members
        if row["logical_name"] == "somph_predictor_bundle.py"
    )
    payload: dict[str, Any] = {
        "schema": predictor.SOMPH_FORMAL_POLICY_AUTHORIZATION_SCHEMA,
        "status": predictor.SOMPH_FORMAL_POLICY_AUTHORIZATION_STATUS,
        "formal_launch_authority": True,
        "formal_metric_claim_allowed": False,
        "package_root_sha256": manifest["package_root_sha256"],
        "package_detached_seal_sha256": seal_sha256,
        "artifact_member_allowlist_sha256": seal[
            "artifact_member_allowlist_sha256"
        ],
        "manifest_sha256": seal["manifest_sha256"],
        "overlay_provenance_sha256": manifest["overlay_provenance_sha256"],
        "authority_commit_sha256": authority_commit_sha256,
        "authority_lock_sha256": commit["authority_lock_sha256"],
        "authority_attestation_sha256": _member_sha(
            commit, authority.AUTHORITY_ATTESTATION_NAME
        ),
        "receiver": manifest["receiver"],
        "seed": manifest["seed"],
        "stage": manifest["stage"],
        "registration_state": manifest["registration_state"],
        "k_shot": manifest["k_shot"],
        "cache_scope": "stage2_registered",
        "old_tx_ids": list(lock["old_tx_ids"]),
        "new_tx_ids": list(new_tx_ids),
        "cache_sha256_by_scenario": dict(lock["cache_sha256_by_scenario"]),
        "channel_config_sha256_by_scenario": dict(
            lock["channel_config_sha256_by_scenario"]
        ),
        "structural_receipt_sha256": attestation["structural_receipt_sha256"],
        "dataset_authority_root_sha256": attestation[
            "dataset_authority_root_sha256"
        ],
        "cache_role_inputs_root_sha256": attestation[
            "cache_role_inputs_root_sha256"
        ],
        "physical_sample_ids_sha256_by_scenario": dict(
            attestation["physical_sample_ids_sha256_by_scenario"]
        ),
        "physical_sample_scenario_assignment_sha256": attestation[
            "physical_sample_scenario_assignment_sha256"
        ],
        "post_channel_iq_sha256_root_by_scenario": dict(
            lock["post_channel_iq_sha256_root_by_scenario"]
        ),
        "overlay_ids_sha256_by_scenario": dict(
            lock["overlay_ids_sha256_by_scenario"]
        ),
        "preflight_code_sha256": preflight_code_sha256,
        "formal_policy_sha256": formal_policy_sha256,
        "code_closure_sha256": code_closure_sha256,
        "physical_sample_scenario_assignment_policy": runtime_trust.PHYSICAL_SAMPLE_SCENARIO_ASSIGNMENT_POLICY,
        "cross_scenario_physical_disjointness_audit": "PASS",
        "single_observation_contract_audit": "PASS",
        **runtime_trust.PHASE2_SINGLE_OBSERVATION_CONTRACT,
        **package_roots,
        **dict(membership_roots),
    }
    try:
        predictor._validate_path_free_authorization_shape(payload)
    except predictor.PredictorPackageError as exc:
        raise SomphRuntimeAuthorizationSigningError(str(exc)) from exc
    return payload


def _envelope(
    authorization: Mapping[str, Any],
    *,
    issuer: str,
    key_id: str,
) -> dict[str, Any]:
    return {
        "schema": predictor.SOMPH_SIGNED_POLICY_ENVELOPE_SCHEMA,
        "domain": predictor.SOMPH_SIGNED_POLICY_ENVELOPE_DOMAIN,
        "issuer": issuer,
        "key_id": key_id,
        "authorization_canonical_sha256": authority.sha256_bytes(
            authority.canonical_json_bytes(dict(authorization))
        ),
        "formal_policy_sha256": authorization["formal_policy_sha256"],
        "package_root_sha256": authorization["package_root_sha256"],
        "package_detached_seal_sha256": authorization[
            "package_detached_seal_sha256"
        ],
        "authority_commit_sha256": authorization["authority_commit_sha256"],
        "receiver": authorization["receiver"],
        "seed": authorization["seed"],
        "stage": authorization["stage"],
        "registration_state": authorization["registration_state"],
        "k_shot": authorization["k_shot"],
        "code_closure_sha256": authorization["code_closure_sha256"],
        "signature_ed25519_hex": "",
    }


def _reject_output_reachability(value: Any, *, location: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if any(fragment in normalized for fragment in _FORBIDDEN_OUTPUT_FRAGMENTS):
                raise SomphRuntimeAuthorizationSigningError(
                    f"forbidden runtime output key at {location}"
                )
            _reject_output_reachability(item, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_output_reachability(item, location=f"{location}[{index}]")
    elif isinstance(value, str):
        lowered = value.lower()
        if (
            "/" in value
            or "\\" in value
            or lowered.startswith("file:")
            or lowered.endswith((".pkl", ".npz", ".pt", ".pth"))
            or any(fragment in lowered for fragment in ("raw_iq", "clean_sample", "build_spec", "cache_build"))
        ):
            raise SomphRuntimeAuthorizationSigningError(
                f"forbidden runtime output value at {location}"
            )


def _publish_output_root(output_root: Path, payloads: Mapping[str, bytes]) -> None:
    if output_root.exists():
        raise FileExistsError("refusing to overwrite runtime authorization output root")
    parent = output_root.parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    lock_signer._fsync_directory(parent)
    staging = parent / f".{output_root.name}.staging-{secrets.token_hex(8)}"
    staging.mkdir()
    published = False
    try:
        for name, payload in payloads.items():
            lock_signer._write_new_readonly(staging / name, payload)
        lock_signer._fsync_directory(staging)
        os.rename(staging, output_root)
        published = True
        lock_signer._fsync_directory(parent)
    except BaseException:
        cleanup = output_root if published else staging
        if cleanup.exists():
            authority._remove_tree(cleanup)
        lock_signer._fsync_directory(parent)
        raise


def _prepare_formal_authorization_pair(
    *,
    actual_cache_manifest_path: str | Path,
    authority_bundle_root: str | Path,
    expected_authority_commit_sha256: str,
    verified_cache_root: str | Path,
    before_package_root: str | Path,
    before_detached_seal_path: str | Path,
    after_package_root: str | Path,
    after_detached_seal_path: str | Path,
    formal_policy_path: str | Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    try:
        lock, attestation, commit = authority.verify_somph_lineage_authority_bundle(
            authority_bundle_root,
            expected_commit_sha256=expected_authority_commit_sha256,
        )
    except authority.SomphLineageAuthorityError as exc:
        raise SomphRuntimeAuthorizationSigningError(str(exc)) from exc
    committed_manifest_sha = _member_sha(commit, authority.CACHE_SPEC_MANIFEST_NAME)
    _read_actual_formal_manifest(
        actual_cache_manifest_path,
        expected_sha256=committed_manifest_sha,
        lock=lock,
    )
    cache_arrays = _load_authority_bound_cache_set(
        verified_cache_root, lock=lock, attestation=attestation
    )
    _policy, formal_policy_sha = _read_policy(formal_policy_path)
    code_members, code_closure_sha = _runtime_code_closure()
    (
        before_manifest,
        before_seal,
        before_provenance,
        before_seal_sha,
        before_payloads,
    ) = (
        _preflight_enrollment_package(before_package_root, before_detached_seal_path)
    )
    (
        after_manifest,
        after_seal,
        after_provenance,
        after_seal_sha,
        after_payloads,
    ) = (
        _preflight_enrollment_package(after_package_root, after_detached_seal_path)
    )
    new_tx_ids = _require_package_pair(before_manifest, after_manifest)
    if (
        before_manifest["receiver"] != lock["receiver"]
        or before_manifest["seed"] != lock["seed"]
        or list(lock["old_tx_ids"]) != list(predictor.OLD_TX_IDS)
        or list(lock["new_tx_ids"]) != list(predictor.NEW_TX_IDS)
    ):
        raise SomphRuntimeAuthorizationSigningError(
            "package pair/verified authority cell identity drift"
        )
    before_membership, before_membership_roots = _verify_support_cache_membership(
        state="before",
        manifest=before_manifest,
        payloads=before_payloads,
        cache_arrays=cache_arrays,
        new_tx_ids=new_tx_ids,
    )
    after_membership, after_membership_roots = _verify_support_cache_membership(
        state="after",
        manifest=after_manifest,
        payloads=after_payloads,
        cache_arrays=cache_arrays,
        new_tx_ids=new_tx_ids,
    )
    _require_before_after_old_membership_stable(
        before_membership, after_membership
    )

    inputs = {
        "before": (
            before_manifest,
            before_seal,
            before_provenance,
            before_seal_sha,
            before_membership_roots,
        ),
        "after": (
            after_manifest,
            after_seal,
            after_provenance,
            after_seal_sha,
            after_membership_roots,
        ),
    }
    authorizations = {
        state: _authorization(
            manifest=values[0],
            seal=values[1],
            provenance=values[2],
            seal_sha256=values[3],
            new_tx_ids=new_tx_ids,
            formal_policy_sha256=formal_policy_sha,
            code_members=code_members,
            code_closure_sha256=code_closure_sha,
            lock=lock,
            attestation=attestation,
            commit=commit,
            authority_commit_sha256=expected_authority_commit_sha256,
            membership_roots=values[4],
        )
        for state, values in inputs.items()
    }
    evidence = {
        "authority_commit_sha256": expected_authority_commit_sha256,
        "actual_cache_manifest_sha256": committed_manifest_sha,
        "verified_cache_set_sha256": lock["cache_set_manifest"]["sha256"],
        "code_closure_sha256": code_closure_sha,
        "formal_policy_sha256": formal_policy_sha,
        "selected_membership_roots": {
            "before": before_membership_roots,
            "after": after_membership_roots,
        },
    }
    return authorizations, evidence


def _make_production_runtime_authorization_signer() -> Callable[..., dict[str, Any]]:
    pinned_issuer = str(runtime_trust.PINNED_AUTHORITY_ISSUER)
    pinned_key_id = str(runtime_trust.PINNED_AUTHORITY_KEY_ID)
    pinned_public_key = bytes.fromhex(runtime_trust.PINNED_AUTHORITY_PUBLIC_KEY_HEX)
    pinned_public_key_sha256 = str(
        runtime_trust.PINNED_AUTHORITY_PUBLIC_KEY_SHA256
    )
    pinned_verifier = runtime_trust.verify_ed25519
    if (
        hashlib.sha256(pinned_public_key).hexdigest()
        != pinned_public_key_sha256
        or runtime_trust.PINNED_AUTHORITY_PUBLIC_KEY_HEX
        != authority.PINNED_AUTHORITY_PUBLIC_KEY_HEX
        or pinned_public_key_sha256 != authority.PINNED_AUTHORITY_PUBLIC_KEY_SHA256
        or pinned_issuer != authority.PINNED_AUTHORITY_ISSUER
        or pinned_key_id != authority.PINNED_AUTHORITY_KEY_ID
    ):
        raise SomphRuntimeAuthorizationSigningError(
            "offline authority and runtime trust identity drift"
        )

    def sign_runtime_authorization_pair(
        *,
        actual_cache_manifest_path: str | Path,
        authority_bundle_root: str | Path,
        expected_authority_commit_sha256: str,
        verified_cache_root: str | Path,
        before_package_root: str | Path,
        before_detached_seal_path: str | Path,
        after_package_root: str | Path,
        after_detached_seal_path: str | Path,
        formal_policy_path: str | Path,
        private_key_path: str | Path,
        openssl_bin: str | Path,
        output_root: str | Path,
    ) -> dict[str, Any]:
        destination = Path(output_root).resolve(strict=False)
        if destination.exists():
            raise FileExistsError(
                "refusing to overwrite runtime authorization output root"
            )
        authorizations, evidence = _prepare_formal_authorization_pair(
            actual_cache_manifest_path=actual_cache_manifest_path,
            authority_bundle_root=authority_bundle_root,
            expected_authority_commit_sha256=expected_authority_commit_sha256,
            verified_cache_root=verified_cache_root,
            before_package_root=before_package_root,
            before_detached_seal_path=before_detached_seal_path,
            after_package_root=after_package_root,
            after_detached_seal_path=after_detached_seal_path,
            formal_policy_path=formal_policy_path,
        )
        envelopes = {
            state: _envelope(
                authorization, issuer=pinned_issuer, key_id=pinned_key_id
            )
            for state, authorization in authorizations.items()
        }
        for envelope in envelopes.values():
            if (
                envelope.get("schema")
                != predictor.SOMPH_SIGNED_POLICY_ENVELOPE_SCHEMA
                or envelope.get("domain")
                != predictor.SOMPH_SIGNED_POLICY_ENVELOPE_DOMAIN
                or envelope.get("issuer") != pinned_issuer
                or envelope.get("key_id") != pinned_key_id
            ):
                raise SomphRuntimeAuthorizationSigningError(
                    "formal signed envelope pinned identity drift"
                )

        _, openssl_bytes, openssl_sha, runtime_files = (
            lock_signer._pinned_openssl_binary(openssl_bin)
        )
        private_key = lock_signer._resolved_regular_file(
            private_key_path, context="Ed25519 private key"
        )
        with lock_signer._private_openssl_executable(
            verified_bytes=openssl_bytes,
            expected_sha256=openssl_sha,
            runtime_files=runtime_files,
        ) as private_openssl:
            for envelope in envelopes.values():
                signature = lock_signer._sign_with_openssl(
                    openssl_binary=private_openssl,
                    private_key=private_key,
                    message=predictor._policy_signature_message(envelope),
                )
                envelope["signature_ed25519_hex"] = signature.hex()
                try:
                    pinned_verifier(
                        pinned_public_key,
                        predictor._policy_signature_message(envelope),
                        signature,
                    )
                except ValueError as exc:
                    raise SomphRuntimeAuthorizationSigningError(
                        "Ed25519 runtime authorization signature invalid"
                    ) from exc

        payloads: dict[str, bytes] = {}
        result: dict[str, Any] = {**evidence, "outputs": {}}
        for state in ("before", "after"):
            _reject_output_reachability(authorizations[state])
            _reject_output_reachability(envelopes[state])
            auth_bytes = _json_line(authorizations[state])
            envelope_bytes = _json_line(envelopes[state])
            payloads[AUTHORIZATION_NAMES[state]] = auth_bytes
            payloads[ENVELOPE_NAMES[state]] = envelope_bytes
            result["outputs"][state] = {
                "authorization_name": AUTHORIZATION_NAMES[state],
                "authorization_canonical_sha256": envelopes[state][
                    "authorization_canonical_sha256"
                ],
                "envelope_name": ENVELOPE_NAMES[state],
                "envelope_file_sha256": hashlib.sha256(envelope_bytes).hexdigest(),
            }
        _publish_output_root(destination, payloads)
        result["output_root"] = str(destination)
        return result

    return sign_runtime_authorization_pair


sign_runtime_authorization_pair = _make_production_runtime_authorization_signer()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual-cache-manifest", type=Path, required=True)
    parser.add_argument("--authority-bundle", type=Path, required=True)
    parser.add_argument("--authority-commit-sha256", required=True)
    parser.add_argument("--verified-cache-root", type=Path, required=True)
    parser.add_argument("--before-package", type=Path, required=True)
    parser.add_argument("--before-seal", type=Path, required=True)
    parser.add_argument("--after-package", type=Path, required=True)
    parser.add_argument("--after-seal", type=Path, required=True)
    parser.add_argument("--formal-policy", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument(
        "--openssl-bin",
        type=Path,
        default=Path(lock_signer.PINNED_OPENSSL_BINARY_PATH),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = sign_runtime_authorization_pair(
        actual_cache_manifest_path=args.actual_cache_manifest,
        authority_bundle_root=args.authority_bundle,
        expected_authority_commit_sha256=args.authority_commit_sha256,
        verified_cache_root=args.verified_cache_root,
        before_package_root=args.before_package,
        before_detached_seal_path=args.before_seal,
        after_package_root=args.after_package,
        after_detached_seal_path=args.after_seal,
        formal_policy_path=args.formal_policy,
        private_key_path=args.private_key,
        openssl_bin=args.openssl_bin,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
